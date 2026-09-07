//! Hull-White one-factor calibration to swaption-vol matrices.
//!
//! References:
//! - Hull and White (1990), one-factor short-rate model.
//! - Jamshidian (1989), decomposition into zero-coupon bond options.
//!
//! Quotes are absolute Bachelier ATM volatilities for annual-pay, physically
//! settled single-curve swaptions. Model NPVs are converted to the same normal
//! volatility convention. Optimization residuals are in normal-vol basis
//! points; reported instrument errors retain absolute volatility units.

use serde::{Deserialize, Serialize};

use crate::calibration::core::{
    BoxConstraints, CalibrationResult, Calibrator, finite_metric, matrix_condition_number,
    matrix_to_rows, sanitize_convergence,
};
use crate::calibration::diagnostics::diagnostics;
use crate::calibration::instruments::{SwaptionVolQuote, make_error_record};
use crate::calibration::optimizers::{
    LmOptions, NelderMeadOptions, levenberg_marquardt, nelder_mead,
};
use crate::models::calibrate_hull_white_params;
use crate::rates::{Swaption, YieldCurve};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HullWhiteCalibrationParams {
    pub a: f64,
    pub sigma: f64,
}

impl HullWhiteCalibrationParams {
    fn from_slice(x: &[f64]) -> Option<Self> {
        if x.len() != 2 {
            return None;
        }
        Some(Self {
            a: x[0],
            sigma: x[1],
        })
    }

    fn to_vec(self) -> Vec<f64> {
        vec![self.a, self.sigma]
    }
}

#[derive(Debug, Clone)]
pub struct HullWhiteCalibrator {
    /// Valuation-date curve used for physical, annual-pay ATM swaptions.
    pub curve: YieldCurve,
    pub bounds: BoxConstraints,
    pub lm_options: LmOptions,
    pub nm_options: NelderMeadOptions,
    pub use_nelder_mead_fallback: bool,
}

impl HullWhiteCalibrator {
    /// Fits absolute Bachelier ATM volatilities by repricing physical swaptions.
    pub fn new(curve: YieldCurve) -> Self {
        Self {
            curve,
            bounds: BoxConstraints::new(vec![1e-5, 1e-5], vec![1.0, 0.2]).expect("valid HW bounds"),
            lm_options: LmOptions {
                max_iterations: 100,
                gradient_tolerance: 1.0e-7,
                objective_tolerance: 1.0e-14,
                step_tolerance: 1.0e-10,
                finite_diff_epsilon: 1.0e-6,
                ..LmOptions::default()
            },
            nm_options: NelderMeadOptions::default(),
            use_nelder_mead_fallback: true,
        }
    }
}

impl HullWhiteCalibrator {
    fn initial_guess(&self, instruments: &[SwaptionVolQuote]) -> Vec<f64> {
        let tuples: Vec<(f64, f64, f64)> = instruments
            .iter()
            .map(|q| (q.expiry, q.tenor, q.market_vol))
            .collect();

        if let Some((a, sigma)) = calibrate_hull_white_params(&tuples) {
            self.bounds.clamp(&[a, sigma])
        } else {
            self.bounds.clamp(&[0.05, 0.01])
        }
    }

    fn model_vols(&self, x: &[f64], instruments: &[SwaptionVolQuote]) -> Option<Vec<f64>> {
        let p = HullWhiteCalibrationParams::from_slice(x)?;
        Some(
            instruments
                .iter()
                .map(|quote| {
                    let mut swaption = Swaption {
                        notional: 1.0,
                        strike: 0.0,
                        option_expiry: quote.expiry,
                        swap_tenor: quote.tenor,
                        is_payer: true,
                    };
                    swaption.strike = swaption.forward_swap_rate(&self.curve);
                    swaption
                        .price_hull_white(&self.curve, &crate::models::HullWhite::new(p.a, p.sigma))
                        .map(|price| {
                            price
                                / (swaption.annuity_factor(&self.curve)
                                    * quote.expiry.sqrt()
                                    * crate::math::normal_pdf(0.0))
                        })
                        .unwrap_or(f64::NAN)
                })
                .collect(),
        )
    }

    fn residuals(&self, x: &[f64], instruments: &[SwaptionVolQuote]) -> Vec<f64> {
        let Some(model) = self.model_vols(x, instruments) else {
            return vec![1e6; instruments.len()];
        };

        model
            .iter()
            .zip(instruments.iter())
            .map(|(m, q)| {
                let e = make_error_record(q, *m);
                10_000.0 * e.effective_error * q.weight.max(1e-12).sqrt()
            })
            .collect()
    }

    fn objective(&self, x: &[f64], instruments: &[SwaptionVolQuote]) -> f64 {
        let r = self.residuals(x, instruments);
        0.5 * r.iter().map(|v| v * v).sum::<f64>()
    }
}

impl Calibrator<HullWhiteCalibrationParams> for HullWhiteCalibrator {
    type Instrument = SwaptionVolQuote;

    fn name(&self) -> &'static str {
        "hull-white"
    }

    fn calibrate(
        &self,
        instruments: &[Self::Instrument],
    ) -> Result<CalibrationResult<HullWhiteCalibrationParams>, String> {
        if instruments.is_empty() {
            return Err("hull-white calibration requires non-empty instrument set".to_string());
        }
        if instruments.iter().any(|q| {
            q.expiry <= 0.0
                || q.tenor <= 0.0
                || q.market_vol <= 0.0
                || !q.expiry.is_finite()
                || !q.tenor.is_finite()
                || !q.market_vol.is_finite()
        }) {
            return Err("invalid Hull-White swaption quote set".to_string());
        }

        let start = self.initial_guess(instruments);
        if self
            .model_vols(&start, instruments)
            .is_none_or(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("curve must support finite ATM swaption prices and annuities".into());
        }
        let mut lm = levenberg_marquardt(&start, &self.bounds, self.lm_options, |x| {
            self.residuals(x, instruments)
        })?;

        if self.use_nelder_mead_fallback && !lm.convergence.converged {
            let nm = nelder_mead(&lm.x, &self.bounds, self.nm_options, |x| {
                self.objective(x, instruments)
            })?;
            let lm2 = levenberg_marquardt(&nm.x, &self.bounds, self.lm_options, |x| {
                self.residuals(x, instruments)
            })?;
            if lm2.objective < lm.objective {
                lm = lm2;
            }
        }

        let params = HullWhiteCalibrationParams::from_slice(&lm.x)
            .ok_or_else(|| "failed to decode calibrated Hull-White params".to_string())?;

        let model = self
            .model_vols(&params.to_vec(), instruments)
            .ok_or_else(|| "failed to evaluate calibrated Hull-White vols".to_string())?;

        let errors: Vec<_> = instruments
            .iter()
            .zip(model.iter())
            .map(|(q, m)| make_error_record(q, *m))
            .collect();

        let condition_number = finite_metric(matrix_condition_number(&lm.jacobian));
        let convergence = sanitize_convergence(lm.convergence);
        let diagnostics = diagnostics(
            &errors,
            &convergence,
            condition_number,
            Some(&self.bounds),
            Some(&lm.x),
            None,
        );

        Ok(CalibrationResult {
            params,
            objective: lm.objective,
            per_instrument_error: errors,
            jacobian: matrix_to_rows(&lm.jacobian),
            condition_number,
            convergence,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_curve() -> YieldCurve {
        YieldCurve::new(vec![(30.0, (-0.03_f64 * 30.0).exp())])
    }

    fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && error <= tolerance,
            "{label}: actual={actual:.16e}, expected={expected:.16e}, error={error:.3e}, tolerance={tolerance:.3e}"
        );
    }

    /// SciPy 1.17.1 expiry-forward Gaussian payoff quadrature, absolute
    /// normal-vol errors <1.3e-16; annual 2x5 swaption with a=.06, sigma=.011.
    #[test]
    fn calibration_prices_respect_negative_and_nonflat_discount_curves() {
        let references = [
            (
                YieldCurve::new(vec![(30.0, (0.015_f64 * 30.0).exp())]),
                0.008_809_903_160_634_465,
            ),
            (
                YieldCurve::new(vec![(1.0, 0.99), (3.0, 0.91), (7.0, 0.72), (15.0, 0.5)]),
                0.009_461_768_452_801_047,
            ),
        ];
        let quote = SwaptionVolQuote::new("2x5", 2.0, 5.0, 0.01);
        for (curve, reference) in references {
            let calibrator = HullWhiteCalibrator::new(curve);
            let model_quotes = calibrator
                .model_vols(&[0.06, 0.011], std::slice::from_ref(&quote))
                .unwrap();
            assert_close("curve-dependent quote", model_quotes[0], reference, 1.0e-14);
        }
    }

    /// QuantLib-Python 1.43, FlatForward(2025-01-02,.03,Actual365Fixed),
    /// HullWhite(.06,.011): annual cashflows and discountBondOption-based
    /// Jamshidian NPVs divided by ATM Bachelier annuity/expiry factors.
    #[test]
    fn recovers_parameters_from_quantlib_1_43_prices() {
        let true_params = HullWhiteCalibrationParams {
            a: 0.06,
            sigma: 0.011,
        };

        let references = [
            [
                0.010679657425268224,
                0.01037322565789539,
                0.009539516908173729,
                0.00839433699169527,
            ],
            [
                0.010373310897310823,
                0.010075558432800991,
                0.009265185809660618,
                0.008151655651596592,
            ],
            [
                0.010081785628633518,
                0.00979230559203658,
                0.009004208300888812,
                0.0079209384911933,
            ],
            [
                0.00954011021686305,
                0.00926603228153922,
                0.008519488070666683,
                0.007492784574779565,
            ],
        ];
        let mut quotes = Vec::new();
        for (expiry_index, expiry) in [1.0, 2.0, 3.0, 5.0].into_iter().enumerate() {
            for (tenor_index, tenor) in [1.0, 2.0, 5.0, 10.0].into_iter().enumerate() {
                let vol = references[expiry_index][tenor_index];
                let mut q =
                    SwaptionVolQuote::new(format!("{expiry:.0}x{tenor:.0}"), expiry, tenor, vol);
                q.liquid = tenor <= 5.0;
                quotes.push(q);
            }
        }

        let cal = HullWhiteCalibrator::new(reference_curve());
        let result = cal.calibrate(&quotes).expect("calibration succeeds");

        assert_eq!(result.per_instrument_error.len(), quotes.len());
        assert!(result.objective.is_finite());
        assert!(result.condition_number.is_finite());
        assert!(result.jacobian.iter().flatten().all(|x| x.is_finite()));
        assert!(
            result.convergence.converged,
            "optimizer did not converge: {:?}",
            result.convergence
        );

        assert_close("a", result.params.a, true_params.a, 3.3e-8);
        assert_close("sigma", result.params.sigma, true_params.sigma, 1.3e-9);

        for (error, quote) in result.per_instrument_error.iter().zip(&quotes) {
            assert_eq!(error.id, quote.id);
            assert_close(
                &format!("{} model vol", quote.id),
                error.model,
                quote.market_vol,
                9e-10,
            );
            assert_close(
                &format!("{} recorded signed error", quote.id),
                error.signed_error,
                error.model - quote.market_vol,
                f64::EPSILON,
            );
        }
    }

    #[test]
    fn helper_paths_decode_fallback_and_penalize_wrong_dimensions_exactly() {
        assert!(HullWhiteCalibrationParams::from_slice(&[0.1]).is_none());
        let params = HullWhiteCalibrationParams::from_slice(&[0.07, 0.012]).unwrap();
        assert_eq!(params.to_vec(), vec![0.07, 0.012]);

        let calibrator = HullWhiteCalibrator::new(reference_curve());
        assert_eq!(calibrator.name(), "hull-white");
        assert_eq!(calibrator.initial_guess(&[]), vec![0.05, 0.01]);

        let quotes = [
            SwaptionVolQuote::new("1x2", 1.0, 2.0, 0.01),
            SwaptionVolQuote::new("2x5", 2.0, 5.0, 0.012),
        ];
        assert!(calibrator.model_vols(&[0.05], &quotes).is_none());
        assert_eq!(calibrator.residuals(&[0.05], &quotes), vec![1.0e6; 2]);
        assert_eq!(calibrator.objective(&[0.05], &quotes), 1.0e12);
    }

    #[test]
    fn calibration_rejects_empty_and_malformed_quotes() {
        let calibrator = HullWhiteCalibrator::new(reference_curve());
        assert_eq!(
            calibrator.calibrate(&[]).unwrap_err(),
            "hull-white calibration requires non-empty instrument set"
        );

        let valid = SwaptionVolQuote::new("q", 1.0, 2.0, 0.01);
        let invalid = [
            SwaptionVolQuote {
                expiry: 0.0,
                ..valid.clone()
            },
            SwaptionVolQuote {
                expiry: f64::NAN,
                ..valid.clone()
            },
            SwaptionVolQuote {
                tenor: 0.0,
                ..valid.clone()
            },
            SwaptionVolQuote {
                tenor: f64::INFINITY,
                ..valid.clone()
            },
            SwaptionVolQuote {
                market_vol: 0.0,
                ..valid.clone()
            },
            SwaptionVolQuote {
                market_vol: f64::NAN,
                ..valid.clone()
            },
            SwaptionVolQuote {
                market_vol: f64::INFINITY,
                ..valid
            },
        ];
        for quote in invalid {
            assert_eq!(
                calibrator.calibrate(&[quote]).unwrap_err(),
                "invalid Hull-White swaption quote set"
            );
        }
    }

    #[test]
    fn forced_nonconvergence_and_nelder_mead_fallback_improves_objective() {
        let quotes = [
            SwaptionVolQuote::new("1x1", 1.0, 1.0, 0.0103),
            SwaptionVolQuote::new("2x5", 2.0, 5.0, 0.0074),
            SwaptionVolQuote::new("5x10", 5.0, 10.0, 0.0041),
        ];
        let no_fallback = HullWhiteCalibrator {
            lm_options: LmOptions {
                max_iterations: 0,
                ..LmOptions::default()
            },
            use_nelder_mead_fallback: false,
            ..HullWhiteCalibrator::new(reference_curve())
        };
        let result = no_fallback.calibrate(&quotes).unwrap();
        assert!(!result.convergence.converged);
        assert_eq!(
            result.convergence.reason,
            crate::calibration::TerminationReason::MaxIterations
        );
        assert!(
            result
                .diagnostics
                .warning_flags
                .contains(&crate::calibration::CalibrationWarningFlag::NonConvergent)
        );

        let fallback = HullWhiteCalibrator {
            lm_options: LmOptions {
                max_iterations: 0,
                ..LmOptions::default()
            },
            nm_options: NelderMeadOptions {
                max_iterations: 80,
                ..NelderMeadOptions::default()
            },
            use_nelder_mead_fallback: true,
            ..HullWhiteCalibrator::new(reference_curve())
        };
        let fallback_result = fallback.calibrate(&quotes).unwrap();
        assert!(!fallback_result.convergence.converged);
        assert!(
            fallback_result.objective < result.objective,
            "Nelder-Mead fallback did not improve the forced-zero-iteration LM: baseline={}, fallback={}",
            result.objective,
            fallback_result.objective
        );
        assert_eq!(fallback_result.per_instrument_error.len(), quotes.len());
    }
}
