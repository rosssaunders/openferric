//! Hull-White one-factor calibration to swaption-vol matrices.
//!
//! References:
//! - Hull and White (1990), one-factor short-rate model.
//! - Brigo and Mercurio (2006), swaption calibration approximations.

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
use crate::models::{calibrate_hull_white_params, hw_atm_swaption_vol_approx};

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
    pub bounds: BoxConstraints,
    pub lm_options: LmOptions,
    pub nm_options: NelderMeadOptions,
    pub use_nelder_mead_fallback: bool,
}

impl Default for HullWhiteCalibrator {
    fn default() -> Self {
        Self {
            bounds: BoxConstraints::new(vec![1e-5, 1e-5], vec![1.0, 0.2]).expect("valid HW bounds"),
            lm_options: LmOptions {
                max_iterations: 32,
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
                .map(|q| hw_atm_swaption_vol_approx(p.a, p.sigma, q.expiry, q.tenor).max(1e-8))
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
                e.effective_error * q.weight.max(1e-12).sqrt()
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

    #[test]
    fn calibrates_hull_white_swaption_matrix_with_sub_half_vol_point_error() {
        let true_params = HullWhiteCalibrationParams {
            a: 0.06,
            sigma: 0.011,
        };

        let mut quotes = Vec::new();
        for expiry in [1.0, 2.0, 3.0, 5.0] {
            for tenor in [1.0, 2.0, 5.0, 10.0] {
                let vol =
                    hw_atm_swaption_vol_approx(true_params.a, true_params.sigma, expiry, tenor);
                let mut q =
                    SwaptionVolQuote::new(format!("{expiry:.0}x{tenor:.0}"), expiry, tenor, vol);
                q.liquid = tenor <= 5.0;
                quotes.push(q);
            }
        }

        let cal = HullWhiteCalibrator::default();
        let result = cal.calibrate(&quotes).expect("calibration succeeds");

        assert!(
            result.diagnostics.fit_quality.liquid_rmse < 0.005,
            "liquid RMSE too high: {}",
            result.diagnostics.fit_quality.liquid_rmse
        );
    }
}
