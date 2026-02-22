//! SABR per-expiry calibration with optional beta pinning.
//!
//! References:
//! - Hagan et al. (2002), SABR implied-vol asymptotics.
//! - West (2005), practical SABR calibration heuristics.

use serde::{Deserialize, Serialize};

use crate::calibration::core::{
    BoxConstraints, CalibrationResult, Calibrator, finite_metric, matrix_condition_number,
    matrix_to_rows, sanitize_convergence,
};
use crate::calibration::diagnostics::diagnostics;
use crate::calibration::instruments::{OptionVolQuote, make_error_record};
use crate::calibration::optimizers::{
    DifferentialEvolutionOptions, LmOptions, NelderMeadOptions, differential_evolution,
    levenberg_marquardt, nelder_mead,
};
use crate::vol::sabr::{SabrParams, fit_sabr};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SabrCalibrationParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
}

impl From<SabrParams> for SabrCalibrationParams {
    fn from(value: SabrParams) -> Self {
        Self {
            alpha: value.alpha,
            beta: value.beta,
            rho: value.rho,
            nu: value.nu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SabrCalibrator {
    pub forward: f64,
    pub maturity: f64,
    pub beta_pin: Option<f64>,
    pub lm_options: LmOptions,
    pub de_options: DifferentialEvolutionOptions,
    pub nm_options: NelderMeadOptions,
    pub use_global_search: bool,
    pub use_nelder_mead_fallback: bool,
}

impl Default for SabrCalibrator {
    fn default() -> Self {
        Self {
            forward: 100.0,
            maturity: 1.0,
            beta_pin: Some(0.5),
            lm_options: LmOptions {
                max_iterations: 18,
                ..LmOptions::default()
            },
            de_options: DifferentialEvolutionOptions {
                max_generations: 40,
                population_size: 20,
                ..DifferentialEvolutionOptions::default()
            },
            nm_options: NelderMeadOptions::default(),
            use_global_search: false,
            use_nelder_mead_fallback: true,
        }
    }
}

impl SabrCalibrator {
    fn bounds(&self) -> BoxConstraints {
        if self.beta_pin.is_some() {
            BoxConstraints::new(vec![1e-8, -0.999, 1e-8], vec![5.0, 0.999, 5.0])
                .expect("valid SABR bounds")
        } else {
            BoxConstraints::new(vec![1e-8, 0.0, -0.999, 1e-8], vec![5.0, 1.0, 0.999, 5.0])
                .expect("valid SABR bounds")
        }
    }

    fn decode_params(&self, x: &[f64]) -> Option<SabrCalibrationParams> {
        if let Some(beta) = self.beta_pin {
            if x.len() != 3 {
                return None;
            }
            Some(SabrCalibrationParams {
                alpha: x[0],
                beta,
                rho: x[1],
                nu: x[2],
            })
        } else {
            if x.len() != 4 {
                return None;
            }
            Some(SabrCalibrationParams {
                alpha: x[0],
                beta: x[1],
                rho: x[2],
                nu: x[3],
            })
        }
    }

    fn encode_params(&self, p: SabrCalibrationParams) -> Vec<f64> {
        if self.beta_pin.is_some() {
            vec![p.alpha, p.rho, p.nu]
        } else {
            vec![p.alpha, p.beta, p.rho, p.nu]
        }
    }

    fn initial_guess(&self, instruments: &[OptionVolQuote]) -> Vec<f64> {
        let strikes: Vec<f64> = instruments.iter().map(|q| q.strike).collect();
        let vols: Vec<f64> = instruments.iter().map(|q| q.market_vol).collect();

        let guess = if let Some(beta) = self.beta_pin {
            fit_sabr(self.forward, &strikes, &vols, self.maturity, beta).into()
        } else {
            let atm = instruments
                .iter()
                .min_by(|a, b| {
                    (a.strike - self.forward)
                        .abs()
                        .total_cmp(&(b.strike - self.forward).abs())
                })
                .map(|q| q.market_vol)
                .unwrap_or(0.2)
                .max(1e-4);
            SabrCalibrationParams {
                alpha: atm * self.forward.powf(0.5),
                beta: 0.5,
                rho: -0.2,
                nu: 0.6,
            }
        };

        self.bounds().clamp(&self.encode_params(guess))
    }

    fn model_vols(&self, x: &[f64], instruments: &[OptionVolQuote]) -> Option<Vec<f64>> {
        let p = self.decode_params(x)?;
        let params = SabrParams {
            alpha: p.alpha.max(1e-8),
            beta: p.beta.clamp(0.0, 1.0),
            rho: p.rho.clamp(-0.999, 0.999),
            nu: p.nu.max(1e-8),
        };
        Some(
            instruments
                .iter()
                .map(|q| {
                    params
                        .implied_vol(self.forward, q.strike, self.maturity)
                        .max(1e-8)
                })
                .collect(),
        )
    }

    fn residuals(&self, x: &[f64], instruments: &[OptionVolQuote]) -> Vec<f64> {
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

    fn objective(&self, x: &[f64], instruments: &[OptionVolQuote]) -> f64 {
        let r = self.residuals(x, instruments);
        0.5 * r.iter().map(|v| v * v).sum::<f64>()
    }
}

impl Calibrator<SabrCalibrationParams> for SabrCalibrator {
    type Instrument = OptionVolQuote;

    fn name(&self) -> &'static str {
        "sabr"
    }

    fn calibrate(
        &self,
        instruments: &[Self::Instrument],
    ) -> Result<CalibrationResult<SabrCalibrationParams>, String> {
        if instruments.is_empty() {
            return Err("sabr calibration requires non-empty instrument set".to_string());
        }
        if self.forward <= 0.0 || self.maturity <= 0.0 {
            return Err("sabr forward and maturity must be > 0".to_string());
        }
        if instruments.iter().any(|q| {
            q.strike <= 0.0
                || q.market_vol <= 0.0
                || !q.market_vol.is_finite()
                || (q.maturity - self.maturity).abs() > 1e-12
        }) {
            return Err("invalid SABR quote set for configured expiry".to_string());
        }

        let bounds = self.bounds();
        let mut start = self.initial_guess(instruments);

        if self.use_global_search {
            let de = differential_evolution(&bounds, self.de_options, |x| {
                self.objective(x, instruments)
            })?;
            start = bounds.clamp(&de.x);
        }

        let mut lm = levenberg_marquardt(&start, &bounds, self.lm_options, |x| {
            self.residuals(x, instruments)
        })?;

        if self.use_nelder_mead_fallback && !lm.convergence.converged {
            let nm = nelder_mead(&lm.x, &bounds, self.nm_options, |x| {
                self.objective(x, instruments)
            })?;
            let lm2 = levenberg_marquardt(&nm.x, &bounds, self.lm_options, |x| {
                self.residuals(x, instruments)
            })?;
            if lm2.objective < lm.objective {
                lm = lm2;
            }
        }

        let params = self
            .decode_params(&lm.x)
            .ok_or_else(|| "failed to decode calibrated SABR params".to_string())?;

        let model = self
            .model_vols(&lm.x, instruments)
            .ok_or_else(|| "failed to evaluate calibrated SABR vols".to_string())?;

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
            Some(&bounds),
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
    use std::time::Instant;

    use super::*;

    #[test]
    fn calibrates_sabr_per_expiry_with_sub_half_vol_point_error_under_50ms() {
        let true_params = SabrParams {
            alpha: 0.22,
            beta: 0.5,
            rho: -0.3,
            nu: 0.7,
        };

        let forward = 100.0;
        let maturity = 2.0;
        let strikes = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];

        let quotes: Vec<OptionVolQuote> = strikes
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let mut q = OptionVolQuote::new(
                    format!("q{i}"),
                    *k,
                    maturity,
                    true_params.implied_vol(forward, *k, maturity),
                );
                q.liquid = (*k / forward - 1.0).abs() <= 0.2;
                q
            })
            .collect();

        let cal = SabrCalibrator {
            forward,
            maturity,
            beta_pin: Some(0.5),
            use_global_search: false,
            ..SabrCalibrator::default()
        };

        let t0 = Instant::now();
        let result = cal.calibrate(&quotes).expect("calibration succeeds");
        let elapsed = t0.elapsed().as_secs_f64();

        assert!(
            result.diagnostics.fit_quality.liquid_rmse < 0.005,
            "liquid RMSE too high: {}",
            result.diagnostics.fit_quality.liquid_rmse
        );
        assert!(elapsed < 0.05, "SABR calibration took {elapsed:.4}s");
    }

    #[test]
    fn calibration_result_is_serde_roundtrip_safe() {
        let quotes = vec![
            OptionVolQuote::new("a", 95.0, 1.0, 0.22),
            OptionVolQuote::new("b", 100.0, 1.0, 0.20),
            OptionVolQuote::new("c", 105.0, 1.0, 0.21),
        ];

        let cal = SabrCalibrator {
            forward: 100.0,
            maturity: 1.0,
            ..SabrCalibrator::default()
        };
        let result = cal.calibrate(&quotes).expect("calibration succeeds");

        let blob = serde_json::to_string(&result).expect("serialize");
        let roundtrip: CalibrationResult<SabrCalibrationParams> =
            serde_json::from_str(&blob).expect("deserialize");

        assert_eq!(result.params, roundtrip.params);
        assert_eq!(
            result.per_instrument_error.len(),
            roundtrip.per_instrument_error.len()
        );
    }
}
