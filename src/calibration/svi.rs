//! SVI calibration (raw and jump-wings) with Gatheral no-arbitrage constraints.
//!
//! References:
//! - Gatheral (2004, 2006), SVI parameterization and static-arbitrage checks.
//! - Lee (2004), moment formula and wing slope bounds.

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
use crate::vol::surface::{SviParams, calibrate_svi};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SviParameterization {
    Raw,
    JumpWings,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SviRawCalibrationParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl SviRawCalibrationParams {
    fn from_slice(x: &[f64]) -> Option<Self> {
        if x.len() != 5 {
            return None;
        }
        Some(Self {
            a: x[0],
            b: x[1],
            rho: x[2],
            m: x[3],
            sigma: x[4],
        })
    }

    fn to_vec(self) -> Vec<f64> {
        vec![self.a, self.b, self.rho, self.m, self.sigma]
    }

    fn to_svi(self) -> SviParams {
        SviParams {
            a: self.a,
            b: self.b,
            rho: self.rho,
            m: self.m,
            sigma: self.sigma,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SviJumpWingsCalibrationParams {
    pub v: f64,
    pub psi: f64,
    pub p: f64,
    pub c: f64,
    pub vt: f64,
    pub maturity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "parameterization", content = "params")]
pub enum SviCalibrationParams {
    Raw(SviRawCalibrationParams),
    JumpWings(SviJumpWingsCalibrationParams),
}

#[derive(Debug, Clone)]
pub struct SviCalibrator {
    pub forward: f64,
    pub maturity: f64,
    pub parameterization: SviParameterization,
    pub lm_options: LmOptions,
    pub de_options: DifferentialEvolutionOptions,
    pub nm_options: NelderMeadOptions,
    pub use_global_search: bool,
    pub use_nelder_mead_fallback: bool,
}

impl Default for SviCalibrator {
    fn default() -> Self {
        Self {
            forward: 100.0,
            maturity: 1.0,
            parameterization: SviParameterization::Raw,
            lm_options: LmOptions {
                max_iterations: 24,
                ..LmOptions::default()
            },
            de_options: DifferentialEvolutionOptions {
                max_generations: 60,
                population_size: 30,
                ..DifferentialEvolutionOptions::default()
            },
            nm_options: NelderMeadOptions::default(),
            use_global_search: true,
            use_nelder_mead_fallback: true,
        }
    }
}

impl SviCalibrator {
    fn bounds(&self) -> BoxConstraints {
        BoxConstraints::new(
            vec![-0.5, 1e-8, -0.999, -5.0, 1e-5],
            vec![2.0, 3.0, 0.999, 5.0, 3.0],
        )
        .expect("valid SVI bounds")
    }

    fn project_no_arb(&self, x: &[f64], bounds: &BoxConstraints) -> Vec<f64> {
        let mut p = bounds.clamp(x);
        let rho = p[2].clamp(-0.999, 0.999);
        p[2] = rho;

        // Lee wing slopes: b(1 +/- rho) < 2
        let b_max = 1.999 / (1.0 + rho.abs()).max(1e-6);
        p[1] = p[1].min(b_max).max(1e-8);
        p[4] = p[4].max(1e-5);

        // Gatheral positivity at minimum total variance.
        let floor = 1e-8 - p[1] * p[4] * (1.0 - rho * rho).max(0.0).sqrt();
        p[0] = p[0].max(floor);

        p
    }

    fn initial_guess(&self, instruments: &[OptionVolQuote], bounds: &BoxConstraints) -> Vec<f64> {
        let points: Vec<(f64, f64)> = instruments
            .iter()
            .map(|q| {
                let k = (q.strike / self.forward).ln();
                let w = q.market_vol * q.market_vol * self.maturity;
                (k, w)
            })
            .collect();

        let atm_w = instruments
            .iter()
            .min_by(|a, b| {
                (a.strike - self.forward)
                    .abs()
                    .total_cmp(&(b.strike - self.forward).abs())
            })
            .map(|q| q.market_vol * q.market_vol * self.maturity)
            .unwrap_or(0.04 * self.maturity)
            .max(1e-4);

        let init = SviParams {
            a: (0.5 * atm_w).max(1e-6),
            b: 0.15,
            rho: -0.2,
            m: 0.0,
            sigma: 0.25,
        };
        let fit = calibrate_svi(&points, init, 1500, 2e-3);

        self.project_no_arb(
            &SviRawCalibrationParams {
                a: fit.a,
                b: fit.b,
                rho: fit.rho,
                m: fit.m,
                sigma: fit.sigma,
            }
            .to_vec(),
            bounds,
        )
    }

    fn raw_from_vector(
        &self,
        x: &[f64],
        bounds: &BoxConstraints,
    ) -> Option<SviRawCalibrationParams> {
        SviRawCalibrationParams::from_slice(&self.project_no_arb(x, bounds))
    }

    fn model_vols(
        &self,
        x: &[f64],
        instruments: &[OptionVolQuote],
        bounds: &BoxConstraints,
    ) -> Option<Vec<f64>> {
        let raw = self.raw_from_vector(x, bounds)?;
        let svi = raw.to_svi();

        Some(
            instruments
                .iter()
                .map(|q| {
                    let k = (q.strike / self.forward).ln();
                    let w = svi.total_variance(k).max(1e-12);
                    (w / self.maturity).sqrt().max(1e-8)
                })
                .collect(),
        )
    }

    fn residuals(
        &self,
        x: &[f64],
        instruments: &[OptionVolQuote],
        bounds: &BoxConstraints,
    ) -> Vec<f64> {
        let Some(model) = self.model_vols(x, instruments, bounds) else {
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

    fn objective(&self, x: &[f64], instruments: &[OptionVolQuote], bounds: &BoxConstraints) -> f64 {
        let r = self.residuals(x, instruments, bounds);
        0.5 * r.iter().map(|v| v * v).sum::<f64>()
    }

    fn raw_to_jump_wings(&self, raw: SviRawCalibrationParams) -> SviJumpWingsCalibrationParams {
        let svi = raw.to_svi();
        let t = self.maturity.max(1e-8);
        let w_atm = svi.total_variance(0.0).max(1e-12);
        let x = -raw.m;
        let sqrt_term = (x * x + raw.sigma * raw.sigma).sqrt();
        let dw_dk_atm = raw.b * (raw.rho + x / sqrt_term);

        SviJumpWingsCalibrationParams {
            v: w_atm / t,
            psi: dw_dk_atm / (2.0 * w_atm.sqrt()),
            p: raw.b * (1.0 - raw.rho) / w_atm.sqrt(),
            c: raw.b * (1.0 + raw.rho) / w_atm.sqrt(),
            vt: (raw.a + raw.b * raw.sigma * (1.0 - raw.rho * raw.rho).max(0.0).sqrt()) / t,
            maturity: t,
        }
    }
}

impl Calibrator<SviCalibrationParams> for SviCalibrator {
    type Instrument = OptionVolQuote;

    fn name(&self) -> &'static str {
        "svi"
    }

    fn calibrate(
        &self,
        instruments: &[Self::Instrument],
    ) -> Result<CalibrationResult<SviCalibrationParams>, String> {
        if instruments.is_empty() {
            return Err("svi calibration requires non-empty instrument set".to_string());
        }
        if self.forward <= 0.0 || self.maturity <= 0.0 {
            return Err("svi forward and maturity must be > 0".to_string());
        }
        if instruments.iter().any(|q| {
            q.strike <= 0.0
                || q.market_vol <= 0.0
                || !q.market_vol.is_finite()
                || (q.maturity - self.maturity).abs() > 1e-12
        }) {
            return Err("invalid SVI quote set for configured expiry".to_string());
        }

        let bounds = self.bounds();
        let mut start = self.initial_guess(instruments, &bounds);

        if self.use_global_search {
            let de = differential_evolution(&bounds, self.de_options, |x| {
                self.objective(x, instruments, &bounds)
            })?;
            start = self.project_no_arb(&de.x, &bounds);
        }

        let mut lm = levenberg_marquardt(&start, &bounds, self.lm_options, |x| {
            self.residuals(x, instruments, &bounds)
        })?;

        if self.use_nelder_mead_fallback && !lm.convergence.converged {
            let nm = nelder_mead(&lm.x, &bounds, self.nm_options, |x| {
                self.objective(x, instruments, &bounds)
            })?;
            let lm2 = levenberg_marquardt(&nm.x, &bounds, self.lm_options, |x| {
                self.residuals(x, instruments, &bounds)
            })?;
            if lm2.objective < lm.objective {
                lm = lm2;
            }
        }

        let raw = self
            .raw_from_vector(&lm.x, &bounds)
            .ok_or_else(|| "failed to decode calibrated SVI params".to_string())?;

        let model = self
            .model_vols(&lm.x, instruments, &bounds)
            .ok_or_else(|| "failed to evaluate calibrated SVI vols".to_string())?;

        let errors: Vec<_> = instruments
            .iter()
            .zip(model.iter())
            .map(|(q, m)| make_error_record(q, *m))
            .collect();

        let params = match self.parameterization {
            SviParameterization::Raw => SviCalibrationParams::Raw(raw),
            SviParameterization::JumpWings => {
                SviCalibrationParams::JumpWings(self.raw_to_jump_wings(raw))
            }
        };

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
    use super::*;

    fn synthetic_quotes() -> (f64, f64, Vec<OptionVolQuote>) {
        let forward = 100.0;
        let maturity = 1.5;
        let true_p = SviRawCalibrationParams {
            a: 0.01,
            b: 0.18,
            rho: -0.3,
            m: 0.02,
            sigma: 0.25,
        };
        let svi = true_p.to_svi();

        let strikes: [f64; 9] = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];
        let quotes = strikes
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let kk = (*k / forward).ln();
                let vol = (svi.total_variance(kk) / maturity).sqrt();
                let mut q = OptionVolQuote::new(format!("q{i}"), *k, maturity, vol);
                q.liquid = (*k / forward - 1.0).abs() <= 0.2;
                q
            })
            .collect();
        (forward, maturity, quotes)
    }

    #[test]
    fn calibrates_raw_svi_with_sub_half_vol_point_error() {
        let (forward, maturity, quotes) = synthetic_quotes();
        let cal = SviCalibrator {
            forward,
            maturity,
            parameterization: SviParameterization::Raw,
            ..SviCalibrator::default()
        };

        let result = cal.calibrate(&quotes).expect("svi calibration succeeds");
        assert!(
            result.diagnostics.fit_quality.liquid_rmse < 0.005,
            "liquid RMSE too high: {}",
            result.diagnostics.fit_quality.liquid_rmse
        );
    }

    #[test]
    fn calibrates_jump_wings_output_with_sub_half_vol_point_error() {
        let (forward, maturity, quotes) = synthetic_quotes();
        let cal = SviCalibrator {
            forward,
            maturity,
            parameterization: SviParameterization::JumpWings,
            ..SviCalibrator::default()
        };

        let result = cal.calibrate(&quotes).expect("svi calibration succeeds");
        assert!(matches!(result.params, SviCalibrationParams::JumpWings(_)));
        assert!(
            result.diagnostics.fit_quality.liquid_rmse < 0.005,
            "liquid RMSE too high: {}",
            result.diagnostics.fit_quality.liquid_rmse
        );
    }
}
