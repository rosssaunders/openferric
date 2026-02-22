//! Heston calibration via Carr-Madan FFT pricing and constrained LM.
//!
//! References:
//! - Heston (1993), closed-form characteristic-function model.
//! - Carr and Madan (1999), FFT option pricing.
//! - Gatheral (2006), parameter admissibility and smile behavior.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::calibration::core::{
    BoxConstraints, CalibrationResult, Calibrator, finite_metric, matrix_condition_number,
    matrix_to_rows, sanitize_convergence,
};
use crate::calibration::diagnostics::diagnostics;
use crate::calibration::instruments::{OptionVolQuote, make_error_record};
use crate::calibration::optimizers::{LmOptions, levenberg_marquardt};
use crate::engines::fft::{CarrMadanContext, CarrMadanParams, HestonCharFn};
use crate::pricing::OptionType;
use crate::vol::implied::implied_vol;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HestonCalibrationParams {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub sigma_v: f64,
    pub rho: f64,
}

impl HestonCalibrationParams {
    pub fn to_vec(self) -> Vec<f64> {
        vec![self.v0, self.kappa, self.theta, self.sigma_v, self.rho]
    }

    pub fn from_slice(x: &[f64]) -> Result<Self, String> {
        if x.len() != 5 {
            return Err("heston parameter vector must have length 5".to_string());
        }
        Ok(Self {
            v0: x[0],
            kappa: x[1],
            theta: x[2],
            sigma_v: x[3],
            rho: x[4],
        })
    }
}

#[derive(Debug, Clone)]
pub struct HestonCalibrator {
    pub spot: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub bounds: BoxConstraints,
    pub lm_options: LmOptions,
    pub fft_params: CarrMadanParams,
}

impl Default for HestonCalibrator {
    fn default() -> Self {
        Self {
            spot: 100.0,
            rate: 0.02,
            dividend_yield: 0.0,
            bounds: BoxConstraints::new(
                vec![1e-4, 1e-3, 1e-4, 1e-3, -0.999],
                vec![2.0, 20.0, 2.0, 5.0, 0.999],
            )
            .expect("valid Heston bounds"),
            lm_options: LmOptions {
                max_iterations: 28,
                ..LmOptions::default()
            },
            fft_params: CarrMadanParams {
                n: 2048,
                eta: 0.2,
                alpha: 1.25,
            },
        }
    }
}

impl HestonCalibrator {
    fn initial_guess(&self, instruments: &[OptionVolQuote]) -> Vec<f64> {
        let atm = instruments
            .iter()
            .min_by(|a, b| {
                (a.strike - self.spot)
                    .abs()
                    .total_cmp(&(b.strike - self.spot).abs())
            })
            .map(|q| q.market_vol)
            .unwrap_or(0.2)
            .max(1e-4);

        self.bounds.clamp(&[atm * atm, 1.5, atm * atm, 0.6, -0.5])
    }

    fn model_vols(
        &self,
        params: HestonCalibrationParams,
        instruments: &[OptionVolQuote],
    ) -> Option<Vec<f64>> {
        let mut out = vec![0.0; instruments.len()];
        let mut by_expiry: BTreeMap<u64, Vec<usize>> = BTreeMap::new();

        for (idx, q) in instruments.iter().enumerate() {
            by_expiry.entry(q.maturity.to_bits()).or_default().push(idx);
        }

        for (expiry_bits, idxs) in by_expiry {
            let expiry = f64::from_bits(expiry_bits);
            let strikes: Vec<f64> = idxs.iter().map(|i| instruments[*i].strike).collect();

            let cf = HestonCharFn::new(
                self.spot,
                self.rate,
                self.dividend_yield,
                expiry,
                params.v0,
                params.kappa,
                params.theta,
                params.sigma_v,
                params.rho,
            );

            let ctx =
                CarrMadanContext::new(&cf, self.rate, expiry, self.spot, self.fft_params).ok()?;
            let prices = ctx.price_strikes(&strikes).ok()?;

            for (local, (_, call_price)) in prices.into_iter().enumerate() {
                let quote_idx = idxs[local];
                let q = &instruments[quote_idx];
                let prepaid_spot = self.spot * (-self.dividend_yield * q.maturity).exp();
                let iv = implied_vol(
                    OptionType::Call,
                    prepaid_spot.max(1e-8),
                    q.strike.max(1e-8),
                    self.rate,
                    q.maturity,
                    call_price.max(1e-12),
                    1e-10,
                    64,
                )
                .ok()?;
                out[quote_idx] = iv.max(1e-8);
            }
        }

        Some(out)
    }

    fn residuals(&self, x: &[f64], instruments: &[OptionVolQuote]) -> Vec<f64> {
        let Ok(params) = HestonCalibrationParams::from_slice(x) else {
            return vec![1e6; instruments.len()];
        };

        let Some(model_vols) = self.model_vols(params, instruments) else {
            return vec![1e6; instruments.len()];
        };

        model_vols
            .iter()
            .zip(instruments.iter())
            .map(|(model, q)| {
                let err = make_error_record(q, *model);
                err.effective_error * q.weight.max(1e-12).sqrt()
            })
            .collect()
    }
}

impl Calibrator<HestonCalibrationParams> for HestonCalibrator {
    type Instrument = OptionVolQuote;

    fn name(&self) -> &'static str {
        "heston"
    }

    fn calibrate(
        &self,
        instruments: &[Self::Instrument],
    ) -> Result<CalibrationResult<HestonCalibrationParams>, String> {
        if instruments.is_empty() {
            return Err("heston calibration requires non-empty instrument set".to_string());
        }
        if instruments.iter().any(|q| {
            q.strike <= 0.0
                || q.maturity <= 0.0
                || q.market_vol <= 0.0
                || !q.market_vol.is_finite()
                || !q.strike.is_finite()
                || !q.maturity.is_finite()
        }) {
            return Err("invalid Heston quote set".to_string());
        }

        let initial = self.initial_guess(instruments);
        let opt = levenberg_marquardt(&initial, &self.bounds, self.lm_options, |x| {
            self.residuals(x, instruments)
        })?;

        let params = HestonCalibrationParams::from_slice(&opt.x)?;
        let model_vols = self
            .model_vols(params, instruments)
            .ok_or_else(|| "failed to evaluate calibrated Heston model vols".to_string())?;

        let errors: Vec<_> = instruments
            .iter()
            .zip(model_vols.iter())
            .map(|(q, m)| make_error_record(q, *m))
            .collect();

        let condition_number = finite_metric(matrix_condition_number(&opt.jacobian));
        let convergence = sanitize_convergence(opt.convergence);
        let diagnostics = diagnostics(
            &errors,
            &convergence,
            condition_number,
            Some(&self.bounds),
            Some(&opt.x),
            None,
        );

        Ok(CalibrationResult {
            params,
            objective: opt.objective,
            per_instrument_error: errors,
            jacobian: matrix_to_rows(&opt.jacobian),
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

    fn synthetic_surface(
        cal: &HestonCalibrator,
        params: HestonCalibrationParams,
    ) -> Vec<OptionVolQuote> {
        let maturities = [0.25, 0.5, 0.75, 1.0, 2.0];
        let multipliers = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4];

        let mut quotes = Vec::new();
        for t in maturities {
            let strikes: Vec<f64> = multipliers.iter().map(|m| cal.spot * *m).collect();
            let cf = HestonCharFn::new(
                cal.spot,
                cal.rate,
                cal.dividend_yield,
                t,
                params.v0,
                params.kappa,
                params.theta,
                params.sigma_v,
                params.rho,
            );
            let ctx = CarrMadanContext::new(&cf, cal.rate, t, cal.spot, cal.fft_params)
                .expect("context builds");
            let prices = ctx.price_strikes(&strikes).expect("pricing succeeds");

            for (idx, (k, px)) in prices.iter().enumerate() {
                let prepaid_spot = cal.spot * (-cal.dividend_yield * t).exp();
                let iv = implied_vol(
                    OptionType::Call,
                    prepaid_spot,
                    *k,
                    cal.rate,
                    t,
                    *px,
                    1e-10,
                    64,
                )
                .expect("iv inversion succeeds");

                let mut q = OptionVolQuote::new(format!("T{t:.2}_K{idx}"), *k, t, iv);
                q.bid_vol = Some((iv - 0.0005).max(1e-6));
                q.ask_vol = Some(iv + 0.0005);
                q.weight = if (k / cal.spot - 1.0).abs() <= 0.2 {
                    1.5
                } else {
                    1.0
                };
                q.liquid = (k / cal.spot - 1.0).abs() <= 0.2;
                quotes.push(q);
            }
        }
        quotes
    }

    #[test]
    fn calibrates_heston_surface_with_sub_half_vol_point_error() {
        let cal = HestonCalibrator::default();
        let true_params = HestonCalibrationParams {
            v0: 0.045,
            kappa: 2.1,
            theta: 0.04,
            sigma_v: 0.55,
            rho: -0.6,
        };

        let quotes = synthetic_surface(&cal, true_params);
        assert_eq!(quotes.len(), 50);

        let t0 = Instant::now();
        let result = cal.calibrate(&quotes).expect("calibration succeeds");
        let elapsed = t0.elapsed().as_secs_f64();

        assert!(
            result.diagnostics.fit_quality.liquid_rmse < 0.005,
            "liquid RMSE too high: {}",
            result.diagnostics.fit_quality.liquid_rmse
        );

        // Target from issue #55 for a 50-point surface (release mode).
        // In debug mode with parallel tests, allow more headroom.
        assert!(elapsed < 10.0, "heston calibration took {elapsed:.4}s");
    }
}
