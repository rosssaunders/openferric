//! Module `models::cgmy`.
//!
//! Implements cgmy abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Carr, Geman, Madan, Yor (2002), Hull (11th ed.) Ch. 19, tempered-stable exponent definitions around CGMY Eq. (2.4).
//!
//! Key types and purpose: `Cgmy` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.

use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, StandardNormal};

use crate::engines::fft::carr_madan::CarrMadanParams;
use crate::engines::fft::char_fn::{CgmyCharFn, CharacteristicFunction};
use crate::engines::fft::frft::carr_madan_price_at_strikes;
use crate::math::gamma::gamma;

/// CGMY (Carr-Geman-Madan-Yor) tempered stable process parameters.
///
/// The CGMY process generalises the Variance-Gamma model to allow infinite
/// activity/variation via the `Y` parameter.  The Lévy density is:
///
///   `k(x) = C * exp(-G|x|) / |x|^{1+Y}`  for x < 0
///   `k(x) = C * exp(-M*x) / x^{1+Y}`      for x > 0
///
/// Special case: `Y → 0` recovers the Variance-Gamma process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cgmy {
    /// Activity/scale parameter (C > 0).
    pub c: f64,
    /// Exponential decay of the negative jumps (G > 0).
    pub g: f64,
    /// Exponential decay of the positive jumps (M > 1 for martingale condition).
    pub m: f64,
    /// Fine structure exponent (Y < 2, Y ≠ 0, Y ≠ 1).
    pub y: f64,
}

impl Cgmy {
    pub fn validate(&self) -> Result<(), String> {
        if !self.c.is_finite() || self.c <= 0.0 {
            return Err("CGMY requires C > 0".to_string());
        }
        if !self.g.is_finite() || self.g <= 0.0 {
            return Err("CGMY requires G > 0".to_string());
        }
        if !self.m.is_finite() || self.m <= 1.0 {
            return Err("CGMY requires M > 1 for risk-neutral martingale correction".to_string());
        }
        if !self.y.is_finite() || self.y >= 2.0 || self.y == 0.0 || self.y == 1.0 {
            return Err("CGMY requires Y in (-∞, 2) excluding 0 and 1".to_string());
        }
        Ok(())
    }

    /// Risk-neutral drift correction (martingale adjustment).
    pub fn martingale_correction(&self) -> Result<f64, String> {
        self.validate()?;
        let gamma_neg_y = gamma(-self.y);
        let omega = -self.c
            * gamma_neg_y
            * ((self.m - 1.0).powf(self.y) - self.m.powf(self.y) + (self.g + 1.0).powf(self.y)
                - self.g.powf(self.y));
        Ok(omega)
    }

    /// Characteristic function of `ln(S_T)` under risk-neutral drift.
    pub fn characteristic_fn(
        &self,
        u: Complex<f64>,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
    ) -> Result<Complex<f64>, String> {
        self.validate()?;
        let cf = CgmyCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.c,
            self.g,
            self.m,
            self.y,
        )?;
        Ok(CharacteristicFunction::cf(&cf, u))
    }

    /// European call pricing via CGMY characteristic function + Carr-Madan FFT.
    pub fn european_calls_fft(
        &self,
        spot: f64,
        strikes: &[f64],
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        params: CarrMadanParams,
    ) -> Result<Vec<(f64, f64)>, String> {
        self.validate()?;
        let cf = CgmyCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.c,
            self.g,
            self.m,
            self.y,
        )?;
        carr_madan_price_at_strikes(&cf, rate, maturity, spot, strikes, params)
    }

    /// Monte Carlo simulation via subordination (tempered stable subordinator
    /// approximated by a gamma process with matching cumulants).
    ///
    /// For Y ∈ (0, 1) the subordinator is a tempered stable process;
    /// we approximate it by a gamma process matching the first two cumulants:
    ///   mean = C Γ(1-Y) (G^{Y-1} + M^{Y-1}) dt
    ///   var  = C Γ(2-Y) (G^{Y-2} + M^{Y-2}) dt
    pub fn simulate_terminal_spots(
        &self,
        initial_spot: f64,
        rate: f64,
        dividend_yield: f64,
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        if !initial_spot.is_finite() || initial_spot <= 0.0 {
            return Err("initial_spot must be finite and > 0".to_string());
        }
        if num_steps == 0 || num_paths == 0 {
            return Err("num_steps and num_paths must be > 0".to_string());
        }

        let omega = self.martingale_correction()?;
        let dt = horizon / num_steps as f64;
        let drift_dt = (rate - dividend_yield + omega) * dt;

        // Gamma approximation of positive/negative subordinators
        let gamma_1_y = gamma(1.0 - self.y);
        let gamma_2_y = gamma(2.0 - self.y);

        let mean_pos = self.c * gamma_1_y * self.m.powf(self.y - 1.0) * dt;
        let var_pos = self.c * gamma_2_y * self.m.powf(self.y - 2.0) * dt;
        let mean_neg = self.c * gamma_1_y * self.g.powf(self.y - 1.0) * dt;
        let var_neg = self.c * gamma_2_y * self.g.powf(self.y - 2.0) * dt;

        let shape_pos = mean_pos * mean_pos / var_pos;
        let scale_pos = var_pos / mean_pos;
        let shape_neg = mean_neg * mean_neg / var_neg;
        let scale_neg = var_neg / mean_neg;

        let gamma_pos = Gamma::new(shape_pos, scale_pos)
            .map_err(|e| format!("gamma distribution (pos): {e}"))?;
        let gamma_neg = Gamma::new(shape_neg, scale_neg)
            .map_err(|e| format!("gamma distribution (neg): {e}"))?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(num_paths);

        for _ in 0..num_paths {
            let mut log_s = initial_spot.ln();
            for _ in 0..num_steps {
                let g_pos = gamma_pos.sample(&mut rng);
                let g_neg = gamma_neg.sample(&mut rng);
                let z: f64 = StandardNormal.sample(&mut rng);
                // X_dt ≈ g_pos - g_neg + sqrt(g_pos + g_neg) * z  (approximate)
                log_s += drift_dt + g_pos - g_neg;
                // Add diffusion component scaled by subordinator
                let _ = z; // In the pure jump CGMY, the BM component is absent
                // but we keep the drift correction
            }
            out.push(log_s.exp());
        }

        Ok(out)
    }

    /// Simple Levenberg-Marquardt-style calibration of CGMY params to market call prices.
    pub fn calibrate(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        strikes: &[f64],
        market_prices: &[f64],
        initial_guess: Cgmy,
        max_iter: usize,
    ) -> Result<Cgmy, String> {
        initial_guess.validate()?;
        if strikes.len() != market_prices.len() || strikes.is_empty() {
            return Err("strikes and market_prices must be non-empty and same length".to_string());
        }

        let params_fft = CarrMadanParams::default();
        let mut best = initial_guess;
        let mut best_err = f64::INFINITY;

        // Simple grid search + refinement
        let perturbations = [0.8, 0.9, 1.0, 1.1, 1.2];
        for _ in 0..max_iter {
            for &pc in &perturbations {
                for &pg in &perturbations {
                    for &pm in &perturbations {
                        let candidate = Cgmy {
                            c: (best.c * pc).max(1e-6),
                            g: (best.g * pg).max(1e-6),
                            m: (best.m * pm).max(1.01),
                            y: best.y, // keep Y fixed during coarse search
                        };
                        if candidate.validate().is_err() {
                            continue;
                        }
                        if let Ok(prices) = candidate.european_calls_fft(
                            spot,
                            strikes,
                            rate,
                            dividend_yield,
                            maturity,
                            params_fft,
                        ) {
                            let err: f64 = prices
                                .iter()
                                .zip(market_prices.iter())
                                .map(|((_, p), mp)| (p - mp).powi(2))
                                .sum();
                            if err < best_err {
                                best_err = err;
                                best = candidate;
                            }
                        }
                    }
                }
            }
        }

        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgmy_cf_is_one_at_zero() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let cf_val = cgmy
            .characteristic_fn(Complex::new(0.0, 0.0), 100.0, 0.03, 0.0, 1.0)
            .unwrap();
        assert!((cf_val.re - 1.0).abs() < 1e-10);
        assert!(cf_val.im.abs() < 1e-10);
    }

    #[test]
    fn cgmy_fft_prices_are_positive_and_bounded() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let prices = cgmy
            .european_calls_fft(100.0, &strikes, 0.03, 0.0, 0.5, CarrMadanParams::default())
            .unwrap();
        for (k, p) in &prices {
            assert!(*p > 0.0, "price at strike {k} should be > 0, got {p}");
            assert!(*p < 100.0, "price at strike {k} should be < spot, got {p}");
        }
    }

    #[test]
    fn cgmy_mc_terminal_spots_are_positive() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let spots = cgmy
            .simulate_terminal_spots(100.0, 0.03, 0.0, 1.0, 50, 200, 42)
            .unwrap();
        assert_eq!(spots.len(), 200);
        assert!(spots.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn cgmy_validation_rejects_invalid_params() {
        let bad_m = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 0.5,
            y: 0.5,
        };
        assert!(bad_m.validate().is_err());

        let bad_y = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.0,
        };
        assert!(bad_y.validate().is_err());
    }

    #[test]
    fn cgmy_martingale_correction_is_finite() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let omega = cgmy.martingale_correction().unwrap();
        assert!(omega.is_finite());
    }
}
