//! Module `models::nig`.
//!
//! Implements nig abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Barndorff-Nielsen (1997), Cont and Tankov (2004), Hull (11th ed.) Ch. 19, NIG characteristic forms around Eq. (3.1).
//!
//! Key types and purpose: `Nig` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.

use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, InverseGaussian, StandardNormal};

use crate::engines::fft::carr_madan::CarrMadanParams;
use crate::engines::fft::char_fn::{CharacteristicFunction, NigCharFn};
use crate::engines::fft::frft::carr_madan_price_at_strikes;

/// Normal Inverse Gaussian (NIG) process parameters.
///
/// The NIG process is a subordinated Brownian motion:
///   `X(t) = beta * delta^2 * I(t) + delta * W(I(t))`
/// where `I(t)` is an inverse Gaussian process with parameters `(delta * t, delta^2 * t)`.
///
/// Parametrisation follows Barndorff-Nielsen (1997):
///   alpha: tail heaviness / steepness (alpha > 0)
///   beta:  asymmetry (-alpha < beta < alpha)
///   delta: scale (delta > 0)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nig {
    /// Tail heaviness parameter (alpha > 0, alpha > |beta|).
    pub alpha: f64,
    /// Asymmetry parameter (-alpha < beta < alpha).
    pub beta: f64,
    /// Scale parameter (delta > 0).
    pub delta: f64,
}

impl Nig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err("NIG requires alpha > 0".to_string());
        }
        if !self.beta.is_finite() || self.beta.abs() >= self.alpha {
            return Err("NIG requires |beta| < alpha".to_string());
        }
        if !self.delta.is_finite() || self.delta <= 0.0 {
            return Err("NIG requires delta > 0".to_string());
        }
        Ok(())
    }

    /// Helper: `gamma_bar = sqrt(alpha^2 - beta^2)`.
    fn gamma_bar(&self) -> f64 {
        (self.alpha * self.alpha - self.beta * self.beta).sqrt()
    }

    /// Risk-neutral drift correction (martingale adjustment).
    pub fn martingale_correction(&self) -> Result<f64, String> {
        self.validate()?;
        let gb = self.gamma_bar();
        // omega = delta * (sqrt(alpha^2 - (beta+1)^2) - sqrt(alpha^2 - beta^2))
        let beta_plus_1 = self.beta + 1.0;
        if beta_plus_1.abs() >= self.alpha {
            return Err("NIG martingale condition requires |beta + 1| < alpha".to_string());
        }
        let gb1 = (self.alpha * self.alpha - beta_plus_1 * beta_plus_1).sqrt();
        Ok(self.delta * (gb1 - gb))
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
        let cf = NigCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.alpha,
            self.beta,
            self.delta,
        )?;
        Ok(CharacteristicFunction::cf(&cf, u))
    }

    /// European call pricing via NIG characteristic function + Carr-Madan FFT.
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
        let cf = NigCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.alpha,
            self.beta,
            self.delta,
        )?;
        carr_madan_price_at_strikes(&cf, rate, maturity, spot, strikes, params)
    }

    /// Monte Carlo simulation via inverse Gaussian subordination.
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

        // Inverse Gaussian subordinator over dt: IG(mean = delta*dt/gamma_bar, shape = (delta*dt)^2),
        // so that Var(I_dt) = mean^3/shape = delta*dt/gamma_bar^3 scales linearly in dt.
        let gb = self.gamma_bar();
        let ig_mean = self.delta * dt / gb;
        let ig_shape = (self.delta * dt) * (self.delta * dt);

        let ig_dist = InverseGaussian::new(ig_mean, ig_shape)
            .map_err(|e| format!("InverseGaussian distribution: {e}"))?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(num_paths);

        for _ in 0..num_paths {
            let mut log_s = initial_spot.ln();
            for _ in 0..num_steps {
                let ig: f64 = ig_dist.sample(&mut rng);
                let z: f64 = StandardNormal.sample(&mut rng);
                log_s += drift_dt + self.beta * ig + ig.sqrt() * z;
            }
            out.push(log_s.exp());
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nig_cf_is_one_at_zero() {
        let nig = Nig {
            alpha: 15.0,
            beta: -5.0,
            delta: 0.5,
        };
        let cf_val = nig
            .characteristic_fn(Complex::new(0.0, 0.0), 100.0, 0.03, 0.0, 1.0)
            .unwrap();
        assert!((cf_val.re - 1.0).abs() < 1e-10);
        assert!(cf_val.im.abs() < 1e-10);
    }

    #[test]
    fn nig_fft_prices_match_independent_scipy_quadrature() {
        let nig = Nig {
            alpha: 15.0,
            beta: -5.0,
            delta: 0.5,
        };
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let prices = nig
            .european_calls_fft(100.0, &strikes, 0.03, 0.0, 0.5, CarrMadanParams::default())
            .unwrap();

        // Independently evaluated with SciPy 1.17.1 integrate.quad on the
        // continuous Carr-Madan integral (epsabs=epsrel=2e-12, upper tail
        // extended to 1000), rather than another FFT grid.
        let references = [
            12.894_367_932_776_754,
            9.188_650_543_941_383,
            6.124_743_548_019_659,
            3.800_296_619_395_705_7,
            2.200_950_921_022_865,
        ];
        for ((k, price), reference) in prices.iter().zip(references) {
            assert!(
                (price - reference).abs() <= 2.0e-10,
                "K={k}: fft={price}, scipy_quad={reference}"
            );
        }
    }

    #[test]
    fn nig_characteristic_function_satisfies_martingale_moment() {
        let nig = Nig {
            alpha: 15.0,
            beta: 0.0,
            delta: 0.5,
        };
        let spot: f64 = 100.0;
        let rate: f64 = 0.03;
        let dividend: f64 = 0.01;
        let maturity: f64 = 1.25;
        let moment = nig
            .characteristic_fn(Complex::new(0.0, -1.0), spot, rate, dividend, maturity)
            .unwrap();
        let forward_moment = spot * ((rate - dividend) * maturity).exp();
        assert!((moment.re - forward_moment).abs() <= 2.0e-12 * forward_moment);
        assert!(moment.im.abs() <= 2.0e-12 * forward_moment);
    }

    #[test]
    fn nig_mc_terminal_spots_are_positive() {
        let nig = Nig {
            alpha: 15.0,
            beta: -5.0,
            delta: 0.5,
        };
        let spots = nig
            .simulate_terminal_spots(100.0, 0.03, 0.0, 1.0, 50, 200, 42)
            .unwrap();
        assert_eq!(spots.len(), 200);
        assert!(spots.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn nig_mc_variance_matches_analytic_with_many_steps() {
        // Var(X_T) = delta*T*alpha^2/gamma^3 with gamma = sqrt(alpha^2 - beta^2).
        // With the wrong IG shape (delta^2*dt instead of (delta*dt)^2) the
        // beta^2*Var(I) contribution vanishes as steps grow (~11% low here).
        let nig = Nig {
            alpha: 15.0,
            beta: -5.0,
            delta: 0.5,
        };
        let (alpha, beta, delta) = (nig.alpha, nig.beta, nig.delta);
        let gamma = (alpha * alpha - beta * beta).sqrt();
        let horizon = 1.0;
        let var_analytic = delta * horizon * alpha * alpha / gamma.powi(3);

        let spots = nig
            .simulate_terminal_spots(100.0, 0.0, 0.0, horizon, 64, 40_000, 7)
            .unwrap();
        let logs: Vec<f64> = spots.iter().map(|s| (s / 100.0).ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        let var_mc =
            logs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (logs.len() - 1) as f64;

        let fourth_central =
            logs.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / logs.len() as f64;
        // Asymptotic standard error of the unbiased sample variance, with the
        // fourth central moment estimated from the same exact-terminal draws.
        let n = logs.len() as f64;
        let variance_stderr = ((fourth_central - ((n - 3.0) / (n - 1.0)) * var_mc * var_mc) / n)
            .max(0.0)
            .sqrt();
        assert!(
            (var_mc - var_analytic).abs() <= 4.0 * variance_stderr,
            "MC variance {var_mc} vs analytic {var_analytic}, variance_stderr={variance_stderr}"
        );

        let mean_analytic =
            nig.martingale_correction().unwrap() * horizon + beta * delta * horizon / gamma;
        let mean_stderr = (var_analytic / n).sqrt();
        assert!(
            (mean - mean_analytic).abs() <= 4.0 * mean_stderr,
            "MC mean {mean} vs analytic {mean_analytic}, mean_stderr={mean_stderr}"
        );
    }

    #[test]
    fn nig_validation_rejects_invalid_params() {
        let bad = Nig {
            alpha: 5.0,
            beta: 6.0, // |beta| > alpha
            delta: 0.5,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn nig_martingale_correction_matches_closed_form_value() {
        let nig = Nig {
            alpha: 15.0,
            beta: -5.0,
            delta: 0.5,
        };
        let omega = nig.martingale_correction().unwrap();
        assert!((omega - 0.157_348_335_535_004_94).abs() <= 2.0e-15);
    }
}
