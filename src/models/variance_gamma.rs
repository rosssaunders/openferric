//! Variance-Gamma (VG) model utilities: characteristic function pricing and path simulation.
//!
//! [`VarianceGamma`] encodes `(sigma, theta, nu)` for the time-changed Brownian process
//! `X_t = theta*G_t + sigma*W_{G_t}`, with martingale drift correction for risk-neutral pricing.
//! The module supports Carr-Madan FFT call pricing via characteristic functions and
//! Monte Carlo simulation using gamma increments.
//! References: Madan and Seneta (1990); Madan, Carr, and Chang (1998); Carr and Madan (1999).
//! Numerical checks enforce the VG martingale condition `1-theta*nu-0.5*sigma^2*nu>0`
//! and strict positivity for spot/maturity/path counts.
//! Use this module for jump/skew modeling beyond lognormal diffusion assumptions.

use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, StandardNormal};

use crate::engines::fft::carr_madan::CarrMadanParams;
use crate::engines::fft::char_fn::VarianceGammaCharFn;
use crate::engines::fft::frft::carr_madan_price_at_strikes;

/// Variance-Gamma (VG) process parameters.
///
/// `X(t) = theta * G(t) + sigma * W(G(t))`, where `G` is a gamma process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceGamma {
    /// Diffusion volatility parameter.
    pub sigma: f64,
    /// Asymmetry/drift in the subordinator time.
    pub theta: f64,
    /// Variance-rate parameter of the gamma clock.
    pub nu: f64,
}

impl VarianceGamma {
    pub fn validate(&self) -> Result<(), String> {
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err("variance-gamma sigma must be finite and >= 0".to_string());
        }
        if !self.theta.is_finite() {
            return Err("variance-gamma theta must be finite".to_string());
        }
        if !self.nu.is_finite() || self.nu <= 0.0 {
            return Err("variance-gamma nu must be finite and > 0".to_string());
        }
        Ok(())
    }

    /// Instantaneous variance proxy `Var[X(1)] = sigma^2 + theta^2 * nu`.
    pub fn variance_rate(&self) -> f64 {
        self.sigma * self.sigma + self.theta * self.theta * self.nu
    }

    /// Risk-neutral drift correction `omega` from martingale condition.
    pub fn martingale_correction(&self) -> Result<f64, String> {
        self.validate()?;
        let inner = 1.0 - self.theta * self.nu - 0.5 * self.sigma * self.sigma * self.nu;
        if inner <= 0.0 {
            return Err(
                "variance-gamma martingale condition violated: 1 - theta*nu - 0.5*sigma^2*nu must be > 0"
                    .to_string(),
            );
        }
        Ok(inner.ln() / self.nu)
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
        let cf = VarianceGammaCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.sigma,
            self.theta,
            self.nu,
        )?;
        Ok(crate::engines::fft::char_fn::CharacteristicFunction::cf(
            &cf, u,
        ))
    }

    /// European call pricing via VG characteristic function + Carr-Madan FFT.
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
        let cf = VarianceGammaCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.sigma,
            self.theta,
            self.nu,
        )?;
        carr_madan_price_at_strikes(&cf, rate, maturity, spot, strikes, params)
    }

    /// Simulates one VG spot path under risk-neutral dynamics.
    pub fn simulate_path(
        &self,
        initial_spot: f64,
        rate: f64,
        dividend_yield: f64,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        if !initial_spot.is_finite() || initial_spot <= 0.0 {
            return Err("initial_spot must be finite and > 0".to_string());
        }
        if !rate.is_finite() || !dividend_yield.is_finite() {
            return Err("rate and dividend_yield must be finite".to_string());
        }
        if !horizon.is_finite() || horizon < 0.0 {
            return Err("horizon must be finite and >= 0".to_string());
        }
        if num_steps == 0 {
            return Err("num_steps must be > 0".to_string());
        }

        let omega = self.martingale_correction()?;
        let dt = horizon / num_steps as f64;
        let drift_dt = (rate - dividend_yield + omega) * dt;
        let gamma_dist = Gamma::new(dt / self.nu, self.nu)
            .map_err(|e| format!("failed to build gamma increment distribution: {e}"))?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut log_s = initial_spot.ln();
        let mut path = Vec::with_capacity(num_steps + 1);
        path.push(initial_spot);

        for _ in 0..num_steps {
            let g = gamma_dist.sample(&mut rng);
            let z: f64 = StandardNormal.sample(&mut rng);
            log_s += drift_dt + self.theta * g + self.sigma * g.sqrt() * z;
            path.push(log_s.exp());
        }

        Ok(path)
    }

    /// Simulates terminal spots for Monte Carlo pricing.
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
        if !rate.is_finite() || !dividend_yield.is_finite() {
            return Err("rate and dividend_yield must be finite".to_string());
        }
        if !horizon.is_finite() || horizon < 0.0 {
            return Err("horizon must be finite and >= 0".to_string());
        }
        if num_steps == 0 || num_paths == 0 {
            return Err("num_steps and num_paths must be > 0".to_string());
        }

        let omega = self.martingale_correction()?;
        let dt = horizon / num_steps as f64;
        let drift_dt = (rate - dividend_yield + omega) * dt;
        let gamma_dist = Gamma::new(dt / self.nu, self.nu)
            .map_err(|e| format!("failed to build gamma increment distribution: {e}"))?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(num_paths);

        for _ in 0..num_paths {
            let mut log_s = initial_spot.ln();
            for _ in 0..num_steps {
                let g = gamma_dist.sample(&mut rng);
                let z: f64 = StandardNormal.sample(&mut rng);
                log_s += drift_dt + self.theta * g + self.sigma * g.sqrt() * z;
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
    fn variance_rate_is_non_negative() {
        let vg = VarianceGamma {
            sigma: 0.2,
            theta: -0.1,
            nu: 0.3,
        };
        assert!(vg.variance_rate() > 0.0);
    }

    #[test]
    fn simulation_returns_requested_length() {
        let vg = VarianceGamma {
            sigma: 0.2,
            theta: -0.1,
            nu: 0.2,
        };
        let terminals = vg
            .simulate_terminal_spots(100.0, 0.02, 0.0, 1.0, 64, 100, 7)
            .unwrap();
        assert_eq!(terminals.len(), 100);
        assert!(terminals.iter().all(|x| x.is_finite() && *x > 0.0));
    }
}
