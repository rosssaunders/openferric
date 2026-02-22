//! Commodity factor models and convenience-yield term-structure helpers.
//!
//! Implements exact-step simulation for the one-factor Schwartz log-spot OU model
//! and the two-factor Schwartz-Smith decomposition (`ln S = chi + xi`) with correlated shocks.
//! Key types are [`SchwartzOneFactor`], [`SchwartzSmithTwoFactor`], and [`CommodityForwardCurve`].
//! References: Schwartz (1997); Schwartz and Smith (2000); Gibson and Schwartz (1990).
//! Numerical design uses closed-form transition moments (not Euler) and positivity-preserving
//! spot reconstruction, with strict input validation for maturities/vols/correlation.
//! Companion utilities infer convenience yields from futures quotes and build interpolated forward curves.
//! Use this module for commodity dynamics/calibration inputs; product payoff schemas live in `instruments::commodity`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Exchange futures quote at a given maturity (in years).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuturesQuote {
    pub maturity: f64,
    pub price: f64,
}

/// One-factor Schwartz commodity spot model in spot form:
/// `dS = kappa * (mu - ln(S)) * S dt + sigma * S dW`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwartzOneFactor {
    pub kappa: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl SchwartzOneFactor {
    pub fn validate(&self) -> Result<(), String> {
        if !self.kappa.is_finite() || self.kappa <= 0.0 {
            return Err("kappa must be finite and > 0".to_string());
        }
        if !self.mu.is_finite() {
            return Err("mu must be finite".to_string());
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err("sigma must be finite and >= 0".to_string());
        }
        Ok(())
    }

    /// Long-run mean of `ln(S_t)` under this specification.
    pub fn long_run_log_mean(&self) -> f64 {
        self.mu - 0.5 * self.sigma * self.sigma / self.kappa
    }

    /// Exact step for `ln(S)` (Ornstein-Uhlenbeck with Ito adjustment).
    pub fn step_log_exact(&self, log_spot: f64, dt: f64, z: f64) -> Result<f64, String> {
        self.validate()?;
        if !log_spot.is_finite() {
            return Err("log_spot must be finite".to_string());
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err("dt must be finite and >= 0".to_string());
        }
        if !z.is_finite() {
            return Err("z must be finite".to_string());
        }

        if dt == 0.0 {
            return Ok(log_spot);
        }

        let kdt = self.kappa * dt;
        let exp_kdt = (-kdt).exp();
        let variance = (1.0 - (-2.0 * kdt).exp()) / (2.0 * self.kappa);
        let vol = self.sigma * variance.max(0.0).sqrt();
        let mean = self.long_run_log_mean() + (log_spot - self.long_run_log_mean()) * exp_kdt;

        Ok(mean + vol * z)
    }

    /// Exact step in spot space.
    pub fn step_exact(&self, spot: f64, dt: f64, z: f64) -> Result<f64, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        let next_log = self.step_log_exact(spot.ln(), dt, z)?;
        Ok(next_log.exp())
    }

    /// Simulates one spot path including the initial level.
    pub fn simulate_path(
        &self,
        initial_spot: f64,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        if !initial_spot.is_finite() || initial_spot <= 0.0 {
            return Err("initial_spot must be finite and > 0".to_string());
        }
        if !horizon.is_finite() || horizon < 0.0 {
            return Err("horizon must be finite and >= 0".to_string());
        }
        if num_steps == 0 {
            return Err("num_steps must be > 0".to_string());
        }

        let dt = horizon / num_steps as f64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut log_s = initial_spot.ln();
        let mut path = Vec::with_capacity(num_steps + 1);
        path.push(initial_spot);

        for _ in 0..num_steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            log_s = self.step_log_exact(log_s, dt, z)?;
            path.push(log_s.exp());
        }

        Ok(path)
    }

    /// Simulates terminal spots for Monte Carlo use.
    pub fn simulate_terminal_spots(
        &self,
        initial_spot: f64,
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        if !initial_spot.is_finite() || initial_spot <= 0.0 {
            return Err("initial_spot must be finite and > 0".to_string());
        }
        if !horizon.is_finite() || horizon < 0.0 {
            return Err("horizon must be finite and >= 0".to_string());
        }
        if num_steps == 0 || num_paths == 0 {
            return Err("num_steps and num_paths must be > 0".to_string());
        }

        let dt = horizon / num_steps as f64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut terminals = Vec::with_capacity(num_paths);

        for _ in 0..num_paths {
            let mut log_s = initial_spot.ln();
            for _ in 0..num_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                log_s = self.step_log_exact(log_s, dt, z)?;
            }
            terminals.push(log_s.exp());
        }

        Ok(terminals)
    }
}

/// Schwartz-Smith two-factor commodity model where `ln(S) = chi + xi`.
///
/// - `chi`: short-term mean-reverting factor.
/// - `xi`: long-term GBM log-factor (arithmetic Brownian in logs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwartzSmithTwoFactor {
    pub kappa: f64,
    pub sigma_chi: f64,
    pub mu_xi: f64,
    pub sigma_xi: f64,
    pub rho: f64,
}

impl SchwartzSmithTwoFactor {
    pub fn validate(&self) -> Result<(), String> {
        if !self.kappa.is_finite() || self.kappa <= 0.0 {
            return Err("kappa must be finite and > 0".to_string());
        }
        if !self.sigma_chi.is_finite() || self.sigma_chi < 0.0 {
            return Err("sigma_chi must be finite and >= 0".to_string());
        }
        if !self.mu_xi.is_finite() {
            return Err("mu_xi must be finite".to_string());
        }
        if !self.sigma_xi.is_finite() || self.sigma_xi < 0.0 {
            return Err("sigma_xi must be finite and >= 0".to_string());
        }
        if !self.rho.is_finite() || !(-1.0..=1.0).contains(&self.rho) {
            return Err("rho must be finite and in [-1, 1]".to_string());
        }
        Ok(())
    }

    /// Exact step for `(chi, xi)` using correlated Gaussian shocks.
    pub fn step_exact(
        &self,
        chi: f64,
        xi: f64,
        dt: f64,
        z1: f64,
        z2: f64,
    ) -> Result<(f64, f64), String> {
        self.validate()?;
        if !chi.is_finite() || !xi.is_finite() {
            return Err("chi and xi must be finite".to_string());
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err("dt must be finite and >= 0".to_string());
        }
        if !z1.is_finite() || !z2.is_finite() {
            return Err("z1 and z2 must be finite".to_string());
        }
        if dt == 0.0 {
            return Ok((chi, xi));
        }

        let exp_kdt = (-self.kappa * dt).exp();
        let chi_var = (1.0 - (-2.0 * self.kappa * dt).exp()) / (2.0 * self.kappa);
        let chi_next = chi * exp_kdt + self.sigma_chi * chi_var.max(0.0).sqrt() * z1;

        let z_corr = self.rho * z1 + (1.0 - self.rho * self.rho).max(0.0).sqrt() * z2;
        let xi_next = xi + self.mu_xi * dt + self.sigma_xi * dt.sqrt() * z_corr;

        Ok((chi_next, xi_next))
    }

    pub fn spot_from_factors(chi: f64, xi: f64) -> f64 {
        (chi + xi).exp()
    }

    /// Simulates one `(chi, xi, spot)` path including initial state.
    pub fn simulate_path(
        &self,
        initial_chi: f64,
        initial_xi: f64,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> Result<Vec<(f64, f64, f64)>, String> {
        self.validate()?;
        if !initial_chi.is_finite() || !initial_xi.is_finite() {
            return Err("initial factors must be finite".to_string());
        }
        if !horizon.is_finite() || horizon < 0.0 {
            return Err("horizon must be finite and >= 0".to_string());
        }
        if num_steps == 0 {
            return Err("num_steps must be > 0".to_string());
        }

        let dt = horizon / num_steps as f64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut chi = initial_chi;
        let mut xi = initial_xi;

        let mut path = Vec::with_capacity(num_steps + 1);
        path.push((chi, xi, Self::spot_from_factors(chi, xi)));

        for _ in 0..num_steps {
            let z1: f64 = StandardNormal.sample(&mut rng);
            let z2: f64 = StandardNormal.sample(&mut rng);
            (chi, xi) = self.step_exact(chi, xi, dt, z1, z2)?;
            path.push((chi, xi, Self::spot_from_factors(chi, xi)));
        }

        Ok(path)
    }
}

/// Implied convenience yield from a single futures quote under
/// `F(T) = S0 * exp((r + u - y) * T)`.
pub fn implied_convenience_yield(
    spot: f64,
    futures_price: f64,
    risk_free_rate: f64,
    storage_cost: f64,
    maturity: f64,
) -> Option<f64> {
    if !spot.is_finite()
        || !futures_price.is_finite()
        || !risk_free_rate.is_finite()
        || !storage_cost.is_finite()
        || !maturity.is_finite()
        || spot <= 0.0
        || futures_price <= 0.0
        || maturity <= 0.0
    {
        return None;
    }

    Some(risk_free_rate + storage_cost - (futures_price / spot).ln() / maturity)
}

/// Convenience yield term structure inferred from futures quotes.
pub fn convenience_yield_from_term_structure(
    spot: f64,
    quotes: &[FuturesQuote],
    risk_free_rate: f64,
    storage_cost: f64,
) -> Result<Vec<(f64, f64)>, String> {
    if quotes.is_empty() {
        return Err("quotes cannot be empty".to_string());
    }

    let mut out = Vec::with_capacity(quotes.len());
    let mut prev_t = 0.0;
    for q in quotes {
        if !q.maturity.is_finite() || q.maturity <= 0.0 || q.maturity <= prev_t {
            return Err("quote maturities must be finite, > 0 and strictly increasing".to_string());
        }
        if !q.price.is_finite() || q.price <= 0.0 {
            return Err("quote prices must be finite and > 0".to_string());
        }

        let y = implied_convenience_yield(spot, q.price, risk_free_rate, storage_cost, q.maturity)
            .ok_or_else(|| "failed to infer convenience yield".to_string())?;
        out.push((q.maturity, y));
        prev_t = q.maturity;
    }

    Ok(out)
}

/// Piecewise-linear forward curve built from futures prices.
#[derive(Debug, Clone, PartialEq)]
pub struct CommodityForwardCurve {
    maturities: Vec<f64>,
    forwards: Vec<f64>,
}

impl CommodityForwardCurve {
    pub fn from_futures_quotes(quotes: &[FuturesQuote]) -> Result<Self, String> {
        if quotes.is_empty() {
            return Err("quotes cannot be empty".to_string());
        }

        let mut maturities = Vec::with_capacity(quotes.len());
        let mut forwards = Vec::with_capacity(quotes.len());
        let mut prev_t = 0.0;

        for q in quotes {
            if !q.maturity.is_finite() || q.maturity <= 0.0 || q.maturity <= prev_t {
                return Err(
                    "quote maturities must be finite, > 0 and strictly increasing".to_string(),
                );
            }
            if !q.price.is_finite() || q.price <= 0.0 {
                return Err("quote prices must be finite and > 0".to_string());
            }
            maturities.push(q.maturity);
            forwards.push(q.price);
            prev_t = q.maturity;
        }

        Ok(Self {
            maturities,
            forwards,
        })
    }

    pub fn maturities(&self) -> &[f64] {
        &self.maturities
    }

    pub fn forwards(&self) -> &[f64] {
        &self.forwards
    }

    /// Linear interpolation with flat extrapolation.
    pub fn forward(&self, maturity: f64) -> f64 {
        if self.maturities.is_empty() || !maturity.is_finite() {
            return f64::NAN;
        }

        if maturity <= self.maturities[0] {
            return self.forwards[0];
        }

        let n = self.maturities.len();
        if maturity >= self.maturities[n - 1] {
            return self.forwards[n - 1];
        }

        let mut hi = 1;
        while hi < n && self.maturities[hi] < maturity {
            hi += 1;
        }
        let lo = hi - 1;

        let t0 = self.maturities[lo];
        let t1 = self.maturities[hi];
        let f0 = self.forwards[lo];
        let f1 = self.forwards[hi];
        let w = (maturity - t0) / (t1 - t0);

        f0 + w * (f1 - f0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn implied_convenience_yield_matches_manual_formula() {
        let y = implied_convenience_yield(100.0, 102.0, 0.05, 0.01, 1.0).unwrap();
        assert_relative_eq!(y, 0.040_197_372_7, epsilon = 1e-10);
    }

    #[test]
    fn forward_curve_interpolates() {
        let curve = CommodityForwardCurve::from_futures_quotes(&[
            FuturesQuote {
                maturity: 0.5,
                price: 98.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 102.0,
            },
        ])
        .unwrap();

        assert_relative_eq!(curve.forward(0.5), 98.0, epsilon = 1e-12);
        assert_relative_eq!(curve.forward(0.75), 100.0, epsilon = 1e-12);
        assert_relative_eq!(curve.forward(1.5), 102.0, epsilon = 1e-12);
    }
}
