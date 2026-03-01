//! Module `models::commodity`.
//!
//! Implements commodity workflows with concrete routines such as
//! `implied_convenience_yield`, `CommodityForwardCurve` interpolation,
//! seasonality handling, two-factor spread Monte Carlo, storage valuation,
//! and volume-constrained swing valuation utilities.
//!
//! References: Hull (11th ed.) Ch. 31, Schwartz (1997), and
//! Schwartz-Smith (2000) stochastic-factor dynamics.
//!
//! Numerical considerations: parameter admissibility constraints are essential
//! (positivity/integrability/stationarity) to avoid unstable simulation or
//! invalid estimates.
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::core::OptionType;

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

    let mut sorted = quotes.to_vec();
    sorted.sort_by(|a, b| a.maturity.total_cmp(&b.maturity));

    let mut out = Vec::with_capacity(sorted.len());
    let mut prev_t = 0.0;
    for q in &sorted {
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

/// Interpolation mode for commodity forward curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardInterpolation {
    /// Step function between listed contracts.
    PiecewiseFlat,
    /// Piecewise-linear interpolation.
    Linear,
    /// Natural cubic spline interpolation.
    CubicSpline,
}

/// Forward-curve shape classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveStructure {
    /// Forward levels are monotonically increasing with maturity.
    Contango,
    /// Forward levels are monotonically decreasing with maturity.
    Backwardation,
    /// Forward levels are effectively flat.
    Flat,
    /// Term structure changes sign in slope.
    Mixed,
}

/// Commodity forward curve bootstrapped from listed futures contracts.
///
/// The curve reproduces quoted futures exactly at contract maturities.
#[derive(Debug, Clone, PartialEq)]
pub struct CommodityForwardCurve {
    maturities: Vec<f64>,
    forwards: Vec<f64>,
    interpolation: ForwardInterpolation,
    spline_second_derivatives: Option<Vec<f64>>,
}

impl CommodityForwardCurve {
    /// Builds a curve from futures quotes with default linear interpolation.
    pub fn from_futures_quotes(quotes: &[FuturesQuote]) -> Result<Self, String> {
        Self::from_futures_quotes_with_interpolation(quotes, ForwardInterpolation::Linear)
    }

    /// Bootstraps a forward curve from futures quotes.
    pub fn bootstrap_from_futures(quotes: &[FuturesQuote]) -> Result<Self, String> {
        Self::from_futures_quotes(quotes)
    }

    /// Builds a curve from futures quotes with an explicit interpolation mode.
    pub fn from_futures_quotes_with_interpolation(
        quotes: &[FuturesQuote],
        interpolation: ForwardInterpolation,
    ) -> Result<Self, String> {
        if quotes.is_empty() {
            return Err("quotes cannot be empty".to_string());
        }

        let mut sorted = quotes.to_vec();
        sorted.sort_by(|a, b| a.maturity.total_cmp(&b.maturity));

        let mut maturities = Vec::with_capacity(sorted.len());
        let mut forwards = Vec::with_capacity(sorted.len());
        let mut prev_t = 0.0;

        for q in &sorted {
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

        let spline_second_derivatives = match interpolation {
            ForwardInterpolation::CubicSpline => {
                Some(natural_cubic_second_derivatives(&maturities, &forwards))
            }
            _ => None,
        };

        Ok(Self {
            maturities,
            forwards,
            interpolation,
            spline_second_derivatives,
        })
    }

    /// Returns a copy of this curve with a different interpolation mode.
    pub fn with_interpolation(&self, interpolation: ForwardInterpolation) -> Self {
        let spline_second_derivatives = match interpolation {
            ForwardInterpolation::CubicSpline => Some(natural_cubic_second_derivatives(
                &self.maturities,
                &self.forwards,
            )),
            _ => None,
        };

        Self {
            maturities: self.maturities.clone(),
            forwards: self.forwards.clone(),
            interpolation,
            spline_second_derivatives,
        }
    }

    pub fn maturities(&self) -> &[f64] {
        &self.maturities
    }

    pub fn forwards(&self) -> &[f64] {
        &self.forwards
    }

    pub fn interpolation(&self) -> ForwardInterpolation {
        self.interpolation
    }

    /// Curve lookup using the configured interpolation mode.
    pub fn forward(&self, maturity: f64) -> f64 {
        self.forward_with_method(maturity, self.interpolation)
    }

    /// Curve lookup using an explicit interpolation mode.
    pub fn forward_with_method(&self, maturity: f64, method: ForwardInterpolation) -> f64 {
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

        let hi = self
            .maturities
            .partition_point(|&x| x < maturity)
            .clamp(1, n - 1);
        let lo = hi - 1;

        let t0 = self.maturities[lo];
        let t1 = self.maturities[hi];
        let f0 = self.forwards[lo];
        let f1 = self.forwards[hi];

        // Exact contract-date reproduction.
        if (maturity - t0).abs() <= 1.0e-14 {
            return f0;
        }
        if (maturity - t1).abs() <= 1.0e-14 {
            return f1;
        }

        match method {
            ForwardInterpolation::PiecewiseFlat => f0,
            ForwardInterpolation::Linear => {
                let w = (maturity - t0) / (t1 - t0);
                f0 + w * (f1 - f0)
            }
            ForwardInterpolation::CubicSpline => {
                if let Some(m) = &self.spline_second_derivatives {
                    let h = t1 - t0;
                    let a = (t1 - maturity) / h;
                    let b = (maturity - t0) / h;
                    a * f0
                        + b * f1
                        + ((a * a * a - a) * m[lo] + (b * b * b - b) * m[hi]) * h * h / 6.0
                } else {
                    let w = (maturity - t0) / (t1 - t0);
                    f0 + w * (f1 - f0)
                }
            }
        }
    }

    /// Classifies the curve as contango, backwardation, flat, or mixed.
    pub fn structure(&self) -> CurveStructure {
        if self.forwards.len() < 2 {
            return CurveStructure::Flat;
        }

        let tol = 1.0e-10;
        let mut has_up = false;
        let mut has_down = false;

        for window in self.forwards.windows(2) {
            let diff = window[1] - window[0];
            if diff > tol {
                has_up = true;
            } else if diff < -tol {
                has_down = true;
            }
        }

        match (has_up, has_down) {
            (true, false) => CurveStructure::Contango,
            (false, true) => CurveStructure::Backwardation,
            (false, false) => CurveStructure::Flat,
            (true, true) => CurveStructure::Mixed,
        }
    }

    /// True when the curve is monotone-increasing with maturity.
    pub fn is_contango(&self) -> bool {
        self.structure() == CurveStructure::Contango
    }

    /// True when the curve is monotone-decreasing with maturity.
    pub fn is_backwardation(&self) -> bool {
        self.structure() == CurveStructure::Backwardation
    }

    /// Implied convenience yield at every listed contract.
    pub fn convenience_yield_curve(
        &self,
        spot: f64,
        risk_free_rate: f64,
        storage_cost: f64,
    ) -> Result<Vec<(f64, f64)>, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("spot must be finite and > 0".to_string());
        }
        let mut out = Vec::with_capacity(self.maturities.len());
        for i in 0..self.maturities.len() {
            let t = self.maturities[i];
            let fwd = self.forwards[i];
            let y = implied_convenience_yield(spot, fwd, risk_free_rate, storage_cost, t)
                .ok_or_else(|| "failed to infer convenience yield from curve node".to_string())?;
            out.push((t, y));
        }
        Ok(out)
    }
}

fn natural_cubic_second_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n <= 2 {
        return vec![0.0; n];
    }

    let mut lower = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    diag[0] = 1.0;
    diag[n - 1] = 1.0;

    for i in 1..(n - 1) {
        let h_prev = x[i] - x[i - 1];
        let h_next = x[i + 1] - x[i];
        lower[i] = h_prev;
        diag[i] = 2.0 * (h_prev + h_next);
        upper[i] = h_next;
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h_next - (y[i] - y[i - 1]) / h_prev);
    }

    // Thomas algorithm.
    for i in 1..n {
        let w = if diag[i - 1].abs() > 1.0e-16 {
            lower[i] / diag[i - 1]
        } else {
            0.0
        };
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    let mut m = vec![0.0; n];
    m[n - 1] = if diag[n - 1].abs() > 1.0e-16 {
        rhs[n - 1] / diag[n - 1]
    } else {
        0.0
    };

    for i in (0..(n - 1)).rev() {
        m[i] = if diag[i].abs() > 1.0e-16 {
            (rhs[i] - upper[i] * m[i + 1]) / diag[i]
        } else {
            0.0
        };
    }

    m
}

/// Seasonality model mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalityMode {
    /// Adjust level as `base + factor`.
    Additive,
    /// Adjust level as `base * factor`.
    Multiplicative,
}

/// Calendar month seasonality model.
///
/// Factors are indexed by month number (1 = January, ..., 12 = December).
#[derive(Debug, Clone, PartialEq)]
pub struct CommoditySeasonalityModel {
    mode: SeasonalityMode,
    monthly_factors: [f64; 12],
}

impl CommoditySeasonalityModel {
    /// Creates a seasonality model from monthly factors.
    pub fn from_monthly_factors(
        mode: SeasonalityMode,
        monthly_factors: [f64; 12],
    ) -> Result<Self, String> {
        for factor in monthly_factors {
            if !factor.is_finite() {
                return Err("monthly factors must be finite".to_string());
            }
            if matches!(mode, SeasonalityMode::Multiplicative) && factor <= 0.0 {
                return Err("multiplicative monthly factors must be > 0".to_string());
            }
        }

        Ok(Self {
            mode,
            monthly_factors,
        })
    }

    /// Additive seasonality constructor.
    pub fn additive(monthly_additives: [f64; 12]) -> Result<Self, String> {
        Self::from_monthly_factors(SeasonalityMode::Additive, monthly_additives)
    }

    /// Multiplicative seasonality constructor.
    pub fn multiplicative(monthly_multipliers: [f64; 12]) -> Result<Self, String> {
        Self::from_monthly_factors(SeasonalityMode::Multiplicative, monthly_multipliers)
    }

    /// Natural-gas style winter/summer seasonal pattern.
    ///
    /// Winter months are Nov-Mar, summer months are Jun-Sep, and shoulder is Apr-May/Oct.
    pub fn natural_gas_winter_summer(
        mode: SeasonalityMode,
        winter_factor: f64,
        summer_factor: f64,
        shoulder_factor: f64,
    ) -> Result<Self, String> {
        let winter = [11_u32, 12, 1, 2, 3];
        let summer = [6_u32, 7, 8, 9];

        let mut factors = [shoulder_factor; 12];
        for &m in &winter {
            factors[(m - 1) as usize] = winter_factor;
        }
        for &m in &summer {
            factors[(m - 1) as usize] = summer_factor;
        }

        Self::from_monthly_factors(mode, factors)
    }

    /// Default multiplicative natural-gas pattern.
    ///
    /// Winter: 1.18, Shoulder: 1.00, Summer: 0.88.
    pub fn default_natural_gas() -> Self {
        Self::natural_gas_winter_summer(SeasonalityMode::Multiplicative, 1.18, 0.88, 1.0)
            .expect("default natural-gas pattern should be valid")
    }

    pub fn mode(&self) -> SeasonalityMode {
        self.mode
    }

    pub fn monthly_factors(&self) -> &[f64; 12] {
        &self.monthly_factors
    }

    /// Returns the seasonal factor for a month number in `[1, 12]`.
    pub fn factor_for_month(&self, month: u32) -> Result<f64, String> {
        if !(1..=12).contains(&month) {
            return Err("month must be in [1, 12]".to_string());
        }
        Ok(self.monthly_factors[(month - 1) as usize])
    }

    /// Applies seasonality to a base level.
    pub fn apply(&self, base_level: f64, month: u32) -> Result<f64, String> {
        if !base_level.is_finite() {
            return Err("base_level must be finite".to_string());
        }
        let factor = self.factor_for_month(month)?;
        Ok(match self.mode {
            SeasonalityMode::Additive => base_level + factor,
            SeasonalityMode::Multiplicative => base_level * factor,
        })
    }

    /// Removes seasonality from observed levels.
    pub fn deseasonalise(&self, observations: &[(u32, f64)]) -> Result<Vec<f64>, String> {
        if observations.is_empty() {
            return Err("observations cannot be empty".to_string());
        }

        observations
            .iter()
            .map(|&(month, value)| {
                if !value.is_finite() || value <= 0.0 {
                    return Err("observed values must be finite and > 0".to_string());
                }
                let factor = self.factor_for_month(month)?;
                match self.mode {
                    SeasonalityMode::Additive => {
                        let adjusted = value - factor;
                        if adjusted <= 0.0 {
                            return Err(
                                "additive deseasonalised value must stay > 0 for log returns"
                                    .to_string(),
                            );
                        }
                        Ok(adjusted)
                    }
                    SeasonalityMode::Multiplicative => Ok(value / factor),
                }
            })
            .collect()
    }

    /// Computes deseasonalised log returns.
    pub fn deseasonalised_log_returns(
        &self,
        observations: &[(u32, f64)],
    ) -> Result<Vec<f64>, String> {
        let series = self.deseasonalise(observations)?;
        if series.len() < 2 {
            return Err("at least two observations are required".to_string());
        }

        let mut out = Vec::with_capacity(series.len() - 1);
        for window in series.windows(2) {
            out.push((window[1] / window[0]).ln());
        }
        Ok(out)
    }

    /// Estimates annualized volatility from deseasonalised log returns.
    pub fn estimate_deseasonalised_volatility(
        &self,
        observations: &[(u32, f64)],
        observations_per_year: f64,
    ) -> Result<f64, String> {
        if !observations_per_year.is_finite() || observations_per_year <= 0.0 {
            return Err("observations_per_year must be finite and > 0".to_string());
        }

        let returns = self.deseasonalised_log_returns(observations)?;
        sample_stddev(&returns).map(|sigma| sigma * observations_per_year.sqrt())
    }
}

fn sample_stddev(values: &[f64]) -> Result<f64, String> {
    if values.len() < 2 {
        return Err("at least two data points are required".to_string());
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(var.max(0.0).sqrt())
}

/// Per-commodity two-factor process parameters used in spread simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoFactorCommodityProcess {
    /// Mean-reversion speed of fast factor.
    pub kappa_fast: f64,
    /// Fast-factor volatility.
    pub sigma_fast: f64,
    /// Slow-factor (Brownian) volatility.
    pub sigma_slow: f64,
}

impl TwoFactorCommodityProcess {
    pub fn validate(&self) -> Result<(), String> {
        if !self.kappa_fast.is_finite() || self.kappa_fast <= 0.0 {
            return Err("kappa_fast must be finite and > 0".to_string());
        }
        if !self.sigma_fast.is_finite() || self.sigma_fast < 0.0 {
            return Err("sigma_fast must be finite and >= 0".to_string());
        }
        if !self.sigma_slow.is_finite() || self.sigma_slow < 0.0 {
            return Err("sigma_slow must be finite and >= 0".to_string());
        }
        Ok(())
    }

    fn variance_over(&self, maturity: f64) -> f64 {
        let fast_var =
            self.sigma_fast * self.sigma_fast * (1.0 - (-2.0 * self.kappa_fast * maturity).exp())
                / (2.0 * self.kappa_fast);
        let slow_var = self.sigma_slow * self.sigma_slow * maturity;
        fast_var + slow_var
    }
}

/// Two-factor spread model for commodity spread options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoFactorSpreadModel {
    /// Process for first leg.
    pub leg_1: TwoFactorCommodityProcess,
    /// Process for second leg.
    pub leg_2: TwoFactorCommodityProcess,
    /// Correlation between fast factors.
    pub rho_fast: f64,
    /// Correlation between slow factors.
    pub rho_slow: f64,
}

impl TwoFactorSpreadModel {
    pub fn validate(&self) -> Result<(), String> {
        self.leg_1.validate()?;
        self.leg_2.validate()?;
        if !self.rho_fast.is_finite() || !(-1.0..=1.0).contains(&self.rho_fast) {
            return Err("rho_fast must be finite and in [-1, 1]".to_string());
        }
        if !self.rho_slow.is_finite() || !(-1.0..=1.0).contains(&self.rho_slow) {
            return Err("rho_slow must be finite and in [-1, 1]".to_string());
        }
        Ok(())
    }

    /// Monte Carlo price for `max(q1*F1 - q2*F2 - K, 0)` or put analogue.
    #[allow(clippy::too_many_arguments)]
    pub fn price_spread_option_mc(
        &self,
        option_type: OptionType,
        forward_1: f64,
        forward_2: f64,
        strike: f64,
        quantity_1: f64,
        quantity_2: f64,
        risk_free_rate: f64,
        maturity: f64,
        num_paths: usize,
        seed: u64,
    ) -> Result<(f64, f64), String> {
        self.validate()?;

        if !forward_1.is_finite() || forward_1 <= 0.0 {
            return Err("forward_1 must be finite and > 0".to_string());
        }
        if !forward_2.is_finite() || forward_2 <= 0.0 {
            return Err("forward_2 must be finite and > 0".to_string());
        }
        if !strike.is_finite() || strike < 0.0 {
            return Err("strike must be finite and >= 0".to_string());
        }
        if !quantity_1.is_finite() || quantity_1 <= 0.0 {
            return Err("quantity_1 must be finite and > 0".to_string());
        }
        if !quantity_2.is_finite() || quantity_2 <= 0.0 {
            return Err("quantity_2 must be finite and > 0".to_string());
        }
        if !risk_free_rate.is_finite() {
            return Err("risk_free_rate must be finite".to_string());
        }
        if !maturity.is_finite() || maturity <= 0.0 {
            return Err("maturity must be finite and > 0".to_string());
        }
        if num_paths == 0 {
            return Err("num_paths must be > 0".to_string());
        }

        let fast_sd_1 = (self.leg_1.sigma_fast.powi(2)
            * (1.0 - (-2.0 * self.leg_1.kappa_fast * maturity).exp())
            / (2.0 * self.leg_1.kappa_fast))
            .max(0.0)
            .sqrt();
        let fast_sd_2 = (self.leg_2.sigma_fast.powi(2)
            * (1.0 - (-2.0 * self.leg_2.kappa_fast * maturity).exp())
            / (2.0 * self.leg_2.kappa_fast))
            .max(0.0)
            .sqrt();

        let slow_sd_1 = self.leg_1.sigma_slow * maturity.sqrt();
        let slow_sd_2 = self.leg_2.sigma_slow * maturity.sqrt();

        let var_1 = self.leg_1.variance_over(maturity);
        let var_2 = self.leg_2.variance_over(maturity);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut payoffs = Vec::with_capacity(num_paths);

        let corr_tail_fast = (1.0 - self.rho_fast * self.rho_fast).max(0.0).sqrt();
        let corr_tail_slow = (1.0 - self.rho_slow * self.rho_slow).max(0.0).sqrt();

        for _ in 0..num_paths {
            let z_fast_1: f64 = StandardNormal.sample(&mut rng);
            let z_fast_indep: f64 = StandardNormal.sample(&mut rng);
            let z_slow_1: f64 = StandardNormal.sample(&mut rng);
            let z_slow_indep: f64 = StandardNormal.sample(&mut rng);

            let z_fast_2 = self.rho_fast * z_fast_1 + corr_tail_fast * z_fast_indep;
            let z_slow_2 = self.rho_slow * z_slow_1 + corr_tail_slow * z_slow_indep;

            let log_return_1 = -0.5 * var_1 + fast_sd_1 * z_fast_1 + slow_sd_1 * z_slow_1;
            let log_return_2 = -0.5 * var_2 + fast_sd_2 * z_fast_2 + slow_sd_2 * z_slow_2;

            let f1_t = forward_1 * log_return_1.exp();
            let f2_t = forward_2 * log_return_2.exp();

            let spread = quantity_1 * f1_t - quantity_2 * f2_t - strike;
            let payoff = match option_type {
                OptionType::Call => spread.max(0.0),
                OptionType::Put => (-spread).max(0.0),
            };

            payoffs.push(payoff);
        }

        let n = num_paths as f64;
        let sum = payoffs.iter().sum::<f64>();
        let mean = sum / n;
        let var = if num_paths > 1 {
            payoffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        let discount = (-risk_free_rate * maturity).exp();
        Ok((discount * mean, discount * (var / n).sqrt()))
    }
}

/// Storage-contract settings for intrinsic and LSM valuation.
#[derive(Debug, Clone, PartialEq)]
pub struct CommodityStorageContract {
    /// Decision times in years (strictly increasing).
    pub decision_times: Vec<f64>,
    /// Minimum inventory bound.
    pub min_inventory: f64,
    /// Maximum inventory bound.
    pub max_inventory: f64,
    /// Initial inventory.
    pub initial_inventory: f64,
    /// Max injection volume per decision.
    pub max_injection: f64,
    /// Max withdrawal volume per decision.
    pub max_withdrawal: f64,
    /// Variable handling cost per moved unit.
    pub variable_cost: f64,
    /// Optional terminal inventory target.
    pub terminal_inventory_target: Option<f64>,
}

impl CommodityStorageContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.decision_times.is_empty() {
            return Err("decision_times cannot be empty".to_string());
        }
        if self
            .decision_times
            .iter()
            .any(|&t| !t.is_finite() || t <= 0.0)
        {
            return Err("decision_times must be finite and > 0".to_string());
        }
        if self
            .decision_times
            .windows(2)
            .any(|w| w[1] <= w[0] || !w[1].is_finite())
        {
            return Err("decision_times must be strictly increasing".to_string());
        }
        if !self.min_inventory.is_finite()
            || !self.max_inventory.is_finite()
            || self.max_inventory <= self.min_inventory
        {
            return Err(
                "inventory bounds must be finite with max_inventory > min_inventory".to_string(),
            );
        }
        if !self.initial_inventory.is_finite()
            || self.initial_inventory < self.min_inventory
            || self.initial_inventory > self.max_inventory
        {
            return Err("initial_inventory must lie within inventory bounds".to_string());
        }
        if !self.max_injection.is_finite() || self.max_injection <= 0.0 {
            return Err("max_injection must be finite and > 0".to_string());
        }
        if !self.max_withdrawal.is_finite() || self.max_withdrawal <= 0.0 {
            return Err("max_withdrawal must be finite and > 0".to_string());
        }
        if !self.variable_cost.is_finite() || self.variable_cost < 0.0 {
            return Err("variable_cost must be finite and >= 0".to_string());
        }
        if let Some(target) = self.terminal_inventory_target
            && (!target.is_finite() || target < self.min_inventory || target > self.max_inventory)
        {
            return Err(
                "terminal_inventory_target must be finite and inside inventory bounds".to_string(),
            );
        }
        Ok(())
    }
}

/// LSM simulation settings for storage extrinsic valuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StorageLsmConfig {
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Mean reversion of log spot residual around forwards.
    pub kappa: f64,
    /// Residual volatility.
    pub sigma: f64,
    /// RNG seed.
    pub seed: u64,
}

impl StorageLsmConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_paths < 8 {
            return Err("num_paths must be >= 8".to_string());
        }
        if !self.kappa.is_finite() || self.kappa <= 0.0 {
            return Err("kappa must be finite and > 0".to_string());
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err("sigma must be finite and >= 0".to_string());
        }
        Ok(())
    }
}

/// Storage valuation split into intrinsic and extrinsic components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StorageValuation {
    /// Deterministic intrinsic value under current forward curve.
    pub intrinsic: f64,
    /// Extrinsic value from stochastic optionality.
    pub extrinsic: f64,
    /// Total value (`intrinsic + extrinsic`).
    pub total: f64,
    /// Monte Carlo standard error on total.
    pub stderr: f64,
}

/// Values storage intrinsically and extrinsically via LSM.
pub fn value_storage_intrinsic_extrinsic(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    risk_free_rate: f64,
    inventory_grid: usize,
    lsm_config: StorageLsmConfig,
) -> Result<StorageValuation, String> {
    contract.validate()?;
    lsm_config.validate()?;
    if !risk_free_rate.is_finite() {
        return Err("risk_free_rate must be finite".to_string());
    }
    if inventory_grid < 3 {
        return Err("inventory_grid must be >= 3".to_string());
    }

    let intrinsic = intrinsic_storage_value(contract, curve, risk_free_rate, inventory_grid)?;
    let (lsm_total, stderr) =
        extrinsic_storage_value_lsm(contract, curve, risk_free_rate, inventory_grid, lsm_config)?;

    Ok(StorageValuation {
        intrinsic,
        extrinsic: lsm_total - intrinsic,
        total: lsm_total,
        stderr,
    })
}

/// Deterministic intrinsic storage value under the forward curve.
pub fn intrinsic_storage_value(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    risk_free_rate: f64,
    inventory_grid: usize,
) -> Result<f64, String> {
    contract.validate()?;
    if !risk_free_rate.is_finite() {
        return Err("risk_free_rate must be finite".to_string());
    }
    if inventory_grid < 3 {
        return Err("inventory_grid must be >= 3".to_string());
    }

    let grid = inventory_grid_values(
        contract.min_inventory,
        contract.max_inventory,
        inventory_grid,
    );
    let n_t = contract.decision_times.len();

    let mut next_values = vec![0.0_f64; grid.len()];
    let t_last = contract.decision_times[n_t - 1];
    let terminal_price = curve.forward(t_last);

    for (i, &inv) in grid.iter().enumerate() {
        let mut terminal = terminal_price * inv;
        if let Some(target) = contract.terminal_inventory_target {
            terminal -= 1.0e6 * (inv - target).abs();
        }
        next_values[i] = terminal;
    }

    for ti in (0..n_t).rev() {
        let t = contract.decision_times[ti];
        let dt = if ti + 1 < n_t {
            contract.decision_times[ti + 1] - t
        } else {
            0.0
        };
        let disc = (-risk_free_rate * dt).exp();
        let price = curve.forward(t);

        let mut current = vec![f64::NEG_INFINITY; grid.len()];

        for (inv_idx, &inv) in grid.iter().enumerate() {
            let min_next_inv = (inv - contract.max_withdrawal).max(contract.min_inventory);
            let max_next_inv = (inv + contract.max_injection).min(contract.max_inventory);

            for (next_idx, &next_inv) in grid.iter().enumerate() {
                if next_inv < min_next_inv - 1.0e-12 || next_inv > max_next_inv + 1.0e-12 {
                    continue;
                }

                let delta = next_inv - inv;
                let immediate_cash = -delta * price - contract.variable_cost * delta.abs();
                let total = immediate_cash + disc * next_values[next_idx];
                if total > current[inv_idx] {
                    current[inv_idx] = total;
                }
            }
        }

        next_values = current;
    }

    Ok(interpolate_inventory_value(
        contract.initial_inventory,
        &grid,
        &next_values,
    ))
}

fn extrinsic_storage_value_lsm(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    risk_free_rate: f64,
    inventory_grid: usize,
    lsm_config: StorageLsmConfig,
) -> Result<(f64, f64), String> {
    let grid = inventory_grid_values(
        contract.min_inventory,
        contract.max_inventory,
        inventory_grid,
    );
    let n_inv = grid.len();
    let n_t = contract.decision_times.len();

    let spots = simulate_storage_spot_paths(contract, curve, lsm_config)?;

    let mut values_next = vec![vec![0.0_f64; n_inv]; lsm_config.num_paths];

    let terminal_time = contract.decision_times[n_t - 1];
    let terminal_disc = (-risk_free_rate * terminal_time).exp();

    for p in 0..lsm_config.num_paths {
        let s_t = spots[p][n_t - 1];
        for (inv_idx, &inv) in grid[..n_inv].iter().enumerate() {
            let mut terminal = terminal_disc * s_t * inv;
            if let Some(target) = contract.terminal_inventory_target {
                terminal -= 1.0e6 * (inv - target).abs();
            }
            values_next[p][inv_idx] = terminal;
        }
    }

    for ti in (0..n_t).rev() {
        let t = contract.decision_times[ti];
        let dt = if ti + 1 < n_t {
            contract.decision_times[ti + 1] - t
        } else {
            0.0
        };
        let disc_step = (-risk_free_rate * dt).exp();

        let mut values_cur = vec![vec![f64::NEG_INFINITY; n_inv]; lsm_config.num_paths];

        for (inv_idx, &inv) in grid[..n_inv].iter().enumerate() {
            let min_next_inv = (inv - contract.max_withdrawal).max(contract.min_inventory);
            let max_next_inv = (inv + contract.max_injection).min(contract.max_inventory);

            let feasible_next: Vec<usize> = grid
                .iter()
                .enumerate()
                .filter_map(|(j, &x)| {
                    (x >= min_next_inv - 1.0e-12 && x <= max_next_inv + 1.0e-12).then_some(j)
                })
                .collect();

            if feasible_next.is_empty() {
                continue;
            }

            let x: Vec<f64> = spots.iter().map(|path| path[ti]).collect();

            let mut continuation_models = Vec::with_capacity(feasible_next.len());
            for &next_idx in &feasible_next {
                let y: Vec<f64> = (0..lsm_config.num_paths)
                    .map(|p| disc_step * values_next[p][next_idx])
                    .collect();
                continuation_models.push(regress_quadratic(&x, &y));
            }

            for p in 0..lsm_config.num_paths {
                let s = spots[p][ti];
                let mut best = f64::NEG_INFINITY;
                for (k, &next_idx) in feasible_next.iter().enumerate() {
                    let next_inv = grid[next_idx];
                    let delta = next_inv - inv;
                    let immediate = -delta * s - contract.variable_cost * delta.abs();

                    let cont_est = eval_quadratic(continuation_models[k], s);
                    let objective_est = immediate + cont_est;

                    if objective_est > best {
                        best = immediate + disc_step * values_next[p][next_idx];
                    }
                }
                values_cur[p][inv_idx] = best;
            }
        }

        values_next = values_cur;
    }

    let init_idx = nearest_inventory_index(contract.initial_inventory, &grid);
    let path_values: Vec<f64> = values_next.iter().map(|row| row[init_idx]).collect();
    let mean = path_values.iter().sum::<f64>() / lsm_config.num_paths as f64;
    let stderr = if lsm_config.num_paths > 1 {
        let var = path_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (lsm_config.num_paths as f64 - 1.0);
        (var / lsm_config.num_paths as f64).sqrt()
    } else {
        0.0
    };

    Ok((mean, stderr))
}

fn simulate_storage_spot_paths(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    lsm_config: StorageLsmConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let n_t = contract.decision_times.len();
    let mut paths = vec![vec![0.0_f64; n_t]; lsm_config.num_paths];

    let mut rng = StdRng::seed_from_u64(lsm_config.seed);

    for path in paths.iter_mut().take(lsm_config.num_paths) {
        let mut residual = 0.0_f64;
        let mut residual_var = 0.0_f64;

        for (i, &t) in contract.decision_times.iter().enumerate().take(n_t) {
            let dt = if i == 0 {
                t
            } else {
                t - contract.decision_times[i - 1]
            };
            let a = (-lsm_config.kappa * dt).exp();
            let var_step =
                lsm_config.sigma * lsm_config.sigma * (1.0 - a * a) / (2.0 * lsm_config.kappa);
            let z: f64 = StandardNormal.sample(&mut rng);
            residual = a * residual + var_step.max(0.0).sqrt() * z;
            residual_var = a * a * residual_var + var_step;

            let base_forward = curve.forward(t);
            path[i] = base_forward * (residual - 0.5 * residual_var).exp();
        }
    }

    Ok(paths)
}

fn regress_quadratic(x: &[f64], y: &[f64]) -> [f64; 3] {
    if x.len() != y.len() || x.len() < 3 {
        let mean = if y.is_empty() {
            0.0
        } else {
            y.iter().sum::<f64>() / y.len() as f64
        };
        return [mean, 0.0, 0.0];
    }

    let mut s1 = 0.0;
    let mut sx = 0.0;
    let mut sx2 = 0.0;
    let mut sx3 = 0.0;
    let mut sx4 = 0.0;
    let mut sy = 0.0;
    let mut sxy = 0.0;
    let mut sx2y = 0.0;

    for i in 0..x.len() {
        let xi = x[i];
        let yi = y[i];
        let xi2 = xi * xi;
        s1 += 1.0;
        sx += xi;
        sx2 += xi2;
        sx3 += xi2 * xi;
        sx4 += xi2 * xi2;
        sy += yi;
        sxy += xi * yi;
        sx2y += xi2 * yi;
    }

    let a = [[s1, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
    let b = [sy, sxy, sx2y];

    solve_3x3(a, b).unwrap_or_else(|| {
        let mean = sy / s1.max(1.0);
        [mean, 0.0, 0.0]
    })
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1.0e-14 {
        return None;
    }

    let det0 = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

    let det1 = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

    let det2 = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    Some([det0 / det, det1 / det, det2 / det])
}

fn eval_quadratic(beta: [f64; 3], x: f64) -> f64 {
    beta[0] + beta[1] * x + beta[2] * x * x
}

fn inventory_grid_values(min_inv: f64, max_inv: f64, grid: usize) -> Vec<f64> {
    let step = (max_inv - min_inv) / (grid - 1) as f64;
    (0..grid).map(|i| min_inv + i as f64 * step).collect()
}

fn nearest_inventory_index(value: f64, grid: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, &x) in grid.iter().enumerate() {
        let d = (value - x).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

fn interpolate_inventory_value(value: f64, grid: &[f64], values: &[f64]) -> f64 {
    if value <= grid[0] {
        return values[0];
    }
    let n = grid.len();
    if value >= grid[n - 1] {
        return values[n - 1];
    }

    let hi = grid.partition_point(|&x| x < value).clamp(1, n - 1);
    let lo = hi - 1;
    let w = (value - grid[lo]) / (grid[hi] - grid[lo]);
    values[lo] + w * (values[hi] - values[lo])
}

/// Volume-constrained commodity swing contract.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeConstrainedSwing {
    /// Exercise times in years.
    pub exercise_times: Vec<f64>,
    /// Per-period strike.
    pub strike: f64,
    /// Option type for per-period payoff.
    pub option_type: OptionType,
    /// Minimum exercised volume per period.
    pub min_period_volume: f64,
    /// Maximum exercised volume per period.
    pub max_period_volume: f64,
    /// Minimum cumulative exercised volume across all periods.
    pub min_total_volume: f64,
    /// Maximum cumulative exercised volume across all periods.
    pub max_total_volume: f64,
}

impl VolumeConstrainedSwing {
    pub fn validate(&self) -> Result<(), String> {
        if self.exercise_times.is_empty() {
            return Err("exercise_times cannot be empty".to_string());
        }
        if self
            .exercise_times
            .iter()
            .any(|&t| !t.is_finite() || t <= 0.0)
        {
            return Err("exercise_times must be finite and > 0".to_string());
        }
        if self
            .exercise_times
            .windows(2)
            .any(|w| !w[1].is_finite() || w[1] <= w[0])
        {
            return Err("exercise_times must be strictly increasing".to_string());
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err("strike must be finite and > 0".to_string());
        }
        if !self.min_period_volume.is_finite()
            || !self.max_period_volume.is_finite()
            || self.min_period_volume < 0.0
            || self.max_period_volume <= 0.0
            || self.max_period_volume < self.min_period_volume
        {
            return Err(
                "period volume bounds must satisfy 0 <= min_period_volume <= max_period_volume"
                    .to_string(),
            );
        }
        if !self.min_total_volume.is_finite()
            || !self.max_total_volume.is_finite()
            || self.min_total_volume < 0.0
            || self.max_total_volume < self.min_total_volume
        {
            return Err("total volume bounds are invalid".to_string());
        }

        let n = self.exercise_times.len() as f64;
        let min_feasible = n * self.min_period_volume;
        let max_feasible = n * self.max_period_volume;
        if self.min_total_volume > max_feasible + 1.0e-12
            || self.max_total_volume < min_feasible - 1.0e-12
        {
            return Err(
                "total volume constraints are infeasible with period volume limits".to_string(),
            );
        }

        Ok(())
    }

    /// Intrinsic swing value with explicit volume constraints.
    pub fn intrinsic_value(
        &self,
        curve: &CommodityForwardCurve,
        risk_free_rate: f64,
        total_volume_grid: usize,
    ) -> Result<f64, String> {
        self.validate()?;
        if !risk_free_rate.is_finite() {
            return Err("risk_free_rate must be finite".to_string());
        }
        if total_volume_grid < 3 {
            return Err("total_volume_grid must be >= 3".to_string());
        }

        let n_t = self.exercise_times.len();
        let grid = inventory_grid_values(
            self.min_total_volume,
            self.max_total_volume,
            total_volume_grid,
        );

        let mut next = vec![0.0_f64; grid.len()];

        for ti in (0..n_t).rev() {
            let t = self.exercise_times[ti];
            let dt = if ti + 1 < n_t {
                self.exercise_times[ti + 1] - t
            } else {
                0.0
            };
            let disc = (-risk_free_rate * dt).exp();
            let fwd = curve.forward(t);
            let payoff_per_unit = match self.option_type {
                OptionType::Call => (fwd - self.strike).max(0.0),
                OptionType::Put => (self.strike - fwd).max(0.0),
            };

            let mut current = vec![f64::NEG_INFINITY; grid.len()];
            for used_idx in 0..grid.len() {
                let used = grid[used_idx];
                for next_idx in 0..grid.len() {
                    let next_used = grid[next_idx];
                    let v = next_used - used;
                    if v < self.min_period_volume - 1.0e-12 || v > self.max_period_volume + 1.0e-12
                    {
                        continue;
                    }

                    let remaining = (n_t - ti - 1) as f64;
                    let min_remaining = remaining * self.min_period_volume;
                    let max_remaining = remaining * self.max_period_volume;

                    if next_used + min_remaining > self.max_total_volume + 1.0e-12 {
                        continue;
                    }
                    if next_used + max_remaining < self.min_total_volume - 1.0e-12 {
                        continue;
                    }

                    let immediate = v * payoff_per_unit;
                    let total = immediate + disc * next[next_idx];
                    if total > current[used_idx] {
                        current[used_idx] = total;
                    }
                }
            }

            next = current;
        }

        let start_volume = 0.0_f64.clamp(self.min_total_volume, self.max_total_volume);
        Ok(interpolate_inventory_value(start_volume, &grid, &next))
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

    #[test]
    fn piecewise_flat_and_spline_interpolation_work() {
        let quotes = [
            FuturesQuote {
                maturity: 0.25,
                price: 50.0,
            },
            FuturesQuote {
                maturity: 0.5,
                price: 54.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 60.0,
            },
        ];

        let flat = CommodityForwardCurve::from_futures_quotes_with_interpolation(
            &quotes,
            ForwardInterpolation::PiecewiseFlat,
        )
        .unwrap();
        let spline = CommodityForwardCurve::from_futures_quotes_with_interpolation(
            &quotes,
            ForwardInterpolation::CubicSpline,
        )
        .unwrap();

        assert_relative_eq!(flat.forward(0.49), 50.0, epsilon = 1.0e-12);
        assert_relative_eq!(flat.forward(0.5), 54.0, epsilon = 1.0e-12);

        for q in &quotes {
            assert_relative_eq!(spline.forward(q.maturity), q.price, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn curve_structure_detection_works() {
        let contango = CommodityForwardCurve::from_futures_quotes(&[
            FuturesQuote {
                maturity: 0.5,
                price: 100.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 102.0,
            },
        ])
        .unwrap();

        let backward = CommodityForwardCurve::from_futures_quotes(&[
            FuturesQuote {
                maturity: 0.5,
                price: 102.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 100.0,
            },
        ])
        .unwrap();

        assert_eq!(contango.structure(), CurveStructure::Contango);
        assert_eq!(backward.structure(), CurveStructure::Backwardation);
    }

    #[test]
    fn natural_gas_seasonality_highlights_winter() {
        let season = CommoditySeasonalityModel::default_natural_gas();
        let jan = season.apply(100.0, 1).unwrap();
        let jul = season.apply(100.0, 7).unwrap();
        assert!(jan > jul);
    }

    #[test]
    fn deseasonalised_vol_recovers_signal() {
        let season = CommoditySeasonalityModel::default_natural_gas();

        let mut obs = Vec::new();
        for i in 0..60 {
            let month = (i % 12 + 1) as u32;
            let base = 100.0 * (0.01 * i as f64).exp();
            let observed = season.apply(base, month).unwrap();
            obs.push((month, observed));
        }

        let vol = season
            .estimate_deseasonalised_volatility(&obs, 12.0)
            .unwrap();

        assert!(vol > 0.0);
        assert!(vol < 0.20);
    }

    #[test]
    fn two_factor_spread_mc_is_finite() {
        let model = TwoFactorSpreadModel {
            leg_1: TwoFactorCommodityProcess {
                kappa_fast: 2.5,
                sigma_fast: 0.18,
                sigma_slow: 0.12,
            },
            leg_2: TwoFactorCommodityProcess {
                kappa_fast: 2.0,
                sigma_fast: 0.16,
                sigma_slow: 0.10,
            },
            rho_fast: 0.55,
            rho_slow: 0.45,
        };

        let (price, stderr) = model
            .price_spread_option_mc(
                OptionType::Call,
                95.0,
                88.0,
                2.0,
                1.0,
                1.0,
                0.03,
                1.0,
                25_000,
                7,
            )
            .unwrap();

        assert!(price.is_finite() && price > 0.0);
        assert!(stderr.is_finite() && stderr > 0.0);
    }

    #[test]
    fn storage_intrinsic_is_finite() {
        let curve = CommodityForwardCurve::from_futures_quotes(&[
            FuturesQuote {
                maturity: 0.25,
                price: 40.0,
            },
            FuturesQuote {
                maturity: 0.5,
                price: 42.0,
            },
            FuturesQuote {
                maturity: 0.75,
                price: 45.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 47.0,
            },
        ])
        .unwrap();

        let contract = CommodityStorageContract {
            decision_times: vec![0.25, 0.5, 0.75, 1.0],
            min_inventory: 0.0,
            max_inventory: 100.0,
            initial_inventory: 50.0,
            max_injection: 25.0,
            max_withdrawal: 25.0,
            variable_cost: 0.2,
            terminal_inventory_target: Some(50.0),
        };

        let v = intrinsic_storage_value(&contract, &curve, 0.03, 21).unwrap();
        assert!(v.is_finite());
    }

    #[test]
    fn volume_constrained_swing_prices() {
        let curve = CommodityForwardCurve::from_futures_quotes(&[
            FuturesQuote {
                maturity: 0.25,
                price: 48.0,
            },
            FuturesQuote {
                maturity: 0.5,
                price: 50.0,
            },
            FuturesQuote {
                maturity: 0.75,
                price: 52.0,
            },
            FuturesQuote {
                maturity: 1.0,
                price: 55.0,
            },
        ])
        .unwrap();

        let swing = VolumeConstrainedSwing {
            exercise_times: vec![0.25, 0.5, 0.75, 1.0],
            strike: 49.0,
            option_type: OptionType::Call,
            min_period_volume: 0.0,
            max_period_volume: 10.0,
            min_total_volume: 5.0,
            max_total_volume: 25.0,
        };

        let value = swing.intrinsic_value(&curve, 0.03, 41).unwrap();
        assert!(value.is_finite());
        assert!(value >= 0.0);
    }
}
