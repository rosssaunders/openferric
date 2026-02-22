//! Module `vol::forward`.
//!
//! Forward-variance and term-structure utilities derived from implied-vol surfaces.
//!
//! This module fills the gap between static implied-volatility surfaces and
//! stochastic-volatility dynamics by exposing:
//! - ATM forward-variance curves `w(T) = sigma_atm(T)^2 T`,
//! - forward volatility between two maturities,
//! - forward variance-swap fair levels and PV,
//! - VIX-style model-free index replication from a surface,
//! - ATM skew term structures,
//! - Heston/SABR vol-of-vol term-structure containers.
//!
//! References:
//! - CBOE VIX White Paper, model-free variance-swap replication.
//! - Gatheral (2006), total variance and smile dynamics.
//! - Hagan et al. (2002), SABR parameter interpretation.
//!
//! Numerical notes:
//! - Total variance is expected to be non-decreasing in expiry at ATM.
//! - All APIs guard against non-finite and negative variances.
//! - VIX integration uses a finite strike grid; tighter tolerances require
//!   wider and denser strike coverage.

use crate::pricing::OptionType;
use crate::pricing::european::black_76_price;
use crate::vol::sabr::SabrParams;

/// Minimal implied-vol surface view required for forward-variance analytics.
///
/// Implement this for any surface representation that can return:
/// - implied vol at `(strike, expiry)`,
/// - forward level at `expiry`,
/// - the native expiry grid.
pub trait ForwardVarianceSource {
    /// Implied volatility at strike/expiry.
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64;

    /// Forward level used to define ATM at a given expiry.
    fn forward_price(&self, expiry: f64) -> f64;

    /// Native expiry grid for the source surface.
    fn expiries(&self) -> &[f64];
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Node on an ATM forward-variance curve.
///
/// - `expiry`: node time in years.
/// - `total_variance`: ATM total variance `w(T) = sigma_atm(T)^2 T`.
/// - `forward_variance`: interval forward variance over `(T_prev, T]`.
pub struct ForwardVariancePoint {
    pub expiry: f64,
    pub total_variance: f64,
    pub forward_variance: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Piecewise-linear ATM total-variance curve with interval forward variances.
pub struct ForwardVarianceCurve {
    points: Vec<ForwardVariancePoint>,
}

impl ForwardVarianceCurve {
    /// Builds a curve from `(expiry, total_variance)` points.
    ///
    /// The points are sorted by expiry. Calendar-arbitrage consistency is
    /// enforced: total variance must be non-decreasing (within a small tolerance).
    pub fn new(mut total_variance_points: Vec<(f64, f64)>) -> Result<Self, String> {
        if total_variance_points.is_empty() {
            return Err("forward variance curve requires at least one point".to_string());
        }

        total_variance_points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if total_variance_points
            .windows(2)
            .any(|w| w[1].0 <= w[0].0 + 1.0e-14)
        {
            return Err("forward variance curve expiries must be strictly increasing".to_string());
        }

        if total_variance_points
            .iter()
            .any(|(t, w)| !t.is_finite() || !w.is_finite() || *t <= 0.0 || *w < 0.0)
        {
            return Err(
                "forward variance curve points must be finite with expiry > 0 and variance >= 0"
                    .to_string(),
            );
        }

        let mut points = Vec::with_capacity(total_variance_points.len());
        let mut prev_t = 0.0;
        let mut prev_w = 0.0;
        let tol = 1.0e-10;

        for (expiry, total_variance) in total_variance_points {
            if total_variance + tol < prev_w {
                return Err(format!(
                    "calendar arbitrage at expiry {expiry}: total variance {total_variance} < previous {prev_w}"
                ));
            }

            let dt = expiry - prev_t;
            let mut fwd = (total_variance - prev_w) / dt;
            if fwd < 0.0 && fwd.abs() <= tol {
                fwd = 0.0;
            }
            if fwd < 0.0 {
                return Err(format!(
                    "negative forward variance segment ending at expiry {expiry}: {fwd}"
                ));
            }

            points.push(ForwardVariancePoint {
                expiry,
                total_variance,
                forward_variance: fwd,
            });
            prev_t = expiry;
            prev_w = total_variance;
        }

        Ok(Self { points })
    }

    /// Extracts ATM total variance from a surface on the provided expiry grid.
    ///
    /// ATM is defined as `strike = forward_price(expiry)`.
    pub fn from_surface<S: ForwardVarianceSource>(
        surface: &S,
        expiries: &[f64],
    ) -> Result<Self, String> {
        if expiries.is_empty() {
            return Err("expiry grid cannot be empty".to_string());
        }

        let mut points = Vec::with_capacity(expiries.len());
        for &expiry in expiries {
            if !expiry.is_finite() || expiry <= 0.0 {
                return Err("all expiries must be finite and > 0".to_string());
            }
            let atm = surface.forward_price(expiry);
            if !atm.is_finite() || atm <= 0.0 {
                return Err(format!("invalid forward price at expiry {expiry}: {atm}"));
            }
            let vol = surface.implied_vol(atm, expiry);
            if !vol.is_finite() || vol <= 0.0 {
                return Err(format!("invalid implied vol at expiry {expiry}: {vol}"));
            }
            points.push((expiry, vol * vol * expiry));
        }

        Self::new(points)
    }

    /// Extracts ATM total variance on the source surface's native expiry grid.
    pub fn from_surface_expiries<S: ForwardVarianceSource>(surface: &S) -> Result<Self, String> {
        Self::from_surface(surface, surface.expiries())
    }

    /// Curve nodes.
    pub fn points(&self) -> &[ForwardVariancePoint] {
        &self.points
    }

    /// Node expiries.
    pub fn expiries(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.expiry).collect()
    }

    /// ATM total variance `w(T)` via piecewise-linear interpolation in time.
    ///
    /// For `T <= first_expiry`, interpolation is anchored at `(0, 0)`.
    /// For `T > last_expiry`, linear extrapolation uses the last forward variance.
    pub fn total_variance(&self, expiry: f64) -> f64 {
        if self.points.is_empty() || expiry <= 0.0 {
            return 0.0;
        }

        let t = expiry;
        let first = self.points[0];
        if t <= first.expiry {
            return first.total_variance * (t / first.expiry);
        }

        for i in 0..self.points.len() - 1 {
            let p0 = self.points[i];
            let p1 = self.points[i + 1];
            if t <= p1.expiry {
                let w = (t - p0.expiry) / (p1.expiry - p0.expiry);
                return p0.total_variance + (p1.total_variance - p0.total_variance) * w;
            }
        }

        let last = self.points[self.points.len() - 1];
        let prev = if self.points.len() >= 2 {
            self.points[self.points.len() - 2]
        } else {
            ForwardVariancePoint {
                expiry: 0.0,
                total_variance: 0.0,
                forward_variance: last.total_variance / last.expiry.max(1e-12),
            }
        };
        let slope = (last.total_variance - prev.total_variance) / (last.expiry - prev.expiry);
        (last.total_variance + slope * (t - last.expiry)).max(0.0)
    }

    /// Forward variance over `[t1, t2]`, i.e. `(w(t2) - w(t1)) / (t2 - t1)`.
    pub fn forward_variance(&self, t1: f64, t2: f64) -> Result<f64, String> {
        if !t1.is_finite() || !t2.is_finite() || t1 < 0.0 || t2 <= t1 {
            return Err("require finite times with 0 <= t1 < t2".to_string());
        }
        let w1 = self.total_variance(t1);
        let w2 = self.total_variance(t2);
        let fwd = (w2 - w1) / (t2 - t1);
        if fwd < -1.0e-10 {
            return Err(format!(
                "negative forward variance between {t1} and {t2}: {fwd}"
            ));
        }
        Ok(fwd.max(0.0))
    }

    /// Forward volatility over `[t1, t2]`.
    pub fn forward_vol(&self, t1: f64, t2: f64) -> Result<f64, String> {
        Ok(self.forward_variance(t1, t2)?.sqrt())
    }

    /// Fair forward variance-swap strike in variance units.
    pub fn fair_forward_variance_swap(&self, start: f64, end: f64) -> Result<f64, String> {
        self.forward_variance(start, end)
    }

    /// Fair forward variance-swap strike in volatility units.
    pub fn fair_forward_vol_swap(&self, start: f64, end: f64) -> Result<f64, String> {
        Ok(self.fair_forward_variance_swap(start, end)?.sqrt())
    }

    /// Present value of a forward variance swap.
    ///
    /// Formula:
    /// `PV = variance_notional * (fair_variance - strike_variance) * exp(-r * end)`
    pub fn price_forward_variance_swap(
        &self,
        start: f64,
        end: f64,
        strike_variance: f64,
        variance_notional: f64,
        risk_free_rate: f64,
    ) -> Result<f64, String> {
        if strike_variance < 0.0 || variance_notional < 0.0 || !risk_free_rate.is_finite() {
            return Err(
                "strike variance, notional, and rate must be finite and non-negative".to_string(),
            );
        }

        let fair = self.fair_forward_variance_swap(start, end)?;
        let df = (-risk_free_rate * end).exp();
        Ok(variance_notional * (fair - strike_variance) * df)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// ATM skew node with derivative in log-moneyness.
///
/// `skew = d sigma(K, T) / d ln(K/F(T)) |_{K = F(T)}`.
pub struct AtmSkewPoint {
    pub expiry: f64,
    pub skew: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// ATM skew term structure sampled across expiries.
pub struct AtmSkewTermStructure {
    points: Vec<AtmSkewPoint>,
}

impl AtmSkewTermStructure {
    /// Computes ATM skew term structure from a surface using central differences.
    pub fn from_surface<S: ForwardVarianceSource>(
        surface: &S,
        expiries: &[f64],
    ) -> Result<Self, String> {
        if expiries.is_empty() {
            return Err("expiry grid cannot be empty".to_string());
        }

        let eps = 1.0e-3_f64;
        let mut points = Vec::with_capacity(expiries.len());
        for &expiry in expiries {
            if expiry <= 0.0 || !expiry.is_finite() {
                return Err("all expiries must be finite and > 0".to_string());
            }

            let fwd = surface.forward_price(expiry);
            if !fwd.is_finite() || fwd <= 0.0 {
                return Err(format!("invalid forward price at expiry {expiry}: {fwd}"));
            }

            let k_dn = fwd * (-eps).exp();
            let k_up = fwd * eps.exp();
            let iv_dn = surface.implied_vol(k_dn, expiry);
            let iv_up = surface.implied_vol(k_up, expiry);
            if !iv_dn.is_finite() || !iv_up.is_finite() {
                return Err(format!(
                    "non-finite implied vol around ATM at expiry {expiry}"
                ));
            }

            let skew = (iv_up - iv_dn) / (2.0 * eps);
            points.push(AtmSkewPoint { expiry, skew });
        }

        points.sort_by(|a, b| a.expiry.total_cmp(&b.expiry));
        if points.windows(2).any(|w| w[1].expiry <= w[0].expiry) {
            return Err("ATM skew expiries must be strictly increasing".to_string());
        }

        Ok(Self { points })
    }

    /// Computes ATM skew term structure on the source native expiry grid.
    pub fn from_surface_expiries<S: ForwardVarianceSource>(surface: &S) -> Result<Self, String> {
        Self::from_surface(surface, surface.expiries())
    }

    /// Term-structure nodes.
    pub fn points(&self) -> &[AtmSkewPoint] {
        &self.points
    }

    /// Linearly interpolated ATM skew at expiry.
    pub fn skew(&self, expiry: f64) -> f64 {
        interpolate_piecewise(
            expiry,
            &self.points.iter().map(|p| p.expiry).collect::<Vec<_>>(),
            &self.points.iter().map(|p| p.skew).collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Heston vol-of-vol (`sigma_v`/`xi`) node by expiry.
pub struct HestonVolOfVolPoint {
    pub expiry: f64,
    pub sigma_v: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Heston vol-of-vol term structure.
pub struct HestonVolOfVolTermStructure {
    points: Vec<HestonVolOfVolPoint>,
}

impl HestonVolOfVolTermStructure {
    /// Creates a term structure from `(expiry, sigma_v)` nodes.
    pub fn new(mut points: Vec<HestonVolOfVolPoint>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("heston vol-of-vol term structure requires at least one point".to_string());
        }

        points.sort_by(|a, b| a.expiry.total_cmp(&b.expiry));
        if points.windows(2).any(|w| w[1].expiry <= w[0].expiry) {
            return Err("heston vol-of-vol expiries must be strictly increasing".to_string());
        }
        if points.iter().any(|p| {
            !p.expiry.is_finite() || p.expiry <= 0.0 || !p.sigma_v.is_finite() || p.sigma_v < 0.0
        }) {
            return Err(
                "heston vol-of-vol points must be finite with expiry > 0 and sigma_v >= 0"
                    .to_string(),
            );
        }

        Ok(Self { points })
    }

    /// Underlying nodes.
    pub fn points(&self) -> &[HestonVolOfVolPoint] {
        &self.points
    }

    /// Interpolated `sigma_v` at expiry.
    pub fn sigma_v(&self, expiry: f64) -> f64 {
        interpolate_piecewise(
            expiry,
            &self.points.iter().map(|p| p.expiry).collect::<Vec<_>>(),
            &self.points.iter().map(|p| p.sigma_v).collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// SABR vol-of-vol node by expiry.
///
/// `alpha` is included alongside `nu` so the same structure can feed
/// per-expiry SABR initialization/calibration.
pub struct SabrVolOfVolPoint {
    pub expiry: f64,
    pub alpha: f64,
    pub nu: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// SABR `(alpha, nu)` term structure by expiry.
pub struct SabrVolOfVolTermStructure {
    points: Vec<SabrVolOfVolPoint>,
}

impl SabrVolOfVolTermStructure {
    /// Creates a SABR term structure from explicit points.
    pub fn new(mut points: Vec<SabrVolOfVolPoint>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("sabr vol-of-vol term structure requires at least one point".to_string());
        }

        points.sort_by(|a, b| a.expiry.total_cmp(&b.expiry));
        if points.windows(2).any(|w| w[1].expiry <= w[0].expiry) {
            return Err("sabr term-structure expiries must be strictly increasing".to_string());
        }
        if points.iter().any(|p| {
            !p.expiry.is_finite()
                || p.expiry <= 0.0
                || !p.alpha.is_finite()
                || p.alpha <= 0.0
                || !p.nu.is_finite()
                || p.nu < 0.0
        }) {
            return Err(
                "sabr points must be finite with expiry > 0, alpha > 0, and nu >= 0".to_string(),
            );
        }

        Ok(Self { points })
    }

    /// Builds the structure from calibrated SABR parameters by expiry.
    pub fn from_sabr_params(points: &[(f64, SabrParams)]) -> Result<Self, String> {
        let mut ts = Vec::with_capacity(points.len());
        for (expiry, params) in points {
            ts.push(SabrVolOfVolPoint {
                expiry: *expiry,
                alpha: params.alpha,
                nu: params.nu,
            });
        }
        Self::new(ts)
    }

    /// Underlying nodes.
    pub fn points(&self) -> &[SabrVolOfVolPoint] {
        &self.points
    }

    /// Interpolated SABR alpha at expiry.
    pub fn alpha(&self, expiry: f64) -> f64 {
        interpolate_piecewise(
            expiry,
            &self.points.iter().map(|p| p.expiry).collect::<Vec<_>>(),
            &self.points.iter().map(|p| p.alpha).collect::<Vec<_>>(),
        )
    }

    /// Interpolated SABR nu at expiry.
    pub fn nu(&self, expiry: f64) -> f64 {
        interpolate_piecewise(
            expiry,
            &self.points.iter().map(|p| p.expiry).collect::<Vec<_>>(),
            &self.points.iter().map(|p| p.nu).collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Settings for CBOE-style VIX replication from a volatility surface.
pub struct VixSettings {
    /// Target tenor in calendar days (`30` for standard VIX).
    pub target_days: f64,
    /// Number of strike points in integration grid (auto-adjusted to odd and >= 11).
    pub strike_count: usize,
    /// Half-width of strike grid in log-moneyness, centered on forward.
    pub log_moneyness_span: f64,
}

impl Default for VixSettings {
    fn default() -> Self {
        Self {
            target_days: 30.0,
            strike_count: 1501,
            log_moneyness_span: 2.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Output of a VIX-style calculation.
pub struct VixStyleIndex {
    pub target_days: f64,
    pub near_expiry: f64,
    pub next_expiry: f64,
    pub near_variance: f64,
    pub next_variance: f64,
    pub target_variance: f64,
    pub index: f64,
}

/// Computes a CBOE-style VIX index from an implied-vol surface.
///
/// The algorithm:
/// 1. Selects the two expiries bracketing the target tenor.
/// 2. Computes model-free variance for each expiry using OTM option prices.
/// 3. Interpolates in variance-time to the target tenor.
/// 4. Returns `100 * sqrt(target_variance / target_years)`.
pub fn vix_style_index_from_surface<S: ForwardVarianceSource>(
    surface: &S,
    risk_free_rate: f64,
    settings: VixSettings,
) -> Result<VixStyleIndex, String> {
    if !risk_free_rate.is_finite() {
        return Err("risk-free rate must be finite".to_string());
    }
    if settings.target_days <= 0.0 || !settings.target_days.is_finite() {
        return Err("target_days must be finite and > 0".to_string());
    }

    let target_t = settings.target_days / 365.0;
    let expiries = surface.expiries();
    if expiries.len() < 2 {
        return Err("VIX calculation requires at least two expiries".to_string());
    }
    if expiries.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return Err("surface expiries must be finite and > 0".to_string());
    }

    let (near_idx, next_idx) = bracketing_expiry_indices(expiries, target_t)?;
    let t1 = expiries[near_idx];
    let t2 = expiries[next_idx];

    let var1 = model_free_variance_for_expiry(surface, t1, risk_free_rate, settings)?;
    let var2 = model_free_variance_for_expiry(surface, t2, risk_free_rate, settings)?;

    let target_var_time = if (t2 - t1).abs() < 1.0e-12 {
        t1 * var1
    } else {
        t1 * var1 * (t2 - target_t) / (t2 - t1) + t2 * var2 * (target_t - t1) / (t2 - t1)
    };
    let target_variance = (target_var_time / target_t).max(0.0);
    let index = 100.0 * target_variance.sqrt();

    Ok(VixStyleIndex {
        target_days: settings.target_days,
        near_expiry: t1,
        next_expiry: t2,
        near_variance: var1,
        next_variance: var2,
        target_variance,
        index,
    })
}

fn bracketing_expiry_indices(expiries: &[f64], target: f64) -> Result<(usize, usize), String> {
    if expiries.len() < 2 {
        return Err("need at least two expiries".to_string());
    }

    if target <= expiries[0] {
        return Ok((0, 1));
    }
    if target >= expiries[expiries.len() - 1] {
        let n = expiries.len();
        return Ok((n - 2, n - 1));
    }

    let mut next = 1;
    for (i, t) in expiries.iter().enumerate() {
        if *t >= target {
            next = i;
            break;
        }
    }
    let near = next.saturating_sub(1);
    Ok((near, next))
}

fn model_free_variance_for_expiry<S: ForwardVarianceSource>(
    surface: &S,
    expiry: f64,
    rate: f64,
    settings: VixSettings,
) -> Result<f64, String> {
    let fwd = surface.forward_price(expiry);
    if !fwd.is_finite() || fwd <= 0.0 {
        return Err(format!("invalid forward at expiry {expiry}: {fwd}"));
    }

    let mut n = settings.strike_count.max(11);
    if n % 2 == 0 {
        n += 1;
    }
    let span = settings.log_moneyness_span.max(0.3);
    let dk = 2.0 * span / (n - 1) as f64;

    let mut strikes = Vec::with_capacity(n);
    let mut q_otm = Vec::with_capacity(n);
    for i in 0..n {
        let k_log = -span + i as f64 * dk;
        let strike = fwd * k_log.exp();
        let vol = surface.implied_vol(strike, expiry).max(1e-8);
        if !vol.is_finite() {
            return Err(format!(
                "non-finite implied vol at expiry {expiry}, strike {strike}"
            ));
        }

        let call = black_76_price(OptionType::Call, fwd, strike, rate, vol, expiry).max(0.0);
        let put = black_76_price(OptionType::Put, fwd, strike, rate, vol, expiry).max(0.0);

        strikes.push(strike);
        q_otm.push((call, put));
    }

    let mut k0_idx = 0usize;
    for (i, k) in strikes.iter().enumerate() {
        if *k <= fwd {
            k0_idx = i;
        } else {
            break;
        }
    }
    let k0 = strikes[k0_idx];

    let mut sum = 0.0;
    for i in 0..n {
        let k = strikes[i];
        let delta_k = if i == 0 {
            strikes[1] - strikes[0]
        } else if i == n - 1 {
            strikes[n - 1] - strikes[n - 2]
        } else {
            0.5 * (strikes[i + 1] - strikes[i - 1])
        };

        let (call, put) = q_otm[i];
        let q = if i < k0_idx {
            put
        } else if i > k0_idx {
            call
        } else {
            0.5 * (call + put)
        };

        sum += (delta_k / (k * k)) * q;
    }

    let term1 = 2.0 * (rate * expiry).exp() * sum / expiry;
    let term2 = ((fwd / k0) - 1.0).powi(2) / expiry;
    Ok((term1 - term2).max(0.0))
}

fn interpolate_piecewise(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    if xs.len() == 1 || ys.len() == 1 {
        return ys[0];
    }

    let x = if x.is_finite() { x } else { xs[0] };
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }

    for i in 0..xs.len() - 1 {
        if x >= xs[i] && x <= xs[i + 1] {
            let w = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + (ys[i + 1] - ys[i]) * w;
        }
    }
    ys[ys.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[derive(Debug, Clone)]
    struct FlatSurface {
        vol: f64,
        expiries: Vec<f64>,
        forward: f64,
    }

    impl ForwardVarianceSource for FlatSurface {
        fn implied_vol(&self, _strike: f64, _expiry: f64) -> f64 {
            self.vol
        }

        fn forward_price(&self, _expiry: f64) -> f64 {
            self.forward
        }

        fn expiries(&self) -> &[f64] {
            &self.expiries
        }
    }

    #[derive(Debug, Clone)]
    struct CalendarArbSurface {
        expiries: Vec<f64>,
        forward: f64,
    }

    impl ForwardVarianceSource for CalendarArbSurface {
        fn implied_vol(&self, _strike: f64, expiry: f64) -> f64 {
            if expiry <= 0.3 { 0.35 } else { 0.09 }
        }

        fn forward_price(&self, _expiry: f64) -> f64 {
            self.forward
        }

        fn expiries(&self) -> &[f64] {
            &self.expiries
        }
    }

    #[test]
    fn forward_variance_is_consistent_with_total_variance_differences() {
        let surface = FlatSurface {
            vol: 0.22,
            expiries: vec![0.25, 0.5, 1.0, 2.0],
            forward: 100.0,
        };
        let curve = ForwardVarianceCurve::from_surface_expiries(&surface).unwrap();

        for w in curve.points().windows(2) {
            let p0 = w[0];
            let p1 = w[1];
            let lhs = p1.total_variance - p0.total_variance;
            let rhs = p1.forward_variance * (p1.expiry - p0.expiry);
            assert!(p1.forward_variance >= 0.0);
            assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
        }

        let fwd = curve.forward_variance(0.5, 1.5).unwrap();
        assert_relative_eq!(fwd, 0.22_f64.powi(2), epsilon = 1e-12);
    }

    #[test]
    fn forward_variance_curve_rejects_calendar_arbitrage() {
        let surface = CalendarArbSurface {
            expiries: vec![0.25, 1.0],
            forward: 100.0,
        };
        let err = ForwardVarianceCurve::from_surface_expiries(&surface).unwrap_err();
        assert!(err.contains("calendar arbitrage"));
    }

    #[test]
    fn vix_matches_flat_vol_within_tenth_of_point() {
        let sigma = 0.20;
        let surface = FlatSurface {
            vol: sigma,
            expiries: vec![20.0 / 365.0, 45.0 / 365.0, 90.0 / 365.0],
            forward: 100.0,
        };

        let settings = VixSettings {
            strike_count: 3001,
            log_moneyness_span: 3.0,
            ..VixSettings::default()
        };
        let out = vix_style_index_from_surface(&surface, 0.01, settings).unwrap();
        let expected = sigma * 100.0;
        assert!(
            (out.index - expected).abs() < 0.1,
            "vix={} expected={expected}",
            out.index
        );
    }

    #[test]
    fn atm_skew_is_near_zero_for_flat_surface() {
        let surface = FlatSurface {
            vol: 0.18,
            expiries: vec![0.25, 0.5, 1.0],
            forward: 100.0,
        };
        let skew = AtmSkewTermStructure::from_surface_expiries(&surface).unwrap();
        assert!(skew.points().iter().all(|p| p.skew.abs() < 1.0e-12));
    }

    #[test]
    fn sabr_term_structure_interpolates_alpha_and_nu() {
        let points = vec![
            SabrVolOfVolPoint {
                expiry: 0.5,
                alpha: 0.20,
                nu: 0.60,
            },
            SabrVolOfVolPoint {
                expiry: 2.0,
                alpha: 0.16,
                nu: 0.40,
            },
        ];
        let ts = SabrVolOfVolTermStructure::new(points).unwrap();
        assert_relative_eq!(ts.alpha(1.25), 0.18, epsilon = 1e-12);
        assert_relative_eq!(ts.nu(1.25), 0.5, epsilon = 1e-12);
    }
}
