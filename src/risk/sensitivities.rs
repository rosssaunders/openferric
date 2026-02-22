//! Finite-difference risk sensitivities, reporting transforms, and lightweight risk aggregation.
//!
//! This module centers on bump-and-reprice utilities across curves, vol surfaces, and spot:
//! - IR: `parallel_dv01`, `bucket_dv01`, `key_rate_duration`, `gamma_ladder`, `cross_gamma`
//!   with configurable bump domain (`zero`, `par`, `log-discount`) and differencing stencil.
//! - Vol: expiry and strike-expiry vegas on [`QuoteVolSurface`].
//! - Spot: `fx_delta` and `commodity_delta`.
//!
//! It also includes higher-level plumbing used in risk production workflows:
//! - chain-rule Jacobian propagation through bootstraps (`jacobian_via_bootstrap`),
//! - SIMM/FRTB-style class mapping, CRIF-like CSV serialization, and concentration-aware
//!   charge aggregation (`compute_risk_charges`),
//! - scenario P&L explain and per-trade attribution.
//!
//! Numerical notes: finite-difference step size and stencil choice control truncation vs.
//! cancellation error; this module applies positive bump floors to avoid divide-by-zero on
//! near-zero states/quotes. Central differencing is generally more accurate but doubles
//! revaluation cost.
//!
//! References:
//! - Glasserman, *Monte Carlo Methods in Financial Engineering* (2004), bump-and-reprice.
//! - ISDA SIMM methodology and CRIF data conventions (for naming/export semantics).

use std::collections::BTreeMap;

use crate::market::VolSurface;
use crate::rates::YieldCurve;

use super::portfolio::Portfolio;

/// Bump magnitude specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BumpSize {
    /// Additive bump in absolute units.
    Absolute(f64),
    /// Relative bump as a fraction of the current value.
    Relative(f64),
}

impl BumpSize {
    #[inline]
    fn step(self, base: f64) -> f64 {
        match self {
            Self::Absolute(b) => b.abs().max(1.0e-12),
            Self::Relative(r) => (base.abs() * r.abs()).max(1.0e-12),
        }
    }
}

/// Finite-difference stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferencingScheme {
    /// Uses `f(x+h) - f(x)`.
    Forward,
    /// Uses `f(x+h) - f(x-h)`.
    Central,
}

/// Curve state to perturb for bump-and-reprice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveBumpMode {
    /// Perturb continuously-compounded zero rates.
    ZeroRate,
    /// Perturb par rates inferred on the input pillars.
    ParRate,
    /// Perturb log discount factors (`ln DF`).
    LogDiscount,
}

/// Curve bump configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveBumpConfig {
    pub bump_size: BumpSize,
    pub differencing: DifferencingScheme,
    pub mode: CurveBumpMode,
}

impl Default for CurveBumpConfig {
    fn default() -> Self {
        Self {
            bump_size: BumpSize::Absolute(1.0e-4),
            differencing: DifferencingScheme::Central,
            mode: CurveBumpMode::ZeroRate,
        }
    }
}

/// Surface bump target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceBumpMode {
    /// Perturb all surface quotes.
    Flat,
    /// Perturb all strikes for one expiry bucket.
    PerExpiry { expiry_index: usize },
    /// Perturb a single strike-expiry cell.
    PerStrikeExpiry {
        expiry_index: usize,
        strike_index: usize,
    },
}

/// Surface bump configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceBumpConfig {
    pub bump_size: BumpSize,
    pub differencing: DifferencingScheme,
}

impl Default for SurfaceBumpConfig {
    fn default() -> Self {
        Self {
            bump_size: BumpSize::Absolute(0.01),
            differencing: DifferencingScheme::Central,
        }
    }
}

/// Spot bump configuration for FX/commodity deltas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotBumpConfig {
    pub bump_size: BumpSize,
    pub differencing: DifferencingScheme,
}

impl Default for SpotBumpConfig {
    fn default() -> Self {
        Self {
            bump_size: BumpSize::Relative(0.01),
            differencing: DifferencingScheme::Central,
        }
    }
}

/// Generic curve bucket sensitivity record.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketSensitivity {
    pub pillar: f64,
    pub bump: f64,
    pub value: f64,
}

/// Key-rate duration point.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyRateDurationPoint {
    pub pillar: f64,
    pub bump: f64,
    pub duration: f64,
}

/// Gamma ladder point.
#[derive(Debug, Clone, PartialEq)]
pub struct GammaLadderPoint {
    pub pillar: f64,
    pub bump: f64,
    pub gamma: f64,
}

/// Expiry-bucket vega point.
#[derive(Debug, Clone, PartialEq)]
pub struct VegaExpiryPoint {
    pub expiry: f64,
    pub bump: f64,
    pub vega: f64,
}

/// Strike-expiry-bucket vega point.
#[derive(Debug, Clone, PartialEq)]
pub struct VegaStrikeExpiryPoint {
    pub expiry: f64,
    pub strike: f64,
    pub bump: f64,
    pub vega: f64,
}

/// Quote-vol surface with bucket access and bilinear interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteVolSurface {
    expiries: Vec<f64>,
    strikes: Vec<f64>,
    quotes: Vec<Vec<f64>>, // [expiry][strike]
}

impl QuoteVolSurface {
    /// Build from sorted strikes/expiries and a rectangular quote matrix.
    pub fn new(
        expiries: Vec<f64>,
        strikes: Vec<f64>,
        quotes: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        if expiries.is_empty() || strikes.is_empty() {
            return Err("expiries and strikes must be non-empty".to_string());
        }
        if quotes.len() != expiries.len() {
            return Err("quotes rows must match expiry count".to_string());
        }
        if quotes.iter().any(|row| row.len() != strikes.len()) {
            return Err("each quote row must match strike count".to_string());
        }
        if expiries.windows(2).any(|w| w[1] <= w[0]) {
            return Err("expiries must be strictly increasing".to_string());
        }
        if strikes.windows(2).any(|w| w[1] <= w[0]) {
            return Err("strikes must be strictly increasing".to_string());
        }
        if quotes.iter().flatten().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("all vol quotes must be finite and > 0".to_string());
        }

        Ok(Self {
            expiries,
            strikes,
            quotes,
        })
    }

    #[inline]
    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    #[inline]
    pub fn strikes(&self) -> &[f64] {
        &self.strikes
    }

    #[inline]
    pub fn quote(&self, expiry_index: usize, strike_index: usize) -> f64 {
        self.quotes[expiry_index][strike_index]
    }

    #[inline]
    pub fn set_quote(&mut self, expiry_index: usize, strike_index: usize, quote: f64) {
        self.quotes[expiry_index][strike_index] = quote.max(1.0e-8);
    }

    fn bumped(&self, mode: SurfaceBumpMode, bump_size: BumpSize, sign: f64) -> Self {
        let mut out = self.clone();

        match mode {
            SurfaceBumpMode::Flat => {
                for i in 0..out.expiries.len() {
                    for j in 0..out.strikes.len() {
                        let base = out.quotes[i][j];
                        let bump = bump_size.step(base);
                        out.quotes[i][j] = (base + sign * bump).max(1.0e-8);
                    }
                }
            }
            SurfaceBumpMode::PerExpiry { expiry_index } => {
                if expiry_index < out.expiries.len() {
                    for j in 0..out.strikes.len() {
                        let base = out.quotes[expiry_index][j];
                        let bump = bump_size.step(base);
                        out.quotes[expiry_index][j] = (base + sign * bump).max(1.0e-8);
                    }
                }
            }
            SurfaceBumpMode::PerStrikeExpiry {
                expiry_index,
                strike_index,
            } => {
                if expiry_index < out.expiries.len() && strike_index < out.strikes.len() {
                    let base = out.quotes[expiry_index][strike_index];
                    let bump = bump_size.step(base);
                    out.quotes[expiry_index][strike_index] = (base + sign * bump).max(1.0e-8);
                }
            }
        }

        out
    }
}

impl VolSurface for QuoteVolSurface {
    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        let (ei0, ei1, ew) = locate_bounds(&self.expiries, expiry);
        let (si0, si1, sw) = locate_bounds(&self.strikes, strike);

        if ei0 == ei1 && si0 == si1 {
            return self.quotes[ei0][si0];
        }
        if ei0 == ei1 {
            let v0 = self.quotes[ei0][si0];
            let v1 = self.quotes[ei0][si1];
            return v0 + (v1 - v0) * sw;
        }
        if si0 == si1 {
            let v0 = self.quotes[ei0][si0];
            let v1 = self.quotes[ei1][si0];
            return v0 + (v1 - v0) * ew;
        }

        let v00 = self.quotes[ei0][si0];
        let v01 = self.quotes[ei0][si1];
        let v10 = self.quotes[ei1][si0];
        let v11 = self.quotes[ei1][si1];

        let v0 = v00 + (v01 - v00) * sw;
        let v1 = v10 + (v11 - v10) * sw;
        v0 + (v1 - v0) * ew
    }
}

fn locate_bounds(grid: &[f64], x: f64) -> (usize, usize, f64) {
    if x <= grid[0] {
        return (0, 0, 0.0);
    }
    let last = grid.len() - 1;
    if x >= grid[last] {
        return (last, last, 0.0);
    }

    let mut lo = 0usize;
    for i in 0..last {
        if x >= grid[i] && x <= grid[i + 1] {
            lo = i;
            break;
        }
    }
    let hi = lo + 1;
    let w = (x - grid[lo]) / (grid[hi] - grid[lo]);
    (lo, hi, w)
}

fn first_order_pnl(base: f64, up: f64, down: Option<f64>, scheme: DifferencingScheme) -> f64 {
    match scheme {
        DifferencingScheme::Forward => up - base,
        DifferencingScheme::Central => {
            let dn = down.expect("central differencing requires down scenario");
            0.5 * (up - dn)
        }
    }
}

fn derivative(base: f64, up: f64, down: Option<f64>, h: f64, scheme: DifferencingScheme) -> f64 {
    match scheme {
        DifferencingScheme::Forward => (up - base) / h,
        DifferencingScheme::Central => {
            let dn = down.expect("central differencing requires down scenario");
            (up - dn) / (2.0 * h)
        }
    }
}

fn second_derivative<F>(
    mut eval: F,
    base_state: &[f64],
    idx: usize,
    h: f64,
    scheme: DifferencingScheme,
) -> f64
where
    F: FnMut(&[f64]) -> f64,
{
    match scheme {
        DifferencingScheme::Central => {
            let mut up = base_state.to_vec();
            up[idx] += h;
            let mut dn = base_state.to_vec();
            dn[idx] -= h;

            let p0 = eval(base_state);
            let p_up = eval(&up);
            let p_dn = eval(&dn);
            (p_up - 2.0 * p0 + p_dn) / (h * h)
        }
        DifferencingScheme::Forward => {
            let mut up = base_state.to_vec();
            up[idx] += h;
            let mut up2 = base_state.to_vec();
            up2[idx] += 2.0 * h;

            let p0 = eval(base_state);
            let p_up = eval(&up);
            let p_up2 = eval(&up2);
            (p_up2 - 2.0 * p_up + p0) / (h * h)
        }
    }
}

fn curve_state(curve: &YieldCurve, mode: CurveBumpMode) -> Vec<f64> {
    match mode {
        CurveBumpMode::ZeroRate => curve
            .tenors
            .iter()
            .map(|(t, df)| if *t > 0.0 { -df.ln() / t } else { 0.0 })
            .collect(),
        CurveBumpMode::ParRate => curve_to_par_rates(curve),
        CurveBumpMode::LogDiscount => curve.tenors.iter().map(|(_, df)| df.ln()).collect(),
    }
}

fn curve_from_state(template: &YieldCurve, mode: CurveBumpMode, state: &[f64]) -> YieldCurve {
    assert_eq!(template.tenors.len(), state.len());

    match mode {
        CurveBumpMode::ZeroRate => {
            let points: Vec<(f64, f64)> = template
                .tenors
                .iter()
                .zip(state.iter())
                .map(|((t, _), z)| (*t, (-(z.max(-5.0)) * *t).exp().max(1.0e-12)))
                .collect();
            YieldCurve::new(points)
        }
        CurveBumpMode::ParRate => par_rates_to_curve(template, state),
        CurveBumpMode::LogDiscount => {
            let points: Vec<(f64, f64)> = template
                .tenors
                .iter()
                .zip(state.iter())
                .map(|((t, _), ldf)| (*t, ldf.exp().max(1.0e-12)))
                .collect();
            YieldCurve::new(points)
        }
    }
}

fn curve_to_par_rates(curve: &YieldCurve) -> Vec<f64> {
    let mut out = Vec::with_capacity(curve.tenors.len());
    let mut annuity = 0.0;
    let mut prev_t = 0.0;

    for (t, _) in &curve.tenors {
        let dt = (t - prev_t).max(1.0e-8);
        let df = curve.discount_factor(*t);
        annuity += dt * df;
        let par = if annuity > 0.0 {
            (1.0 - df) / annuity
        } else {
            0.0
        };
        out.push(par);
        prev_t = *t;
    }

    out
}

fn par_rates_to_curve(template: &YieldCurve, par_rates: &[f64]) -> YieldCurve {
    let mut points = Vec::with_capacity(template.tenors.len());
    let mut annuity = 0.0;
    let mut prev_t = 0.0;

    for ((t, _), rate) in template.tenors.iter().zip(par_rates.iter()) {
        let dt = (t - prev_t).max(1.0e-8);
        let num = (1.0 - rate * annuity).max(1.0e-12);
        let den = (1.0 + rate * dt).max(1.0e-12);
        let df = (num / den).max(1.0e-12);

        points.push((*t, df));
        annuity += dt * df;
        prev_t = *t;
    }

    YieldCurve::new(points)
}

/// Parallel DV01 (flat curve bump) under the selected curve bump mode.
///
/// Returns PV change for the configured bump magnitude.
pub fn parallel_dv01<P>(curve: &YieldCurve, config: CurveBumpConfig, pricer: P) -> f64
where
    P: Fn(&YieldCurve) -> f64,
{
    let state = curve_state(curve, config.mode);
    let base = pricer(curve);

    let mut up = state.clone();
    for i in 0..up.len() {
        up[i] += config.bump_size.step(state[i]);
    }
    let up_curve = curve_from_state(curve, config.mode, &up);
    let p_up = pricer(&up_curve);

    let p_dn = if config.differencing == DifferencingScheme::Central {
        let mut dn = state.clone();
        for i in 0..dn.len() {
            dn[i] -= config.bump_size.step(state[i]);
        }
        let dn_curve = curve_from_state(curve, config.mode, &dn);
        Some(pricer(&dn_curve))
    } else {
        None
    };

    first_order_pnl(base, p_up, p_dn, config.differencing)
}

/// Bucket DV01 by curve pillar.
pub fn bucket_dv01<P>(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: P,
) -> Vec<BucketSensitivity>
where
    P: Fn(&YieldCurve) -> f64,
{
    let state = curve_state(curve, config.mode);
    let base = pricer(curve);

    curve
        .tenors
        .iter()
        .enumerate()
        .map(|(idx, (pillar, _))| {
            let h = config.bump_size.step(state[idx]);

            let mut up = state.clone();
            up[idx] += h;
            let up_curve = curve_from_state(curve, config.mode, &up);
            let p_up = pricer(&up_curve);

            let p_dn = if config.differencing == DifferencingScheme::Central {
                let mut dn = state.clone();
                dn[idx] -= h;
                let dn_curve = curve_from_state(curve, config.mode, &dn);
                Some(pricer(&dn_curve))
            } else {
                None
            };

            BucketSensitivity {
                pillar: *pillar,
                bump: h,
                value: first_order_pnl(base, p_up, p_dn, config.differencing),
            }
        })
        .collect()
}

/// Key-rate duration by curve pillar.
///
/// `KRD_i = -(1 / PV) * dPV/dr_i`.
pub fn key_rate_duration<P>(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: P,
) -> Vec<KeyRateDurationPoint>
where
    P: Fn(&YieldCurve) -> f64,
{
    let state = curve_state(curve, config.mode);
    let base = pricer(curve);

    if base.abs() <= 1.0e-14 {
        return curve
            .tenors
            .iter()
            .map(|(pillar, _)| KeyRateDurationPoint {
                pillar: *pillar,
                bump: 0.0,
                duration: 0.0,
            })
            .collect();
    }

    curve
        .tenors
        .iter()
        .enumerate()
        .map(|(idx, (pillar, _))| {
            let h = config.bump_size.step(state[idx]);

            let mut up = state.clone();
            up[idx] += h;
            let up_curve = curve_from_state(curve, config.mode, &up);
            let p_up = pricer(&up_curve);

            let p_dn = if config.differencing == DifferencingScheme::Central {
                let mut dn = state.clone();
                dn[idx] -= h;
                let dn_curve = curve_from_state(curve, config.mode, &dn);
                Some(pricer(&dn_curve))
            } else {
                None
            };

            let dp_dr = derivative(base, p_up, p_dn, h, config.differencing);
            KeyRateDurationPoint {
                pillar: *pillar,
                bump: h,
                duration: -dp_dr / base,
            }
        })
        .collect()
}

/// Gamma ladder (second derivative) by curve pillar.
pub fn gamma_ladder<P>(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: P,
) -> Vec<GammaLadderPoint>
where
    P: Fn(&YieldCurve) -> f64,
{
    let state = curve_state(curve, config.mode);

    curve
        .tenors
        .iter()
        .enumerate()
        .map(|(idx, (pillar, _))| {
            let h = config.bump_size.step(state[idx]);
            let gamma = second_derivative(
                |x| {
                    let c = curve_from_state(curve, config.mode, x);
                    pricer(&c)
                },
                &state,
                idx,
                h,
                config.differencing,
            );

            GammaLadderPoint {
                pillar: *pillar,
                bump: h,
                gamma,
            }
        })
        .collect()
}

/// Cross-gamma between two curve pillars.
pub fn cross_gamma<P>(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pillar_i: usize,
    pillar_j: usize,
    pricer: P,
) -> f64
where
    P: Fn(&YieldCurve) -> f64,
{
    assert!(pillar_i < curve.tenors.len());
    assert!(pillar_j < curve.tenors.len());

    let state = curve_state(curve, config.mode);
    let h_i = config.bump_size.step(state[pillar_i]);
    let h_j = config.bump_size.step(state[pillar_j]);

    match config.differencing {
        DifferencingScheme::Central => {
            let mut pp = state.clone();
            pp[pillar_i] += h_i;
            pp[pillar_j] += h_j;

            let mut pm = state.clone();
            pm[pillar_i] += h_i;
            pm[pillar_j] -= h_j;

            let mut mp = state.clone();
            mp[pillar_i] -= h_i;
            mp[pillar_j] += h_j;

            let mut mm = state.clone();
            mm[pillar_i] -= h_i;
            mm[pillar_j] -= h_j;

            let v_pp = pricer(&curve_from_state(curve, config.mode, &pp));
            let v_pm = pricer(&curve_from_state(curve, config.mode, &pm));
            let v_mp = pricer(&curve_from_state(curve, config.mode, &mp));
            let v_mm = pricer(&curve_from_state(curve, config.mode, &mm));

            (v_pp - v_pm - v_mp + v_mm) / (4.0 * h_i * h_j)
        }
        DifferencingScheme::Forward => {
            let base = pricer(curve);

            let mut pi = state.clone();
            pi[pillar_i] += h_i;

            let mut pj = state.clone();
            pj[pillar_j] += h_j;

            let mut pij = state.clone();
            pij[pillar_i] += h_i;
            pij[pillar_j] += h_j;

            let v_i = pricer(&curve_from_state(curve, config.mode, &pi));
            let v_j = pricer(&curve_from_state(curve, config.mode, &pj));
            let v_ij = pricer(&curve_from_state(curve, config.mode, &pij));

            (v_ij - v_i - v_j + base) / (h_i * h_j)
        }
    }
}

/// Vega by expiry bucket.
pub fn vega_by_expiry_bucket<P>(
    surface: &QuoteVolSurface,
    config: SurfaceBumpConfig,
    pricer: P,
) -> Vec<VegaExpiryPoint>
where
    P: Fn(&QuoteVolSurface) -> f64,
{
    let base = pricer(surface);

    (0..surface.expiries().len())
        .map(|i| {
            let avg = (0..surface.strikes().len())
                .map(|j| surface.quote(i, j))
                .sum::<f64>()
                / surface.strikes().len() as f64;
            let h = config.bump_size.step(avg);

            let up = surface.bumped(
                SurfaceBumpMode::PerExpiry { expiry_index: i },
                config.bump_size,
                1.0,
            );
            let p_up = pricer(&up);

            let p_dn = if config.differencing == DifferencingScheme::Central {
                let dn = surface.bumped(
                    SurfaceBumpMode::PerExpiry { expiry_index: i },
                    config.bump_size,
                    -1.0,
                );
                Some(pricer(&dn))
            } else {
                None
            };

            VegaExpiryPoint {
                expiry: surface.expiries()[i],
                bump: h,
                vega: first_order_pnl(base, p_up, p_dn, config.differencing),
            }
        })
        .collect()
}

/// Vega by strike-expiry bucket.
pub fn vega_by_strike_expiry_bucket<P>(
    surface: &QuoteVolSurface,
    config: SurfaceBumpConfig,
    pricer: P,
) -> Vec<VegaStrikeExpiryPoint>
where
    P: Fn(&QuoteVolSurface) -> f64,
{
    let base = pricer(surface);
    let mut out = Vec::with_capacity(surface.expiries().len() * surface.strikes().len());

    for i in 0..surface.expiries().len() {
        for j in 0..surface.strikes().len() {
            let base_quote = surface.quote(i, j);
            let h = config.bump_size.step(base_quote);

            let mode = SurfaceBumpMode::PerStrikeExpiry {
                expiry_index: i,
                strike_index: j,
            };

            let up = surface.bumped(mode, config.bump_size, 1.0);
            let p_up = pricer(&up);

            let p_dn = if config.differencing == DifferencingScheme::Central {
                let dn = surface.bumped(mode, config.bump_size, -1.0);
                Some(pricer(&dn))
            } else {
                None
            };

            out.push(VegaStrikeExpiryPoint {
                expiry: surface.expiries()[i],
                strike: surface.strikes()[j],
                bump: h,
                vega: first_order_pnl(base, p_up, p_dn, config.differencing),
            });
        }
    }

    out
}

/// FX delta via bump-and-reprice.
pub fn fx_delta<P>(spot: f64, config: SpotBumpConfig, pricer: P) -> f64
where
    P: Fn(f64) -> f64,
{
    let h = config.bump_size.step(spot);
    let base = pricer(spot);
    let up = pricer((spot + h).max(1.0e-12));
    let dn = if config.differencing == DifferencingScheme::Central {
        Some(pricer((spot - h).max(1.0e-12)))
    } else {
        None
    };
    derivative(base, up, dn, h, config.differencing)
}

/// Commodity delta via bump-and-reprice.
#[inline]
pub fn commodity_delta<P>(spot: f64, config: SpotBumpConfig, pricer: P) -> f64
where
    P: Fn(f64) -> f64,
{
    fx_delta(spot, config, pricer)
}

/// Chain-rule Jacobian output.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainRuleJacobian {
    /// First derivative of PV wrt calibrated state factors.
    pub d_pv_d_state: Vec<f64>,
    /// Jacobian of calibrated state wrt market quotes (`state_dim x quote_dim`).
    pub d_state_d_quote: Vec<Vec<f64>>,
    /// Chain-rule result `dPV/dQuote`.
    pub d_pv_d_quote: Vec<f64>,
}

/// Compute `dPV/dQuote` via chain rule through a bootstrap/calibration map.
///
/// The function evaluates:
/// `dPV/dQuote = (dPV/dState) * (dState/dQuote)`.
pub fn jacobian_via_bootstrap<B, P>(
    market_quotes: &[f64],
    bump_size: BumpSize,
    differencing: DifferencingScheme,
    bootstrap: B,
    pv_from_state: P,
) -> ChainRuleJacobian
where
    B: Fn(&[f64]) -> Vec<f64>,
    P: Fn(&[f64]) -> f64,
{
    assert!(!market_quotes.is_empty(), "market_quotes must not be empty");

    let state0 = bootstrap(market_quotes);
    assert!(!state0.is_empty(), "bootstrap state must not be empty");

    let state_dim = state0.len();
    let quote_dim = market_quotes.len();

    let pv0 = pv_from_state(&state0);

    let mut d_pv_d_state = vec![0.0; state_dim];
    for i in 0..state_dim {
        let h = bump_size.step(state0[i]);
        let mut up = state0.clone();
        up[i] += h;
        let pv_up = pv_from_state(&up);

        let pv_dn = if differencing == DifferencingScheme::Central {
            let mut dn = state0.clone();
            dn[i] -= h;
            Some(pv_from_state(&dn))
        } else {
            None
        };

        d_pv_d_state[i] = derivative(pv0, pv_up, pv_dn, h, differencing);
    }

    let mut d_state_d_quote = vec![vec![0.0; quote_dim]; state_dim];
    for j in 0..quote_dim {
        let h = bump_size.step(market_quotes[j]);

        let mut q_up = market_quotes.to_vec();
        q_up[j] += h;
        let s_up = bootstrap(&q_up);
        assert_eq!(
            s_up.len(),
            state_dim,
            "bootstrap state dimension must remain stable under bumps"
        );

        let s_dn = if differencing == DifferencingScheme::Central {
            let mut q_dn = market_quotes.to_vec();
            q_dn[j] -= h;
            let s_dn = bootstrap(&q_dn);
            assert_eq!(
                s_dn.len(),
                state_dim,
                "bootstrap state dimension must remain stable under bumps"
            );
            Some(s_dn)
        } else {
            None
        };

        for i in 0..state_dim {
            let val = match differencing {
                DifferencingScheme::Forward => (s_up[i] - state0[i]) / h,
                DifferencingScheme::Central => {
                    let sd = s_dn
                        .as_ref()
                        .expect("central differencing requires down state");
                    (s_up[i] - sd[i]) / (2.0 * h)
                }
            };
            d_state_d_quote[i][j] = val;
        }
    }

    let mut d_pv_d_quote = vec![0.0; quote_dim];
    for (j, out) in d_pv_d_quote.iter_mut().enumerate() {
        let mut acc = 0.0;
        for i in 0..state_dim {
            acc += d_pv_d_state[i] * d_state_d_quote[i][j];
        }
        *out = acc;
    }

    ChainRuleJacobian {
        d_pv_d_state,
        d_state_d_quote,
        d_pv_d_quote,
    }
}

/// Regulatory risk class for FRTB/SIMM style sensitivities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegulatoryRiskClass {
    IR,
    FX,
    EQ,
    COMM,
    Credit,
}

impl RegulatoryRiskClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IR => "IR",
            Self::FX => "FX",
            Self::EQ => "EQ",
            Self::COMM => "COMM",
            Self::Credit => "Credit",
        }
    }
}

/// Heuristic risk-class mapping from label/type text.
pub fn map_risk_class(label: &str) -> RegulatoryRiskClass {
    let key = label.to_ascii_lowercase();
    if key.contains("fx") || key.contains("ccy") || key.contains("usd/") || key.contains("eur/") {
        RegulatoryRiskClass::FX
    } else if key.contains("eq")
        || key.contains("equity")
        || key.contains("stock")
        || key.contains("index")
    {
        RegulatoryRiskClass::EQ
    } else if key.contains("comm")
        || key.contains("commodity")
        || key.contains("oil")
        || key.contains("gas")
        || key.contains("power")
        || key.contains("metal")
    {
        RegulatoryRiskClass::COMM
    } else if key.contains("credit") || key.contains("cds") || key.contains("spread") {
        RegulatoryRiskClass::Credit
    } else {
        RegulatoryRiskClass::IR
    }
}

/// Sensitivity measure type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensitivityMeasure {
    Delta,
    Vega,
    Curvature,
}

impl SensitivityMeasure {
    fn as_crif_suffix(self) -> &'static str {
        match self {
            Self::Delta => "Delta",
            Self::Vega => "Vega",
            Self::Curvature => "Curvature",
        }
    }
}

/// Internal sensitivity record used for CRIF export and risk aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityRecord {
    pub portfolio_id: String,
    pub trade_id: String,
    pub risk_class: RegulatoryRiskClass,
    pub measure: SensitivityMeasure,
    pub qualifier: String,
    pub bucket: String,
    pub label1: String,
    pub label2: String,
    pub amount: f64,
    pub amount_currency: String,
}

impl SensitivityRecord {
    pub fn to_crif(&self) -> CrifRecord {
        CrifRecord {
            portfolio_id: self.portfolio_id.clone(),
            trade_id: self.trade_id.clone(),
            risk_type: format!(
                "Risk_{}{}",
                self.risk_class.as_str(),
                self.measure.as_crif_suffix()
            ),
            qualifier: self.qualifier.clone(),
            bucket: self.bucket.clone(),
            label1: self.label1.clone(),
            label2: self.label2.clone(),
            amount: self.amount,
            amount_currency: self.amount_currency.clone(),
        }
    }
}

/// CRIF line representation (subset of commonly consumed columns).
#[derive(Debug, Clone, PartialEq)]
pub struct CrifRecord {
    pub portfolio_id: String,
    pub trade_id: String,
    pub risk_type: String,
    pub qualifier: String,
    pub bucket: String,
    pub label1: String,
    pub label2: String,
    pub amount: f64,
    pub amount_currency: String,
}

impl CrifRecord {
    fn csv_header() -> &'static str {
        "PortfolioId,TradeId,RiskType,Qualifier,Bucket,Label1,Label2,Amount,AmountCurrency"
    }

    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.10},{}",
            csv_escape(&self.portfolio_id),
            csv_escape(&self.trade_id),
            csv_escape(&self.risk_type),
            csv_escape(&self.qualifier),
            csv_escape(&self.bucket),
            csv_escape(&self.label1),
            csv_escape(&self.label2),
            self.amount,
            csv_escape(&self.amount_currency)
        )
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Serialize records in CRIF-like CSV format.
pub fn to_crif_csv(records: &[SensitivityRecord]) -> String {
    let mut lines = Vec::with_capacity(records.len() + 1);
    lines.push(CrifRecord::csv_header().to_string());
    for r in records {
        lines.push(r.to_crif().to_csv_row());
    }
    lines.join("\n")
}

/// Risk-aggregation parameters for one risk class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskClassChargeConfig {
    pub risk_class: RegulatoryRiskClass,
    pub delta_weight: f64,
    pub vega_weight: f64,
    pub curvature_weight: f64,
    pub intra_bucket_corr: f64,
    pub inter_bucket_corr: f64,
    pub concentration_threshold: f64,
}

/// SIMM/FRTB-style risk charge configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskChargeConfig {
    pub class_configs: Vec<RiskClassChargeConfig>,
}

impl RiskChargeConfig {
    /// Reasonable default magnitudes for demonstration/testing.
    pub fn baseline() -> Self {
        Self {
            class_configs: vec![
                RiskClassChargeConfig {
                    risk_class: RegulatoryRiskClass::IR,
                    delta_weight: 0.017,
                    vega_weight: 0.21,
                    curvature_weight: 0.50,
                    intra_bucket_corr: 0.98,
                    inter_bucket_corr: 0.35,
                    concentration_threshold: 230_000_000.0,
                },
                RiskClassChargeConfig {
                    risk_class: RegulatoryRiskClass::FX,
                    delta_weight: 0.077,
                    vega_weight: 0.32,
                    curvature_weight: 0.50,
                    intra_bucket_corr: 0.60,
                    inter_bucket_corr: 0.35,
                    concentration_threshold: 8_400_000_000.0,
                },
                RiskClassChargeConfig {
                    risk_class: RegulatoryRiskClass::EQ,
                    delta_weight: 0.24,
                    vega_weight: 0.30,
                    curvature_weight: 0.50,
                    intra_bucket_corr: 0.15,
                    inter_bucket_corr: 0.15,
                    concentration_threshold: 3_300_000_000.0,
                },
                RiskClassChargeConfig {
                    risk_class: RegulatoryRiskClass::COMM,
                    delta_weight: 0.18,
                    vega_weight: 0.38,
                    curvature_weight: 0.50,
                    intra_bucket_corr: 0.20,
                    inter_bucket_corr: 0.20,
                    concentration_threshold: 2_100_000_000.0,
                },
                RiskClassChargeConfig {
                    risk_class: RegulatoryRiskClass::Credit,
                    delta_weight: 0.10,
                    vega_weight: 0.52,
                    curvature_weight: 0.50,
                    intra_bucket_corr: 0.30,
                    inter_bucket_corr: 0.18,
                    concentration_threshold: 1_300_000_000.0,
                },
            ],
        }
    }

    pub fn for_class(&self, risk_class: RegulatoryRiskClass) -> Option<RiskClassChargeConfig> {
        self.class_configs
            .iter()
            .copied()
            .find(|c| c.risk_class == risk_class)
    }
}

/// Per-class risk charge decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassRiskCharge {
    pub risk_class: RegulatoryRiskClass,
    pub delta: f64,
    pub vega: f64,
    pub curvature: f64,
    pub total: f64,
}

/// Portfolio-level SIMM/FRTB sensitivity charges.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskChargeSummary {
    pub by_class: Vec<ClassRiskCharge>,
    pub delta_total: f64,
    pub vega_total: f64,
    pub curvature_total: f64,
    pub total: f64,
}

/// Compute delta/vega/curvature risk charges with concentration scaling.
pub fn compute_risk_charges(
    sensitivities: &[SensitivityRecord],
    config: &RiskChargeConfig,
) -> RiskChargeSummary {
    let mut by_class = Vec::new();

    for class_cfg in &config.class_configs {
        let delta = class_kind_charge(
            sensitivities,
            class_cfg,
            SensitivityMeasure::Delta,
            class_cfg.delta_weight,
        );
        let vega = class_kind_charge(
            sensitivities,
            class_cfg,
            SensitivityMeasure::Vega,
            class_cfg.vega_weight,
        );
        let curvature = class_kind_charge(
            sensitivities,
            class_cfg,
            SensitivityMeasure::Curvature,
            class_cfg.curvature_weight,
        );
        let total = (delta * delta + vega * vega + curvature * curvature).sqrt();

        by_class.push(ClassRiskCharge {
            risk_class: class_cfg.risk_class,
            delta,
            vega,
            curvature,
            total,
        });
    }

    let delta_total = (by_class.iter().map(|x| x.delta * x.delta).sum::<f64>()).sqrt();
    let vega_total = (by_class.iter().map(|x| x.vega * x.vega).sum::<f64>()).sqrt();
    let curvature_total = (by_class
        .iter()
        .map(|x| x.curvature * x.curvature)
        .sum::<f64>())
    .sqrt();
    let total = (by_class.iter().map(|x| x.total * x.total).sum::<f64>()).sqrt();

    RiskChargeSummary {
        by_class,
        delta_total,
        vega_total,
        curvature_total,
        total,
    }
}

fn class_kind_charge(
    sensitivities: &[SensitivityRecord],
    cfg: &RiskClassChargeConfig,
    measure: SensitivityMeasure,
    risk_weight: f64,
) -> f64 {
    let mut buckets: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for s in sensitivities {
        if s.risk_class == cfg.risk_class && s.measure == measure {
            buckets.entry(s.bucket.clone()).or_default().push(s.amount);
        }
    }

    if buckets.is_empty() {
        return 0.0;
    }

    let mut kbs = Vec::with_capacity(buckets.len());
    for values in buckets.values() {
        let gross = values.iter().map(|x| x.abs()).sum::<f64>();
        let threshold = cfg.concentration_threshold.max(1.0e-12);
        let concentration = (gross / threshold).sqrt().max(1.0);

        let ws: Vec<f64> = values
            .iter()
            .map(|x| x * risk_weight * concentration)
            .collect();

        let mut kb2 = 0.0;
        for i in 0..ws.len() {
            for j in 0..ws.len() {
                let corr = if i == j { 1.0 } else { cfg.intra_bucket_corr };
                kb2 += corr * ws[i] * ws[j];
            }
        }
        kbs.push(kb2.max(0.0).sqrt());
    }

    let mut charge2 = 0.0;
    for i in 0..kbs.len() {
        for j in 0..kbs.len() {
            let corr = if i == j { 1.0 } else { cfg.inter_bucket_corr };
            charge2 += corr * kbs[i] * kbs[j];
        }
    }

    charge2.max(0.0).sqrt()
}

/// Scenario shock for portfolio attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioShock {
    pub name: String,
    pub as_of: Option<String>,
    pub spot_shock_pct: f64,
    pub vol_shock_pct: f64,
    pub rate_shock_abs: f64,
    pub horizon_years: f64,
}

impl ScenarioShock {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            as_of: None,
            spot_shock_pct: 0.0,
            vol_shock_pct: 0.0,
            rate_shock_abs: 0.0,
            horizon_years: 0.0,
        }
    }
}

/// PnL explain decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct PnlExplain {
    pub observed_pnl: f64,
    pub theta: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub rho: f64,
    pub explained: f64,
    pub unexplained: f64,
    pub unexplained_ratio: f64,
}

/// Scenario PnL output row.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioPnlRow {
    pub scenario: String,
    pub as_of: Option<String>,
    pub pnl: f64,
}

/// Trade-level contribution row.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeRiskContribution {
    pub trade_index: usize,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub total: f64,
    pub share_of_total: f64,
}

fn trade_components<I>(
    position: &super::portfolio::Position<I>,
    scenario: &ScenarioShock,
) -> (f64, f64, f64, f64, f64, f64) {
    let ds = position.spot * scenario.spot_shock_pct;
    let dvol = position.implied_vol * scenario.vol_shock_pct;

    let delta = position.quantity * position.greeks.delta * ds;
    let gamma = 0.5 * position.quantity * position.greeks.gamma * ds * ds;
    let vega = position.quantity * position.greeks.vega * dvol;
    let theta = position.quantity * position.greeks.theta * scenario.horizon_years;
    let rho = position.quantity * position.greeks.rho * scenario.rate_shock_abs;

    (
        delta,
        gamma,
        vega,
        theta,
        rho,
        delta + gamma + vega + theta + rho,
    )
}

/// PnL explain (`theta + delta + gamma + vega + rho + unexplained`).
pub fn pnl_explain<I>(
    portfolio: &Portfolio<I>,
    observed_pnl: f64,
    scenario: &ScenarioShock,
) -> PnlExplain {
    let mut delta = 0.0;
    let mut gamma = 0.0;
    let mut vega = 0.0;
    let mut theta = 0.0;
    let mut rho = 0.0;

    for p in &portfolio.positions {
        let (d, g, v, t, r, _) = trade_components(p, scenario);
        delta += d;
        gamma += g;
        vega += v;
        theta += t;
        rho += r;
    }

    let explained = theta + delta + gamma + vega + rho;
    let unexplained = observed_pnl - explained;
    let unexplained_ratio = if observed_pnl.abs() > 1.0e-12 {
        unexplained.abs() / observed_pnl.abs()
    } else {
        0.0
    };

    PnlExplain {
        observed_pnl,
        theta,
        delta,
        gamma,
        vega,
        rho,
        explained,
        unexplained,
        unexplained_ratio,
    }
}

/// Scenario PnL report for historical/hypothetical scenario sets.
pub fn scenario_pnl_report<I>(
    portfolio: &Portfolio<I>,
    scenarios: &[ScenarioShock],
) -> Vec<ScenarioPnlRow> {
    scenarios
        .iter()
        .map(|scenario| {
            let pnl = portfolio
                .positions
                .iter()
                .map(|p| trade_components(p, scenario).5)
                .sum();
            ScenarioPnlRow {
                scenario: scenario.name.clone(),
                as_of: scenario.as_of.clone(),
                pnl,
            }
        })
        .collect()
}

/// Per-trade contribution for a selected scenario.
pub fn risk_contribution_per_trade<I>(
    portfolio: &Portfolio<I>,
    scenario: &ScenarioShock,
) -> Vec<TradeRiskContribution> {
    let mut rows = Vec::with_capacity(portfolio.positions.len());
    for (idx, p) in portfolio.positions.iter().enumerate() {
        let (delta, gamma, vega, theta, rho, total) = trade_components(p, scenario);
        rows.push(TradeRiskContribution {
            trade_index: idx,
            delta,
            gamma,
            vega,
            theta,
            rho,
            total,
            share_of_total: 0.0,
        });
    }

    let total_pnl: f64 = rows.iter().map(|r| r.total).sum();
    if total_pnl.abs() > 1.0e-12 {
        for row in &mut rows {
            row.share_of_total = row.total / total_pnl;
        }
    }

    rows
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::core::Greeks;
    use crate::risk::portfolio::{Portfolio, Position};

    use super::*;

    fn sample_curve() -> YieldCurve {
        YieldCurve::new(vec![
            (1.0, (-0.02_f64).exp()),
            (2.0, (-0.025_f64 * 2.0).exp()),
            (5.0, (-0.03_f64 * 5.0).exp()),
        ])
    }

    #[test]
    fn parallel_dv01_matches_sum_of_bucket_dv01_for_linear_model() {
        let curve = sample_curve();
        let cfg = CurveBumpConfig {
            bump_size: BumpSize::Absolute(1.0e-4),
            differencing: DifferencingScheme::Forward,
            mode: CurveBumpMode::ZeroRate,
        };

        let weights = [100.0, -50.0, 20.0];
        let pricer = |c: &YieldCurve| {
            c.tenors
                .iter()
                .enumerate()
                .map(|(i, (t, _))| weights[i] * c.zero_rate(*t))
                .sum::<f64>()
        };

        let parallel = parallel_dv01(&curve, cfg, pricer);
        let buckets = bucket_dv01(&curve, cfg, pricer);
        let bucket_sum: f64 = buckets.iter().map(|b| b.value).sum();

        assert_relative_eq!(parallel, bucket_sum, epsilon = 1.0e-12);
    }

    #[test]
    fn key_rate_duration_recovers_zero_coupon_duration() {
        let curve = YieldCurve::new(vec![(5.0, (-0.03_f64 * 5.0).exp())]);
        let cfg = CurveBumpConfig::default();
        let pricer = |c: &YieldCurve| c.discount_factor(5.0);

        let krd = key_rate_duration(&curve, cfg, pricer);
        assert_eq!(krd.len(), 1);
        assert_relative_eq!(krd[0].duration, 5.0, epsilon = 5.0e-3);
    }

    #[test]
    fn gamma_ladder_and_cross_gamma_match_quadratic_model() {
        let curve = YieldCurve::new(vec![
            (1.0, (-0.01_f64).exp()),
            (2.0, (-0.02_f64 * 2.0).exp()),
        ]);
        let cfg = CurveBumpConfig::default();

        let pricer = |c: &YieldCurve| {
            let x0 = c.zero_rate(1.0);
            let x1 = c.zero_rate(2.0);
            0.5 * (x0 * x0 + 2.0 * x0 * x1 + 3.0 * x1 * x1)
        };

        let gamma = gamma_ladder(&curve, cfg, pricer);
        assert_relative_eq!(gamma[0].gamma, 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(gamma[1].gamma, 3.0, epsilon = 1.0e-6);

        let cg = cross_gamma(&curve, cfg, 0, 1, pricer);
        assert_relative_eq!(cg, 1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn curve_bump_modes_produce_finite_values() {
        let curve = sample_curve();
        let pricer = |c: &YieldCurve| c.discount_factor(1.5) + c.discount_factor(4.0);

        for mode in [
            CurveBumpMode::ZeroRate,
            CurveBumpMode::ParRate,
            CurveBumpMode::LogDiscount,
        ] {
            let cfg = CurveBumpConfig {
                mode,
                ..CurveBumpConfig::default()
            };
            let dv01 = parallel_dv01(&curve, cfg, pricer);
            assert!(dv01.is_finite());

            let buckets = bucket_dv01(&curve, cfg, pricer);
            assert_eq!(buckets.len(), curve.tenors.len());
            assert!(buckets.iter().all(|b| b.value.is_finite()));
        }
    }

    #[test]
    fn vega_bucket_reports_match_linear_surface_pricer() {
        let surface = QuoteVolSurface::new(
            vec![0.5, 1.0],
            vec![90.0, 100.0],
            vec![vec![0.20, 0.21], vec![0.22, 0.23]],
        )
        .unwrap();

        let cfg = SurfaceBumpConfig {
            bump_size: BumpSize::Absolute(0.01),
            differencing: DifferencingScheme::Forward,
        };

        let weights = [[10.0, -5.0], [2.0, 1.0]];
        let pricer = |s: &QuoteVolSurface| {
            let mut v = 0.0;
            for (i, row) in weights.iter().enumerate() {
                for (j, w) in row.iter().enumerate() {
                    v += w * s.quote(i, j);
                }
            }
            v
        };

        let by_expiry = vega_by_expiry_bucket(&surface, cfg, pricer);
        assert_relative_eq!(by_expiry[0].vega, (10.0 - 5.0) * 0.01, epsilon = 1.0e-12);
        assert_relative_eq!(by_expiry[1].vega, (2.0 + 1.0) * 0.01, epsilon = 1.0e-12);

        let by_bucket = vega_by_strike_expiry_bucket(&surface, cfg, pricer);
        assert_relative_eq!(by_bucket[0].vega, 10.0 * 0.01, epsilon = 1.0e-12);
        assert_relative_eq!(by_bucket[1].vega, -5.0 * 0.01, epsilon = 1.0e-12);
        assert_relative_eq!(by_bucket[2].vega, 2.0 * 0.01, epsilon = 1.0e-12);
        assert_relative_eq!(by_bucket[3].vega, 1.0 * 0.01, epsilon = 1.0e-12);
    }

    #[test]
    fn fx_and_commodity_delta_match_analytic_derivatives() {
        let cfg = SpotBumpConfig::default();

        let fx = fx_delta(120.0, cfg, |s| 2.0 * s + 3.0);
        assert_relative_eq!(fx, 2.0, epsilon = 1.0e-10);

        let comm = commodity_delta(80.0, cfg, |s| -0.5 * s * s);
        assert_relative_eq!(comm, -80.0, epsilon = 1.0e-6);
    }

    #[test]
    fn jacobian_chain_rule_matches_closed_form() {
        let quotes = vec![1.0, 2.0];
        let result = jacobian_via_bootstrap(
            &quotes,
            BumpSize::Absolute(1.0e-5),
            DifferencingScheme::Central,
            |q: &[f64]| vec![q[0] * q[0] + q[1], q[0] - 2.0 * q[1]],
            |s: &[f64]| 3.0 * s[0] + 4.0 * s[1],
        );

        assert_relative_eq!(result.d_pv_d_state[0], 3.0, epsilon = 1.0e-8);
        assert_relative_eq!(result.d_pv_d_state[1], 4.0, epsilon = 1.0e-8);
        assert_relative_eq!(result.d_pv_d_quote[0], 10.0, epsilon = 1.0e-4);
        assert_relative_eq!(result.d_pv_d_quote[1], -5.0, epsilon = 1.0e-4);
    }

    #[test]
    fn risk_class_mapping_and_crif_export_work() {
        assert_eq!(map_risk_class("USD 5Y swap"), RegulatoryRiskClass::IR);
        assert_eq!(map_risk_class("EUR/USD spot"), RegulatoryRiskClass::FX);
        assert_eq!(map_risk_class("equity index"), RegulatoryRiskClass::EQ);
        assert_eq!(map_risk_class("Brent commodity"), RegulatoryRiskClass::COMM);
        assert_eq!(map_risk_class("credit spread"), RegulatoryRiskClass::Credit);

        let rows = vec![SensitivityRecord {
            portfolio_id: "P1".to_string(),
            trade_id: "T1".to_string(),
            risk_class: RegulatoryRiskClass::IR,
            measure: SensitivityMeasure::Delta,
            qualifier: "USD".to_string(),
            bucket: "1".to_string(),
            label1: "5Y".to_string(),
            label2: String::new(),
            amount: 1234.56,
            amount_currency: "USD".to_string(),
        }];

        let csv = to_crif_csv(&rows);
        assert!(csv.contains("PortfolioId,TradeId,RiskType"));
        assert!(csv.contains("Risk_IRDelta"));
        assert!(csv.contains("1234.5600000000"));
    }

    #[test]
    fn risk_charges_include_concentration_effects() {
        let cfg = RiskChargeConfig::baseline();

        let base = vec![
            SensitivityRecord {
                portfolio_id: "P".to_string(),
                trade_id: "A".to_string(),
                risk_class: RegulatoryRiskClass::IR,
                measure: SensitivityMeasure::Delta,
                qualifier: "USD".to_string(),
                bucket: "1".to_string(),
                label1: "2Y".to_string(),
                label2: String::new(),
                amount: 1_000_000.0,
                amount_currency: "USD".to_string(),
            },
            SensitivityRecord {
                portfolio_id: "P".to_string(),
                trade_id: "B".to_string(),
                risk_class: RegulatoryRiskClass::IR,
                measure: SensitivityMeasure::Delta,
                qualifier: "USD".to_string(),
                bucket: "2".to_string(),
                label1: "10Y".to_string(),
                label2: String::new(),
                amount: -700_000.0,
                amount_currency: "USD".to_string(),
            },
        ];

        let mut stressed = base.clone();
        stressed.push(SensitivityRecord {
            portfolio_id: "P".to_string(),
            trade_id: "C".to_string(),
            risk_class: RegulatoryRiskClass::IR,
            measure: SensitivityMeasure::Delta,
            qualifier: "USD".to_string(),
            bucket: "1".to_string(),
            label1: "30Y".to_string(),
            label2: String::new(),
            amount: 500_000_000.0,
            amount_currency: "USD".to_string(),
        });

        let q1 = compute_risk_charges(&base, &cfg);
        let q2 = compute_risk_charges(&stressed, &cfg);

        assert!(q1.delta_total > 0.0);
        assert!(q2.delta_total > q1.delta_total);
        assert!(q2.total > q1.total);
    }

    fn position(
        delta: f64,
        gamma: f64,
        vega: f64,
        theta: f64,
        rho: f64,
        quantity: f64,
    ) -> Position<&'static str> {
        Position::new(
            "trade",
            quantity,
            Greeks {
                delta,
                gamma,
                vega,
                theta,
                rho,
            },
            100.0,
            0.2,
        )
    }

    #[test]
    fn pnl_explain_and_scenario_contributions_balance() {
        let portfolio = Portfolio::new(vec![
            position(1.2, 0.4, 2.5, -0.3, 0.8, 10.0),
            position(-0.6, 0.1, 1.1, -0.2, -0.4, 5.0),
        ]);

        let mut scenario = ScenarioShock::new("hypo");
        scenario.spot_shock_pct = 0.02;
        scenario.vol_shock_pct = 0.10;
        scenario.rate_shock_abs = 0.005;
        scenario.horizon_years = 1.0 / 252.0;

        let rows = scenario_pnl_report(&portfolio, &[scenario.clone()]);
        assert_eq!(rows.len(), 1);

        let observed = rows[0].pnl + 1.0;
        let explain = pnl_explain(&portfolio, observed, &scenario);
        assert_relative_eq!(explain.unexplained, 1.0, epsilon = 1.0e-12);

        let contributions = risk_contribution_per_trade(&portfolio, &scenario);
        let contrib_sum: f64 = contributions.iter().map(|r| r.total).sum();
        assert_relative_eq!(contrib_sum, rows[0].pnl, epsilon = 1.0e-12);

        let share_sum: f64 = contributions.iter().map(|r| r.share_of_total).sum();
        assert_relative_eq!(share_sum, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn scenario_report_handles_historical_dates() {
        let portfolio = Portfolio::new(vec![position(1.0, 0.0, 0.0, 0.0, 0.0, 1.0)]);

        let mut s1 = ScenarioShock::new("hist-1");
        s1.as_of = Some("2024-10-15".to_string());
        s1.spot_shock_pct = -0.01;

        let mut s2 = ScenarioShock::new("hist-2");
        s2.as_of = Some("2025-04-02".to_string());
        s2.spot_shock_pct = 0.02;

        let rows = scenario_pnl_report(&portfolio, &[s1, s2]);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].pnl.is_finite());
        assert!(rows[1].pnl.is_finite());
        assert_eq!(rows[0].as_of.as_deref(), Some("2024-10-15"));
    }

    #[test]
    fn risk_charge_config_lookup_returns_expected_class() {
        let cfg = RiskChargeConfig::baseline();
        let ir = cfg.for_class(RegulatoryRiskClass::IR).unwrap();
        assert!(ir.delta_weight > 0.0);
    }
}
