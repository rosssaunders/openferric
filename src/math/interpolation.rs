//! Yield-curve interpolation and parametric term-structure models.
//!
//! This module provides a common [`Interpolator`] trait plus a set of
//! market-standard interpolation schemes used in rates analytics.
//!
//! References:
//! - Hagan, P. S., and West, G. (2006). "Interpolation methods for curve construction."
//! - Nelson, C. R., and Siegel, A. F. (1987). "Parsimonious modeling of yield curves."
//! - Svensson, L. E. O. (1994). "Estimating and interpreting forward interest rates."
//! - EIOPA technical documentation for Smith-Wilson (Solvency II).

use nalgebra::{DMatrix, DVector};

/// Errors returned by interpolation and calibration routines.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationError {
    /// Input validation failure.
    InvalidInput(&'static str),
    /// Query point lies outside the calibrated range while extrapolation is disabled.
    ExtrapolationDisabled { t: f64, min: f64, max: f64 },
    /// Model calibration failed.
    CalibrationFailed(&'static str),
}

/// Extrapolation policy for points outside the node range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMode {
    /// No extrapolation. Return an error when querying outside node range.
    None,
    /// Hold boundary value flat outside node range.
    Flat,
    /// Extend using the local boundary slope.
    Linear,
}

/// Interpolation method selector for yield curves.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation on continuously-compounded zero rates.
    LinearZeroRate,
    /// Linear interpolation on `ln(df)` (log-linear discount factors).
    LogLinearDiscount,
    /// Monotone-convex style interpolation using a monotone forward proxy.
    MonotoneConvex,
    /// Cubic Hermite spline with a tension parameter in `[0, 1)`.
    TensionSpline { tension: f64 },
    /// Local shape-preserving cubic Hermite interpolation.
    HermiteCubic,
    /// Monotone cubic interpolation on `ln(df)`.
    LogCubicMonotone,
    /// Nelson-Siegel parametric term structure.
    NelsonSiegel { tau: Option<f64> },
    /// Nelson-Siegel-Svensson parametric term structure.
    NelsonSiegelSvensson {
        tau1: Option<f64>,
        tau2: Option<f64>,
    },
    /// Smith-Wilson model with Ultimate Forward Rate and convergence speed.
    SmithWilson { ufr: f64, alpha: f64 },
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::LogLinearDiscount
    }
}

/// Shared interface for discount-factor curve interpolators.
///
/// Implementations return discount factors, their first time derivative,
/// and Jacobians of zero rates with respect to node discount factors.
pub trait Interpolator {
    /// Input curve nodes `(tenor_years, discount_factor)`.
    fn input_nodes(&self) -> &[(f64, f64)];

    /// Discount factor `P(t)`.
    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError>;

    /// First derivative `dP(t)/dt`.
    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError>;

    /// Continuously-compounded zero rate `z(t) = -ln(P(t))/t`.
    fn zero_rate(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let df = self.discount_factor(t)?;
        Ok(-df.ln() / t)
    }

    /// First derivative `dz(t)/dt`.
    fn zero_rate_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let df = self.discount_factor(t)?;
        let ddf = self.discount_factor_derivative(t)?;
        // z(t) = -ln(P(t))/t
        Ok(-ddf / (t * df) + df.ln() / (t * t))
    }

    /// Jacobian `d z(t) / d input_df_i`.
    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError>;
}

#[derive(Debug, Clone, PartialEq)]
enum Region {
    Left,
    Interior(usize),
    Right,
}

#[inline]
fn locate_interval(x: &[f64], t: f64) -> Region {
    if t < x[0] {
        return Region::Left;
    }
    let n = x.len();
    if t > x[n - 1] {
        return Region::Right;
    }

    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        if t < x[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Region::Interior(lo)
}

fn validate_nodes(mut nodes: Vec<(f64, f64)>) -> Result<Vec<(f64, f64)>, InterpolationError> {
    nodes.retain(|(t, df)| t.is_finite() && df.is_finite() && *t > 0.0 && *df > 0.0);
    nodes.sort_by(|a, b| a.0.total_cmp(&b.0));
    if nodes.len() < 2 {
        return Err(InterpolationError::InvalidInput(
            "need at least two positive nodes",
        ));
    }
    if nodes.windows(2).any(|w| w[1].0 <= w[0].0) {
        return Err(InterpolationError::InvalidInput(
            "tenors must be strictly increasing",
        ));
    }
    Ok(nodes)
}

fn finite_difference_jacobian<F>(
    nodes: &[(f64, f64)],
    mut eval_zero: F,
) -> Result<Vec<f64>, InterpolationError>
where
    F: FnMut(&[(f64, f64)]) -> Result<f64, InterpolationError>,
{
    let n = nodes.len();
    let mut jac = vec![0.0; n];
    for i in 0..n {
        let base = nodes[i].1;
        let h = (base.abs().max(1.0)) * 1.0e-6;

        let mut up = nodes.to_vec();
        let mut dn = nodes.to_vec();
        up[i].1 = base + h;
        dn[i].1 = (base - h).max(1.0e-12);

        if (up[i].1 - dn[i].1).abs() <= 1.0e-14 {
            jac[i] = 0.0;
            continue;
        }

        let zu = eval_zero(&up)?;
        let zd = eval_zero(&dn)?;
        jac[i] = (zu - zd) / (up[i].1 - dn[i].1);
    }
    Ok(jac)
}

#[derive(Debug, Clone, PartialEq)]
struct PiecewiseHermite {
    x: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl PiecewiseHermite {
    fn new(x: Vec<f64>, y: Vec<f64>, m: Vec<f64>, extrapolation: ExtrapolationMode) -> Self {
        Self {
            x,
            y,
            m,
            extrapolation,
        }
    }

    fn value_derivative(&self, t: f64) -> Result<(f64, f64), InterpolationError> {
        match locate_interval(&self.x, t) {
            Region::Left => match self.extrapolation {
                ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                    t,
                    min: self.x[0],
                    max: self.x[self.x.len() - 1],
                }),
                ExtrapolationMode::Flat => Ok((self.y[0], 0.0)),
                ExtrapolationMode::Linear => {
                    Ok((self.y[0] + self.m[0] * (t - self.x[0]), self.m[0]))
                }
            },
            Region::Right => {
                let n = self.x.len() - 1;
                match self.extrapolation {
                    ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                        t,
                        min: self.x[0],
                        max: self.x[n],
                    }),
                    ExtrapolationMode::Flat => Ok((self.y[n], 0.0)),
                    ExtrapolationMode::Linear => {
                        Ok((self.y[n] + self.m[n] * (t - self.x[n]), self.m[n]))
                    }
                }
            }
            Region::Interior(i) => {
                let h = self.x[i + 1] - self.x[i];
                let s = (t - self.x[i]) / h;
                let s2 = s * s;
                let s3 = s2 * s;

                let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                let h10 = s3 - 2.0 * s2 + s;
                let h01 = -2.0 * s3 + 3.0 * s2;
                let h11 = s3 - s2;

                let value = h00 * self.y[i]
                    + h10 * h * self.m[i]
                    + h01 * self.y[i + 1]
                    + h11 * h * self.m[i + 1];

                let dh00 = (6.0 * s2 - 6.0 * s) / h;
                let dh10 = 3.0 * s2 - 4.0 * s + 1.0;
                let dh01 = (-6.0 * s2 + 6.0 * s) / h;
                let dh11 = 3.0 * s2 - 2.0 * s;

                let deriv = dh00 * self.y[i]
                    + dh10 * self.m[i]
                    + dh01 * self.y[i + 1]
                    + dh11 * self.m[i + 1];

                Ok((value, deriv))
            }
        }
    }
}

fn secant_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(wx, wy)| (wy[1] - wy[0]) / (wx[1] - wx[0]))
        .collect()
}

fn fritsch_carlson_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let d = secant_slopes(x, y);
    if n == 2 {
        return vec![d[0], d[0]];
    }

    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let mut m = vec![0.0; n];

    for i in 1..(n - 1) {
        let d0 = d[i - 1];
        let d1 = d[i];
        if d0 * d1 <= 0.0 {
            m[i] = 0.0;
        } else {
            let w1 = 2.0 * h[i] + h[i - 1];
            let w2 = h[i] + 2.0 * h[i - 1];
            m[i] = (w1 + w2) / (w1 / d0 + w2 / d1);
        }
    }

    let mut m0 = ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1]);
    if m0.signum() != d[0].signum() {
        m0 = 0.0;
    } else if d[0].signum() != d[1].signum() && m0.abs() > 3.0 * d[0].abs() {
        m0 = 3.0 * d[0];
    }
    m[0] = m0;

    let k = n - 1;
    let mut mk =
        ((2.0 * h[k - 1] + h[k - 2]) * d[k - 1] - h[k - 1] * d[k - 2]) / (h[k - 1] + h[k - 2]);
    if mk.signum() != d[k - 1].signum() {
        mk = 0.0;
    } else if d[k - 1].signum() != d[k - 2].signum() && mk.abs() > 3.0 * d[k - 1].abs() {
        mk = 3.0 * d[k - 1];
    }
    m[k] = mk;

    m
}

fn harmonic_forward_slopes(x: &[f64], g: &[f64]) -> Vec<f64> {
    let n = x.len();
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let f = secant_slopes(x, g);
    if n == 2 {
        return vec![f[0], f[0]];
    }

    let mut m = vec![0.0; n];
    m[0] = f[0];
    m[n - 1] = f[n - 2];
    for i in 1..(n - 1) {
        let fl = f[i - 1];
        let fr = f[i];
        if fl * fr <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = (h[i - 1] + h[i]) / (h[i] / fl + h[i - 1] / fr);
        }
    }
    m
}

fn tension_slopes(x: &[f64], y: &[f64], tension: f64) -> Result<Vec<f64>, InterpolationError> {
    if !(0.0..1.0).contains(&tension) {
        return Err(InterpolationError::InvalidInput(
            "tension must be in [0, 1)",
        ));
    }

    let n = x.len();
    let d = secant_slopes(x, y);
    if n == 2 {
        return Ok(vec![(1.0 - tension) * d[0], (1.0 - tension) * d[0]]);
    }
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let mut m = vec![0.0; n];
    m[0] = (1.0 - tension) * d[0];
    m[n - 1] = (1.0 - tension) * d[n - 2];

    for i in 1..(n - 1) {
        let weighted = (h[i] * d[i - 1] + h[i - 1] * d[i]) / (h[i - 1] + h[i]);
        m[i] = (1.0 - tension) * weighted;
    }

    Ok(m)
}

/// Linear interpolation on continuously-compounded zero rates.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearZeroRateInterpolator {
    nodes: Vec<(f64, f64)>,
    x: Vec<f64>,
    dfs: Vec<f64>,
    zeros: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl LinearZeroRateInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let dfs: Vec<f64> = nodes.iter().map(|(_, df)| *df).collect();
        let zeros: Vec<f64> = x
            .iter()
            .zip(dfs.iter())
            .map(|(&t, &df)| -df.ln() / t)
            .collect();
        Ok(Self {
            nodes,
            x,
            dfs,
            zeros,
            extrapolation,
        })
    }

    fn zero_with_slope_weights(&self, t: f64) -> Result<(f64, f64, Vec<f64>), InterpolationError> {
        let n = self.x.len();
        let mut w = vec![0.0; n];

        let build = |i: usize, j: usize, tq: f64, w: &mut [f64]| {
            let x0 = self.x[i];
            let x1 = self.x[j];
            let h = x1 - x0;
            let wi = (x1 - tq) / h;
            let wj = (tq - x0) / h;
            w[i] = wi;
            w[j] = wj;
            let z = wi * self.zeros[i] + wj * self.zeros[j];
            let dz = (self.zeros[j] - self.zeros[i]) / h;
            (z, dz)
        };

        match locate_interval(&self.x, t) {
            Region::Left => match self.extrapolation {
                ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                    t,
                    min: self.x[0],
                    max: self.x[n - 1],
                }),
                ExtrapolationMode::Flat => {
                    w[0] = 1.0;
                    Ok((self.zeros[0], 0.0, w))
                }
                ExtrapolationMode::Linear => {
                    let (z, dz) = build(0, 1, t, &mut w);
                    Ok((z, dz, w))
                }
            },
            Region::Right => match self.extrapolation {
                ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                    t,
                    min: self.x[0],
                    max: self.x[n - 1],
                }),
                ExtrapolationMode::Flat => {
                    w[n - 1] = 1.0;
                    Ok((self.zeros[n - 1], 0.0, w))
                }
                ExtrapolationMode::Linear => {
                    let (z, dz) = build(n - 2, n - 1, t, &mut w);
                    Ok((z, dz, w))
                }
            },
            Region::Interior(i) => {
                let (z, dz) = build(i, i + 1, t, &mut w);
                Ok((z, dz, w))
            }
        }
    }
}

impl Interpolator for LinearZeroRateInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (z, _, _) = self.zero_with_slope_weights(t)?;
        Ok((-t * z).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (z, dzdt, _) = self.zero_with_slope_weights(t)?;
        let df = (-t * z).exp();
        Ok(-df * (z + t * dzdt))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        let (_, _, wz) = self.zero_with_slope_weights(t)?;
        let mut jac = vec![0.0; self.nodes.len()];
        for (i, ji) in jac.iter_mut().enumerate() {
            *ji = wz[i] * (-(1.0 / (self.x[i] * self.dfs[i])));
        }
        Ok(jac)
    }
}

/// Log-linear interpolation on discount factors.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLinearDiscountInterpolator {
    nodes: Vec<(f64, f64)>,
    x: Vec<f64>,
    dfs: Vec<f64>,
    ln_dfs: Vec<f64>,
    extrapolation: ExtrapolationMode,
    anchor_at_zero: bool,
}

impl LogLinearDiscountInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    ///
    /// If `anchor_at_zero` is true, left interpolation uses `(0, 1)` to first node.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        extrapolation: ExtrapolationMode,
        anchor_at_zero: bool,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let dfs: Vec<f64> = nodes.iter().map(|(_, df)| *df).collect();
        let ln_dfs: Vec<f64> = dfs.iter().map(|df| df.ln()).collect();
        Ok(Self {
            nodes,
            x,
            dfs,
            ln_dfs,
            extrapolation,
            anchor_at_zero,
        })
    }

    fn ln_df_with_slope_weights(&self, t: f64) -> Result<(f64, f64, Vec<f64>), InterpolationError> {
        let n = self.x.len();
        let mut w = vec![0.0; n];

        let build = |i: usize, j: usize, tq: f64, w: &mut [f64]| {
            let x0 = self.x[i];
            let x1 = self.x[j];
            let h = x1 - x0;
            let wi = (x1 - tq) / h;
            let wj = (tq - x0) / h;
            w[i] = wi;
            w[j] = wj;
            let l = wi * self.ln_dfs[i] + wj * self.ln_dfs[j];
            let dl = (self.ln_dfs[j] - self.ln_dfs[i]) / h;
            (l, dl)
        };

        match locate_interval(&self.x, t) {
            Region::Left => {
                if self.anchor_at_zero {
                    let x1 = self.x[0];
                    let l1 = self.ln_dfs[0];
                    w[0] = t / x1;
                    return Ok((w[0] * l1, l1 / x1, w));
                }
                match self.extrapolation {
                    ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                        t,
                        min: self.x[0],
                        max: self.x[n - 1],
                    }),
                    ExtrapolationMode::Flat => {
                        w[0] = 1.0;
                        Ok((self.ln_dfs[0], 0.0, w))
                    }
                    ExtrapolationMode::Linear => {
                        let (l, dl) = build(0, 1, t, &mut w);
                        Ok((l, dl, w))
                    }
                }
            }
            Region::Right => match self.extrapolation {
                ExtrapolationMode::None => Err(InterpolationError::ExtrapolationDisabled {
                    t,
                    min: self.x[0],
                    max: self.x[n - 1],
                }),
                ExtrapolationMode::Flat => {
                    w[n - 1] = 1.0;
                    Ok((self.ln_dfs[n - 1], 0.0, w))
                }
                ExtrapolationMode::Linear => {
                    let (l, dl) = build(n - 2, n - 1, t, &mut w);
                    Ok((l, dl, w))
                }
            },
            Region::Interior(i) => {
                let (l, dl) = build(i, i + 1, t, &mut w);
                Ok((l, dl, w))
            }
        }
    }
}

impl Interpolator for LogLinearDiscountInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (l, _, _) = self.ln_df_with_slope_weights(t)?;
        Ok(l.exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (l, dl, _) = self.ln_df_with_slope_weights(t)?;
        let df = l.exp();
        Ok(df * dl)
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        let (_, _, wln) = self.ln_df_with_slope_weights(t)?;
        let mut jac = vec![0.0; self.nodes.len()];
        for (i, ji) in jac.iter_mut().enumerate() {
            *ji = -(wln[i] / (t * self.dfs[i]));
        }
        Ok(jac)
    }
}

/// Monotone-convex style interpolation using a Hermite model on
/// `g(t) = -ln(P(t))` with harmonic forward slopes.
#[derive(Debug, Clone, PartialEq)]
pub struct MonotoneConvexInterpolator {
    nodes: Vec<(f64, f64)>,
    spline: PiecewiseHermite,
    extrapolation: ExtrapolationMode,
}

impl MonotoneConvexInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let g: Vec<f64> = nodes.iter().map(|(_, df)| -df.ln()).collect();
        let m = harmonic_forward_slopes(&x, &g);
        let spline = PiecewiseHermite::new(x, g, m, extrapolation);
        Ok(Self {
            nodes,
            spline,
            extrapolation,
        })
    }
}

impl Interpolator for MonotoneConvexInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (g, _) = self.spline.value_derivative(t)?;
        Ok((-g).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (g, dg) = self.spline.value_derivative(t)?;
        let df = (-g).exp();
        Ok(-df * dg)
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = MonotoneConvexInterpolator::new(trial_nodes.to_vec(), self.extrapolation)?;
            itp.zero_rate(t)
        })
    }
}

/// Tension spline interpolation on zero rates.
#[derive(Debug, Clone, PartialEq)]
pub struct TensionSplineInterpolator {
    nodes: Vec<(f64, f64)>,
    spline: PiecewiseHermite,
    tension: f64,
    extrapolation: ExtrapolationMode,
}

impl TensionSplineInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        tension: f64,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let z: Vec<f64> = nodes.iter().map(|(t, df)| -df.ln() / t).collect();
        let m = tension_slopes(&x, &z, tension)?;
        let spline = PiecewiseHermite::new(x, z, m, extrapolation);
        Ok(Self {
            nodes,
            spline,
            tension,
            extrapolation,
        })
    }
}

impl Interpolator for TensionSplineInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (z, _) = self.spline.value_derivative(t)?;
        Ok((-t * z).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (z, dz) = self.spline.value_derivative(t)?;
        let df = (-t * z).exp();
        Ok(-df * (z + t * dz))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = TensionSplineInterpolator::new(
                trial_nodes.to_vec(),
                self.tension,
                self.extrapolation,
            )?;
            itp.zero_rate(t)
        })
    }
}

/// Local monotone shape-preserving cubic Hermite interpolation on zero rates.
#[derive(Debug, Clone, PartialEq)]
pub struct HermiteCubicInterpolator {
    nodes: Vec<(f64, f64)>,
    spline: PiecewiseHermite,
    extrapolation: ExtrapolationMode,
}

impl HermiteCubicInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let z: Vec<f64> = nodes.iter().map(|(t, df)| -df.ln() / t).collect();
        let m = fritsch_carlson_slopes(&x, &z);
        let spline = PiecewiseHermite::new(x, z, m, extrapolation);
        Ok(Self {
            nodes,
            spline,
            extrapolation,
        })
    }
}

impl Interpolator for HermiteCubicInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (z, _) = self.spline.value_derivative(t)?;
        Ok((-t * z).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (z, dz) = self.spline.value_derivative(t)?;
        let df = (-t * z).exp();
        Ok(-df * (z + t * dz))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = HermiteCubicInterpolator::new(trial_nodes.to_vec(), self.extrapolation)?;
            itp.zero_rate(t)
        })
    }
}

/// Monotone cubic interpolation on `ln(P(t))`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogCubicMonotoneInterpolator {
    nodes: Vec<(f64, f64)>,
    spline: PiecewiseHermite,
    extrapolation: ExtrapolationMode,
}

impl LogCubicMonotoneInterpolator {
    /// Creates the interpolator from `(tenor, discount_factor)` nodes.
    pub fn new(
        nodes: Vec<(f64, f64)>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let x: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let l: Vec<f64> = nodes.iter().map(|(_, df)| df.ln()).collect();
        let m = fritsch_carlson_slopes(&x, &l);
        let spline = PiecewiseHermite::new(x, l, m, extrapolation);
        Ok(Self {
            nodes,
            spline,
            extrapolation,
        })
    }
}

impl Interpolator for LogCubicMonotoneInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (l, _) = self.spline.value_derivative(t)?;
        Ok(l.exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (l, dl) = self.spline.value_derivative(t)?;
        let df = l.exp();
        Ok(df * dl)
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = LogCubicMonotoneInterpolator::new(trial_nodes.to_vec(), self.extrapolation)?;
            itp.zero_rate(t)
        })
    }
}

fn ns_loading(t: f64, tau: f64) -> (f64, f64, f64) {
    let x = (t / tau).max(1.0e-12);
    let e = (-x).exp();
    let l1 = (1.0 - e) / x;
    let l2 = l1 - e;
    (1.0, l1, l2)
}

fn nss_loading(t: f64, tau1: f64, tau2: f64) -> (f64, f64, f64, f64) {
    let (_, l1, l2) = ns_loading(t, tau1);
    let x2 = (t / tau2).max(1.0e-12);
    let e2 = (-x2).exp();
    let l3 = (1.0 - e2) / x2 - e2;
    (1.0, l1, l2, l3)
}

fn fit_linear_factor_model(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<DVector<f64>, InterpolationError> {
    let xt = x.transpose();
    let xtx = &xt * x;
    let xty = &xt * y;
    xtx.lu().solve(&xty).ok_or(InterpolationError::CalibrationFailed(
        "least-squares solve failed",
    ))
}

/// Nelson-Siegel parametric model:
/// `z(t) = β0 + β1 * ((1-e^{-t/τ})/(t/τ)) + β2 * (((1-e^{-t/τ})/(t/τ)) - e^{-t/τ})`.
#[derive(Debug, Clone, PartialEq)]
pub struct NelsonSiegelInterpolator {
    nodes: Vec<(f64, f64)>,
    betas: [f64; 3],
    tau: f64,
    tau_fixed: Option<f64>,
    extrapolation: ExtrapolationMode,
}

impl NelsonSiegelInterpolator {
    /// Fits Nelson-Siegel to node zero rates.
    ///
    /// If `tau` is `None`, a one-dimensional grid search is performed for `tau`.
    pub fn fit(
        nodes: Vec<(f64, f64)>,
        tau: Option<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let times: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let zr: Vec<f64> = nodes.iter().map(|(t, df)| -df.ln() / t).collect();

        let (best_tau, best_beta) = if let Some(tau) = tau {
            if tau <= 0.0 {
                return Err(InterpolationError::InvalidInput("tau must be positive"));
            }
            let b = Self::fit_betas(&times, &zr, tau)?;
            (tau, b)
        } else {
            let mut best_err = f64::INFINITY;
            let mut best_tau = 1.0;
            let mut best_beta = [0.0; 3];
            for k in 0..160 {
                let tau = 0.05 * (1.06_f64).powi(k);
                let beta = Self::fit_betas(&times, &zr, tau)?;
                let err = Self::rmse(&times, &zr, tau, &beta);
                if err < best_err {
                    best_err = err;
                    best_tau = tau;
                    best_beta = beta;
                }
            }
            (best_tau, best_beta)
        };

        Ok(Self {
            nodes,
            betas: best_beta,
            tau: best_tau,
            tau_fixed: tau,
            extrapolation,
        })
    }

    fn fit_betas(times: &[f64], zr: &[f64], tau: f64) -> Result<[f64; 3], InterpolationError> {
        let n = times.len();
        let mut data = Vec::with_capacity(n * 3);
        for &t in times {
            let (c0, c1, c2) = ns_loading(t, tau);
            data.push(c0);
            data.push(c1);
            data.push(c2);
        }
        let x = DMatrix::from_row_slice(n, 3, &data);
        let y = DVector::from_vec(zr.to_vec());
        let beta = fit_linear_factor_model(&x, &y)?;
        Ok([beta[0], beta[1], beta[2]])
    }

    fn rmse(times: &[f64], zr: &[f64], tau: f64, beta: &[f64; 3]) -> f64 {
        let mse = times
            .iter()
            .zip(zr.iter())
            .map(|(&t, &z)| {
                let (c0, c1, c2) = ns_loading(t, tau);
                let zh = beta[0] * c0 + beta[1] * c1 + beta[2] * c2;
                let e = zh - z;
                e * e
            })
            .sum::<f64>()
            / times.len() as f64;
        mse.sqrt()
    }

    fn model_zero(&self, t: f64) -> f64 {
        let (c0, c1, c2) = ns_loading(t.max(1.0e-12), self.tau);
        self.betas[0] * c0 + self.betas[1] * c1 + self.betas[2] * c2
    }

    fn effective_t(&self, t: f64) -> Result<(f64, bool), InterpolationError> {
        let min = self.nodes[0].0;
        let max = self.nodes[self.nodes.len() - 1].0;
        if (min..=max).contains(&t) {
            return Ok((t, false));
        }
        match self.extrapolation {
            ExtrapolationMode::None => {
                Err(InterpolationError::ExtrapolationDisabled { t, min, max })
            }
            ExtrapolationMode::Flat => Ok((t.clamp(min, max), true)),
            ExtrapolationMode::Linear => Ok((t, false)),
        }
    }
}

impl Interpolator for NelsonSiegelInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (te, flat) = self.effective_t(t)?;
        if flat {
            let z = self.model_zero(te);
            return Ok((-t * z).exp());
        }
        let z = self.model_zero(te);
        Ok((-t * z).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (_, flat) = self.effective_t(t)?;
        if flat {
            return Ok(0.0);
        }
        let h = 1.0e-5_f64.max(1.0e-5 * t.abs());
        let up = self.discount_factor(t + h)?;
        let dn = self.discount_factor((t - h).max(1.0e-10))?;
        Ok((up - dn) / (2.0 * h))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = NelsonSiegelInterpolator::fit(
                trial_nodes.to_vec(),
                self.tau_fixed,
                self.extrapolation,
            )?;
            itp.zero_rate(t)
        })
    }
}

/// Nelson-Siegel-Svensson parametric model:
/// `z(t) = β0 + β1 L1(t,τ1) + β2 L2(t,τ1) + β3 L2(t,τ2)`.
#[derive(Debug, Clone, PartialEq)]
pub struct NelsonSiegelSvenssonInterpolator {
    nodes: Vec<(f64, f64)>,
    betas: [f64; 4],
    tau1: f64,
    tau2: f64,
    tau1_fixed: Option<f64>,
    tau2_fixed: Option<f64>,
    extrapolation: ExtrapolationMode,
}

impl NelsonSiegelSvenssonInterpolator {
    /// Fits NSS to node zero rates.
    ///
    /// If either `tau1` or `tau2` is `None`, a two-dimensional grid search is used.
    pub fn fit(
        nodes: Vec<(f64, f64)>,
        tau1: Option<f64>,
        tau2: Option<f64>,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        let nodes = validate_nodes(nodes)?;
        let times: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();
        let zr: Vec<f64> = nodes.iter().map(|(t, df)| -df.ln() / t).collect();

        let (best_tau1, best_tau2, best_beta) = if let (Some(t1), Some(t2)) = (tau1, tau2) {
            if t1 <= 0.0 || t2 <= 0.0 {
                return Err(InterpolationError::InvalidInput("taus must be positive"));
            }
            let b = Self::fit_betas(&times, &zr, t1, t2)?;
            (t1, t2, b)
        } else {
            let mut best = (f64::INFINITY, 1.0, 3.0, [0.0; 4]);
            for i in 0..36 {
                for j in 0..36 {
                    let t1 = 0.10 * (1.08_f64).powi(i);
                    let t2 = 0.20 * (1.08_f64).powi(j);
                    if (t1 - t2).abs() < 1.0e-3 {
                        continue;
                    }
                    let beta = Self::fit_betas(&times, &zr, t1, t2)?;
                    let err = Self::rmse(&times, &zr, t1, t2, &beta);
                    if err < best.0 {
                        best = (err, t1, t2, beta);
                    }
                }
            }
            (best.1, best.2, best.3)
        };

        Ok(Self {
            nodes,
            betas: best_beta,
            tau1: best_tau1,
            tau2: best_tau2,
            tau1_fixed: tau1,
            tau2_fixed: tau2,
            extrapolation,
        })
    }

    fn fit_betas(
        times: &[f64],
        zr: &[f64],
        tau1: f64,
        tau2: f64,
    ) -> Result<[f64; 4], InterpolationError> {
        let n = times.len();
        let mut data = Vec::with_capacity(n * 4);
        for &t in times {
            let (c0, c1, c2, c3) = nss_loading(t, tau1, tau2);
            data.push(c0);
            data.push(c1);
            data.push(c2);
            data.push(c3);
        }
        let x = DMatrix::from_row_slice(n, 4, &data);
        let y = DVector::from_vec(zr.to_vec());
        let beta = fit_linear_factor_model(&x, &y)?;
        Ok([beta[0], beta[1], beta[2], beta[3]])
    }

    fn rmse(times: &[f64], zr: &[f64], tau1: f64, tau2: f64, beta: &[f64; 4]) -> f64 {
        let mse = times
            .iter()
            .zip(zr.iter())
            .map(|(&t, &z)| {
                let (c0, c1, c2, c3) = nss_loading(t, tau1, tau2);
                let zh = beta[0] * c0 + beta[1] * c1 + beta[2] * c2 + beta[3] * c3;
                let e = zh - z;
                e * e
            })
            .sum::<f64>()
            / times.len() as f64;
        mse.sqrt()
    }

    fn model_zero(&self, t: f64) -> f64 {
        let (c0, c1, c2, c3) = nss_loading(t.max(1.0e-12), self.tau1, self.tau2);
        self.betas[0] * c0 + self.betas[1] * c1 + self.betas[2] * c2 + self.betas[3] * c3
    }

    fn effective_t(&self, t: f64) -> Result<(f64, bool), InterpolationError> {
        let min = self.nodes[0].0;
        let max = self.nodes[self.nodes.len() - 1].0;
        if (min..=max).contains(&t) {
            return Ok((t, false));
        }
        match self.extrapolation {
            ExtrapolationMode::None => {
                Err(InterpolationError::ExtrapolationDisabled { t, min, max })
            }
            ExtrapolationMode::Flat => Ok((t.clamp(min, max), true)),
            ExtrapolationMode::Linear => Ok((t, false)),
        }
    }
}

impl Interpolator for NelsonSiegelSvenssonInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (te, _) = self.effective_t(t)?;
        let z = self.model_zero(te);
        Ok((-t * z).exp())
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (_, flat) = self.effective_t(t)?;
        if flat {
            return Ok(0.0);
        }
        let h = 1.0e-5_f64.max(1.0e-5 * t.abs());
        let up = self.discount_factor(t + h)?;
        let dn = self.discount_factor((t - h).max(1.0e-10))?;
        Ok((up - dn) / (2.0 * h))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }
        finite_difference_jacobian(&self.nodes, |trial_nodes| {
            let itp = NelsonSiegelSvenssonInterpolator::fit(
                trial_nodes.to_vec(),
                self.tau1_fixed,
                self.tau2_fixed,
                self.extrapolation,
            )?;
            itp.zero_rate(t)
        })
    }
}

fn wilson_function(t: f64, u: f64, ufr: f64, alpha: f64) -> f64 {
    let min_tu = t.min(u);
    let max_tu = t.max(u);
    let b = alpha * min_tu - (-alpha * max_tu).exp() * (alpha * min_tu).sinh();
    (-(ufr) * (t + u)).exp() * b
}

/// Smith-Wilson term structure model used in Solvency II extrapolation.
///
/// The model solves:
/// `P(t) = exp(-ufr * t) + sum_i zeta_i * W(t, u_i)`,
/// where `W` is the Wilson kernel and `zeta` is calibrated to input nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct SmithWilsonInterpolator {
    nodes: Vec<(f64, f64)>,
    times: Vec<f64>,
    ufr: f64,
    alpha: f64,
    zeta: DVector<f64>,
    w_inv: DMatrix<f64>,
    extrapolation: ExtrapolationMode,
}

impl SmithWilsonInterpolator {
    /// Fits Smith-Wilson to `(tenor, discount_factor)` nodes.
    pub fn fit(
        nodes: Vec<(f64, f64)>,
        ufr: f64,
        alpha: f64,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        if ufr < 0.0 {
            return Err(InterpolationError::InvalidInput("ufr must be >= 0"));
        }
        if alpha <= 0.0 {
            return Err(InterpolationError::InvalidInput("alpha must be > 0"));
        }

        let nodes = validate_nodes(nodes)?;
        let n = nodes.len();
        let times: Vec<f64> = nodes.iter().map(|(t, _)| *t).collect();

        let mut w_data = Vec::with_capacity(n * n);
        for &ti in &times {
            for &tj in &times {
                w_data.push(wilson_function(ti, tj, ufr, alpha));
            }
        }
        let w = DMatrix::from_row_slice(n, n, &w_data);
        let w_inv = w
            .clone()
            .try_inverse()
            .ok_or(InterpolationError::CalibrationFailed(
                "Wilson matrix not invertible",
            ))?;

        let p = DVector::from_iterator(n, nodes.iter().map(|(_, df)| *df));
        let p_ufr = DVector::from_iterator(n, times.iter().map(|t| (-ufr * t).exp()));
        let rhs = p - p_ufr;
        let zeta = &w_inv * rhs;

        Ok(Self {
            nodes,
            times,
            ufr,
            alpha,
            zeta,
            w_inv,
            extrapolation,
        })
    }

    fn base_df(&self, t: f64) -> f64 {
        let mut df = (-self.ufr * t).exp();
        for (j, &u) in self.times.iter().enumerate() {
            df += self.zeta[j] * wilson_function(t, u, self.ufr, self.alpha);
        }
        df.max(1.0e-16)
    }

    fn effective_t(&self, t: f64) -> Result<(f64, bool), InterpolationError> {
        let min = self.times[0];
        let max = self.times[self.times.len() - 1];
        if (min..=max).contains(&t) {
            return Ok((t, false));
        }
        match self.extrapolation {
            ExtrapolationMode::None => {
                Err(InterpolationError::ExtrapolationDisabled { t, min, max })
            }
            ExtrapolationMode::Flat => Ok((t.clamp(min, max), true)),
            ExtrapolationMode::Linear => Ok((t, false)),
        }
    }
}

impl Interpolator for SmithWilsonInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        let (te, _) = self.effective_t(t)?;
        Ok(self.base_df(te))
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let (_, flat) = self.effective_t(t)?;
        if flat {
            return Ok(0.0);
        }
        let h = 1.0e-5_f64.max(1.0e-5 * t.abs());
        let up = self.discount_factor(t + h)?;
        let dn = self.discount_factor((t - h).max(1.0e-10))?;
        Ok((up - dn) / (2.0 * h))
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.nodes.len()]);
        }

        let (te, _) = self.effective_t(t)?;
        let mut w_t = DVector::zeros(self.times.len());
        for (j, &u) in self.times.iter().enumerate() {
            w_t[j] = wilson_function(te, u, self.ufr, self.alpha);
        }
        let sens_df = self.w_inv.transpose() * w_t;
        let df = self.discount_factor(t)?;

        let mut jac = vec![0.0; self.nodes.len()];
        for (i, ji) in jac.iter_mut().enumerate() {
            *ji = -sens_df[i] / (t * df);
        }
        Ok(jac)
    }
}

/// Type-erased interpolator container.
#[derive(Debug, Clone, PartialEq)]
pub enum AnyInterpolator {
    LinearZeroRate(LinearZeroRateInterpolator),
    LogLinearDiscount(LogLinearDiscountInterpolator),
    MonotoneConvex(MonotoneConvexInterpolator),
    TensionSpline(TensionSplineInterpolator),
    HermiteCubic(HermiteCubicInterpolator),
    LogCubicMonotone(LogCubicMonotoneInterpolator),
    NelsonSiegel(NelsonSiegelInterpolator),
    NelsonSiegelSvensson(NelsonSiegelSvenssonInterpolator),
    SmithWilson(SmithWilsonInterpolator),
}

impl Interpolator for AnyInterpolator {
    fn input_nodes(&self) -> &[(f64, f64)] {
        match self {
            Self::LinearZeroRate(x) => x.input_nodes(),
            Self::LogLinearDiscount(x) => x.input_nodes(),
            Self::MonotoneConvex(x) => x.input_nodes(),
            Self::TensionSpline(x) => x.input_nodes(),
            Self::HermiteCubic(x) => x.input_nodes(),
            Self::LogCubicMonotone(x) => x.input_nodes(),
            Self::NelsonSiegel(x) => x.input_nodes(),
            Self::NelsonSiegelSvensson(x) => x.input_nodes(),
            Self::SmithWilson(x) => x.input_nodes(),
        }
    }

    fn discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        match self {
            Self::LinearZeroRate(x) => x.discount_factor(t),
            Self::LogLinearDiscount(x) => x.discount_factor(t),
            Self::MonotoneConvex(x) => x.discount_factor(t),
            Self::TensionSpline(x) => x.discount_factor(t),
            Self::HermiteCubic(x) => x.discount_factor(t),
            Self::LogCubicMonotone(x) => x.discount_factor(t),
            Self::NelsonSiegel(x) => x.discount_factor(t),
            Self::NelsonSiegelSvensson(x) => x.discount_factor(t),
            Self::SmithWilson(x) => x.discount_factor(t),
        }
    }

    fn discount_factor_derivative(&self, t: f64) -> Result<f64, InterpolationError> {
        match self {
            Self::LinearZeroRate(x) => x.discount_factor_derivative(t),
            Self::LogLinearDiscount(x) => x.discount_factor_derivative(t),
            Self::MonotoneConvex(x) => x.discount_factor_derivative(t),
            Self::TensionSpline(x) => x.discount_factor_derivative(t),
            Self::HermiteCubic(x) => x.discount_factor_derivative(t),
            Self::LogCubicMonotone(x) => x.discount_factor_derivative(t),
            Self::NelsonSiegel(x) => x.discount_factor_derivative(t),
            Self::NelsonSiegelSvensson(x) => x.discount_factor_derivative(t),
            Self::SmithWilson(x) => x.discount_factor_derivative(t),
        }
    }

    fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        match self {
            Self::LinearZeroRate(x) => x.jacobian_zero_rate(t),
            Self::LogLinearDiscount(x) => x.jacobian_zero_rate(t),
            Self::MonotoneConvex(x) => x.jacobian_zero_rate(t),
            Self::TensionSpline(x) => x.jacobian_zero_rate(t),
            Self::HermiteCubic(x) => x.jacobian_zero_rate(t),
            Self::LogCubicMonotone(x) => x.jacobian_zero_rate(t),
            Self::NelsonSiegel(x) => x.jacobian_zero_rate(t),
            Self::NelsonSiegelSvensson(x) => x.jacobian_zero_rate(t),
            Self::SmithWilson(x) => x.jacobian_zero_rate(t),
        }
    }
}

/// Factory for building interpolators from method metadata.
pub fn build_interpolator(
    nodes: Vec<(f64, f64)>,
    method: &InterpolationMethod,
    extrapolation: ExtrapolationMode,
) -> Result<AnyInterpolator, InterpolationError> {
    match method {
        InterpolationMethod::LinearZeroRate => Ok(AnyInterpolator::LinearZeroRate(
            LinearZeroRateInterpolator::new(nodes, extrapolation)?,
        )),
        InterpolationMethod::LogLinearDiscount => Ok(AnyInterpolator::LogLinearDiscount(
            LogLinearDiscountInterpolator::new(nodes, extrapolation, true)?,
        )),
        InterpolationMethod::MonotoneConvex => Ok(AnyInterpolator::MonotoneConvex(
            MonotoneConvexInterpolator::new(nodes, extrapolation)?,
        )),
        InterpolationMethod::TensionSpline { tension } => Ok(AnyInterpolator::TensionSpline(
            TensionSplineInterpolator::new(nodes, *tension, extrapolation)?,
        )),
        InterpolationMethod::HermiteCubic => Ok(AnyInterpolator::HermiteCubic(
            HermiteCubicInterpolator::new(nodes, extrapolation)?,
        )),
        InterpolationMethod::LogCubicMonotone => Ok(AnyInterpolator::LogCubicMonotone(
            LogCubicMonotoneInterpolator::new(nodes, extrapolation)?,
        )),
        InterpolationMethod::NelsonSiegel { tau } => Ok(AnyInterpolator::NelsonSiegel(
            NelsonSiegelInterpolator::fit(nodes, *tau, extrapolation)?,
        )),
        InterpolationMethod::NelsonSiegelSvensson { tau1, tau2 } => {
            Ok(AnyInterpolator::NelsonSiegelSvensson(
                NelsonSiegelSvenssonInterpolator::fit(nodes, *tau1, *tau2, extrapolation)?,
            ))
        }
        InterpolationMethod::SmithWilson { ufr, alpha } => Ok(AnyInterpolator::SmithWilson(
            SmithWilsonInterpolator::fit(nodes, *ufr, *alpha, extrapolation)?,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn flat_nodes(rate: f64) -> Vec<(f64, f64)> {
        [0.5, 1.0, 2.0, 5.0, 10.0]
            .iter()
            .map(|&t| (t, (-rate * t).exp()))
            .collect()
    }

    #[test]
    fn linear_zero_and_log_linear_match_flat_curve() {
        let nodes = flat_nodes(0.03);
        let lin =
            LinearZeroRateInterpolator::new(nodes.clone(), ExtrapolationMode::Linear).unwrap();
        let loglin =
            LogLinearDiscountInterpolator::new(nodes.clone(), ExtrapolationMode::Linear, true)
                .unwrap();

        for &t in &[0.25, 0.75, 1.5, 3.0, 7.0, 12.0] {
            let expected = (-0.03_f64 * t).exp();
            assert_relative_eq!(lin.discount_factor(t).unwrap(), expected, epsilon = 1.0e-10);
            assert_relative_eq!(
                loglin.discount_factor(t).unwrap(),
                expected,
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    fn monotone_methods_preserve_positive_discount() {
        let nodes = vec![
            (1.0, 0.99),
            (2.0, 0.965),
            (3.0, 0.94),
            (5.0, 0.89),
            (10.0, 0.77),
        ];
        let mc = MonotoneConvexInterpolator::new(nodes.clone(), ExtrapolationMode::Linear).unwrap();
        let hc = HermiteCubicInterpolator::new(nodes.clone(), ExtrapolationMode::Linear).unwrap();
        let lc = LogCubicMonotoneInterpolator::new(nodes, ExtrapolationMode::Linear).unwrap();

        for i in 1..200 {
            let t = i as f64 * 0.05;
            let df_mc = mc.discount_factor(t).unwrap();
            let df_hc = hc.discount_factor(t).unwrap();
            let df_lc = lc.discount_factor(t).unwrap();
            assert!(df_mc > 0.0);
            assert!(df_hc > 0.0);
            assert!(df_lc > 0.0);
        }
    }

    #[test]
    fn tension_spline_limits_match_linear_at_high_tension() {
        let nodes = flat_nodes(0.025);
        let lin =
            LinearZeroRateInterpolator::new(nodes.clone(), ExtrapolationMode::Linear).unwrap();
        let tns =
            TensionSplineInterpolator::new(nodes.clone(), 0.999_999, ExtrapolationMode::Linear)
                .unwrap();
        for &t in &[0.75, 1.4, 2.3, 6.0] {
            assert_relative_eq!(
                tns.zero_rate(t).unwrap(),
                lin.zero_rate(t).unwrap(),
                epsilon = 1.0e-5
            );
        }
    }

    #[test]
    fn extrapolation_none_errors_outside_range() {
        let nodes = flat_nodes(0.02);
        let itp = HermiteCubicInterpolator::new(nodes, ExtrapolationMode::None).unwrap();
        assert!(matches!(
            itp.discount_factor(12.0),
            Err(InterpolationError::ExtrapolationDisabled { .. })
        ));
    }

    #[test]
    fn ns_and_nss_fit_synthetic_curves() {
        // Synthetic NS.
        let ns_true = NelsonSiegelInterpolator {
            nodes: flat_nodes(0.03),
            betas: [0.02, -0.01, 0.03],
            tau: 1.8,
            tau_fixed: Some(1.8),
            extrapolation: ExtrapolationMode::Linear,
        };
        let ns_nodes: Vec<(f64, f64)> = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
            .iter()
            .map(|&t| (t, ns_true.discount_factor(t).unwrap()))
            .collect();
        let ns_fit =
            NelsonSiegelInterpolator::fit(ns_nodes, None, ExtrapolationMode::Linear).unwrap();
        for &t in &[0.75, 1.5, 4.0, 8.0] {
            assert_relative_eq!(
                ns_fit.zero_rate(t).unwrap(),
                ns_true.zero_rate(t).unwrap(),
                epsilon = 1.0e-3
            );
        }

        // Synthetic NSS.
        let nss_true = NelsonSiegelSvenssonInterpolator {
            nodes: flat_nodes(0.03),
            betas: [0.018, -0.015, 0.020, -0.007],
            tau1: 1.5,
            tau2: 4.0,
            tau1_fixed: Some(1.5),
            tau2_fixed: Some(4.0),
            extrapolation: ExtrapolationMode::Linear,
        };
        let nss_nodes: Vec<(f64, f64)> = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]
            .iter()
            .map(|&t| (t, nss_true.discount_factor(t).unwrap()))
            .collect();
        let nss_fit =
            NelsonSiegelSvenssonInterpolator::fit(nss_nodes, None, None, ExtrapolationMode::Linear)
                .unwrap();
        for &t in &[0.75, 1.5, 4.0, 8.0, 15.0] {
            assert_relative_eq!(
                nss_fit.zero_rate(t).unwrap(),
                nss_true.zero_rate(t).unwrap(),
                epsilon = 1.5e-3
            );
        }
    }

    #[test]
    fn smith_wilson_reproduces_input_nodes() {
        let nodes = vec![
            (1.0, 0.98),
            (2.0, 0.955),
            (3.0, 0.93),
            (5.0, 0.885),
            (10.0, 0.79),
        ];
        let sw =
            SmithWilsonInterpolator::fit(nodes.clone(), 0.032, 0.12, ExtrapolationMode::Linear)
                .unwrap();
        for (t, df) in nodes {
            assert_relative_eq!(sw.discount_factor(t).unwrap(), df, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn jacobian_matches_finite_difference_for_log_linear() {
        let nodes = vec![(1.0, 0.98), (2.0, 0.95), (3.0, 0.92), (5.0, 0.86)];
        let itp =
            LogLinearDiscountInterpolator::new(nodes.clone(), ExtrapolationMode::Linear, true)
                .unwrap();
        let t = 2.4;
        let jac = itp.jacobian_zero_rate(t).unwrap();
        let jac_fd = finite_difference_jacobian(&nodes, |trial| {
            let tmp = LogLinearDiscountInterpolator::new(
                trial.to_vec(),
                ExtrapolationMode::Linear,
                true,
            )?;
            tmp.zero_rate(t)
        })
        .unwrap();
        for (a, b) in jac.iter().zip(jac_fd.iter()) {
            assert_relative_eq!(a, b, epsilon = 1.0e-8);
        }
    }
}
