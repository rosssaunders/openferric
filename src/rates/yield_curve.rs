//! Yield-curve construction and interpolation.
//!
//! The default behavior remains log-linear interpolation on discount factors,
//! but this module also supports production methods used by rates desks,
//! including monotone-convex, shape-preserving cubic variants, and parametric
//! families.
//!
//! References:
//! - Hagan and West (2006), *Interpolation Methods for Curve Construction*.
//! - Nelson and Siegel (1987); Svensson (1994).
//! - EIOPA Smith-Wilson technical specification (Solvency II).

use crate::math::interpolation::{
    AnyInterpolator, ExtrapolationMode, HermiteMonotoneInterpolator, InterpolationError,
    Interpolator, LinearInterpolator, LogCubicMonotoneInterpolator, LogLinearInterpolator,
    MonotoneConvexInterpolator, NelsonSiegelInterpolator, NelsonSiegelSvenssonInterpolator,
    SmithWilsonInterpolator, TensionSplineInterpolator,
};

/// Interpolation method used to convert input nodes into a full curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum YieldCurveInterpolationMethod {
    /// Log-linear interpolation on discount factors.
    LogLinearDiscount,
    /// Piecewise-linear interpolation on continuously-compounded zero rates.
    LinearZeroRate,
    /// Hagan-West inspired monotone-convex cubic interpolation on zero rates.
    MonotoneConvex,
    /// Cardinal tension spline on zero rates.
    TensionSpline { tension: f64 },
    /// Monotone cubic Hermite interpolation on zero rates.
    HermiteMonotone,
    /// Monotone cubic interpolation on `ln(df)` values.
    LogCubicMonotone,
    /// Nelson-Siegel parametric fit on zero rates.
    NelsonSiegel,
    /// Nelson-Siegel-Svensson parametric fit on zero rates.
    NelsonSiegelSvensson,
    /// Smith-Wilson discount-factor fit.
    SmithWilson { ufr: f64, alpha: f64 },
}

/// Interpolation configuration for [`YieldCurve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YieldCurveInterpolationSettings {
    pub method: YieldCurveInterpolationMethod,
    pub extrapolation: ExtrapolationMode,
}

impl Default for YieldCurveInterpolationSettings {
    fn default() -> Self {
        Self {
            method: YieldCurveInterpolationMethod::LogLinearDiscount,
            extrapolation: ExtrapolationMode::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterpolatedQuantity {
    DiscountFactor,
    ZeroRate,
}

#[derive(Debug, Clone)]
struct InterpolationState {
    interpolator: AnyInterpolator,
    quantity: InterpolatedQuantity,
    has_origin_node: bool,
    origin_linked_to_first_input: bool,
}

/// Discount-factor term structure keyed by maturity tenor in years.
///
/// # Examples
/// ```rust
/// use openferric::rates::YieldCurve;
///
/// let yc = YieldCurve::new(vec![(0.5, 0.98), (1.0, 0.95)]);
/// assert!(yc.discount_factor(0.75) < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct YieldCurve {
    /// Curve nodes as `(tenor, discount_factor)`.
    pub tenors: Vec<(f64, f64)>,
    interpolation: YieldCurveInterpolationSettings,
    interpolation_state: Option<InterpolationState>,
}

impl PartialEq for YieldCurve {
    fn eq(&self, other: &Self) -> bool {
        self.tenors == other.tenors && self.interpolation == other.interpolation
    }
}

impl YieldCurve {
    /// Creates a curve from unsorted discount-factor nodes using default interpolation
    /// (`log-linear discount`, `linear` extrapolation).
    pub fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self::new_with_settings(tenors, YieldCurveInterpolationSettings::default())
            .unwrap_or_else(|_| Self::empty())
    }

    /// Creates a curve with explicit interpolation settings.
    pub fn new_with_settings(
        tenors: Vec<(f64, f64)>,
        settings: YieldCurveInterpolationSettings,
    ) -> Result<Self, InterpolationError> {
        let cleaned = sanitize_curve_nodes(tenors);
        let interpolation_state = build_interpolation_state(&cleaned, settings)?;

        Ok(Self {
            tenors: cleaned,
            interpolation: settings,
            interpolation_state,
        })
    }

    fn empty() -> Self {
        Self {
            tenors: Vec::new(),
            interpolation: YieldCurveInterpolationSettings::default(),
            interpolation_state: None,
        }
    }

    /// Returns active interpolation settings.
    pub fn interpolation_settings(&self) -> YieldCurveInterpolationSettings {
        self.interpolation
    }

    /// Returns discount factor at tenor `t`.
    ///
    /// If extrapolation mode is `Error`, prefer `try_discount_factor` to receive
    /// explicit errors. This method returns `NaN` on extrapolation failures.
    pub fn discount_factor(&self, t: f64) -> f64 {
        self.try_discount_factor(t).unwrap_or(f64::NAN)
    }

    /// Returns discount factor at tenor `t` with explicit error handling.
    pub fn try_discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        if self.tenors.is_empty() {
            return Ok(1.0);
        }

        let Some(state) = &self.interpolation_state else {
            return Ok(discount_factor_one_point(
                self.tenors[0].0,
                self.tenors[0].1,
                t,
                self.interpolation.extrapolation,
            ));
        };

        let raw = state.interpolator.value(t)?;
        let df = match state.quantity {
            InterpolatedQuantity::DiscountFactor => raw,
            InterpolatedQuantity::ZeroRate => (-raw * t).exp(),
        };
        Ok(df.max(1.0e-14))
    }

    /// Returns continuously-compounded zero rate at tenor `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        self.try_zero_rate(t).unwrap_or(f64::NAN)
    }

    /// Returns continuously-compounded zero rate at tenor `t` with explicit errors.
    pub fn try_zero_rate(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }

        let Some(state) = &self.interpolation_state else {
            return Ok(-discount_factor_one_point(
                self.tenors[0].0,
                self.tenors[0].1,
                t,
                self.interpolation.extrapolation,
            )
            .ln()
                / t);
        };

        match state.quantity {
            InterpolatedQuantity::ZeroRate => state.interpolator.value(t),
            InterpolatedQuantity::DiscountFactor => {
                let df = state.interpolator.value(t)?;
                Ok(-df.max(1.0e-14).ln() / t)
            }
        }
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        self.try_forward_rate(t1, t2).unwrap_or(f64::NAN)
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`
    /// with explicit interpolation errors.
    pub fn try_forward_rate(&self, t1: f64, t2: f64) -> Result<f64, InterpolationError> {
        assert!(t2 > t1, "t2 must be greater than t1");
        let df1 = self.try_discount_factor(t1)?;
        let df2 = self.try_discount_factor(t2)?;
        Ok((df1 / df2).ln() / (t2 - t1))
    }

    /// Instantaneous forward rate `f(t) = -d/dt ln(P(t))`.
    pub fn instantaneous_forward_rate(&self, t: f64) -> f64 {
        self.try_instantaneous_forward_rate(t).unwrap_or(f64::NAN)
    }

    /// Instantaneous forward rate with explicit interpolation errors.
    pub fn try_instantaneous_forward_rate(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(self.try_zero_rate(1.0e-8)?);
        }

        let Some(state) = &self.interpolation_state else {
            return Ok(self.try_zero_rate(t)?);
        };

        match state.quantity {
            InterpolatedQuantity::DiscountFactor => {
                let p = state.interpolator.value(t)?;
                let dp = state.interpolator.derivative(t)?;
                Ok(-(dp / p.max(1.0e-14)))
            }
            InterpolatedQuantity::ZeroRate => {
                let z = state.interpolator.value(t)?;
                let dz = state.interpolator.derivative(t)?;
                Ok(z + t * dz)
            }
        }
    }

    /// Jacobian of discount factor wrt input pillar discount factors.
    pub fn discount_factor_jacobian(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if self.tenors.is_empty() {
            return Ok(Vec::new());
        }
        if t <= 0.0 {
            return Ok(vec![0.0; self.tenors.len()]);
        }

        let Some(state) = &self.interpolation_state else {
            return Ok(vec![0.0; self.tenors.len()]);
        };

        let raw_j = state.interpolator.jacobian(t)?;
        let j_no_origin = map_jacobian_from_interpolator(raw_j, state, self.tenors.len());

        match state.quantity {
            InterpolatedQuantity::DiscountFactor => Ok(j_no_origin),
            InterpolatedQuantity::ZeroRate => {
                let z = state.interpolator.value(t)?;
                let df = (-z * t).exp();
                let mut out = vec![0.0; self.tenors.len()];
                for i in 0..self.tenors.len() {
                    let ti = self.tenors[i].0;
                    let dzi_ddfi = -1.0 / (ti * self.tenors[i].1.max(1.0e-14));
                    out[i] = -t * df * j_no_origin[i] * dzi_ddfi;
                }
                Ok(out)
            }
        }
    }

    /// Jacobian of zero rate wrt input pillar discount factors.
    pub fn zero_rate_jacobian(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if self.tenors.is_empty() {
            return Ok(Vec::new());
        }
        if t <= 0.0 {
            return Ok(vec![0.0; self.tenors.len()]);
        }

        let Some(state) = &self.interpolation_state else {
            return Ok(vec![0.0; self.tenors.len()]);
        };

        let raw_j = state.interpolator.jacobian(t)?;
        let j_no_origin = map_jacobian_from_interpolator(raw_j, state, self.tenors.len());

        match state.quantity {
            InterpolatedQuantity::DiscountFactor => {
                let p = state.interpolator.value(t)?.max(1.0e-14);
                Ok(j_no_origin
                    .iter()
                    .map(|dpi| -(dpi / (t * p)))
                    .collect::<Vec<_>>())
            }
            InterpolatedQuantity::ZeroRate => {
                let mut out = vec![0.0; self.tenors.len()];
                for i in 0..self.tenors.len() {
                    let ti = self.tenors[i].0;
                    let dzi_ddfi = -1.0 / (ti * self.tenors[i].1.max(1.0e-14));
                    out[i] = j_no_origin[i] * dzi_ddfi;
                }
                Ok(out)
            }
        }
    }
}

/// Helper constructors for simple curve bootstrapping.
pub struct YieldCurveBuilder;

impl YieldCurveBuilder {
    /// Builds a curve from simple deposit rates `(tenor, rate)`.
    pub fn from_deposits(deposits: &[(f64, f64)]) -> YieldCurve {
        Self::from_deposits_with_settings(deposits, YieldCurveInterpolationSettings::default())
            .unwrap_or_else(|_| YieldCurve::empty())
    }

    /// Builds a curve from simple deposit rates with explicit interpolation.
    pub fn from_deposits_with_settings(
        deposits: &[(f64, f64)],
        settings: YieldCurveInterpolationSettings,
    ) -> Result<YieldCurve, InterpolationError> {
        let points = deposits
            .iter()
            .filter(|(tenor, _)| *tenor > 0.0)
            .map(|(tenor, rate)| (*tenor, 1.0 / (1.0 + rate * tenor)))
            .collect();
        YieldCurve::new_with_settings(points, settings)
    }

    /// Bootstraps discount factors from par swap rates `(tenor, fixed_rate)`.
    pub fn from_swap_rates(swap_rates: &[(f64, f64)], frequency: usize) -> YieldCurve {
        Self::from_swap_rates_with_settings(
            swap_rates,
            frequency,
            YieldCurveInterpolationSettings::default(),
        )
        .unwrap_or_else(|_| YieldCurve::empty())
    }

    /// Bootstraps discount factors from par swap rates with explicit interpolation.
    pub fn from_swap_rates_with_settings(
        swap_rates: &[(f64, f64)],
        frequency: usize,
        settings: YieldCurveInterpolationSettings,
    ) -> Result<YieldCurve, InterpolationError> {
        assert!(frequency > 0, "frequency must be > 0");

        let mut sorted = swap_rates.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut points: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
        let freq = frequency as f64;
        let dt = 1.0 / freq;

        for (tenor, swap_rate) in sorted {
            if tenor <= 0.0 {
                continue;
            }

            let periods = (tenor * freq).round() as usize;
            if periods == 0 {
                continue;
            }

            let coupon = swap_rate / freq;
            let mut pv_coupons = 0.0;
            for i in 1..periods {
                let ti = i as f64 * dt;
                pv_coupons += coupon * discount_factor_from_points(&points, ti, settings)?;
            }

            let mut df = (1.0 - pv_coupons) / (1.0 + coupon);
            if df <= 0.0 {
                df = 1.0e-12;
            }
            points.push((tenor, df));
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        YieldCurve::new_with_settings(points, settings)
    }
}

fn sanitize_curve_nodes(mut tenors: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    tenors.retain(|(t, df)| *t > 0.0 && *df > 0.0 && t.is_finite() && df.is_finite());
    tenors.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut out: Vec<(f64, f64)> = Vec::with_capacity(tenors.len());
    for (t, df) in tenors {
        if let Some(last) = out.last_mut()
            && (last.0 - t).abs() <= 1.0e-12
        {
            last.1 = df.max(1.0e-14);
            continue;
        }
        out.push((t, df.max(1.0e-14)));
    }
    out
}

fn build_interpolation_state(
    tenors: &[(f64, f64)],
    settings: YieldCurveInterpolationSettings,
) -> Result<Option<InterpolationState>, InterpolationError> {
    if tenors.is_empty() {
        return Ok(None);
    }

    let base_x_df: Vec<f64> = tenors.iter().map(|(t, _)| *t).collect();
    let base_y_df: Vec<f64> = tenors.iter().map(|(_, df)| *df).collect();
    let base_x_zero = base_x_df.clone();
    let base_y_zero: Vec<f64> = tenors
        .iter()
        .map(|(t, df)| -df.ln() / *t)
        .collect::<Vec<_>>();

    let needs_origin = base_x_df[0] > 0.0;

    let with_df_origin = || {
        if !needs_origin {
            return (base_x_df.clone(), base_y_df.clone(), false, false);
        }
        let mut x = base_x_df.clone();
        let mut y = base_y_df.clone();
        x.insert(0, 0.0);
        y.insert(0, 1.0);
        (x, y, true, false)
    };

    let with_zero_origin = || {
        if !needs_origin {
            return (base_x_zero.clone(), base_y_zero.clone(), false, false);
        }
        let mut x = base_x_zero.clone();
        let mut y = base_y_zero.clone();
        x.insert(0, 0.0);
        y.insert(0, base_y_zero[0]);
        (x, y, true, true)
    };

    let (interpolator, quantity, has_origin_node, origin_linked_to_first_input) = match settings
        .method
    {
        YieldCurveInterpolationMethod::LogLinearDiscount => {
            let (x, y, has_origin, origin_link) = with_df_origin();
            (
                AnyInterpolator::LogLinear(LogLinearInterpolator::new(
                    x,
                    y,
                    settings.extrapolation,
                )?),
                InterpolatedQuantity::DiscountFactor,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::LinearZeroRate => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            (
                AnyInterpolator::Linear(LinearInterpolator::new(x, y, settings.extrapolation)?),
                InterpolatedQuantity::ZeroRate,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::MonotoneConvex => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            (
                AnyInterpolator::MonotoneConvex(MonotoneConvexInterpolator::new(
                    x,
                    y,
                    settings.extrapolation,
                )?),
                InterpolatedQuantity::ZeroRate,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::TensionSpline { tension } => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            (
                AnyInterpolator::TensionSpline(TensionSplineInterpolator::new(
                    x,
                    y,
                    tension,
                    settings.extrapolation,
                )?),
                InterpolatedQuantity::ZeroRate,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::HermiteMonotone => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            (
                AnyInterpolator::HermiteMonotone(HermiteMonotoneInterpolator::new(
                    x,
                    y,
                    settings.extrapolation,
                )?),
                InterpolatedQuantity::ZeroRate,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::LogCubicMonotone => {
            let (x, y, has_origin, origin_link) = with_df_origin();
            (
                AnyInterpolator::LogCubicMonotone(LogCubicMonotoneInterpolator::new(
                    x,
                    y,
                    settings.extrapolation,
                )?),
                InterpolatedQuantity::DiscountFactor,
                has_origin,
                origin_link,
            )
        }
        YieldCurveInterpolationMethod::NelsonSiegel => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            if x.len() < 3 {
                (
                    AnyInterpolator::Linear(LinearInterpolator::new(x, y, settings.extrapolation)?),
                    InterpolatedQuantity::ZeroRate,
                    has_origin,
                    origin_link,
                )
            } else {
                (
                    AnyInterpolator::NelsonSiegel(NelsonSiegelInterpolator::fit(
                        x,
                        y,
                        settings.extrapolation,
                    )?),
                    InterpolatedQuantity::ZeroRate,
                    has_origin,
                    origin_link,
                )
            }
        }
        YieldCurveInterpolationMethod::NelsonSiegelSvensson => {
            let (x, y, has_origin, origin_link) = with_zero_origin();
            if x.len() < 4 {
                (
                    AnyInterpolator::NelsonSiegel(NelsonSiegelInterpolator::fit(
                        x,
                        y,
                        settings.extrapolation,
                    )?),
                    InterpolatedQuantity::ZeroRate,
                    has_origin,
                    origin_link,
                )
            } else {
                (
                    AnyInterpolator::NelsonSiegelSvensson(NelsonSiegelSvenssonInterpolator::fit(
                        x,
                        y,
                        settings.extrapolation,
                    )?),
                    InterpolatedQuantity::ZeroRate,
                    has_origin,
                    origin_link,
                )
            }
        }
        YieldCurveInterpolationMethod::SmithWilson { ufr, alpha } => {
            if base_x_df.len() < 2 {
                (
                    AnyInterpolator::LogLinear(LogLinearInterpolator::new(
                        base_x_df.clone(),
                        base_y_df.clone(),
                        settings.extrapolation,
                    )?),
                    InterpolatedQuantity::DiscountFactor,
                    false,
                    false,
                )
            } else {
                (
                    AnyInterpolator::SmithWilson(SmithWilsonInterpolator::new(
                        base_x_df.clone(),
                        base_y_df.clone(),
                        ufr,
                        alpha,
                        settings.extrapolation,
                    )?),
                    InterpolatedQuantity::DiscountFactor,
                    false,
                    false,
                )
            }
        }
    };

    Ok(Some(InterpolationState {
        interpolator,
        quantity,
        has_origin_node,
        origin_linked_to_first_input,
    }))
}

fn map_jacobian_from_interpolator(
    jacobian: Vec<f64>,
    state: &InterpolationState,
    n_inputs: usize,
) -> Vec<f64> {
    if !state.has_origin_node {
        return jacobian;
    }

    if state.origin_linked_to_first_input {
        let mut out = vec![0.0; n_inputs];
        if !jacobian.is_empty() {
            out[0] += jacobian[0];
        }
        for i in 0..n_inputs {
            if i + 1 < jacobian.len() {
                out[i] += jacobian[i + 1];
            }
        }
        out
    } else {
        jacobian.into_iter().skip(1).take(n_inputs).collect()
    }
}

fn discount_factor_one_point(t1: f64, df1: f64, t: f64, extrapolation: ExtrapolationMode) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    let z1 = -df1.ln() / t1;
    match extrapolation {
        ExtrapolationMode::Flat => {
            if t <= t1 {
                (-z1 * t).exp()
            } else {
                df1
            }
        }
        ExtrapolationMode::Linear | ExtrapolationMode::Error => (-z1 * t).exp(),
    }
}

fn discount_factor_from_points(
    points: &[(f64, f64)],
    t: f64,
    settings: YieldCurveInterpolationSettings,
) -> Result<f64, InterpolationError> {
    if t <= 0.0 || points.is_empty() {
        return Ok(1.0);
    }
    let curve = YieldCurve::new_with_settings(points.to_vec(), settings)?;
    curve.try_discount_factor(t)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn settings(method: YieldCurveInterpolationMethod) -> YieldCurveInterpolationSettings {
        YieldCurveInterpolationSettings {
            method,
            extrapolation: ExtrapolationMode::Linear,
        }
    }

    #[test]
    fn default_curve_keeps_log_linear_discount_behavior() {
        let yc = YieldCurve::new(vec![(1.0, 0.95), (2.0, 0.90)]);
        let mid = yc.discount_factor(1.5);
        let expected = (0.5 * 0.95_f64.ln() + 0.5 * 0.90_f64.ln()).exp();
        assert_relative_eq!(mid, expected, epsilon = 1e-12);
    }

    #[test]
    fn linear_zero_rate_curve_is_flat_for_flat_inputs() {
        let r = 0.03_f64;
        let yc = YieldCurve::new_with_settings(
            vec![
                (1.0, (-r).exp()),
                (5.0, (-r * 5.0).exp()),
                (10.0, (-r * 10.0).exp()),
            ],
            settings(YieldCurveInterpolationMethod::LinearZeroRate),
        )
        .unwrap();

        for t in [0.5, 2.0, 7.0, 10.0] {
            assert_relative_eq!(yc.zero_rate(t), r, epsilon = 1e-10);
        }
    }

    #[test]
    fn curve_methods_produce_positive_discount_factors() {
        let points = vec![
            (0.5, 0.99),
            (1.0, 0.975),
            (2.0, 0.945),
            (5.0, 0.86),
            (10.0, 0.72),
        ];

        let methods = [
            YieldCurveInterpolationMethod::LogLinearDiscount,
            YieldCurveInterpolationMethod::LinearZeroRate,
            YieldCurveInterpolationMethod::MonotoneConvex,
            YieldCurveInterpolationMethod::TensionSpline { tension: 0.25 },
            YieldCurveInterpolationMethod::HermiteMonotone,
            YieldCurveInterpolationMethod::LogCubicMonotone,
            YieldCurveInterpolationMethod::NelsonSiegel,
            YieldCurveInterpolationMethod::NelsonSiegelSvensson,
            YieldCurveInterpolationMethod::SmithWilson {
                ufr: 0.032,
                alpha: 0.12,
            },
        ];

        for method in methods {
            let yc = YieldCurve::new_with_settings(points.clone(), settings(method)).unwrap();
            for t in [0.25, 0.75, 1.5, 3.0, 8.0, 15.0] {
                let df = yc.discount_factor(t);
                assert!(df > 0.0 && df <= 1.5, "method={method:?}, t={t}, df={df}");
            }
        }
    }

    #[test]
    fn zero_rate_jacobian_matches_finite_difference() {
        let points = vec![(1.0, 0.98), (2.0, 0.95), (5.0, 0.87), (10.0, 0.74)];
        let s = settings(YieldCurveInterpolationMethod::HermiteMonotone);
        let yc = YieldCurve::new_with_settings(points.clone(), s).unwrap();

        let t = 3.5;
        let j = yc.zero_rate_jacobian(t).unwrap();
        let eps = 1.0e-6;

        for i in 0..points.len() {
            let mut up = points.clone();
            up[i].1 += eps;
            let r_up = YieldCurve::new_with_settings(up, s).unwrap().zero_rate(t);

            let mut dn = points.clone();
            dn[i].1 -= eps;
            let r_dn = YieldCurve::new_with_settings(dn, s).unwrap().zero_rate(t);

            let fd = (r_up - r_dn) / (2.0 * eps);
            assert_relative_eq!(j[i], fd, epsilon = 2.0e-4);
        }
    }

    #[test]
    fn extrapolation_error_mode_surfaces_error_in_try_methods() {
        let settings = YieldCurveInterpolationSettings {
            method: YieldCurveInterpolationMethod::LinearZeroRate,
            extrapolation: ExtrapolationMode::Error,
        };
        let yc = YieldCurve::new_with_settings(vec![(1.0, 0.98), (2.0, 0.95)], settings).unwrap();

        assert!(yc.try_discount_factor(3.0).is_err());
        assert!(yc.discount_factor(3.0).is_nan());
    }
}
