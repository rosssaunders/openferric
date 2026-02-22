use crate::math::{
    AnyInterpolator, ExtrapolationMode, InterpolationError, InterpolationMethod, Interpolator,
    build_interpolator,
};

/// Discount-factor term structure keyed by maturity tenor in years.
#[derive(Debug, Clone, PartialEq)]
pub struct YieldCurve {
    /// Curve nodes as `(tenor, discount_factor)`.
    pub tenors: Vec<(f64, f64)>,
    interpolation_method: InterpolationMethod,
    extrapolation: ExtrapolationMode,
    interpolator: Option<AnyInterpolator>,
}

impl YieldCurve {
    /// Creates a curve from unsorted discount-factor nodes using the historical
    /// default interpolation: log-linear discount factors with linear extrapolation.
    pub fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self::try_new_with_interpolation(
            tenors,
            InterpolationMethod::LogLinearDiscount,
            ExtrapolationMode::Linear,
        )
        .expect("default yield curve construction should not fail")
    }

    /// Creates a curve with a user-selected interpolation method and extrapolation mode.
    pub fn try_new_with_interpolation(
        mut tenors: Vec<(f64, f64)>,
        interpolation_method: InterpolationMethod,
        extrapolation: ExtrapolationMode,
    ) -> Result<Self, InterpolationError> {
        tenors.retain(|(t, df)| *t > 0.0 && *df > 0.0 && t.is_finite() && df.is_finite());
        tenors.sort_by(|a, b| a.0.total_cmp(&b.0));
        if tenors.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(InterpolationError::InvalidInput(
                "tenors must be strictly increasing",
            ));
        }

        let interpolator = if tenors.len() >= 2 {
            Some(build_interpolator(
                tenors.clone(),
                &interpolation_method,
                extrapolation,
            )?)
        } else {
            None
        };

        Ok(Self {
            tenors,
            interpolation_method,
            extrapolation,
            interpolator,
        })
    }

    /// Convenience constructor that panics if the interpolator setup is invalid.
    pub fn with_interpolation(
        tenors: Vec<(f64, f64)>,
        interpolation_method: InterpolationMethod,
        extrapolation: ExtrapolationMode,
    ) -> Self {
        Self::try_new_with_interpolation(tenors, interpolation_method, extrapolation)
            .expect("invalid yield-curve interpolation configuration")
    }

    /// Selected interpolation method.
    pub fn interpolation_method(&self) -> &InterpolationMethod {
        &self.interpolation_method
    }

    /// Selected extrapolation mode.
    pub fn extrapolation_mode(&self) -> ExtrapolationMode {
        self.extrapolation
    }

    /// Returns discount factor at tenor `t`.
    ///
    /// If the configured extrapolation mode is `None` and `t` is outside the
    /// interpolation range, this method will panic. Use [`Self::try_discount_factor`]
    /// for fallible handling.
    pub fn discount_factor(&self, t: f64) -> f64 {
        self.try_discount_factor(t)
            .expect("discount factor query failed; use try_discount_factor for error handling")
    }

    /// Fallible discount-factor query.
    pub fn try_discount_factor(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(1.0);
        }
        if let Some(interpolator) = &self.interpolator {
            return interpolator.discount_factor(t);
        }
        Ok(discount_factor_from_points_default(&self.tenors, t))
    }

    /// Returns continuously-compounded zero rate at tenor `t`.
    ///
    /// Panics on interpolation errors. Use [`Self::try_zero_rate`] for fallible handling.
    pub fn zero_rate(&self, t: f64) -> f64 {
        self.try_zero_rate(t)
            .expect("zero rate query failed; use try_zero_rate for error handling")
    }

    /// Fallible zero-rate query.
    pub fn try_zero_rate(&self, t: f64) -> Result<f64, InterpolationError> {
        if t <= 0.0 {
            return Ok(0.0);
        }
        let df = self.try_discount_factor(t)?;
        Ok(-df.ln() / t)
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`.
    ///
    /// Panics on interpolation errors. Use [`Self::try_forward_rate`] for fallible handling.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        self.try_forward_rate(t1, t2)
            .expect("forward rate query failed; use try_forward_rate for error handling")
    }

    /// Fallible forward-rate query.
    pub fn try_forward_rate(&self, t1: f64, t2: f64) -> Result<f64, InterpolationError> {
        if t2 <= t1 {
            return Err(InterpolationError::InvalidInput(
                "t2 must be greater than t1",
            ));
        }
        let df1 = self.try_discount_factor(t1)?;
        let df2 = self.try_discount_factor(t2)?;
        Ok((df1 / df2).ln() / (t2 - t1))
    }

    /// Jacobian of zero rate at tenor `t` with respect to input discount nodes.
    ///
    /// Returns a vector with `self.tenors.len()` entries.
    pub fn jacobian_zero_rate(&self, t: f64) -> Result<Vec<f64>, InterpolationError> {
        if t <= 0.0 {
            return Ok(vec![0.0; self.tenors.len()]);
        }
        if let Some(interpolator) = &self.interpolator {
            return interpolator.jacobian_zero_rate(t);
        }

        // Degenerate fallback for 0/1 node curves.
        if self.tenors.is_empty() {
            return Ok(Vec::new());
        }
        let (tn, dfn) = self.tenors[0];
        let j = -1.0 / (tn * dfn);
        Ok(vec![j])
    }
}

/// Helper constructors for simple curve bootstrapping.
pub struct YieldCurveBuilder;

impl YieldCurveBuilder {
    /// Builds a curve from simple deposit rates `(tenor, rate)` with default interpolation.
    pub fn from_deposits(deposits: &[(f64, f64)]) -> YieldCurve {
        Self::from_deposits_with_interpolation(
            deposits,
            InterpolationMethod::LogLinearDiscount,
            ExtrapolationMode::Linear,
        )
        .expect("default deposit curve construction should not fail")
    }

    /// Builds a curve from simple deposit rates `(tenor, rate)` with interpolation config.
    pub fn from_deposits_with_interpolation(
        deposits: &[(f64, f64)],
        interpolation_method: InterpolationMethod,
        extrapolation: ExtrapolationMode,
    ) -> Result<YieldCurve, InterpolationError> {
        let points: Vec<(f64, f64)> = deposits
            .iter()
            .filter(|(tenor, _)| *tenor > 0.0)
            .map(|(tenor, rate)| (*tenor, 1.0 / (1.0 + rate * tenor)))
            .collect();
        YieldCurve::try_new_with_interpolation(points, interpolation_method, extrapolation)
    }

    /// Bootstraps discount factors from par swap rates `(tenor, fixed_rate)`
    /// with default interpolation.
    pub fn from_swap_rates(swap_rates: &[(f64, f64)], frequency: usize) -> YieldCurve {
        Self::from_swap_rates_with_interpolation(
            swap_rates,
            frequency,
            InterpolationMethod::LogLinearDiscount,
            ExtrapolationMode::Linear,
        )
        .expect("default swap curve bootstrap should not fail")
    }

    /// Bootstraps discount factors from par swap rates `(tenor, fixed_rate)`
    /// using the chosen interpolation and extrapolation settings.
    pub fn from_swap_rates_with_interpolation(
        swap_rates: &[(f64, f64)],
        frequency: usize,
        interpolation_method: InterpolationMethod,
        extrapolation: ExtrapolationMode,
    ) -> Result<YieldCurve, InterpolationError> {
        if frequency == 0 {
            return Err(InterpolationError::InvalidInput("frequency must be > 0"));
        }

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
                pv_coupons += coupon
                    * bootstrap_discount_from_points(
                        &points,
                        ti,
                        &interpolation_method,
                        extrapolation,
                    )?;
            }

            let mut df = (1.0 - pv_coupons) / (1.0 + coupon);
            if df <= 0.0 {
                df = 1.0e-12;
            }
            points.push((tenor, df));
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        YieldCurve::try_new_with_interpolation(points, interpolation_method, extrapolation)
    }
}

fn bootstrap_discount_from_points(
    points: &[(f64, f64)],
    t: f64,
    method: &InterpolationMethod,
    extrapolation: ExtrapolationMode,
) -> Result<f64, InterpolationError> {
    if t <= 0.0 {
        return Ok(1.0);
    }
    if points.is_empty() {
        return Ok(1.0);
    }
    if points.len() == 1 {
        return Ok(discount_factor_from_points_default(points, t));
    }
    let curve =
        YieldCurve::try_new_with_interpolation(points.to_vec(), method.clone(), extrapolation)?;
    curve.try_discount_factor(t)
}

fn discount_factor_from_points_default(points: &[(f64, f64)], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    if points.is_empty() {
        return 1.0;
    }

    let first = points[0];
    if t <= first.0 {
        return log_linear_df(0.0, 1.0, first.0, first.1, t);
    }

    for window in points.windows(2) {
        let left = window[0];
        let right = window[1];
        if t <= right.0 {
            return log_linear_df(left.0, left.1, right.0, right.1, t);
        }
    }

    if points.len() == 1 {
        let (t1, df1) = points[0];
        let z = -df1.ln() / t1;
        return (-z * t).exp();
    }

    let left = points[points.len() - 2];
    let right = points[points.len() - 1];
    log_linear_df(left.0, left.1, right.0, right.1, t)
}

fn log_linear_df(t1: f64, df1: f64, t2: f64, df2: f64, t: f64) -> f64 {
    if (t2 - t1).abs() <= f64::EPSILON {
        return df2;
    }
    let w = (t - t1) / (t2 - t1);
    (df1.ln() + w * (df2.ln() - df1.ln())).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn default_constructor_keeps_log_linear_behavior() {
        let r = 0.05_f64;
        let curve = YieldCurve::new(vec![
            (0.5, (-r * 0.5).exp()),
            (1.0, (-r * 1.0).exp()),
            (2.0, (-r * 2.0).exp()),
            (5.0, (-r * 5.0).exp()),
        ]);
        for t in [0.25, 0.75, 1.5, 3.0, 7.0] {
            assert_relative_eq!(curve.discount_factor(t), (-r * t).exp(), epsilon = 1.0e-12);
        }
    }

    #[test]
    fn explicit_interpolator_construction_works() {
        let nodes = vec![(1.0, 0.98), (2.0, 0.955), (3.0, 0.93), (5.0, 0.88)];
        let curve = YieldCurve::try_new_with_interpolation(
            nodes,
            InterpolationMethod::HermiteCubic,
            ExtrapolationMode::Linear,
        )
        .unwrap();
        let df = curve.discount_factor(2.5);
        assert!(df > 0.0 && df < 1.0);
    }

    #[test]
    fn extrapolation_none_returns_error_on_try_api() {
        let nodes = vec![(1.0, 0.98), (2.0, 0.95), (3.0, 0.92)];
        let curve = YieldCurve::try_new_with_interpolation(
            nodes,
            InterpolationMethod::LinearZeroRate,
            ExtrapolationMode::None,
        )
        .unwrap();
        assert!(matches!(
            curve.try_discount_factor(5.0),
            Err(InterpolationError::ExtrapolationDisabled { .. })
        ));
    }

    #[test]
    fn jacobian_vector_has_expected_size() {
        let nodes = vec![(1.0, 0.98), (2.0, 0.95), (3.0, 0.92), (5.0, 0.87)];
        let curve = YieldCurve::try_new_with_interpolation(
            nodes.clone(),
            InterpolationMethod::LogLinearDiscount,
            ExtrapolationMode::Linear,
        )
        .unwrap();
        let jac = curve.jacobian_zero_rate(2.4).unwrap();
        assert_eq!(jac.len(), nodes.len());
    }

    #[test]
    fn swap_bootstrap_with_custom_interpolation_builds_valid_curve() {
        let swaps = vec![(1.0, 0.03), (2.0, 0.032), (3.0, 0.034), (5.0, 0.036)];
        let curve = YieldCurveBuilder::from_swap_rates_with_interpolation(
            &swaps,
            2,
            InterpolationMethod::MonotoneConvex,
            ExtrapolationMode::Linear,
        )
        .unwrap();
        assert!(!curve.tenors.is_empty());
        for (_, df) in &curve.tenors {
            assert!(*df > 0.0 && *df <= 1.0);
        }
    }
}
