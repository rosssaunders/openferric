//! Yield-curve primitives and simple bootstrap helpers.
//!
//! Curve interpolation is log-linear in discount factors, which corresponds
//! to linear interpolation of continuously-compounded zero rates.

/// Discount-factor term structure keyed by maturity tenor in years.
///
/// Tenors are interpreted as year fractions from valuation date.
#[derive(Debug, Clone, PartialEq)]
pub struct YieldCurve {
    /// Curve nodes as `(tenor, discount_factor)`.
    pub tenors: Vec<(f64, f64)>,
}

impl YieldCurve {
    /// Creates a curve from unsorted discount-factor nodes.
    ///
    /// Invalid nodes (`tenor <= 0` or `discount_factor <= 0`) are dropped.
    ///
    /// # Examples
    /// ```
    /// use openferric::rates::YieldCurve;
    ///
    /// let curve = YieldCurve::new(vec![(2.0, 0.90), (1.0, 0.95)]);
    /// assert_eq!(curve.tenors[0].0, 1.0);
    /// ```
    pub fn new(mut tenors: Vec<(f64, f64)>) -> Self {
        tenors.retain(|(t, df)| *t > 0.0 && *df > 0.0);
        tenors.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { tenors }
    }

    /// Returns discount factor at tenor `t` using log-linear interpolation.
    ///
    /// # Numerical notes
    /// - `t <= 0` returns `1.0`.
    /// - For extrapolation beyond the last node, the last segment slope is reused.
    ///
    /// # Examples
    /// ```
    /// use openferric::rates::YieldCurve;
    ///
    /// let curve = YieldCurve::new(vec![(1.0, 0.95), (2.0, 0.90)]);
    /// let df = curve.discount_factor(1.5);
    /// assert!(df < 0.95 && df > 0.90);
    /// ```
    pub fn discount_factor(&self, t: f64) -> f64 {
        discount_factor_from_points(&self.tenors, t)
    }

    /// Returns continuously-compounded zero rate at tenor `t`.
    ///
    /// Uses `z(t) = -ln(DF(t))/t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        -self.discount_factor(t).ln() / t
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`.
    ///
    /// Uses `f(t1,t2) = ln(DF(t1)/DF(t2)) / (t2-t1)`.
    ///
    /// # Panics
    /// Panics when `t2 <= t1`.
    ///
    /// # Examples
    /// ```
    /// use openferric::rates::YieldCurve;
    ///
    /// let curve = YieldCurve::new(vec![(1.0, 0.95), (2.0, 0.90)]);
    /// let fwd = curve.forward_rate(1.0, 2.0);
    /// assert!(fwd.is_finite());
    /// ```
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        assert!(t2 > t1, "t2 must be greater than t1");
        (self.discount_factor(t1) / self.discount_factor(t2)).ln() / (t2 - t1)
    }
}

/// Helper constructors for simple curve bootstrapping.
pub struct YieldCurveBuilder;

impl YieldCurveBuilder {
    /// Builds a curve from simple deposit rates `(tenor, rate)`.
    ///
    /// Uses simple-compounding conversion `DF = 1 / (1 + rT)`.
    pub fn from_deposits(deposits: &[(f64, f64)]) -> YieldCurve {
        let points = deposits
            .iter()
            .filter(|(tenor, _)| *tenor > 0.0)
            .map(|(tenor, rate)| (*tenor, 1.0 / (1.0 + rate * tenor)))
            .collect();
        YieldCurve::new(points)
    }

    /// Bootstraps discount factors from par swap rates `(tenor, fixed_rate)`.
    ///
    /// This assumes standard fixed-leg accrual on a regular payment grid and
    /// solves discount factors sequentially (Hull, swap bootstrapping chapter).
    ///
    /// # Examples
    /// ```
    /// use openferric::rates::YieldCurveBuilder;
    ///
    /// let swaps = vec![(1.0, 0.03), (2.0, 0.032), (3.0, 0.034)];
    /// let curve = YieldCurveBuilder::from_swap_rates(&swaps, 2);
    /// assert_eq!(curve.tenors.len(), 3);
    /// ```
    pub fn from_swap_rates(swap_rates: &[(f64, f64)], frequency: usize) -> YieldCurve {
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
                pv_coupons += coupon * discount_factor_from_points(&points, ti);
            }

            let mut df = (1.0 - pv_coupons) / (1.0 + coupon);
            if df <= 0.0 {
                df = 1.0e-12;
            }
            points.push((tenor, df));
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        YieldCurve::new(points)
    }
}

fn discount_factor_from_points(points: &[(f64, f64)], t: f64) -> f64 {
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
