//! Module `rates::yield_curve`.
//!
//! Implements yield curve abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `YieldCurve`, `YieldCurveBuilder` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
/// Discount-factor term structure keyed by maturity tenor in years.
///
/// # Examples
/// ```rust
/// use openferric::rates::YieldCurve;
///
/// let yc = YieldCurve::new(vec![(0.5, 0.98), (1.0, 0.95)]);
/// assert!(yc.discount_factor(0.75) < 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct YieldCurve {
    /// Curve nodes as `(tenor, discount_factor)`.
    pub tenors: Vec<(f64, f64)>,
}

impl YieldCurve {
    /// Creates a curve from unsorted discount-factor nodes.
    pub fn new(mut tenors: Vec<(f64, f64)>) -> Self {
        tenors.retain(|(t, df)| *t > 0.0 && *df > 0.0);
        tenors.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { tenors }
    }

    /// Returns discount factor at tenor `t` using log-linear interpolation.
    pub fn discount_factor(&self, t: f64) -> f64 {
        discount_factor_from_points(&self.tenors, t)
    }

    /// Returns continuously-compounded zero rate at tenor `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        -self.discount_factor(t).ln() / t
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`.
    ///
    /// # Examples
    /// ```rust
    /// use openferric::rates::YieldCurve;
    ///
    /// let yc = YieldCurve::new(vec![(1.0, 0.95), (2.0, 0.90)]);
    /// let fwd = yc.forward_rate(1.0, 2.0);
    /// assert!(fwd > 0.0);
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
    pub fn from_deposits(deposits: &[(f64, f64)]) -> YieldCurve {
        let points = deposits
            .iter()
            .filter(|(tenor, _)| *tenor > 0.0)
            .map(|(tenor, rate)| (*tenor, 1.0 / (1.0 + rate * tenor)))
            .collect();
        YieldCurve::new(points)
    }

    /// Bootstraps discount factors from par swap rates `(tenor, fixed_rate)`.
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
