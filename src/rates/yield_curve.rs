use crate::math::Tape;

/// Discount-factor term structure keyed by maturity tenor in years.
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

    /// Returns discount factor and sensitivities with respect to input discount-factor nodes.
    ///
    /// The gradient vector is aligned to `self.tenors` order.
    pub fn discount_factor_with_sensitivities_aad(&self, t: f64) -> (f64, Vec<f64>) {
        if t <= 0.0 || self.tenors.is_empty() {
            return (1.0, vec![0.0; self.tenors.len()]);
        }

        let mut tape = Tape::with_capacity(self.tenors.len() * 8 + 16);
        let inputs: Vec<_> = self.tenors.iter().map(|(_, df)| tape.input(*df)).collect();

        let out = discount_factor_from_points_aad(&mut tape, &self.tenors, &inputs, t);
        let gradient = tape.gradient(out, &inputs);
        (tape.value(out), gradient)
    }

    /// Returns continuously-compounded zero rate at tenor `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        -self.discount_factor(t).ln() / t
    }

    /// Returns zero rate and sensitivities with respect to input discount-factor nodes.
    ///
    /// The gradient vector is aligned to `self.tenors` order.
    pub fn zero_rate_with_sensitivities_aad(&self, t: f64) -> (f64, Vec<f64>) {
        if t <= 0.0 || self.tenors.is_empty() {
            return (0.0, vec![0.0; self.tenors.len()]);
        }

        let mut tape = Tape::with_capacity(self.tenors.len() * 8 + 24);
        let inputs: Vec<_> = self.tenors.iter().map(|(_, df)| tape.input(*df)).collect();
        let df = discount_factor_from_points_aad(&mut tape, &self.tenors, &inputs, t);
        let ln_df = tape.ln(df);
        let scaled = tape.mul_const(ln_df, -1.0 / t);
        let gradient = tape.gradient(scaled, &inputs);
        (tape.value(scaled), gradient)
    }

    /// Returns continuously-compounded forward rate between `t1` and `t2`.
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

fn discount_factor_from_points_aad(
    tape: &mut Tape,
    points: &[(f64, f64)],
    inputs: &[crate::math::VarId],
    t: f64,
) -> crate::math::VarId {
    if t <= 0.0 || points.is_empty() {
        return tape.constant(1.0);
    }

    let first = points[0];
    if t <= first.0 {
        let one = tape.constant(1.0);
        return log_linear_df_aad(tape, 0.0, one, first.0, inputs[0], t);
    }

    for i in 1..points.len() {
        if t <= points[i].0 {
            return log_linear_df_aad(
                tape,
                points[i - 1].0,
                inputs[i - 1],
                points[i].0,
                inputs[i],
                t,
            );
        }
    }

    if points.len() == 1 {
        let t1 = points[0].0;
        let df1 = inputs[0];
        let ln_df1 = tape.ln(df1);
        let scale = tape.mul_const(ln_df1, -t / t1);
        return tape.exp(scale);
    }

    let i = points.len() - 2;
    log_linear_df_aad(
        tape,
        points[i].0,
        inputs[i],
        points[i + 1].0,
        inputs[i + 1],
        t,
    )
}

fn log_linear_df(t1: f64, df1: f64, t2: f64, df2: f64, t: f64) -> f64 {
    if (t2 - t1).abs() <= f64::EPSILON {
        return df2;
    }
    let w = (t - t1) / (t2 - t1);
    (df1.ln() + w * (df2.ln() - df1.ln())).exp()
}

fn log_linear_df_aad(
    tape: &mut Tape,
    t1: f64,
    df1: crate::math::VarId,
    t2: f64,
    df2: crate::math::VarId,
    t: f64,
) -> crate::math::VarId {
    if (t2 - t1).abs() <= f64::EPSILON {
        return df2;
    }

    let w = (t - t1) / (t2 - t1);
    let ln_df1 = tape.ln(df1);
    let ln_df2 = tape.ln(df2);
    let dln = tape.sub(ln_df2, ln_df1);
    let wdln = tape.mul_const(dln, w);
    let ln_df = tape.add(ln_df1, wdln);
    tape.exp(ln_df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn discount_factor_aad_matches_finite_differences() {
        let curve = YieldCurve::new(vec![(0.5, 0.99), (1.0, 0.965), (2.0, 0.91)]);
        let t = 1.35;
        let (df, grad) = curve.discount_factor_with_sensitivities_aad(t);

        let bump = 1e-6;
        for i in 0..curve.tenors.len() {
            let mut up = curve.clone();
            up.tenors[i].1 += bump;
            let mut dn = curve.clone();
            dn.tenors[i].1 -= bump;
            let fd = (up.discount_factor(t) - dn.discount_factor(t)) / (2.0 * bump);
            assert_relative_eq!(grad[i], fd, epsilon = 5e-8);
        }

        assert_relative_eq!(df, curve.discount_factor(t), epsilon = 1e-14);
    }
}
