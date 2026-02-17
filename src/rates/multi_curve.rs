/// Multi-curve framework for post-2008 interest rate modeling.
///
/// Separates discounting (OIS) from forwarding (IBOR/SOFR tenor curves).
/// Implements dual-curve bootstrap and tenor basis modeling.
///
/// References:
/// - Henrard, "Interest Rate Modelling in the Multi-Curve Framework" (2014)
/// - Ametrano, Bianchetti, "Everything You Always Wanted to Know About
///   Multiple Interest Rate Curve Bootstrapping" (2013)

use crate::rates::yield_curve::YieldCurve;

/// Multi-curve environment: one discount curve + multiple forwarding curves.
#[derive(Debug, Clone)]
pub struct MultiCurveEnvironment {
    /// OIS discount curve (e.g., SOFR, €STR).
    pub discount_curve: YieldCurve,
    /// Forward curves keyed by tenor name (e.g., "3M", "6M", "SOFR").
    pub forward_curves: Vec<(String, YieldCurve)>,
}

impl MultiCurveEnvironment {
    pub fn new(discount_curve: YieldCurve) -> Self {
        Self {
            discount_curve,
            forward_curves: Vec::new(),
        }
    }

    /// Add a forwarding curve for a specific tenor.
    pub fn add_forward_curve(&mut self, tenor_name: &str, curve: YieldCurve) {
        self.forward_curves
            .push((tenor_name.to_string(), curve));
    }

    /// Get discount factor from OIS curve.
    pub fn discount_factor(&self, t: f64) -> f64 {
        self.discount_curve.discount_factor(t)
    }

    /// Get forward rate from a specific tenor curve.
    pub fn forward_rate(&self, tenor_name: &str, t1: f64, t2: f64) -> Option<f64> {
        self.forward_curves
            .iter()
            .find(|(name, _)| name == tenor_name)
            .map(|(_, curve)| {
                let df1 = curve.discount_factor(t1);
                let df2 = curve.discount_factor(t2);
                if t2 > t1 && df2 > 0.0 {
                    (df1 / df2 - 1.0) / (t2 - t1)
                } else {
                    0.0
                }
            })
    }

    /// Tenor basis spread between two forwarding curves.
    pub fn tenor_basis(&self, tenor1: &str, tenor2: &str, t1: f64, t2: f64) -> Option<f64> {
        let fwd1 = self.forward_rate(tenor1, t1, t2)?;
        let fwd2 = self.forward_rate(tenor2, t1, t2)?;
        Some(fwd1 - fwd2)
    }
}

/// Dual-curve bootstrap: build forward curve from swap rates using OIS discounting.
///
/// Given par swap rates and an OIS discount curve, bootstraps the forward curve
/// so that swaps are priced at par under OIS discounting.
///
/// # Arguments
/// * `swap_rates` - `(tenor, par_rate)` pairs, sorted by tenor
/// * `ois_curve` - OIS discount curve
/// * `frequency` - Payment frequency per year (e.g., 4 for quarterly)
pub fn dual_curve_bootstrap(
    swap_rates: &[(f64, f64)],
    ois_curve: &YieldCurve,
    frequency: usize,
) -> YieldCurve {
    assert!(frequency > 0);

    let mut sorted = swap_rates.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    let dt = 1.0 / frequency as f64;
    let mut fwd_points: Vec<(f64, f64)> = Vec::new();

    for &(tenor, par_rate) in &sorted {
        // PV of fixed leg = par_rate * sum(DF_ois(t_i) * dt)
        // PV of float leg = sum(f(t_{i-1}, t_i) * DF_ois(t_i) * dt)
        // At par: fixed PV = float PV + (DF_fwd(0) - DF_fwd(T)) * DF_ois(T)/DF_fwd(T)
        //
        // Simplified: solve for DF_fwd(T) such that swap is at par
        let n_periods = (tenor * frequency as f64).round() as usize;
        if n_periods == 0 {
            continue;
        }

        let mut annuity = 0.0;
        let mut float_pv = 0.0;

        for i in 1..n_periods {
            let t_i = i as f64 * dt;
            let ois_df = ois_curve.discount_factor(t_i);
            annuity += ois_df * dt;

            // Use already-bootstrapped forward DFs for intermediate periods
            if let Some(fwd_df_prev) = interpolate_df(&fwd_points, (i - 1) as f64 * dt) {
                if let Some(fwd_df_curr) = interpolate_df(&fwd_points, t_i) {
                    let fwd_rate = (fwd_df_prev / fwd_df_curr - 1.0) / dt;
                    float_pv += fwd_rate * ois_df * dt;
                }
            }
        }

        let t_n = n_periods as f64 * dt;
        let ois_df_n = ois_curve.discount_factor(t_n);
        annuity += ois_df_n * dt;

        // Solve for DF_fwd(T): par_rate * annuity = float_pv + (1 - DF_fwd(T)) * ois_df_n / ... 
        // Simplified: DF_fwd(T) = (1 - par_rate * annuity + float_pv) if float_pv is already computed
        // More precisely, for the last period:
        let fwd_df_prev = if fwd_points.is_empty() {
            1.0
        } else {
            interpolate_df(&fwd_points, (n_periods - 1) as f64 * dt).unwrap_or(1.0)
        };

        // The last forward rate satisfies:
        // par_rate * annuity = float_pv + fwd_rate_last * ois_df_n * dt
        // fwd_rate_last = (fwd_df_prev / fwd_df_n - 1) / dt
        // Solving: fwd_df_n = fwd_df_prev / (1 + (par_rate * annuity - float_pv) / (ois_df_n * dt) * dt)
        let remaining = par_rate * annuity - float_pv;
        let implied_fwd_rate = remaining / (ois_df_n * dt);
        let fwd_df_n = fwd_df_prev / (1.0 + implied_fwd_rate * dt);

        if fwd_df_n > 0.0 && fwd_df_n.is_finite() {
            fwd_points.push((t_n, fwd_df_n));
        }
    }

    YieldCurve::new(fwd_points)
}

fn interpolate_df(points: &[(f64, f64)], t: f64) -> Option<f64> {
    if points.is_empty() || t <= 0.0 {
        return Some(1.0);
    }
    // Simple: use the YieldCurve interpolation
    if points.len() == 1 {
        let (t0, df0) = points[0];
        if t <= t0 {
            return Some((-(-df0.ln() / t0) * t).exp());
        }
    }
    let curve = YieldCurve::new(points.to_vec());
    Some(curve.discount_factor(t))
}

/// Price a vanilla IRS under multi-curve framework.
///
/// Fixed leg discounted with OIS, floating leg uses forward curve + OIS discounting.
pub fn price_irs_multi_curve(
    env: &MultiCurveEnvironment,
    forward_tenor: &str,
    notional: f64,
    fixed_rate: f64,
    tenor: f64,
    frequency: usize,
) -> Option<f64> {
    let dt = 1.0 / frequency as f64;
    let n_periods = (tenor * frequency as f64).round() as usize;

    let mut fixed_pv = 0.0;
    let mut float_pv = 0.0;

    for i in 1..=n_periods {
        let t_prev = (i - 1) as f64 * dt;
        let t_i = i as f64 * dt;
        let ois_df = env.discount_factor(t_i);

        fixed_pv += fixed_rate * notional * dt * ois_df;

        let fwd = env.forward_rate(forward_tenor, t_prev, t_i)?;
        float_pv += fwd * notional * dt * ois_df;
    }

    Some(float_pv - fixed_pv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_curve(rate: f64) -> YieldCurve {
        let points: Vec<(f64, f64)> = (1..=40)
            .map(|i| {
                let t = i as f64 * 0.25;
                (t, (-rate * t).exp())
            })
            .collect();
        YieldCurve::new(points)
    }

    #[test]
    fn multi_curve_discount_uses_ois() {
        let ois = make_flat_curve(0.03);
        let env = MultiCurveEnvironment::new(ois);
        let df = env.discount_factor(1.0);
        assert!((df - (-0.03_f64).exp()).abs() < 0.001);
    }

    #[test]
    fn multi_curve_forward_rate() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035); // 3M IBOR at 3.5%
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m);

        let fwd = env.forward_rate("3M", 1.0, 1.25).unwrap();
        // For flat curve at 3.5%, forward ≈ 3.5%
        assert!((fwd - 0.035).abs() < 0.005);
    }

    #[test]
    fn tenor_basis_is_difference_of_forwards() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035);
        let fwd_6m = make_flat_curve(0.037);
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m);
        env.add_forward_curve("6M", fwd_6m);

        let basis = env.tenor_basis("6M", "3M", 1.0, 1.5).unwrap();
        assert!(basis > 0.0); // 6M > 3M
    }

    #[test]
    fn dual_curve_bootstrap_produces_valid_curve() {
        let ois = make_flat_curve(0.03);
        let swap_rates = vec![
            (1.0, 0.035),
            (2.0, 0.036),
            (3.0, 0.037),
            (5.0, 0.038),
        ];
        let fwd_curve = dual_curve_bootstrap(&swap_rates, &ois, 4);
        assert!(!fwd_curve.tenors.is_empty());
        // All DFs should be positive and decreasing
        for &(_, df) in &fwd_curve.tenors {
            assert!(df > 0.0);
            assert!(df <= 1.0);
        }
    }

    #[test]
    fn irs_at_par_rate_has_near_zero_value() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035);
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m.clone());

        // If we set fixed rate = forward rate, PV should be near zero
        let pv = price_irs_multi_curve(&env, "3M", 1_000_000.0, 0.035, 5.0, 4).unwrap();
        // Won't be exactly zero due to interpolation, but should be small relative to notional
        assert!(pv.abs() < 5000.0); // < 0.5% of notional
    }

    #[test]
    fn irs_receiver_benefits_from_higher_fixed_rate() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035);
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m);

        let pv_low = price_irs_multi_curve(&env, "3M", 1_000_000.0, 0.03, 5.0, 4).unwrap();
        let pv_high = price_irs_multi_curve(&env, "3M", 1_000_000.0, 0.04, 5.0, 4).unwrap();
        // Float - Fixed: higher fixed rate → lower PV for payer
        assert!(pv_high < pv_low);
    }
}
