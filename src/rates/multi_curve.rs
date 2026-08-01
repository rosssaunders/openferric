//! Module `rates::multi_curve`.
//!
//! Implements multi curve workflows with concrete routines such as `dual_curve_bootstrap`, `price_irs_multi_curve`.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `MultiCurveEnvironment` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
/// Multi-curve framework for post-2008 interest rate modeling.
///
/// Separates discounting (OIS) from forwarding (IBOR/SOFR tenor curves).
/// Implements dual-curve bootstrap and tenor basis modeling.
///
/// References:
/// - Henrard, "Interest Rate Modelling in the Multi-Curve Framework" (2014)
/// - Ametrano, Bianchetti, "Everything You Always Wanted to Know About
///   Multiple Interest Rate Curve Bootstrapping" (2013)
use crate::rates::yield_curve::{YieldCurve, solve_monotone_root};

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
        self.forward_curves.push((tenor_name.to_string(), curve));
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
        // Par condition under OIS discounting:
        //   par_rate * sum_i DF_ois(t_i) * dt = sum_i f(t_{i-1}, t_i) * DF_ois(t_i) * dt
        // with the simple forward f(t1, t2) = (DF_fwd(t1)/DF_fwd(t2) - 1) / (t2 - t1).
        //
        // The pillar DF_fwd(T) is solved with a 1-D root-finder so the swap
        // reprices exactly: intermediate coupon DFs between the last pillar and
        // T interpolate against the candidate pillar value.
        let n_periods = (tenor * frequency as f64).round() as usize;
        if n_periods == 0 {
            continue;
        }
        let t_n = n_periods as f64 * dt;

        // OIS annuity is independent of the candidate forward curve.
        let mut fixed_pv = 0.0;
        let mut ois_dfs = Vec::with_capacity(n_periods);
        for i in 1..=n_periods {
            let ois_df = ois_curve.discount_factor(i as f64 * dt);
            ois_dfs.push(ois_df);
            fixed_pv += par_rate * ois_df * dt;
        }

        // Residual of the par equation as a function of the candidate pillar
        // DF. One candidate curve is built per evaluation (not per coupon).
        // Raising the pillar DF lowers every projected forward, so the
        // residual is monotone in the candidate value.
        let residual = |df_n: f64| {
            let mut candidate = fwd_points.clone();
            candidate.push((t_n, df_n));
            let fwd_curve = YieldCurve::new(candidate);

            let mut float_pv = 0.0;
            let mut fwd_df_prev = 1.0;
            for (i, ois_df) in ois_dfs.iter().enumerate() {
                let t_i = (i + 1) as f64 * dt;
                let fwd_df_curr = fwd_curve.discount_factor(t_i);
                let fwd_rate = (fwd_df_prev / fwd_df_curr - 1.0) / dt;
                float_pv += fwd_rate * ois_df * dt;
                fwd_df_prev = fwd_df_curr;
            }
            Ok(fixed_pv - float_pv)
        };

        // Quotes whose residual cannot be bracketed (`Ok(None)`) or whose
        // candidate curve fails to build (`Err`) are skipped: storing a
        // bracket endpoint would put a nonsense pillar on the curve.
        if let Ok(Some(fwd_df_n)) = solve_monotone_root(residual, 1.0e-10, 4.0)
            && fwd_df_n > 0.0
            && fwd_df_n.is_finite()
        {
            fwd_points.push((t_n, fwd_df_n));
        }
    }

    YieldCurve::new(fwd_points)
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
    use approx::assert_relative_eq;

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
        assert_relative_eq!(df, (-0.03_f64).exp(), epsilon = 1.0e-12);
    }

    #[test]
    fn multi_curve_forward_rate() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035); // 3M IBOR at 3.5%
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m);

        let fwd = env.forward_rate("3M", 1.0, 1.25).unwrap();
        let expected = (0.035_f64 * 0.25).exp_m1() / 0.25;
        assert_relative_eq!(fwd, expected, epsilon = 1.0e-12);
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
        let expected = (0.037_f64 * 0.5).exp_m1() / 0.5 - (0.035_f64 * 0.5).exp_m1() / 0.5;
        assert_relative_eq!(basis, expected, epsilon = 1.0e-12);
    }

    #[test]
    fn dual_curve_bootstrap_produces_valid_curve() {
        let ois = make_flat_curve(0.03);
        let swap_rates = vec![(1.0, 0.035), (2.0, 0.036), (3.0, 0.037), (5.0, 0.038)];
        let fwd_curve = dual_curve_bootstrap(&swap_rates, &ois, 4);
        assert!(!fwd_curve.tenors.is_empty());
        // All DFs should be positive and decreasing
        for &(_, df) in &fwd_curve.tenors {
            assert!(df > 0.0);
            assert!(df <= 1.0);
        }
    }

    #[test]
    fn dual_curve_bootstrap_reprices_par_swaps() {
        // Repricing each input swap with the same conventions the bootstrap
        // assumes (simple forwards (DF1/DF2 - 1)/dt projected off the forward
        // curve, OIS discounting) must give NPV ~ 0 at the input par rate.
        let ois = make_flat_curve(0.03);
        let swap_rates = vec![(1.0, 0.035), (2.0, 0.036), (3.0, 0.037), (5.0, 0.038)];
        let fwd_curve = dual_curve_bootstrap(&swap_rates, &ois, 4);

        let mut env = MultiCurveEnvironment::new(make_flat_curve(0.03));
        env.add_forward_curve("3M", fwd_curve);

        for &(tenor, rate) in &swap_rates {
            let pv = price_irs_multi_curve(&env, "3M", 1.0, rate, tenor, 4).unwrap();
            assert!(
                pv.abs() < 1.0e-10,
                "swap tenor={tenor} should reprice at par, pv={pv}"
            );
        }
    }

    #[test]
    fn irs_at_par_rate_has_near_zero_value() {
        let ois = make_flat_curve(0.03);
        let fwd_3m = make_flat_curve(0.035);
        let mut env = MultiCurveEnvironment::new(ois);
        env.add_forward_curve("3M", fwd_3m.clone());

        // On a flat continuous 3.5% projection curve, each quarterly simple
        // forward is exactly exp(0.035/4)-1 over a quarter.
        let par = (0.035_f64 * 0.25).exp_m1() / 0.25;
        let pv = price_irs_multi_curve(&env, "3M", 1_000_000.0, par, 5.0, 4).unwrap();
        assert_relative_eq!(pv, 0.0, epsilon = 1.0e-9);
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
