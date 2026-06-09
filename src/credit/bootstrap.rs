//! Module `credit::bootstrap`.
//!
//! Implements bootstrap workflows with concrete routines such as `bootstrap_survival_curve_from_cds_spreads`.
//!
//! References: Hull (11th ed.) Ch. 24-25, O'Kane (2008) Ch. 3, representative cashflow identities as in Eq. (24.7) and Eq. (25.5).
//!
//! Primary API surface: free functions `bootstrap_survival_curve_from_cds_spreads`.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use these routines for CDS/tranche and survival-curve workflows; consider structural credit models when capital-structure dynamics are required explicitly.
use crate::rates::YieldCurve;

use super::{cds::payment_times, survival_curve::SurvivalCurve};

/// Lambda-invariant per-period data for the periods after the previous pillar.
struct TailPeriod {
    /// Period end (payment) time.
    t: f64,
    /// Period accrual length.
    dt: f64,
    /// Discount factor at the payment time.
    df_pay: f64,
    /// Discount factor at the period midpoint.
    df_mid: f64,
}

/// Bootstraps a survival curve from CDS par spreads under piecewise-constant hazard rates.
///
/// For each pillar the premium/protection leg contributions of periods ending at
/// or before the previous pillar are invariant in the new hazard, so they are
/// accumulated once and only the tail periods are re-evaluated inside the
/// bisection. A single working curve is reused across iterations by updating
/// the survival probability of its last node in place.
pub fn bootstrap_survival_curve_from_cds_spreads(
    cds_spreads: &[(f64, f64)],
    recovery_rate: f64,
    payment_freq: usize,
    discount_curve: &YieldCurve,
) -> SurvivalCurve {
    if payment_freq == 0 || !(0.0..1.0).contains(&recovery_rate) {
        return SurvivalCurve::new(vec![]);
    }

    let mut quotes = cds_spreads
        .iter()
        .copied()
        .filter(|(tenor, spread)| *tenor > 0.0 && *spread >= 0.0)
        .collect::<Vec<_>>();
    quotes.sort_by(|a, b| a.0.total_cmp(&b.0));
    quotes.dedup_by(|a, b| (a.0 - b.0).abs() <= 1.0e-12);

    let mut pillar_times: Vec<f64> = Vec::with_capacity(quotes.len());
    let mut hazards: Vec<f64> = Vec::with_capacity(quotes.len());
    // Solved curve nodes, mirroring `SurvivalCurve::from_piecewise_hazard` output
    // (including its probability clamping) for the pillars solved so far.
    let mut solved_points: Vec<(f64, f64)> = Vec::with_capacity(quotes.len());
    let mut cum_hazard = 0.0_f64;

    for (tenor, spread) in quotes {
        let prev_t = solved_points.last().map_or(0.0, |p| p.0);
        let prev_prob = solved_points.last().map_or(1.0, |p| p.1);
        if tenor <= prev_t {
            continue;
        }

        // One working curve per pillar: solved nodes plus a mutable trial node.
        let mut work = SurvivalCurve {
            tenors: solved_points.clone(),
        };
        work.tenors.push((tenor, prev_prob));
        let trial_idx = work.tenors.len() - 1;

        let times = payment_times(tenor, payment_freq);

        // Invariant prefix: periods ending at or before the previous pillar.
        let mut prefix_coupon = 0.0;
        let mut prefix_accrual = 0.0;
        let mut prefix_protection = 0.0;
        let mut split = 0usize;
        let mut t_prev = 0.0_f64;
        let mut survival_prev = 1.0_f64;
        while split < times.len() && times[split] <= prev_t {
            let t = times[split];
            let dt = t - t_prev;
            let survival_t = work.survival_prob(t);

            let df_pay = discount_curve.discount_factor(t);
            prefix_coupon += dt * df_pay * survival_t;

            let default_prob = (survival_prev - survival_t).clamp(0.0, 1.0);
            let t_mid = 0.5 * (t_prev + t);
            let df_mid = discount_curve.discount_factor(t_mid);
            prefix_accrual += 0.5 * dt * df_mid * default_prob;
            prefix_protection += df_mid * default_prob;

            t_prev = t;
            survival_prev = survival_t;
            split += 1;
        }

        // Lambda-invariant tail data: accrual lengths and discount factors.
        let tail: Vec<TailPeriod> = times[split..]
            .iter()
            .scan(t_prev, |prev, &t| {
                let period = TailPeriod {
                    t,
                    dt: t - *prev,
                    df_pay: discount_curve.discount_factor(t),
                    df_mid: discount_curve.discount_factor(0.5 * (*prev + t)),
                };
                *prev = t;
                Some(period)
            })
            .collect();
        let tail_start_survival = survival_prev;

        let mut eval_npv = |lambda: f64| {
            // Trial node survival, replicating `from_piecewise_hazard` + `new()`.
            let trial_prob = (-(cum_hazard + lambda.max(0.0) * (tenor - prev_t)))
                .exp()
                .clamp(1.0e-12, 1.0)
                .min(prev_prob);
            work.tenors[trial_idx].1 = trial_prob;

            let mut coupon_annuity = prefix_coupon;
            let mut accrual_annuity = prefix_accrual;
            let mut protection_term = prefix_protection;
            let mut survival_prev = tail_start_survival;
            for period in &tail {
                let survival_t = work.survival_prob(period.t);
                coupon_annuity += period.dt * period.df_pay * survival_t;

                let default_prob = (survival_prev - survival_t).clamp(0.0, 1.0);
                accrual_annuity += 0.5 * period.dt * period.df_mid * default_prob;
                protection_term += period.df_mid * default_prob;

                survival_prev = survival_t;
            }

            (1.0 - recovery_rate) * protection_term - spread * (coupon_annuity + accrual_annuity)
        };

        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        let mut f_lo = eval_npv(lo);
        if f_lo.abs() <= 1.0e-13 {
            pillar_times.push(tenor);
            hazards.push(lo);
            // Zero marginal hazard: survival stays flat at the previous level.
            let solved_prob = (-cum_hazard).exp().clamp(1.0e-12, 1.0).min(prev_prob);
            solved_points.push((tenor, solved_prob));
            continue;
        }

        let mut f_hi = eval_npv(hi);
        let mut grow_iter = 0usize;
        while f_lo.signum() == f_hi.signum() && grow_iter < 50 {
            hi *= 2.0;
            f_hi = eval_npv(hi);
            grow_iter += 1;
        }

        let solved = if f_lo.signum() != f_hi.signum() {
            for _ in 0..120 {
                let mid = 0.5 * (lo + hi);
                let f_mid = eval_npv(mid);
                if f_mid.abs() <= 1.0e-13 || (hi - lo).abs() <= 1.0e-12 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if f_mid.signum() == f_lo.signum() {
                    lo = mid;
                    f_lo = f_mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        } else if f_lo.abs() <= f_hi.abs() {
            lo
        } else {
            hi
        };

        let solved = solved.max(0.0);
        pillar_times.push(tenor);
        hazards.push(solved);
        cum_hazard += solved * (tenor - prev_t);
        let solved_prob = (-cum_hazard).exp().clamp(1.0e-12, 1.0).min(prev_prob);
        solved_points.push((tenor, solved_prob));
    }

    SurvivalCurve::from_piecewise_hazard(&pillar_times, &hazards)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::super::cds::Cds;
    use super::*;

    /// Survival probabilities captured from the pre-optimization implementation
    /// (per-iteration vector clones + full curve rebuild + full repricing from
    /// t = 0). The optimized bootstrap must reproduce them to 1e-12.
    #[test]
    fn bootstrap_matches_pre_optimization_baseline() {
        // Set A: flat 5% discounting, R = 0.4, quarterly premiums.
        let dc_a = YieldCurve::new(
            (1..=48)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-0.05 * t).exp())
                })
                .collect(),
        );
        let quotes_a = vec![
            (1.0, 0.0060),
            (3.0, 0.0080),
            (5.0, 0.0100),
            (7.0, 0.0115),
            (10.0, 0.0130),
        ];
        let curve_a = bootstrap_survival_curve_from_cds_spreads(&quotes_a, 0.4, 4, &dc_a);
        let expected_a = [
            (0.5, 0.9950434357987565),
            (1.0, 0.990111439126194),
            (2.0, 0.9753014923883152),
            (3.0, 0.9607130707371199),
            (4.3, 0.9330774832897206),
            (5.0, 0.9185275272635182),
            (6.0, 0.894004726429613),
            (7.0, 0.8701366340751925),
            (8.5, 0.8317940618740909),
            (10.0, 0.7951410551796285),
            (12.0, 0.7487703876987137),
        ];
        for &(t, expected) in &expected_a {
            assert_relative_eq!(curve_a.survival_prob(t), expected, epsilon = 1.0e-12);
        }

        // Set B: flat 2% discounting, R = 0.25, semiannual premiums.
        let dc_b = YieldCurve::new(
            (1..=16)
                .map(|i| {
                    let t = i as f64 * 0.5;
                    (t, (-0.02 * t).exp())
                })
                .collect(),
        );
        let quotes_b = vec![(0.5, 0.0020), (2.0, 0.0045), (4.0, 0.0070), (6.0, 0.0095)];
        let curve_b = bootstrap_survival_curve_from_cds_spreads(&quotes_b, 0.25, 2, &dc_b);
        let expected_b = [
            (0.25, 0.9993368760082683),
            (0.5, 0.998674191749965),
            (1.0, 0.9951328202563808),
            (2.0, 0.9880877066222792),
            (3.0, 0.9755136686439234),
            (4.0, 0.9630996432130585),
            (5.5, 0.9346968206315072),
            (6.0, 0.9254165880239882),
            (7.0, 0.9071316271948568),
        ];
        for &(t, expected) in &expected_b {
            assert_relative_eq!(curve_b.survival_prob(t), expected, epsilon = 1.0e-12);
        }

        // Set C: single 5y quote, R = 0.5, quarterly premiums, flat 3% discounting.
        let dc_c = YieldCurve::new(
            (1..=20)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-0.03 * t).exp())
                })
                .collect(),
        );
        let quotes_c = vec![(5.0, 0.0200)];
        let curve_c = bootstrap_survival_curve_from_cds_spreads(&quotes_c, 0.5, 4, &dc_c);
        let expected_c = [
            (1.0, 0.9609322650319612),
            (2.5, 0.9051737263978786),
            (5.0, 0.8193394749610216),
            (6.0, 0.7873297375043923),
        ];
        for &(t, expected) in &expected_c {
            assert_relative_eq!(curve_c.survival_prob(t), expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn bootstrapped_curve_reprices_input_term_structure() {
        let discount_rate = 0.05;
        let discount_curve = YieldCurve::new(
            (1..=40)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-discount_rate * t).exp())
                })
                .collect(),
        );

        let recovery = 0.4;
        let quotes = vec![
            (1.0, 0.0060),
            (3.0, 0.0080),
            (5.0, 0.0100),
            (7.0, 0.0115),
            (10.0, 0.0130),
        ];

        let curve =
            bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

        for (tenor, spread) in quotes {
            let cds = Cds {
                notional: 1.0,
                spread,
                maturity: tenor,
                recovery_rate: recovery,
                payment_freq: 4,
            };
            assert_relative_eq!(cds.npv(&discount_curve, &curve), 0.0, epsilon = 1.0e-9);
        }
    }
}
