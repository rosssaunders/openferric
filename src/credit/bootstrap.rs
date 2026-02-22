//! Credit analytics for Bootstrap.
//!
//! Module openferric::credit::bootstrap provides pricing helpers and model utilities for credit products.

use crate::rates::YieldCurve;

use super::{cds::Cds, survival_curve::SurvivalCurve};

/// Bootstraps a survival curve from CDS par spreads under piecewise-constant hazard rates.
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

    let mut pillar_times = Vec::with_capacity(quotes.len());
    let mut hazards = Vec::with_capacity(quotes.len());

    for (tenor, spread) in quotes {
        let eval_npv = |lambda: f64| {
            let mut t = pillar_times.clone();
            t.push(tenor);
            let mut h = hazards.clone();
            h.push(lambda.max(0.0));

            let survival = SurvivalCurve::from_piecewise_hazard(&t, &h);
            let cds = Cds {
                notional: 1.0,
                spread,
                maturity: tenor,
                recovery_rate,
                payment_freq,
            };
            cds.npv(discount_curve, &survival)
        };

        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        let mut f_lo = eval_npv(lo);
        if f_lo.abs() <= 1.0e-13 {
            pillar_times.push(tenor);
            hazards.push(lo);
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

        pillar_times.push(tenor);
        hazards.push(solved.max(0.0));
    }

    SurvivalCurve::from_piecewise_hazard(&pillar_times, &hazards)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

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
