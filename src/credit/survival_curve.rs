use crate::rates::YieldCurve;

use super::cds::Cds;

/// Survival-probability term structure keyed by maturity tenor in years.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvivalCurve {
    /// Curve nodes as `(tenor, survival_probability)`.
    pub tenors: Vec<(f64, f64)>,
}

impl SurvivalCurve {
    /// Creates a survival curve from unsorted nodes.
    pub fn new(mut tenors: Vec<(f64, f64)>) -> Self {
        tenors.retain(|(t, p)| *t > 0.0 && *p > 0.0);
        tenors.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Keep nodes monotone non-increasing in probability.
        let mut cleaned: Vec<(f64, f64)> = Vec::with_capacity(tenors.len());
        let mut prev_prob = 1.0_f64;
        for (t, p) in tenors {
            let prob = p.clamp(1.0e-12, 1.0).min(prev_prob);
            if let Some(last) = cleaned.last_mut()
                && (last.0 - t).abs() <= 1.0e-12
            {
                last.1 = prob;
                prev_prob = prob;
                continue;
            }
            cleaned.push((t, prob));
            prev_prob = prob;
        }

        Self { tenors: cleaned }
    }

    /// Builds a survival curve from piecewise-constant hazard rates.
    pub fn from_piecewise_hazard(tenors: &[f64], hazards: &[f64]) -> Self {
        assert_eq!(
            tenors.len(),
            hazards.len(),
            "tenors and hazards must have same length"
        );

        let mut points = Vec::with_capacity(tenors.len());
        let mut cum_hazard = 0.0;
        let mut prev_t = 0.0;
        for (&t, &h) in tenors.iter().zip(hazards.iter()) {
            if t <= prev_t {
                continue;
            }
            cum_hazard += h.max(0.0) * (t - prev_t);
            points.push((t, (-cum_hazard).exp()));
            prev_t = t;
        }

        Self::new(points)
    }

    /// Bootstraps a survival curve from par CDS spreads `(maturity, spread)`.
    pub fn bootstrap_from_cds_spreads(
        cds_spreads: &[(f64, f64)],
        recovery_rate: f64,
        payment_freq: usize,
        discount_curve: &YieldCurve,
    ) -> Self {
        if payment_freq == 0 || !(0.0..1.0).contains(&recovery_rate) {
            return Self::new(vec![]);
        }

        let mut quotes = cds_spreads
            .iter()
            .copied()
            .filter(|(maturity, spread)| *maturity > 0.0 && *spread >= 0.0)
            .collect::<Vec<_>>();
        quotes.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut pillar_times = Vec::with_capacity(quotes.len());
        let mut hazards = Vec::with_capacity(quotes.len());

        for (maturity, spread) in quotes {
            let eval = |lambda: f64| {
                let mut t = pillar_times.clone();
                t.push(maturity);
                let mut h = hazards.clone();
                h.push(lambda.max(1.0e-10));

                let survival_curve = Self::from_piecewise_hazard(&t, &h);
                let cds = Cds {
                    notional: 1.0,
                    spread,
                    maturity,
                    recovery_rate,
                    payment_freq,
                };
                cds.npv(discount_curve, &survival_curve)
            };

            let mut lo = 1.0e-8;
            let mut hi = 1.0;
            let mut f_lo = eval(lo);
            let mut f_hi = eval(hi);

            let mut grow_iter = 0;
            while f_lo.signum() == f_hi.signum() && grow_iter < 40 {
                hi *= 2.0;
                f_hi = eval(hi);
                grow_iter += 1;
            }

            let solved = if f_lo.signum() != f_hi.signum() {
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    let f_mid = eval(mid);

                    if f_mid.abs() < 1.0e-12 || (hi - lo) < 1.0e-10 {
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

            pillar_times.push(maturity);
            hazards.push(solved.max(1.0e-10));
        }

        Self::from_piecewise_hazard(&pillar_times, &hazards)
    }

    /// Returns survival probability at tenor `t` using log-linear interpolation.
    pub fn survival_prob(&self, t: f64) -> f64 {
        survival_prob_from_points(&self.tenors, t)
    }

    /// Returns piecewise-constant hazard rate at tenor `t`.
    pub fn hazard_rate(&self, t: f64) -> f64 {
        if self.tenors.is_empty() {
            return 0.0;
        }

        let first = self.tenors[0];
        if t <= first.0 {
            return hazard_between(0.0, 1.0, first.0, first.1);
        }

        for window in self.tenors.windows(2) {
            let left = window[0];
            let right = window[1];
            if t <= right.0 {
                return hazard_between(left.0, left.1, right.0, right.1);
            }
        }

        if self.tenors.len() == 1 {
            return hazard_between(0.0, 1.0, first.0, first.1);
        }

        let left = self.tenors[self.tenors.len() - 2];
        let right = self.tenors[self.tenors.len() - 1];
        hazard_between(left.0, left.1, right.0, right.1)
    }

    /// Returns default probability in `(t1, t2]`.
    pub fn default_prob(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 {
            return 0.0;
        }
        (self.survival_prob(t1) - self.survival_prob(t2)).clamp(0.0, 1.0)
    }
}

fn survival_prob_from_points(points: &[(f64, f64)], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    if points.is_empty() {
        return 1.0;
    }

    let first = points[0];
    if t <= first.0 {
        return log_linear_prob(0.0, 1.0, first.0, first.1, t);
    }

    for window in points.windows(2) {
        let left = window[0];
        let right = window[1];
        if t <= right.0 {
            return log_linear_prob(left.0, left.1, right.0, right.1, t);
        }
    }

    if points.len() == 1 {
        let (t1, p1) = points[0];
        let h = hazard_between(0.0, 1.0, t1, p1);
        return (-h * t).exp();
    }

    let (t_last, p_last) = points[points.len() - 1];
    let left = points[points.len() - 2];
    let h_tail = hazard_between(left.0, left.1, t_last, p_last);
    p_last * (-h_tail * (t - t_last)).exp()
}

fn hazard_between(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    if t2 <= t1 {
        return 0.0;
    }
    let h = -(p2.ln() - p1.ln()) / (t2 - t1);
    if h.is_finite() { h.max(0.0) } else { 0.0 }
}

fn log_linear_prob(t1: f64, p1: f64, t2: f64, p2: f64, t: f64) -> f64 {
    if (t2 - t1).abs() <= f64::EPSILON {
        return p2;
    }
    let w = (t - t1) / (t2 - t1);
    (p1.ln() + w * (p2.ln() - p1.ln())).exp()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn survival_default_hazard_are_consistent() {
        let curve = SurvivalCurve::new(vec![(1.0, 0.98), (3.0, 0.92), (5.0, 0.86)]);

        assert_relative_eq!(curve.survival_prob(0.0), 1.0, epsilon = 1e-12);
        assert!(curve.hazard_rate(2.0) > 0.0);
        assert_relative_eq!(
            curve.default_prob(1.0, 3.0),
            curve.survival_prob(1.0) - curve.survival_prob(3.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn bootstrap_reprices_input_cds_spreads() {
        let r = 0.02;
        let discount_curve = YieldCurve::new(
            (1..=15)
                .map(|t| {
                    let tt = t as f64;
                    (tt, (-r * tt).exp())
                })
                .collect(),
        );

        let recovery = 0.4;
        let hazard = 0.025;
        let pillars = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        let true_curve =
            SurvivalCurve::from_piecewise_hazard(&pillars, &vec![hazard; pillars.len()]);

        let maturities = vec![1.0, 3.0, 5.0, 7.0];
        let quotes = maturities
            .iter()
            .map(|&maturity| {
                let cds = Cds {
                    notional: 1.0,
                    spread: 0.0,
                    maturity,
                    recovery_rate: recovery,
                    payment_freq: 4,
                };
                (maturity, cds.fair_spread(&discount_curve, &true_curve))
            })
            .collect::<Vec<_>>();

        let bootstrapped =
            SurvivalCurve::bootstrap_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

        for (maturity, spread) in quotes {
            let cds = Cds {
                notional: 1.0,
                spread: 0.0,
                maturity,
                recovery_rate: recovery,
                payment_freq: 4,
            };
            let repriced = cds.fair_spread(&discount_curve, &bootstrapped);
            assert_relative_eq!(repriced, spread, epsilon = 1e-8);
        }

        assert_relative_eq!(
            bootstrapped.survival_prob(6.0),
            true_curve.survival_prob(6.0),
            epsilon = 2e-3
        );
    }
}
