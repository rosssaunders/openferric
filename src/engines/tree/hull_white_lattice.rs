//! Curve-fitted Hull-White lattice in the centered Ornstein-Uhlenbeck state.
//!
//! Exact one-step OU moments determine positive trinomial probabilities.
//! Arrow-Debreu state prices fit each grid-date discount factor, avoiding
//! differentiation of discontinuous forwards at interpolated curve pillars.
//! Analytic conditional bonds use the continuous-model short rate; the
//! separately fitted discount shift removes finite-grid zero-bond bias.

use crate::models::HullWhite;
use crate::rates::YieldCurve;

pub(crate) struct HullWhiteLattice {
    dt: f64,
    spacing: f64,
    persistence: f64,
    discount_shifts: Vec<f64>,
    short_rate_shifts: Vec<f64>,
}

impl HullWhiteLattice {
    pub(crate) fn new(
        model: &HullWhite,
        curve: &YieldCurve,
        horizon: f64,
        steps: usize,
    ) -> Result<Self, String> {
        if steps == 0
            || !horizon.is_finite()
            || horizon <= 0.0
            || !model.a.is_finite()
            || model.a < 0.0
            || !model.sigma.is_finite()
            || model.sigma < 0.0
        {
            return Err("Hull-White lattice requires finite non-negative model parameters and positive horizon/steps".to_string());
        }
        let dt = horizon / steps as f64;
        let variance_time = if model.a == 0.0 {
            dt
        } else {
            -(-2.0 * model.a * dt).exp_m1() / (2.0 * model.a)
        };
        let spacing = model.sigma * (3.0 * variance_time).sqrt();
        let mut lattice = Self {
            dt,
            spacing,
            persistence: (-model.a * dt).exp(),
            discount_shifts: Vec::with_capacity(steps),
            short_rate_shifts: Vec::with_capacity(steps + 1),
        };
        for step in 0..=steps {
            let time = step as f64 * dt;
            let mean_time = if model.a == 0.0 {
                time
            } else {
                -(-model.a * time).exp_m1() / model.a
            };
            let shift = HullWhite::instantaneous_forward(curve, time)
                + 0.5 * (model.sigma * mean_time).powi(2);
            if !shift.is_finite() {
                return Err("Hull-White curve produces a non-finite short rate".to_string());
            }
            lattice.short_rate_shifts.push(shift);
        }
        let mut state_prices = vec![1.0];
        for step in 0..steps {
            let unshifted_discount: f64 = state_prices
                .iter()
                .enumerate()
                .map(|(index, state_price)| {
                    let state = index as isize - step as isize;
                    state_price * (-(state as f64 * spacing) * dt).exp()
                })
                .sum();
            let target_discount = curve.discount_factor((step + 1) as f64 * dt);
            let shift = (unshifted_discount / target_discount).ln() / dt;
            if !shift.is_finite() || target_discount <= 0.0 {
                return Err(
                    "Hull-White curve cannot be fitted to finite positive discount factors"
                        .to_string(),
                );
            }
            lattice.discount_shifts.push(shift);
            let mut next_prices = vec![0.0; 2 * step + 3];
            for (index, state_price) in state_prices.iter().enumerate() {
                let state = index as isize - step as isize;
                let (center, probabilities) = lattice.branches(state);
                let discounted = state_price * lattice.discount(step, state);
                for (branch, probability) in probabilities.into_iter().enumerate() {
                    let target = (center + step as isize + branch as isize) as usize;
                    next_prices[target] += discounted * probability;
                }
            }
            state_prices = next_prices;
        }
        Ok(lattice)
    }

    #[inline]
    fn branches(&self, state: isize) -> (isize, [f64; 3]) {
        if self.spacing == 0.0 {
            return (0, [0.0, 1.0, 0.0]);
        }
        let mean = state as f64 * self.persistence;
        let center = mean.round() as isize;
        let residual = mean - center as f64;
        let second_moment = 1.0 / 3.0 + residual * residual;
        (
            center,
            [
                0.5 * (second_moment - residual),
                1.0 - second_moment,
                0.5 * (second_moment + residual),
            ],
        )
    }

    #[inline]
    pub(crate) fn short_rate(&self, step: usize, state: isize) -> f64 {
        self.short_rate_shifts[step] + state as f64 * self.spacing
    }

    #[inline]
    fn discount(&self, step: usize, state: isize) -> f64 {
        (-(self.discount_shifts[step] + state as f64 * self.spacing) * self.dt).exp()
    }

    #[inline]
    pub(crate) fn continuation(&self, step: usize, state: isize, next_values: &[f64]) -> f64 {
        let (center, probabilities) = self.branches(state);
        let first = (center + step as isize) as usize;
        self.discount(step, state)
            * (probabilities[0] * next_values[first]
                + probabilities[1] * next_values[first + 1]
                + probabilities[2] * next_values[first + 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn every_grid_date_zero_bond_reprices_nonflat_curve() {
        let curve = YieldCurve::new(vec![(0.25, 1.01), (0.5, 0.99), (1.0, 0.94), (2.0, 0.80)]);
        for mean_reversion in [0.0, 0.1, 5.0] {
            for volatility in [0.0, 0.01, 0.05] {
                let lattice = HullWhiteLattice::new(
                    &HullWhite::new(mean_reversion, volatility),
                    &curve,
                    2.0,
                    80,
                )
                .unwrap();
                for maturity_step in 1..=80 {
                    let mut values = vec![1.0; 2 * maturity_step + 1];
                    for step in (0..maturity_step).rev() {
                        values = (-(step as isize)..=step as isize)
                            .map(|state| lattice.continuation(step, state, &values))
                            .collect();
                    }
                    assert_relative_eq!(
                        values[0],
                        curve.discount_factor(maturity_step as f64 * 0.025),
                        epsilon = 2.0e-14
                    );
                }
            }
        }
    }

    #[test]
    fn transitions_match_exact_ou_moments_without_probability_clipping() {
        let curve = YieldCurve::new(vec![(2.0, 0.9)]);
        for mean_reversion in [0.0, 0.1, 5.0] {
            let lattice =
                HullWhiteLattice::new(&HullWhite::new(mean_reversion, 0.03), &curve, 2.0, 80)
                    .unwrap();
            for state in -80..=80 {
                let (center, probabilities) = lattice.branches(state);
                let mean = state as f64 * lattice.persistence;
                let mut actual_mean = 0.0;
                let mut variance = 0.0;
                for (branch, probability) in probabilities.into_iter().enumerate() {
                    assert!((0.0..=1.0).contains(&probability));
                    let target = (center + branch as isize - 1) as f64;
                    actual_mean += probability * target;
                    variance += probability * (target - mean).powi(2);
                }
                assert_relative_eq!(actual_mean, mean, epsilon = 3.0e-14);
                assert_relative_eq!(variance, 1.0 / 3.0, epsilon = 2.0e-15);
                assert_relative_eq!(probabilities.iter().sum::<f64>(), 1.0, epsilon = 2.0e-16);
            }
        }
    }
}
