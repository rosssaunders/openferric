//! Module `credit::cds_index`.
//!
//! Implements cds index workflows with concrete routines such as `first_to_default_spread_copula`.
//!
//! References: Hull (11th ed.) Ch. 24-25, O'Kane (2008) Ch. 3, representative cashflow identities as in Eq. (24.7) and Eq. (25.5).
//!
//! Key types and purpose: `CdsIndex`, `NthToDefaultBasket` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use these routines for CDS/tranche and survival-curve workflows; consider structural credit models when capital-structure dynamics are required explicitly.
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::rates::YieldCurve;

use super::{GaussianCopula, SurvivalCurve, cds::Cds};

/// CDS index as a weighted basket of single-name CDS constituents.
#[derive(Debug, Clone, PartialEq)]
pub struct CdsIndex {
    pub constituents: Vec<Cds>,
    /// Weights associated with each constituent. They are normalized internally.
    pub weights: Vec<f64>,
}

impl CdsIndex {
    /// Weighted average NPV from protection-buyer perspective.
    pub fn npv(&self, discount_curve: &YieldCurve, survival_curves: &[SurvivalCurve]) -> f64 {
        if !self.is_valid(survival_curves.len()) {
            return 0.0;
        }

        self.normalized_weights()
            .iter()
            .zip(self.constituents.iter())
            .zip(survival_curves.iter())
            .map(|((w, cds), curve)| w * cds.npv(discount_curve, curve))
            .sum()
    }

    /// Weighted average fair spread.
    pub fn fair_spread(
        &self,
        discount_curve: &YieldCurve,
        survival_curves: &[SurvivalCurve],
    ) -> f64 {
        if !self.is_valid(survival_curves.len()) {
            return 0.0;
        }

        self.normalized_weights()
            .iter()
            .zip(self.constituents.iter())
            .zip(survival_curves.iter())
            .map(|((w, cds), curve)| w * cds.fair_spread(discount_curve, curve))
            .sum()
    }

    fn normalized_weights(&self) -> Vec<f64> {
        let sum = self
            .weights
            .iter()
            .copied()
            .filter(|w| *w > 0.0)
            .sum::<f64>();
        if sum <= 1.0e-14 {
            return vec![0.0; self.weights.len()];
        }
        self.weights
            .iter()
            .map(|w| if *w > 0.0 { *w / sum } else { 0.0 })
            .collect()
    }

    fn is_valid(&self, n_curves: usize) -> bool {
        !self.constituents.is_empty()
            && self.constituents.len() == self.weights.len()
            && self.constituents.len() == n_curves
            && self.weights.iter().any(|w| *w > 0.0)
    }
}

/// Nth-to-default basket CDS with common maturity and recovery.
#[derive(Debug, Clone, PartialEq)]
pub struct NthToDefaultBasket {
    pub n: usize,
    pub notional: f64,
    pub maturity: f64,
    pub recovery_rate: f64,
    pub payment_freq: usize,
}

impl NthToDefaultBasket {
    /// Fair spread from a discrete-time default-count distribution approximation.
    pub fn fair_spread(
        &self,
        discount_curve: &YieldCurve,
        survival_curves: &[SurvivalCurve],
    ) -> f64 {
        if !self.is_valid(survival_curves.len()) {
            return 0.0;
        }

        let payment_times = payment_times(self.maturity, self.payment_freq);
        if payment_times.is_empty() {
            return 0.0;
        }

        let mut premium_annuity = 0.0;
        let mut protection_term = 0.0;

        let mut p_triggered_prev = 0.0;
        let mut t_prev = 0.0;

        for &t in &payment_times {
            let dt = t - t_prev;
            let p_triggered_t = prob_at_least_n_defaults(self.n, t, survival_curves);
            let p_alive_mid = (1.0 - 0.5 * (p_triggered_prev + p_triggered_t)).clamp(0.0, 1.0);

            premium_annuity += dt * discount_curve.discount_factor(t) * p_alive_mid;

            let delta_trigger = (p_triggered_t - p_triggered_prev).max(0.0);
            let t_mid = 0.5 * (t_prev + t);
            protection_term += discount_curve.discount_factor(t_mid) * delta_trigger;

            p_triggered_prev = p_triggered_t;
            t_prev = t;
        }

        if premium_annuity <= 1.0e-14 {
            0.0
        } else {
            (1.0 - self.recovery_rate).max(0.0) * protection_term / premium_annuity
        }
    }

    /// Protection-buyer NPV for a running spread.
    pub fn npv(
        &self,
        running_spread: f64,
        discount_curve: &YieldCurve,
        survival_curves: &[SurvivalCurve],
    ) -> f64 {
        let fair = self.fair_spread(discount_curve, survival_curves);
        if fair <= 0.0 {
            return 0.0;
        }

        let unit_annuity = self.premium_leg_annuity(discount_curve, survival_curves);
        self.notional * (fair - running_spread.max(0.0)) * unit_annuity
    }

    fn premium_leg_annuity(
        &self,
        discount_curve: &YieldCurve,
        survival_curves: &[SurvivalCurve],
    ) -> f64 {
        let payment_times = payment_times(self.maturity, self.payment_freq);
        let mut annuity = 0.0;

        let mut p_triggered_prev = 0.0;
        let mut t_prev = 0.0;
        for &t in &payment_times {
            let dt = t - t_prev;
            let p_triggered_t = prob_at_least_n_defaults(self.n, t, survival_curves);
            let p_alive_mid = (1.0 - 0.5 * (p_triggered_prev + p_triggered_t)).clamp(0.0, 1.0);
            annuity += dt * discount_curve.discount_factor(t) * p_alive_mid;
            p_triggered_prev = p_triggered_t;
            t_prev = t;
        }

        annuity
    }

    fn is_valid(&self, num_names: usize) -> bool {
        self.notional > 0.0
            && self.n > 0
            && self.n <= num_names
            && self.maturity > 0.0
            && (0.0..1.0).contains(&self.recovery_rate)
            && self.payment_freq > 0
    }
}

/// First-to-default fair spread from Gaussian copula simulation.
pub fn first_to_default_spread_copula(
    notional: f64,
    maturity: f64,
    recovery_rate: f64,
    payment_freq: usize,
    discount_curve: &YieldCurve,
    survival_curves: &[SurvivalCurve],
    copula: &GaussianCopula,
    num_paths: usize,
    seed: u64,
) -> f64 {
    if notional <= 0.0
        || maturity <= 0.0
        || payment_freq == 0
        || survival_curves.is_empty()
        || !(0.0..1.0).contains(&recovery_rate)
        || num_paths == 0
    {
        return 0.0;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let times = payment_times(maturity, payment_freq);
    if times.is_empty() {
        return 0.0;
    }

    let mut protection_pv_sum = 0.0;
    let mut premium_annuity_sum = 0.0;

    for _ in 0..num_paths {
        let sim = copula.simulate(survival_curves, &mut rng);
        let tau = sim
            .default_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        if tau.is_finite() && tau <= maturity {
            protection_pv_sum += discount_curve.discount_factor(tau);
        }

        let mut annuity_path = 0.0;
        let mut t_prev = 0.0;
        for &t in &times {
            if tau.is_finite() {
                if tau <= t_prev {
                    break;
                }
                if tau < t {
                    annuity_path += (tau - t_prev).max(0.0) * discount_curve.discount_factor(tau);
                    break;
                }
            }

            annuity_path += (t - t_prev) * discount_curve.discount_factor(t);
            t_prev = t;
        }

        premium_annuity_sum += annuity_path;
    }

    let protection_leg = notional * (1.0 - recovery_rate) * protection_pv_sum / num_paths as f64;
    let premium_leg_unit = notional * premium_annuity_sum / num_paths as f64;

    if premium_leg_unit <= 1.0e-14 {
        0.0
    } else {
        protection_leg / premium_leg_unit
    }
}

fn prob_at_least_n_defaults(n: usize, t: f64, survival_curves: &[SurvivalCurve]) -> f64 {
    let mut dist = vec![0.0_f64; survival_curves.len() + 1];
    dist[0] = 1.0;

    for curve in survival_curves {
        let q = (1.0 - curve.survival_prob(t)).clamp(0.0, 1.0);
        for k in (1..dist.len()).rev() {
            dist[k] = dist[k] * (1.0 - q) + dist[k - 1] * q;
        }
        dist[0] *= 1.0 - q;
    }

    dist.iter().skip(n).sum::<f64>().clamp(0.0, 1.0)
}

fn payment_times(maturity: f64, payment_freq: usize) -> Vec<f64> {
    if maturity <= 0.0 || payment_freq == 0 {
        return vec![];
    }

    let dt = 1.0 / payment_freq as f64;
    let mut t = 0.0;
    let mut times = Vec::new();
    while t + dt < maturity - 1.0e-12 {
        t += dt;
        times.push(t);
    }
    times.push(maturity);
    times
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn homogeneous_index_spread_matches_single_name() {
        let discount_curve = YieldCurve::new(
            (1..=80)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-0.03 * t).exp())
                })
                .collect(),
        );

        let single_curve = SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.02]);
        let single = Cds {
            notional: 1.0,
            spread: 0.01,
            maturity: 5.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        };

        let index = CdsIndex {
            constituents: vec![single.clone(); 5],
            weights: vec![1.0; 5],
        };

        let curves = vec![single_curve.clone(); 5];
        let index_spread = index.fair_spread(&discount_curve, &curves);
        let single_spread = single.fair_spread(&discount_curve, &single_curve);

        assert_relative_eq!(index_spread, single_spread, epsilon = 1.0e-12);
    }
}
