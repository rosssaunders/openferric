//! Module `engines::tree::bermudan_swaption`.
//!
//! Implements bermudan swaption abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13, Cox-Ross-Rubinstein (1979), and backward-induction recursions around Eq. (13.10).
//!
//! Key types and purpose: `BermudanSwaptionEngine` define the core data contracts for this module.
//!
//! Numerical considerations: convergence is first- to second-order in time-step count depending on tree parameterization; deep ITM/OTM nodes may need larger depth.
//!
//! When to use: use trees for early-exercise intuition and lattice diagnostics; use analytic formulas for plain vanillas and Monte Carlo/PDE for richer dynamics.
use super::hull_white_lattice::HullWhiteLattice;
use crate::models::HullWhite;
use crate::rates::{Swaption, YieldCurve};

/// Trinomial-tree Bermudan swaption engine under one-factor Hull-White.
///
/// Each exercise starts a new swap of `swap_tenor`, not a co-terminal swap.
/// The centered-OU lattice fits the input discount curve at every grid date.
#[derive(Debug, Clone)]
pub struct BermudanSwaptionEngine {
    /// Hull-White model parameters.
    pub hw_model: HullWhite,
    /// Number of lattice time steps.
    pub steps: usize,
}

impl BermudanSwaptionEngine {
    /// Creates a Bermudan swaption tree engine.
    pub fn new(hw_model: HullWhite, steps: usize) -> Self {
        Self { hw_model, steps }
    }

    /// Prices a Bermudan swaption with the supplied exercise dates.
    pub fn price(&self, swaption: &Swaption, exercise_dates: &[f64], curve: &YieldCurve) -> f64 {
        if self.steps == 0
            || swaption.notional <= 0.0
            || swaption.strike < 0.0
            || swaption.swap_tenor <= 0.0
            || exercise_dates.is_empty()
            || !self.hw_model.a.is_finite()
            || self.hw_model.a < 0.0
            || !self.hw_model.sigma.is_finite()
            || self.hw_model.sigma < 0.0
            || [swaption.notional, swaption.strike, swaption.swap_tenor]
                .iter()
                .any(|value| !value.is_finite())
            || exercise_dates
                .iter()
                .any(|time| !time.is_finite() || *time < 0.0)
        {
            return f64::NAN;
        }

        let horizon = exercise_dates
            .iter()
            .copied()
            .filter(|t| *t >= 0.0 && t.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if !horizon.is_finite() || horizon < 0.0 {
            return f64::NAN;
        }
        if horizon == 0.0 {
            let cache = SliceBondCache::new(swaption, &self.hw_model, curve, 0.0);
            return cache.exercise_value(swaption, HullWhite::instantaneous_forward(curve, 0.0));
        }

        let dt = horizon / self.steps as f64;
        let model = &self.hw_model;
        let Ok(lattice) = HullWhiteLattice::new(model, curve, horizon, self.steps) else {
            return f64::NAN;
        };

        let exercise_flags = map_exercise_steps(exercise_dates, horizon, self.steps);

        // Two reusable buffers ping-ponged across backward-induction steps;
        // slice `i` occupies the first `2*i + 1` entries.
        let max_width = 2 * self.steps + 1;
        let mut values = vec![0.0_f64; max_width];
        let mut scratch = vec![0.0_f64; max_width];

        if exercise_flags[self.steps] {
            let cache = SliceBondCache::new(swaption, model, curve, horizon);
            for j in -(self.steps as isize)..=(self.steps as isize) {
                let idx = (j + self.steps as isize) as usize;
                let rate = lattice.short_rate(self.steps, j);
                values[idx] = cache.exercise_value(swaption, rate);
            }
        }

        for i in (0..self.steps).rev() {
            let t = i as f64 * dt;
            let slice_cache = if exercise_flags[i] {
                Some(SliceBondCache::new(swaption, model, curve, t))
            } else {
                None
            };

            for j in -(i as isize)..=(i as isize) {
                let rate = lattice.short_rate(i, j);
                let continuation = lattice.continuation(i, j, &values);

                let idx = (j + i as isize) as usize;
                scratch[idx] = match &slice_cache {
                    Some(cache) => continuation.max(cache.exercise_value(swaption, rate)),
                    None => continuation,
                };
            }

            std::mem::swap(&mut values, &mut scratch);
        }

        values[0]
    }
}

fn map_exercise_steps(dates: &[f64], horizon: f64, steps: usize) -> Vec<bool> {
    let mut flags = vec![false; steps + 1];
    for &t in dates {
        if !t.is_finite() || t < 0.0 {
            continue;
        }
        let idx = ((t / horizon) * steps as f64).round() as usize;
        flags[idx.min(steps)] = true;
    }
    flags
}

/// Per-time-slice cache of the `(t, T)`-dependent affine bond terms
/// `P(t, T, r) = A(t, T) * exp(-B(t, T) * r)` for every fixed-leg coupon date
/// of the underlying swap, so each tree node only pays one `exp` per cashflow.
struct SliceBondCache {
    /// `(accrual, A, B)` per fixed-leg coupon date.
    coupons: Vec<(f64, f64, f64)>,
    /// `(A, B)` for the swap end date.
    end: (f64, f64),
}

impl SliceBondCache {
    fn new(swaption: &Swaption, model: &HullWhite, curve: &YieldCurve, exercise_time: f64) -> Self {
        let end = exercise_time + swaption.swap_tenor;
        let mut prev = exercise_time;
        let mut coupons = Vec::new();

        loop {
            let next = (prev + 1.0).min(end);
            if next <= prev {
                break;
            }
            let delta = next - prev;
            let (a_t, b_t) = Self::bond_terms(model, curve, exercise_time, next);
            coupons.push((delta, a_t, b_t));

            if next >= end - 1.0e-12 {
                break;
            }
            prev = next;
        }

        Self {
            coupons,
            end: Self::bond_terms(model, curve, exercise_time, end),
        }
    }

    /// Splits the affine Hull-White bond price into `(A(t, T), B(t, T))`.
    ///
    /// `A` is recovered as the model bond price at `r = 0`; `B` replicates the
    /// model's `(1 - exp(-a tau)) / a` term.
    fn bond_terms(model: &HullWhite, curve: &YieldCurve, t: f64, maturity: f64) -> (f64, f64) {
        let a_t = model.bond_price(t, maturity, 0.0, curve);
        let tau = maturity - t;
        let b_t = if tau <= 0.0 {
            0.0
        } else if model.a.abs() <= 1.0e-12 {
            tau
        } else {
            (1.0 - (-model.a * tau).exp()) / model.a
        };
        (a_t, b_t)
    }

    fn exercise_value(&self, swaption: &Swaption, short_rate: f64) -> f64 {
        let mut annuity = 0.0;
        for &(delta, a_t, b_t) in &self.coupons {
            annuity += delta * a_t * (-b_t * short_rate).exp();
        }
        if annuity <= 0.0 {
            return 0.0;
        }

        let df_end = self.end.0 * (-self.end.1 * short_rate).exp();
        let float_leg = swaption.notional * (1.0 - df_end);
        let fixed_leg = swaption.notional * swaption.strike * annuity;
        let swap_value = if swaption.is_payer {
            float_leg - fixed_leg
        } else {
            fixed_leg - float_leg
        };

        swap_value.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bermudan_swaption_has_early_exercise_premium_over_black_european() {
        let flat_rate = 0.05;
        let curve = YieldCurve::new(
            (1..=120)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-flat_rate * t).exp())
                })
                .collect(),
        );

        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.05,
            option_expiry: 5.0,
            swap_tenor: 5.0,
            is_payer: true,
        };
        let exercise_dates = (1..=20).map(|i| i as f64 * 0.25).collect::<Vec<_>>();

        let hw_model = HullWhite::new(0.05, 0.01);
        let engine = BermudanSwaptionEngine::new(hw_model, 300);

        let bermudan = engine.price(&swaption, &exercise_dates, &curve);
        let european_black = swaption.price(&curve, 0.01);
        let european_tree = engine.price(&swaption, &[5.0], &curve);

        assert!(bermudan.is_finite());
        assert!(bermudan > 0.0);
        assert!(
            bermudan > european_black,
            "bermudan={} black_european={}",
            bermudan,
            european_black
        );
        assert!(
            bermudan >= european_tree - 1.0e-8,
            "bermudan={} tree_european={}",
            bermudan,
            european_tree
        );
    }

    #[test]
    fn zero_vol_bermudan_matches_discounted_deterministic_swap_value() {
        // With sigma=0 on a flat curve, the short-rate path is deterministic.
        // The payer swap is ITM at every exercise date, and identical forward
        // swap cashflows make the earliest exercise optimal.  This provides a
        // closed-form oracle for the complete exercise/backward-induction path.
        let flat_rate = 0.05_f64;
        let curve = YieldCurve::new(
            (0..=80)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-flat_rate * t).exp())
                })
                .collect(),
        );
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.04,
            option_expiry: 3.0,
            swap_tenor: 5.0,
            is_payer: true,
        };
        let exercise_dates = [1.0, 2.0, 3.0];
        let engine = BermudanSwaptionEngine::new(HullWhite::new(0.05, 0.0), 300);

        let actual = engine.price(&swaption, &exercise_dates, &curve);
        let annuity_at_exercise: f64 = (1..=5).map(|year| (-flat_rate * year as f64).exp()).sum();
        let swap_value_at_exercise = swaption.notional
            * (1.0
                - (-flat_rate * swaption.swap_tenor).exp()
                - swaption.strike * annuity_at_exercise);
        let expected = (-flat_rate * exercise_dates[0]).exp() * swap_value_at_exercise;
        // The model recovers its short-rate drift from finite differences of
        // the input curve, so allow the measured 300-step discretization error
        // against the continuum cashflow value (well below one millionth of a bp).
        let tolerance = 5.0e-10 * expected.abs().max(1.0);

        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}
