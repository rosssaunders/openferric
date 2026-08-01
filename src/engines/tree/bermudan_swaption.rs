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
use crate::models::HullWhite;
use crate::rates::{Swaption, YieldCurve};

/// Trinomial-tree Bermudan swaption engine under one-factor Hull-White.
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
            || swaption.strike <= 0.0
            || swaption.swap_tenor <= 0.0
            || exercise_dates.is_empty()
        {
            return f64::NAN;
        }

        let horizon = exercise_dates
            .iter()
            .copied()
            .filter(|t| *t >= 0.0 && t.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if !horizon.is_finite() || horizon <= 0.0 {
            return f64::NAN;
        }

        let dt = horizon / self.steps as f64;
        let r0 = HullWhite::instantaneous_forward(curve, 0.0);

        let mut model = self.hw_model.clone();
        let time_grid = (0..=self.steps).map(|i| i as f64 * dt).collect::<Vec<_>>();
        model.calibrate_theta(curve, &time_grid);

        let dx = if model.sigma.abs() <= 1.0e-14 {
            (3.0 * dt).sqrt() * 1.0e-6
        } else {
            model.sigma * (3.0 * dt).sqrt()
        };

        let exercise_flags = map_exercise_steps(exercise_dates, horizon, self.steps);

        // Two reusable buffers ping-ponged across backward-induction steps;
        // slice `i` occupies the first `2*i + 1` entries.
        let max_width = 2 * self.steps + 1;
        let mut values = vec![0.0_f64; max_width];
        let mut scratch = vec![0.0_f64; max_width];

        if exercise_flags[self.steps] {
            let cache = SliceBondCache::new(swaption, &model, curve, horizon);
            for j in -(self.steps as isize)..=(self.steps as isize) {
                let idx = (j + self.steps as isize) as usize;
                let rate = r0 + j as f64 * dx;
                values[idx] = cache.exercise_value(swaption, rate);
            }
        }

        for i in (0..self.steps).rev() {
            let t = i as f64 * dt;
            // theta(t), and the (t, T)-dependent bond terms used by the
            // exercise value, are constant on a time slice: hoist them out of
            // the per-node loop.
            let theta_t = model.theta_at(t);
            let slice_cache = if exercise_flags[i] {
                Some(SliceBondCache::new(swaption, &model, curve, t))
            } else {
                None
            };

            for j in -(i as isize)..=(i as isize) {
                let rate = r0 + j as f64 * dx;
                // Branching offsets reachable inside the next slice [-(i+1), i+1].
                let k_min = (-(i as isize) - j).max(-1);
                let k_max = ((i as isize) - j).min(1);
                let (k, pu, pm, pd) =
                    trinomial_probs(theta_t, model.a, model.sigma, rate, dt, dx, k_min, k_max);
                let disc = (-rate * dt).exp();

                let next_shift = (i + 1) as isize;
                let up_idx = (j + k + next_shift + 1) as usize;
                let mid_idx = (j + k + next_shift) as usize;
                let down_idx = (j + k + next_shift - 1) as usize;
                let continuation =
                    disc * (pu * values[up_idx] + pm * values[mid_idx] + pd * values[down_idx]);

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

/// Hull-White adaptive trinomial branching (Hull, ch. 32).
///
/// Returns `(k, pu, pm, pd)`: probabilities of branching from node `j` to
/// nodes `j + k + 1`, `j + k`, `j + k - 1`, where `k` is the branching offset
/// (`0` standard, `+1` upward branching at the low edge, `-1` downward
/// branching at the high edge).
///
/// Derivation (first principles): with conditional mean `m1 = mu * dt`
/// (`mu = theta(t) - a * r`) and variance `V = sigma^2 * dt`, write
/// `eta = m1 / dx`, pick the central target `k = round(eta)` (clamped to the
/// branching offsets reachable in the next slice), and let
/// `alpha = eta - k` be the residual drift. Matching
/// `pu - pd = alpha` (mean) and
/// `pu (k+1)^2 + pm k^2 + pd (k-1)^2 = (V + m1^2) / dx^2` (second moment)
/// with `pu + pm + pd = 1` gives
/// `pu = (V/dx^2 + alpha^2 + alpha) / 2`, `pd = (V/dx^2 + alpha^2 - alpha) / 2`,
/// `pm = 1 - V/dx^2 - alpha^2`. With `dx = sigma * sqrt(3 dt)` this is Hull's
/// `p_u = 1/6 + (alpha^2 + alpha)/2` form, and all probabilities are
/// non-negative whenever `|alpha| <= sqrt(2/3)` (guaranteed for
/// `k = round(eta)` with `|eta| <= 1.5`).
#[allow(clippy::too_many_arguments)]
fn trinomial_probs(
    theta_t: f64,
    a: f64,
    sigma: f64,
    rate: f64,
    dt: f64,
    dx: f64,
    k_min: isize,
    k_max: isize,
) -> (isize, f64, f64, f64) {
    if dx <= 0.0 {
        return (0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let mu = theta_t - a * rate;
    let m1 = mu * dt;
    let eta = m1 / dx;
    let k = (eta.round() as isize).clamp(k_min.max(-1), k_max.min(1));
    let alpha = eta - k as f64;

    let v_ratio = sigma * sigma * dt / (dx * dx);
    let pu = 0.5 * (v_ratio + alpha * alpha + alpha);
    let pd = 0.5 * (v_ratio + alpha * alpha - alpha);
    let pm = 1.0 - pu - pd;

    if pu >= 0.0 && pm >= 0.0 && pd >= 0.0 {
        return (k, pu, pm, pd);
    }

    // Fallback for extreme drift (|alpha| > sqrt(2/3) even after adaptive
    // branching, e.g. far grid wings under very strong mean reversion where
    // the tree geometry caps |k| at 1): clamp and renormalize. This trades
    // exact moment matching for stability and is only reachable outside the
    // adaptive-branching regime.
    let pu = pu.max(0.0);
    let pm = pm.max(0.0);
    let pd = pd.max(0.0);
    let norm = pu + pm + pd;
    if norm <= 1.0e-14 {
        (k, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    } else {
        (k, pu / norm, pm / norm, pd / norm)
    }
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
    fn trinomial_probs_match_first_two_moments_at_every_node() {
        // Strong mean reversion so outer grid nodes force adaptive branching
        // (the old clamp-and-renormalize silently broke moment matching here).
        let a = 0.5_f64;
        let sigma = 0.01_f64;
        let dt = 0.1_f64;
        let dx = sigma * (3.0 * dt).sqrt();
        let r0 = 0.05_f64;
        // theta = a * r0 keeps the drift zero at the tree center and large at
        // the wings.
        let theta_t = a * r0;
        let steps = 25_usize;

        let mut saw_adaptive = false;
        for i in 0..steps {
            for j in -(i as isize)..=(i as isize) {
                let rate = r0 + j as f64 * dx;
                let k_min = (-(i as isize) - j).max(-1);
                let k_max = ((i as isize) - j).min(1);
                let (k, pu, pm, pd) =
                    trinomial_probs(theta_t, a, sigma, rate, dt, dx, k_min, k_max);
                if k != 0 {
                    saw_adaptive = true;
                }

                assert!(
                    pu >= 0.0 && pm >= 0.0 && pd >= 0.0,
                    "node ({i},{j}): negative prob"
                );
                assert!(
                    (pu + pm + pd - 1.0).abs() < 1.0e-12,
                    "node ({i},{j}): probs sum to {}",
                    pu + pm + pd
                );

                let m1 = (theta_t - a * rate) * dt;
                let mean = dx * (k as f64 + pu - pd);
                assert!(
                    (mean - m1).abs() < 1.0e-12,
                    "node ({i},{j}): mean {mean} vs target {m1}"
                );

                let kf = k as f64;
                let second = dx
                    * dx
                    * (pu * (kf + 1.0) * (kf + 1.0) + pm * kf * kf + pd * (kf - 1.0) * (kf - 1.0));
                let target = sigma * sigma * dt + m1 * m1;
                assert!(
                    (second - target).abs() < 1.0e-12,
                    "node ({i},{j}): second moment {second} vs target {target}"
                );
            }
        }
        assert!(
            saw_adaptive,
            "test parameters should exercise adaptive (non-standard) branching"
        );
    }

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
