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
        let mut values = vec![0.0_f64; 2 * self.steps + 1];
        for j in -(self.steps as isize)..=(self.steps as isize) {
            let idx = (j + self.steps as isize) as usize;
            let rate = r0 + j as f64 * dx;
            values[idx] = if exercise_flags[self.steps] {
                exercise_value(swaption, &model, curve, horizon, rate)
            } else {
                0.0
            };
        }

        for i in (0..self.steps).rev() {
            let mut next_values = vec![0.0_f64; 2 * i + 1];
            let t = i as f64 * dt;

            for j in -(i as isize)..=(i as isize) {
                let rate = r0 + j as f64 * dx;
                let (pu, pm, pd) = trinomial_probs(&model, t, rate, dt, dx);
                let disc = (-rate * dt).exp();

                let next_shift = i + 1;
                let up_idx = (j + next_shift as isize + 1) as usize;
                let mid_idx = (j + next_shift as isize) as usize;
                let down_idx = (j + next_shift as isize - 1) as usize;
                let continuation =
                    disc * (pu * values[up_idx] + pm * values[mid_idx] + pd * values[down_idx]);

                let idx = (j + i as isize) as usize;
                if exercise_flags[i] {
                    let exercise = exercise_value(swaption, &model, curve, t, rate);
                    next_values[idx] = continuation.max(exercise);
                } else {
                    next_values[idx] = continuation;
                }
            }

            values = next_values;
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

fn trinomial_probs(model: &HullWhite, t: f64, rate: f64, dt: f64, dx: f64) -> (f64, f64, f64) {
    if dx <= 0.0 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let mu = model.theta_at(t) - model.a * rate;
    let m1 = mu * dt;
    let variance = model.sigma * model.sigma * dt;
    let total = (variance + m1 * m1) / (dx * dx);

    let mut pu = 0.5 * (total + m1 / dx);
    let mut pd = 0.5 * (total - m1 / dx);
    let mut pm = 1.0 - total;

    pu = pu.max(0.0);
    pm = pm.max(0.0);
    pd = pd.max(0.0);

    let norm = pu + pm + pd;
    if norm <= 1.0e-14 {
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    } else {
        (pu / norm, pm / norm, pd / norm)
    }
}

fn exercise_value(
    swaption: &Swaption,
    model: &HullWhite,
    curve: &YieldCurve,
    exercise_time: f64,
    short_rate: f64,
) -> f64 {
    let end = exercise_time + swaption.swap_tenor;
    let mut prev = exercise_time;
    let mut annuity = 0.0;

    loop {
        let next = (prev + 1.0).min(end);
        if next <= prev {
            break;
        }
        let delta = next - prev;
        let df = model.bond_price(exercise_time, next, short_rate, curve);
        annuity += delta * df;

        if next >= end - 1.0e-12 {
            break;
        }
        prev = next;
    }

    if annuity <= 0.0 {
        return 0.0;
    }

    let df_end = model.bond_price(exercise_time, end, short_rate, curve);
    let float_leg = swaption.notional * (1.0 - df_end);
    let fixed_leg = swaption.notional * swaption.strike * annuity;
    let swap_value = if swaption.is_payer {
        float_leg - fixed_leg
    } else {
        fixed_leg - float_leg
    };

    swap_value.max(0.0)
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
}
