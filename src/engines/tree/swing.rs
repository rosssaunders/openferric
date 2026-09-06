//! Module `engines::tree::swing`.
//!
//! Implements swing abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13, Cox-Ross-Rubinstein (1979), and backward-induction recursions around Eq. (13.10).
//!
//! Key types and purpose: `SwingTreeEngine` define the core data contracts for this module.
//!
//! Numerical considerations: convergence is first- to second-order in time-step count depending on tree parameterization; deep ITM/OTM nodes may need larger depth.
//!
//! When to use: use trees for early-exercise intuition and lattice diagnostics; use analytic formulas for plain vanillas and Monte Carlo/PDE for richer dynamics.
use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::swing::SwingOption;
use crate::market::Market;

/// Binomial-tree swing option engine with exercise-right state.
#[derive(Debug, Clone)]
pub struct SwingTreeEngine {
    /// Number of tree steps.
    pub steps: usize,
}

impl SwingTreeEngine {
    /// Creates a swing tree engine.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

fn expected_continuation(up: f64, down: f64, p: f64, disc: f64) -> f64 {
    if p <= 0.0 {
        return disc * down;
    }
    if p >= 1.0 {
        return disc * up;
    }
    if !up.is_finite() || !down.is_finite() {
        return f64::NEG_INFINITY;
    }
    disc * (p * up + (1.0 - p) * down)
}

fn exercise_steps(dates: &[f64], maturity: f64, steps: usize) -> Vec<bool> {
    let mut flags = vec![false; steps + 1];
    if maturity <= 0.0 {
        return flags;
    }

    for &t in dates {
        let idx = ((t / maturity) * steps as f64).round() as usize;
        flags[idx.min(steps)] = true;
    }
    flags
}

fn remaining_opportunities(flags: &[bool]) -> Vec<usize> {
    let mut out = vec![0_usize; flags.len()];
    let mut running = 0_usize;
    for i in (0..flags.len()).rev() {
        if flags[i] {
            running += 1;
        }
        out[i] = running;
    }
    out
}

impl PricingEngine<SwingOption> for SwingTreeEngine {
    fn price(
        &self,
        instrument: &SwingOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "swing tree steps must be > 0".to_string(),
            ));
        }

        let maturity = instrument
            .exercise_dates
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !maturity.is_finite() || maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "swing maturity inferred from exercise_dates must be > 0".to_string(),
            ));
        }

        let vol = market.checked_vol_for(instrument.strike, maturity)?;
        market.require_continuous_dividends(maturity)?;

        let dt = maturity / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let effective_dividend_yield = market.effective_dividend_yield(maturity);
        let growth = ((market.rate - effective_dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-market.rate * dt).exp();

        let exercise_flags = exercise_steps(&instrument.exercise_dates, maturity, self.steps);
        let opportunities = remaining_opportunities(&exercise_flags);
        if opportunities[0] < instrument.min_exercises {
            return Err(PricingError::InvalidInput(
                "swing tree discretization cannot satisfy min_exercises".to_string(),
            ));
        }
        if opportunities[0] < instrument.max_exercises {
            return Err(PricingError::InvalidInput(
                "swing tree discretization cannot satisfy max_exercises".to_string(),
            ));
        }

        let rights = instrument.max_exercises;

        // Multiplicative recurrence: spot * u^j * d^(n-j) = spot * d^n * (u/d)^j
        let ratio = u / d;

        let mut next = vec![vec![0.0_f64; self.steps + 1]; rights + 1];
        for (rem, next_rem) in next.iter_mut().enumerate().take(rights + 1) {
            let used = rights - rem;
            let feasible_without_exercise = used >= instrument.min_exercises;

            let hold_value = if feasible_without_exercise {
                0.0
            } else {
                f64::NEG_INFINITY
            };

            if exercise_flags[self.steps] && rem > 0 {
                let used_if_exercise = rights - (rem - 1);
                if used_if_exercise >= instrument.min_exercises {
                    let mut st = market.spot * d.powi(self.steps as i32);
                    for value in next_rem.iter_mut().take(self.steps + 1) {
                        let payoff =
                            instrument.payoff_per_exercise * (st - instrument.strike).max(0.0);
                        *value = hold_value.max(payoff);
                        st *= ratio;
                    }
                } else {
                    for value in next_rem.iter_mut().take(self.steps + 1) {
                        *value = hold_value;
                    }
                }
            } else {
                for value in next_rem.iter_mut().take(self.steps + 1) {
                    *value = hold_value;
                }
            }
        }

        for i in (0..self.steps).rev() {
            let mut current = vec![vec![f64::NEG_INFINITY; i + 1]; rights + 1];

            for rem in 0..=rights {
                let used = rights - rem;
                let min_needed = instrument.min_exercises.saturating_sub(used);
                let max_possible = opportunities[i].min(rem);
                if max_possible < min_needed {
                    continue;
                }

                let needs_exercise_check = exercise_flags[i] && rem > 0;
                let exercise_feasible = if needs_exercise_check {
                    let used_if_exercise = rights - (rem - 1);
                    let min_needed_after =
                        instrument.min_exercises.saturating_sub(used_if_exercise);
                    let max_possible_after = opportunities[i + 1].min(rem - 1);
                    max_possible_after >= min_needed_after
                } else {
                    false
                };

                if exercise_feasible {
                    let mut st = market.spot * d.powi(i as i32);
                    for j in 0..=i {
                        let hold = expected_continuation(next[rem][j + 1], next[rem][j], p, disc);
                        let mut best = hold;
                        let continuation_after_exercise =
                            expected_continuation(next[rem - 1][j + 1], next[rem - 1][j], p, disc);
                        if continuation_after_exercise.is_finite() {
                            let payoff =
                                instrument.payoff_per_exercise * (st - instrument.strike).max(0.0);
                            best = best.max(payoff + continuation_after_exercise);
                        }
                        current[rem][j] = best;
                        st *= ratio;
                    }
                } else {
                    for j in 0..=i {
                        current[rem][j] =
                            expected_continuation(next[rem][j + 1], next[rem][j], p, disc);
                    }
                }
            }

            next = current;
        }

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("vol", vol);
        diagnostics.insert("exercise_dates", instrument.exercise_dates.len() as f64);
        diagnostics.insert("min_exercises", instrument.min_exercises as f64);
        diagnostics.insert("max_exercises", instrument.max_exercises as f64);

        let price = next[rights][0];
        if !price.is_finite() {
            return Err(PricingError::NumericalError(
                "swing tree produced non-finite value; check exercise constraints and discretization"
                    .to_string(),
            ));
        }

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PricingEngine;
    use crate::engines::tree::binomial::BinomialTreeEngine;
    use crate::instruments::vanilla::VanillaOption;
    use statrs::distribution::{ContinuousCDF, Normal};

    fn black_scholes_call(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        maturity: f64,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sigma_sqrt_t = volatility * maturity.sqrt();
        let d1 = ((spot / strike).ln()
            + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity)
            / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        spot * (-dividend_yield * maturity).exp() * normal.cdf(d1)
            - strike * (-rate * maturity).exp() * normal.cdf(d2)
    }

    #[test]
    fn unconstrained_swing_matches_strip_of_black_scholes_calls() {
        // With one right for every exercise date, exercising never consumes a
        // scarce right. The swing value is exactly a strip of European calls.
        const CRR_DISCRETIZATION_BUDGET: f64 = 6.5e-3;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let swing = SwingOption::new(0, 2, vec![0.5, 1.0], 100.0, 1.0);
        let actual = SwingTreeEngine::new(800)
            .price(&swing, &market)
            .unwrap()
            .price;
        let reference = black_scholes_call(100.0, 100.0, 0.05, 0.0, 0.20, 0.5)
            + black_scholes_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

        assert!(
            (actual - reference).abs() <= CRR_DISCRETIZATION_BUDGET,
            "swing strip price {actual:.12} differs from BS strip {reference:.12}"
        );
    }

    #[test]
    fn constrained_monthly_swing_matches_quantlib_fd_reference_and_converges() {
        // Independent oracle generated with QuantLib 1.43
        // FdSimpleBSSwingEngine(t_grid=960, x_grid=1200) for this contract.
        const QUANTLIB_REFERENCE: f64 = 54.200_380_350_565_76;
        const FINE_GRID_DISCRETIZATION_BUDGET: f64 = 7.5e-3;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let exercise_dates: Vec<f64> = (1..=12).map(|m| m as f64 / 12.0).collect();
        let swing = SwingOption::new(0, 6, exercise_dates, 100.0, 1.0);

        let coarse_price = SwingTreeEngine::new(960)
            .price(&swing, &market)
            .unwrap()
            .price;
        let fine_price = SwingTreeEngine::new(1_920)
            .price(&swing, &market)
            .unwrap()
            .price;
        let coarse_error = (coarse_price - QUANTLIB_REFERENCE).abs();
        let fine_error = (fine_price - QUANTLIB_REFERENCE).abs();

        assert!(
            fine_error <= FINE_GRID_DISCRETIZATION_BUDGET,
            "fine swing price {fine_price:.12} differs from QuantLib reference \
             {QUANTLIB_REFERENCE:.12} by {fine_error:.12}"
        );
        assert!(
            fine_error <= 0.51 * coarse_error,
            "expected first-order convergence: coarse error={coarse_error:.12}, \
             fine error={fine_error:.12}"
        );

        // Supplemental economic bounds retained from the original smoke test.
        let single_euro = VanillaOption::european_call(100.0, 0.5);
        let euro_price = BinomialTreeEngine::new(960)
            .price(&single_euro, &market)
            .unwrap()
            .price;

        assert!(
            fine_price > 6.0 * euro_price,
            "expected swing > 6x single european: swing={} european={}",
            fine_price,
            euro_price
        );
        assert!(
            fine_price < 12.0 * euro_price,
            "expected swing < 12x single european: swing={} european={}",
            fine_price,
            euro_price
        );
    }

    #[test]
    fn minimum_rights_swing_matches_quantlib_fd_reference_and_converges() {
        // Offline QuantLib-Python 1.43 reference: evaluation date 2025-01-01,
        // monthly SwingExercise dates through 2026-01-01, 30/360 Bond Basis
        // (so the times are exactly m/12), flat continuous r=5%, q=0%,
        // BlackConstantVol=20%, PlainVanillaPayoff(Call, K=100), and
        // VanillaSwingOption(minExerciseRights=2, maxExerciseRights=6).
        // FdSimpleBSSwingEngine(tGrid=960, xGrid=1200) gives the literal below.
        // Because every exercise payoff is floored at zero, the two minimum
        // rights can be spent for zero and do not alter this particular NPV.
        // This is supplemental state-path execution coverage, not evidence
        // that a positive minimum has an economically binding effect.
        const QUANTLIB_REFERENCE: f64 = 54.200_380_350_565_76;
        const FINE_GRID_DISCRETIZATION_BUDGET: f64 = 7.5e-3;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();
        let exercise_dates: Vec<f64> = (1..=12).map(|m| m as f64 / 12.0).collect();
        let swing = SwingOption::new(2, 6, exercise_dates, 100.0, 1.0);

        let coarse_price = SwingTreeEngine::new(960)
            .price(&swing, &market)
            .unwrap()
            .price;
        let fine_price = SwingTreeEngine::new(1_920)
            .price(&swing, &market)
            .unwrap()
            .price;
        let coarse_error = (coarse_price - QUANTLIB_REFERENCE).abs();
        let fine_error = (fine_price - QUANTLIB_REFERENCE).abs();

        assert!(
            fine_error <= FINE_GRID_DISCRETIZATION_BUDGET,
            "fine minimum-rights swing price {fine_price:.12} differs from QuantLib reference \
             {QUANTLIB_REFERENCE:.12} by {fine_error:.12}"
        );
        assert!(
            fine_error <= 0.51 * coarse_error,
            "expected first-order convergence for minimum-rights swing: \
             coarse error={coarse_error:.12}, fine error={fine_error:.12}"
        );
    }
}
