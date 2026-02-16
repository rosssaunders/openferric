use std::sync::{Arc, Mutex};

use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::arena::PricingArena;

/// Cox-Ross-Rubinstein binomial tree engine for American vanilla options.
#[derive(Debug, Clone)]
pub struct AmericanBinomialEngine {
    /// Number of tree steps.
    pub steps: usize,
    arena: Option<Arc<Mutex<PricingArena>>>,
}

impl Default for AmericanBinomialEngine {
    fn default() -> Self {
        Self {
            steps: 500,
            arena: None,
        }
    }
}

impl AmericanBinomialEngine {
    /// Creates an American binomial engine with a custom number of steps.
    pub fn new(steps: usize) -> Self {
        Self { steps, arena: None }
    }

    /// Creates an engine that reuses a shared pre-allocated pricing arena.
    pub fn with_arena(steps: usize, arena: Arc<Mutex<PricingArena>>) -> Self {
        Self {
            steps,
            arena: Some(arena),
        }
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn rollback_american_binomial(
    values: &mut [f64],
    steps: usize,
    spot0: f64,
    strike: f64,
    option_type: OptionType,
    u: f64,
    d: f64,
    p: f64,
    disc: f64,
) -> f64 {
    debug_assert!(values.len() >= steps + 1);

    for (j, value) in values.iter_mut().enumerate().take(steps + 1) {
        let st = spot0 * u.powf(j as f64) * d.powf((steps - j) as f64);
        *value = intrinsic(option_type, st, strike);
    }

    for i in (0..steps).rev() {
        for j in 0..=i {
            let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            let st = spot0 * u.powf(j as f64) * d.powf((i - j) as f64);
            let exercise = intrinsic(option_type, st, strike);
            values[j] = continuation.max(exercise);
        }
    }

    values[0]
}

impl PricingEngine<VanillaOption> for AmericanBinomialEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if !matches!(instrument.exercise, ExerciseStyle::American) {
            return Err(PricingError::InvalidInput(
                "AmericanBinomialEngine supports American exercise only".to_string(),
            ));
        }

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "binomial steps must be > 0".to_string(),
            ));
        }

        if instrument.expiry == 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let dt = instrument.expiry / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let growth = ((market.rate - market.dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-market.rate * dt).exp();

        let price = if let Some(arena) = &self.arena {
            let mut guard = arena.lock().unwrap_or_else(|poison| poison.into_inner());
            let values = guard.tree_slice(self.steps + 1);
            rollback_american_binomial(
                values,
                self.steps,
                market.spot,
                instrument.strike,
                instrument.option_type,
                u,
                d,
                p,
                disc,
            )
        } else {
            let mut values = vec![0.0_f64; self.steps + 1];
            rollback_american_binomial(
                &mut values,
                self.steps,
                market.spot,
                instrument.strike,
                instrument.option_type,
                u,
                d,
                p,
                disc,
            )
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("vol", vol);

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
    use std::sync::{Arc, Mutex};

    use super::AmericanBinomialEngine;
    use crate::core::PricingEngine;
    use crate::instruments::VanillaOption;
    use crate::market::Market;
    use crate::math::arena::PricingArena;

    fn setup_market() -> Market {
        Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.25)
            .build()
            .expect("valid market")
    }

    #[test]
    fn american_binomial_with_arena_matches_non_arena_exactly() {
        let market = setup_market();
        let option = VanillaOption::american_put(100.0, 1.0);
        let steps = 400;

        let baseline = AmericanBinomialEngine::new(steps)
            .price(&option, &market)
            .expect("baseline pricing succeeds");

        let arena = Arc::new(Mutex::new(PricingArena::with_capacity(1, steps)));
        let arena_result = AmericanBinomialEngine::with_arena(steps, Arc::clone(&arena))
            .price(&option, &market)
            .expect("arena pricing succeeds");

        assert_eq!(arena_result.price, baseline.price);
    }

    #[test]
    fn american_binomial_shared_arena_is_reusable() {
        let market = setup_market();
        let option = VanillaOption::american_call(95.0, 1.5);
        let steps = 300;

        let shared = Arc::new(Mutex::new(PricingArena::with_capacity(1, 32)));
        let engine = AmericanBinomialEngine::with_arena(steps, Arc::clone(&shared));

        let first = engine
            .price(&option, &market)
            .expect("first pricing succeeds");
        let second = engine
            .price(&option, &market)
            .expect("second pricing succeeds");

        assert_eq!(first.price, second.price);

        let guard = shared.lock().unwrap_or_else(|poison| poison.into_inner());
        assert!(guard.tree_buffer.len() >= steps + 1);
    }
}
