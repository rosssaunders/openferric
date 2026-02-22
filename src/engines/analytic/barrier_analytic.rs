//! Closed-form analytic pricing routines for Barrier Analytic.
//!
//! This module implements formulas and sensitivities used by fast deterministic engines.

use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::barrier::BarrierOption;
use crate::market::Market;
use crate::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;

/// Reiner-Rubinstein style analytic engine for barrier options.
#[derive(Debug, Clone, Default)]
pub struct BarrierAnalyticEngine;

impl BarrierAnalyticEngine {
    /// Creates a barrier analytic engine.
    pub fn new() -> Self {
        Self
    }
}

impl PricingEngine<BarrierOption> for BarrierAnalyticEngine {
    fn price(
        &self,
        instrument: &BarrierOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let price = barrier_price_closed_form_with_carry_and_rebate(
            instrument.option_type,
            instrument.barrier.style,
            instrument.barrier.direction,
            market.spot,
            instrument.strike,
            instrument.barrier.level,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.expiry,
            instrument.barrier.rebate,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("barrier_level", instrument.barrier.level);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
