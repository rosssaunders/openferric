use std::collections::HashMap;

use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::barrier::BarrierOption;
use crate::market::Market;
use crate::pricing::barrier::barrier_price_closed_form;

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

        if instrument.barrier.rebate.abs() > 0.0 {
            return Err(PricingError::InvalidInput(
                "BarrierAnalyticEngine currently supports rebate = 0 only".to_string(),
            ));
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let price = barrier_price_closed_form(
            instrument.option_type,
            instrument.barrier.style,
            instrument.barrier.direction,
            market.spot,
            instrument.strike,
            instrument.barrier.level,
            market.rate,
            vol,
            instrument.expiry,
        );

        let mut diagnostics = HashMap::new();
        diagnostics.insert("vol".to_string(), vol);
        diagnostics.insert("barrier_level".to_string(), instrument.barrier.level);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
