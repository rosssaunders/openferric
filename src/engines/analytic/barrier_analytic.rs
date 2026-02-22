//! Module `engines::analytic::barrier_analytic`.
//!
//! Implements barrier analytic abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `BarrierAnalyticEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
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
        let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);

        let price = barrier_price_closed_form_with_carry_and_rebate(
            instrument.option_type,
            instrument.barrier.style,
            instrument.barrier.direction,
            market.spot,
            instrument.strike,
            instrument.barrier.level,
            market.rate,
            effective_dividend_yield,
            vol,
            instrument.expiry,
            instrument.barrier.rebate,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);
        diagnostics.insert_key(crate::core::DiagKey::BarrierLevel, instrument.barrier.level);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
