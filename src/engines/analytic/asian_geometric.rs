//! Module `engines::analytic::asian_geometric`.
//!
//! Implements asian geometric abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `GeometricAsianEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{Averaging, PricingEngine, PricingError, PricingResult, StrikeType};
use crate::instruments::asian::AsianOption;
use crate::market::{DividendKind, Market};
use crate::pricing::asian::geometric_asian_discrete_fixed_expected_payoff;

/// Analytic engine for geometric-average fixed-strike Asian options.
/// Proportional dividends enter at their actual fixing weights. Cash dividends
/// on or before a fixing require a numerical engine instead of this lognormal formula.
#[derive(Debug, Clone, Default)]
pub struct GeometricAsianEngine;

impl GeometricAsianEngine {
    /// Creates a geometric Asian analytic engine.
    pub fn new() -> Self {
        Self
    }
}

impl PricingEngine<AsianOption> for GeometricAsianEngine {
    fn price(
        &self,
        instrument: &AsianOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if instrument.asian.averaging != Averaging::Geometric {
            return Err(PricingError::InvalidInput(
                "GeometricAsianEngine requires Averaging::Geometric".to_string(),
            ));
        }

        if instrument.asian.strike_type != StrikeType::Fixed {
            return Err(PricingError::InvalidInput(
                "GeometricAsianEngine currently supports StrikeType::Fixed only".to_string(),
            ));
        }

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;

        let observations = &instrument.asian.observation_times;
        let mut log_dividend_factor = 0.0;
        for event in market.dividends().events() {
            let affected = observations
                .iter()
                .filter(|&&time| time >= event.time)
                .count();
            if affected == 0 {
                continue;
            }
            match event.kind {
                DividendKind::Proportional(ratio) => {
                    log_dividend_factor +=
                        (-ratio).ln_1p() * affected as f64 / observations.len() as f64;
                }
                DividendKind::Cash(_) => {
                    return Err(PricingError::InvalidInput(
                        "GeometricAsianEngine does not support cash dividends on or before a fixing; use Monte Carlo".to_string(),
                    ));
                }
            }
        }

        let expected_payoff = geometric_asian_discrete_fixed_expected_payoff(
            instrument.option_type,
            market.spot * log_dividend_factor.exp(),
            instrument.strike,
            market.rate,
            market.dividend_yield,
            vol,
            &instrument.asian.observation_times,
        );
        let price = (-market.rate * instrument.expiry).exp() * expected_payoff;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert(
            "observation_count",
            instrument.asian.observation_times.len() as f64,
        );

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
