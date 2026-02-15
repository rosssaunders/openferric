use std::collections::HashMap;

use crate::core::{Averaging, PricingEngine, PricingError, PricingResult, StrikeType};
use crate::instruments::asian::AsianOption;
use crate::market::Market;
use crate::pricing::asian::geometric_asian_fixed_closed_form;

/// Analytic engine for geometric-average fixed-strike Asian options.
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

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let price = geometric_asian_fixed_closed_form(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            vol,
            instrument.expiry,
        );

        let mut diagnostics = HashMap::new();
        diagnostics.insert("vol".to_string(), vol);
        diagnostics.insert(
            "observation_count".to_string(),
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
