//! Module `instruments::fx`.
//!
//! Implements fx abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `FxOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// European FX option parameters for Garman-Kohlhagen pricing.
#[derive(Debug, Clone, PartialEq)]
pub struct FxOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Domestic continuously compounded rate.
    pub domestic_rate: f64,
    /// Foreign continuously compounded rate.
    pub foreign_rate: f64,
    /// Spot FX level.
    pub spot_fx: f64,
    /// Strike FX level.
    pub strike_fx: f64,
    /// Volatility.
    pub vol: f64,
    /// Maturity in years.
    pub maturity: f64,
}

impl FxOption {
    /// Creates a new FX option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        option_type: OptionType,
        domestic_rate: f64,
        foreign_rate: f64,
        spot_fx: f64,
        strike_fx: f64,
        vol: f64,
        maturity: f64,
    ) -> Self {
        Self {
            option_type,
            domestic_rate,
            foreign_rate,
            spot_fx,
            strike_fx,
            vol,
            maturity,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.spot_fx <= 0.0 {
            return Err(PricingError::InvalidInput(
                "fx spot_fx must be > 0".to_string(),
            ));
        }
        if self.strike_fx <= 0.0 {
            return Err(PricingError::InvalidInput(
                "fx strike_fx must be > 0".to_string(),
            ));
        }
        if self.vol <= 0.0 {
            return Err(PricingError::InvalidInput("fx vol must be > 0".to_string()));
        }
        if self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "fx maturity must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for FxOption {
    fn instrument_type(&self) -> &str {
        "FxOption"
    }
}
