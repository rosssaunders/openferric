//! Module `instruments::black76`.
//!
//! Implements black76 abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `FuturesOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// European option on a forward/futures price for Black-76 style models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuturesOption {
    /// Forward/futures level.
    pub forward: f64,
    /// Strike.
    pub strike: f64,
    /// Lognormal volatility.
    pub vol: f64,
    /// Continuously compounded risk-free rate.
    pub r: f64,
    /// Time to expiry in years.
    pub t: f64,
    /// Call or put.
    pub option_type: OptionType,
}

impl FuturesOption {
    /// Creates a new futures/forward option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward: f64,
        strike: f64,
        vol: f64,
        r: f64,
        t: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            forward,
            strike,
            vol,
            r,
            t,
            option_type,
        }
    }

    /// Creates a call on forward/futures.
    pub fn call(forward: f64, strike: f64, vol: f64, r: f64, t: f64) -> Self {
        Self::new(forward, strike, vol, r, t, OptionType::Call)
    }

    /// Creates a put on forward/futures.
    pub fn put(forward: f64, strike: f64, vol: f64, r: f64, t: f64) -> Self {
        Self::new(forward, strike, vol, r, t, OptionType::Put)
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.forward.is_finite() || self.forward <= 0.0 {
            return Err(PricingError::InvalidInput(
                "futures option forward must be finite and > 0".to_string(),
            ));
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "futures option strike must be finite and > 0".to_string(),
            ));
        }
        if !self.vol.is_finite() || self.vol < 0.0 {
            return Err(PricingError::InvalidInput(
                "futures option vol must be finite and >= 0".to_string(),
            ));
        }
        if !self.r.is_finite() {
            return Err(PricingError::InvalidInput(
                "futures option r must be finite".to_string(),
            ));
        }
        if !self.t.is_finite() || self.t < 0.0 {
            return Err(PricingError::InvalidInput(
                "futures option t must be finite and >= 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for FuturesOption {
    fn instrument_type(&self) -> &str {
        "FuturesOption"
    }
}
