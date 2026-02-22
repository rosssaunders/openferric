//! Module `instruments::power`.
//!
//! Implements power abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `PowerOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Power option with payoff `max(sign * (S^alpha - K), 0)` where sign is +1 (call), -1 (put).
#[derive(Debug, Clone, PartialEq)]
pub struct PowerOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike in transformed units.
    pub strike: f64,
    /// Power exponent `alpha`.
    pub alpha: f64,
    /// Expiry in years.
    pub expiry: f64,
}

impl PowerOption {
    /// Creates a power option.
    pub fn new(option_type: OptionType, strike: f64, alpha: f64, expiry: f64) -> Self {
        Self {
            option_type,
            strike,
            alpha,
            expiry,
        }
    }

    /// Builds a power call.
    pub fn call(strike: f64, alpha: f64, expiry: f64) -> Self {
        Self::new(OptionType::Call, strike, alpha, expiry)
    }

    /// Builds a power put.
    pub fn put(strike: f64, alpha: f64, expiry: f64) -> Self {
        Self::new(OptionType::Put, strike, alpha, expiry)
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "power option strike must be finite and > 0".to_string(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(PricingError::InvalidInput(
                "power option alpha must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "power option expiry must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for PowerOption {
    fn instrument_type(&self) -> &str {
        "PowerOption"
    }
}
