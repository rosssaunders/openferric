//! Asian option contract schema and validation rules.
//!
//! [`AsianOption`] wraps [`crate::core::AsianSpec`] (arithmetic/geometric averaging,
//! fixed/floating strike convention, observation schedule) with option side, strike, and expiry.
//! References for payoff conventions: Kemna and Vorst (1990), Turnbull and Wakeman (1991).
//! Validation enforces non-empty observation times and bounds each fixing to `[0, expiry]`.
//! This module intentionally does not price; pricing engines consume this schema in Monte Carlo
//! and control-variate workflows (for example geometric-Asian closed-form variance reduction).
//! Use this type when you need explicit averaging schedule semantics in instrument definitions.

use crate::core::{AsianSpec, Instrument, OptionType, PricingError};

/// Asian option instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct AsianOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike level (used for fixed-strike variants).
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Asian-specific contract terms.
    pub asian: AsianSpec,
}

impl AsianOption {
    /// Builds a new Asian option.
    pub fn new(option_type: OptionType, strike: f64, expiry: f64, asian: AsianSpec) -> Self {
        Self {
            option_type,
            strike,
            expiry,
            asian,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "asian expiry must be >= 0".to_string(),
            ));
        }
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "asian strike must be > 0".to_string(),
            ));
        }
        if self.asian.observation_times.is_empty() {
            return Err(PricingError::InvalidInput(
                "asian observation_times cannot be empty".to_string(),
            ));
        }
        if self
            .asian
            .observation_times
            .iter()
            .any(|&t| t < 0.0 || t > self.expiry)
        {
            return Err(PricingError::InvalidInput(
                "asian observation_times must lie in [0, expiry]".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for AsianOption {
    fn instrument_type(&self) -> &str {
        "AsianOption"
    }
}
