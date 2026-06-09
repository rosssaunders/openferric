//! Module `instruments::asian`.
//!
//! Implements asian abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `AsianOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{AsianSpec, Instrument, OptionType, PricingError};

/// Asian option instrument.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "asian expiry must be finite and >= 0".to_string(),
            ));
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "asian strike must be finite and > 0".to_string(),
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
            .any(|&t| !t.is_finite() || t < 0.0 || t > self.expiry)
        {
            return Err(PricingError::InvalidInput(
                "asian observation_times must be finite and lie in [0, expiry]".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Averaging, StrikeType};

    fn valid_option() -> AsianOption {
        AsianOption::new(
            OptionType::Call,
            100.0,
            1.0,
            AsianSpec {
                averaging: Averaging::Arithmetic,
                strike_type: StrikeType::Fixed,
                observation_times: vec![0.25, 0.5, 0.75, 1.0],
            },
        )
    }

    #[test]
    fn validate_rejects_nan_fields() {
        assert!(valid_option().validate().is_ok());

        let cases: Vec<(&str, AsianOption)> = vec![
            ("NaN strike", {
                let mut o = valid_option();
                o.strike = f64::NAN;
                o
            }),
            ("NaN expiry", {
                let mut o = valid_option();
                o.expiry = f64::NAN;
                o
            }),
            ("NaN observation time", {
                let mut o = valid_option();
                o.asian.observation_times[1] = f64::NAN;
                o
            }),
        ];

        for (label, option) in cases {
            assert!(option.validate().is_err(), "{label} must be rejected");
        }
    }
}
