//! Module `instruments::vanilla`.
//!
//! Implements vanilla abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `VanillaOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{ExerciseStyle, Instrument, OptionType, PricingError};

/// Vanilla option instrument.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VanillaOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike level.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Exercise style.
    pub exercise: ExerciseStyle,
}

impl VanillaOption {
    /// Builds a European call option.
    pub fn european_call(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Call,
            strike,
            expiry,
            exercise: ExerciseStyle::European,
        }
    }

    /// Builds a European put option.
    pub fn european_put(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Put,
            strike,
            expiry,
            exercise: ExerciseStyle::European,
        }
    }

    /// Builds an American call option.
    pub fn american_call(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Call,
            strike,
            expiry,
            exercise: ExerciseStyle::American,
        }
    }

    /// Builds an American put option.
    pub fn american_put(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Put,
            strike,
            expiry,
            exercise: ExerciseStyle::American,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "vanilla strike must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "vanilla expiry must be finite and >= 0".to_string(),
            ));
        }

        if let ExerciseStyle::Bermudan { dates } = &self.exercise {
            if dates.is_empty() {
                return Err(PricingError::InvalidInput(
                    "bermudan exercise dates cannot be empty".to_string(),
                ));
            }
            if dates
                .iter()
                .any(|&d| !d.is_finite() || d <= 0.0 || d > self.expiry)
            {
                return Err(PricingError::InvalidInput(
                    "bermudan exercise dates must be finite and lie in (0, expiry]".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Instrument for VanillaOption {
    fn instrument_type(&self) -> &str {
        "VanillaOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_nan_and_infinite_fields() {
        let cases: Vec<(&str, VanillaOption)> = vec![
            ("NaN strike", VanillaOption::european_call(f64::NAN, 1.0)),
            (
                "infinite strike",
                VanillaOption::european_call(f64::INFINITY, 1.0),
            ),
            ("NaN expiry", VanillaOption::european_call(100.0, f64::NAN)),
            (
                "infinite expiry",
                VanillaOption::european_put(100.0, f64::INFINITY),
            ),
            ("NaN bermudan date", {
                let mut o = VanillaOption::european_call(100.0, 1.0);
                o.exercise = ExerciseStyle::Bermudan {
                    dates: vec![0.5, f64::NAN],
                };
                o
            }),
        ];

        for (label, option) in cases {
            assert!(option.validate().is_err(), "{label} must be rejected");
        }

        assert!(VanillaOption::european_call(100.0, 1.0).validate().is_ok());
    }
}
