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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid(option: &PowerOption, message: &str) {
        assert_eq!(
            option.validate(),
            Err(PricingError::InvalidInput(message.to_string()))
        );
    }

    #[test]
    fn constructors_preserve_terms_and_option_side_through_serialization() {
        let call = PowerOption::call(10_000.0, 2.0, 0.5);
        let put = PowerOption::put(85.0, 0.75, 2.25);

        assert_eq!(call, PowerOption::new(OptionType::Call, 10_000.0, 2.0, 0.5));
        assert_eq!(put, PowerOption::new(OptionType::Put, 85.0, 0.75, 2.25));
        assert_eq!(call.instrument_type(), "PowerOption");
        assert_eq!(put.instrument_type(), "PowerOption");
        assert_eq!(call.validate(), Ok(()));
        assert_eq!(put.validate(), Ok(()));

        for (option, serialized_side) in [(call, "Call"), (put, "Put")] {
            let value = serde_json::to_value(&option).expect("serialize power option");
            assert_eq!(value["option_type"], serialized_side);
            assert_eq!(value["strike"], option.strike);
            assert_eq!(value["alpha"], option.alpha);
            assert_eq!(value["expiry"], option.expiry);
            assert_eq!(
                serde_json::from_value::<PowerOption>(value).expect("deserialize power option"),
                option
            );
        }
    }

    #[test]
    fn validation_accepts_exact_boundary_values() {
        let option = PowerOption::call(f64::MIN_POSITIVE, f64::MIN_POSITIVE, 0.0);
        assert_eq!(option.validate(), Ok(()));
    }

    #[test]
    fn validation_rejects_non_finite_and_out_of_domain_fields() {
        const STRIKE_ERROR: &str = "power option strike must be finite and > 0";
        const ALPHA_ERROR: &str = "power option alpha must be finite and > 0";
        const EXPIRY_ERROR: &str = "power option expiry must be finite and >= 0";
        let base = PowerOption::put(100.0, 1.5, 1.0);

        for strike in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_invalid(
                &PowerOption {
                    strike,
                    ..base.clone()
                },
                STRIKE_ERROR,
            );
        }
        for alpha in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
            assert_invalid(
                &PowerOption {
                    alpha,
                    ..base.clone()
                },
                ALPHA_ERROR,
            );
        }
        for expiry in [-f64::EPSILON, f64::NAN, f64::INFINITY] {
            assert_invalid(
                &PowerOption {
                    expiry,
                    ..base.clone()
                },
                EXPIRY_ERROR,
            );
        }
    }
}
