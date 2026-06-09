//! Module `instruments::digital`.
//!
//! Implements digital abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `CashOrNothingOption`, `AssetOrNothingOption`, `GapOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Cash-or-nothing digital option.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CashOrNothingOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Trigger strike.
    pub strike: f64,
    /// Fixed cash amount paid when the option expires in the money.
    pub cash: f64,
    /// Expiry in years.
    pub expiry: f64,
}

impl CashOrNothingOption {
    /// Creates a cash-or-nothing option.
    pub fn new(option_type: OptionType, strike: f64, cash: f64, expiry: f64) -> Self {
        Self {
            option_type,
            strike,
            cash,
            expiry,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing strike must be finite and > 0".to_string(),
            ));
        }
        if !self.cash.is_finite() || self.cash < 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing cash must be finite and >= 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing expiry must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for CashOrNothingOption {
    fn instrument_type(&self) -> &str {
        "CashOrNothingOption"
    }
}

/// Asset-or-nothing digital option.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssetOrNothingOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Trigger strike.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
}

impl AssetOrNothingOption {
    /// Creates an asset-or-nothing option.
    pub fn new(option_type: OptionType, strike: f64, expiry: f64) -> Self {
        Self {
            option_type,
            strike,
            expiry,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "asset-or-nothing strike must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "asset-or-nothing expiry must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for AssetOrNothingOption {
    fn instrument_type(&self) -> &str {
        "AssetOrNothingOption"
    }
}

/// Gap option with distinct trigger and payoff strikes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GapOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Payoff strike `K1` in `S_T - K1` (call) or `K1 - S_T` (put).
    pub payoff_strike: f64,
    /// Trigger strike `K2` controlling in-the-money event.
    pub trigger_strike: f64,
    /// Expiry in years.
    pub expiry: f64,
}

impl GapOption {
    /// Creates a gap option.
    pub fn new(
        option_type: OptionType,
        payoff_strike: f64,
        trigger_strike: f64,
        expiry: f64,
    ) -> Self {
        Self {
            option_type,
            payoff_strike,
            trigger_strike,
            expiry,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.payoff_strike.is_finite() || self.payoff_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "gap payoff_strike must be finite and > 0".to_string(),
            ));
        }
        if !self.trigger_strike.is_finite() || self.trigger_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "gap trigger_strike must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "gap expiry must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for GapOption {
    fn instrument_type(&self) -> &str {
        "GapOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cash_or_nothing_rejects_nan_fields() {
        let valid = CashOrNothingOption::new(OptionType::Call, 100.0, 10.0, 1.0);
        assert!(valid.validate().is_ok());

        let cases = [
            CashOrNothingOption::new(OptionType::Call, f64::NAN, 10.0, 1.0),
            CashOrNothingOption::new(OptionType::Call, 100.0, f64::NAN, 1.0),
            CashOrNothingOption::new(OptionType::Call, 100.0, 10.0, f64::NAN),
        ];
        for (i, option) in cases.iter().enumerate() {
            assert!(option.validate().is_err(), "NaN case {i} must be rejected");
        }
    }

    #[test]
    fn asset_or_nothing_rejects_nan_fields() {
        let valid = AssetOrNothingOption::new(OptionType::Put, 100.0, 1.0);
        assert!(valid.validate().is_ok());

        let cases = [
            AssetOrNothingOption::new(OptionType::Put, f64::NAN, 1.0),
            AssetOrNothingOption::new(OptionType::Put, 100.0, f64::NAN),
        ];
        for (i, option) in cases.iter().enumerate() {
            assert!(option.validate().is_err(), "NaN case {i} must be rejected");
        }
    }

    #[test]
    fn gap_option_rejects_nan_fields() {
        let valid = GapOption::new(OptionType::Call, 95.0, 100.0, 1.0);
        assert!(valid.validate().is_ok());

        let cases = [
            GapOption::new(OptionType::Call, f64::NAN, 100.0, 1.0),
            GapOption::new(OptionType::Call, 95.0, f64::NAN, 1.0),
            GapOption::new(OptionType::Call, 95.0, 100.0, f64::NAN),
        ];
        for (i, option) in cases.iter().enumerate() {
            assert!(option.validate().is_err(), "NaN case {i} must be rejected");
        }
    }
}
