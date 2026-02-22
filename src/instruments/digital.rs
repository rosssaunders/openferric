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
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing strike must be > 0".to_string(),
            ));
        }
        if self.cash < 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing cash must be >= 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "cash-or-nothing expiry must be >= 0".to_string(),
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
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "asset-or-nothing strike must be > 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "asset-or-nothing expiry must be >= 0".to_string(),
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
        if self.payoff_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "gap payoff_strike must be > 0".to_string(),
            ));
        }
        if self.trigger_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "gap trigger_strike must be > 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "gap expiry must be >= 0".to_string(),
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
