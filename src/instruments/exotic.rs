//! Instrument definition for Exotic contracts.
//!
//! Module openferric::instruments::exotic contains payoff parameters and validation logic.

use crate::core::{Instrument, OptionType, PricingError};

/// Floating-strike lookback option.
#[derive(Debug, Clone, PartialEq)]
pub struct LookbackFloatingOption {
    /// Call (payoff `S_T - S_min`) or put (payoff `S_max - S_T`).
    pub option_type: OptionType,
    /// Expiry in years.
    pub expiry: f64,
    /// Observed running extreme up to valuation time.
    /// For calls this is `S_min`; for puts this is `S_max`.
    pub observed_extreme: Option<f64>,
}

/// Fixed-strike lookback option.
#[derive(Debug, Clone, PartialEq)]
pub struct LookbackFixedOption {
    /// Call (payoff `max(S_max - K, 0)`) or put (payoff `max(K - S_min, 0)`).
    pub option_type: OptionType,
    /// Strike level.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Observed running extreme up to valuation time.
    /// For calls this is `S_max`; for puts this is `S_min`.
    pub observed_extreme: Option<f64>,
}

/// Simple chooser option where the holder chooses call or put at `choose_time`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChooserOption {
    /// Strike level shared by call and put.
    pub strike: f64,
    /// Final expiry in years.
    pub expiry: f64,
    /// Choice time in years.
    pub choose_time: f64,
}

/// Quanto European option with fixed FX conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantoOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike in foreign asset units.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Fixed FX conversion rate into domestic currency.
    pub fx_rate: f64,
    /// Foreign risk-free rate used in quanto drift.
    pub foreign_rate: f64,
    /// Volatility of the FX rate process.
    pub fx_vol: f64,
    /// Correlation between asset and FX shocks.
    pub asset_fx_corr: f64,
}

/// Compound option on a vanilla option.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundOption {
    /// Outer option type (call/put on the underlying option value).
    pub option_type: OptionType,
    /// Inner vanilla option type.
    pub underlying_option_type: OptionType,
    /// Compound strike paid at compound expiry.
    pub compound_strike: f64,
    /// Strike of the underlying vanilla option.
    pub underlying_strike: f64,
    /// Compound option expiry `T1`.
    pub compound_expiry: f64,
    /// Underlying vanilla option expiry `T2` with `T2 >= T1`.
    pub underlying_expiry: f64,
}

/// Unified exotic option instrument.
#[derive(Debug, Clone, PartialEq)]
pub enum ExoticOption {
    /// Floating-strike lookback option.
    LookbackFloating(LookbackFloatingOption),
    /// Fixed-strike lookback option.
    LookbackFixed(LookbackFixedOption),
    /// Chooser option.
    Chooser(ChooserOption),
    /// Quanto option.
    Quanto(QuantoOption),
    /// Compound option.
    Compound(CompoundOption),
}

impl ExoticOption {
    /// Builds a floating-strike lookback call.
    pub fn lookback_floating_call(expiry: f64) -> Self {
        Self::LookbackFloating(LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry,
            observed_extreme: None,
        })
    }

    /// Builds a floating-strike lookback put.
    pub fn lookback_floating_put(expiry: f64) -> Self {
        Self::LookbackFloating(LookbackFloatingOption {
            option_type: OptionType::Put,
            expiry,
            observed_extreme: None,
        })
    }

    /// Builds a fixed-strike lookback call.
    pub fn lookback_fixed_call(strike: f64, expiry: f64) -> Self {
        Self::LookbackFixed(LookbackFixedOption {
            option_type: OptionType::Call,
            strike,
            expiry,
            observed_extreme: None,
        })
    }

    /// Builds a fixed-strike lookback put.
    pub fn lookback_fixed_put(strike: f64, expiry: f64) -> Self {
        Self::LookbackFixed(LookbackFixedOption {
            option_type: OptionType::Put,
            strike,
            expiry,
            observed_extreme: None,
        })
    }

    /// Validates exotic instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        match self {
            Self::LookbackFloating(spec) => {
                if spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback expiry must be >= 0".to_string(),
                    ));
                }
                if let Some(extreme) = spec.observed_extreme
                    && extreme <= 0.0
                {
                    return Err(PricingError::InvalidInput(
                        "lookback observed_extreme must be > 0".to_string(),
                    ));
                }
            }
            Self::LookbackFixed(spec) => {
                if spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed strike must be > 0".to_string(),
                    ));
                }
                if spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed expiry must be >= 0".to_string(),
                    ));
                }
                if let Some(extreme) = spec.observed_extreme
                    && extreme <= 0.0
                {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed observed_extreme must be > 0".to_string(),
                    ));
                }
            }
            Self::Chooser(spec) => {
                if spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "chooser strike must be > 0".to_string(),
                    ));
                }
                if spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "chooser expiry must be >= 0".to_string(),
                    ));
                }
                if spec.choose_time < 0.0 || spec.choose_time > spec.expiry {
                    return Err(PricingError::InvalidInput(
                        "chooser choose_time must lie in [0, expiry]".to_string(),
                    ));
                }
            }
            Self::Quanto(spec) => {
                if spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto strike must be > 0".to_string(),
                    ));
                }
                if spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto expiry must be >= 0".to_string(),
                    ));
                }
                if spec.fx_rate <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto fx_rate must be > 0".to_string(),
                    ));
                }
                if spec.fx_vol < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto fx_vol must be >= 0".to_string(),
                    ));
                }
                if spec.asset_fx_corr < -1.0 || spec.asset_fx_corr > 1.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto asset_fx_corr must be in [-1, 1]".to_string(),
                    ));
                }
            }
            Self::Compound(spec) => {
                if spec.compound_strike <= 0.0 || spec.underlying_strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "compound strikes must be > 0".to_string(),
                    ));
                }
                if spec.compound_expiry < 0.0 || spec.underlying_expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "compound expiries must be >= 0".to_string(),
                    ));
                }
                if spec.compound_expiry > spec.underlying_expiry {
                    return Err(PricingError::InvalidInput(
                        "compound_expiry must be <= underlying_expiry".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Instrument for ExoticOption {
    fn instrument_type(&self) -> &str {
        "ExoticOption"
    }
}
