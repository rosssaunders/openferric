//! Module `instruments::exotic`.
//!
//! Implements exotic abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `LookbackFloatingOption`, `LookbackFixedOption`, `ChooserOption`, `QuantoOption`, `CompoundOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Floating-strike lookback option.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChooserOption {
    /// Strike level shared by call and put.
    pub strike: f64,
    /// Final expiry in years.
    pub expiry: f64,
    /// Choice time in years.
    pub choose_time: f64,
}

/// Quanto European option with fixed FX conversion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
                if !spec.expiry.is_finite() || spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback expiry must be finite and >= 0".to_string(),
                    ));
                }
                if let Some(extreme) = spec.observed_extreme
                    && (!extreme.is_finite() || extreme <= 0.0)
                {
                    return Err(PricingError::InvalidInput(
                        "lookback observed_extreme must be finite and > 0".to_string(),
                    ));
                }
            }
            Self::LookbackFixed(spec) => {
                if !spec.strike.is_finite() || spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed strike must be finite and > 0".to_string(),
                    ));
                }
                if !spec.expiry.is_finite() || spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed expiry must be finite and >= 0".to_string(),
                    ));
                }
                if let Some(extreme) = spec.observed_extreme
                    && (!extreme.is_finite() || extreme <= 0.0)
                {
                    return Err(PricingError::InvalidInput(
                        "lookback fixed observed_extreme must be finite and > 0".to_string(),
                    ));
                }
            }
            Self::Chooser(spec) => {
                if !spec.strike.is_finite() || spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "chooser strike must be finite and > 0".to_string(),
                    ));
                }
                if !spec.expiry.is_finite() || spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "chooser expiry must be finite and >= 0".to_string(),
                    ));
                }
                if !spec.choose_time.is_finite()
                    || spec.choose_time < 0.0
                    || spec.choose_time > spec.expiry
                {
                    return Err(PricingError::InvalidInput(
                        "chooser choose_time must be finite and lie in [0, expiry]".to_string(),
                    ));
                }
            }
            Self::Quanto(spec) => {
                if !spec.strike.is_finite() || spec.strike <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto strike must be finite and > 0".to_string(),
                    ));
                }
                if !spec.expiry.is_finite() || spec.expiry < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto expiry must be finite and >= 0".to_string(),
                    ));
                }
                if !spec.fx_rate.is_finite() || spec.fx_rate <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto fx_rate must be finite and > 0".to_string(),
                    ));
                }
                if !spec.foreign_rate.is_finite() {
                    return Err(PricingError::InvalidInput(
                        "quanto foreign_rate must be finite".to_string(),
                    ));
                }
                if !spec.fx_vol.is_finite() || spec.fx_vol < 0.0 {
                    return Err(PricingError::InvalidInput(
                        "quanto fx_vol must be finite and >= 0".to_string(),
                    ));
                }
                if !spec.asset_fx_corr.is_finite()
                    || spec.asset_fx_corr < -1.0
                    || spec.asset_fx_corr > 1.0
                {
                    return Err(PricingError::InvalidInput(
                        "quanto asset_fx_corr must be finite and in [-1, 1]".to_string(),
                    ));
                }
            }
            Self::Compound(spec) => {
                if !spec.compound_strike.is_finite()
                    || !spec.underlying_strike.is_finite()
                    || spec.compound_strike <= 0.0
                    || spec.underlying_strike <= 0.0
                {
                    return Err(PricingError::InvalidInput(
                        "compound strikes must be finite and > 0".to_string(),
                    ));
                }
                if !spec.compound_expiry.is_finite()
                    || !spec.underlying_expiry.is_finite()
                    || spec.compound_expiry < 0.0
                    || spec.underlying_expiry < 0.0
                {
                    return Err(PricingError::InvalidInput(
                        "compound expiries must be finite and >= 0".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookback_constructors_create_valid_contracts() {
        for option in [
            ExoticOption::lookback_floating_call(1.0),
            ExoticOption::lookback_floating_put(1.0),
            ExoticOption::lookback_fixed_call(100.0, 1.0),
            ExoticOption::lookback_fixed_put(100.0, 1.0),
        ] {
            assert!(option.validate().is_ok());
            assert_eq!(option.instrument_type(), "ExoticOption");
        }
    }

    #[test]
    fn lookback_validation_rejects_each_invalid_domain() {
        let invalid = [
            ExoticOption::lookback_floating_call(-1.0),
            ExoticOption::LookbackFloating(LookbackFloatingOption {
                option_type: OptionType::Put,
                expiry: 1.0,
                observed_extreme: Some(0.0),
            }),
            ExoticOption::lookback_fixed_call(0.0, 1.0),
            ExoticOption::lookback_fixed_put(100.0, -1.0),
            ExoticOption::LookbackFixed(LookbackFixedOption {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: 1.0,
                observed_extreme: Some(-1.0),
            }),
        ];
        for option in invalid {
            assert!(option.validate().is_err(), "unexpectedly valid: {option:?}");
        }
    }

    #[test]
    fn exotic_validation_rejects_nonfinite_scalar_fields() {
        let floating = LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry: 1.0,
            observed_extreme: Some(90.0),
        };
        let fixed = LookbackFixedOption {
            option_type: OptionType::Put,
            strike: 100.0,
            expiry: 1.0,
            observed_extreme: Some(110.0),
        };
        let chooser = ChooserOption {
            strike: 100.0,
            expiry: 1.0,
            choose_time: 0.5,
        };
        let quanto = QuantoOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            fx_rate: 1.2,
            foreign_rate: 0.02,
            fx_vol: 0.15,
            asset_fx_corr: -0.35,
        };
        let compound = CompoundOption {
            option_type: OptionType::Put,
            underlying_option_type: OptionType::Call,
            compound_strike: 8.0,
            underlying_strike: 100.0,
            compound_expiry: 0.5,
            underlying_expiry: 1.0,
        };

        let invalid = [
            ExoticOption::LookbackFloating(LookbackFloatingOption {
                expiry: f64::NAN,
                ..floating.clone()
            }),
            ExoticOption::LookbackFloating(LookbackFloatingOption {
                observed_extreme: Some(f64::NAN),
                ..floating
            }),
            ExoticOption::LookbackFixed(LookbackFixedOption {
                strike: f64::NAN,
                ..fixed.clone()
            }),
            ExoticOption::LookbackFixed(LookbackFixedOption {
                expiry: f64::NAN,
                ..fixed.clone()
            }),
            ExoticOption::LookbackFixed(LookbackFixedOption {
                observed_extreme: Some(f64::NAN),
                ..fixed
            }),
            ExoticOption::Chooser(ChooserOption {
                strike: f64::NAN,
                ..chooser.clone()
            }),
            ExoticOption::Chooser(ChooserOption {
                expiry: f64::NAN,
                ..chooser.clone()
            }),
            ExoticOption::Chooser(ChooserOption {
                choose_time: f64::NAN,
                ..chooser
            }),
            ExoticOption::Quanto(QuantoOption {
                strike: f64::NAN,
                ..quanto.clone()
            }),
            ExoticOption::Quanto(QuantoOption {
                expiry: f64::NAN,
                ..quanto.clone()
            }),
            ExoticOption::Quanto(QuantoOption {
                fx_rate: f64::NAN,
                ..quanto.clone()
            }),
            ExoticOption::Quanto(QuantoOption {
                foreign_rate: f64::NAN,
                ..quanto.clone()
            }),
            ExoticOption::Quanto(QuantoOption {
                fx_vol: f64::NAN,
                ..quanto.clone()
            }),
            ExoticOption::Quanto(QuantoOption {
                asset_fx_corr: f64::NAN,
                ..quanto
            }),
            ExoticOption::Compound(CompoundOption {
                compound_strike: f64::NAN,
                ..compound.clone()
            }),
            ExoticOption::Compound(CompoundOption {
                underlying_strike: f64::NAN,
                ..compound.clone()
            }),
            ExoticOption::Compound(CompoundOption {
                compound_expiry: f64::NAN,
                ..compound.clone()
            }),
            ExoticOption::Compound(CompoundOption {
                underlying_expiry: f64::NAN,
                ..compound
            }),
        ];

        for option in invalid {
            assert!(option.validate().is_err(), "unexpectedly valid: {option:?}");
        }
    }

    #[test]
    fn chooser_quanto_and_compound_validation_cover_all_contract_rules() {
        let chooser = ExoticOption::Chooser(ChooserOption {
            strike: 100.0,
            expiry: 2.0,
            choose_time: 0.75,
        });
        let quanto = ExoticOption::Quanto(QuantoOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            fx_rate: 1.2,
            foreign_rate: 0.02,
            fx_vol: 0.15,
            asset_fx_corr: -0.35,
        });
        let compound = ExoticOption::Compound(CompoundOption {
            option_type: OptionType::Put,
            underlying_option_type: OptionType::Call,
            compound_strike: 8.0,
            underlying_strike: 100.0,
            compound_expiry: 0.5,
            underlying_expiry: 1.0,
        });
        for option in [&chooser, &quanto, &compound] {
            assert!(option.validate().is_ok());
        }

        let invalid = [
            ExoticOption::Chooser(ChooserOption {
                strike: 0.0,
                expiry: 1.0,
                choose_time: 0.5,
            }),
            ExoticOption::Chooser(ChooserOption {
                strike: 100.0,
                expiry: -1.0,
                choose_time: 0.0,
            }),
            ExoticOption::Chooser(ChooserOption {
                strike: 100.0,
                expiry: 1.0,
                choose_time: 1.5,
            }),
            ExoticOption::Quanto(QuantoOption {
                option_type: OptionType::Call,
                strike: 0.0,
                expiry: 1.0,
                fx_rate: 1.0,
                foreign_rate: 0.0,
                fx_vol: 0.2,
                asset_fx_corr: 0.0,
            }),
            ExoticOption::Quanto(QuantoOption {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: -1.0,
                fx_rate: 1.0,
                foreign_rate: 0.0,
                fx_vol: 0.2,
                asset_fx_corr: 0.0,
            }),
            ExoticOption::Quanto(QuantoOption {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: 1.0,
                fx_rate: 0.0,
                foreign_rate: 0.0,
                fx_vol: 0.2,
                asset_fx_corr: 0.0,
            }),
            ExoticOption::Quanto(QuantoOption {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: 1.0,
                fx_rate: 1.0,
                foreign_rate: 0.0,
                fx_vol: -0.1,
                asset_fx_corr: 0.0,
            }),
            ExoticOption::Quanto(QuantoOption {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: 1.0,
                fx_rate: 1.0,
                foreign_rate: 0.0,
                fx_vol: 0.2,
                asset_fx_corr: 1.1,
            }),
            ExoticOption::Compound(CompoundOption {
                option_type: OptionType::Call,
                underlying_option_type: OptionType::Put,
                compound_strike: 0.0,
                underlying_strike: 100.0,
                compound_expiry: 0.5,
                underlying_expiry: 1.0,
            }),
            ExoticOption::Compound(CompoundOption {
                option_type: OptionType::Call,
                underlying_option_type: OptionType::Put,
                compound_strike: 5.0,
                underlying_strike: 100.0,
                compound_expiry: -0.5,
                underlying_expiry: 1.0,
            }),
            ExoticOption::Compound(CompoundOption {
                option_type: OptionType::Call,
                underlying_option_type: OptionType::Put,
                compound_strike: 5.0,
                underlying_strike: 100.0,
                compound_expiry: 1.5,
                underlying_expiry: 1.0,
            }),
        ];
        for option in invalid {
            assert!(option.validate().is_err(), "unexpectedly valid: {option:?}");
        }
    }
}
