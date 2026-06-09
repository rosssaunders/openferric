//! Module `instruments::fx`.
//!
//! Implements fx abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `FxOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// European FX option parameters for Garman-Kohlhagen pricing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FxOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Domestic continuously compounded rate.
    pub domestic_rate: f64,
    /// Foreign continuously compounded rate.
    pub foreign_rate: f64,
    /// Spot FX level.
    pub spot_fx: f64,
    /// Strike FX level.
    pub strike_fx: f64,
    /// Volatility.
    pub vol: f64,
    /// Maturity in years.
    pub maturity: f64,
}

impl FxOption {
    /// Creates a new FX option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        option_type: OptionType,
        domestic_rate: f64,
        foreign_rate: f64,
        spot_fx: f64,
        strike_fx: f64,
        vol: f64,
        maturity: f64,
    ) -> Self {
        Self {
            option_type,
            domestic_rate,
            foreign_rate,
            spot_fx,
            strike_fx,
            vol,
            maturity,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.spot_fx.is_finite() || self.spot_fx <= 0.0 {
            return Err(PricingError::InvalidInput(
                "fx spot_fx must be finite and > 0".to_string(),
            ));
        }
        if !self.strike_fx.is_finite() || self.strike_fx <= 0.0 {
            return Err(PricingError::InvalidInput(
                "fx strike_fx must be finite and > 0".to_string(),
            ));
        }
        if !self.vol.is_finite() || self.vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "fx vol must be finite and > 0".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "fx maturity must be finite and >= 0".to_string(),
            ));
        }
        if !self.domestic_rate.is_finite() || !self.foreign_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "fx domestic_rate and foreign_rate must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for FxOption {
    fn instrument_type(&self) -> &str {
        "FxOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_option() -> FxOption {
        FxOption::new(OptionType::Call, 0.03, 0.01, 1.25, 1.30, 0.1, 1.0)
    }

    #[test]
    fn validate_rejects_nan_fields() {
        assert!(valid_option().validate().is_ok());

        let cases: Vec<(&str, FxOption)> = vec![
            ("NaN spot_fx", {
                let mut o = valid_option();
                o.spot_fx = f64::NAN;
                o
            }),
            ("NaN strike_fx", {
                let mut o = valid_option();
                o.strike_fx = f64::NAN;
                o
            }),
            ("NaN vol", {
                let mut o = valid_option();
                o.vol = f64::NAN;
                o
            }),
            ("NaN maturity", {
                let mut o = valid_option();
                o.maturity = f64::NAN;
                o
            }),
            ("NaN domestic_rate", {
                let mut o = valid_option();
                o.domestic_rate = f64::NAN;
                o
            }),
            ("NaN foreign_rate", {
                let mut o = valid_option();
                o.foreign_rate = f64::NAN;
                o
            }),
        ];

        for (label, option) in cases {
            assert!(option.validate().is_err(), "{label} must be rejected");
        }
    }
}
