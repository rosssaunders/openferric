//! Module `instruments::basket`.
//!
//! Implements basket abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `BasketType`, `BasketOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Basket payoff definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BasketType {
    /// Weighted arithmetic basket `sum_i w_i * S_i(T)`.
    Average,
    /// Best performer on normalized returns `max_i S_i(T)/S_i(0)`.
    BestOf,
    /// Worst performer on normalized returns `min_i S_i(T)/S_i(0)`.
    WorstOf,
}

/// Multi-asset basket option.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BasketOption {
    /// Weights for weighted-average basket.
    pub weights: Vec<f64>,
    pub strike: f64,
    pub maturity: f64,
    pub is_call: bool,
    pub basket_type: BasketType,
}

impl BasketOption {
    /// Validates basket fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.strike < 0.0 {
            return Err(PricingError::InvalidInput(
                "basket strike must be >= 0".to_string(),
            ));
        }
        if self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "basket maturity must be >= 0".to_string(),
            ));
        }

        if self.weights.iter().any(|w| !w.is_finite()) {
            return Err(PricingError::InvalidInput(
                "basket weights must be finite".to_string(),
            ));
        }

        if matches!(self.basket_type, BasketType::Average) && self.weights.is_empty() {
            return Err(PricingError::InvalidInput(
                "average basket requires non-empty weights".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for BasketOption {
    fn instrument_type(&self) -> &str {
        "BasketOption"
    }
}

/// Outperformance basket option.
///
/// Payoff (call): `max(S_leader(T) / B_lagger(T) - K, 0)`, where
/// `B_lagger(T) = sum_i w_i * S_i(T)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OutperformanceBasketOption {
    /// Index of the leader asset in spot/vol arrays.
    pub leader_index: usize,
    /// Lagger basket weights (same length as the asset universe).
    pub lagger_weights: Vec<f64>,
    /// Strike on outperformance ratio.
    pub strike: f64,
    /// Maturity in years.
    pub maturity: f64,
    /// Call or put side.
    pub option_type: OptionType,
}

impl OutperformanceBasketOption {
    /// Validates static fields (dimension checks are done by pricing routines).
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.lagger_weights.is_empty() {
            return Err(PricingError::InvalidInput(
                "outperformance lagger weights cannot be empty".to_string(),
            ));
        }
        if self.strike < 0.0 {
            return Err(PricingError::InvalidInput(
                "outperformance strike must be >= 0".to_string(),
            ));
        }
        if self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "outperformance maturity must be >= 0".to_string(),
            ));
        }
        if self.lagger_weights.iter().any(|w| !w.is_finite()) {
            return Err(PricingError::InvalidInput(
                "outperformance lagger weights must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for OutperformanceBasketOption {
    fn instrument_type(&self) -> &str {
        "OutperformanceBasketOption"
    }
}

/// Quanto basket option settled in domestic currency with a fixed FX conversion rate.
///
/// The underlying basket follows foreign-asset dynamics and uses a quanto drift
/// adjustment via asset/FX correlations.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QuantoBasketOption {
    /// Base basket payoff definition.
    pub basket: BasketOption,
    /// Fixed FX conversion rate into domestic currency.
    pub fx_rate: f64,
    /// FX volatility.
    pub fx_vol: f64,
    /// Asset/FX correlations, one per basket asset.
    pub asset_fx_corr: Vec<f64>,
    /// Domestic risk-free rate for discounting.
    pub domestic_rate: f64,
    /// Foreign risk-free rate driving asset drifts before quanto adjustment.
    pub foreign_rate: f64,
}

impl QuantoBasketOption {
    /// Validates quanto fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        self.basket.validate()?;
        if self.fx_rate <= 0.0 || !self.fx_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "quanto fx_rate must be finite and > 0".to_string(),
            ));
        }
        if self.fx_vol < 0.0 || !self.fx_vol.is_finite() {
            return Err(PricingError::InvalidInput(
                "quanto fx_vol must be finite and >= 0".to_string(),
            ));
        }
        if self
            .asset_fx_corr
            .iter()
            .any(|rho| !rho.is_finite() || !(-1.0..=1.0).contains(rho))
        {
            return Err(PricingError::InvalidInput(
                "quanto asset_fx_corr entries must be finite and in [-1, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for QuantoBasketOption {
    fn instrument_type(&self) -> &str {
        "QuantoBasketOption"
    }
}
