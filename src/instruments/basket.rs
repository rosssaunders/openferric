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
use crate::core::{Instrument, PricingError};

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
