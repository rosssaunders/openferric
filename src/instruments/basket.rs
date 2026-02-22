//! Multi-asset basket option definitions (average, best-of, and worst-of payoffs).
//!
//! [`BasketType`] distinguishes weighted arithmetic baskets from rank-based extrema,
//! and [`BasketOption`] stores shared strike/maturity/call-put metadata.
//! References: Stulz (1982) and Johnson (1987) for two-asset extreme payoffs,
//! plus standard basket-option treatments in Hull.
//! Validation requires finite weights and enforces non-empty weights for arithmetic baskets.
//! Ranking-based baskets ignore explicit weights by design in this schema.
//! Use this module for product definition; pricing choice (moment matching, MC, copula) is external.

use crate::core::{Instrument, PricingError};

/// Basket payoff definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasketType {
    /// Weighted arithmetic basket `sum_i w_i * S_i(T)`.
    Average,
    /// Best performer on normalized returns `max_i S_i(T)/S_i(0)`.
    BestOf,
    /// Worst performer on normalized returns `min_i S_i(T)/S_i(0)`.
    WorstOf,
}

/// Multi-asset basket option.
#[derive(Debug, Clone, PartialEq)]
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
