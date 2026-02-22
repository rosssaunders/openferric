//! Module `instruments::rainbow`.
//!
//! Implements rainbow abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `BestOfTwoCallOption`, `WorstOfTwoCallOption`, `TwoAssetCorrelationOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

fn validate_common(
    s1: f64,
    s2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
) -> Result<(), PricingError> {
    if s1 <= 0.0 || s2 <= 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow spots s1 and s2 must be > 0".to_string(),
        ));
    }
    if vol1 <= 0.0 || vol2 <= 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow volatilities vol1 and vol2 must be > 0".to_string(),
        ));
    }
    if !(-1.0..=1.0).contains(&rho) {
        return Err(PricingError::InvalidInput(
            "rainbow correlation rho must be in [-1, 1]".to_string(),
        ));
    }
    if t < 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow maturity t must be >= 0".to_string(),
        ));
    }
    Ok(())
}

/// Two-asset best-of call: `max(max(S1_T, S2_T) - K, 0)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BestOfTwoCallOption {
    pub s1: f64,
    pub s2: f64,
    pub k: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl BestOfTwoCallOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.k < 0.0 {
            return Err(PricingError::InvalidInput(
                "best-of strike k must be >= 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)
    }
}

impl Instrument for BestOfTwoCallOption {
    fn instrument_type(&self) -> &str {
        "BestOfTwoCallOption"
    }
}

/// Two-asset worst-of call: `max(min(S1_T, S2_T) - K, 0)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorstOfTwoCallOption {
    pub s1: f64,
    pub s2: f64,
    pub k: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl WorstOfTwoCallOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.k < 0.0 {
            return Err(PricingError::InvalidInput(
                "worst-of strike k must be >= 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)
    }
}

impl Instrument for WorstOfTwoCallOption {
    fn instrument_type(&self) -> &str {
        "WorstOfTwoCallOption"
    }
}

/// Two-asset correlation option.
///
/// Call payoff: `1_{S2_T > K2} * max(S1_T - K1, 0)`
/// Put payoff:  `1_{S2_T < K2} * max(K1 - S1_T, 0)`
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TwoAssetCorrelationOption {
    pub option_type: OptionType,
    pub s1: f64,
    pub s2: f64,
    pub k1: f64,
    pub k2: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl TwoAssetCorrelationOption {
    /// Validates option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.k1 <= 0.0 || self.k2 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "correlation option strikes k1 and k2 must be > 0".to_string(),
            ));
        }
        validate_common(self.s1, self.s2, self.vol1, self.vol2, self.rho, self.t)
    }
}

impl Instrument for TwoAssetCorrelationOption {
    fn instrument_type(&self) -> &str {
        "TwoAssetCorrelationOption"
    }
}
