//! Module `instruments::double_barrier`.
//!
//! Implements double barrier abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `DoubleBarrierType`, `DoubleBarrierOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Double-barrier knock behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DoubleBarrierType {
    /// Option deactivates if either barrier is hit.
    KnockOut,
    /// Option activates if either barrier is hit.
    KnockIn,
}

/// Double-barrier option with lower/upper barriers.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DoubleBarrierOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike level.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Lower barrier level.
    pub lower_barrier: f64,
    /// Upper barrier level.
    pub upper_barrier: f64,
    /// Knock-in or knock-out type.
    pub barrier_type: DoubleBarrierType,
    /// Cash rebate amount.
    pub rebate: f64,
}

impl DoubleBarrierOption {
    /// Creates a new double-barrier option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        option_type: OptionType,
        strike: f64,
        expiry: f64,
        lower_barrier: f64,
        upper_barrier: f64,
        barrier_type: DoubleBarrierType,
        rebate: f64,
    ) -> Self {
        Self {
            option_type,
            strike,
            expiry,
            lower_barrier,
            upper_barrier,
            barrier_type,
            rebate,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier strike must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier expiry must be finite and >= 0".to_string(),
            ));
        }
        if !self.lower_barrier.is_finite() || self.lower_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier lower_barrier must be finite and > 0".to_string(),
            ));
        }
        if !self.upper_barrier.is_finite() || self.upper_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier upper_barrier must be finite and > 0".to_string(),
            ));
        }
        if self.lower_barrier >= self.upper_barrier {
            return Err(PricingError::InvalidInput(
                "double-barrier lower_barrier must be < upper_barrier".to_string(),
            ));
        }
        if !self.rebate.is_finite() || self.rebate < 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier rebate must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for DoubleBarrierOption {
    fn instrument_type(&self) -> &str {
        "DoubleBarrierOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_option() -> DoubleBarrierOption {
        DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            1.0,
            80.0,
            120.0,
            DoubleBarrierType::KnockOut,
            0.0,
        )
    }

    #[test]
    fn validate_rejects_nan_fields() {
        assert!(valid_option().validate().is_ok());

        let cases: Vec<(&str, DoubleBarrierOption)> = vec![
            ("NaN strike", {
                let mut o = valid_option();
                o.strike = f64::NAN;
                o
            }),
            ("NaN expiry", {
                let mut o = valid_option();
                o.expiry = f64::NAN;
                o
            }),
            ("NaN lower_barrier", {
                let mut o = valid_option();
                o.lower_barrier = f64::NAN;
                o
            }),
            ("NaN upper_barrier", {
                let mut o = valid_option();
                o.upper_barrier = f64::NAN;
                o
            }),
            ("NaN rebate", {
                let mut o = valid_option();
                o.rebate = f64::NAN;
                o
            }),
        ];

        for (label, option) in cases {
            assert!(option.validate().is_err(), "{label} must be rejected");
        }
    }
}
