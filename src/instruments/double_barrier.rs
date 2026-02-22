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
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier strike must be > 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier expiry must be >= 0".to_string(),
            ));
        }
        if self.lower_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier lower_barrier must be > 0".to_string(),
            ));
        }
        if self.upper_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier upper_barrier must be > 0".to_string(),
            ));
        }
        if self.lower_barrier >= self.upper_barrier {
            return Err(PricingError::InvalidInput(
                "double-barrier lower_barrier must be < upper_barrier".to_string(),
            ));
        }
        if self.rebate < 0.0 {
            return Err(PricingError::InvalidInput(
                "double-barrier rebate must be >= 0".to_string(),
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
