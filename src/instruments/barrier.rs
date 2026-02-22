//! Single-barrier option schema plus ergonomic builder for knock-in/out variants.
//!
//! [`BarrierOption`] couples vanilla fields with [`crate::core::BarrierSpec`], and
//! [`BarrierOptionBuilder`] provides explicit constructors (`up_and_out`, `down_and_in`, etc.)
//! to reduce invalid state combinations at call sites.
//! References: Reiner and Rubinstein (1991); Haug (2007), barrier-option formulas.
//! Validation enforces positive strike/barrier level and non-negative rebate.
//! Path-hit semantics are evaluated by pricing engines (for example Monte Carlo barrier checks);
//! this module only defines contract state and input constraints.
//! Prefer this type over ad hoc tuples when barriers and rebates must remain explicit.

use crate::core::{
    BarrierDirection, BarrierSpec, BarrierStyle, Instrument, OptionType, PricingError,
};

/// Barrier option instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct BarrierOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike level.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Barrier specification.
    pub barrier: BarrierSpec,
}

impl BarrierOption {
    /// Starts a barrier option builder.
    pub fn builder() -> BarrierOptionBuilder {
        BarrierOptionBuilder::default()
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "barrier strike must be > 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "barrier expiry must be >= 0".to_string(),
            ));
        }
        if self.barrier.level <= 0.0 {
            return Err(PricingError::InvalidInput(
                "barrier level must be > 0".to_string(),
            ));
        }
        if self.barrier.rebate < 0.0 {
            return Err(PricingError::InvalidInput(
                "barrier rebate must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for BarrierOption {
    fn instrument_type(&self) -> &str {
        "BarrierOption"
    }
}

/// Builder for [`BarrierOption`].
#[derive(Debug, Clone, Default)]
pub struct BarrierOptionBuilder {
    option_type: Option<OptionType>,
    strike: Option<f64>,
    expiry: Option<f64>,
    direction: Option<BarrierDirection>,
    style: Option<BarrierStyle>,
    level: Option<f64>,
    rebate: Option<f64>,
}

impl BarrierOptionBuilder {
    /// Sets option side to call.
    pub fn call(mut self) -> Self {
        self.option_type = Some(OptionType::Call);
        self
    }

    /// Sets option side to put.
    pub fn put(mut self) -> Self {
        self.option_type = Some(OptionType::Put);
        self
    }

    /// Sets strike.
    pub fn strike(mut self, strike: f64) -> Self {
        self.strike = Some(strike);
        self
    }

    /// Sets expiry in years.
    pub fn expiry(mut self, expiry: f64) -> Self {
        self.expiry = Some(expiry);
        self
    }

    /// Sets up-and-out barrier.
    pub fn up_and_out(mut self, level: f64) -> Self {
        self.direction = Some(BarrierDirection::Up);
        self.style = Some(BarrierStyle::Out);
        self.level = Some(level);
        self
    }

    /// Sets up-and-in barrier.
    pub fn up_and_in(mut self, level: f64) -> Self {
        self.direction = Some(BarrierDirection::Up);
        self.style = Some(BarrierStyle::In);
        self.level = Some(level);
        self
    }

    /// Sets down-and-out barrier.
    pub fn down_and_out(mut self, level: f64) -> Self {
        self.direction = Some(BarrierDirection::Down);
        self.style = Some(BarrierStyle::Out);
        self.level = Some(level);
        self
    }

    /// Sets down-and-in barrier.
    pub fn down_and_in(mut self, level: f64) -> Self {
        self.direction = Some(BarrierDirection::Down);
        self.style = Some(BarrierStyle::In);
        self.level = Some(level);
        self
    }

    /// Sets cash rebate.
    pub fn rebate(mut self, rebate: f64) -> Self {
        self.rebate = Some(rebate);
        self
    }

    /// Validates and builds a barrier option.
    pub fn build(self) -> Result<BarrierOption, PricingError> {
        let option_type = self.option_type.ok_or_else(|| {
            PricingError::InvalidInput("barrier option type is required".to_string())
        })?;
        let strike = self
            .strike
            .ok_or_else(|| PricingError::InvalidInput("barrier strike is required".to_string()))?;
        let expiry = self
            .expiry
            .ok_or_else(|| PricingError::InvalidInput("barrier expiry is required".to_string()))?;
        let direction = self.direction.ok_or_else(|| {
            PricingError::InvalidInput("barrier direction is required".to_string())
        })?;
        let style = self
            .style
            .ok_or_else(|| PricingError::InvalidInput("barrier style is required".to_string()))?;
        let level = self
            .level
            .ok_or_else(|| PricingError::InvalidInput("barrier level is required".to_string()))?;
        let rebate = self.rebate.unwrap_or(0.0);

        let option = BarrierOption {
            option_type,
            strike,
            expiry,
            barrier: BarrierSpec {
                direction,
                style,
                level,
                rebate,
            },
        };
        option.validate()?;
        Ok(option)
    }
}
