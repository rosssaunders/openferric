use crate::core::{ExerciseStyle, Instrument, OptionType, PricingError};

/// Vanilla option instrument.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VanillaOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike level.
    pub strike: f64,
    /// Expiry in years.
    pub expiry: f64,
    /// Exercise style.
    pub exercise: ExerciseStyle,
}

impl VanillaOption {
    /// Builds a European call option.
    pub fn european_call(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Call,
            strike,
            expiry,
            exercise: ExerciseStyle::European,
        }
    }

    /// Builds a European put option.
    pub fn european_put(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Put,
            strike,
            expiry,
            exercise: ExerciseStyle::European,
        }
    }

    /// Builds an American call option.
    pub fn american_call(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Call,
            strike,
            expiry,
            exercise: ExerciseStyle::American,
        }
    }

    /// Builds an American put option.
    pub fn american_put(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Put,
            strike,
            expiry,
            exercise: ExerciseStyle::American,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "vanilla strike must be > 0".to_string(),
            ));
        }
        if self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "vanilla expiry must be >= 0".to_string(),
            ));
        }

        if let ExerciseStyle::Bermudan { dates } = &self.exercise {
            if dates.is_empty() {
                return Err(PricingError::InvalidInput(
                    "bermudan exercise dates cannot be empty".to_string(),
                ));
            }
            if dates.iter().any(|&d| d <= 0.0 || d > self.expiry) {
                return Err(PricingError::InvalidInput(
                    "bermudan exercise dates must lie in (0, expiry]".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Instrument for VanillaOption {
    fn instrument_type(&self) -> &str {
        "VanillaOption"
    }
}
