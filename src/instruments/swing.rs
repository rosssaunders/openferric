//! Instrument definition for Swing contracts.
//!
//! Module openferric::instruments::swing contains payoff parameters and validation logic.

use crate::core::{Instrument, PricingError};

/// Multi-exercise swing option commonly used in energy contracts.
#[derive(Debug, Clone, PartialEq)]
pub struct SwingOption {
    /// Minimum number of exercise rights that must be used.
    pub min_exercises: usize,
    /// Maximum number of exercise rights that can be used.
    pub max_exercises: usize,
    /// Allowed exercise dates in year fractions.
    pub exercise_dates: Vec<f64>,
    /// Strike level for each exercise.
    pub strike: f64,
    /// Quantity multiplier paid per exercise.
    pub payoff_per_exercise: f64,
}

impl SwingOption {
    /// Creates a new swing option.
    pub fn new(
        min_exercises: usize,
        max_exercises: usize,
        exercise_dates: Vec<f64>,
        strike: f64,
        payoff_per_exercise: f64,
    ) -> Self {
        Self {
            min_exercises,
            max_exercises,
            exercise_dates,
            strike,
            payoff_per_exercise,
        }
    }

    /// Validates swing option terms.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.exercise_dates.is_empty() {
            return Err(PricingError::InvalidInput(
                "swing exercise_dates cannot be empty".to_string(),
            ));
        }
        if self.max_exercises == 0 {
            return Err(PricingError::InvalidInput(
                "swing max_exercises must be > 0".to_string(),
            ));
        }
        if self.min_exercises > self.max_exercises {
            return Err(PricingError::InvalidInput(
                "swing min_exercises must be <= max_exercises".to_string(),
            ));
        }
        if self.max_exercises > self.exercise_dates.len() {
            return Err(PricingError::InvalidInput(
                "swing max_exercises cannot exceed number of exercise_dates".to_string(),
            ));
        }
        if self
            .exercise_dates
            .iter()
            .any(|&t| !t.is_finite() || t <= 0.0)
        {
            return Err(PricingError::InvalidInput(
                "swing exercise_dates must be finite and > 0".to_string(),
            ));
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "swing strike must be finite and > 0".to_string(),
            ));
        }
        if !self.payoff_per_exercise.is_finite() || self.payoff_per_exercise < 0.0 {
            return Err(PricingError::InvalidInput(
                "swing payoff_per_exercise must be finite and >= 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for SwingOption {
    fn instrument_type(&self) -> &str {
        "SwingOption"
    }
}
