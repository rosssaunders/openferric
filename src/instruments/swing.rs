//! Module `instruments::swing`.
//!
//! Implements swing abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `SwingOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, PricingError};

/// Multi-exercise swing option commonly used in energy contracts.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
        if self.exercise_dates.windows(2).any(|w| w[1] <= w[0]) {
            return Err(PricingError::InvalidInput(
                "swing exercise_dates must be strictly increasing".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn option() -> SwingOption {
        SwingOption::new(1, 2, vec![0.25, 0.50, 1.0], 100.0, 10.0)
    }

    fn assert_invalid(option: &SwingOption, message: &str) {
        assert_eq!(
            option.validate(),
            Err(PricingError::InvalidInput(message.to_string()))
        );
    }

    #[test]
    fn constructor_preserves_terms_and_serialization_shape() {
        let swing = option();
        assert_eq!(swing.min_exercises, 1);
        assert_eq!(swing.max_exercises, 2);
        assert_eq!(swing.exercise_dates, vec![0.25, 0.50, 1.0]);
        assert_eq!(swing.strike, 100.0);
        assert_eq!(swing.payoff_per_exercise, 10.0);
        assert_eq!(swing.validate(), Ok(()));
        assert_eq!(swing.instrument_type(), "SwingOption");

        let value = serde_json::to_value(&swing).expect("serialize swing option");
        assert_eq!(value["min_exercises"], 1);
        assert_eq!(value["max_exercises"], 2);
        assert_eq!(value["exercise_dates"], serde_json::json!([0.25, 0.5, 1.0]));
        assert_eq!(value["strike"], 100.0);
        assert_eq!(value["payoff_per_exercise"], 10.0);
        assert_eq!(
            serde_json::from_value::<SwingOption>(value).expect("deserialize swing option"),
            swing
        );
    }

    #[test]
    fn validation_accepts_optional_minimum_and_zero_payoff_boundaries() {
        let swing = SwingOption::new(0, 3, vec![0.25, 0.50, 1.0], 0.01, 0.0);
        assert_eq!(swing.validate(), Ok(()));
    }

    #[test]
    fn validation_rejects_invalid_rights_and_schedule_shape() {
        const EMPTY_ERROR: &str = "swing exercise_dates cannot be empty";
        const MAX_ZERO_ERROR: &str = "swing max_exercises must be > 0";
        const MIN_MAX_ERROR: &str = "swing min_exercises must be <= max_exercises";
        const TOO_MANY_ERROR: &str = "swing max_exercises cannot exceed number of exercise_dates";
        const DATE_ERROR: &str = "swing exercise_dates must be finite and > 0";
        const ORDER_ERROR: &str = "swing exercise_dates must be strictly increasing";

        assert_invalid(
            &SwingOption {
                exercise_dates: vec![],
                ..option()
            },
            EMPTY_ERROR,
        );
        assert_invalid(
            &SwingOption {
                min_exercises: 0,
                max_exercises: 0,
                ..option()
            },
            MAX_ZERO_ERROR,
        );
        assert_invalid(
            &SwingOption {
                min_exercises: 3,
                max_exercises: 2,
                ..option()
            },
            MIN_MAX_ERROR,
        );
        assert_invalid(
            &SwingOption {
                max_exercises: 4,
                ..option()
            },
            TOO_MANY_ERROR,
        );

        for exercise_dates in [
            vec![0.0, 0.5, 1.0],
            vec![0.25, f64::NAN, 1.0],
            vec![0.25, 0.5, f64::INFINITY],
        ] {
            assert_invalid(
                &SwingOption {
                    exercise_dates,
                    ..option()
                },
                DATE_ERROR,
            );
        }
        for exercise_dates in [vec![0.25, 0.25, 1.0], vec![0.50, 0.25, 1.0]] {
            assert_invalid(
                &SwingOption {
                    exercise_dates,
                    ..option()
                },
                ORDER_ERROR,
            );
        }
    }

    #[test]
    fn validation_rejects_invalid_strike_and_payoff() {
        const STRIKE_ERROR: &str = "swing strike must be finite and > 0";
        const PAYOFF_ERROR: &str = "swing payoff_per_exercise must be finite and >= 0";

        for strike in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_invalid(&SwingOption { strike, ..option() }, STRIKE_ERROR);
        }
        for payoff_per_exercise in [-f64::EPSILON, f64::NAN, f64::INFINITY] {
            assert_invalid(
                &SwingOption {
                    payoff_per_exercise,
                    ..option()
                },
                PAYOFF_ERROR,
            );
        }
    }
}
