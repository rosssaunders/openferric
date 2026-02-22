//! Module `instruments::bermudan`.
//!
//! Bermudan vanilla option with an explicit exercise-date schedule and
//! per-date strike schedule (supports step-down / step-up structures).
//!
//! References:
//! - Longstaff and Schwartz (2001), *Valuing American Options by Simulation*.
//! - Tavella and Randall (2000), finite-difference methods for free-boundary
//!   option problems.

use crate::core::{Instrument, OptionType, PricingError};

/// Bermudan option with discrete exercise rights and strike schedule.
///
/// `exercise_dates[i]` pairs with `strike_schedule[i]`.
/// If the final exercise date is strictly below expiry, engines append an
/// implicit final exercise right at expiry with the last scheduled strike.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BermudanOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Expiry in years.
    pub expiry: f64,
    /// Exercise dates in year fractions.
    pub exercise_dates: Vec<f64>,
    /// Strike corresponding to each exercise date.
    pub strike_schedule: Vec<f64>,
}

impl BermudanOption {
    /// Creates a Bermudan option with explicit per-date strikes.
    pub fn new(
        option_type: OptionType,
        expiry: f64,
        exercise_dates: Vec<f64>,
        strike_schedule: Vec<f64>,
    ) -> Self {
        Self {
            option_type,
            expiry,
            exercise_dates,
            strike_schedule,
        }
    }

    /// Creates a Bermudan option with constant strike across all exercise dates.
    pub fn with_constant_strike(
        option_type: OptionType,
        strike: f64,
        expiry: f64,
        exercise_dates: Vec<f64>,
    ) -> Self {
        let strike_schedule = vec![strike; exercise_dates.len()];
        Self::new(option_type, expiry, exercise_dates, strike_schedule)
    }

    /// Number of scheduled exercise rights.
    #[inline]
    pub fn num_exercise_dates(&self) -> usize {
        self.exercise_dates.len()
    }

    /// Returns the strike applied at a specific exercise date.
    #[inline]
    pub fn strike_at_exercise_index(&self, index: usize) -> Option<f64> {
        self.strike_schedule.get(index).copied()
    }

    /// Returns the strike active at `time` using a right-continuous step
    /// schedule induced by `exercise_dates`.
    pub fn strike_at_time(&self, time: f64) -> Result<f64, PricingError> {
        self.validate()?;
        let idx = self.exercise_dates.partition_point(|&d| d <= time);
        if idx == 0 {
            Ok(self.strike_schedule[0])
        } else {
            Ok(self.strike_schedule[idx - 1])
        }
    }

    /// Returns effective `(exercise_time, strike)` pairs.
    ///
    /// If the last exercise date is `< expiry`, an extra pair `(expiry, last_strike)`
    /// is appended so payoff-at-expiry is always well-defined.
    pub fn effective_schedule(&self) -> Result<Vec<(f64, f64)>, PricingError> {
        self.validate()?;

        let mut out = self
            .exercise_dates
            .iter()
            .copied()
            .zip(self.strike_schedule.iter().copied())
            .collect::<Vec<_>>();

        let last = *out.last().ok_or_else(|| {
            PricingError::InvalidInput("bermudan schedule cannot be empty".to_string())
        })?;
        if self.expiry - last.0 > 1.0e-12 {
            out.push((self.expiry, last.1));
        }

        Ok(out)
    }

    /// Validates schedule and strike inputs.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.expiry <= 0.0 || !self.expiry.is_finite() {
            return Err(PricingError::InvalidInput(
                "bermudan expiry must be finite and > 0".to_string(),
            ));
        }
        if self.exercise_dates.is_empty() {
            return Err(PricingError::InvalidInput(
                "bermudan exercise_dates cannot be empty".to_string(),
            ));
        }
        if self.exercise_dates.len() != self.strike_schedule.len() {
            return Err(PricingError::InvalidInput(
                "bermudan strike_schedule must match exercise_dates length".to_string(),
            ));
        }
        if self
            .exercise_dates
            .iter()
            .any(|&t| !t.is_finite() || t <= 0.0 || t > self.expiry)
        {
            return Err(PricingError::InvalidInput(
                "bermudan exercise_dates must be finite and in (0, expiry]".to_string(),
            ));
        }
        if self.exercise_dates.windows(2).any(|w| w[1] <= w[0]) {
            return Err(PricingError::InvalidInput(
                "bermudan exercise_dates must be strictly increasing".to_string(),
            ));
        }
        if self
            .strike_schedule
            .iter()
            .any(|&k| !k.is_finite() || k <= 0.0)
        {
            return Err(PricingError::InvalidInput(
                "bermudan strike_schedule entries must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Instrument for BermudanOption {
    fn instrument_type(&self) -> &str {
        "BermudanOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_expiry_with_last_strike_when_missing() {
        let b = BermudanOption::new(
            OptionType::Put,
            1.0,
            vec![0.25, 0.5, 0.75],
            vec![100.0, 98.0, 96.0],
        );
        let sch = b.effective_schedule().unwrap();
        assert_eq!(sch.len(), 4);
        assert!((sch[3].0 - 1.0).abs() < 1.0e-12);
        assert!((sch[3].1 - 96.0).abs() < 1.0e-12);
    }

    #[test]
    fn strike_at_time_is_stepwise() {
        let b = BermudanOption::new(
            OptionType::Put,
            1.0,
            vec![0.25, 0.5, 1.0],
            vec![100.0, 95.0, 90.0],
        );
        assert_eq!(b.strike_at_time(0.10).unwrap(), 100.0);
        assert_eq!(b.strike_at_time(0.40).unwrap(), 100.0);
        assert_eq!(b.strike_at_time(0.75).unwrap(), 95.0);
        assert_eq!(b.strike_at_time(1.00).unwrap(), 90.0);
    }
}
