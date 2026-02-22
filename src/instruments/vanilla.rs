//! Canonical plain-vanilla option contract definition used throughout the library.
//!
//! [`VanillaOption`] stores side, strike, expiry, and exercise rights
//! ([`crate::core::ExerciseStyle`]: European/American/Bermudan).
//! References: Hull (2018), Ch. 10-13 for payoff and exercise conventions.
//! Validation accepts `expiry == 0` (intrinsic-value edge case) and enforces
//! Bermudan-date consistency within `(0, expiry]`.
//! This type is the default input for Black-Scholes, lattice, PDE, and Monte Carlo engines.
//! Use this module unless a product requires explicit path dependence or additional state.

use crate::core::{ExerciseStyle, Instrument, OptionType, PricingError};

/// Vanilla option contract.
///
/// This is the canonical input for Black-Scholes/Black-76 style engines:
/// strike `K`, expiry `T`, option side, and exercise rights.
///
/// # Examples
/// ```
/// use openferric::core::{ExerciseStyle, OptionType};
/// use openferric::instruments::VanillaOption;
///
/// let option = VanillaOption {
///     option_type: OptionType::Call,
///     strike: 100.0,
///     expiry: 1.0,
///     exercise: ExerciseStyle::European,
/// };
/// assert!(option.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
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
    ///
    /// `strike` and `expiry` are interpreted in spot units and year fractions.
    ///
    /// # Examples
    /// ```
    /// use openferric::core::{ExerciseStyle, OptionType};
    /// use openferric::instruments::VanillaOption;
    ///
    /// let call = VanillaOption::european_call(100.0, 1.0);
    /// assert_eq!(call.option_type, OptionType::Call);
    /// assert!(matches!(call.exercise, ExerciseStyle::European));
    /// ```
    pub fn european_call(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Call,
            strike,
            expiry,
            exercise: ExerciseStyle::European,
        }
    }

    /// Builds a European put option.
    ///
    /// # Examples
    /// ```
    /// use openferric::core::OptionType;
    /// use openferric::instruments::VanillaOption;
    ///
    /// let put = VanillaOption::european_put(95.0, 0.5);
    /// assert_eq!(put.option_type, OptionType::Put);
    /// ```
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
    ///
    /// # Examples
    /// ```
    /// use openferric::core::ExerciseStyle;
    /// use openferric::instruments::VanillaOption;
    ///
    /// let put = VanillaOption::american_put(100.0, 2.0);
    /// assert!(matches!(put.exercise, ExerciseStyle::American));
    /// ```
    pub fn american_put(strike: f64, expiry: f64) -> Self {
        Self {
            option_type: OptionType::Put,
            strike,
            expiry,
            exercise: ExerciseStyle::American,
        }
    }

    /// Validates instrument fields.
    ///
    /// # Errors
    /// Returns [`PricingError::InvalidInput`] when:
    /// - `strike <= 0`
    /// - `expiry < 0`
    /// - Bermudan exercise dates are empty or outside `(0, expiry]`
    ///
    /// # Numerical notes
    /// `expiry == 0` is accepted to support immediate-expiry intrinsic-value pricing.
    ///
    /// # Examples
    /// ```
    /// use openferric::core::{ExerciseStyle, OptionType};
    /// use openferric::instruments::VanillaOption;
    ///
    /// let bermudan = VanillaOption {
    ///     option_type: OptionType::Call,
    ///     strike: 100.0,
    ///     expiry: 1.0,
    ///     exercise: ExerciseStyle::Bermudan { dates: vec![0.5, 1.0] },
    /// };
    /// assert!(bermudan.validate().is_ok());
    /// ```
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
