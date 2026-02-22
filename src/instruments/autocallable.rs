//! Data models for worst-of autocallable and Phoenix-style structured notes.
//!
//! [`Autocallable`] encodes call dates, autocall/knock-in barriers, coupon rate, and underlyings,
//! while [`PhoenixAutocallable`] extends this with coupon barrier and memory-coupon behavior.
//! References: common equity/FX autocall term-sheet design (for example Wystup, 2017).
//! Validation checks strictly increasing observation dates, positive barriers/notional,
//! and unique underlying indices to prevent ambiguous path state updates.
//! This module is contract-definition only; Monte Carlo payoff engines should implement the
//! path-dependent redemption and coupon logic on top of these validated parameters.
//! Use this when structuring callable retail-note style products.

use std::collections::BTreeSet;

use crate::core::{Instrument, PricingError};

/// Worst-of autocallable note with knock-in downside at maturity.
#[derive(Debug, Clone, PartialEq)]
pub struct Autocallable {
    /// Indices into the global spot/vol vectors.
    pub underlyings: Vec<usize>,
    pub notional: f64,
    /// Observation dates in years.
    pub autocall_dates: Vec<f64>,
    /// Autocall trigger on worst-of ratio.
    pub autocall_barrier: f64,
    /// Annual coupon rate used for accrued/final coupon.
    pub coupon_rate: f64,
    /// Knock-in barrier on running worst-of ratio.
    pub ki_barrier: f64,
    /// Knock-in strike on final worst-of ratio.
    pub ki_strike: f64,
    pub maturity: f64,
}

impl Autocallable {
    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.underlyings.is_empty() {
            return Err(PricingError::InvalidInput(
                "autocallable underlyings cannot be empty".to_string(),
            ));
        }
        if self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable notional must be > 0".to_string(),
            ));
        }
        if self.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable maturity must be > 0".to_string(),
            ));
        }
        if self.autocall_dates.is_empty() {
            return Err(PricingError::InvalidInput(
                "autocallable autocall_dates cannot be empty".to_string(),
            ));
        }
        if self.coupon_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable coupon_rate must be >= 0".to_string(),
            ));
        }
        if self.autocall_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable autocall_barrier must be > 0".to_string(),
            ));
        }
        if self.ki_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable ki_barrier must be > 0".to_string(),
            ));
        }
        if self.ki_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable ki_strike must be > 0".to_string(),
            ));
        }

        if self
            .autocall_dates
            .iter()
            .any(|&t| t <= 0.0 || t > self.maturity)
        {
            return Err(PricingError::InvalidInput(
                "autocallable dates must lie in (0, maturity]".to_string(),
            ));
        }
        if self.autocall_dates.windows(2).any(|w| w[1] <= w[0]) {
            return Err(PricingError::InvalidInput(
                "autocallable dates must be strictly increasing".to_string(),
            ));
        }

        let unique = self.underlyings.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != self.underlyings.len() {
            return Err(PricingError::InvalidInput(
                "autocallable underlyings must be unique".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for Autocallable {
    fn instrument_type(&self) -> &str {
        "Autocallable"
    }
}

/// Phoenix-style autocallable with coupon barrier and optional memory.
#[derive(Debug, Clone, PartialEq)]
pub struct PhoenixAutocallable {
    /// Indices into the global spot/vol vectors.
    pub underlyings: Vec<usize>,
    pub notional: f64,
    /// Observation dates in years.
    pub autocall_dates: Vec<f64>,
    /// Autocall trigger on worst-of ratio.
    pub autocall_barrier: f64,
    /// Coupon trigger on worst-of ratio.
    pub coupon_barrier: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Whether missed coupons are remembered and paid later.
    pub memory: bool,
    /// Knock-in barrier on running worst-of ratio.
    pub ki_barrier: f64,
    /// Knock-in strike on final worst-of ratio.
    pub ki_strike: f64,
    pub maturity: f64,
}

impl PhoenixAutocallable {
    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        let base = Autocallable {
            underlyings: self.underlyings.clone(),
            notional: self.notional,
            autocall_dates: self.autocall_dates.clone(),
            autocall_barrier: self.autocall_barrier,
            coupon_rate: self.coupon_rate,
            ki_barrier: self.ki_barrier,
            ki_strike: self.ki_strike,
            maturity: self.maturity,
        };
        base.validate()?;

        if self.coupon_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "phoenix coupon_barrier must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for PhoenixAutocallable {
    fn instrument_type(&self) -> &str {
        "PhoenixAutocallable"
    }
}
