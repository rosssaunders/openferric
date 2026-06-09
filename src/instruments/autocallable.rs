//! Module `instruments::autocallable`.
//!
//! Implements autocallable abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `Autocallable`, `PhoenixAutocallable` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use std::collections::BTreeSet;

use crate::core::{Instrument, PricingError};

/// Worst-of autocallable note with knock-in downside at maturity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable notional must be finite and > 0".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable maturity must be finite and > 0".to_string(),
            ));
        }
        if self.autocall_dates.is_empty() {
            return Err(PricingError::InvalidInput(
                "autocallable autocall_dates cannot be empty".to_string(),
            ));
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable coupon_rate must be finite and >= 0".to_string(),
            ));
        }
        if !self.autocall_barrier.is_finite() || self.autocall_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable autocall_barrier must be finite and > 0".to_string(),
            ));
        }
        if !self.ki_barrier.is_finite() || self.ki_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable ki_barrier must be finite and > 0".to_string(),
            ));
        }
        if !self.ki_strike.is_finite() || self.ki_strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "autocallable ki_strike must be finite and > 0".to_string(),
            ));
        }

        if self
            .autocall_dates
            .iter()
            .any(|&t| !t.is_finite() || t <= 0.0 || t > self.maturity)
        {
            return Err(PricingError::InvalidInput(
                "autocallable dates must be finite and lie in (0, maturity]".to_string(),
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

        if !self.coupon_barrier.is_finite() || self.coupon_barrier <= 0.0 {
            return Err(PricingError::InvalidInput(
                "phoenix coupon_barrier must be finite and > 0".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_note() -> Autocallable {
        Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 1.0,
            coupon_rate: 0.08,
            ki_barrier: 0.6,
            ki_strike: 1.0,
            maturity: 1.0,
        }
    }

    fn valid_phoenix() -> PhoenixAutocallable {
        PhoenixAutocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 1.0,
            coupon_barrier: 0.7,
            coupon_rate: 0.08,
            memory: true,
            ki_barrier: 0.6,
            ki_strike: 1.0,
            maturity: 1.0,
        }
    }

    #[test]
    fn autocallable_validate_rejects_nan_fields() {
        assert!(valid_note().validate().is_ok());

        let cases: Vec<(&str, Autocallable)> = vec![
            ("NaN notional", {
                let mut n = valid_note();
                n.notional = f64::NAN;
                n
            }),
            ("NaN maturity", {
                let mut n = valid_note();
                n.maturity = f64::NAN;
                n
            }),
            ("NaN coupon_rate", {
                let mut n = valid_note();
                n.coupon_rate = f64::NAN;
                n
            }),
            ("NaN autocall_barrier", {
                let mut n = valid_note();
                n.autocall_barrier = f64::NAN;
                n
            }),
            ("NaN ki_barrier", {
                let mut n = valid_note();
                n.ki_barrier = f64::NAN;
                n
            }),
            ("NaN ki_strike", {
                let mut n = valid_note();
                n.ki_strike = f64::NAN;
                n
            }),
            ("NaN autocall date", {
                let mut n = valid_note();
                n.autocall_dates[1] = f64::NAN;
                n
            }),
        ];

        for (label, note) in cases {
            assert!(note.validate().is_err(), "{label} must be rejected");
        }
    }

    #[test]
    fn phoenix_validate_rejects_nan_coupon_barrier() {
        assert!(valid_phoenix().validate().is_ok());

        let mut p = valid_phoenix();
        p.coupon_barrier = f64::NAN;
        assert!(p.validate().is_err(), "NaN coupon_barrier must be rejected");
    }
}
