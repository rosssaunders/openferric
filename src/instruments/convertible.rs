//! Convertible-bond contract schema (no valuation engine in this module).
//!
//! [`ConvertibleBond`] captures face value, coupon, maturity, conversion ratio,
//! and optional issuer call / holder put provisions.
//! References: Tsiveriotis and Fernandes (1998), standard convertible-bond treatment in Hull.
//! Validation enforces sign and positivity constraints on contractual fields.
//! Numerical and exercise-feature interactions (credit/equity coupling, call triggers,
//! soft-call constraints) are intentionally out of scope here and belong in dedicated pricers.
//! Use this type as the canonical input object for lattice/PDE convertible engines.

use crate::core::{Instrument, PricingError};

/// Convertible bond with optional issuer call and holder put features.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertibleBond {
    /// Notional/face amount.
    pub face_value: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Maturity in years.
    pub maturity: f64,
    /// Shares received per bond when converted.
    pub conversion_ratio: f64,
    /// Optional issuer call price cap.
    pub call_price: Option<f64>,
    /// Optional holder put floor.
    pub put_price: Option<f64>,
}

impl ConvertibleBond {
    /// Creates a new convertible bond.
    pub fn new(
        face_value: f64,
        coupon_rate: f64,
        maturity: f64,
        conversion_ratio: f64,
        call_price: Option<f64>,
        put_price: Option<f64>,
    ) -> Self {
        Self {
            face_value,
            coupon_rate,
            maturity,
            conversion_ratio,
            call_price,
            put_price,
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.face_value <= 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible face_value must be > 0".to_string(),
            ));
        }
        if self.coupon_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible coupon_rate must be >= 0".to_string(),
            ));
        }
        if self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible maturity must be >= 0".to_string(),
            ));
        }
        if self.conversion_ratio < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible conversion_ratio must be >= 0".to_string(),
            ));
        }
        if self.call_price.is_some_and(|x| x <= 0.0) {
            return Err(PricingError::InvalidInput(
                "convertible call_price must be > 0 when provided".to_string(),
            ));
        }
        if self.put_price.is_some_and(|x| x <= 0.0) {
            return Err(PricingError::InvalidInput(
                "convertible put_price must be > 0 when provided".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for ConvertibleBond {
    fn instrument_type(&self) -> &str {
        "ConvertibleBond"
    }
}
