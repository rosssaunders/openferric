//! Module `instruments::convertible`.
//!
//! Implements convertible abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `ConvertibleBond` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, PricingError};

/// Convertible bond with optional issuer call and holder put features.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    ///
    /// Engines model this as a flat call level exercisable at every time step
    /// before maturity (no call-protection schedule). A value below
    /// `face_value` therefore lets the issuer redeem below par at any time;
    /// it never caps the redemption of an already-matured bond.
    pub call_price: Option<f64>,
    /// Optional holder put floor.
    ///
    /// Modeled as a flat put level exercisable at every time step (no put
    /// schedule).
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
        if !self.face_value.is_finite() || self.face_value <= 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible face_value must be finite and > 0".to_string(),
            ));
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible coupon_rate must be finite and >= 0".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible maturity must be finite and >= 0".to_string(),
            ));
        }
        if !self.conversion_ratio.is_finite() || self.conversion_ratio < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible conversion_ratio must be finite and >= 0".to_string(),
            ));
        }
        if self.call_price.is_some_and(|x| !x.is_finite() || x <= 0.0) {
            return Err(PricingError::InvalidInput(
                "convertible call_price must be finite and > 0 when provided".to_string(),
            ));
        }
        if self.put_price.is_some_and(|x| !x.is_finite() || x <= 0.0) {
            return Err(PricingError::InvalidInput(
                "convertible put_price must be finite and > 0 when provided".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_bond() -> ConvertibleBond {
        ConvertibleBond::new(100.0, 0.05, 5.0, 2.0, Some(110.0), Some(90.0))
    }

    #[test]
    fn validate_rejects_nan_fields() {
        assert!(valid_bond().validate().is_ok());

        let cases: Vec<(&str, ConvertibleBond)> = vec![
            ("NaN face_value", {
                let mut b = valid_bond();
                b.face_value = f64::NAN;
                b
            }),
            ("NaN coupon_rate", {
                let mut b = valid_bond();
                b.coupon_rate = f64::NAN;
                b
            }),
            ("NaN maturity", {
                let mut b = valid_bond();
                b.maturity = f64::NAN;
                b
            }),
            ("NaN conversion_ratio", {
                let mut b = valid_bond();
                b.conversion_ratio = f64::NAN;
                b
            }),
            ("NaN call_price", {
                let mut b = valid_bond();
                b.call_price = Some(f64::NAN);
                b
            }),
            ("NaN put_price", {
                let mut b = valid_bond();
                b.put_price = Some(f64::NAN);
                b
            }),
        ];

        for (label, bond) in cases {
            assert!(bond.validate().is_err(), "{label} must be rejected");
        }
    }
}
