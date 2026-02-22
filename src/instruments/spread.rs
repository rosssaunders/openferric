use crate::core::{Instrument, PricingError};

/// Two-asset spread option input bundle.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpreadOption {
    pub s1: f64,
    pub s2: f64,
    pub k: f64,
    pub vol1: f64,
    pub vol2: f64,
    pub rho: f64,
    pub q1: f64,
    pub q2: f64,
    pub r: f64,
    pub t: f64,
}

impl SpreadOption {
    /// Validates spread option fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.s1 <= 0.0 || self.s2 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "spread spots s1 and s2 must be > 0".to_string(),
            ));
        }
        if self.k < 0.0 {
            return Err(PricingError::InvalidInput(
                "spread strike k must be >= 0".to_string(),
            ));
        }
        if self.vol1 < 0.0 || self.vol2 < 0.0 {
            return Err(PricingError::InvalidInput(
                "spread volatilities vol1 and vol2 must be >= 0".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(PricingError::InvalidInput(
                "spread correlation rho must be in [-1, 1]".to_string(),
            ));
        }
        if self.t < 0.0 {
            return Err(PricingError::InvalidInput(
                "spread maturity t must be >= 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Effective Margrabe volatility `sqrt(vol1^2 - 2*rho*vol1*vol2 + vol2^2)`.
    pub fn effective_volatility(&self) -> Result<f64, PricingError> {
        let variance =
            self.vol1 * self.vol1 - 2.0 * self.rho * self.vol1 * self.vol2 + self.vol2 * self.vol2;

        if variance < -1.0e-14 {
            return Err(PricingError::InvalidInput(
                "spread effective variance is negative".to_string(),
            ));
        }

        Ok(variance.max(0.0).sqrt())
    }
}

impl Instrument for SpreadOption {
    fn instrument_type(&self) -> &str {
        "SpreadOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn effective_volatility_matches_hand_calculation() {
        let option = SpreadOption {
            s1: 100.0,
            s2: 105.0,
            k: 0.0,
            vol1: 0.20,
            vol2: 0.15,
            rho: 0.5,
            q1: 0.04,
            q2: 0.06,
            r: 0.05,
            t: 1.0,
        };

        let sigma = option.effective_volatility().expect("valid effective vol");
        assert_relative_eq!(sigma, 0.180_277_563_8, epsilon = 1e-10);
    }
}
