//! Module `instruments::cliquet`.
//!
//! Implements cliquet abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `ForwardStartOption`, `CliquetOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};
use crate::math::normal_cdf;

/// Forward-start option with strike set as a multiple of spot at start time.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForwardStartOption {
    pub option_type: OptionType,
    pub spot: f64,
    /// Strike multiple `alpha`, so strike at reset is `alpha * S(t_start)`.
    pub strike_ratio: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub vol: f64,
    pub t_start: f64,
    pub expiry: f64,
}

/// Cliquet option alias for a single-reset forward-start contract.
pub type CliquetOption = ForwardStartOption;

impl ForwardStartOption {
    /// Builds an ATM forward-start call (`strike_ratio = 1.0`).
    pub fn atm_call(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        t_start: f64,
        expiry: f64,
    ) -> Self {
        Self {
            option_type: OptionType::Call,
            spot,
            strike_ratio: 1.0,
            rate,
            dividend_yield,
            vol,
            t_start,
            expiry,
        }
    }

    /// Validates contract fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "forward-start spot must be > 0".to_string(),
            ));
        }
        if self.strike_ratio <= 0.0 {
            return Err(PricingError::InvalidInput(
                "forward-start strike_ratio must be > 0".to_string(),
            ));
        }
        if self.vol < 0.0 {
            return Err(PricingError::InvalidInput(
                "forward-start volatility must be >= 0".to_string(),
            ));
        }
        if self.t_start < 0.0 || self.expiry < 0.0 {
            return Err(PricingError::InvalidInput(
                "forward-start times must be >= 0".to_string(),
            ));
        }
        if self.t_start > self.expiry {
            return Err(PricingError::InvalidInput(
                "forward-start requires t_start <= expiry".to_string(),
            ));
        }

        Ok(())
    }

    /// Rubinstein (1991) forward-start option formula.
    pub fn price_rubinstein(&self) -> Result<f64, PricingError> {
        self.validate()?;

        let tau = self.expiry - self.t_start;
        let scale = self.spot * (-self.dividend_yield * self.t_start).exp();

        if tau <= 0.0 {
            return Ok(scale
                * match self.option_type {
                    OptionType::Call => (1.0 - self.strike_ratio).max(0.0),
                    OptionType::Put => (self.strike_ratio - 1.0).max(0.0),
                });
        }

        if self.vol <= 0.0 {
            let growth = ((self.rate - self.dividend_yield) * tau).exp();
            let deterministic_forward = match self.option_type {
                OptionType::Call => (growth - self.strike_ratio).max(0.0),
                OptionType::Put => (self.strike_ratio - growth).max(0.0),
            };
            return Ok(scale * (-self.rate * tau).exp() * deterministic_forward);
        }

        let sqrt_tau = tau.sqrt();
        let sig_sqrt_tau = self.vol * sqrt_tau;
        let d1 = ((1.0 / self.strike_ratio).ln()
            + (self.rate - self.dividend_yield + 0.5 * self.vol * self.vol) * tau)
            / sig_sqrt_tau;
        let d2 = d1 - sig_sqrt_tau;

        let unit_price = match self.option_type {
            OptionType::Call => {
                (-self.dividend_yield * tau).exp() * normal_cdf(d1)
                    - self.strike_ratio * (-self.rate * tau).exp() * normal_cdf(d2)
            }
            OptionType::Put => {
                self.strike_ratio * (-self.rate * tau).exp() * normal_cdf(-d2)
                    - (-self.dividend_yield * tau).exp() * normal_cdf(-d1)
            }
        };

        Ok(scale * unit_price)
    }
}

impl Instrument for ForwardStartOption {
    fn instrument_type(&self) -> &str {
        "ForwardStartOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn forward_start_atm_call_haug_reference_case() {
        let option = ForwardStartOption::atm_call(100.0, 0.08, 0.04, 0.30, 0.25, 1.0);
        let price = option.price_rubinstein().expect("pricing succeeds");

        // Independently evaluated with SciPy 1.17.1's normal CDF using the
        // Rubinstein formula and these exact year fractions.  The previous
        // 10.3290 target used a different convention and only passed because
        // its tolerance was an entire currency unit.
        assert_relative_eq!(price, 11.268_293_381_170_055, epsilon = 2.0e-12);
    }

    #[test]
    fn forward_start_call_matches_quantlib_haug_reference() {
        // QuantLib test-suite/cliquetoption.cpp retains Haug's rounded 4.4064
        // value (p.37).  The full target below is the NPV independently
        // generated by QuantLib 1.43's AnalyticCliquetEngine using its exact
        // Actual/360 reset and maturity dates (90 and 360 days).
        let option = ForwardStartOption {
            option_type: OptionType::Call,
            spot: 60.0,
            strike_ratio: 1.1,
            rate: 0.08,
            dividend_yield: 0.04,
            vol: 0.30,
            t_start: 0.25,
            expiry: 1.0,
        };
        let price = option.price_rubinstein().expect("pricing succeeds");

        assert_relative_eq!(price, 4.406_454_339_365_002, epsilon = 2.0e-12);
    }

    #[test]
    fn forward_start_reduces_to_ratio_intrinsic_at_zero_vol() {
        let option = ForwardStartOption {
            option_type: OptionType::Call,
            spot: 100.0,
            strike_ratio: 1.0,
            rate: 0.03,
            dividend_yield: 0.01,
            vol: 0.0,
            t_start: 0.5,
            expiry: 1.0,
        };

        let price = option.price_rubinstein().unwrap();

        let tau = option.expiry - option.t_start;
        let expected = option.spot
            * (-option.dividend_yield * option.t_start).exp()
            * (-option.rate * tau).exp()
            * (((option.rate - option.dividend_yield) * tau).exp() - option.strike_ratio).max(0.0);

        assert_eq!(price.to_bits(), expected.to_bits());
    }
}
