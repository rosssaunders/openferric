use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::spread::SpreadOption;
use crate::market::Market;
use crate::math::normal_cdf;

/// Pricing method for spread options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadAnalyticMethod {
    /// Margrabe exchange option (best suited for `k = 0`).
    Margrabe,
    /// Kirk approximation for `max(S1 - S2 - K, 0)`.
    Kirk,
}

/// Analytic spread option engine.
#[derive(Debug, Clone, Copy)]
pub struct SpreadAnalyticEngine {
    method: SpreadAnalyticMethod,
}

impl SpreadAnalyticEngine {
    pub fn new(method: SpreadAnalyticMethod) -> Self {
        Self { method }
    }
}

impl Default for SpreadAnalyticEngine {
    fn default() -> Self {
        Self::new(SpreadAnalyticMethod::Kirk)
    }
}

impl PricingEngine<SpreadOption> for SpreadAnalyticEngine {
    fn price(
        &self,
        instrument: &SpreadOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        let price = match self.method {
            SpreadAnalyticMethod::Margrabe => margrabe_exchange_price(instrument)?,
            SpreadAnalyticMethod::Kirk => kirk_spread_price(instrument)?,
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("effective_vol", instrument.effective_volatility()?);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

/// Margrabe exchange option value for `max(S1 - S2, 0)`.
#[inline]
pub fn margrabe_exchange_price(option: &SpreadOption) -> Result<f64, PricingError> {
    option.validate()?;

    if option.k.abs() > 1.0e-12 {
        return Err(PricingError::InvalidInput(
            "Margrabe pricing requires k = 0".to_string(),
        ));
    }

    if option.t <= 0.0 {
        return Ok((option.s1 - option.s2).max(0.0));
    }

    let sigma = option.effective_volatility()?;
    let s1_df = option.s1 * (-option.q1 * option.t).exp();
    let s2_df = option.s2 * (-option.q2 * option.t).exp();

    if sigma <= 0.0 {
        return Ok((s1_df - s2_df).max(0.0));
    }

    let sqrt_t = option.t.sqrt();
    let sig_sqrt_t = sigma * sqrt_t;
    let d1 = ((option.s1 / option.s2).ln()
        + (0.5 * sigma).mul_add(sigma, option.q2 - option.q1) * option.t)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    Ok(s1_df.mul_add(normal_cdf(d1), -(s2_df * normal_cdf(d2))))
}

/// Kirk approximation for spread call `max(S1 - S2 - K, 0)`.
#[inline]
pub fn kirk_spread_price(option: &SpreadOption) -> Result<f64, PricingError> {
    option.validate()?;

    if option.t <= 0.0 {
        return Ok((option.s1 - option.s2 - option.k).max(0.0));
    }

    let f1 = option.s1 * ((option.r - option.q1) * option.t).exp();
    let f2 = option.s2 * ((option.r - option.q2) * option.t).exp();
    let denominator = f2 + option.k;

    if denominator <= 0.0 {
        return Err(PricingError::InvalidInput(
            "Kirk approximation requires F2 + K > 0".to_string(),
        ));
    }

    let beta = f2 / denominator;
    // sigma_k^2 = vol1^2 - 2*beta*rho*vol1*vol2 + beta^2*vol2^2 via FMA
    let sigma_k2 = option.vol1.mul_add(
        option.vol1,
        (-2.0 * beta * option.rho * option.vol1).mul_add(option.vol2, beta * beta * option.vol2 * option.vol2),
    );

    if sigma_k2 < -1.0e-14 {
        return Err(PricingError::InvalidInput(
            "Kirk effective variance is negative".to_string(),
        ));
    }

    let sigma_k = sigma_k2.max(0.0).sqrt();
    let discount = (-option.r * option.t).exp();

    if sigma_k <= 0.0 {
        return Ok(discount * (f1 - denominator).max(0.0));
    }

    let sqrt_t = option.t.sqrt();
    let sig_sqrt_t = sigma_k * sqrt_t;

    let d1 = ((f1 / denominator).ln() + 0.5 * sigma_k2 * option.t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    Ok(discount * (f1 * normal_cdf(d1) - denominator * normal_cdf(d2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;
    use crate::pricing::european::black_scholes_price;
    use approx::assert_relative_eq;

    #[test]
    fn margrabe_matches_ratio_call_representation() {
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

        let sigma = option.effective_volatility().unwrap();
        assert_relative_eq!(sigma, 0.180_277_563_8, epsilon = 1e-10);

        let margrabe = margrabe_exchange_price(&option).unwrap();

        // Equivalent representation via option on ratio S1/S2 with strike 1.0.
        let ratio_call = option.s2
            * (-option.q1 * option.t).exp()
            * black_scholes_price(
                OptionType::Call,
                option.s1 / option.s2,
                1.0,
                option.q2 - option.q1,
                sigma,
                option.t,
            );

        assert_relative_eq!(margrabe, ratio_call, epsilon = 1e-10);
        assert_relative_eq!(margrabe, 5.687_130_39, epsilon = 2e-5);
    }

    #[test]
    fn kirk_reference_case() {
        let option = SpreadOption {
            s1: 100.0,
            s2: 96.0,
            k: 3.0,
            vol1: 0.20,
            vol2: 0.15,
            rho: 0.5,
            q1: 0.0,
            q2: 0.0,
            r: 0.05,
            t: 0.5,
        };

        let kirk = kirk_spread_price(&option).unwrap();
        assert_relative_eq!(kirk, 5.577_021_91, epsilon = 2e-5);
    }

    #[test]
    fn kirk_reduces_to_margrabe_when_k_zero() {
        let option = SpreadOption {
            s1: 110.0,
            s2: 100.0,
            k: 0.0,
            vol1: 0.22,
            vol2: 0.18,
            rho: 0.4,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.2,
        };

        let kirk = kirk_spread_price(&option).unwrap();
        let margrabe = margrabe_exchange_price(&option).unwrap();

        assert_relative_eq!(kirk, margrabe, epsilon = 1e-10);
    }
}
