
use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::power::PowerOption;
use crate::market::Market;
use crate::math::normal_cdf;

/// Analytic power option engine based on Haug-style transformed Black pricing.
#[derive(Debug, Clone, Default)]
pub struct PowerOptionEngine;

impl PowerOptionEngine {
    /// Creates a power option engine.
    pub fn new() -> Self {
        Self
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64, alpha: f64) -> f64 {
    let transformed_spot = spot.powf(alpha);
    match option_type {
        OptionType::Call => (transformed_spot - strike).max(0.0),
        OptionType::Put => (strike - transformed_spot).max(0.0),
    }
}

/// Power option price for payoff `max(S^alpha - K, 0)` / `max(K - S^alpha, 0)`.
#[allow(clippy::too_many_arguments)]
pub fn power_option_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    alpha: f64,
    expiry: f64,
) -> Result<f64, PricingError> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err(PricingError::InvalidInput(
            "power option spot must be finite and > 0".to_string(),
        ));
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err(PricingError::InvalidInput(
            "power option strike must be finite and > 0".to_string(),
        ));
    }
    if !rate.is_finite() || !dividend_yield.is_finite() {
        return Err(PricingError::InvalidInput(
            "power option rates must be finite".to_string(),
        ));
    }
    if !vol.is_finite() || vol < 0.0 {
        return Err(PricingError::InvalidInput(
            "power option vol must be finite and >= 0".to_string(),
        ));
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(PricingError::InvalidInput(
            "power option alpha must be finite and > 0".to_string(),
        ));
    }
    if !expiry.is_finite() || expiry < 0.0 {
        return Err(PricingError::InvalidInput(
            "power option expiry must be finite and >= 0".to_string(),
        ));
    }

    if expiry <= 0.0 {
        return Ok(intrinsic(option_type, spot, strike, alpha));
    }

    // Haug-style transformed Black representation.
    let pv_forward = spot.powf(alpha)
        * (((alpha - 1.0) * (rate + 0.5 * alpha * vol * vol) - alpha * dividend_yield) * expiry)
            .exp();
    let discount = (-rate * expiry).exp();
    let discounted_strike = strike * discount;

    let vol_adj = alpha * vol;
    if vol_adj <= 0.0 {
        return Ok(match option_type {
            OptionType::Call => (pv_forward - discounted_strike).max(0.0),
            OptionType::Put => (discounted_strike - pv_forward).max(0.0),
        });
    }

    let sig_sqrt_t = vol_adj * expiry.sqrt();
    let d1 =
        ((pv_forward / discounted_strike).ln() + 0.5 * vol_adj * vol_adj * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    Ok(match option_type {
        OptionType::Call => pv_forward * normal_cdf(d1) - discounted_strike * normal_cdf(d2),
        OptionType::Put => discounted_strike * normal_cdf(-d2) - pv_forward * normal_cdf(-d1),
    })
}

impl PricingEngine<PowerOption> for PowerOptionEngine {
    fn price(
        &self,
        instrument: &PowerOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        let implied_strike = instrument.strike.powf(1.0 / instrument.alpha);
        let vol = market.vol_for(implied_strike, instrument.expiry.max(1.0e-12));
        if vol < 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be >= 0".to_string(),
            ));
        }

        let price = power_option_price(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.alpha,
            instrument.expiry,
        )?;

        let pv_forward = market.spot.powf(instrument.alpha)
            * (((instrument.alpha - 1.0) * (market.rate + 0.5 * instrument.alpha * vol * vol)
                - instrument.alpha * market.dividend_yield)
                * instrument.expiry)
                .exp();

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("vol_adj", instrument.alpha * vol);
        diagnostics.insert("pv_forward", pv_forward);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn power_option_reference_case() {
        let price = power_option_price(
            OptionType::Call,
            100.0,
            10_000.0,
            0.05,
            0.0,
            0.20,
            2.0,
            0.50,
        )
        .unwrap();

        assert_relative_eq!(price, 1_524.601_214_70, epsilon = 2e-8);
    }

    #[test]
    fn power_option_engine_matches_formula() {
        let instrument = PowerOption::call(10_000.0, 2.0, 0.50);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let formula = power_option_price(
            OptionType::Call,
            100.0,
            10_000.0,
            0.05,
            0.0,
            0.20,
            2.0,
            0.50,
        )
        .unwrap();
        let engine = PowerOptionEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(engine, formula, epsilon = 1e-12);
    }
}
