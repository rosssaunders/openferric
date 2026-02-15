use std::collections::HashMap;

use crate::core::{ExerciseStyle, Greeks, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::{normal_cdf, normal_pdf};

/// Analytic Black-Scholes engine for European vanilla options.
#[derive(Debug, Clone, Default)]
pub struct BlackScholesEngine;

impl BlackScholesEngine {
    /// Creates a Black-Scholes engine instance.
    pub fn new() -> Self {
        Self
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

fn bs_price_greeks_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, Greeks, f64, f64) {
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 =
        ((spot / strike).ln() + (rate - dividend_yield + 0.5 * vol * vol) * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    let price = match option_type {
        OptionType::Call => spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2),
        OptionType::Put => strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1),
    };

    let delta = match option_type {
        OptionType::Call => df_q * normal_cdf(d1),
        OptionType::Put => df_q * (normal_cdf(d1) - 1.0),
    };
    let gamma = df_q * normal_pdf(d1) / (spot * vol * sqrt_t);
    let vega = spot * df_q * normal_pdf(d1) * sqrt_t;

    let theta = match option_type {
        OptionType::Call => {
            -spot * df_q * normal_pdf(d1) * vol / (2.0 * sqrt_t)
                + dividend_yield * spot * df_q * normal_cdf(d1)
                - rate * strike * df_r * normal_cdf(d2)
        }
        OptionType::Put => {
            -spot * df_q * normal_pdf(d1) * vol / (2.0 * sqrt_t)
                - dividend_yield * spot * df_q * normal_cdf(-d1)
                + rate * strike * df_r * normal_cdf(-d2)
        }
    };

    let rho = match option_type {
        OptionType::Call => strike * expiry * df_r * normal_cdf(d2),
        OptionType::Put => -strike * expiry * df_r * normal_cdf(-d2),
    };

    (
        price,
        Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        },
        d1,
        d2,
    )
}

impl PricingEngine<VanillaOption> for BlackScholesEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "BlackScholesEngine supports European exercise only".to_string(),
            ));
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                stderr: None,
                greeks: Some(Greeks {
                    delta: 0.0,
                    gamma: 0.0,
                    vega: 0.0,
                    theta: 0.0,
                    rho: 0.0,
                }),
                diagnostics: HashMap::new(),
            });
        }

        let (price, greeks, d1, d2) = bs_price_greeks_with_dividend(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.expiry,
        );

        let mut diagnostics = HashMap::new();
        diagnostics.insert("vol".to_string(), vol);
        diagnostics.insert("d1".to_string(), d1);
        diagnostics.insert("d2".to_string(), d2);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(greeks),
            diagnostics,
        })
    }
}

/// One-liner convenience wrapper for Black-Scholes pricing.
pub fn black_scholes(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
) -> Result<f64, PricingError> {
    let instrument = VanillaOption {
        option_type,
        strike,
        expiry,
        exercise: ExerciseStyle::European,
    };
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(0.0)
        .flat_vol(vol)
        .build()?;
    let engine = BlackScholesEngine::new();
    Ok(engine.price(&instrument, &market)?.price)
}
