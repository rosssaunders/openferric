
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

#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    normal_cdf(x)
}

#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    normal_pdf(x)
}

#[inline]
fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
fn d1_d2(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, f64) {
    let sig_sqrt_t = vol * expiry.sqrt();
    let d1 =
        ((spot / strike).ln() + (rate - dividend_yield + 0.5 * vol * vol) * expiry) / sig_sqrt_t;
    (d1, d1 - sig_sqrt_t)
}

#[inline]
pub fn bs_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 {
        return intrinsic(option_type, spot, strike);
    }
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    if vol <= 0.0 {
        return match option_type {
            OptionType::Call => (spot * df_q - strike * df_r).max(0.0),
            OptionType::Put => (strike * df_r - spot * df_q).max(0.0),
        };
    }

    let (d1, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    match option_type {
        OptionType::Call => spot * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2),
        OptionType::Put => strike * df_r * norm_cdf(-d2) - spot * df_q * norm_cdf(-d1),
    }
}

#[inline]
pub fn bs_delta(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_q = (-dividend_yield * expiry).exp();
    match option_type {
        OptionType::Call => df_q * norm_cdf(d1),
        OptionType::Put => df_q * (norm_cdf(d1) - 1.0),
    }
}

#[inline]
pub fn bs_gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 || spot <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_q = (-dividend_yield * expiry).exp();
    df_q * norm_pdf(d1) / (spot * vol * expiry.sqrt())
}

#[inline]
pub fn bs_vega(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 || spot <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_q = (-dividend_yield * expiry).exp();
    spot * df_q * norm_pdf(d1) * expiry.sqrt()
}

#[inline]
pub fn bs_theta(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 || spot <= 0.0 {
        return 0.0;
    }
    let (d1, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let sqrt_t = expiry.sqrt();
    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    match option_type {
        OptionType::Call => {
            -spot * df_q * norm_pdf(d1) * vol / (2.0 * sqrt_t)
                + dividend_yield * spot * df_q * norm_cdf(d1)
                - rate * strike * df_r * norm_cdf(d2)
        }
        OptionType::Put => {
            -spot * df_q * norm_pdf(d1) * vol / (2.0 * sqrt_t)
                - dividend_yield * spot * df_q * norm_cdf(-d1)
                + rate * strike * df_r * norm_cdf(-d2)
        }
    }
}

#[inline]
pub fn bs_rho(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 || spot <= 0.0 {
        return 0.0;
    }
    let (_, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_r = (-rate * expiry).exp();
    match option_type {
        OptionType::Call => strike * expiry * df_r * norm_cdf(d2),
        OptionType::Put => -strike * expiry * df_r * norm_cdf(-d2),
    }
}

#[inline]
fn bs_price_greeks_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, Greeks, f64, f64) {
    let (d1, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let price = bs_price(option_type, spot, strike, rate, dividend_yield, vol, expiry);
    let delta = bs_delta(option_type, spot, strike, rate, dividend_yield, vol, expiry);
    let gamma = bs_gamma(spot, strike, rate, dividend_yield, vol, expiry);
    let vega = bs_vega(spot, strike, rate, dividend_yield, vol, expiry);
    let theta = bs_theta(option_type, spot, strike, rate, dividend_yield, vol, expiry);
    let rho = bs_rho(option_type, spot, strike, rate, dividend_yield, vol, expiry);

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
                diagnostics: crate::core::Diagnostics::new(),
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

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d1", d1);
        diagnostics.insert("d2", d2);

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
