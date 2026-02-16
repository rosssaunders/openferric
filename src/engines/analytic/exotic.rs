use std::collections::HashMap;

use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFloatingOption, QuantoOption,
};
use crate::market::Market;
use crate::math::{gauss_legendre_integrate, normal_cdf, normal_pdf};

/// Analytic/semi-analytic engine for selected exotic options.
#[derive(Debug, Clone, Default)]
pub struct ExoticAnalyticEngine;

impl ExoticAnalyticEngine {
    /// Creates an exotic analytic engine.
    pub fn new() -> Self {
        Self
    }
}

impl PricingEngine<ExoticOption> for ExoticAnalyticEngine {
    fn price(
        &self,
        instrument: &ExoticOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        let mut diagnostics = HashMap::new();

        let price = match instrument {
            ExoticOption::LookbackFloating(spec) => {
                let vol = market.vol_for(market.spot, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol".to_string(), vol);
                floating_lookback_price(spec, market, vol)
            }
            ExoticOption::Chooser(spec) => {
                let vol = market.vol_for(spec.strike, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol".to_string(), vol);
                chooser_price(spec, market, vol)
            }
            ExoticOption::Quanto(spec) => {
                let vol = market.vol_for(spec.strike, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol".to_string(), vol);
                quanto_price(spec, market, vol)
            }
            ExoticOption::Compound(spec) => {
                let vol =
                    market.vol_for(spec.underlying_strike, spec.underlying_expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol".to_string(), vol);
                compound_price(spec, market, vol)?
            }
        };

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

fn bs_price_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 {
        return match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        };
    }

    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 =
        ((spot / strike).ln() + (rate - dividend_yield + 0.5 * vol * vol) * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    match option_type {
        OptionType::Call => spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2),
        OptionType::Put => strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1),
    }
}

fn floating_lookback_price(spec: &LookbackFloatingOption, market: &Market, vol: f64) -> f64 {
    if spec.expiry <= 0.0 {
        let extreme = spec.observed_extreme.unwrap_or(market.spot);
        return match spec.option_type {
            OptionType::Call => (market.spot - extreme).max(0.0),
            OptionType::Put => (extreme - market.spot).max(0.0),
        };
    }

    let b = market.rate - market.dividend_yield;
    match spec.option_type {
        OptionType::Call => {
            let s_min = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .min(market.spot);
            floating_lookback_call_formula(
                market.spot,
                s_min,
                market.rate,
                market.dividend_yield,
                b,
                vol,
                spec.expiry,
            )
        }
        OptionType::Put => {
            let s_max = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .max(market.spot);
            floating_lookback_put_formula(
                market.spot,
                s_max,
                market.rate,
                market.dividend_yield,
                b,
                vol,
                spec.expiry,
            )
        }
    }
}

fn floating_lookback_call_formula(
    spot: f64,
    s_min: f64,
    rate: f64,
    dividend_yield: f64,
    carry: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if carry.abs() < 1.0e-8 {
        let eps = 1.0e-5;
        let q_hi = rate - eps;
        let q_lo = rate + eps;
        let hi = floating_lookback_call_formula(spot, s_min, rate, q_hi, eps, vol, expiry);
        let lo = floating_lookback_call_formula(spot, s_min, rate, q_lo, -eps, vol, expiry);
        return 0.5 * (hi + lo);
    }

    let sqrt_t = expiry.sqrt();
    let a1 = ((spot / s_min).ln() + (carry + 0.5 * vol * vol) * expiry) / (vol * sqrt_t);
    let a2 = a1 - vol * sqrt_t;
    let a3 = a1 - 2.0 * carry * sqrt_t / vol;
    let y = (2.0 * carry / (vol * vol)) * (spot / s_min).ln();

    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let phi = (vol * vol) / (2.0 * carry);

    spot * df_q * normal_cdf(a1)
        - spot * df_q * phi * normal_cdf(-a1)
        - s_min * df_r * normal_cdf(a2)
        + s_min * df_r * y.exp() * normal_cdf(-a3)
}

fn floating_lookback_put_formula(
    spot: f64,
    s_max: f64,
    rate: f64,
    dividend_yield: f64,
    carry: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if carry.abs() < 1.0e-8 {
        let eps = 1.0e-5;
        let q_hi = rate - eps;
        let q_lo = rate + eps;
        let hi = floating_lookback_put_formula(spot, s_max, rate, q_hi, eps, vol, expiry);
        let lo = floating_lookback_put_formula(spot, s_max, rate, q_lo, -eps, vol, expiry);
        return 0.5 * (hi + lo);
    }

    let sqrt_t = expiry.sqrt();
    let b1 = ((spot / s_max).ln() + (carry + 0.5 * vol * vol) * expiry) / (vol * sqrt_t);
    let b2 = b1 - vol * sqrt_t;
    let b3 = b1 - 2.0 * carry * sqrt_t / vol;
    let y = (2.0 * carry / (vol * vol)) * (spot / s_max).ln();

    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let phi = (vol * vol) / (2.0 * carry);

    s_max * df_r * normal_cdf(-b2) - spot * df_q * normal_cdf(-b1)
        + spot * df_q * phi * normal_cdf(b1)
        - s_max * df_r * y.exp() * normal_cdf(b3)
}

fn chooser_price(spec: &ChooserOption, market: &Market, vol: f64) -> f64 {
    let call = bs_price_with_dividend(
        OptionType::Call,
        market.spot,
        spec.strike,
        market.rate,
        market.dividend_yield,
        vol,
        spec.expiry,
    );
    let put = bs_price_with_dividend(
        OptionType::Put,
        market.spot,
        spec.strike,
        market.rate,
        market.dividend_yield,
        vol,
        spec.expiry,
    );

    if spec.choose_time <= 0.0 {
        return call.max(put);
    }

    let tau = (spec.expiry - spec.choose_time).max(0.0);
    let d1_choose = ((market.spot / spec.strike).ln()
        + (market.rate - market.dividend_yield + 0.5 * vol * vol) * spec.choose_time)
        / (vol * spec.choose_time.sqrt());

    call + put * (-market.dividend_yield * tau).exp() * normal_cdf(-d1_choose)
}

fn quanto_price(spec: &QuantoOption, market: &Market, vol: f64) -> f64 {
    if spec.expiry <= 0.0 {
        let intrinsic = match spec.option_type {
            OptionType::Call => (market.spot - spec.strike).max(0.0),
            OptionType::Put => (spec.strike - market.spot).max(0.0),
        };
        return spec.fx_rate * intrinsic;
    }

    // Quanto drift adjustment under domestic measure.
    let mu_adj = spec.foreign_rate - market.dividend_yield - spec.asset_fx_corr * vol * spec.fx_vol;
    let sqrt_t = spec.expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 =
        ((market.spot / spec.strike).ln() + (mu_adj + 0.5 * vol * vol) * spec.expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_dom = (-market.rate * spec.expiry).exp();
    let carry_discounted = ((mu_adj - market.rate) * spec.expiry).exp();

    let foreign_price = match spec.option_type {
        OptionType::Call => {
            market.spot * carry_discounted * normal_cdf(d1) - spec.strike * df_dom * normal_cdf(d2)
        }
        OptionType::Put => {
            spec.strike * df_dom * normal_cdf(-d2)
                - market.spot * carry_discounted * normal_cdf(-d1)
        }
    };

    spec.fx_rate * foreign_price
}

fn compound_price(spec: &CompoundOption, market: &Market, vol: f64) -> Result<f64, PricingError> {
    if spec.compound_expiry <= 0.0 {
        let inner = bs_price_with_dividend(
            spec.underlying_option_type,
            market.spot,
            spec.underlying_strike,
            market.rate,
            market.dividend_yield,
            vol,
            spec.underlying_expiry,
        );
        return Ok(match spec.option_type {
            OptionType::Call => (inner - spec.compound_strike).max(0.0),
            OptionType::Put => (spec.compound_strike - inner).max(0.0),
        });
    }

    let t1 = spec.compound_expiry;
    let t2 = spec.underlying_expiry;
    let tau = (t2 - t1).max(0.0);
    let drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * t1;
    let vol_t1 = vol * t1.sqrt();

    // Numerical quadrature approximation to the Geske expectation.
    let integral = gauss_legendre_integrate(
        |z| {
            let s_t1 = market.spot * (drift + vol_t1 * z).exp();
            let inner = bs_price_with_dividend(
                spec.underlying_option_type,
                s_t1,
                spec.underlying_strike,
                market.rate,
                market.dividend_yield,
                vol,
                tau,
            );

            let payoff = match spec.option_type {
                OptionType::Call => (inner - spec.compound_strike).max(0.0),
                OptionType::Put => (spec.compound_strike - inner).max(0.0),
            };

            payoff * normal_pdf(z)
        },
        -8.0,
        8.0,
        96,
    )
    .map_err(|e| PricingError::NumericalError(format!("compound quadrature failed: {e:?}")))?;

    Ok((-market.rate * t1).exp() * integral)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn lookback_call_matches_haug_reference_value() {
        // Goldman-Sosin-Gatto floating-strike lookback reference setup.
        // S=120, S_min=100, T=0.5, r=0.10, q=0.06, sigma=0.30.
        let market = Market::builder()
            .spot(120.0)
            .rate(0.10)
            .dividend_yield(0.06)
            .flat_vol(0.30)
            .build()
            .unwrap();

        let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry: 0.5,
            observed_extreme: Some(100.0),
        });

        let engine = ExoticAnalyticEngine::new();
        let price = engine.price(&option, &market).unwrap().price;

        assert_relative_eq!(price, 25.862_607_39, epsilon = 3e-6);
    }

    #[test]
    fn quanto_option_reduces_to_vanilla_when_fx_terms_zero() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();

        let spec = QuantoOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            fx_rate: 1.0,
            foreign_rate: market.rate,
            fx_vol: 0.0,
            asset_fx_corr: 0.0,
        };

        let vanilla = bs_price_with_dividend(
            OptionType::Call,
            market.spot,
            spec.strike,
            market.rate,
            market.dividend_yield,
            0.2,
            1.0,
        );

        let price = quanto_price(&spec, &market, 0.2);
        assert_relative_eq!(price, vanilla, epsilon = 1e-10);
    }

    #[test]
    fn compound_option_price_is_non_negative() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.04)
            .dividend_yield(0.01)
            .flat_vol(0.25)
            .build()
            .unwrap();

        let spec = CompoundOption {
            option_type: OptionType::Call,
            underlying_option_type: OptionType::Call,
            compound_strike: 6.0,
            underlying_strike: 100.0,
            compound_expiry: 0.5,
            underlying_expiry: 1.0,
        };

        let price = compound_price(&spec, &market, 0.25).unwrap();
        assert!(price >= 0.0);
    }
}
