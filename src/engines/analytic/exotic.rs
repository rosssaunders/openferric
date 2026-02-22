//! Module `engines::analytic::exotic`.
//!
//! Implements exotic abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `ExoticAnalyticEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFixedOption, LookbackFloatingOption,
    QuantoOption,
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

        let mut diagnostics = crate::core::Diagnostics::new();

        let price = match instrument {
            ExoticOption::LookbackFloating(spec) => {
                let vol = market.vol_for(market.spot, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol", vol);
                floating_lookback_price(spec, market, vol)
            }
            ExoticOption::LookbackFixed(spec) => {
                let vol = market.vol_for(spec.strike, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol", vol);
                fixed_lookback_price(spec, market, vol)
            }
            ExoticOption::Chooser(spec) => {
                let vol = market.vol_for(spec.strike, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol", vol);
                chooser_price(spec, market, vol)
            }
            ExoticOption::Quanto(spec) => {
                let vol = market.vol_for(spec.strike, spec.expiry.max(1.0e-12));
                if vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be > 0".to_string(),
                    ));
                }
                diagnostics.insert("vol", vol);
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
                diagnostics.insert("vol", vol);
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

#[inline]
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
    let d1 = ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    // Compute call price, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = spot.mul_add(df_q * nd1, -(strike * df_r * nd2));
    match option_type {
        OptionType::Call => call,
        OptionType::Put => call - spot * df_q + strike * df_r,
    }
}

#[inline]
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

#[inline]
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

    let vol_sq = vol * vol;
    let sqrt_t = expiry.sqrt();
    let a1 = ((spot / s_min).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / (vol * sqrt_t);
    let a2 = a1 - vol * sqrt_t;
    let a3 = a1 - 2.0 * carry * sqrt_t / vol;
    let ln_ratio = (spot / s_min).ln();
    let y = (-2.0 * carry / vol_sq) * ln_ratio;

    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let phi = vol_sq / (2.0 * carry);

    spot.mul_add(df_q * normal_cdf(a1), -(s_min * df_r * normal_cdf(a2)))
        + spot
            * df_r
            * phi
            * y.exp()
                .mul_add(normal_cdf(-a3), -((carry * expiry).exp() * normal_cdf(-a1)))
}

#[inline]
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

    let vol_sq = vol * vol;
    let sqrt_t = expiry.sqrt();
    let b1 = ((spot / s_max).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / (vol * sqrt_t);
    let b2 = b1 - vol * sqrt_t;
    let b3 = b1 - 2.0 * carry * sqrt_t / vol;
    let ln_ratio = (spot / s_max).ln();
    let y = (-2.0 * carry / vol_sq) * ln_ratio;

    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let phi = vol_sq / (2.0 * carry);

    s_max.mul_add(df_r * normal_cdf(-b2), -(spot * df_q * normal_cdf(-b1)))
        + spot
            * df_r
            * phi
            * (carry * expiry)
                .exp()
                .mul_add(normal_cdf(b1), -(y.exp() * normal_cdf(b3)))
}

#[inline]
fn fixed_lookback_price(spec: &LookbackFixedOption, market: &Market, vol: f64) -> f64 {
    if spec.expiry <= 0.0 {
        return match spec.option_type {
            OptionType::Call => {
                let s_max = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .max(market.spot);
                (s_max - spec.strike).max(0.0)
            }
            OptionType::Put => {
                let s_min = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .min(market.spot);
                (spec.strike - s_min).max(0.0)
            }
        };
    }

    let carry = market.rate - market.dividend_yield;
    match spec.option_type {
        OptionType::Call => {
            let s_max = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .max(market.spot);
            fixed_lookback_call_formula(
                market.spot,
                spec.strike,
                s_max,
                market.rate,
                market.dividend_yield,
                carry,
                vol,
                spec.expiry,
            )
        }
        OptionType::Put => {
            let s_min = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .min(market.spot);
            fixed_lookback_put_formula(
                market.spot,
                spec.strike,
                s_min,
                market.rate,
                market.dividend_yield,
                carry,
                vol,
                spec.expiry,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn fixed_lookback_call_formula(
    spot: f64,
    strike: f64,
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
        let hi = fixed_lookback_call_formula(spot, strike, s_max, rate, q_hi, eps, vol, expiry);
        let lo = fixed_lookback_call_formula(spot, strike, s_max, rate, q_lo, -eps, vol, expiry);
        return 0.5 * (hi + lo);
    }

    let vol_sq = vol * vol;
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let correction_scale = spot * df_r * (vol_sq / (2.0 * carry));
    let carry_term = (carry * expiry).exp();
    let shift = 2.0 * carry * sqrt_t / vol;
    let power_exp = -2.0 * carry / vol_sq;

    if strike > s_max {
        let d1 = ((spot / strike).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / sig_sqrt_t;
        let d2 = d1 - sig_sqrt_t;
        let d3 = d1 - shift;
        let power = (spot / strike).powf(power_exp);

        spot.mul_add(df_q * normal_cdf(d1), -(strike * df_r * normal_cdf(d2)))
            + correction_scale * carry_term.mul_add(normal_cdf(d1), -(power * normal_cdf(d3)))
    } else {
        let e1 = ((spot / s_max).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / sig_sqrt_t;
        let e2 = e1 - sig_sqrt_t;
        let e3 = e1 - shift;
        let power = (spot / s_max).powf(power_exp);

        df_r * (s_max - strike)
            + spot.mul_add(df_q * normal_cdf(e1), -(s_max * df_r * normal_cdf(e2)))
            + correction_scale * carry_term.mul_add(normal_cdf(e1), -(power * normal_cdf(e3)))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn fixed_lookback_put_formula(
    spot: f64,
    strike: f64,
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
        let hi = fixed_lookback_put_formula(spot, strike, s_min, rate, q_hi, eps, vol, expiry);
        let lo = fixed_lookback_put_formula(spot, strike, s_min, rate, q_lo, -eps, vol, expiry);
        return 0.5 * (hi + lo);
    }

    let vol_sq = vol * vol;
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let correction_scale = spot * df_r * (vol_sq / (2.0 * carry));
    let carry_term = (carry * expiry).exp();
    let shift = 2.0 * carry * sqrt_t / vol;
    let power_exp = -2.0 * carry / vol_sq;

    if strike < s_min {
        let d1 = ((spot / strike).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / sig_sqrt_t;
        let d2 = d1 - sig_sqrt_t;
        let d3 = -d1 + shift;
        let power = (spot / strike).powf(power_exp);

        strike.mul_add(df_r * normal_cdf(-d2), -(spot * df_q * normal_cdf(-d1)))
            + correction_scale * power.mul_add(normal_cdf(d3), -(carry_term * normal_cdf(-d1)))
    } else {
        let f1 = ((spot / s_min).ln() + (0.5 * vol).mul_add(vol, carry) * expiry) / sig_sqrt_t;
        let f2 = f1 - sig_sqrt_t;
        let f3 = -f1 + shift;
        let power = (spot / s_min).powf(power_exp);

        df_r * (strike - s_min)
            + s_min.mul_add(df_r * normal_cdf(-f2), -(spot * df_q * normal_cdf(-f1)))
            + correction_scale * power.mul_add(normal_cdf(f3), -(carry_term * normal_cdf(-f1)))
    }
}

#[inline]
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
        + (0.5 * vol).mul_add(vol, market.rate - market.dividend_yield) * spec.choose_time)
        / (vol * spec.choose_time.sqrt());

    put.mul_add(
        (-market.dividend_yield * tau).exp() * normal_cdf(-d1_choose),
        call,
    )
}

#[inline]
fn quanto_price(spec: &QuantoOption, market: &Market, vol: f64) -> f64 {
    if spec.expiry <= 0.0 {
        let intrinsic = match spec.option_type {
            OptionType::Call => (market.spot - spec.strike).max(0.0),
            OptionType::Put => (spec.strike - market.spot).max(0.0),
        };
        return spec.fx_rate * intrinsic;
    }

    // Quanto drift adjustment under domestic measure: mu_adj = rf - q - rho * vol * fx_vol
    let mu_adj =
        (-spec.asset_fx_corr * spec.fx_vol).mul_add(vol, spec.foreign_rate - market.dividend_yield);
    let sqrt_t = spec.expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((market.spot / spec.strike).ln() + (0.5 * vol).mul_add(vol, mu_adj) * spec.expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_dom = (-market.rate * spec.expiry).exp();
    let carry_discounted = ((mu_adj - market.rate) * spec.expiry).exp();

    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = market
        .spot
        .mul_add(carry_discounted * nd1, -(spec.strike * df_dom * nd2));
    let foreign_price = match spec.option_type {
        OptionType::Call => call,
        OptionType::Put => call - market.spot * carry_discounted + spec.strike * df_dom,
    };

    spec.fx_rate * foreign_price
}

#[inline]
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
    // drift = (r - q - 0.5 * vol^2) * t1 via FMA
    let drift = (-0.5 * vol).mul_add(vol, market.rate - market.dividend_yield) * t1;
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
    use crate::mc::{GbmPathGenerator, MonteCarloEngine};
    use crate::models::Gbm;

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

        assert_relative_eq!(price, 25.353_355_27, epsilon = 2e-5);
    }

    #[test]
    fn fixed_lookback_call_matches_haug_reference_values() {
        // Source: Haug (1998), pp. 63-64.
        let references = [
            (95.0, 0.10, 13.2687),
            (95.0, 0.20, 18.9263),
            (95.0, 0.30, 24.9857),
            (100.0, 0.10, 8.5126),
            (100.0, 0.20, 14.1702),
            (100.0, 0.30, 20.2296),
        ];

        let engine = ExoticAnalyticEngine::new();
        for (strike, vol, expected) in references {
            let market = Market::builder()
                .spot(100.0)
                .rate(0.10)
                .dividend_yield(0.00)
                .flat_vol(vol)
                .build()
                .expect("valid market");

            let option = ExoticOption::LookbackFixed(LookbackFixedOption {
                option_type: OptionType::Call,
                strike,
                expiry: 0.50,
                observed_extreme: Some(100.0),
            });

            let price = engine
                .price(&option, &market)
                .expect("pricing succeeds")
                .price;
            assert_relative_eq!(price, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn mc_floating_lookback_converges_to_analytic_within_two_percent() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .expect("valid market");

        let option = LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry: 0.50,
            observed_extreme: Some(90.0),
        };

        let analytic = ExoticAnalyticEngine::new()
            .price(&ExoticOption::LookbackFloating(option.clone()), &market)
            .expect("analytic pricing succeeds")
            .price;

        let vol = market.vol_for(market.spot, option.expiry);
        let generator = GbmPathGenerator {
            model: Gbm {
                mu: market.rate - market.dividend_yield,
                sigma: vol,
            },
            s0: market.spot,
            maturity: option.expiry,
            steps: 756,
        };
        let discount_factor = (-market.rate * option.expiry).exp();
        let observed_min = option.observed_extreme.unwrap_or(market.spot);

        let (mc, _stderr) = MonteCarloEngine::new(50_000, 42).with_antithetic(true).run(
            &generator,
            |path| {
                let path_min = path.iter().fold(observed_min, |acc, &s| acc.min(s));
                (path[path.len() - 1] - path_min).max(0.0)
            },
            discount_factor,
        );

        let rel_err = ((mc - analytic) / analytic).abs();
        assert!(
            rel_err <= 0.02,
            "MC floating-lookback mismatch: mc={} analytic={} rel_err={}",
            mc,
            analytic,
            rel_err
        );
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
