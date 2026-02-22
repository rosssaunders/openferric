use crate::core::{
    AadPricingResult, AadSensitivity, ExerciseStyle, Greeks, OptionType, PricingEngine,
    PricingError, PricingResult,
};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::{Tape, VarId, normal_cdf, normal_pdf};

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

    #[cfg(target_arch = "x86_64")]
    if super::bs_inline::has_fma_bs_kernel() {
        return super::bs_inline::bs_price_asm(
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
            matches!(option_type, OptionType::Call),
        );
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

fn bs_price_tape(
    tape: &mut Tape,
    option_type: OptionType,
    spot: VarId,
    strike: f64,
    rate: VarId,
    dividend_yield: VarId,
    vol: VarId,
    expiry: VarId,
) -> VarId {
    let sqrt_t = tape.sqrt(expiry);
    let sig_sqrt_t = tape.mul(vol, sqrt_t);

    let spot_over_strike = tape.mul_const(spot, 1.0 / strike);
    let log_moneyness = tape.ln(spot_over_strike);

    let vol_sq = tape.mul(vol, vol);
    let half_vol_sq = tape.mul_const(vol_sq, 0.5);
    let r_minus_q = tape.sub(rate, dividend_yield);
    let carry = tape.add(r_minus_q, half_vol_sq);
    let drift = tape.mul(carry, expiry);
    let numerator = tape.add(log_moneyness, drift);
    let d1 = tape.div(numerator, sig_sqrt_t);
    let d2 = tape.sub(d1, sig_sqrt_t);

    let rt = tape.mul(rate, expiry);
    let neg_rt = tape.neg(rt);
    let df_r = tape.exp(neg_rt);

    let qt = tape.mul(dividend_yield, expiry);
    let neg_qt = tape.neg(qt);
    let df_q = tape.exp(neg_qt);

    match option_type {
        OptionType::Call => {
            let nd1 = tape.normal_cdf(d1);
            let nd2 = tape.normal_cdf(d2);

            let spot_df_q = tape.mul(spot, df_q);
            let left = tape.mul(spot_df_q, nd1);

            let strike_df_r = tape.mul_const(df_r, strike);
            let right = tape.mul(strike_df_r, nd2);
            tape.sub(left, right)
        }
        OptionType::Put => {
            let neg_d1 = tape.neg(d1);
            let neg_d2 = tape.neg(d2);
            let n_neg_d1 = tape.normal_cdf(neg_d1);
            let n_neg_d2 = tape.normal_cdf(neg_d2);

            let strike_df_r = tape.mul_const(df_r, strike);
            let left = tape.mul(strike_df_r, n_neg_d2);

            let spot_df_q = tape.mul(spot, df_q);
            let right = tape.mul(spot_df_q, n_neg_d1);
            tape.sub(left, right)
        }
    }
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

    fn price_with_greeks_aad(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<AadPricingResult, PricingError> {
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
            return Ok(AadPricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                gradient: vec![
                    AadSensitivity {
                        factor: "spot".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "rate".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "dividend_yield".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "vol".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "expiry".to_string(),
                        value: 0.0,
                    },
                ],
            });
        }

        let mut tape = Tape::with_capacity(128);
        let spot = tape.input(market.spot);
        let rate = tape.input(market.rate);
        let dividend = tape.input(market.dividend_yield);
        let sigma = tape.input(vol);
        let expiry = tape.input(instrument.expiry);
        let price_node = bs_price_tape(
            &mut tape,
            instrument.option_type,
            spot,
            instrument.strike,
            rate,
            dividend,
            sigma,
            expiry,
        );
        let grads = tape.gradient(price_node, &[spot, rate, dividend, sigma, expiry]);

        Ok(AadPricingResult {
            price: tape.value(price_node),
            gradient: vec![
                AadSensitivity {
                    factor: "spot".to_string(),
                    value: grads[0],
                },
                AadSensitivity {
                    factor: "rate".to_string(),
                    value: grads[1],
                },
                AadSensitivity {
                    factor: "dividend_yield".to_string(),
                    value: grads[2],
                },
                AadSensitivity {
                    factor: "vol".to_string(),
                    value: grads[3],
                },
                AadSensitivity {
                    factor: "expiry".to_string(),
                    value: grads[4],
                },
            ],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greeks::black_scholes_merton_greeks;
    use approx::assert_relative_eq;

    fn grad(result: &AadPricingResult, factor: &str) -> f64 {
        result
            .gradient
            .iter()
            .find(|g| g.factor == factor)
            .map(|g| g.value)
            .unwrap()
    }

    #[test]
    fn aad_matches_analytic_greeks_to_1e10() {
        let option = VanillaOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.5,
            exercise: ExerciseStyle::European,
        };
        let market = Market::builder()
            .spot(110.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();

        let engine = BlackScholesEngine::new();
        let aad = engine.price_with_greeks_aad(&option, &market).unwrap();
        let cf = black_scholes_merton_greeks(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.2,
            option.expiry,
        );

        assert_relative_eq!(
            aad.price,
            bs_price(
                option.option_type,
                market.spot,
                option.strike,
                market.rate,
                market.dividend_yield,
                0.2,
                option.expiry
            ),
            epsilon = 1e-12
        );
        assert_relative_eq!(grad(&aad, "spot"), cf.delta, epsilon = 1e-10);
        assert_relative_eq!(grad(&aad, "vol"), cf.vega, epsilon = 1e-10);
        assert_relative_eq!(grad(&aad, "rate"), cf.rho, epsilon = 1e-10);
    }
}
