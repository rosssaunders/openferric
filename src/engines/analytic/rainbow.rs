//! Closed-form analytic pricing routines for Rainbow.
//!
//! This module implements formulas and sensitivities used by fast deterministic engines.

use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::rainbow::{
    BestOfTwoCallOption, TwoAssetCorrelationOption, WorstOfTwoCallOption,
};
use crate::market::Market;
use crate::math::{bivariate_normal_cdf, normal_cdf};

/// Analytic engine for selected two-asset rainbow options.
#[derive(Debug, Clone, Default)]
pub struct RainbowAnalyticEngine;

impl RainbowAnalyticEngine {
    /// Creates a rainbow analytic engine.
    pub fn new() -> Self {
        Self
    }
}

fn effective_volatility(vol1: f64, vol2: f64, rho: f64) -> Result<f64, PricingError> {
    let variance = vol1 * vol1 - 2.0 * rho * vol1 * vol2 + vol2 * vol2;
    if variance < -1.0e-14 {
        return Err(PricingError::InvalidInput(
            "rainbow effective variance is negative".to_string(),
        ));
    }
    Ok(variance.max(0.0).sqrt())
}

fn black_scholes_call_with_dividend(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    t: f64,
) -> f64 {
    if t <= 0.0 {
        return (spot - strike).max(0.0);
    }

    if vol <= 0.0 {
        let forward = spot * ((rate - dividend_yield) * t).exp();
        return (-rate * t).exp() * (forward - strike).max(0.0);
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * vol * vol) * t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    spot * (-dividend_yield * t).exp() * normal_cdf(d1)
        - strike * (-rate * t).exp() * normal_cdf(d2)
}

/// Stulz (1982) best-of-two call price.
pub fn best_of_two_call_price(option: &BestOfTwoCallOption) -> Result<f64, PricingError> {
    option.validate()?;

    if option.t <= 0.0 {
        return Ok((option.s1.max(option.s2) - option.k).max(0.0));
    }

    let sigma = effective_volatility(option.vol1, option.vol2, option.rho)?;
    if sigma <= 0.0 {
        return Err(PricingError::InvalidInput(
            "rainbow effective volatility must be > 0".to_string(),
        ));
    }

    let sqrt_t = option.t.sqrt();
    let sig1 = option.vol1 * sqrt_t;
    let sig2 = option.vol2 * sqrt_t;
    let sig = sigma * sqrt_t;

    let d1 = ((option.s1 / option.k).ln()
        + (option.r - option.q1 + 0.5 * option.vol1 * option.vol1) * option.t)
        / sig1;
    let d2 = ((option.s2 / option.k).ln()
        + (option.r - option.q2 + 0.5 * option.vol2 * option.vol2) * option.t)
        / sig2;

    let d1m = d1 - sig1;
    let d2m = d2 - sig2;

    let y1 = ((option.s1 / option.s2).ln()
        + (option.q2 - option.q1 + 0.5 * sigma * sigma) * option.t)
        / sig;
    let y2 = ((option.s2 / option.s1).ln()
        + (option.q1 - option.q2 + 0.5 * sigma * sigma) * option.t)
        / sig;

    let rho1 = ((option.vol1 - option.rho * option.vol2) / sigma).clamp(-1.0, 1.0);
    let rho2 = ((option.vol2 - option.rho * option.vol1) / sigma).clamp(-1.0, 1.0);

    let df1 = (-option.q1 * option.t).exp();
    let df2 = (-option.q2 * option.t).exp();
    let dfr = (-option.r * option.t).exp();

    let prob_union = (1.0 - bivariate_normal_cdf(-d1m, -d2m, option.rho)).clamp(0.0, 1.0);

    Ok(option.s1 * df1 * bivariate_normal_cdf(d1, y1, rho1)
        + option.s2 * df2 * bivariate_normal_cdf(d2, y2, rho2)
        - option.k * dfr * prob_union)
}

/// Worst-of-two call value from rainbow parity:
/// `(best-of call) + (worst-of call) = call(S1) + call(S2)`.
pub fn worst_of_two_call_price(option: &WorstOfTwoCallOption) -> Result<f64, PricingError> {
    option.validate()?;

    if option.t <= 0.0 {
        return Ok((option.s1.min(option.s2) - option.k).max(0.0));
    }

    let best = best_of_two_call_price(&BestOfTwoCallOption {
        s1: option.s1,
        s2: option.s2,
        k: option.k,
        vol1: option.vol1,
        vol2: option.vol2,
        rho: option.rho,
        q1: option.q1,
        q2: option.q2,
        r: option.r,
        t: option.t,
    })?;

    let call1 = black_scholes_call_with_dividend(
        option.s1,
        option.k,
        option.r,
        option.q1,
        option.vol1,
        option.t,
    );
    let call2 = black_scholes_call_with_dividend(
        option.s2,
        option.k,
        option.r,
        option.q2,
        option.vol2,
        option.t,
    );

    Ok((call1 + call2 - best).max(0.0))
}

/// Two-asset correlation option price.
///
/// Call payoff: `1_{S2_T > K2} * max(S1_T - K1, 0)`
/// Put payoff:  `1_{S2_T < K2} * max(K1 - S1_T, 0)`
pub fn two_asset_correlation_price(
    option: &TwoAssetCorrelationOption,
) -> Result<f64, PricingError> {
    option.validate()?;

    if option.t <= 0.0 {
        return Ok(match option.option_type {
            OptionType::Call if option.s2 > option.k2 => (option.s1 - option.k1).max(0.0),
            OptionType::Put if option.s2 < option.k2 => (option.k1 - option.s1).max(0.0),
            _ => 0.0,
        });
    }

    let sqrt_t = option.t.sqrt();
    let sig1 = option.vol1 * sqrt_t;
    let sig2 = option.vol2 * sqrt_t;

    let d1 = ((option.s1 / option.k1).ln()
        + (option.r - option.q1 + 0.5 * option.vol1 * option.vol1) * option.t)
        / sig1;
    let d2 = d1 - sig1;

    let e2 = ((option.s2 / option.k2).ln()
        + (option.r - option.q2 - 0.5 * option.vol2 * option.vol2) * option.t)
        / sig2;
    let e1_tilted = e2 + option.rho * option.vol1 * sqrt_t;

    let df_q1 = (-option.q1 * option.t).exp();
    let df_r = (-option.r * option.t).exp();

    let price = match option.option_type {
        OptionType::Call => {
            option.s1 * df_q1 * bivariate_normal_cdf(d1, e1_tilted, option.rho)
                - option.k1 * df_r * bivariate_normal_cdf(d2, e2, option.rho)
        }
        OptionType::Put => {
            option.k1 * df_r * bivariate_normal_cdf(-d2, -e2, option.rho)
                - option.s1 * df_q1 * bivariate_normal_cdf(-d1, -e1_tilted, option.rho)
        }
    };

    Ok(price.max(0.0))
}

impl PricingEngine<BestOfTwoCallOption> for RainbowAnalyticEngine {
    fn price(
        &self,
        instrument: &BestOfTwoCallOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        let price = best_of_two_call_price(instrument)?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert(
            "effective_vol",
            effective_volatility(instrument.vol1, instrument.vol2, instrument.rho)?,
        );

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<WorstOfTwoCallOption> for RainbowAnalyticEngine {
    fn price(
        &self,
        instrument: &WorstOfTwoCallOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        let price = worst_of_two_call_price(instrument)?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert(
            "effective_vol",
            effective_volatility(instrument.vol1, instrument.vol2, instrument.rho)?,
        );

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<TwoAssetCorrelationOption> for RainbowAnalyticEngine {
    fn price(
        &self,
        instrument: &TwoAssetCorrelationOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        let price = two_asset_correlation_price(instrument)?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("rho", instrument.rho);

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
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn best_and_worst_of_two_match_stulz_reference_values() {
        let best = BestOfTwoCallOption {
            s1: 100.0,
            s2: 100.0,
            k: 100.0,
            r: 0.05,
            q1: 0.0,
            q2: 0.0,
            vol1: 0.20,
            vol2: 0.20,
            rho: 0.5,
            t: 1.0,
        };
        let worst = WorstOfTwoCallOption {
            s1: best.s1,
            s2: best.s2,
            k: best.k,
            r: best.r,
            q1: best.q1,
            q2: best.q2,
            vol1: best.vol1,
            vol2: best.vol2,
            rho: best.rho,
            t: best.t,
        };

        let best_price = best_of_two_call_price(&best).unwrap();
        let worst_price = worst_of_two_call_price(&worst).unwrap();

        assert_relative_eq!(best_price, 15.5185, epsilon = 5e-3);
        assert_relative_eq!(worst_price, 5.3826, epsilon = 5e-3);
    }

    #[test]
    fn best_plus_worst_equals_sum_of_individual_calls() {
        let best = BestOfTwoCallOption {
            s1: 100.0,
            s2: 100.0,
            k: 100.0,
            r: 0.05,
            q1: 0.0,
            q2: 0.0,
            vol1: 0.20,
            vol2: 0.20,
            rho: 0.5,
            t: 1.0,
        };
        let worst = WorstOfTwoCallOption {
            s1: best.s1,
            s2: best.s2,
            k: best.k,
            r: best.r,
            q1: best.q1,
            q2: best.q2,
            vol1: best.vol1,
            vol2: best.vol2,
            rho: best.rho,
            t: best.t,
        };

        let best_price = best_of_two_call_price(&best).unwrap();
        let worst_price = worst_of_two_call_price(&worst).unwrap();

        let c1 =
            black_scholes_call_with_dividend(best.s1, best.k, best.r, best.q1, best.vol1, best.t);
        let c2 =
            black_scholes_call_with_dividend(best.s2, best.k, best.r, best.q2, best.vol2, best.t);

        assert_relative_eq!(best_price + worst_price, c1 + c2, epsilon = 1e-10);
    }

    #[test]
    fn correlation_option_reduces_to_independence_case_when_rho_zero() {
        let option = TwoAssetCorrelationOption {
            option_type: OptionType::Call,
            s1: 100.0,
            s2: 95.0,
            k1: 100.0,
            k2: 95.0,
            vol1: 0.20,
            vol2: 0.25,
            rho: 0.0,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.2,
        };

        let price = two_asset_correlation_price(&option).unwrap();

        let call1 = black_scholes_call_with_dividend(
            option.s1,
            option.k1,
            option.r,
            option.q1,
            option.vol1,
            option.t,
        );

        let sqrt_t = option.t.sqrt();
        let d2_2 = ((option.s2 / option.k2).ln()
            + (option.r - option.q2 - 0.5 * option.vol2 * option.vol2) * option.t)
            / (option.vol2 * sqrt_t);
        let prob_s2_above_k2 = normal_cdf(d2_2);

        assert_relative_eq!(price, call1 * prob_s2_above_k2, epsilon = 2e-5);
    }
}
