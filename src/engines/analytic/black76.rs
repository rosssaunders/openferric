//! Module `engines::analytic::black76`.
//!
//! Implements black76 workflows with concrete routines such as `black76_price`, `black76_greeks`.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `Black76Engine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{Greeks, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::black76::FuturesOption;
use crate::market::Market;
use crate::math::{normal_cdf, normal_pdf};

/// Analytic Black-76 engine for European options on forwards/futures.
#[derive(Debug, Clone, Default)]
pub struct Black76Engine;

impl Black76Engine {
    /// Creates a Black-76 engine.
    pub fn new() -> Self {
        Self
    }
}

#[inline]
fn intrinsic(option_type: OptionType, forward: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (forward - strike).max(0.0),
        OptionType::Put => (strike - forward).max(0.0),
    }
}

#[inline]
fn black76_price_greeks(option: &FuturesOption) -> (f64, Greeks, f64, f64) {
    let sqrt_t = option.t.sqrt();
    let sig_sqrt_t = option.vol * sqrt_t;
    let d1 = ((option.forward / option.strike).ln() + 0.5 * option.vol * option.vol * option.t)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df = (-option.r * option.t).exp();
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let pdf_d1 = normal_pdf(d1);

    // Compute call price, derive put via put-call parity.
    let call = df * option.forward.mul_add(nd1, -(option.strike * nd2));
    let price = match option.option_type {
        OptionType::Call => call,
        OptionType::Put => call - df * (option.forward - option.strike),
    };

    let delta = match option.option_type {
        OptionType::Call => df * nd1,
        OptionType::Put => df * (nd1 - 1.0),
    };
    let gamma = df * pdf_d1 / (option.forward * option.vol * sqrt_t);
    let vega = df * option.forward * pdf_d1 * sqrt_t;
    let theta = option.r.mul_add(
        price,
        -(df * option.forward * pdf_d1 * option.vol / (2.0 * sqrt_t)),
    );

    // Sensitivity to r for fixed forward input.
    let rho = -option.t * price;

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

/// Closed-form Black-76 price for European options on forwards/futures.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn black76_price(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    r: f64,
    vol: f64,
    t: f64,
) -> Result<f64, PricingError> {
    let option = FuturesOption::new(forward, strike, vol, r, t, option_type);
    option.validate()?;

    if option.t <= 0.0 {
        return Ok(intrinsic(option.option_type, option.forward, option.strike));
    }

    if option.vol <= 0.0 {
        return Ok((-option.r * option.t).exp()
            * intrinsic(option.option_type, option.forward, option.strike));
    }

    Ok(black76_price_greeks(&option).0)
}

/// Closed-form Black-76 Greeks for European options on forwards/futures.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn black76_greeks(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    r: f64,
    vol: f64,
    t: f64,
) -> Result<Greeks, PricingError> {
    let option = FuturesOption::new(forward, strike, vol, r, t, option_type);
    option.validate()?;

    if option.t <= 0.0 || option.vol <= 0.0 {
        return Ok(Greeks {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        });
    }

    Ok(black76_price_greeks(&option).1)
}

impl PricingEngine<FuturesOption> for Black76Engine {
    fn price(
        &self,
        instrument: &FuturesOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.t <= 0.0 {
            return Ok(PricingResult {
                price: intrinsic(
                    instrument.option_type,
                    instrument.forward,
                    instrument.strike,
                ),
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

        if instrument.vol <= 0.0 {
            return Ok(PricingResult {
                price: (-instrument.r * instrument.t).exp()
                    * intrinsic(
                        instrument.option_type,
                        instrument.forward,
                        instrument.strike,
                    ),
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

        let (price, greeks, d1, d2) = black76_price_greeks(instrument);
        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("d1", d1);
        diagnostics.insert("d2", d2);
        diagnostics.insert("discount_factor", (-instrument.r * instrument.t).exp());

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(greeks),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::core::{ExerciseStyle, PricingEngine};
    use crate::engines::analytic::black_scholes::BlackScholesEngine;
    use crate::instruments::vanilla::VanillaOption;

    fn dummy_market() -> Market {
        Market::builder()
            .spot(1.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap()
    }

    #[test]
    fn reference_values() {
        let c1 = black76_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0).unwrap();
        let c2 = black76_price(OptionType::Call, 100.0, 90.0, 0.05, 0.20, 0.50).unwrap();
        let p1 = black76_price(OptionType::Put, 100.0, 110.0, 0.05, 0.20, 0.50).unwrap();

        assert_relative_eq!(c1, 7.577_082_146_4, epsilon = 2e-4);
        assert_relative_eq!(c2, 11.481_788_247_2, epsilon = 2e-4);
        assert_relative_eq!(p1, 11.909_749_684_9, epsilon = 2e-4);
    }

    #[test]
    fn matches_black_scholes_with_q_equal_r() {
        let f = 100.0;
        let k = 100.0;
        let r = 0.05;
        let vol = 0.20;
        let t = 1.0;

        let black76 = black76_price(OptionType::Call, f, k, r, vol, t).unwrap();

        let option = VanillaOption {
            option_type: OptionType::Call,
            strike: k,
            expiry: t,
            exercise: ExerciseStyle::European,
        };
        let market = Market::builder()
            .spot(f)
            .rate(r)
            .dividend_yield(r)
            .flat_vol(vol)
            .build()
            .unwrap();
        let bsm = BlackScholesEngine::new()
            .price(&option, &market)
            .unwrap()
            .price;

        assert_relative_eq!(black76, bsm, epsilon = 1e-10);
    }

    #[test]
    fn put_call_parity_holds() {
        let f = 100.0;
        let k = 95.0;
        let r = 0.03;
        let vol = 0.22;
        let t = 1.4;

        let c = black76_price(OptionType::Call, f, k, r, vol, t).unwrap();
        let p = black76_price(OptionType::Put, f, k, r, vol, t).unwrap();

        assert_relative_eq!(c - p, (-r * t).exp() * (f - k), epsilon = 2e-10);
    }

    #[test]
    fn engine_exposes_greeks_and_diagnostics() {
        let instrument = FuturesOption::call(100.0, 100.0, 0.20, 0.05, 1.0);
        let result = Black76Engine::new()
            .price(&instrument, &dummy_market())
            .unwrap();

        assert!(result.greeks.is_some());
        assert!(result.diagnostics.contains_key("d1"));
        assert!(result.diagnostics.contains_key("d2"));
    }
}
