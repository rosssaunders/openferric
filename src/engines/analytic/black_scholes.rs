//! Module `engines::analytic::black_scholes`.
//!
//! Implements black scholes workflows with concrete routines such as `norm_cdf`, `norm_pdf`, `bs_price`, `bs_delta`.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `BlackScholesEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
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
    let d1 = ((spot / strike).ln()
        + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
        / sig_sqrt_t;
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
    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let s_df_q = spot * df_q;
    let k_df_r = strike * df_r;
    let call = s_df_q.mul_add(nd1, -(k_df_r * nd2));
    match option_type {
        OptionType::Call => call,
        OptionType::Put => call - s_df_q + k_df_r,
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
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let theta_common = -spot * df_q * norm_pdf(d1) * vol / (2.0 * sqrt_t);
    match option_type {
        OptionType::Call => {
            theta_common
                + dividend_yield * spot * df_q * nd1
                - rate * strike * df_r * nd2
        }
        OptionType::Put => {
            theta_common
                - dividend_yield * spot * df_q * (1.0 - nd1)
                + rate * strike * df_r * (1.0 - nd2)
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
    let nd2 = norm_cdf(d2);
    match option_type {
        OptionType::Call => strike * expiry * df_r * nd2,
        OptionType::Put => -strike * expiry * df_r * (1.0 - nd2),
    }
}

/// Single-pass computation of price + all Greeks.
///
/// Computes d1, d2, discount factors, CDF and PDF values once and derives
/// every output from those shared intermediates. This eliminates the 5x
/// redundant d1/d2 calculations, 6x redundant exp() calls, and 6+
/// redundant CDF/PDF evaluations that the previous per-greek approach had.
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
    // Shared intermediates — computed exactly once.
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln()
        + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let pdf_d1 = norm_pdf(d1);

    // Price: compute call, derive put via put-call parity.
    let s_df_q = spot * df_q;
    let k_df_r = strike * df_r;
    let call = s_df_q.mul_add(nd1, -(k_df_r * nd2));
    let theta_common = -s_df_q * pdf_d1 * vol / (2.0 * sqrt_t);

    let (price, delta, theta) = match option_type {
        OptionType::Call => {
            let d = df_q * nd1;
            let th = theta_common
                + dividend_yield * s_df_q * nd1
                - rate * k_df_r * nd2;
            (call, d, th)
        }
        OptionType::Put => {
            let nmd1 = 1.0 - nd1;
            let nmd2 = 1.0 - nd2;
            let p = call - s_df_q + k_df_r;
            let d = df_q * (nd1 - 1.0);
            let th = theta_common
                - dividend_yield * s_df_q * nmd1
                + rate * k_df_r * nmd2;
            (p, d, th)
        }
    };

    // Greeks that are independent of call/put.
    let gamma = df_q * pdf_d1 / (spot * vol * sqrt_t);
    let vega = spot * df_q * pdf_d1 * sqrt_t;
    let rho = match option_type {
        OptionType::Call => strike * expiry * df_r * nd2,
        OptionType::Put => -strike * expiry * df_r * (1.0 - nd2),
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
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);
        diagnostics.insert_key(crate::core::DiagKey::D1, d1);
        diagnostics.insert_key(crate::core::DiagKey::D2, d2);

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
