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
use crate::math::{black_scholes_price_greeks_aad, normal_cdf, normal_pdf};

#[inline]
fn invalid_inputs(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> bool {
    !spot.is_finite()
        || !strike.is_finite()
        || !rate.is_finite()
        || !dividend_yield.is_finite()
        || !vol.is_finite()
        || !expiry.is_finite()
        || spot < 0.0
        || strike < 0.0
        || vol < 0.0
        || (spot == 0.0 && strike == 0.0)
}

#[inline]
fn deterministic_delta_theta_rho(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
) -> (f64, f64, f64) {
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    let forward_value = spot * df_q - strike * df_r;

    match forward_value.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) if option_type == OptionType::Call => (
            df_q,
            dividend_yield * spot * df_q - rate * strike * df_r,
            strike * expiry * df_r,
        ),
        Some(std::cmp::Ordering::Less) if option_type == OptionType::Put => (
            -df_q,
            -dividend_yield * spot * df_q + rate * strike * df_r,
            -strike * expiry * df_r,
        ),
        Some(std::cmp::Ordering::Equal) => (f64::NAN, f64::NAN, f64::NAN),
        Some(_) => (0.0, 0.0, 0.0),
        None => (f64::NAN, f64::NAN, f64::NAN),
    }
}

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
    super::bs_inline::stable_d1_d2(spot, strike, rate, dividend_yield, vol, expiry)
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return intrinsic(option_type, spot, strike);
    }
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    if spot == 0.0 {
        return if matches!(option_type, OptionType::Call) {
            0.0
        } else {
            strike * df_r
        };
    }
    if strike == 0.0 {
        return if matches!(option_type, OptionType::Call) {
            spot * df_q
        } else {
            0.0
        };
    }
    if vol == 0.0 {
        return match option_type {
            OptionType::Call => (spot * df_q - strike * df_r).max(0.0),
            OptionType::Put => (strike * df_r - spot * df_q).max(0.0),
        };
    }
    let is_call = matches!(option_type, OptionType::Call);
    if super::bs_inline::requires_stable_price(
        spot,
        strike,
        rate,
        dividend_yield,
        vol,
        expiry,
        is_call,
    ) {
        return super::bs_inline::stable_short_total_vol_price(
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
        );
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
            is_call,
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return 0.0;
    }
    if vol == 0.0 || spot == 0.0 || strike == 0.0 {
        return deterministic_delta_theta_rho(
            option_type,
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        )
        .0;
    }
    let (d1, _) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_q = (-dividend_yield * expiry).exp();
    match option_type {
        OptionType::Call => df_q * norm_cdf(d1),
        OptionType::Put => -df_q * norm_cdf(-d1),
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return 0.0;
    }
    if vol == 0.0 {
        let (delta, _, _) = deterministic_delta_theta_rho(
            OptionType::Call,
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        );
        return if delta.is_nan() { f64::NAN } else { 0.0 };
    }
    if spot == 0.0 || strike == 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_q = (-dividend_yield * expiry).exp();
    super::bs_inline::stable_gamma(
        df_q * norm_pdf(d1),
        spot,
        vol,
        expiry.sqrt(),
        d1,
        -dividend_yield * expiry,
    )
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return 0.0;
    }
    if vol == 0.0 {
        let (delta, _, _) = deterministic_delta_theta_rho(
            OptionType::Call,
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        );
        return if delta.is_nan() { f64::NAN } else { 0.0 };
    }
    if spot == 0.0 || strike == 0.0 {
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return 0.0;
    }
    if vol == 0.0 || spot == 0.0 || strike == 0.0 {
        return deterministic_delta_theta_rho(
            option_type,
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        )
        .1;
    }
    let (d1, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let sqrt_t = expiry.sqrt();
    let df_q = (-dividend_yield * expiry).exp();
    let df_r = (-rate * expiry).exp();
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let theta_common =
        super::bs_inline::stable_theta_diffusion(spot * df_q * norm_pdf(d1), vol, sqrt_t);
    match option_type {
        OptionType::Call => {
            theta_common + dividend_yield * spot * df_q * nd1 - rate * strike * df_r * nd2
        }
        OptionType::Put => {
            theta_common - dividend_yield * spot * df_q * norm_cdf(-d1)
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return 0.0;
    }
    if vol == 0.0 || spot == 0.0 || strike == 0.0 {
        return deterministic_delta_theta_rho(
            option_type,
            spot,
            strike,
            rate,
            dividend_yield,
            expiry,
        )
        .2;
    }
    let (_, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    let df_r = (-rate * expiry).exp();
    let nd2 = norm_cdf(d2);
    match option_type {
        OptionType::Call => strike * expiry * df_r * nd2,
        OptionType::Put => -strike * expiry * df_r * norm_cdf(-d2),
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
    if spot == 0.0 || strike == 0.0 {
        let d = if spot == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        return (
            bs_price(option_type, spot, strike, rate, dividend_yield, vol, expiry),
            Greeks {
                delta: bs_delta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
                gamma: bs_gamma(spot, strike, rate, dividend_yield, vol, expiry),
                vega: bs_vega(spot, strike, rate, dividend_yield, vol, expiry),
                theta: bs_theta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
                rho: bs_rho(option_type, spot, strike, rate, dividend_yield, vol, expiry),
            },
            d,
            d,
        );
    }

    // Shared intermediates — computed exactly once.
    let sqrt_t = expiry.sqrt();
    let (d1, d2) = d1_d2(spot, strike, rate, dividend_yield, vol, expiry);

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let pdf_d1 = norm_pdf(d1);

    // Price: compute call, derive put via put-call parity except where that
    // subtraction would erase tiny but representable time value.
    let s_df_q = spot * df_q;
    let k_df_r = strike * df_r;
    let call = s_df_q.mul_add(nd1, -(k_df_r * nd2));
    let is_call = matches!(option_type, OptionType::Call);
    let sensitive_price = super::bs_inline::requires_stable_price(
        spot,
        strike,
        rate,
        dividend_yield,
        vol,
        expiry,
        is_call,
    )
    .then(|| {
        super::bs_inline::stable_short_total_vol_price(
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
        )
    });
    let theta_common = super::bs_inline::stable_theta_diffusion(s_df_q * pdf_d1, vol, sqrt_t);

    let (price, delta, theta) = match option_type {
        OptionType::Call => {
            let d = df_q * nd1;
            let th = theta_common + dividend_yield * s_df_q * nd1 - rate * k_df_r * nd2;
            (sensitive_price.unwrap_or(call), d, th)
        }
        OptionType::Put => {
            let nmd1 = norm_cdf(-d1);
            let nmd2 = norm_cdf(-d2);
            let p = call - s_df_q + k_df_r;
            let d = -df_q * nmd1;
            let th = theta_common - dividend_yield * s_df_q * nmd1 + rate * k_df_r * nmd2;
            (sensitive_price.unwrap_or(p), d, th)
        }
    };

    // Greeks that are independent of call/put.
    let gamma = super::bs_inline::stable_gamma(
        df_q * pdf_d1,
        spot,
        vol,
        sqrt_t,
        d1,
        -dividend_yield * expiry,
    );
    let vega = spot * df_q * pdf_d1 * sqrt_t;
    let rho = match option_type {
        OptionType::Call => strike * expiry * df_r * nd2,
        OptionType::Put => -strike * expiry * df_r * norm_cdf(-d2),
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
        market.validate()?;
        instrument.validate()?;

        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "BlackScholesEngine supports European exercise only".to_string(),
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

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;

        let (spot, dividend_yield) = if market.has_discrete_dividends() {
            (market.prepaid_forward_spot(instrument.expiry), 0.0)
        } else {
            (market.spot, market.dividend_yield)
        };

        let (price, greeks, d1, d2) = bs_price_greeks_with_dividend(
            instrument.option_type,
            spot,
            instrument.strike,
            market.rate,
            dividend_yield,
            vol,
            instrument.expiry,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);
        if d1.is_finite() {
            diagnostics.insert_key(crate::core::DiagKey::D1, d1);
        }
        if d2.is_finite() {
            diagnostics.insert_key(crate::core::DiagKey::D2, d2);
        }

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
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;
        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "BlackScholesEngine supports European exercise only".to_string(),
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

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;

        let (spot, dividend_yield) = if market.has_discrete_dividends() {
            (market.prepaid_forward_spot(instrument.expiry), 0.0)
        } else {
            (market.spot, market.dividend_yield)
        };

        let (price, greeks) = black_scholes_price_greeks_aad(
            instrument.option_type,
            spot,
            instrument.strike,
            market.rate,
            dividend_yield,
            vol,
            instrument.expiry,
        );
        let (d1, d2) = if instrument.expiry > 0.0 && vol > 0.0 {
            d1_d2(
                spot,
                instrument.strike,
                market.rate,
                dividend_yield,
                vol,
                instrument.expiry,
            )
        } else {
            (0.0, 0.0)
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);
        if d1.is_finite() {
            diagnostics.insert_key(crate::core::DiagKey::D1, d1);
        }
        if d2.is_finite() {
            diagnostics.insert_key(crate::core::DiagKey::D2, d2);
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_tail_price_is_nonnegative_and_homogeneous() {
        let unit = bs_price(OptionType::Put, 5.0, 1.0, 0.0, 0.0, 0.2, 1.0);
        let scaled = bs_price(OptionType::Put, 5.0e18, 1.0e18, 0.0, 0.0, 0.2, 1.0);
        const EXPECTED_SCALED: f64 = 22.752_884_600_977_636;
        assert!(unit >= 0.0);
        assert!(scaled >= 0.0);
        assert!(
            ((scaled - EXPECTED_SCALED) / EXPECTED_SCALED).abs() <= 2.0e-12,
            "scaled={scaled:.17e}"
        );
        assert!(
            ((scaled / 1.0e18 - unit) / unit).abs() <= 2.0e-12,
            "unit={unit:.17e} scaled={scaled:.17e}"
        );
    }

    #[test]
    fn gamma_avoids_overflowing_spot_times_total_volatility() {
        // Independent closed-form constant: phi(1) / (2 * 1e308).
        const EXPECTED: f64 = 1.209_853_622_595_72e-309;
        let gamma = bs_gamma(1.0e308, 1.0e308, 0.0, 0.0, 2.0, 1.0);
        assert!(gamma > 0.0, "gamma={gamma:.17e}");
        assert!(
            ((gamma - EXPECTED) / EXPECTED).abs() <= 2.0e-14,
            "gamma={gamma:.17e} expected={EXPECTED:.17e}"
        );
    }

    #[test]
    fn gamma_is_zero_not_nan_when_pdf_and_total_volatility_underflow() {
        let gamma = bs_gamma(2.0, 1.0, 0.0, 0.0, 1.0e-300, 1.0e-100);
        assert!(gamma.is_finite(), "gamma={gamma}");
        assert_eq!(gamma, 0.0);
    }
}
