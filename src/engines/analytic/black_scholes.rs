//! Closed-form Black-Scholes-Merton pricing and Greeks.
//!
//! Implements European vanilla formulas under lognormal diffusion
//! (Hull, Ch. 15) with continuous dividend yield `q`.
//! Formulas are written in numerically robust branches for `T -> 0`
//! and `sigma -> 0`.

use crate::core::{ExerciseStyle, Greeks, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::{normal_cdf, normal_pdf};

/// Analytic Black-Scholes engine for European vanilla options.
#[derive(Debug, Clone, Default)]
pub struct BlackScholesEngine;

impl BlackScholesEngine {
    /// Creates a stateless Black-Scholes engine instance.
    ///
    /// # Examples
    /// ```
    /// use openferric::engines::analytic::BlackScholesEngine;
    ///
    /// let engine = BlackScholesEngine::new();
    /// let _ = engine;
    /// ```
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

/// Black-Scholes-Merton price for a European vanilla option.
///
/// Parameters follow standard notation:
/// - `spot` = current underlying level `S_0`
/// - `strike` = strike `K`
/// - `rate` = continuously-compounded risk-free rate `r`
/// - `dividend_yield` = continuous carry/dividend yield `q`
/// - `vol` = lognormal volatility `sigma`
/// - `expiry` = year fraction to maturity `T`
///
/// Uses the closed-form equations in Hull, Ch. 15.
///
/// # Numerical stability
/// - Returns intrinsic payoff when `expiry <= 0`.
/// - Returns discounted intrinsic forward value when `vol <= 0`.
///
/// # Examples
/// ```
/// use openferric::core::OptionType;
/// use openferric::engines::analytic::black_scholes::bs_price;
///
/// let call = bs_price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
/// assert!((call - 10.4506).abs() < 1e-3);
/// ```
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
    // Edge handling for expiry/volatility degeneracies.
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

/// Spot delta `dV/dS` under Black-Scholes-Merton.
///
/// Reference: Hull, Ch. 15.
///
/// # Edge cases
/// Returns `0.0` when `expiry <= 0` or `vol <= 0`.
///
/// # Examples
/// ```
/// use openferric::core::OptionType;
/// use openferric::engines::analytic::black_scholes::bs_delta;
///
/// let d = bs_delta(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
/// assert!(d > 0.5 && d < 0.7);
/// ```
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

/// Spot gamma `d^2V/dS^2` under Black-Scholes-Merton.
///
/// # Numerical stability
/// Returns `0.0` when `spot <= 0`, `expiry <= 0`, or `vol <= 0` to avoid
/// singular behavior as denominator terms vanish.
///
/// # Examples
/// ```
/// use openferric::engines::analytic::black_scholes::bs_gamma;
///
/// let g = bs_gamma(100.0, 100.0, 0.03, 0.0, 0.2, 1.0);
/// assert!(g > 0.0);
/// ```
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

/// Black-Scholes vega `dV/dsigma`.
///
/// Vega is quoted per unit volatility (e.g., `0.01` vol points corresponds
/// to `vega * 0.01` price change).
///
/// # Examples
/// ```
/// use openferric::engines::analytic::black_scholes::bs_vega;
///
/// let v = bs_vega(100.0, 100.0, 0.01, 0.0, 0.2, 1.0);
/// assert!(v > 0.0);
/// ```
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

/// Calendar theta `dV/dT` in annualized units.
///
/// Reference: Hull, Ch. 15 sign conventions.
///
/// # Limitations
/// Theta here is with respect to increasing maturity `T` (not day-decay),
/// so users should negate and scale if they need daily carry convention.
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

/// Interest-rate rho `dV/dr` under Black-Scholes-Merton.
///
/// # Examples
/// ```
/// use openferric::core::OptionType;
/// use openferric::engines::analytic::black_scholes::bs_rho;
///
/// let rho = bs_rho(OptionType::Call, 100.0, 100.0, 0.02, 0.0, 0.25, 1.0);
/// assert!(rho > 0.0);
/// ```
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
///
/// Equivalent to constructing a [`VanillaOption`] with European exercise,
/// creating a [`Market`] with flat volatility, and invoking
/// [`BlackScholesEngine::price`].
///
/// # Errors
/// Returns [`PricingError`] for invalid inputs (for example non-positive
/// spot/volatility through market validation).
///
/// # Examples
/// ```
/// use openferric::core::OptionType;
/// use openferric::engines::analytic::black_scholes;
///
/// let price = black_scholes(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0).unwrap();
/// assert!((price - 10.4506).abs() < 1e-3);
/// ```
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
