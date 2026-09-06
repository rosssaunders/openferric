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
        market.validate()?;
        instrument.validate()?;
        let expiry = match instrument {
            ExoticOption::LookbackFloating(spec) => spec.expiry,
            ExoticOption::LookbackFixed(spec) => spec.expiry,
            ExoticOption::Chooser(spec) => spec.expiry,
            ExoticOption::Quanto(spec) => spec.expiry,
            ExoticOption::Compound(spec) => spec.underlying_expiry,
        };
        market.require_continuous_dividends(expiry)?;

        let mut diagnostics = crate::core::Diagnostics::new();

        let price = match instrument {
            ExoticOption::LookbackFloating(spec) => {
                let vol = market.checked_vol_for(market.spot, spec.expiry.max(1.0e-12))?;
                diagnostics.insert("vol", vol);
                floating_lookback_price(spec, market, vol)
            }
            ExoticOption::LookbackFixed(spec) => {
                let vol = market.checked_vol_for(spec.strike, spec.expiry.max(1.0e-12))?;
                diagnostics.insert("vol", vol);
                fixed_lookback_price(spec, market, vol)
            }
            ExoticOption::Chooser(spec) => {
                let vol = market.checked_vol_for(spec.strike, spec.expiry.max(1.0e-12))?;
                diagnostics.insert("vol", vol);
                chooser_price(spec, market, vol)
            }
            ExoticOption::Quanto(spec) => {
                let vol = market.checked_vol_for(spec.strike, spec.expiry.max(1.0e-12))?;
                diagnostics.insert("vol", vol);
                quanto_price(spec, market, vol)
            }
            ExoticOption::Compound(spec) => {
                let vol = market
                    .checked_vol_for(spec.underlying_strike, spec.underlying_expiry.max(1.0e-12))?;
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
    crate::engines::analytic::black_scholes::bs_price(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        vol,
        expiry,
    )
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

    let q = market.effective_dividend_yield(spec.expiry);
    let b = market.rate - q;
    if vol <= 0.0 {
        let terminal_spot = market.spot * (b * spec.expiry).exp();
        let discount = (-market.rate * spec.expiry).exp();
        return match spec.option_type {
            OptionType::Call => {
                let running_min = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .min(market.spot)
                    .min(terminal_spot);
                discount * (terminal_spot - running_min).max(0.0)
            }
            OptionType::Put => {
                let running_max = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .max(market.spot)
                    .max(terminal_spot);
                discount * (running_max - terminal_spot).max(0.0)
            }
        };
    }

    match spec.option_type {
        OptionType::Call => {
            let s_min = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .min(market.spot);
            floating_lookback_call_formula(market.spot, s_min, market.rate, q, b, vol, spec.expiry)
        }
        OptionType::Put => {
            let s_max = spec
                .observed_extreme
                .unwrap_or(market.spot)
                .max(market.spot);
            floating_lookback_put_formula(market.spot, s_max, market.rate, q, b, vol, spec.expiry)
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

    let q = market.effective_dividend_yield(spec.expiry);
    let carry = market.rate - q;
    if vol <= 0.0 {
        let terminal_spot = market.spot * (carry * spec.expiry).exp();
        let discount = (-market.rate * spec.expiry).exp();
        return match spec.option_type {
            OptionType::Call => {
                let running_max = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .max(market.spot)
                    .max(terminal_spot);
                discount * (running_max - spec.strike).max(0.0)
            }
            OptionType::Put => {
                let running_min = spec
                    .observed_extreme
                    .unwrap_or(market.spot)
                    .min(market.spot)
                    .min(terminal_spot);
                discount * (spec.strike - running_min).max(0.0)
            }
        };
    }

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
                q,
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
                q,
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
    let q_expiry = market.effective_dividend_yield(spec.expiry);
    if vol <= 0.0 && spec.expiry > 0.0 {
        let terminal_spot = market.spot * ((market.rate - q_expiry) * spec.expiry).exp();
        return (-market.rate * spec.expiry).exp() * (terminal_spot - spec.strike).abs();
    }

    let call = bs_price_with_dividend(
        OptionType::Call,
        market.spot,
        spec.strike,
        market.rate,
        q_expiry,
        vol,
        spec.expiry,
    );
    if spec.choose_time <= 0.0 {
        let put = bs_price_with_dividend(
            OptionType::Put,
            market.spot,
            spec.strike,
            market.rate,
            q_expiry,
            vol,
            spec.expiry,
        );
        return call.max(put);
    }

    // Rubinstein (1991) simple chooser (Haug 2.7):
    //   V = c(S, K, T2) + K e^{-r T2} N(-y2) - S e^{-q T2} N(-y1)
    // with y1 = [ln(S/K) + (r - q) T2 + sigma^2 t1 / 2] / (sigma sqrt(t1)),
    //      y2 = y1 - sigma sqrt(t1).
    // At t1 = T2 this reduces to call + put (a straddle).
    let t1 = spec.choose_time.min(spec.expiry);
    let sig_sqrt_t1 = vol * t1.sqrt();
    let y1 = ((market.spot / spec.strike).ln()
        + (market.rate - q_expiry).mul_add(spec.expiry, 0.5 * vol * vol * t1))
        / sig_sqrt_t1;
    let y2 = y1 - sig_sqrt_t1;

    (spec.strike * (-market.rate * spec.expiry).exp()).mul_add(
        normal_cdf(-y2),
        (-(market.spot * (-q_expiry * spec.expiry).exp())).mul_add(normal_cdf(-y1), call),
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

    let q = market.effective_dividend_yield(spec.expiry);
    // Quanto drift adjustment under domestic measure: mu_adj = rf - q - rho * vol * fx_vol
    let mu_adj = (-spec.asset_fx_corr * spec.fx_vol).mul_add(vol, spec.foreign_rate - q);
    if vol <= 0.0 {
        let terminal_spot = market.spot * (mu_adj * spec.expiry).exp();
        let payoff = match spec.option_type {
            OptionType::Call => (terminal_spot - spec.strike).max(0.0),
            OptionType::Put => (spec.strike - terminal_spot).max(0.0),
        };
        return spec.fx_rate * (-market.rate * spec.expiry).exp() * payoff;
    }

    let sqrt_t = spec.expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((market.spot / spec.strike).ln() + (0.5 * vol).mul_add(vol, mu_adj) * spec.expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_dom = (-market.rate * spec.expiry).exp();
    let carry_discounted = ((mu_adj - market.rate) * spec.expiry).exp();

    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = market
        .spot
        .mul_add(carry_discounted * nd1, -(spec.strike * df_dom * nd2));
    let foreign_price = match spec.option_type {
        OptionType::Call => call,
        OptionType::Put => (spec.strike * df_dom).mul_add(
            normal_cdf(-d2),
            -(market.spot * carry_discounted * normal_cdf(-d1)),
        ),
    };

    spec.fx_rate * foreign_price
}

#[inline]
fn compound_price(spec: &CompoundOption, market: &Market, vol: f64) -> Result<f64, PricingError> {
    let q_underlying = market.effective_dividend_yield(spec.underlying_expiry);
    if spec.compound_expiry <= 0.0 {
        let inner = bs_price_with_dividend(
            spec.underlying_option_type,
            market.spot,
            spec.underlying_strike,
            market.rate,
            q_underlying,
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
    let q_t1 = market.effective_dividend_yield(t1.max(1.0e-12));
    let q_tau = market.effective_dividend_yield(tau.max(1.0e-12));
    if vol <= 0.0 {
        let s_t1 = market.spot * ((market.rate - q_t1) * t1).exp();
        let inner = bs_price_with_dividend(
            spec.underlying_option_type,
            s_t1,
            spec.underlying_strike,
            market.rate,
            q_tau,
            vol,
            tau,
        );
        let payoff = match spec.option_type {
            OptionType::Call => (inner - spec.compound_strike).max(0.0),
            OptionType::Put => (spec.compound_strike - inner).max(0.0),
        };
        return Ok((-market.rate * t1).exp() * payoff);
    }

    // drift = (r - q - 0.5 * vol^2) * t1 via FMA
    let drift = (-0.5 * vol).mul_add(vol, market.rate - q_t1) * t1;
    let vol_t1 = vol * t1.sqrt();

    let inner_minus_compound_strike = |z: f64| {
        let s_t1 = market.spot * (drift + vol_t1 * z).exp();
        bs_price_with_dividend(
            spec.underlying_option_type,
            s_t1,
            spec.underlying_strike,
            market.rate,
            q_tau,
            vol,
            tau,
        ) - spec.compound_strike
    };

    // A compound payoff has a kink at the critical T1 spot for which the
    // daughter option is worth the mother strike.  Integrating across that
    // kink with one fixed Gauss-Legendre panel converges slowly (and missed
    // QuantLib's Haug references by more than a cent).  Locate the critical
    // normal variate and integrate only the active payoff interval.  The
    // integrand is smooth on that interval and the same 96 nodes then recover
    // the reference values to their quoted precision.
    const Z_MIN: f64 = -8.0;
    const Z_MAX: f64 = 8.0;
    let mut z_lo = Z_MIN;
    let mut z_hi = Z_MAX;
    let mut f_lo = inner_minus_compound_strike(z_lo);
    let f_hi = inner_minus_compound_strike(z_hi);
    let want_positive = matches!(spec.option_type, OptionType::Call);

    let (integral_lo, integral_hi) = if f_lo.signum() == f_hi.signum() {
        let active = (f_lo > 0.0) == want_positive;
        if !active {
            return Ok(0.0);
        }
        (Z_MIN, Z_MAX)
    } else {
        for _ in 0..80 {
            let z_mid = 0.5 * (z_lo + z_hi);
            let f_mid = inner_minus_compound_strike(z_mid);
            if f_mid == 0.0 {
                z_lo = z_mid;
                z_hi = z_mid;
                break;
            }
            if f_mid.signum() == f_lo.signum() {
                z_lo = z_mid;
                f_lo = f_mid;
            } else {
                z_hi = z_mid;
            }
        }
        let critical_z = 0.5 * (z_lo + z_hi);
        let positive_on_right = matches!(spec.underlying_option_type, OptionType::Call);
        if want_positive == positive_on_right {
            (critical_z, Z_MAX)
        } else {
            (Z_MIN, critical_z)
        }
    };

    // Numerical quadrature approximation to the Geske expectation, split at
    // the critical spot as described above.
    let integral = gauss_legendre_integrate(
        |z| {
            let s_t1 = market.spot * (drift + vol_t1 * z).exp();
            let inner = bs_price_with_dividend(
                spec.underlying_option_type,
                s_t1,
                spec.underlying_strike,
                market.rate,
                q_tau,
                vol,
                tau,
            );

            let payoff = match spec.option_type {
                OptionType::Call => (inner - spec.compound_strike).max(0.0),
                OptionType::Put => (spec.compound_strike - inner).max(0.0),
            };

            payoff * normal_pdf(z)
        },
        integral_lo,
        integral_hi,
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

        // QuantLib's Haug fixture is published as 25.3533.  QuantLib 1.43's
        // AnalyticContinuousFloatingLookbackEngine gives this full-precision
        // value for the exact half-year input used here.
        assert_relative_eq!(price, 25.353_355_271_810_2, epsilon = 5e-12);
    }

    #[test]
    fn fixed_lookback_call_matches_haug_reference_values() {
        // Haug (1998), pp. 63-64 publishes these to four decimals.  These are
        // full-precision QuantLib 1.43
        // AnalyticContinuousFixedLookbackEngine values for the exact inputs.
        let references = [
            (95.0, 0.10, 13.268_722_361_148_942),
            (95.0, 0.20, 18.926_336_989_228_3),
            (95.0, 0.30, 24.985_760_140_325_393),
            (100.0, 0.10, 8.512_575_238_645_372),
            (100.0, 0.20, 14.170_189_866_724_73),
            (100.0, 0.30, 20.229_613_017_821_823),
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
            assert_relative_eq!(price, expected, epsilon = 5e-12);
        }
    }

    #[test]
    fn zero_vol_exotics_equal_discounted_deterministic_cashflows() {
        // The public Market contract currently rejects a zero volatility
        // surface, so exercise the formula layer directly.  These limits also
        // protect callers if the market contract is relaxed in the future.
        let market = Market::builder()
            .spot(100.0)
            .rate(0.04)
            .dividend_yield(0.01)
            .flat_vol(0.20)
            .build()
            .expect("valid test market");

        let assert_cashflow = |label: &str, actual: f64, expected: f64| {
            let tolerance = 64.0 * f64::EPSILON * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
            );
        };

        let expiry = 2.0;
        let terminal = market.spot * ((market.rate - market.dividend_yield) * expiry).exp();
        let discount = (-market.rate * expiry).exp();

        let floating_call = LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry,
            observed_extreme: Some(95.0),
        };
        assert_cashflow(
            "floating lookback call",
            floating_lookback_price(&floating_call, &market, 0.0),
            discount * (terminal - 95.0),
        );

        let floating_put = LookbackFloatingOption {
            option_type: OptionType::Put,
            expiry,
            observed_extreme: Some(110.0),
        };
        assert_cashflow(
            "floating lookback put",
            floating_lookback_price(&floating_put, &market, 0.0),
            discount * (110.0 - terminal),
        );

        let fixed_call = LookbackFixedOption {
            option_type: OptionType::Call,
            strike: 102.0,
            expiry,
            observed_extreme: Some(104.0),
        };
        assert_cashflow(
            "fixed lookback call",
            fixed_lookback_price(&fixed_call, &market, 0.0),
            discount * (terminal - 102.0),
        );

        let fixed_put = LookbackFixedOption {
            option_type: OptionType::Put,
            strike: 103.0,
            expiry,
            observed_extreme: Some(98.0),
        };
        assert_cashflow(
            "fixed lookback put",
            fixed_lookback_price(&fixed_put, &market, 0.0),
            discount * (103.0 - 98.0),
        );

        let chooser = ChooserOption {
            strike: 104.0,
            expiry,
            choose_time: 0.75,
        };
        assert_cashflow(
            "chooser",
            chooser_price(&chooser, &market, 0.0),
            discount * (terminal - chooser.strike).abs(),
        );

        for option_type in [OptionType::Call, OptionType::Put] {
            let quanto = QuantoOption {
                option_type,
                strike: 104.0,
                expiry: 1.25,
                fx_rate: 1.20,
                foreign_rate: 0.025,
                fx_vol: 0.18,
                asset_fx_corr: -0.40,
            };
            let terminal =
                market.spot * ((quanto.foreign_rate - market.dividend_yield) * quanto.expiry).exp();
            let payoff = match option_type {
                OptionType::Call => (terminal - quanto.strike).max(0.0),
                OptionType::Put => (quanto.strike - terminal).max(0.0),
            };
            assert_cashflow(
                "quanto",
                quanto_price(&quanto, &market, 0.0),
                quanto.fx_rate * (-market.rate * quanto.expiry).exp() * payoff,
            );
        }

        for (outer_type, inner_type, mother_strike) in [
            (OptionType::Call, OptionType::Call, 4.0),
            (OptionType::Call, OptionType::Put, 4.0),
            (OptionType::Put, OptionType::Call, 6.0),
            (OptionType::Put, OptionType::Put, 6.0),
        ] {
            let compound = CompoundOption {
                option_type: outer_type,
                underlying_option_type: inner_type,
                compound_strike: mother_strike,
                underlying_strike: match inner_type {
                    OptionType::Call => 100.0,
                    OptionType::Put => 110.0,
                },
                compound_expiry: 0.5,
                underlying_expiry: 1.5,
            };
            let s_t1 = market.spot
                * ((market.rate - market.dividend_yield) * compound.compound_expiry).exp();
            let tau = compound.underlying_expiry - compound.compound_expiry;
            let s_t2 = s_t1 * ((market.rate - market.dividend_yield) * tau).exp();
            let inner_payoff = match inner_type {
                OptionType::Call => (s_t2 - compound.underlying_strike).max(0.0),
                OptionType::Put => (compound.underlying_strike - s_t2).max(0.0),
            };
            let inner_at_t1 = (-market.rate * tau).exp() * inner_payoff;
            let outer_payoff = match outer_type {
                OptionType::Call => (inner_at_t1 - mother_strike).max(0.0),
                OptionType::Put => (mother_strike - inner_at_t1).max(0.0),
            };
            assert_cashflow(
                "compound",
                compound_price(&compound, &market, 0.0).expect("deterministic compound price"),
                (-market.rate * compound.compound_expiry).exp() * outer_payoff,
            );
        }

        // With negative carry, the terminal point is the running minimum.
        let negative_carry_market = Market::builder()
            .spot(100.0)
            .rate(0.01)
            .dividend_yield(0.05)
            .flat_vol(0.20)
            .build()
            .expect("valid test market");
        let falling_terminal = 100.0 * (-0.04_f64 * expiry).exp();
        let falling_fixed_put = LookbackFixedOption {
            option_type: OptionType::Put,
            strike: 103.0,
            expiry,
            observed_extreme: Some(98.0),
        };
        assert_cashflow(
            "falling fixed lookback put",
            fixed_lookback_price(&falling_fixed_put, &negative_carry_market, 0.0),
            (-0.01_f64 * expiry).exp() * (103.0 - falling_terminal),
        );
    }

    #[test]
    fn mc_floating_lookback_matches_independent_scipy_sobol_reference() {
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

        let (mc, stderr) = MonteCarloEngine::new(50_000, 42).with_antithetic(true).run(
            &generator,
            |path| {
                let path_min = path.iter().fold(observed_min, |acc, &s| acc.min(s));
                (path[path.len() - 1] - path_min).max(0.0)
            },
            discount_factor,
        );

        // Independent discrete-monitoring reference: SciPy 1.17.1 Sobol,
        // 16 Owen-scrambled replicates of 2^14 paths in 756 dimensions.
        let reference = 14.816_567_593_152_019;
        let reference_stderr = 0.003_684_381_732_446_104_7;
        let combined_stderr = (stderr * stderr + reference_stderr * reference_stderr).sqrt();
        let tolerance = 4.0 * combined_stderr + 1.0e-12;
        assert!(
            (mc - reference).abs() <= tolerance,
            "MC/SciPy floating-lookback error exceeds combined sampling budget: mc={} reference={} implementation_stderr={} reference_stderr={} tolerance={}",
            mc,
            reference,
            stderr,
            reference_stderr,
            tolerance
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
        let ulp_error = price.to_bits().abs_diff(vanilla.to_bits());
        assert!(
            ulp_error <= 4,
            "zero-FX quanto reduction differs from vanilla by {ulp_error} ulps"
        );
    }

    #[test]
    fn compound_option_matches_independent_scipy_quadrature() {
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

        // Independent SciPy 1.17.1 adaptive quadrature of the discounted T1
        // payoff over the terminal-normal density (epsabs=epsrel=1e-14).
        // This replaces the former non-negativity-only assertion.  QuantLib
        // 1.43's bivariate-normal implementation gives 6.818216699922273 for
        // the same fixture; the direct one-dimensional expectation is the
        // more accurate oracle here.
        assert_relative_eq!(price, 6.818_232_078_481_023, epsilon = 5e-12);
    }
}
