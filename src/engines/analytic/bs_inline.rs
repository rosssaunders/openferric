//! Module `engines::analytic::bs_inline`.
//!
//! Implements bs inline workflows with concrete routines such as `has_fma_bs_kernel`, `has_fma_bs_kernel`, `bs_price_asm`.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Primary API surface: free functions `has_fma_bs_kernel`, `has_fma_bs_kernel`, `bs_price_asm`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.

use crate::math::{normal_cdf, normal_pdf};

pub(crate) const ACCURATE_CDF_TOTAL_VOL: f64 = 1.0e-3;
const LOG_DOMAIN_TAIL_MIDPOINT: f64 = 8.0;

/// `ln(spot / strike)` without overflowing the ratio or erasing adjacent-f64
/// moneyness near the strike.
#[inline]
pub(crate) fn stable_log_spot_strike(spot: f64, strike: f64) -> f64 {
    let relative_moneyness = (spot - strike) / strike;
    if relative_moneyness.abs() <= 0.5 {
        relative_moneyness.ln_1p()
    } else {
        spot.ln() - strike.ln()
    }
}

/// Black-Scholes `d1`/`d2` with a log-domain total-volatility division when
/// `vol * sqrt(expiry)` rounds to zero.
#[inline]
pub(crate) fn stable_d1_d2(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, f64) {
    let width = vol * expiry.sqrt();
    let numerator = stable_log_spot_strike(spot, strike)
        + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry;
    let d1 = if width != 0.0 {
        numerator / width
    } else if numerator == 0.0 {
        0.0
    } else {
        let log_width = vol.ln() + 0.5 * expiry.ln();
        numerator.signum() * (numerator.abs().ln() - log_width).exp()
    };
    (d1, d1 - width)
}

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

/// Black-Scholes price for a small positive total volatility without
/// subtracting two nearly equal CDF values.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn stable_short_total_vol_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> f64 {
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    if spot == 0.0 {
        return if is_call { 0.0 } else { strike * df_r };
    }
    if strike == 0.0 {
        return if is_call { spot * df_q } else { 0.0 };
    }

    #[inline]
    fn nonnegative_or_nan(value: f64) -> f64 {
        if value.is_nan() {
            f64::NAN
        } else {
            value.max(0.0)
        }
    }

    #[inline]
    fn discounted_intrinsic(is_call: bool, discounted_spot: f64, discounted_strike: f64) -> f64 {
        let value = if is_call {
            discounted_spot - discounted_strike
        } else {
            discounted_strike - discounted_spot
        };
        nonnegative_or_nan(value)
    }

    #[inline]
    fn log_positive_expm1(value: f64) -> f64 {
        if value > 50.0 {
            value + (-(-value).exp()).ln_1p()
        } else {
            value.exp_m1().ln()
        }
    }

    #[inline]
    fn log_negative_expm1(value: f64) -> f64 {
        (-value.exp()).ln_1p()
    }

    #[inline]
    fn log_mills_ratio_difference(lower: f64, width: f64, log_width: f64) -> f64 {
        // R(x) = Phi(-x)/phi(x)
        //      ~ 1/x - 1/x^3 + 3/x^5 - 15/x^7 + ...
        // Factor out width before evaluating R(lower)-R(lower+width), so a
        // subnormal dimensionless difference is not lost before a very large
        // strike can rescale it into the representable price range.
        let log_step = if width == 0.0 {
            0.0
        } else {
            (width / lower).ln_1p()
        };
        let inverse = 1.0 / lower;
        let inverse_sq = inverse * inverse;
        let mut power = inverse;
        let mut coefficient = 1.0;
        let mut sum_over_width = 0.0;
        let mut previous_abs = f64::INFINITY;
        for order in 0..128 {
            let exponent = (2 * order + 1) as f64;
            let numerator = -(-exponent * log_step).exp_m1();
            let difference_ratio = if width > 0.0 && numerator > 0.0 {
                numerator / width
            } else {
                exponent / lower
            };
            let term = coefficient * power * difference_ratio;
            if term.abs() >= previous_abs {
                break;
            }
            let updated = sum_over_width + term;
            if updated == sum_over_width && order > 0 {
                break;
            }
            sum_over_width = updated;
            previous_abs = term.abs();
            power *= inverse_sq;
            coefficient *= -((2 * order + 1) as f64);
        }
        log_width + sum_over_width.ln()
    }

    // Division rounds an adjacent-f64 ratio much more coarsely than the
    // original spot/strike difference. Use log1p in the near-ATM region and
    // separate logarithms outside it to avoid both that loss and ratio
    // overflow for finite extreme inputs.
    let log_forward_moneyness =
        stable_log_spot_strike(spot, strike) + (rate - dividend_yield) * expiry;
    let sqrt_t = expiry.sqrt();
    let width = vol * sqrt_t;
    let log_width = vol.ln() + 0.5 * expiry.ln();
    let discounted_spot = spot * df_q;
    let discounted_strike = strike * df_r;

    if log_forward_moneyness.is_nan() {
        return f64::NAN;
    }
    if log_forward_moneyness.is_infinite() {
        return discounted_intrinsic(is_call, discounted_spot, discounted_strike);
    }

    let midpoint = if width == 0.0 {
        if log_forward_moneyness == 0.0 {
            0.0
        } else {
            log_forward_moneyness.signum() * (log_forward_moneyness.abs().ln() - log_width).exp()
        }
    } else {
        log_forward_moneyness / width
    };
    let d1 = midpoint + 0.5 * width;
    let d2 = midpoint - 0.5 * width;

    // CDF/PDF values can underflow while multiplication by a very large
    // finite spot or strike would bring the option value back into range.
    // Price the OTM leg as a log-domain difference and recover the ITM leg
    // through put-call parity.
    if midpoint.abs() >= LOG_DOMAIN_TAIL_MIDPOINT {
        let log_discounted_strike = strike.ln() - rate * expiry;
        let absolute_midpoint = midpoint.abs();
        let lower = absolute_midpoint - 0.5 * width;
        let upper = absolute_midpoint + 0.5 * width;
        let log_mills_difference = log_mills_ratio_difference(lower, width, log_width);
        let (call, put) = if log_forward_moneyness < 0.0 {
            let log_call =
                log_discounted_strike - 0.5 * upper * upper - 0.5 * std::f64::consts::TAU.ln()
                    + log_mills_difference;
            let call = log_call.exp();
            let direct_intrinsic = discounted_intrinsic(false, discounted_spot, discounted_strike);
            let put_intrinsic = if direct_intrinsic.is_nan() {
                (log_discounted_strike + log_negative_expm1(log_forward_moneyness)).exp()
            } else {
                direct_intrinsic
            };
            (call, call + put_intrinsic)
        } else {
            let log_put =
                log_discounted_strike - 0.5 * lower * lower - 0.5 * std::f64::consts::TAU.ln()
                    + log_mills_difference;
            let put = log_put.exp();
            let direct_intrinsic = discounted_intrinsic(true, discounted_spot, discounted_strike);
            let call_intrinsic = if direct_intrinsic.is_nan() {
                (log_discounted_strike + log_positive_expm1(log_forward_moneyness)).exp()
            } else {
                direct_intrinsic
            };
            (put + call_intrinsic, put)
        };
        return if is_call {
            nonnegative_or_nan(call)
        } else {
            nonnegative_or_nan(put)
        };
    }

    let interval_density = if width <= ACCURATE_CDF_TOTAL_VOL {
        let m2 = midpoint * midpoint;
        let w2 = width * width;
        let correction2 = (m2 - 1.0) * w2 / 24.0;
        let correction4 = (m2.mul_add(m2, -6.0 * m2) + 3.0) * w2 * w2 / 1_920.0;
        normal_pdf(midpoint) * (1.0 + correction2 + correction4)
    } else {
        (normal_cdf(d1) - normal_cdf(d2)) / width
    };

    // Multiplying the price scale before vol/sqrt(T) preserves representable
    // time value when the dimensionless total volatility itself underflows.
    #[inline]
    fn scaled_width(scale: f64, width: f64, vol: f64, sqrt_t: f64) -> f64 {
        if width != 0.0 {
            return scale * width;
        }
        let scale_vol_first = (scale * vol) * sqrt_t;
        if scale_vol_first.is_finite() && scale_vol_first != 0.0 {
            scale_vol_first
        } else {
            (scale * sqrt_t) * vol
        }
    }

    if !discounted_spot.is_finite() || !discounted_strike.is_finite() {
        // Keep the finite dimensionless option value separate from an
        // overflowing discount scale. Factoring out total volatility also
        // covers subnormal widths whose price becomes representable only
        // after the scale is applied.
        let forward_excess = log_forward_moneyness.exp_m1();
        let excess_over_width = if width != 0.0 && forward_excess != 0.0 {
            forward_excess / width
        } else {
            midpoint
        };
        let call_over_width = log_forward_moneyness
            .exp()
            .mul_add(interval_density, excess_over_width * normal_cdf(d2));
        let put_over_width = interval_density - excess_over_width * normal_cdf(-d1);
        let dimensionless = if is_call {
            call_over_width
        } else {
            put_over_width
        };
        if dimensionless.is_nan() || dimensionless < 0.0 {
            return f64::NAN;
        }
        return (strike.ln() - rate * expiry + log_width + dimensionless.ln()).exp();
    }

    let discounted_forward_excess = discounted_spot - discounted_strike;
    let call = scaled_width(discounted_spot, width, vol, sqrt_t)
        .mul_add(interval_density, discounted_forward_excess * normal_cdf(d2));
    let put = scaled_width(discounted_strike, width, vol, sqrt_t).mul_add(
        interval_density,
        -(discounted_forward_excess * normal_cdf(-d1)),
    );
    if is_call {
        nonnegative_or_nan(call)
    } else {
        nonnegative_or_nan(put)
    }
}

#[inline]
fn intrinsic(is_call: bool, spot: f64, strike: f64) -> f64 {
    if is_call {
        (spot - strike).max(0.0)
    } else {
        (strike - spot).max(0.0)
    }
}

#[inline]
fn bs_price_scalar_reference(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> f64 {
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry) {
        return f64::NAN;
    }
    if expiry <= 0.0 {
        return intrinsic(is_call, spot, strike);
    }
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    if spot == 0.0 {
        return if is_call { 0.0 } else { strike * df_r };
    }
    if strike == 0.0 {
        return if is_call { spot * df_q } else { 0.0 };
    }
    if vol == 0.0 {
        return if is_call {
            (spot * df_q - strike * df_r).max(0.0)
        } else {
            (strike * df_r - spot * df_q).max(0.0)
        };
    }
    if !(spot * df_q).is_finite() || !(strike * df_r).is_finite() {
        return stable_short_total_vol_price(
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
        );
    }

    let sig_sqrt_t = vol * expiry.sqrt();
    if sig_sqrt_t <= ACCURATE_CDF_TOTAL_VOL {
        return stable_short_total_vol_price(
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
        );
    }
    let (d1, d2) = stable_d1_d2(spot, strike, rate, dividend_yield, vol, expiry);
    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = spot.mul_add(df_q * nd1, -(strike * df_r * nd2));
    if is_call {
        call
    } else {
        call - spot * df_q + strike * df_r
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_fma_bs_kernel() -> bool {
    std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_fma_bs_kernel() -> bool {
    false
}

/// Safe wrapper around the x86 FMA/asm Black-Scholes hot path.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn bs_price_asm(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> f64 {
    let nonfinite_discount_scale = expiry > 0.0
        && ((spot * (-dividend_yield * expiry).exp()).is_infinite()
            || (strike * (-rate * expiry).exp()).is_infinite());
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry)
        || expiry <= 0.0
        || vol == 0.0
        || spot == 0.0
        || strike == 0.0
        || vol * expiry.sqrt() <= ACCURATE_CDF_TOTAL_VOL
        || nonfinite_discount_scale
    {
        return bs_price_scalar_reference(spot, strike, rate, dividend_yield, vol, expiry, is_call);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_fma_bs_kernel() {
            // SAFETY: runtime-detected AVX/FMA support.
            return unsafe {
                bs_price_asm_impl(spot, strike, rate, dividend_yield, vol, expiry, is_call)
            };
        }
    }

    bs_price_scalar_reference(spot, strike, rate, dividend_yield, vol, expiry, is_call)
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx,fma")]
unsafe fn bs_price_asm_impl(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> f64 {
    use std::arch::asm;

    if expiry <= 0.0 {
        return intrinsic(is_call, spot, strike);
    }

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    if vol <= 0.0 {
        return if is_call {
            (spot * df_q - strike * df_r).max(0.0)
        } else {
            (strike * df_r - spot * df_q).max(0.0)
        };
    }

    let mut sqrt_t = expiry;
    // SAFETY: executed only with AVX/FMA enabled by target_feature and runtime detection.
    unsafe {
        asm!(
            "vsqrtsd {x}, {x}, {x}",
            x = inout(xmm_reg) sqrt_t,
            options(pure, nomem, nostack),
        );
    }

    let sig_sqrt_t = vol * sqrt_t;
    let ln_sk = stable_log_spot_strike(spot, strike);

    let mut drift_t = (rate - dividend_yield) * expiry;
    let vol2 = vol * vol;
    let half_t = 0.5 * expiry;
    // SAFETY: executed only with AVX/FMA enabled by target_feature and runtime detection.
    unsafe {
        asm!(
            "vfmadd231sd {acc}, {vol2}, {half_t}",
            acc = inout(xmm_reg) drift_t,
            vol2 = in(xmm_reg) vol2,
            half_t = in(xmm_reg) half_t,
            options(pure, nomem, nostack),
        );
    }

    let (d1, d2) = if sig_sqrt_t == 0.0 {
        stable_d1_d2(spot, strike, rate, dividend_yield, vol, expiry)
    } else {
        let d1 = (ln_sk + drift_t) / sig_sqrt_t;
        (d1, d1 - sig_sqrt_t)
    };

    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = spot.mul_add(df_q * nd1, -(strike * df_r * nd2));
    if is_call {
        call
    } else {
        call - spot * df_q + strike * df_r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asm_wrapper_matches_reference_price_within_1e14() {
        let cases = [
            (100.0, 100.0, 0.03, 0.00, 0.20, 1.00, true),
            (120.0, 100.0, 0.01, 0.02, 0.15, 0.50, true),
            (90.0, 100.0, 0.05, 0.01, 0.30, 2.00, false),
            (75.0, 80.0, 0.00, 0.00, 0.10, 0.25, false),
            (100.0, 100.0, 0.02, 0.00, 0.00, 1.00, true),
        ];

        for (s, k, r, q, vol, t, is_call) in cases {
            let fast = bs_price_asm(s, k, r, q, vol, t, is_call);
            let reference = bs_price_scalar_reference(s, k, r, q, vol, t, is_call);
            assert!(
                (fast - reference).abs() <= 1e-14,
                "s={s} k={k} r={r} q={q} vol={vol} t={t} is_call={is_call} fast={fast} ref={reference}",
            );
        }
    }
}
