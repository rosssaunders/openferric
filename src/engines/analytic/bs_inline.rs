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

/// `(rate - dividend_yield) * expiry` without first overflowing the rate
/// difference when multiplication by a small expiry would bring it back into
/// range.
#[inline]
pub(crate) fn stable_carry(rate: f64, dividend_yield: f64, expiry: f64) -> f64 {
    let direct = (rate - dividend_yield) * expiry;
    if direct.is_finite() {
        direct
    } else {
        let scaled_separately = rate * expiry - dividend_yield * expiry;
        if scaled_separately.is_finite() {
            scaled_separately
        } else {
            direct
        }
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
    let log_forward_moneyness =
        stable_log_spot_strike(spot, strike) + stable_carry(rate, dividend_yield, expiry);
    let midpoint = if width != 0.0 {
        log_forward_moneyness / width
    } else if log_forward_moneyness == 0.0 {
        0.0
    } else {
        let log_width = vol.ln() + 0.5 * expiry.ln();
        log_forward_moneyness.signum() * (log_forward_moneyness.abs().ln() - log_width).exp()
    };
    let half_width = 0.5 * width;
    (midpoint + half_width, midpoint - half_width)
}

/// Scale the Black-Scholes gamma numerator by `spot * vol * sqrt(expiry)`.
///
/// The ordinary arithmetic path is retained for common inputs. If its PDF or
/// denominator underflows/overflows, reconstruct gamma from logarithms so
/// `0 / 0` does not become NaN and a representable subnormal result is not
/// erased.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn stable_gamma(
    numerator: f64,
    spot: f64,
    vol: f64,
    sqrt_expiry: f64,
    d1: f64,
    log_discount: f64,
) -> f64 {
    let grouped_denominator = (spot * vol) * sqrt_expiry;
    if grouped_denominator.is_finite() && grouped_denominator != 0.0 {
        let gamma = numerator / grouped_denominator;
        if gamma.is_finite() && gamma != 0.0 {
            return gamma;
        }
    }

    let log_pdf = -0.5 * d1 * d1 - 0.5 * std::f64::consts::TAU.ln();
    let log_total_vol = vol.ln() + sqrt_expiry.ln();
    (log_discount + log_pdf - spot.ln() - log_total_vol).exp()
}

/// Black-Scholes diffusion theta term
/// `-scale * vol / (2 * sqrt(expiry))` with multiplication ordered to retain
/// subnormal volatilities and a fallback for overflowing scales.
#[inline]
pub(crate) fn stable_theta_diffusion(scale: f64, vol: f64, sqrt_expiry: f64) -> f64 {
    let grouped_numerator = scale * vol;
    if grouped_numerator.is_finite() && grouped_numerator != 0.0 {
        -(0.5 * grouped_numerator) / sqrt_expiry
    } else {
        -(scale / sqrt_expiry) * (0.5 * vol)
    }
}

/// Batch-invariant terms of [`requires_stable_price`], hoisted so a loop over
/// options sharing `(rate, dividend_yield, vol, expiry)` pays the sqrt, exp,
/// and carry setup once and each element only one log-moneyness evaluation.
pub(crate) struct StablePriceInvariants {
    width: f64,
    carry: f64,
    pub(crate) df_r: f64,
    pub(crate) df_q: f64,
    always_stable: bool,
    /// `S/K` at the call-side tail boundary `midpoint == -8`; spot/strike
    /// ratios safely above it cannot be a call deep tail.
    call_tail_bound: f64,
    /// `S/K` at the put-side tail boundary `midpoint == +8`.
    put_tail_bound: f64,
    /// Both ratio bounds are finite and positive, so the multiply-and-compare
    /// fast pass is usable.
    tail_bounds_usable: bool,
}

/// Relative margin around the ratio bounds inside which the guard falls back
/// to the exact log-domain test, so fast-pass conclusions are always certain
/// despite rounding differences between `exp`-bound and `ln`-based forms.
const TAIL_BOUND_MARGIN: f64 = 1.0e-9;

impl StablePriceInvariants {
    #[inline]
    pub(crate) fn new(rate: f64, dividend_yield: f64, vol: f64, expiry: f64) -> Self {
        Self::with_discounts(
            rate,
            dividend_yield,
            vol,
            expiry,
            (-rate * expiry).exp(),
            (-dividend_yield * expiry).exp(),
        )
    }

    /// Variant for callers that already hold the discount factors.
    #[inline]
    pub(crate) fn with_discounts(
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
        df_r: f64,
        df_q: f64,
    ) -> Self {
        let width = vol * expiry.sqrt();
        Self {
            width,
            carry: stable_carry(rate, dividend_yield, expiry),
            df_r,
            df_q,
            always_stable: width <= ACCURATE_CDF_TOTAL_VOL,
            call_tail_bound: f64::NAN,
            put_tail_bound: f64::NAN,
            tail_bounds_usable: false,
        }
    }

    /// Enables the multiply-and-compare tail fast pass in
    /// [`Self::requires_stable`]. Costs two `exp` evaluations, so it pays for
    /// itself only when the guard runs over a batch; single-shot callers that
    /// consume [`Self::ordinary_d1_d2`] should skip it.
    #[inline]
    pub(crate) fn with_ratio_fast_pass(mut self) -> Self {
        let call_tail_bound = ((-LOG_DOMAIN_TAIL_MIDPOINT) * self.width - self.carry).exp();
        let put_tail_bound = (LOG_DOMAIN_TAIL_MIDPOINT * self.width - self.carry).exp();
        self.call_tail_bound = call_tail_bound;
        self.put_tail_bound = put_tail_bound;
        self.tail_bounds_usable = call_tail_bound.is_finite()
            && call_tail_bound > 0.0
            && put_tail_bound.is_finite()
            && put_tail_bound > 0.0;
        self
    }

    #[inline]
    pub(crate) fn requires_stable(&self, spot: f64, strike: f64, is_call: bool) -> bool {
        if self.always_stable
            || !(spot * self.df_q).is_finite()
            || !(strike * self.df_r).is_finite()
        {
            return true;
        }
        // Fast pass: a spot/strike ratio safely inside the tail bounds cannot
        // be a deep tail, decided with one multiply and compare instead of a
        // log evaluation. Only certain conclusions short-circuit; ratios near
        // or beyond a bound (and unusable bounds) take the exact test, so the
        // routing decision is always identical to the log-domain form.
        if self.tail_bounds_usable {
            if is_call {
                if spot > strike * (self.call_tail_bound * (1.0 + TAIL_BOUND_MARGIN)) {
                    return false;
                }
            } else if spot < strike * (self.put_tail_bound * (1.0 - TAIL_BOUND_MARGIN)) {
                return false;
            }
        }
        let midpoint = (stable_log_spot_strike(spot, strike) + self.carry) / self.width;
        if is_call {
            midpoint <= -LOG_DOMAIN_TAIL_MIDPOINT
        } else {
            midpoint >= LOG_DOMAIN_TAIL_MIDPOINT
        }
    }

    /// `Some((d1, d2))` when the ordinary two-CDF formula is safe for this
    /// element, computed with the same operation order as [`stable_d1_d2`]'s
    /// nonzero-width branch so results are bit-identical; `None` when the
    /// element must take the stable scalar path.
    #[inline]
    pub(crate) fn ordinary_d1_d2(
        &self,
        spot: f64,
        strike: f64,
        is_call: bool,
    ) -> Option<(f64, f64)> {
        if self.always_stable
            || !(spot * self.df_q).is_finite()
            || !(strike * self.df_r).is_finite()
        {
            return None;
        }
        let midpoint = (stable_log_spot_strike(spot, strike) + self.carry) / self.width;
        let deep_tail = if is_call {
            midpoint <= -LOG_DOMAIN_TAIL_MIDPOINT
        } else {
            midpoint >= LOG_DOMAIN_TAIL_MIDPOINT
        };
        if deep_tail {
            return None;
        }
        let half_width = 0.5 * self.width;
        Some((midpoint + half_width, midpoint - half_width))
    }
}

/// Whether the ordinary two-CDF formula would lose a representable option
/// value through cancellation, or cannot safely form its discounted scales.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn requires_stable_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> bool {
    StablePriceInvariants::new(rate, dividend_yield, vol, expiry)
        .requires_stable(spot, strike, is_call)
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

    #[inline]
    fn mills_ratio(value: f64) -> f64 {
        let density = normal_pdf(value);
        let tail = normal_cdf(-value);
        if density > 0.0 && tail > 0.0 {
            return tail / density;
        }

        let inverse_sq = 1.0 / (value * value);
        let mut term = 1.0 / value;
        let mut sum = term;
        let mut previous_abs = term.abs();
        let mut odd = 1.0;
        for _ in 0..128 {
            let next = term * (-odd * inverse_sq);
            if next.abs() >= previous_abs {
                break;
            }
            let updated = sum + next;
            if updated == sum {
                break;
            }
            sum = updated;
            term = next;
            previous_abs = next.abs();
            odd += 2.0;
        }
        sum
    }

    // Division rounds an adjacent-f64 ratio much more coarsely than the
    // original spot/strike difference. Use log1p in the near-ATM region and
    // separate logarithms outside it to avoid both that loss and ratio
    // overflow for finite extreme inputs.
    let log_forward_moneyness =
        stable_log_spot_strike(spot, strike) + stable_carry(rate, dividend_yield, expiry);
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
    let absolute_midpoint = midpoint.abs();
    let lower = absolute_midpoint - 0.5 * width;
    if absolute_midpoint >= LOG_DOMAIN_TAIL_MIDPOINT && lower > 0.0 {
        let log_discounted_strike = strike.ln() - rate * expiry;
        let upper = absolute_midpoint + 0.5 * width;
        let log_mills_difference = if lower >= LOG_DOMAIN_TAIL_MIDPOINT {
            log_mills_ratio_difference(lower, width, log_width)
        } else {
            (mills_ratio(lower) - mills_ratio(upper)).ln()
        };
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
    if requires_stable_price(spot, strike, rate, dividend_yield, vol, expiry, is_call) {
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
    if invalid_inputs(spot, strike, rate, dividend_yield, vol, expiry)
        || expiry <= 0.0
        || vol == 0.0
        || spot == 0.0
        || strike == 0.0
        || requires_stable_price(spot, strike, rate, dividend_yield, vol, expiry, is_call)
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

/// x86 kernel entry for inputs already validated and tail-routed by
/// [`crate::engines::analytic::black_scholes::bs_price`]; skips the duplicate
/// guard in [`bs_price_asm`]. The caller must have checked
/// [`has_fma_bs_kernel`].
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn bs_price_asm_prechecked(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> f64 {
    // SAFETY: the caller verified AVX/FMA support via `has_fma_bs_kernel`.
    unsafe { bs_price_asm_impl(spot, strike, rate, dividend_yield, vol, expiry, is_call) }
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

    let (d1, d2) = if sig_sqrt_t == 0.0 {
        stable_d1_d2(spot, strike, rate, dividend_yield, vol, expiry)
    } else {
        let midpoint = ln_sk / sig_sqrt_t + stable_carry(rate, dividend_yield, expiry) / sig_sqrt_t;
        let half_width = 0.5 * sig_sqrt_t;
        (midpoint + half_width, midpoint - half_width)
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
    fn ratio_fast_pass_matches_log_domain_tail_decision() {
        let (r, q, t) = (0.02_f64, 0.01_f64, 1.5_f64);
        for vol in [0.05_f64, 0.2, 0.8] {
            let plain = StablePriceInvariants::new(r, q, vol, t);
            let invariants = StablePriceInvariants::new(r, q, vol, t).with_ratio_fast_pass();
            let width = vol * t.sqrt();
            let carry = stable_carry(r, q, t);
            // Midpoints across and tightly around the ±8 tail boundary.
            for midpoint_target in [
                -12.0_f64, -8.001, -8.0, -7.999, -4.0, 0.0, 4.0, 7.999, 8.0, 8.001, 12.0,
            ] {
                let strike = 100.0_f64;
                let spot = strike * (midpoint_target * width - carry).exp();
                let midpoint = (stable_log_spot_strike(spot, strike) + carry) / width;
                for is_call in [true, false] {
                    let exact = if is_call {
                        midpoint <= -LOG_DOMAIN_TAIL_MIDPOINT
                    } else {
                        midpoint >= LOG_DOMAIN_TAIL_MIDPOINT
                    };
                    assert_eq!(
                        invariants.requires_stable(spot, strike, is_call),
                        exact,
                        "fast-pass mismatch vol={vol} midpoint={midpoint} is_call={is_call}"
                    );
                    assert_eq!(
                        plain.requires_stable(spot, strike, is_call),
                        exact,
                        "plain-guard mismatch vol={vol} midpoint={midpoint} is_call={is_call}"
                    );
                    assert_eq!(
                        invariants.ordinary_d1_d2(spot, strike, is_call).is_none(),
                        exact,
                        "ordinary_d1_d2 mismatch vol={vol} midpoint={midpoint} is_call={is_call}"
                    );
                }
            }
        }
    }

    #[test]
    fn asm_wrapper_matches_reference_price_at_roundoff() {
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

    #[test]
    fn deep_otm_put_preserves_high_precision_tail_value() {
        // 100-decimal mpmath/erfc reference, evaluated independently from the
        // production CDF and pricing implementation.
        const EXPECTED: f64 = 22.752_884_600_977_636;
        let price = bs_price_scalar_reference(5.0e18, 1.0e18, 0.0, 0.0, 0.2, 1.0, false);
        let relative_error = ((price - EXPECTED) / EXPECTED).abs();
        assert!(
            relative_error <= 2.0e-12,
            "price={price:.17e} expected={EXPECTED:.17e} rel={relative_error}"
        );
    }

    #[test]
    fn finite_total_volatility_survives_sigma_squared_overflow() {
        // sigma^2 overflows, but sigma*sqrt(T) is exactly order one.
        const EXPECTED: f64 = 38.292_492_254_802_62;
        let (d1, d2) = stable_d1_d2(100.0, 100.0, 0.0, 0.0, 1.0e155, 1.0e-310);
        assert!((d1 - 0.5).abs() <= 2.0e-14, "d1={d1}");
        assert!((d2 + 0.5).abs() <= 2.0e-14, "d2={d2}");

        let price = bs_price_scalar_reference(100.0, 100.0, 0.0, 0.0, 1.0e155, 1.0e-310, true);
        assert!(
            (price - EXPECTED).abs() <= 2.0e-12,
            "price={price:.17e} expected={EXPECTED:.17e}"
        );
    }

    #[test]
    fn wide_distribution_does_not_use_mills_expansion_outside_the_tail() {
        // Here midpoint=8.1 but d2=0.1. The 100-decimal erfc reference must
        // use the central formula, not the asymptotic Mills expansion.
        const EXPECTED: f64 = 0.435_610_762_433_962_74;
        let price =
            bs_price_scalar_reference(1.925_594_579_148_456_7e56, 1.0, 0.0, 0.0, 16.0, 1.0, false);
        assert!(
            (price - EXPECTED).abs() <= 2.0e-15,
            "price={price:.17e} expected={EXPECTED:.17e}"
        );
    }
}
