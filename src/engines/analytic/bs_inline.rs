//! x86 FMA-accelerated Black-Scholes kernel with scalar fallback.

use crate::math::normal_cdf;

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

    let sig_sqrt_t = vol * expiry.sqrt();
    let d1 =
        ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
            / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;
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
    let ln_sk = (spot / strike).ln();

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

    let d1 = (ln_sk + drift_t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

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
