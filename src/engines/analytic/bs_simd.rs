//! SIMD-accelerated Black-Scholes batch routines with scalar fallback.

/// Scalar Abramowitz & Stegun 7.1.26 normal CDF approximation.
#[inline]
pub fn normal_cdf_approx(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    let z = x.abs();
    let t = 1.0 / (1.0 + P * z);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let pdf = INV_SQRT_2PI * (-0.5 * z * z).exp();
    let approx = 1.0 - pdf * poly;
    if x < 0.0 { 1.0 - approx } else { approx }
}

/// Batch normal CDF approximation with runtime SIMD dispatch and scalar fallback.
pub fn normal_cdf_batch_approx(xs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; xs.len()];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_avx2(xs, &mut out) };
            return out;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_neon(xs, &mut out) };
            return out;
        }
    }

    for (dst, &x) in out.iter_mut().zip(xs.iter()) {
        *dst = normal_cdf_approx(x);
    }
    out
}

/// Black-Scholes price for a batch of options.
///
/// `spots` and `strikes` must have identical lengths.
pub fn bs_price_batch(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> Vec<f64> {
    assert_eq!(
        spots.len(),
        strikes.len(),
        "spots and strikes must have identical lengths",
    );

    let mut out = vec![0.0_f64; spots.len()];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { bs_price_batch_avx2(spots, strikes, r, q, vol, t, is_call, &mut out) };
            return out;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                crate::math::simd_neon::bs_price_neon_batch(spots, strikes, r, q, vol, t, is_call)
            };
        }
    }

    for i in 0..spots.len() {
        out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
    }
    out
}

/// Black-Scholes Greeks (delta, gamma, vega, theta) for a batch of options.
///
/// `spots` and `strikes` must have identical lengths.
pub fn bs_greeks_batch(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(
        spots.len(),
        strikes.len(),
        "spots and strikes must have identical lengths",
    );

    let n = spots.len();
    let mut delta = vec![0.0_f64; n];
    let mut gamma = vec![0.0_f64; n];
    let mut vega = vec![0.0_f64; n];
    let mut theta = vec![0.0_f64; n];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_avx2(
                    spots, strikes, r, q, vol, t, is_call, &mut delta, &mut gamma, &mut vega,
                    &mut theta,
                )
            };
            return (delta, gamma, vega, theta);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_neon(
                    spots, strikes, r, q, vol, t, is_call, &mut delta, &mut gamma, &mut vega,
                    &mut theta,
                )
            };
            return (delta, gamma, vega, theta);
        }
    }

    for i in 0..n {
        let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        delta[i] = d;
        gamma[i] = g;
        vega[i] = v;
        theta[i] = th;
    }

    (delta, gamma, vega, theta)
}

#[inline]
fn bs_price_scalar(spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool) -> f64 {
    if t <= 0.0 || vol <= 0.0 {
        return if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    if is_call {
        spot * df_q * normal_cdf_approx(d1) - strike * df_r * normal_cdf_approx(d2)
    } else {
        strike * df_r * normal_cdf_approx(-d2) - spot * df_q * normal_cdf_approx(-d1)
    }
}

#[inline]
fn normal_pdf_scalar(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

#[inline]
fn bs_greeks_scalar(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> (f64, f64, f64, f64) {
    if t <= 0.0 || vol <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    let pdf = normal_pdf_scalar(d1);

    let delta = if is_call {
        df_q * normal_cdf_approx(d1)
    } else {
        df_q * (normal_cdf_approx(d1) - 1.0)
    };
    let gamma = df_q * pdf / (spot * vol * sqrt_t);
    let vega = spot * df_q * pdf * sqrt_t;

    let theta = if is_call {
        -spot * df_q * pdf * vol / (2.0 * sqrt_t) + q * spot * df_q * normal_cdf_approx(d1)
            - r * strike * df_r * normal_cdf_approx(d2)
    } else {
        -spot * df_q * pdf * vol / (2.0 * sqrt_t) - q * spot * df_q * normal_cdf_approx(-d1)
            + r * strike * df_r * normal_cdf_approx(-d2)
    };

    (delta, gamma, vega, theta)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon_impl {
    use std::arch::aarch64::*;

    use crate::math::simd_neon::{
        load_f64x2, norm_cdf_f64x2, norm_pdf_f64x2, simd_ln_f64x2, splat_f64x2, store_f64x2,
    };

    use super::bs_greeks_scalar;

    pub(super) unsafe fn normal_cdf_batch_neon(xs: &[f64], out: &mut [f64]) {
        let n = xs.len();
        let mut i = 0usize;
        while i + 2 <= n {
            // SAFETY: bounds are checked in loop condition.
            let x = unsafe { load_f64x2(xs, i) };
            // SAFETY: SIMD math helper uses lane-local arithmetic.
            let y = unsafe { norm_cdf_f64x2(x) };
            // SAFETY: bounds are checked in loop condition.
            unsafe { store_f64x2(out, i, y) };
            i += 2;
        }
        while i < n {
            out[i] = super::normal_cdf_approx(xs[i]);
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn bs_greeks_batch_neon(
        spots: &[f64],
        strikes: &[f64],
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
        is_call: bool,
        delta: &mut [f64],
        gamma: &mut [f64],
        vega: &mut [f64],
        theta: &mut [f64],
    ) {
        if t <= 0.0 || vol <= 0.0 {
            for i in 0..spots.len() {
                let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
                delta[i] = d;
                gamma[i] = g;
                vega[i] = v;
                theta[i] = th;
            }
            return;
        }

        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let inv_sig_sqrt_t = 1.0 / sig_sqrt_t;
        let drift = (r - q + 0.5 * vol * vol) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let denom_gamma = vol * sqrt_t;

        let drift_v = unsafe { splat_f64x2(drift) };
        let inv_sig_sqrt_t_v = unsafe { splat_f64x2(inv_sig_sqrt_t) };
        let sig_sqrt_t_v = unsafe { splat_f64x2(sig_sqrt_t) };
        let df_r_v = unsafe { splat_f64x2(df_r) };
        let df_q_v = unsafe { splat_f64x2(df_q) };
        let sqrt_t_v = unsafe { splat_f64x2(sqrt_t) };
        let vol_v = unsafe { splat_f64x2(vol) };
        let q_v = unsafe { splat_f64x2(q) };
        let r_v = unsafe { splat_f64x2(r) };
        let one = unsafe { splat_f64x2(1.0) };
        let two = unsafe { splat_f64x2(2.0) };
        let zero = vdupq_n_f64(0.0);
        let denom_gamma_v = unsafe { splat_f64x2(denom_gamma) };

        let n = spots.len();
        let mut i = 0usize;
        while i + 2 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = unsafe { load_f64x2(spots, i) };
            // SAFETY: bounds are checked in loop condition.
            let k = unsafe { load_f64x2(strikes, i) };

            let ln_sk = unsafe { simd_ln_f64x2(vdivq_f64(s, k)) };
            let d1 = vmulq_f64(vaddq_f64(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = vsubq_f64(d1, sig_sqrt_t_v);

            let nd1 = unsafe { norm_cdf_f64x2(d1) };
            let nd2 = unsafe { norm_cdf_f64x2(d2) };
            let nmd1 = unsafe { norm_cdf_f64x2(vsubq_f64(zero, d1)) };
            let nmd2 = unsafe { norm_cdf_f64x2(vsubq_f64(zero, d2)) };
            let pdf_d1 = unsafe { norm_pdf_f64x2(d1) };

            let delta_call = vmulq_f64(df_q_v, nd1);
            let delta_put = vmulq_f64(df_q_v, vsubq_f64(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = vdivq_f64(vmulq_f64(df_q_v, pdf_d1), vmulq_f64(s, denom_gamma_v));
            let vega_v = vmulq_f64(vmulq_f64(vmulq_f64(s, df_q_v), pdf_d1), sqrt_t_v);

            let theta_common = vmulq_f64(vmulq_f64(vmulq_f64(s, df_q_v), pdf_d1), vol_v);
            let theta_common = vdivq_f64(vsubq_f64(zero, theta_common), vmulq_f64(two, sqrt_t_v));

            let theta_call = vsubq_f64(
                vaddq_f64(
                    theta_common,
                    vmulq_f64(vmulq_f64(q_v, s), vmulq_f64(df_q_v, nd1)),
                ),
                vmulq_f64(vmulq_f64(r_v, k), vmulq_f64(df_r_v, nd2)),
            );

            let theta_put = vaddq_f64(
                vsubq_f64(
                    theta_common,
                    vmulq_f64(vmulq_f64(q_v, s), vmulq_f64(df_q_v, nmd1)),
                ),
                vmulq_f64(vmulq_f64(r_v, k), vmulq_f64(df_r_v, nmd2)),
            );
            let theta_v = if is_call { theta_call } else { theta_put };

            // SAFETY: bounds are checked in loop condition.
            unsafe {
                store_f64x2(delta, i, delta_v);
                store_f64x2(gamma, i, gamma_v);
                store_f64x2(vega, i, vega_v);
                store_f64x2(theta, i, theta_v);
            }

            i += 2;
        }

        while i < n {
            let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            delta[i] = d;
            gamma[i] = g;
            vega[i] = v;
            theta[i] = th;
            i += 1;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2_impl {
    use std::arch::x86_64::*;

    use crate::math::simd_math::{
        ln_f64x4, load_f64x4, norm_cdf_f64x4, norm_pdf_f64x4, splat_f64x4, store_f64x4,
    };

    use super::{bs_greeks_scalar, bs_price_scalar};

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn normal_cdf_batch_avx2(xs: &[f64], out: &mut [f64]) {
        let n = xs.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let x = unsafe { load_f64x4(xs, i) };
            // SAFETY: target feature is enabled by this function attribute.
            let y = unsafe { norm_cdf_f64x4(x) };
            // SAFETY: bounds are checked in loop condition.
            unsafe { store_f64x4(out, i, y) };
            i += 4;
        }
        while i < n {
            out[i] = super::normal_cdf_approx(xs[i]);
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn bs_price_batch_avx2(
        spots: &[f64],
        strikes: &[f64],
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
        is_call: bool,
        out: &mut [f64],
    ) {
        if t <= 0.0 || vol <= 0.0 {
            for i in 0..spots.len() {
                out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            }
            return;
        }

        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let inv_sig_sqrt_t = 1.0 / sig_sqrt_t;
        let drift = (r - q + 0.5 * vol * vol) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let drift_v = unsafe { splat_f64x4(drift) };
        let inv_sig_sqrt_t_v = unsafe { splat_f64x4(inv_sig_sqrt_t) };
        let sig_sqrt_t_v = unsafe { splat_f64x4(sig_sqrt_t) };
        let df_r_v = unsafe { splat_f64x4(df_r) };
        let df_q_v = unsafe { splat_f64x4(df_q) };
        let zero = _mm256_setzero_pd();

        let n = spots.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = unsafe { load_f64x4(spots, i) };
            // SAFETY: bounds are checked in loop condition.
            let k = unsafe { load_f64x4(strikes, i) };
            let ln_sk = unsafe { ln_f64x4(_mm256_div_pd(s, k)) };

            let d1 = _mm256_mul_pd(_mm256_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm256_sub_pd(d1, sig_sqrt_t_v);

            let nd1 = unsafe { norm_cdf_f64x4(d1) };
            let nd2 = unsafe { norm_cdf_f64x4(d2) };

            let call = _mm256_sub_pd(
                _mm256_mul_pd(_mm256_mul_pd(s, df_q_v), nd1),
                _mm256_mul_pd(_mm256_mul_pd(k, df_r_v), nd2),
            );

            let put = _mm256_sub_pd(
                _mm256_mul_pd(_mm256_mul_pd(k, df_r_v), unsafe {
                    norm_cdf_f64x4(_mm256_sub_pd(zero, d2))
                }),
                _mm256_mul_pd(_mm256_mul_pd(s, df_q_v), unsafe {
                    norm_cdf_f64x4(_mm256_sub_pd(zero, d1))
                }),
            );

            // SAFETY: bounds are checked in loop condition.
            unsafe { store_f64x4(out, i, if is_call { call } else { put }) };
            i += 4;
        }

        while i < n {
            out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn bs_greeks_batch_avx2(
        spots: &[f64],
        strikes: &[f64],
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
        is_call: bool,
        delta: &mut [f64],
        gamma: &mut [f64],
        vega: &mut [f64],
        theta: &mut [f64],
    ) {
        if t <= 0.0 || vol <= 0.0 {
            for i in 0..spots.len() {
                let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
                delta[i] = d;
                gamma[i] = g;
                vega[i] = v;
                theta[i] = th;
            }
            return;
        }

        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let inv_sig_sqrt_t = 1.0 / sig_sqrt_t;
        let drift = (r - q + 0.5 * vol * vol) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let denom_gamma = vol * sqrt_t;

        let drift_v = unsafe { splat_f64x4(drift) };
        let inv_sig_sqrt_t_v = unsafe { splat_f64x4(inv_sig_sqrt_t) };
        let sig_sqrt_t_v = unsafe { splat_f64x4(sig_sqrt_t) };
        let df_r_v = unsafe { splat_f64x4(df_r) };
        let df_q_v = unsafe { splat_f64x4(df_q) };
        let sqrt_t_v = unsafe { splat_f64x4(sqrt_t) };
        let vol_v = unsafe { splat_f64x4(vol) };
        let q_v = unsafe { splat_f64x4(q) };
        let r_v = unsafe { splat_f64x4(r) };
        let one = unsafe { splat_f64x4(1.0) };
        let zero = _mm256_setzero_pd();
        let denom_gamma_v = unsafe { splat_f64x4(denom_gamma) };

        let n = spots.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = unsafe { load_f64x4(spots, i) };
            // SAFETY: bounds are checked in loop condition.
            let k = unsafe { load_f64x4(strikes, i) };
            let ln_sk = unsafe { ln_f64x4(_mm256_div_pd(s, k)) };

            let d1 = _mm256_mul_pd(_mm256_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm256_sub_pd(d1, sig_sqrt_t_v);

            let nd1 = unsafe { norm_cdf_f64x4(d1) };
            let nd2 = unsafe { norm_cdf_f64x4(d2) };
            let nmd1 = unsafe { norm_cdf_f64x4(_mm256_sub_pd(zero, d1)) };
            let nmd2 = unsafe { norm_cdf_f64x4(_mm256_sub_pd(zero, d2)) };
            let pdf_d1 = unsafe { norm_pdf_f64x4(d1) };

            let delta_call = _mm256_mul_pd(df_q_v, nd1);
            let delta_put = _mm256_mul_pd(df_q_v, _mm256_sub_pd(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = _mm256_div_pd(
                _mm256_mul_pd(df_q_v, pdf_d1),
                _mm256_mul_pd(s, denom_gamma_v),
            );

            let vega_v = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), sqrt_t_v);

            let theta_common =
                _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), vol_v);
            let theta_common = _mm256_div_pd(
                _mm256_sub_pd(zero, theta_common),
                _mm256_mul_pd(unsafe { splat_f64x4(2.0) }, sqrt_t_v),
            );

            let theta_call = _mm256_sub_pd(
                _mm256_add_pd(
                    theta_common,
                    _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nd1)),
                ),
                _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nd2)),
            );

            let theta_put = _mm256_add_pd(
                _mm256_sub_pd(
                    theta_common,
                    _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nmd1)),
                ),
                _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nmd2)),
            );
            let theta_v = if is_call { theta_call } else { theta_put };

            // SAFETY: bounds are checked in loop condition.
            unsafe {
                store_f64x4(delta, i, delta_v);
                store_f64x4(gamma, i, gamma_v);
                store_f64x4(vega, i, vega_v);
                store_f64x4(theta, i, theta_v);
            }

            i += 4;
        }

        while i < n {
            let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            delta[i] = d;
            gamma[i] = g;
            vega[i] = v;
            theta[i] = th;
            i += 1;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use avx2_impl::{bs_greeks_batch_avx2, bs_price_batch_avx2, normal_cdf_batch_avx2};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use neon_impl::{bs_greeks_batch_neon, normal_cdf_batch_neon};
