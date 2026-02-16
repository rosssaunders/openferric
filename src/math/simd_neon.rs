#![cfg(all(feature = "simd", target_arch = "aarch64"))]

//! Shared AArch64 NEON SIMD math utilities for 2-lane `f64` vectors.

use std::arch::aarch64::*;

const P: f64 = 0.231_641_9;
const A1: f64 = 0.319_381_530;
const A2: f64 = -0.356_563_782;
const A3: f64 = 1.781_477_937;
const A4: f64 = -1.821_255_978;
const A5: f64 = 1.330_274_429;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[inline]
pub unsafe fn splat_f64x2(value: f64) -> float64x2_t {
    vdupq_n_f64(value)
}

#[inline]
pub unsafe fn load_f64x2(values: &[f64], i: usize) -> float64x2_t {
    // SAFETY: caller guarantees there are at least 2 elements starting at `i`.
    unsafe { vld1q_f64(values.as_ptr().add(i)) }
}

#[inline]
pub unsafe fn store_f64x2(values: &mut [f64], i: usize, v: float64x2_t) {
    // SAFETY: caller guarantees there are at least 2 elements starting at `i`.
    unsafe { vst1q_f64(values.as_mut_ptr().add(i), v) };
}

#[inline]
pub unsafe fn simd_exp_f64x2(x: float64x2_t) -> float64x2_t {
    let lanes = [vgetq_lane_f64(x, 0).exp(), vgetq_lane_f64(x, 1).exp()];
    // SAFETY: `lanes` contains exactly two contiguous `f64` values.
    unsafe { vld1q_f64(lanes.as_ptr()) }
}

#[inline]
pub unsafe fn simd_ln_f64x2(x: float64x2_t) -> float64x2_t {
    let lanes = [vgetq_lane_f64(x, 0).ln(), vgetq_lane_f64(x, 1).ln()];
    // SAFETY: `lanes` contains exactly two contiguous `f64` values.
    unsafe { vld1q_f64(lanes.as_ptr()) }
}

#[inline]
pub unsafe fn norm_pdf_f64x2(x: float64x2_t) -> float64x2_t {
    let exponent = vmulq_f64(vdupq_n_f64(-0.5), vmulq_f64(x, x));
    vmulq_f64(vdupq_n_f64(INV_SQRT_2PI), unsafe {
        simd_exp_f64x2(exponent)
    })
}

#[inline]
pub unsafe fn norm_cdf_f64x2(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let zero = vdupq_n_f64(0.0);
    let z = vabsq_f64(x);

    let t = vdivq_f64(one, vaddq_f64(vmulq_f64(vdupq_n_f64(P), z), one));

    let mut poly = vdupq_n_f64(A5);
    poly = vaddq_f64(vmulq_f64(poly, t), vdupq_n_f64(A4));
    poly = vaddq_f64(vmulq_f64(poly, t), vdupq_n_f64(A3));
    poly = vaddq_f64(vmulq_f64(poly, t), vdupq_n_f64(A2));
    poly = vaddq_f64(vmulq_f64(poly, t), vdupq_n_f64(A1));
    poly = vmulq_f64(poly, t);

    let approx = vsubq_f64(one, vmulq_f64(unsafe { norm_pdf_f64x2(z) }, poly));
    let reflected = vsubq_f64(one, approx);
    let neg_mask = vcltq_f64(x, zero);
    vbslq_f64(neg_mask, reflected, approx)
}

#[inline]
fn normal_cdf_scalar(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + P * z);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let pdf = INV_SQRT_2PI * (-0.5 * z * z).exp();
    let approx = 1.0 - pdf * poly;
    if x < 0.0 { 1.0 - approx } else { approx }
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
        spot * df_q * normal_cdf_scalar(d1) - strike * df_r * normal_cdf_scalar(d2)
    } else {
        strike * df_r * normal_cdf_scalar(-d2) - spot * df_q * normal_cdf_scalar(-d1)
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn bs_price_neon_batch(
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

    if t <= 0.0 || vol <= 0.0 {
        for i in 0..spots.len() {
            out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        }
        return out;
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let inv_sig_sqrt_t = 1.0 / sig_sqrt_t;
    let drift = (r - q + 0.5 * vol * vol) * t;
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();

    let drift_v = unsafe { splat_f64x2(drift) };
    let inv_sig_sqrt_t_v = unsafe { splat_f64x2(inv_sig_sqrt_t) };
    let sig_sqrt_t_v = unsafe { splat_f64x2(sig_sqrt_t) };
    let df_r_v = unsafe { splat_f64x2(df_r) };
    let df_q_v = unsafe { splat_f64x2(df_q) };
    let zero = vdupq_n_f64(0.0);

    let n = spots.len();
    let mut i = 0usize;
    while i + 2 <= n {
        // SAFETY: loop condition guarantees in-bounds 2-lane loads.
        let s = unsafe { load_f64x2(spots, i) };
        // SAFETY: loop condition guarantees in-bounds 2-lane loads.
        let k = unsafe { load_f64x2(strikes, i) };

        let ln_sk = unsafe { simd_ln_f64x2(vdivq_f64(s, k)) };
        let d1 = vmulq_f64(vaddq_f64(ln_sk, drift_v), inv_sig_sqrt_t_v);
        let d2 = vsubq_f64(d1, sig_sqrt_t_v);

        let nd1 = unsafe { norm_cdf_f64x2(d1) };
        let nd2 = unsafe { norm_cdf_f64x2(d2) };

        let call = vsubq_f64(
            vmulq_f64(vmulq_f64(s, df_q_v), nd1),
            vmulq_f64(vmulq_f64(k, df_r_v), nd2),
        );

        let put = vsubq_f64(
            vmulq_f64(vmulq_f64(k, df_r_v), unsafe {
                norm_cdf_f64x2(vsubq_f64(zero, d2))
            }),
            vmulq_f64(vmulq_f64(s, df_q_v), unsafe {
                norm_cdf_f64x2(vsubq_f64(zero, d1))
            }),
        );

        // SAFETY: loop condition guarantees in-bounds 2-lane stores.
        unsafe { store_f64x2(&mut out, i, if is_call { call } else { put }) };
        i += 2;
    }

    while i < n {
        out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        i += 1;
    }

    out
}
