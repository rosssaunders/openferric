//! Module `math::simd_neon`.
//!
//! Implements simd neon abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004) Ch. 5, Joe and Kuo (2008), SIMD and random-sequence implementation details tied to Eq. (5.4).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing loops; use higher-level modules when model semantics matter more than raw numerics.
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

/// Vectorized exp(x) for NEON f64x2 using degree-11 Taylor with range reduction.
///
/// Matches the AVX2 `exp_f64x4` algorithm from `simd_math.rs`, ported to 2-lane NEON.
#[inline]
pub unsafe fn simd_exp_f64x2(x: float64x2_t) -> float64x2_t {
    // Clamp to avoid overflow/underflow
    let max_x = vdupq_n_f64(709.782_712_893_384);
    let min_x = vdupq_n_f64(-708.396_418_532_264_1);
    let x = vmaxq_f64(min_x, vminq_f64(x, max_x));

    // Range reduction: x = n*ln(2) + r, |r| <= ln(2)/2
    let log2e = vdupq_n_f64(std::f64::consts::LOG2_E);
    let n_f64 = vrndnq_f64(vmulq_f64(x, log2e)); // round to nearest

    let ln2_hi = vdupq_n_f64(6.931_471_803_691_238e-1);
    let ln2_lo = vdupq_n_f64(1.908_214_929_270_587_7e-10);
    let r = vsubq_f64(x, vmulq_f64(n_f64, ln2_hi));
    let r = vsubq_f64(r, vmulq_f64(n_f64, ln2_lo));

    // Degree-11 Taylor: 1 + r + r^2/2! + ... + r^11/11!
    let c11 = vdupq_n_f64(1.0 / 39_916_800.0);
    let c10 = vdupq_n_f64(1.0 / 3_628_800.0);
    let c9 = vdupq_n_f64(1.0 / 362_880.0);
    let c8 = vdupq_n_f64(1.0 / 40_320.0);
    let c7 = vdupq_n_f64(1.0 / 5_040.0);
    let c6 = vdupq_n_f64(1.0 / 720.0);
    let c5 = vdupq_n_f64(1.0 / 120.0);
    let c4 = vdupq_n_f64(1.0 / 24.0);
    let c3 = vdupq_n_f64(1.0 / 6.0);
    let c2 = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);

    // Horner's method
    let mut poly = c11;
    poly = vfmaq_f64(c10, poly, r);
    poly = vfmaq_f64(c9, poly, r);
    poly = vfmaq_f64(c8, poly, r);
    poly = vfmaq_f64(c7, poly, r);
    poly = vfmaq_f64(c6, poly, r);
    poly = vfmaq_f64(c5, poly, r);
    poly = vfmaq_f64(c4, poly, r);
    poly = vfmaq_f64(c3, poly, r);
    poly = vfmaq_f64(c2, poly, r);
    poly = vfmaq_f64(one, poly, r);
    poly = vfmaq_f64(one, poly, r);

    // Reconstruct: exp(x) = poly * 2^n via IEEE 754 exponent manipulation
    let mut n_lanes = [0_i64; 2];
    vst1q_f64(n_lanes.as_mut_ptr() as *mut f64, n_f64);
    let mut exp_lanes = [0_u64; 2];
    for i in 0..2 {
        exp_lanes[i] = ((n_lanes[i] + 1023) as u64) << 52;
    }
    let two_pow_n: float64x2_t = std::mem::transmute(vld1q_u64(exp_lanes.as_ptr()));
    vmulq_f64(poly, two_pow_n)
}

/// Vectorized ln(x) for NEON f64x2 using fdlibm kernel with IEEE 754 bit extraction.
///
/// Matches the AVX2 `ln_f64x4` algorithm from `simd_math.rs`, ported to 2-lane NEON.
#[inline]
pub unsafe fn simd_ln_f64x2(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let sqrt_half = vdupq_n_f64(std::f64::consts::FRAC_1_SQRT_2);

    // Extract exponent and mantissa via IEEE 754 bit manipulation
    let x_bits: uint64x2_t = std::mem::transmute(x);
    let exp_mask = vdupq_n_u64(0x7ff0_0000_0000_0000);
    let mant_mask = vdupq_n_u64(0x000f_ffff_ffff_ffff);
    let bias_bits = vdupq_n_u64(0x3ff0_0000_0000_0000);

    let exp_bits = vshrq_n_u64(vandq_u64(x_bits, exp_mask), 52);
    let mant_bits = vorrq_u64(vandq_u64(x_bits, mant_mask), bias_bits);
    let mut m: float64x2_t = std::mem::transmute(mant_bits);

    // Compute k = exponent - 1023
    let mut k_lanes = [0.0_f64; 2];
    let mut exp_u64 = [0_u64; 2];
    vst1q_u64(exp_u64.as_mut_ptr(), exp_bits);
    k_lanes[0] = (exp_u64[0] as i64 - 1023) as f64;
    k_lanes[1] = (exp_u64[1] as i64 - 1023) as f64;
    let mut k = vld1q_f64(k_lanes.as_ptr());

    // Fold m into [sqrt(1/2), sqrt(2)] for better accuracy
    let adjust = vcltq_f64(m, sqrt_half);
    let adjust_bits: uint64x2_t = std::mem::transmute(adjust);
    // If m < sqrt(1/2), double m and decrement k
    let m_doubled = vaddq_f64(m, m);
    let k_minus_one = vsubq_f64(k, one);
    m = vbslq_f64(adjust_bits, m_doubled, m);
    k = vbslq_f64(adjust_bits, k_minus_one, k);

    // f = m - 1, s = f / (2 + f)
    let f = vsubq_f64(m, one);
    let two = vdupq_n_f64(2.0);
    let s = vdivq_f64(f, vaddq_f64(two, f));
    let z = vmulq_f64(s, s);
    let w = vmulq_f64(z, z);

    // Minimax polynomial coefficients (fdlibm)
    let lg1 = vdupq_n_f64(6.666_666_666_666_735e-1);
    let lg2 = vdupq_n_f64(3.999_999_999_940_942e-1);
    let lg3 = vdupq_n_f64(2.857_142_874_366_239e-1);
    let lg4 = vdupq_n_f64(2.222_219_843_214_978_4e-1);
    let lg5 = vdupq_n_f64(1.818_357_216_161_805e-1);
    let lg6 = vdupq_n_f64(1.531_383_769_920_937_3e-1);
    let lg7 = vdupq_n_f64(1.479_819_860_511_658_6e-1);

    // t1 = w * (lg2 + w * (lg4 + w * lg6))
    let t1 = vmulq_f64(w, vfmaq_f64(lg2, w, vfmaq_f64(lg4, w, lg6)));
    // t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)))
    let t2 = vmulq_f64(
        z,
        vfmaq_f64(lg1, w, vfmaq_f64(lg3, w, vfmaq_f64(lg5, w, lg7))),
    );
    let r = vaddq_f64(t1, t2);

    let half = vdupq_n_f64(0.5);
    let hfsq = vmulq_f64(half, vmulq_f64(f, f));
    let ln_m = vsubq_f64(f, vsubq_f64(hfsq, vmulq_f64(s, vaddq_f64(hfsq, r))));

    let ln2_hi = vdupq_n_f64(6.931_471_803_691_238e-1);
    let ln2_lo = vdupq_n_f64(1.908_214_929_270_587_7e-10);
    let mut y = vfmaq_f64(ln_m, k, ln2_hi);
    y = vfmaq_f64(y, k, ln2_lo);

    y
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
