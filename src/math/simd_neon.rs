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
#![allow(unsafe_op_in_unsafe_fn)]

//! Shared AArch64 NEON SIMD math utilities for 2-lane `f64` vectors.

use std::arch::aarch64::*;

const P: f64 = 0.231_641_9;
const A1: f64 = 0.319_381_530;
const A2: f64 = -0.356_563_782;
const A3: f64 = 1.781_477_937;
const A4: f64 = -1.821_255_978;
const A5: f64 = 1.330_274_429;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Broadcasts `value` to both lanes of an AArch64 NEON vector.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
#[inline]
pub unsafe fn splat_f64x2(value: f64) -> float64x2_t {
    vdupq_n_f64(value)
}

/// Loads two adjacent values starting at `i`.
///
/// # Safety
///
/// The caller must execute this where NEON is available and ensure that
/// `values[i..]` contains at least two elements.
#[inline]
pub unsafe fn load_f64x2(values: &[f64], i: usize) -> float64x2_t {
    // SAFETY: caller guarantees there are at least 2 elements starting at `i`.
    unsafe { vld1q_f64(values.as_ptr().add(i)) }
}

/// Stores both lanes of `v` starting at `i`.
///
/// # Safety
///
/// The caller must execute this where NEON is available and ensure that
/// `values[i..]` contains at least two elements.
#[inline]
pub unsafe fn store_f64x2(values: &mut [f64], i: usize, v: float64x2_t) {
    // SAFETY: caller guarantees there are at least 2 elements starting at `i`.
    unsafe { vst1q_f64(values.as_mut_ptr().add(i), v) };
}

#[inline]
unsafe fn repair_exp_subnormal_lanes(
    input: float64x2_t,
    result: float64x2_t,
    below_normal_range: uint64x2_t,
) -> float64x2_t {
    // `vmaxvq_u64` is not exposed by Rust's AArch64 intrinsics. Extracting
    // both mask lanes is portable across the supported AArch64 toolchains.
    if (vgetq_lane_u64::<0>(below_normal_range) | vgetq_lane_u64::<1>(below_normal_range)) == 0 {
        return result;
    }

    let mut inputs = [0.0_f64; 2];
    let mut outputs = [0.0_f64; 2];
    vst1q_f64(inputs.as_mut_ptr(), input);
    vst1q_f64(outputs.as_mut_ptr(), result);
    for lane in 0..2 {
        if inputs[lane] < -708.396_418_532_264_1 {
            outputs[lane] = inputs[lane].exp();
        }
    }
    vld1q_f64(outputs.as_ptr())
}

#[inline]
unsafe fn repair_ln_subnormal_lanes(input: float64x2_t, result: float64x2_t) -> float64x2_t {
    let positive = vcgtq_f64(input, vdupq_n_f64(0.0));
    let below_normal = vcltq_f64(input, vdupq_n_f64(f64::MIN_POSITIVE));
    let subnormal = vandq_u64(positive, below_normal);
    if (vgetq_lane_u64::<0>(subnormal) | vgetq_lane_u64::<1>(subnormal)) == 0 {
        return result;
    }

    let mut inputs = [0.0_f64; 2];
    let mut outputs = [0.0_f64; 2];
    vst1q_f64(inputs.as_mut_ptr(), input);
    vst1q_f64(outputs.as_mut_ptr(), result);
    for lane in 0..2 {
        if inputs[lane] > 0.0 && inputs[lane] < f64::MIN_POSITIVE {
            outputs[lane] = inputs[lane].ln();
        }
    }
    vld1q_f64(outputs.as_ptr())
}

/// Vectorized exp(x) for NEON f64x2 using degree-11 Taylor with range reduction.
///
/// Matches the AVX2 `exp_f64x4` algorithm from `simd_math.rs`, ported to 2-lane NEON.
/// Dense differential tests bound relative error by `2e-14` over the practical
/// finite range `[-700, 700]`.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
#[inline]
pub unsafe fn simd_exp_f64x2(x: float64x2_t) -> float64x2_t {
    let input = x;
    // Clamp to avoid overflow/underflow
    let max_x = vdupq_n_f64(709.782_712_893_384);
    let min_x = vdupq_n_f64(-708.396_418_532_264_1);
    // Inputs beyond ln(f64::MAX) must overflow to +inf like std::exp.
    // Lanes whose exact result is subnormal are repaired below, and NaN must
    // propagate.
    // vceqq_f64(x, x) is false only for NaN lanes.
    let overflow = vcgtq_f64(x, max_x);
    let underflow = vcltq_f64(x, min_x);
    let nan_mask = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));
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

    // Reconstruct: exp(x) = poly * 2^n1 * 2^n2 with n1 = n/2, n2 = n - n1.
    // A single 2^n overflows the biased exponent for n = 1024, which
    // round(x*log2e) produces for x in [~709.44, 709.78] where exp(x) is
    // still finite.
    let n_i64 = vcvtq_s64_f64(n_f64);
    let n1_i64 = vshrq_n_s64(n_i64, 1);
    let n2_i64 = vsubq_s64(n_i64, n1_i64);
    let bias = vdupq_n_s64(1023);
    let e1 = vshlq_n_s64(vaddq_s64(n1_i64, bias), 52);
    let e2 = vshlq_n_s64(vaddq_s64(n2_i64, bias), 52);
    let y = vmulq_f64(
        vmulq_f64(poly, vreinterpretq_f64_s64(e1)),
        vreinterpretq_f64_s64(e2),
    );

    let y = vbslq_f64(overflow, vdupq_n_f64(f64::INFINITY), y);
    let y = vbslq_f64(underflow, vdupq_n_f64(0.0), y);
    let y = vbslq_f64(nan_mask, vdupq_n_f64(f64::NAN), y);
    repair_exp_subnormal_lanes(input, y, underflow)
}

/// Faster vectorized exp(x) for NEON f64x2 using a degree-7 polynomial.
///
/// Saves four fused multiply-add operations versus [`simd_exp_f64x2`]. Dense
/// differential tests bound relative error by `8e-9` over the practical finite
/// range `[-700, 700]`.
///
/// # Safety
/// The caller must ensure NEON is available; it is part of the AArch64 baseline.
#[inline]
pub unsafe fn fast_exp_f64x2(x: float64x2_t) -> float64x2_t {
    let input = x;
    let max_x = vdupq_n_f64(709.782_712_893_384);
    let min_x = vdupq_n_f64(-708.396_418_532_264_1);
    let overflow = vcgtq_f64(x, max_x);
    let underflow = vcltq_f64(x, min_x);
    let nan_mask = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));
    let x = vmaxq_f64(min_x, vminq_f64(x, max_x));

    let log2e = vdupq_n_f64(std::f64::consts::LOG2_E);
    let n_f64 = vrndnq_f64(vmulq_f64(x, log2e));

    let ln2_hi = vdupq_n_f64(6.931_471_803_691_238e-1);
    let ln2_lo = vdupq_n_f64(1.908_214_929_270_587_7e-10);
    let r = vsubq_f64(x, vmulq_f64(n_f64, ln2_hi));
    let r = vsubq_f64(r, vmulq_f64(n_f64, ln2_lo));

    // Taylor-like coefficients rounded near reciprocal factorials. This is
    // intentionally the same degree-7 approximation as the x86 kernels.
    let c7 = vdupq_n_f64(1.984_126_984_12e-4);
    let c6 = vdupq_n_f64(1.388_888_889_0e-3);
    let c5 = vdupq_n_f64(8.333_333_333_3e-3);
    let c4 = vdupq_n_f64(4.166_666_666_67e-2);
    let c3 = vdupq_n_f64(1.666_666_666_666_67e-1);
    let c2 = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);

    let mut poly = c7;
    poly = vfmaq_f64(c6, poly, r);
    poly = vfmaq_f64(c5, poly, r);
    poly = vfmaq_f64(c4, poly, r);
    poly = vfmaq_f64(c3, poly, r);
    poly = vfmaq_f64(c2, poly, r);
    poly = vfmaq_f64(one, poly, r);
    poly = vfmaq_f64(one, poly, r);

    let n_i64 = vcvtq_s64_f64(n_f64);
    let n1_i64 = vshrq_n_s64(n_i64, 1);
    let n2_i64 = vsubq_s64(n_i64, n1_i64);
    let bias = vdupq_n_s64(1023);
    let e1 = vshlq_n_s64(vaddq_s64(n1_i64, bias), 52);
    let e2 = vshlq_n_s64(vaddq_s64(n2_i64, bias), 52);
    let y = vmulq_f64(
        vmulq_f64(poly, vreinterpretq_f64_s64(e1)),
        vreinterpretq_f64_s64(e2),
    );

    let y = vbslq_f64(overflow, vdupq_n_f64(f64::INFINITY), y);
    let y = vbslq_f64(underflow, vdupq_n_f64(0.0), y);
    let y = vbslq_f64(nan_mask, vdupq_n_f64(f64::NAN), y);
    repair_exp_subnormal_lanes(input, y, underflow)
}

/// Vectorized ln(x) for NEON f64x2 using fdlibm kernel with IEEE 754 bit extraction.
///
/// Matches the AVX2 `ln_f64x4` algorithm from `simd_math.rs`, ported to 2-lane NEON.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
#[inline]
pub unsafe fn simd_ln_f64x2(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let sqrt_two = vdupq_n_f64(std::f64::consts::SQRT_2);

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

    // Fold m from [1, 2) into [sqrt(1/2), sqrt(2)). Values at or above
    // sqrt(2) are halved and their binary exponent incremented.
    let adjust = vcgeq_f64(m, sqrt_two);
    let m_halved = vmulq_f64(m, vdupq_n_f64(0.5));
    let k_plus_one = vaddq_f64(k, one);
    m = vbslq_f64(adjust, m_halved, m);
    k = vbslq_f64(adjust, k_plus_one, k);

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

    // Special values, mirroring the AVX2/AVX512 implementations:
    // ln(0) = -inf, ln(negative) = NaN, ln(+inf) = +inf, ln(NaN) = NaN. The
    // exponent mask above drops the sign bit, so without these blends ln(-x)
    // would silently return ln(|x|) and ln(0) a finite ~-709. The ordered
    // compares are all false for NaN input, so NaN needs its own check
    // (vceqq_f64(x, x) is false only for NaN lanes) or it would fall through
    // to finite garbage.
    let zero = vdupq_n_f64(0.0);
    let neg = vcltq_f64(x, zero);
    let eq_zero = vceqq_f64(x, zero);
    let is_inf = vceqq_f64(x, vdupq_n_f64(f64::INFINITY));
    let nan_mask = veorq_u64(vceqq_f64(x, x), vdupq_n_u64(u64::MAX));

    y = vbslq_f64(eq_zero, vdupq_n_f64(f64::NEG_INFINITY), y);
    y = vbslq_f64(neg, vdupq_n_f64(f64::NAN), y);
    y = vbslq_f64(is_inf, vdupq_n_f64(f64::INFINITY), y);
    y = vbslq_f64(nan_mask, vdupq_n_f64(f64::NAN), y);
    repair_ln_subnormal_lanes(x, y)
}

/// Evaluates the standard-normal density in both vector lanes.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
#[inline]
pub unsafe fn norm_pdf_f64x2(x: float64x2_t) -> float64x2_t {
    let exponent = vmulq_f64(vdupq_n_f64(-0.5), vmulq_f64(x, x));
    vmulq_f64(vdupq_n_f64(INV_SQRT_2PI), unsafe {
        simd_exp_f64x2(exponent)
    })
}

/// Evaluates the standard-normal cumulative distribution in both vector lanes.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
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
    let result = vbslq_f64(neg_mask, reflected, approx);
    let is_zero = vceqq_f64(x, zero);
    vbslq_f64(is_zero, vdupq_n_f64(0.5), result)
}

#[inline]
fn bs_price_scalar(spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool) -> f64 {
    crate::engines::analytic::black_scholes::bs_price(
        if is_call {
            crate::core::OptionType::Call
        } else {
            crate::core::OptionType::Put
        },
        spot,
        strike,
        r,
        q,
        vol,
        t,
    )
}

/// Prices a homogeneous Black-Scholes batch with AArch64 NEON where safe.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
/// `spots` and `strikes` must have identical lengths.
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
    unsafe {
        bs_price_neon_batch_into(spots, strikes, r, q, vol, t, is_call, &mut out);
    }
    out
}

/// Writes a homogeneous Black-Scholes batch using AArch64 NEON where safe.
///
/// # Safety
///
/// The caller must execute this on an AArch64 target where NEON is available.
/// `spots`, `strikes`, and `out` must have identical lengths.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bs_price_neon_batch_into(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    out: &mut [f64],
) {
    assert_eq!(
        spots.len(),
        strikes.len(),
        "spots and strikes must have identical lengths",
    );
    assert_eq!(
        spots.len(),
        out.len(),
        "output and input slices must have identical lengths",
    );

    let vector_drift = (r - q + 0.5 * vol * vol) * t;
    let requires_scalar = t <= 0.0
        || vol <= 0.0
        || !vector_drift.is_finite()
        || spots.iter().zip(strikes.iter()).any(|(&spot, &strike)| {
            !(spot / strike).is_finite()
                || crate::engines::analytic::bs_inline::requires_stable_price(
                    spot, strike, r, q, vol, t, is_call,
                )
        });
    if requires_scalar {
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
        unsafe { store_f64x2(out, i, if is_call { call } else { put }) };
        i += 2;
    }

    while i < n {
        out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIGH_EXP_RELATIVE_ERROR_BOUND: f64 = 2e-14;
    const FAST_EXP_RELATIVE_ERROR_BOUND: f64 = 8e-9;

    /// Special inputs covering underflow, overflow, infinities and NaN.
    const EXP_LN_SPECIALS: [f64; 11] = [
        f64::NEG_INFINITY,
        -1e308,
        -710.0,
        -708.5,
        f64::from_bits(1),
        f64::from_bits((1_u64 << 52) - 1),
        0.0,
        709.5,
        709.9,
        f64::INFINITY,
        f64::NAN,
    ];

    /// Compare a SIMD exp result against `std::f64::exp`. Rare subnormal lanes
    /// use the scalar repair path and therefore must match bit-for-bit.
    fn check_exp_special(x: f64, got: f64, tolerance: f64) {
        let expected = x.exp();
        if expected.is_nan() {
            assert!(got.is_nan(), "exp({x}) = {got}, expected NaN");
        } else if expected.is_infinite() {
            assert_eq!(got, expected, "exp({x}) = {got}, expected {expected}");
        } else if expected < f64::MIN_POSITIVE {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "exp({x}) = {got}, expected subnormal {expected}"
            );
        } else {
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel <= tolerance,
                "exp({x}) = {got}, expected {expected}, rel={rel}"
            );
        }
    }

    fn check_ln_special(x: f64, got: f64) {
        let expected = x.ln();
        if expected.is_nan() {
            assert!(got.is_nan(), "ln({x}) = {got}, expected NaN");
        } else if expected.is_infinite() {
            assert_eq!(got, expected, "ln({x}) = {got}, expected {expected}");
        } else if x > 0.0 && x < f64::MIN_POSITIVE {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "ln({x}) = {got}, expected repaired subnormal result {expected}"
            );
        } else {
            let abs_err = (got - expected).abs();
            let tolerance = 8.0 * f64::EPSILON * expected.abs().max(1.0);
            assert!(
                abs_err <= tolerance,
                "ln({x}) = {got}, expected {expected}, abs_err={abs_err}, tolerance={tolerance}"
            );
        }
    }

    fn assert_dense_ln_accuracy() {
        const SAMPLES: usize = 16_385;
        const EXPONENTS: [i32; 7] = [-1000, -100, -1, 0, 1, 100, 1000];

        for exponent in EXPONENTS {
            let scale = 2.0_f64.powi(exponent);
            for base in (0..SAMPLES).step_by(2) {
                let mut input = [1.0_f64; 2];
                let valid_lanes = (SAMPLES - base).min(2);
                for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                    let fraction = (base + lane) as f64 / SAMPLES as f64;
                    *value = (1.0 + fraction) * scale;
                }

                let mut out = [0.0_f64; 2];
                // SAFETY: AArch64 guarantees NEON and both arrays contain
                // two lanes.
                unsafe {
                    let x = load_f64x2(&input, 0);
                    store_f64x2(&mut out, 0, simd_ln_f64x2(x));
                }
                for lane in 0..valid_lanes {
                    check_ln_special(input[lane], out[lane]);
                }
            }
        }
    }

    fn assert_dense_exp_accuracy(start: f64, end: f64, samples: usize) {
        let mut max_high = (0.0_f64, 0.0_f64);
        let mut max_fast = (0.0_f64, 0.0_f64);
        let denominator = (samples - 1) as f64;

        for base in (0..samples).step_by(2) {
            let mut input = [0.0_f64; 2];
            let valid_lanes = (samples - base).min(2);
            for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                let sample = (base + lane) as f64;
                *value = (end - start).mul_add(sample / denominator, start);
            }

            let mut high = [0.0_f64; 2];
            let mut fast = [0.0_f64; 2];
            // SAFETY: AArch64 guarantees NEON and all arrays contain two lanes.
            unsafe {
                let x = load_f64x2(&input, 0);
                store_f64x2(&mut high, 0, simd_exp_f64x2(x));
                store_f64x2(&mut fast, 0, fast_exp_f64x2(x));
            }

            for lane in 0..valid_lanes {
                let x = input[lane];
                let expected = x.exp();
                let high_error = ((high[lane] - expected) / expected).abs();
                let fast_error = ((fast[lane] - expected) / expected).abs();
                if high_error > max_high.0 {
                    max_high = (high_error, x);
                }
                if fast_error > max_fast.0 {
                    max_fast = (fast_error, x);
                }
            }
        }

        assert!(
            max_high.0 <= HIGH_EXP_RELATIVE_ERROR_BOUND,
            "degree-11 exp max relative error {} at x={} exceeds {}",
            max_high.0,
            max_high.1,
            HIGH_EXP_RELATIVE_ERROR_BOUND
        );
        assert!(
            max_fast.0 <= FAST_EXP_RELATIVE_ERROR_BOUND,
            "degree-7 exp max relative error {} at x={} exceeds {}",
            max_fast.0,
            max_fast.1,
            FAST_EXP_RELATIVE_ERROR_BOUND
        );
    }

    #[test]
    fn simd_exp_f64x2_dense_accuracy_bounds() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let half_ln_2 = std::f64::consts::LN_2 * 0.5;
        assert_dense_exp_accuracy(-half_ln_2, half_ln_2, 16_385);
        assert_dense_exp_accuracy(-700.0, 700.0, 65_537);
    }

    #[test]
    fn simd_exp_f64x2_special_values_match_std() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(2) {
            let mut input = [0.0_f64; 2];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut high = [0.0_f64; 2];
            let mut fast = [0.0_f64; 2];
            // SAFETY: runtime feature check above; buffers hold 2 lanes.
            unsafe {
                let x = load_f64x2(&input, 0);
                store_f64x2(&mut high, 0, simd_exp_f64x2(x));
                store_f64x2(&mut fast, 0, fast_exp_f64x2(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_exp_special(x, high[i], HIGH_EXP_RELATIVE_ERROR_BOUND);
                check_exp_special(x, fast[i], FAST_EXP_RELATIVE_ERROR_BOUND);
            }
        }
    }

    #[test]
    fn simd_ln_f64x2_special_values_match_std() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(2) {
            let mut input = [1.0_f64; 2];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut out = [0.0_f64; 2];
            // SAFETY: runtime feature check above; buffers hold 2 lanes.
            unsafe {
                let x = load_f64x2(&input, 0);
                store_f64x2(&mut out, 0, simd_ln_f64x2(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_ln_special(x, out[i]);
            }
        }
    }

    #[test]
    fn simd_ln_f64x2_dense_accuracy_bound() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        assert_dense_ln_accuracy();
    }
}
