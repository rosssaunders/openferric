//! Module `math::simd_avx512`.
//!
//! Implements AVX-512 (8-wide f64) SIMD math utilities for high-throughput pricing.
//!
//! References: Glasserman (2004) Ch. 5, Joe and Kuo (2008), SIMD and random-sequence
//! implementation details tied to Eq. (5.4).
//!
//! Primary API surface: module-level functions for 8-wide f64 math operations.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision
//! cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing
//! loops on CPUs with AVX-512F support; use higher-level modules when model semantics
//! matter more than raw numerics.
#![cfg(all(feature = "simd", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)]

//! Shared AVX-512 SIMD math utilities (8-wide f64).

use std::arch::x86_64::*;

use crate::math::fast_norm::accurate_norm_cdf;

const LN_2_HI: f64 = 6.931_471_803_691_238e-1;
const LN_2_LO: f64 = 1.908_214_929_270_587_7e-10;

/// Acklam rational-approximation coefficients for the central region.
const ACKLAM_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];
const ACKLAM_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];
const ACKLAM_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];
const ACKLAM_D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];
const INV_CDF_P_LOW: f64 = 0.024_25;
const INV_CDF_P_HIGH: f64 = 1.0 - INV_CDF_P_LOW;

#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn splat_f64x8(val: f64) -> __m512d {
    _mm512_set1_pd(val)
}

#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available and `values[i..i + 8]` is in-bounds.
pub unsafe fn load_f64x8(values: &[f64], i: usize) -> __m512d {
    // SAFETY: caller guarantees there are at least 8 elements starting at `i`.
    _mm512_loadu_pd(values.as_ptr().add(i))
}

#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available and `values[i..i + 8]` is in-bounds.
pub unsafe fn store_f64x8(values: &mut [f64], i: usize, v: __m512d) {
    // SAFETY: caller guarantees there are at least 8 elements starting at `i`.
    _mm512_storeu_pd(values.as_mut_ptr().add(i), v);
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn repair_exp_subnormal_lanes(
    input: __m512d,
    result: __m512d,
    below_normal_range: __mmask8,
) -> __m512d {
    if below_normal_range == 0 {
        return result;
    }

    let mut inputs = [0.0_f64; 8];
    let mut outputs = [0.0_f64; 8];
    _mm512_storeu_pd(inputs.as_mut_ptr(), input);
    _mm512_storeu_pd(outputs.as_mut_ptr(), result);
    for lane in 0..8 {
        if inputs[lane] < -708.396_418_532_264_1 {
            outputs[lane] = inputs[lane].exp();
        }
    }
    _mm512_loadu_pd(outputs.as_ptr())
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn repair_ln_subnormal_lanes(input: __m512d, result: __m512d) -> __m512d {
    let positive = _mm512_cmp_pd_mask(input, _mm512_setzero_pd(), _CMP_GT_OQ);
    let below_normal = _mm512_cmp_pd_mask(input, _mm512_set1_pd(f64::MIN_POSITIVE), _CMP_LT_OQ);
    if (positive & below_normal) == 0 {
        return result;
    }

    let mut inputs = [0.0_f64; 8];
    let mut outputs = [0.0_f64; 8];
    _mm512_storeu_pd(inputs.as_mut_ptr(), input);
    _mm512_storeu_pd(outputs.as_mut_ptr(), result);
    for lane in 0..8 {
        if inputs[lane] > 0.0 && inputs[lane] < f64::MIN_POSITIVE {
            outputs[lane] = inputs[lane].ln();
        }
    }
    _mm512_loadu_pd(outputs.as_ptr())
}

/// AVX-512 exp() with a degree-11 polynomial.
///
/// Processes 8 f64 values simultaneously using 512-bit vectors. Uses the same
/// Cody-Waite range reduction and degree-11 Taylor polynomial as the AVX2
/// version. Dense differential tests bound relative error by `2e-14` over the
/// practical finite range `[-700, 700]`.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn exp_f64x8(x: __m512d) -> __m512d {
    let input = x;
    let max_x = _mm512_set1_pd(709.782_712_893_384);
    let min_x = _mm512_set1_pd(-708.396_418_532_264_1);
    // Inputs beyond ln(f64::MAX) must overflow to +inf like std::exp.
    // Lanes whose exact result is subnormal are repaired below; NaN must
    // propagate instead of being clamped into exp(max_x).
    let overflow = _mm512_cmp_pd_mask(x, max_x, _CMP_GT_OQ);
    let underflow = _mm512_cmp_pd_mask(x, min_x, _CMP_LT_OQ);
    let nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
    let x = _mm512_max_pd(min_x, _mm512_min_pd(x, max_x));

    let log2e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm512_roundscale_pd(
        _mm512_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // Cody-Waite range reduction: r = x - n * ln(2)
    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_HI), x);
    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_LO), r);

    // Degree-11 polynomial over reduced range |r| <= ln(2)/2.
    let c11 = _mm512_set1_pd(1.0 / 39_916_800.0);
    let c10 = _mm512_set1_pd(1.0 / 3_628_800.0);
    let c9 = _mm512_set1_pd(1.0 / 362_880.0);
    let c8 = _mm512_set1_pd(1.0 / 40_320.0);
    let c7 = _mm512_set1_pd(1.0 / 5_040.0);
    let c6 = _mm512_set1_pd(1.0 / 720.0);
    let c5 = _mm512_set1_pd(1.0 / 120.0);
    let c4 = _mm512_set1_pd(1.0 / 24.0);
    let c3 = _mm512_set1_pd(1.0 / 6.0);
    let c2 = _mm512_set1_pd(0.5);
    let c1 = _mm512_set1_pd(1.0);
    let c0 = _mm512_set1_pd(1.0);

    let mut poly = c11;
    poly = _mm512_fmadd_pd(poly, r, c10);
    poly = _mm512_fmadd_pd(poly, r, c9);
    poly = _mm512_fmadd_pd(poly, r, c8);
    poly = _mm512_fmadd_pd(poly, r, c7);
    poly = _mm512_fmadd_pd(poly, r, c6);
    poly = _mm512_fmadd_pd(poly, r, c5);
    poly = _mm512_fmadd_pd(poly, r, c4);
    poly = _mm512_fmadd_pd(poly, r, c3);
    poly = _mm512_fmadd_pd(poly, r, c2);
    poly = _mm512_fmadd_pd(poly, r, c1);
    poly = _mm512_fmadd_pd(poly, r, c0);

    // Reconstruct 2^n as 2^n1 * 2^n2 (n1 = n/2, n2 = n - n1). A single 2^n
    // overflows the biased exponent for n = 1024, which round(x*log2e)
    // produces for x in [~709.44, 709.78] where exp(x) is still finite.
    let n_i32 = _mm512_cvtpd_epi32(n);
    let n_i64 = _mm512_cvtepi32_epi64(n_i32);
    let n1_i64 = _mm512_srai_epi64(n_i64, 1);
    let n2_i64 = _mm512_sub_epi64(n_i64, n1_i64);
    let bias = _mm512_set1_epi64(1023);
    let e1 = _mm512_slli_epi64(_mm512_add_epi64(n1_i64, bias), 52);
    let e2 = _mm512_slli_epi64(_mm512_add_epi64(n2_i64, bias), 52);
    let y = _mm512_mul_pd(
        _mm512_mul_pd(poly, _mm512_castsi512_pd(e1)),
        _mm512_castsi512_pd(e2),
    );

    let y = _mm512_mask_blend_pd(overflow, y, _mm512_set1_pd(f64::INFINITY));
    let y = _mm512_mask_blend_pd(underflow, y, _mm512_setzero_pd());
    let y = _mm512_mask_blend_pd(nan_mask, y, _mm512_set1_pd(f64::NAN));
    repair_exp_subnormal_lanes(input, y, underflow)
}

/// Fast exp() with a degree-7 polynomial.
///
/// Saves four FMA operations versus the degree-11 version. Dense differential
/// tests bound relative error by `8e-9` over the practical finite range
/// `[-700, 700]`.
///
/// Processes 8 f64 values simultaneously using 512-bit vectors.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn fast_exp_f64x8(x: __m512d) -> __m512d {
    let input = x;
    let max_x = _mm512_set1_pd(709.782_712_893_384);
    let min_x = _mm512_set1_pd(-708.396_418_532_264_1);
    // Special values as in `exp_f64x8`: overflow -> +inf, underflow -> 0.0,
    // NaN propagates.
    let overflow = _mm512_cmp_pd_mask(x, max_x, _CMP_GT_OQ);
    let underflow = _mm512_cmp_pd_mask(x, min_x, _CMP_LT_OQ);
    let nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
    let x = _mm512_max_pd(min_x, _mm512_min_pd(x, max_x));

    let log2e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm512_roundscale_pd(
        _mm512_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_HI), x);
    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_LO), r);

    // Degree-7 Taylor-like polynomial over |r| <= ln(2)/2:
    //   p(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040
    // The coefficients are rounded near reciprocal factorials; this is not a
    // Remez minimax fit. The tested relative-error bound is 8e-9.
    let c7 = _mm512_set1_pd(1.984_126_984_12e-4); // ~ 1/5040
    let c6 = _mm512_set1_pd(1.388_888_889_0e-3); // ~ 1/720
    let c5 = _mm512_set1_pd(8.333_333_333_3e-3); // ~ 1/120
    let c4 = _mm512_set1_pd(4.166_666_666_67e-2); // ~ 1/24
    let c3 = _mm512_set1_pd(1.666_666_666_666_67e-1); // ~ 1/6
    let c2 = _mm512_set1_pd(0.5);
    let c1 = _mm512_set1_pd(1.0);
    let c0 = _mm512_set1_pd(1.0);

    let mut poly = c7;
    poly = _mm512_fmadd_pd(poly, r, c6);
    poly = _mm512_fmadd_pd(poly, r, c5);
    poly = _mm512_fmadd_pd(poly, r, c4);
    poly = _mm512_fmadd_pd(poly, r, c3);
    poly = _mm512_fmadd_pd(poly, r, c2);
    poly = _mm512_fmadd_pd(poly, r, c1);
    poly = _mm512_fmadd_pd(poly, r, c0);

    // Split 2^n into 2^n1 * 2^n2 to avoid biased-exponent overflow at n=1024.
    let n_i32 = _mm512_cvtpd_epi32(n);
    let n_i64 = _mm512_cvtepi32_epi64(n_i32);
    let n1_i64 = _mm512_srai_epi64(n_i64, 1);
    let n2_i64 = _mm512_sub_epi64(n_i64, n1_i64);
    let bias = _mm512_set1_epi64(1023);
    let e1 = _mm512_slli_epi64(_mm512_add_epi64(n1_i64, bias), 52);
    let e2 = _mm512_slli_epi64(_mm512_add_epi64(n2_i64, bias), 52);
    let y = _mm512_mul_pd(
        _mm512_mul_pd(poly, _mm512_castsi512_pd(e1)),
        _mm512_castsi512_pd(e2),
    );

    let y = _mm512_mask_blend_pd(overflow, y, _mm512_set1_pd(f64::INFINITY));
    let y = _mm512_mask_blend_pd(underflow, y, _mm512_setzero_pd());
    let y = _mm512_mask_blend_pd(nan_mask, y, _mm512_set1_pd(f64::NAN));
    repair_exp_subnormal_lanes(input, y, underflow)
}

/// AVX-512 natural logarithm for 8 f64 values simultaneously.
///
/// Uses the same fdlibm-style kernel as the AVX2 version: extract exponent
/// and mantissa, fold into [sqrt(1/2), sqrt(2)], then degree-7 minimax
/// polynomial for ln(1+f).
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn ln_f64x8(x: __m512d) -> __m512d {
    // Extract exponent and mantissa: x = m * 2^k, m in [1, 2).
    let x_bits = _mm512_castpd_si512(x);
    let exp_bits = _mm512_srli_epi64(
        _mm512_and_epi64(x_bits, _mm512_set1_epi64(0x7ff0_0000_0000_0000_u64 as i64)),
        52,
    );

    // Convert biased exponent to f64.
    // Subtract bias 1023 in integer domain, then convert i64 -> i32 -> f64.
    let bias = _mm512_set1_epi64(1023);
    let unbiased = _mm512_sub_epi64(exp_bits, bias);
    // Pack i64 -> i32 (exponents fit in i32), then convert to f64.
    let unbiased_i32 = _mm512_cvtepi64_epi32(unbiased);
    let mut k = _mm512_cvtepi32_pd(unbiased_i32);

    let mant_bits = _mm512_or_epi64(
        _mm512_and_epi64(x_bits, _mm512_set1_epi64(0x000f_ffff_ffff_ffff_u64 as i64)),
        _mm512_set1_epi64(0x3ff0_0000_0000_0000_u64 as i64),
    );
    let mut m = _mm512_castsi512_pd(mant_bits);

    // Fold m from [1, 2) into [sqrt(1/2), sqrt(2)). Values at or above
    // sqrt(2) are halved and their binary exponent incremented.
    let sqrt_two = _mm512_set1_pd(std::f64::consts::SQRT_2);
    let one = _mm512_set1_pd(1.0);
    let adjust = _mm512_cmp_pd_mask(m, sqrt_two, _CMP_GE_OQ);
    m = _mm512_mask_blend_pd(adjust, m, _mm512_mul_pd(m, _mm512_set1_pd(0.5)));
    k = _mm512_mask_blend_pd(adjust, k, _mm512_add_pd(k, one));

    // Degree-7 minimax (fdlibm kernel) for ln(1+f), f = m-1.
    let f = _mm512_sub_pd(m, one);
    let s = _mm512_div_pd(f, _mm512_add_pd(_mm512_set1_pd(2.0), f));
    let z = _mm512_mul_pd(s, s);
    let w = _mm512_mul_pd(z, z);

    let lg1 = _mm512_set1_pd(6.666_666_666_666_735e-1);
    let lg2 = _mm512_set1_pd(3.999_999_999_940_942e-1);
    let lg3 = _mm512_set1_pd(2.857_142_874_366_239e-1);
    let lg4 = _mm512_set1_pd(2.222_219_843_214_978_4e-1);
    let lg5 = _mm512_set1_pd(1.818_357_216_161_805e-1);
    let lg6 = _mm512_set1_pd(1.531_383_769_920_937_3e-1);
    let lg7 = _mm512_set1_pd(1.479_819_860_511_658_6e-1);

    let t1 = _mm512_mul_pd(w, _mm512_fmadd_pd(w, _mm512_fmadd_pd(w, lg6, lg4), lg2));
    let t2 = _mm512_mul_pd(
        z,
        _mm512_fmadd_pd(
            w,
            _mm512_fmadd_pd(w, _mm512_fmadd_pd(w, lg7, lg5), lg3),
            lg1,
        ),
    );
    let r = _mm512_add_pd(t1, t2);

    let hfsq = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(f, f));
    let ln_m = _mm512_sub_pd(
        f,
        _mm512_sub_pd(hfsq, _mm512_mul_pd(s, _mm512_add_pd(hfsq, r))),
    );

    let mut y = _mm512_fmadd_pd(k, _mm512_set1_pd(LN_2_HI), ln_m);
    y = _mm512_fmadd_pd(k, _mm512_set1_pd(LN_2_LO), y);

    // Special values: ln(0)=-inf, ln(neg)=NaN, ln(+inf)=+inf, ln(NaN)=NaN.
    // The ordered compares below are all false for NaN input, so NaN needs
    // its own unordered check or it would fall through to finite garbage.
    let zero = _mm512_setzero_pd();
    let neg = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
    let eq_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
    let is_inf = _mm512_cmp_pd_mask(x, _mm512_set1_pd(f64::INFINITY), _CMP_EQ_OQ);
    let nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);

    y = _mm512_mask_blend_pd(eq_zero, y, _mm512_set1_pd(f64::NEG_INFINITY));
    y = _mm512_mask_blend_pd(neg, y, _mm512_set1_pd(f64::NAN));
    y = _mm512_mask_blend_pd(is_inf, y, _mm512_set1_pd(f64::INFINITY));
    y = _mm512_mask_blend_pd(nan_mask, y, _mm512_set1_pd(f64::NAN));
    repair_ln_subnormal_lanes(x, y)
}

/// AVX-512 standard normal PDF for 8 f64 values simultaneously.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn norm_pdf_f64x8(x: __m512d) -> __m512d {
    let inv_sqrt_2pi = _mm512_set1_pd(0.398_942_280_401_432_7);
    let exponent = _mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(x, x));
    _mm512_mul_pd(inv_sqrt_2pi, exp_f64x8(exponent))
}

/// AVX-512 standard normal CDF (Abramowitz & Stegun 7.1.26) for 8 f64 values.
///
/// Branch-free implementation using AVX-512 mask registers for sign handling.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn norm_cdf_f64x8(x: __m512d) -> __m512d {
    let one = _mm512_set1_pd(1.0);
    let zero = _mm512_setzero_pd();

    // Compute |x| using absolute value via clearing sign bit.
    let sign_mask_bits = _mm512_set1_epi64(0x7fff_ffff_ffff_ffff_u64 as i64);
    let z = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), sign_mask_bits));

    let t = _mm512_div_pd(one, _mm512_fmadd_pd(_mm512_set1_pd(0.231_641_9), z, one));
    let a1 = _mm512_set1_pd(0.319_381_530);
    let a2 = _mm512_set1_pd(-0.356_563_782);
    let a3 = _mm512_set1_pd(1.781_477_937);
    let a4 = _mm512_set1_pd(-1.821_255_978);
    let a5 = _mm512_set1_pd(1.330_274_429);

    let mut poly = a5;
    poly = _mm512_fmadd_pd(poly, t, a4);
    poly = _mm512_fmadd_pd(poly, t, a3);
    poly = _mm512_fmadd_pd(poly, t, a2);
    poly = _mm512_fmadd_pd(poly, t, a1);
    poly = _mm512_mul_pd(poly, t);

    let approx = _mm512_fnmadd_pd(norm_pdf_f64x8(z), poly, one);
    let reflected = _mm512_sub_pd(one, approx);
    let neg_mask = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
    let result = _mm512_mask_blend_pd(neg_mask, approx, reflected);
    let is_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
    _mm512_mask_blend_pd(is_zero, result, _mm512_set1_pd(0.5))
}

/// Production-accuracy normal CDF evaluated independently in each AVX-512 lane.
///
/// AVX-512 has no `erfc` instruction.  The Cody evaluations are therefore
/// scalar, while callers retain vectorized log, discounting, and payoff
/// arithmetic. [`norm_cdf_f64x8`] remains the explicit fast A&S approximation.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn accurate_norm_cdf_f64x8(x: __m512d) -> __m512d {
    let mut lanes = [0.0_f64; 8];
    // SAFETY: `lanes` contains eight contiguous f64 values.
    _mm512_storeu_pd(lanes.as_mut_ptr(), x);
    for value in &mut lanes {
        *value = accurate_norm_cdf(*value);
    }
    // SAFETY: `lanes` contains eight contiguous f64 values.
    _mm512_loadu_pd(lanes.as_ptr())
}

/// AVX-512 vectorized inverse normal CDF for 8 probabilities in `[0, 1]`.
///
/// Uses Acklam's rational approximation with three regions:
///   - low tail  (p < P_LOW):  log-based rational
///   - central   (P_LOW <= p <= P_HIGH): quadratic rational
///   - high tail (p > P_HIGH): reflected low tail
///
/// All three branches are computed simultaneously and blended with AVX-512 masks.
/// This eliminates the data-dependent branching in the scalar version.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn inv_norm_cdf_f64x8(p: __m512d) -> __m512d {
    let one = _mm512_set1_pd(1.0);
    let half = _mm512_set1_pd(0.5);
    let neg_two = _mm512_set1_pd(-2.0);
    let p_low = _mm512_set1_pd(INV_CDF_P_LOW);
    let p_high = _mm512_set1_pd(INV_CDF_P_HIGH);

    // -- Central region: P_LOW <= p <= P_HIGH --
    let q_central = _mm512_sub_pd(p, half);
    let r_central = _mm512_mul_pd(q_central, q_central);

    let mut num_c = _mm512_set1_pd(ACKLAM_A[0]);
    num_c = _mm512_fmadd_pd(num_c, r_central, _mm512_set1_pd(ACKLAM_A[1]));
    num_c = _mm512_fmadd_pd(num_c, r_central, _mm512_set1_pd(ACKLAM_A[2]));
    num_c = _mm512_fmadd_pd(num_c, r_central, _mm512_set1_pd(ACKLAM_A[3]));
    num_c = _mm512_fmadd_pd(num_c, r_central, _mm512_set1_pd(ACKLAM_A[4]));
    num_c = _mm512_fmadd_pd(num_c, r_central, _mm512_set1_pd(ACKLAM_A[5]));
    num_c = _mm512_mul_pd(num_c, q_central);

    let mut den_c = _mm512_set1_pd(ACKLAM_B[0]);
    den_c = _mm512_fmadd_pd(den_c, r_central, _mm512_set1_pd(ACKLAM_B[1]));
    den_c = _mm512_fmadd_pd(den_c, r_central, _mm512_set1_pd(ACKLAM_B[2]));
    den_c = _mm512_fmadd_pd(den_c, r_central, _mm512_set1_pd(ACKLAM_B[3]));
    den_c = _mm512_fmadd_pd(den_c, r_central, _mm512_set1_pd(ACKLAM_B[4]));
    den_c = _mm512_fmadd_pd(den_c, r_central, one);

    let val_central = _mm512_div_pd(num_c, den_c);

    // -- Low tail: p < P_LOW --
    let ln_p = ln_f64x8(p);
    let q_low = _mm512_sqrt_pd(_mm512_mul_pd(neg_two, ln_p));

    let mut num_l = _mm512_set1_pd(ACKLAM_C[0]);
    num_l = _mm512_fmadd_pd(num_l, q_low, _mm512_set1_pd(ACKLAM_C[1]));
    num_l = _mm512_fmadd_pd(num_l, q_low, _mm512_set1_pd(ACKLAM_C[2]));
    num_l = _mm512_fmadd_pd(num_l, q_low, _mm512_set1_pd(ACKLAM_C[3]));
    num_l = _mm512_fmadd_pd(num_l, q_low, _mm512_set1_pd(ACKLAM_C[4]));
    num_l = _mm512_fmadd_pd(num_l, q_low, _mm512_set1_pd(ACKLAM_C[5]));

    let mut den_l = _mm512_set1_pd(ACKLAM_D[0]);
    den_l = _mm512_fmadd_pd(den_l, q_low, _mm512_set1_pd(ACKLAM_D[1]));
    den_l = _mm512_fmadd_pd(den_l, q_low, _mm512_set1_pd(ACKLAM_D[2]));
    den_l = _mm512_fmadd_pd(den_l, q_low, _mm512_set1_pd(ACKLAM_D[3]));
    den_l = _mm512_fmadd_pd(den_l, q_low, one);

    let val_low = _mm512_div_pd(num_l, den_l);

    // -- High tail: p > P_HIGH --
    let one_minus_p = _mm512_sub_pd(one, p);
    let ln_1mp = ln_f64x8(one_minus_p);
    let q_high = _mm512_sqrt_pd(_mm512_mul_pd(neg_two, ln_1mp));

    let mut num_h = _mm512_set1_pd(ACKLAM_C[0]);
    num_h = _mm512_fmadd_pd(num_h, q_high, _mm512_set1_pd(ACKLAM_C[1]));
    num_h = _mm512_fmadd_pd(num_h, q_high, _mm512_set1_pd(ACKLAM_C[2]));
    num_h = _mm512_fmadd_pd(num_h, q_high, _mm512_set1_pd(ACKLAM_C[3]));
    num_h = _mm512_fmadd_pd(num_h, q_high, _mm512_set1_pd(ACKLAM_C[4]));
    num_h = _mm512_fmadd_pd(num_h, q_high, _mm512_set1_pd(ACKLAM_C[5]));

    let mut den_h = _mm512_set1_pd(ACKLAM_D[0]);
    den_h = _mm512_fmadd_pd(den_h, q_high, _mm512_set1_pd(ACKLAM_D[1]));
    den_h = _mm512_fmadd_pd(den_h, q_high, _mm512_set1_pd(ACKLAM_D[2]));
    den_h = _mm512_fmadd_pd(den_h, q_high, _mm512_set1_pd(ACKLAM_D[3]));
    den_h = _mm512_fmadd_pd(den_h, q_high, one);

    // Negate high tail result: val_high = -(num_h / den_h)
    let val_high = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_div_pd(num_h, den_h));

    // -- Blend the three regions --
    let is_low = _mm512_cmp_pd_mask(p, p_low, _CMP_LT_OQ);
    let is_high = _mm512_cmp_pd_mask(p, p_high, _CMP_GT_OQ);

    let result = _mm512_mask_blend_pd(is_low, val_central, val_low);
    let result = _mm512_mask_blend_pd(is_high, result, val_high);

    // -- Domain boundaries: match the scalar contract of
    // `beasley_springer_moro_inv_cdf` exactly so a value's result does not
    // depend on whether it lands in the vector body or scalar remainder:
    //   p == 0 -> -inf, p == 1 -> +inf, p < 0 / p > 1 / NaN -> NaN.
    let zero = _mm512_setzero_pd();
    let is_zero = _mm512_cmp_pd_mask(p, zero, _CMP_EQ_OQ);
    let is_one = _mm512_cmp_pd_mask(p, one, _CMP_EQ_OQ);
    let below = _mm512_cmp_pd_mask(p, zero, _CMP_LT_OQ);
    let above = _mm512_cmp_pd_mask(p, one, _CMP_GT_OQ);
    let unordered = _mm512_cmp_pd_mask(p, p, _CMP_UNORD_Q);
    let invalid = below | above | unordered;

    let result = _mm512_mask_blend_pd(is_zero, result, _mm512_set1_pd(f64::NEG_INFINITY));
    let result = _mm512_mask_blend_pd(is_one, result, _mm512_set1_pd(f64::INFINITY));
    _mm512_mask_blend_pd(invalid, result, _mm512_set1_pd(f64::NAN))
}

/// Batch inverse normal CDF: processes `uniforms` buffer in-place, writing
/// normal variates back into the same slice. Falls back to scalar for
/// the remainder that doesn't fill an 8-wide SIMD register.
///
/// # Safety
/// Caller must ensure AVX-512F is available (runtime check).
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn inv_norm_cdf_batch_avx512(uniforms: &mut [f64]) {
    let n = uniforms.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let p = _mm512_loadu_pd(uniforms.as_ptr().add(i));
        let z = inv_norm_cdf_f64x8(p);
        _mm512_storeu_pd(uniforms.as_mut_ptr().add(i), z);
        i += 8;
    }
    // Scalar remainder
    while i < n {
        uniforms[i] = crate::math::fast_norm::beasley_springer_moro_inv_cdf(uniforms[i]);
        i += 1;
    }
}

/// Generate `n` uniform samples into `buf`, then batch-convert to normals via AVX-512.
///
/// # Safety
/// Caller must ensure AVX-512F is available (runtime check).
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn fill_normals_simd_avx512(
    rng: &mut crate::math::fast_rng::Xoshiro256PlusPlus,
    buf: &mut [f64],
) {
    // Step 1: Fill with uniform open (eps, 1-eps) values.
    let eps = f64::EPSILON;
    let hi = 1.0 - eps;
    for v in buf.iter_mut() {
        let u = rng.next_f64();
        *v = u.max(eps).min(hi);
    }
    // Step 2: Batch inverse CDF transform via AVX-512.
    inv_norm_cdf_batch_avx512(buf);
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
    fn check_exp_special(x: f64, got: f64, tol: f64) {
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
                rel <= tol,
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
            for base in (0..SAMPLES).step_by(8) {
                let mut input = [1.0_f64; 8];
                let valid_lanes = (SAMPLES - base).min(8);
                for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                    let fraction = (base + lane) as f64 / SAMPLES as f64;
                    *value = (1.0 + fraction) * scale;
                }

                let mut out = [0.0_f64; 8];
                // SAFETY: the calling test performs the AVX-512F runtime
                // check and both arrays contain eight lanes.
                unsafe {
                    let x = load_f64x8(&input, 0);
                    store_f64x8(&mut out, 0, ln_f64x8(x));
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

        for base in (0..samples).step_by(8) {
            let mut input = [0.0_f64; 8];
            let valid_lanes = (samples - base).min(8);
            for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                let sample = (base + lane) as f64;
                *value = (end - start).mul_add(sample / denominator, start);
            }

            let mut high = [0.0_f64; 8];
            let mut fast = [0.0_f64; 8];
            // SAFETY: the calling test performs the AVX-512F runtime check and
            // all arrays contain eight lanes.
            unsafe {
                let x = load_f64x8(&input, 0);
                store_f64x8(&mut high, 0, exp_f64x8(x));
                store_f64x8(&mut fast, 0, fast_exp_f64x8(x));
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
    fn exp_f64x8_dense_accuracy_bounds() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let half_ln_2 = std::f64::consts::LN_2 * 0.5;
        assert_dense_exp_accuracy(-half_ln_2, half_ln_2, 16_385);
        assert_dense_exp_accuracy(-700.0, 700.0, 65_537);
    }

    #[test]
    fn exp_f64x8_special_values_match_std() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(8) {
            let mut input = [0.0_f64; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut exact = [0.0_f64; 8];
            let mut fast = [0.0_f64; 8];
            // SAFETY: runtime feature check above; buffers hold 8 lanes.
            unsafe {
                let x = load_f64x8(&input, 0);
                store_f64x8(&mut exact, 0, exp_f64x8(x));
                store_f64x8(&mut fast, 0, fast_exp_f64x8(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_exp_special(x, exact[i], HIGH_EXP_RELATIVE_ERROR_BOUND);
                check_exp_special(x, fast[i], FAST_EXP_RELATIVE_ERROR_BOUND);
            }
        }
    }

    #[test]
    fn ln_f64x8_special_values_match_std() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(8) {
            let mut input = [1.0_f64; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut out = [0.0_f64; 8];
            // SAFETY: runtime feature check above; buffers hold 8 lanes.
            unsafe {
                let x = load_f64x8(&input, 0);
                store_f64x8(&mut out, 0, ln_f64x8(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_ln_special(x, out[i]);
            }
        }
    }

    #[test]
    fn ln_f64x8_dense_accuracy_bound() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        assert_dense_ln_accuracy();
    }
}
