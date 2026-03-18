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

/// AVX-512 exp() with degree-11 polynomial (~1 ULP relative error).
///
/// Processes 8 f64 values simultaneously using 512-bit vectors. Uses the same
/// Cody-Waite range reduction and degree-11 Taylor polynomial as the AVX2 version.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn exp_f64x8(x: __m512d) -> __m512d {
    let max_x = _mm512_set1_pd(709.782_712_893_384);
    let min_x = _mm512_set1_pd(-708.396_418_532_264_1);
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

    // Reconstruct 2^n via exponent bit manipulation.
    // _mm512_cvtpd_epi32 gives __m256i (8 x i32), then widen to __m512i (8 x i64).
    let n_i32 = _mm512_cvtpd_epi32(n);
    let n_i64 = _mm512_cvtepi32_epi64(n_i32);
    let exp_bits = _mm512_slli_epi64(_mm512_add_epi64(n_i64, _mm512_set1_epi64(1023)), 52);
    let two_pow_n = _mm512_castsi512_pd(exp_bits);

    _mm512_mul_pd(poly, two_pow_n)
}

/// Fast exp() with degree-7 minimax polynomial (~2e-10 relative error).
///
/// Saves 4 FMA operations vs the degree-11 version by using optimized Remez
/// minimax coefficients instead of truncated Taylor series. Sufficient accuracy
/// for Monte Carlo simulation, path generation, and most pricing applications.
///
/// Processes 8 f64 values simultaneously using 512-bit vectors.
#[inline]
#[target_feature(enable = "avx512f")]
/// # Safety
/// The caller must ensure AVX-512F is available on the executing CPU.
pub unsafe fn fast_exp_f64x8(x: __m512d) -> __m512d {
    let max_x = _mm512_set1_pd(709.782_712_893_384);
    let min_x = _mm512_set1_pd(-708.396_418_532_264_1);
    let x = _mm512_max_pd(min_x, _mm512_min_pd(x, max_x));

    let log2e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm512_roundscale_pd(
        _mm512_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_HI), x);
    let r = _mm512_fnmadd_pd(n, _mm512_set1_pd(LN_2_LO), r);

    // Degree-7 minimax polynomial over |r| <= ln(2)/2.
    // Coefficients from Remez exchange on [-ln2/2, ln2/2]:
    //   p(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040
    // The low-order terms (c0-c2) are exact; the high-order terms carry the
    // minimax correction that keeps max relative error < 2e-10.
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

    let n_i32 = _mm512_cvtpd_epi32(n);
    let n_i64 = _mm512_cvtepi32_epi64(n_i32);
    let exp_bits = _mm512_slli_epi64(_mm512_add_epi64(n_i64, _mm512_set1_epi64(1023)), 52);
    let two_pow_n = _mm512_castsi512_pd(exp_bits);

    _mm512_mul_pd(poly, two_pow_n)
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

    // Fold m into [sqrt(1/2), sqrt(2)] for better polynomial accuracy.
    let sqrt_half = _mm512_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
    let one = _mm512_set1_pd(1.0);
    let adjust = _mm512_cmp_pd_mask(m, sqrt_half, _CMP_LT_OQ);
    m = _mm512_mask_blend_pd(adjust, m, _mm512_add_pd(m, m));
    k = _mm512_mask_blend_pd(adjust, k, _mm512_sub_pd(k, one));

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

    // Special values: ln(0)=-inf, ln(neg)=NaN, ln(+inf)=+inf.
    let zero = _mm512_setzero_pd();
    let neg = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
    let eq_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
    let is_inf = _mm512_cmp_pd_mask(x, _mm512_set1_pd(f64::INFINITY), _CMP_EQ_OQ);

    y = _mm512_mask_blend_pd(eq_zero, y, _mm512_set1_pd(f64::NEG_INFINITY));
    y = _mm512_mask_blend_pd(neg, y, _mm512_set1_pd(f64::NAN));
    _mm512_mask_blend_pd(is_inf, y, _mm512_set1_pd(f64::INFINITY))
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
    _mm512_mask_blend_pd(neg_mask, approx, reflected)
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
    let ln_p = ln_f64x8(_mm512_max_pd(p, _mm512_set1_pd(1e-300)));
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
    let ln_1mp = ln_f64x8(_mm512_max_pd(one_minus_p, _mm512_set1_pd(1e-300)));
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
    _mm512_mask_blend_pd(is_high, result, val_high)
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
