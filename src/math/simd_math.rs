//! Module `math::simd_math`.
//!
//! Implements simd math abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004) Ch. 5, Joe and Kuo (2008), SIMD and random-sequence implementation details tied to Eq. (5.4).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing loops; use higher-level modules when model semantics matter more than raw numerics.
#![cfg(all(feature = "simd", target_arch = "x86_64"))]

//! Shared AVX2/FMA SIMD math utilities.

use std::arch::x86_64::*;

use crate::math::fast_norm::accurate_norm_cdf;

const LN_2_HI: f64 = 6.931_471_803_691_238e-1;
const LN_2_LO: f64 = 1.908_214_929_270_587_7e-10;

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn splat_f64x4(val: f64) -> __m256d {
    _mm256_set1_pd(val)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available and `values[i..i + 4]` is in-bounds.
pub unsafe fn load_f64x4(values: &[f64], i: usize) -> __m256d {
    // SAFETY: caller guarantees there are at least 4 elements starting at `i`.
    unsafe { _mm256_loadu_pd(values.as_ptr().add(i)) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available and `values[i..i + 4]` is in-bounds.
pub unsafe fn store_f64x4(values: &mut [f64], i: usize, v: __m256d) {
    // SAFETY: caller guarantees there are at least 4 elements starting at `i`.
    unsafe { _mm256_storeu_pd(values.as_mut_ptr().add(i), v) };
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn repair_exp_subnormal_lanes(
    input: __m256d,
    result: __m256d,
    below_normal_range: __m256d,
) -> __m256d {
    if _mm256_movemask_pd(below_normal_range) == 0 {
        return result;
    }

    // Subnormal outputs are exceptionally rare in pricing workloads.  Repair
    // only those lanes with the platform scalar implementation, preserving
    // IEEE-754 semantics without burdening the hot normal-range polynomial.
    let mut inputs = [0.0_f64; 4];
    let mut outputs = [0.0_f64; 4];
    // SAFETY: both arrays contain four contiguous f64 lanes.
    unsafe {
        _mm256_storeu_pd(inputs.as_mut_ptr(), input);
        _mm256_storeu_pd(outputs.as_mut_ptr(), result);
    }
    for lane in 0..4 {
        if inputs[lane] < -708.396_418_532_264_1 {
            outputs[lane] = inputs[lane].exp();
        }
    }
    // SAFETY: `outputs` contains four contiguous f64 lanes.
    unsafe { _mm256_loadu_pd(outputs.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn repair_ln_subnormal_lanes(input: __m256d, result: __m256d) -> __m256d {
    let positive = _mm256_cmp_pd(input, _mm256_setzero_pd(), _CMP_GT_OQ);
    let below_normal = _mm256_cmp_pd(input, _mm256_set1_pd(f64::MIN_POSITIVE), _CMP_LT_OQ);
    let subnormal = _mm256_and_pd(positive, below_normal);
    if _mm256_movemask_pd(subnormal) == 0 {
        return result;
    }

    let mut inputs = [0.0_f64; 4];
    let mut outputs = [0.0_f64; 4];
    // SAFETY: both arrays contain four contiguous f64 lanes.
    unsafe {
        _mm256_storeu_pd(inputs.as_mut_ptr(), input);
        _mm256_storeu_pd(outputs.as_mut_ptr(), result);
    }
    for lane in 0..4 {
        if inputs[lane] > 0.0 && inputs[lane] < f64::MIN_POSITIVE {
            outputs[lane] = inputs[lane].ln();
        }
    }
    // SAFETY: `outputs` contains four contiguous f64 lanes.
    unsafe { _mm256_loadu_pd(outputs.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn exp_f64x4(x: __m256d) -> __m256d {
    let input = x;
    let max_x = _mm256_set1_pd(709.782_712_893_384);
    let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
    // Inputs beyond ln(f64::MAX) must overflow to +inf like std::exp.
    // The polynomial reconstructs normal outputs. Lanes whose exact result is
    // subnormal are repaired below; NaN must propagate instead of being
    // clamped into exp(max_x).
    let overflow = _mm256_cmp_pd(x, max_x, _CMP_GT_OQ);
    let underflow = _mm256_cmp_pd(x, min_x, _CMP_LT_OQ);
    let nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm256_round_pd(
        _mm256_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_HI), x);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_LO), r);

    // Degree-11 Taylor polynomial over |r| <= ln(2)/2. Dense differential
    // tests bound relative error by 2e-14 over the practical range [-700, 700].
    let c11 = _mm256_set1_pd(1.0 / 39_916_800.0);
    let c10 = _mm256_set1_pd(1.0 / 3_628_800.0);
    let c9 = _mm256_set1_pd(1.0 / 362_880.0);
    let c8 = _mm256_set1_pd(1.0 / 40_320.0);
    let c7 = _mm256_set1_pd(1.0 / 5_040.0);
    let c6 = _mm256_set1_pd(1.0 / 720.0);
    let c5 = _mm256_set1_pd(1.0 / 120.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c2 = _mm256_set1_pd(0.5);
    let c1 = _mm256_set1_pd(1.0);
    let c0 = _mm256_set1_pd(1.0);

    let mut poly = c11;
    poly = _mm256_fmadd_pd(poly, r, c10);
    poly = _mm256_fmadd_pd(poly, r, c9);
    poly = _mm256_fmadd_pd(poly, r, c8);
    poly = _mm256_fmadd_pd(poly, r, c7);
    poly = _mm256_fmadd_pd(poly, r, c6);
    poly = _mm256_fmadd_pd(poly, r, c5);
    poly = _mm256_fmadd_pd(poly, r, c4);
    poly = _mm256_fmadd_pd(poly, r, c3);
    poly = _mm256_fmadd_pd(poly, r, c2);
    poly = _mm256_fmadd_pd(poly, r, c1);
    poly = _mm256_fmadd_pd(poly, r, c0);

    // Reconstruct 2^n as 2^n1 * 2^n2 (n1 = n/2, n2 = n - n1). A single 2^n
    // overflows the biased exponent for n = 1024, which round(x*log2e)
    // produces for x in [~709.44, 709.78] where exp(x) is still finite.
    let n_i32 = _mm256_cvtpd_epi32(n);
    let n1_i32 = _mm_srai_epi32::<1>(n_i32);
    let n2_i32 = _mm_sub_epi32(n_i32, n1_i32);
    let bias = _mm256_set1_epi64x(1023);
    let e1 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(n1_i32), bias), 52);
    let e2 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(n2_i32), bias), 52);
    let y = _mm256_mul_pd(
        _mm256_mul_pd(poly, _mm256_castsi256_pd(e1)),
        _mm256_castsi256_pd(e2),
    );

    let y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::INFINITY), overflow);
    let y = _mm256_blendv_pd(y, _mm256_setzero_pd(), underflow);
    let y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NAN), nan_mask);
    unsafe { repair_exp_subnormal_lanes(input, y, underflow) }
}

/// Fast exp() with a degree-7 polynomial.
///
/// Saves four FMA operations versus the degree-11 version. Dense differential
/// tests bound relative error by `8e-9` over the practical finite range
/// `[-700, 700]`.
#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn fast_exp_f64x4(x: __m256d) -> __m256d {
    let input = x;
    let max_x = _mm256_set1_pd(709.782_712_893_384);
    let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
    // Special values as in `exp_f64x4`: overflow -> +inf, underflow -> 0.0,
    // NaN propagates.
    let overflow = _mm256_cmp_pd(x, max_x, _CMP_GT_OQ);
    let underflow = _mm256_cmp_pd(x, min_x, _CMP_LT_OQ);
    let nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm256_round_pd(
        _mm256_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_HI), x);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_LO), r);

    // Degree-7 Taylor-like polynomial over |r| <= ln(2)/2:
    //   p(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
    // The coefficients are rounded near reciprocal factorials; this is not a
    // Remez minimax fit. The tested relative-error bound is 8e-9.
    let c7 = _mm256_set1_pd(1.984_126_984_12e-4); // ≈ 1/5040
    let c6 = _mm256_set1_pd(1.388_888_889_0e-3); // ≈ 1/720
    let c5 = _mm256_set1_pd(8.333_333_333_3e-3); // ≈ 1/120
    let c4 = _mm256_set1_pd(4.166_666_666_67e-2); // ≈ 1/24
    let c3 = _mm256_set1_pd(1.666_666_666_666_67e-1); // ≈ 1/6
    let c2 = _mm256_set1_pd(0.5);
    let c1 = _mm256_set1_pd(1.0);
    let c0 = _mm256_set1_pd(1.0);

    let mut poly = c7;
    poly = _mm256_fmadd_pd(poly, r, c6);
    poly = _mm256_fmadd_pd(poly, r, c5);
    poly = _mm256_fmadd_pd(poly, r, c4);
    poly = _mm256_fmadd_pd(poly, r, c3);
    poly = _mm256_fmadd_pd(poly, r, c2);
    poly = _mm256_fmadd_pd(poly, r, c1);
    poly = _mm256_fmadd_pd(poly, r, c0);

    // Split 2^n into 2^n1 * 2^n2 to avoid biased-exponent overflow at n=1024.
    let n_i32 = _mm256_cvtpd_epi32(n);
    let n1_i32 = _mm_srai_epi32::<1>(n_i32);
    let n2_i32 = _mm_sub_epi32(n_i32, n1_i32);
    let bias = _mm256_set1_epi64x(1023);
    let e1 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(n1_i32), bias), 52);
    let e2 = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(n2_i32), bias), 52);
    let y = _mm256_mul_pd(
        _mm256_mul_pd(poly, _mm256_castsi256_pd(e1)),
        _mm256_castsi256_pd(e2),
    );

    let y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::INFINITY), overflow);
    let y = _mm256_blendv_pd(y, _mm256_setzero_pd(), underflow);
    let y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NAN), nan_mask);
    unsafe { repair_exp_subnormal_lanes(input, y, underflow) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn ln_f64x4(x: __m256d) -> __m256d {
    // Extract exponent and mantissa: x = m * 2^k, m in [1, 2).
    let x_bits = _mm256_castpd_si256(x);
    let exp_bits = _mm256_srli_epi64(
        _mm256_and_si256(x_bits, _mm256_set1_epi64x(0x7ff0_0000_0000_0000_u64 as i64)),
        52,
    );

    // Convert biased exponent to f64 entirely in SIMD (no store-load roundtrip).
    // Subtract bias 1023 in integer domain, then pack 4×i64 → 4×i32 and use
    // hardware i32→f64 conversion. Exponents are in [-1023, 1024] so i32 is safe.
    let bias = _mm256_set1_epi64x(1023);
    let unbiased = _mm256_sub_epi64(exp_bits, bias);
    // Pack: extract low 32 bits of each i64 lane via shuffle, then combine.
    let shuffled = _mm256_shuffle_epi32(unbiased, 0b10_00_10_00);
    let lo128 = _mm256_castsi256_si128(shuffled);
    let hi128 = _mm256_extracti128_si256(shuffled, 1);
    let packed_i32 = _mm_unpacklo_epi64(lo128, hi128); // [k0, k1, k2, k3] as i32
    let mut k = _mm256_cvtepi32_pd(packed_i32);

    let mant_bits = _mm256_or_si256(
        _mm256_and_si256(x_bits, _mm256_set1_epi64x(0x000f_ffff_ffff_ffff_u64 as i64)),
        _mm256_set1_epi64x(0x3ff0_0000_0000_0000_u64 as i64),
    );
    let mut m = _mm256_castsi256_pd(mant_bits);

    // Fold m from [1, 2) into [sqrt(1/2), sqrt(2)). The lower part is
    // already in range; values at or above sqrt(2) must be halved while
    // incrementing the binary exponent.
    let sqrt_two = _mm256_set1_pd(std::f64::consts::SQRT_2);
    let one = _mm256_set1_pd(1.0);
    let adjust = _mm256_cmp_pd(m, sqrt_two, _CMP_GE_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), adjust);
    k = _mm256_blendv_pd(k, _mm256_add_pd(k, one), adjust);

    // Degree-7 minimax (fdlibm kernel) for ln(1+f), f = m-1.
    let f = _mm256_sub_pd(m, one);
    let s = _mm256_div_pd(f, _mm256_add_pd(_mm256_set1_pd(2.0), f));
    let z = _mm256_mul_pd(s, s);
    let w = _mm256_mul_pd(z, z);

    let lg1 = _mm256_set1_pd(6.666_666_666_666_735e-1);
    let lg2 = _mm256_set1_pd(3.999_999_999_940_942e-1);
    let lg3 = _mm256_set1_pd(2.857_142_874_366_239e-1);
    let lg4 = _mm256_set1_pd(2.222_219_843_214_978_4e-1);
    let lg5 = _mm256_set1_pd(1.818_357_216_161_805e-1);
    let lg6 = _mm256_set1_pd(1.531_383_769_920_937_3e-1);
    let lg7 = _mm256_set1_pd(1.479_819_860_511_658_6e-1);

    let t1 = _mm256_mul_pd(w, _mm256_fmadd_pd(w, _mm256_fmadd_pd(w, lg6, lg4), lg2));
    let t2 = _mm256_mul_pd(
        z,
        _mm256_fmadd_pd(
            w,
            _mm256_fmadd_pd(w, _mm256_fmadd_pd(w, lg7, lg5), lg3),
            lg1,
        ),
    );
    let r = _mm256_add_pd(t1, t2);

    let hfsq = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(f, f));
    let ln_m = _mm256_sub_pd(
        f,
        _mm256_sub_pd(hfsq, _mm256_mul_pd(s, _mm256_add_pd(hfsq, r))),
    );

    let mut y = _mm256_fmadd_pd(k, _mm256_set1_pd(LN_2_HI), ln_m);
    y = _mm256_fmadd_pd(k, _mm256_set1_pd(LN_2_LO), y);

    // Special values: ln(0)=-inf, ln(neg)=NaN, ln(+inf)=+inf, ln(NaN)=NaN.
    // The ordered compares below are all false for NaN input, so NaN needs
    // its own unordered check or it would fall through to finite garbage.
    let zero = _mm256_setzero_pd();
    let neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
    let eq_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
    let is_inf = _mm256_cmp_pd(x, _mm256_set1_pd(f64::INFINITY), _CMP_EQ_OQ);
    let nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NEG_INFINITY), eq_zero);
    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NAN), neg);
    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::INFINITY), is_inf);
    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NAN), nan_mask);
    unsafe { repair_ln_subnormal_lanes(x, y) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn norm_pdf_f64x4(x: __m256d) -> __m256d {
    let inv_sqrt_2pi = _mm256_set1_pd(0.398_942_280_401_432_7);
    let exponent = _mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x));
    _mm256_mul_pd(inv_sqrt_2pi, unsafe { exp_f64x4(exponent) })
}

#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn norm_cdf_f64x4(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let zero = _mm256_setzero_pd();
    let sign_mask = _mm256_set1_pd(-0.0);
    let z = _mm256_andnot_pd(sign_mask, x);

    let t = _mm256_div_pd(one, _mm256_fmadd_pd(_mm256_set1_pd(0.231_641_9), z, one));
    let a1 = _mm256_set1_pd(0.319_381_530);
    let a2 = _mm256_set1_pd(-0.356_563_782);
    let a3 = _mm256_set1_pd(1.781_477_937);
    let a4 = _mm256_set1_pd(-1.821_255_978);
    let a5 = _mm256_set1_pd(1.330_274_429);

    let mut poly = a5;
    poly = _mm256_fmadd_pd(poly, t, a4);
    poly = _mm256_fmadd_pd(poly, t, a3);
    poly = _mm256_fmadd_pd(poly, t, a2);
    poly = _mm256_fmadd_pd(poly, t, a1);
    poly = _mm256_mul_pd(poly, t);

    let approx = _mm256_fnmadd_pd(unsafe { norm_pdf_f64x4(z) }, poly, one);
    let reflected = _mm256_sub_pd(one, approx);
    let neg_mask = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
    let result = _mm256_blendv_pd(approx, reflected, neg_mask);
    let is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
    _mm256_blendv_pd(result, _mm256_set1_pd(0.5), is_zero)
}

/// Production-accuracy normal CDF evaluated independently in each AVX2 lane.
///
/// AVX2 has no `erfc` instruction.  The Cody evaluations are therefore scalar,
/// while callers retain vectorized log, discounting, and payoff arithmetic.
/// [`norm_cdf_f64x4`] remains the explicit fast A&S approximation.
#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn accurate_norm_cdf_f64x4(x: __m256d) -> __m256d {
    let mut lanes = [0.0_f64; 4];
    // SAFETY: `lanes` contains four contiguous f64 values.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), x) };
    for value in &mut lanes {
        *value = accurate_norm_cdf(*value);
    }
    // SAFETY: `lanes` contains four contiguous f64 values.
    unsafe { _mm256_loadu_pd(lanes.as_ptr()) }
}

// ──────────────────────────────────────────────────────────────────────────
// AVX2 vectorized inverse normal CDF (Acklam's rational approximation).
//
// Processes 4 values simultaneously. This is the bottleneck in every MC
// path because each random uniform must be mapped to a normal variate.
// ──────────────────────────────────────────────────────────────────────────

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

/// Vectorized inverse normal CDF for 4 probabilities in `[0, 1]`.
///
/// Uses Acklam's rational approximation with three regions:
///   - low tail  (p < P_LOW):  log-based rational
///   - central   (P_LOW <= p <= P_HIGH): quadratic rational
///   - high tail (p > P_HIGH): reflected low tail
///
/// All three branches are computed simultaneously and blended with SIMD masks.
/// This eliminates the data-dependent branching in the scalar version.
#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must ensure AVX2+FMA are available on the executing CPU.
pub unsafe fn inv_norm_cdf_f64x4(p: __m256d) -> __m256d {
    unsafe {
        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);
        let neg_two = _mm256_set1_pd(-2.0);
        let p_low = _mm256_set1_pd(INV_CDF_P_LOW);
        let p_high = _mm256_set1_pd(INV_CDF_P_HIGH);

        // ── Central region: P_LOW <= p <= P_HIGH ──
        let q_central = _mm256_sub_pd(p, half);
        let r_central = _mm256_mul_pd(q_central, q_central);

        let mut num_c = _mm256_set1_pd(ACKLAM_A[0]);
        num_c = _mm256_fmadd_pd(num_c, r_central, _mm256_set1_pd(ACKLAM_A[1]));
        num_c = _mm256_fmadd_pd(num_c, r_central, _mm256_set1_pd(ACKLAM_A[2]));
        num_c = _mm256_fmadd_pd(num_c, r_central, _mm256_set1_pd(ACKLAM_A[3]));
        num_c = _mm256_fmadd_pd(num_c, r_central, _mm256_set1_pd(ACKLAM_A[4]));
        num_c = _mm256_fmadd_pd(num_c, r_central, _mm256_set1_pd(ACKLAM_A[5]));
        num_c = _mm256_mul_pd(num_c, q_central);

        let mut den_c = _mm256_set1_pd(ACKLAM_B[0]);
        den_c = _mm256_fmadd_pd(den_c, r_central, _mm256_set1_pd(ACKLAM_B[1]));
        den_c = _mm256_fmadd_pd(den_c, r_central, _mm256_set1_pd(ACKLAM_B[2]));
        den_c = _mm256_fmadd_pd(den_c, r_central, _mm256_set1_pd(ACKLAM_B[3]));
        den_c = _mm256_fmadd_pd(den_c, r_central, _mm256_set1_pd(ACKLAM_B[4]));
        den_c = _mm256_fmadd_pd(den_c, r_central, one);

        let val_central = _mm256_div_pd(num_c, den_c);

        // ── Low tail: p < P_LOW ──
        let ln_p = ln_f64x4(p);
        let q_low = _mm256_sqrt_pd(_mm256_mul_pd(neg_two, ln_p));

        let mut num_l = _mm256_set1_pd(ACKLAM_C[0]);
        num_l = _mm256_fmadd_pd(num_l, q_low, _mm256_set1_pd(ACKLAM_C[1]));
        num_l = _mm256_fmadd_pd(num_l, q_low, _mm256_set1_pd(ACKLAM_C[2]));
        num_l = _mm256_fmadd_pd(num_l, q_low, _mm256_set1_pd(ACKLAM_C[3]));
        num_l = _mm256_fmadd_pd(num_l, q_low, _mm256_set1_pd(ACKLAM_C[4]));
        num_l = _mm256_fmadd_pd(num_l, q_low, _mm256_set1_pd(ACKLAM_C[5]));

        let mut den_l = _mm256_set1_pd(ACKLAM_D[0]);
        den_l = _mm256_fmadd_pd(den_l, q_low, _mm256_set1_pd(ACKLAM_D[1]));
        den_l = _mm256_fmadd_pd(den_l, q_low, _mm256_set1_pd(ACKLAM_D[2]));
        den_l = _mm256_fmadd_pd(den_l, q_low, _mm256_set1_pd(ACKLAM_D[3]));
        den_l = _mm256_fmadd_pd(den_l, q_low, one);

        let val_low = _mm256_div_pd(num_l, den_l);

        // ── High tail: p > P_HIGH ──
        let one_minus_p = _mm256_sub_pd(one, p);
        let ln_1mp = ln_f64x4(one_minus_p);
        let q_high = _mm256_sqrt_pd(_mm256_mul_pd(neg_two, ln_1mp));

        let mut num_h = _mm256_set1_pd(ACKLAM_C[0]);
        num_h = _mm256_fmadd_pd(num_h, q_high, _mm256_set1_pd(ACKLAM_C[1]));
        num_h = _mm256_fmadd_pd(num_h, q_high, _mm256_set1_pd(ACKLAM_C[2]));
        num_h = _mm256_fmadd_pd(num_h, q_high, _mm256_set1_pd(ACKLAM_C[3]));
        num_h = _mm256_fmadd_pd(num_h, q_high, _mm256_set1_pd(ACKLAM_C[4]));
        num_h = _mm256_fmadd_pd(num_h, q_high, _mm256_set1_pd(ACKLAM_C[5]));

        let mut den_h = _mm256_set1_pd(ACKLAM_D[0]);
        den_h = _mm256_fmadd_pd(den_h, q_high, _mm256_set1_pd(ACKLAM_D[1]));
        den_h = _mm256_fmadd_pd(den_h, q_high, _mm256_set1_pd(ACKLAM_D[2]));
        den_h = _mm256_fmadd_pd(den_h, q_high, _mm256_set1_pd(ACKLAM_D[3]));
        den_h = _mm256_fmadd_pd(den_h, q_high, one);

        let val_high = _mm256_xor_pd(_mm256_div_pd(num_h, den_h), _mm256_set1_pd(-0.0));

        // ── Blend the three regions ──
        let is_low = _mm256_cmp_pd(p, p_low, _CMP_LT_OQ);
        let is_high = _mm256_cmp_pd(p, p_high, _CMP_GT_OQ);

        let result = _mm256_blendv_pd(val_central, val_low, is_low);
        let result = _mm256_blendv_pd(result, val_high, is_high);

        // ── Domain boundaries: match the scalar contract of
        // `beasley_springer_moro_inv_cdf` exactly so a value's result does not
        // depend on whether it lands in the vector body or scalar remainder:
        //   p == 0 -> -inf, p == 1 -> +inf, p < 0 / p > 1 / NaN -> NaN.
        let zero = _mm256_setzero_pd();
        let is_zero = _mm256_cmp_pd(p, zero, _CMP_EQ_OQ);
        let is_one = _mm256_cmp_pd(p, one, _CMP_EQ_OQ);
        let below = _mm256_cmp_pd(p, zero, _CMP_LT_OQ);
        let above = _mm256_cmp_pd(p, one, _CMP_GT_OQ);
        let unordered = _mm256_cmp_pd(p, p, _CMP_UNORD_Q);
        let invalid = _mm256_or_pd(_mm256_or_pd(below, above), unordered);

        let result = _mm256_blendv_pd(result, _mm256_set1_pd(f64::NEG_INFINITY), is_zero);
        let result = _mm256_blendv_pd(result, _mm256_set1_pd(f64::INFINITY), is_one);
        _mm256_blendv_pd(result, _mm256_set1_pd(f64::NAN), invalid)
    }
}

/// Batch inverse normal CDF: processes `uniforms` buffer in-place, writing
/// normal variates back into the same slice. Falls back to scalar for
/// the remainder that doesn't fill a 4-wide SIMD register.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available (runtime check).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn inv_norm_cdf_batch_avx2(uniforms: &mut [f64]) {
    let n = uniforms.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let p = unsafe { _mm256_loadu_pd(uniforms.as_ptr().add(i)) };
        let z = unsafe { inv_norm_cdf_f64x4(p) };
        unsafe { _mm256_storeu_pd(uniforms.as_mut_ptr().add(i), z) };
        i += 4;
    }
    // Scalar remainder
    while i < n {
        uniforms[i] = crate::math::fast_norm::beasley_springer_moro_inv_cdf(uniforms[i]);
        i += 1;
    }
}

/// Generate `n` uniform samples into `buf`, then batch-convert to normals via SIMD.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available (runtime check).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fill_normals_simd(
    rng: &mut crate::math::fast_rng::Xoshiro256PlusPlus,
    buf: &mut [f64],
) {
    // Step 1: Fill with uniform open (ε, 1−ε) values.
    let eps = f64::EPSILON;
    let hi = 1.0 - eps;
    for v in buf.iter_mut() {
        let u = rng.next_f64();
        *v = u.max(eps).min(hi);
    }
    // Step 2: Batch inverse CDF transform via AVX2.
    unsafe { inv_norm_cdf_batch_avx2(buf) };
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
            for base in (0..SAMPLES).step_by(4) {
                let mut input = [1.0_f64; 4];
                let valid_lanes = (SAMPLES - base).min(4);
                for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                    let fraction = (base + lane) as f64 / SAMPLES as f64;
                    *value = (1.0 + fraction) * scale;
                }

                let mut out = [0.0_f64; 4];
                // SAFETY: the calling test performs the AVX2+FMA runtime
                // check and both arrays contain four lanes.
                unsafe {
                    let x = load_f64x4(&input, 0);
                    store_f64x4(&mut out, 0, ln_f64x4(x));
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

        for base in (0..samples).step_by(4) {
            let mut input = [0.0_f64; 4];
            let valid_lanes = (samples - base).min(4);
            for (lane, value) in input.iter_mut().take(valid_lanes).enumerate() {
                let sample = (base + lane) as f64;
                *value = (end - start).mul_add(sample / denominator, start);
            }

            let mut high = [0.0_f64; 4];
            let mut fast = [0.0_f64; 4];
            // SAFETY: the calling tests perform the AVX2+FMA runtime check and
            // all arrays contain four lanes.
            unsafe {
                let x = load_f64x4(&input, 0);
                store_f64x4(&mut high, 0, exp_f64x4(x));
                store_f64x4(&mut fast, 0, fast_exp_f64x4(x));
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
    fn exp_f64x4_dense_accuracy_bounds() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let half_ln_2 = std::f64::consts::LN_2 * 0.5;
        assert_dense_exp_accuracy(-half_ln_2, half_ln_2, 16_385);
        assert_dense_exp_accuracy(-700.0, 700.0, 65_537);
    }

    #[test]
    fn exp_f64x4_special_values_match_std() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(4) {
            let mut input = [0.0_f64; 4];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut exact = [0.0_f64; 4];
            let mut fast = [0.0_f64; 4];
            // SAFETY: runtime feature check above; buffers hold 4 lanes.
            unsafe {
                let x = load_f64x4(&input, 0);
                store_f64x4(&mut exact, 0, exp_f64x4(x));
                store_f64x4(&mut fast, 0, fast_exp_f64x4(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_exp_special(x, exact[i], HIGH_EXP_RELATIVE_ERROR_BOUND);
                check_exp_special(x, fast[i], FAST_EXP_RELATIVE_ERROR_BOUND);
            }
        }
    }

    #[test]
    fn ln_f64x4_special_values_match_std() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for chunk in EXP_LN_SPECIALS.chunks(4) {
            let mut input = [1.0_f64; 4];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut out = [0.0_f64; 4];
            // SAFETY: runtime feature check above; buffers hold 4 lanes.
            unsafe {
                let x = load_f64x4(&input, 0);
                store_f64x4(&mut out, 0, ln_f64x4(x));
            }
            for (i, &x) in chunk.iter().enumerate() {
                check_ln_special(x, out[i]);
            }
        }
    }

    #[test]
    fn ln_f64x4_dense_accuracy_bound() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        assert_dense_ln_accuracy();
    }
}
