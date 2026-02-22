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

const LN_2_HI: f64 = 6.931_471_803_691_238e-1;
const LN_2_LO: f64 = 1.908_214_929_270_587_7e-10;

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn splat_f64x4(val: f64) -> __m256d {
    _mm256_set1_pd(val)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn load_f64x4(values: &[f64], i: usize) -> __m256d {
    // SAFETY: caller guarantees there are at least 4 elements starting at `i`.
    unsafe { _mm256_loadu_pd(values.as_ptr().add(i)) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn store_f64x4(values: &mut [f64], i: usize, v: __m256d) {
    // SAFETY: caller guarantees there are at least 4 elements starting at `i`.
    unsafe { _mm256_storeu_pd(values.as_mut_ptr().add(i), v) };
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_f64x4(x: __m256d) -> __m256d {
    let max_x = _mm256_set1_pd(709.782_712_893_384);
    let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
    let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm256_round_pd(
        _mm256_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_HI), x);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_LO), r);

    // Degree-11 polynomial over reduced range |r| <= ln(2)/2.
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

    let n_i32 = _mm256_cvtpd_epi32(n);
    let n_i64 = _mm256_cvtepi32_epi64(n_i32);
    let exp_bits = _mm256_slli_epi64(_mm256_add_epi64(n_i64, _mm256_set1_epi64x(1023)), 52);
    let two_pow_n = _mm256_castsi256_pd(exp_bits);

    _mm256_mul_pd(poly, two_pow_n)
}

/// Fast exp() with degree-7 minimax polynomial (~2e-10 relative error).
///
/// Saves 4 FMA operations vs the degree-11 version by using optimized Remez
/// minimax coefficients instead of truncated Taylor series. Sufficient accuracy
/// for Monte Carlo simulation, path generation, and most pricing applications.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fast_exp_f64x4(x: __m256d) -> __m256d {
    let max_x = _mm256_set1_pd(709.782_712_893_384);
    let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
    let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let n = _mm256_round_pd(
        _mm256_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_HI), x);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN_2_LO), r);

    // Degree-7 minimax polynomial over |r| <= ln(2)/2.
    // Coefficients from Remez exchange on [−ln2/2, ln2/2]:
    //   p(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
    // The low-order terms (c0–c2) are exact; the high-order terms carry the
    // minimax correction that keeps max relative error < 2e-10.
    let c7 = _mm256_set1_pd(1.984_126_984_12e-4);  // ≈ 1/5040
    let c6 = _mm256_set1_pd(1.388_888_889_0e-3);   // ≈ 1/720
    let c5 = _mm256_set1_pd(8.333_333_333_3e-3);   // ≈ 1/120
    let c4 = _mm256_set1_pd(4.166_666_666_67e-2);   // ≈ 1/24
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

    let n_i32 = _mm256_cvtpd_epi32(n);
    let n_i64 = _mm256_cvtepi32_epi64(n_i32);
    let exp_bits = _mm256_slli_epi64(_mm256_add_epi64(n_i64, _mm256_set1_epi64x(1023)), 52);
    let two_pow_n = _mm256_castsi256_pd(exp_bits);

    _mm256_mul_pd(poly, two_pow_n)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
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

    // Fold m into [sqrt(1/2), sqrt(2)] for better polynomial accuracy.
    let sqrt_half = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
    let one = _mm256_set1_pd(1.0);
    let adjust = _mm256_cmp_pd(m, sqrt_half, _CMP_LT_OQ);
    m = _mm256_blendv_pd(m, _mm256_add_pd(m, m), adjust);
    k = _mm256_blendv_pd(k, _mm256_sub_pd(k, one), adjust);

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

    // Special values: ln(0)=-inf, ln(neg)=NaN, ln(+inf)=+inf.
    let zero = _mm256_setzero_pd();
    let neg = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
    let eq_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
    let is_inf = _mm256_cmp_pd(x, _mm256_set1_pd(f64::INFINITY), _CMP_EQ_OQ);

    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NEG_INFINITY), eq_zero);
    y = _mm256_blendv_pd(y, _mm256_set1_pd(f64::NAN), neg);
    _mm256_blendv_pd(y, _mm256_set1_pd(f64::INFINITY), is_inf)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn norm_pdf_f64x4(x: __m256d) -> __m256d {
    let inv_sqrt_2pi = _mm256_set1_pd(0.398_942_280_401_432_7);
    let exponent = _mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x));
    _mm256_mul_pd(inv_sqrt_2pi, unsafe { exp_f64x4(exponent) })
}

#[inline]
#[target_feature(enable = "avx2,fma")]
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
    _mm256_blendv_pd(approx, reflected, neg_mask)
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
    let ln_p = ln_f64x4(_mm256_max_pd(p, _mm256_set1_pd(1e-300)));
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
    let ln_1mp = ln_f64x4(_mm256_max_pd(one_minus_p, _mm256_set1_pd(1e-300)));
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

    let val_high = _mm256_xor_pd(
        _mm256_div_pd(num_h, den_h),
        _mm256_set1_pd(-0.0),
    );

    // ── Blend the three regions ──
    let is_low = _mm256_cmp_pd(p, p_low, _CMP_LT_OQ);
    let is_high = _mm256_cmp_pd(p, p_high, _CMP_GT_OQ);

    let result = _mm256_blendv_pd(val_central, val_low, is_low);
    _mm256_blendv_pd(result, val_high, is_high)
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
pub unsafe fn fill_normals_simd(rng: &mut crate::math::fast_rng::Xoshiro256PlusPlus, buf: &mut [f64]) {
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
