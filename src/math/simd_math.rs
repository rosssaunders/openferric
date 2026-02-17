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
