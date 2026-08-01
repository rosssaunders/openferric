//! WebAssembly SIMD128 helpers for analytic batch pricing.
//!
//! WebAssembly SIMD has two `f64` lanes but no vector `ln` or `exp`
//! instructions. These helpers deliberately keep those transcendental
//! operations lane-scalar and vectorize the surrounding normal-CDF
//! polynomial and Black-Scholes arithmetic. This still moves the substantial
//! repeated add/multiply/divide work onto explicit `f64x2` instructions while
//! preserving Rust's scalar transcendental behavior.

use std::arch::wasm32::*;

use crate::math::fast_norm::accurate_norm_cdf;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[inline]
pub(crate) fn load_f64x2(values: &[f64], index: usize) -> v128 {
    f64x2(values[index], values[index + 1])
}

#[inline]
pub(crate) fn store_f64x2(values: &mut [f64], index: usize, vector: v128) {
    values[index] = f64x2_extract_lane::<0>(vector);
    values[index + 1] = f64x2_extract_lane::<1>(vector);
}

/// Evaluate the same A&S 7.1.26 approximation as `normal_cdf_approx`.
///
/// The polynomial, sign selection, and PDF scaling use SIMD128. WebAssembly
/// has no vector exponential, so the two exponential evaluations remain
/// lane-scalar.
#[inline]
pub(crate) fn normal_cdf_f64x2(x: v128) -> v128 {
    let one = f64x2_splat(1.0);
    let z = f64x2_abs(x);
    let t = f64x2_div(one, f64x2_add(f64x2_mul(f64x2_splat(0.231_641_9), z), one));

    let mut poly = f64x2_splat(1.330_274_429);
    poly = f64x2_add(f64x2_mul(poly, t), f64x2_splat(-1.821_255_978));
    poly = f64x2_add(f64x2_mul(poly, t), f64x2_splat(1.781_477_937));
    poly = f64x2_add(f64x2_mul(poly, t), f64x2_splat(-0.356_563_782));
    poly = f64x2_add(f64x2_mul(poly, t), f64x2_splat(0.319_381_530));
    poly = f64x2_mul(poly, t);

    let exponent = f64x2_mul(f64x2_splat(-0.5), f64x2_mul(z, z));
    let pdf = f64x2(
        INV_SQRT_2PI * f64x2_extract_lane::<0>(exponent).exp(),
        INV_SQRT_2PI * f64x2_extract_lane::<1>(exponent).exp(),
    );
    let cdf_pos = f64x2_sub(one, f64x2_mul(pdf, poly));
    let cdf_neg = f64x2_sub(one, cdf_pos);

    // Integer comparison observes the IEEE-754 sign bit. Restore the exact
    // mathematical midpoint for both signed zeros after selecting the side.
    let negative = i64x2_lt(x, i64x2_splat(0));
    let result = v128_bitselect(cdf_neg, cdf_pos, negative);
    let is_zero = f64x2_eq(x, f64x2_splat(0.0));
    v128_bitselect(f64x2_splat(0.5), result, is_zero)
}

/// Evaluate the production-accuracy normal CDF independently in each lane.
///
/// WebAssembly SIMD has no vector `erfc`, so the Cody evaluations remain
/// lane-scalar while the surrounding Black-Scholes arithmetic stays `f64x2`.
/// The explicitly named [`normal_cdf_f64x2`] helper above deliberately keeps
/// the faster A&S approximation for `normal_cdf_batch_approx` callers.
#[inline]
pub(crate) fn accurate_normal_cdf_f64x2(x: v128) -> v128 {
    f64x2(
        accurate_norm_cdf(f64x2_extract_lane::<0>(x)),
        accurate_norm_cdf(f64x2_extract_lane::<1>(x)),
    )
}

pub(crate) fn normal_cdf_batch_into(xs: &[f64], out: &mut [f64]) {
    let mut index = 0;
    while index + 2 <= xs.len() {
        let values = load_f64x2(xs, index);
        store_f64x2(out, index, normal_cdf_f64x2(values));
        index += 2;
    }
    if index < xs.len() {
        out[index] = crate::engines::analytic::normal_cdf_approx(xs[index]);
    }
}
