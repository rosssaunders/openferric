//! Module `engines::analytic::bs_simd`.
//!
//! Implements bs simd workflows with concrete routines such as `normal_cdf_approx`, `normal_cdf_batch_approx`, `bs_price_batch`, `bs_greeks_batch`.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Primary API surface: free functions `normal_cdf_approx`, `normal_cdf_batch_approx`, `bs_price_batch`, `bs_greeks_batch`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.

use std::sync::OnceLock;

use crate::math::fast_norm::accurate_norm_cdf;

/// Below this total volatility the absolute error of the fast A&S CDF can be
/// larger than the option's time value.  Use the Cody/erfc implementation for
/// this numerically sensitive region.
const ACCURATE_CDF_TOTAL_VOL: f64 = 1.0e-3;

/// SIMD implementation selected for batch analytic pricing on this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSimdBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    /// Compile-time WebAssembly SIMD128 (`f64x2`) backend.
    WasmSimd128,
}

/// Detect the widest supported batch backend once and cache the result.
pub fn detected_batch_simd_backend() -> BatchSimdBackend {
    static BACKEND: OnceLock<BatchSimdBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            return BatchSimdBackend::Avx512;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return BatchSimdBackend::Avx2;
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return BatchSimdBackend::Neon;
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
        let backend = BatchSimdBackend::WasmSimd128;
        #[cfg(not(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")))]
        let backend = BatchSimdBackend::Scalar;
        backend
    })
}

#[inline]
fn backend_width(backend: BatchSimdBackend) -> usize {
    match backend {
        BatchSimdBackend::Avx512 => 8,
        BatchSimdBackend::Avx2 => 4,
        BatchSimdBackend::Neon | BatchSimdBackend::WasmSimd128 => 2,
        BatchSimdBackend::Scalar => 1,
    }
}

/// A partial vector plus dispatch overhead loses to the scalar loop for the
/// small batches commonly produced by Python and iterator adapters.
#[inline]
fn simd_is_worthwhile(len: usize, backend: BatchSimdBackend) -> bool {
    let width = backend_width(backend);
    width > 1 && len >= width && (len >= 4 * width || len.is_multiple_of(width))
}

#[inline]
fn batch_requires_scalar(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> bool {
    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    let vector_drift = (0.5 * vol).mul_add(vol, r - q) * t;
    !r.is_finite()
        || !q.is_finite()
        || !vol.is_finite()
        || !t.is_finite()
        || !vector_drift.is_finite()
        || spots.iter().zip(strikes.iter()).any(|(&spot, &strike)| {
            !spot.is_finite()
                || !strike.is_finite()
                || spot <= 0.0
                || strike <= 0.0
                || !(spot * df_q).is_finite()
                || !(strike * df_r).is_finite()
                || !(spot / strike).is_finite()
                || super::bs_inline::requires_stable_price(spot, strike, r, q, vol, t, is_call)
        })
}

#[inline]
fn selected_price_backend(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> BatchSimdBackend {
    let backend = detected_batch_simd_backend();
    selected_price_backend_for(backend, spots, strikes, r, q, vol, t, is_call)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn selected_price_backend_for(
    backend: BatchSimdBackend,
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> BatchSimdBackend {
    let sensitive_total_vol = t > 0.0 && vol > 0.0 && vol * t.sqrt() <= ACCURATE_CDF_TOTAL_VOL;
    if t <= 0.0
        || vol <= 0.0
        || sensitive_total_vol
        || !simd_is_worthwhile(spots.len(), backend)
        || batch_requires_scalar(spots, strikes, r, q, vol, t, is_call)
    {
        BatchSimdBackend::Scalar
    } else {
        backend
    }
}

/// Scalar Abramowitz & Stegun 7.1.26 normal CDF approximation.
///
/// Branch-free implementation using bit-level sign extraction,
/// eliminating branch misprediction on random inputs.
#[inline]
pub fn normal_cdf_approx(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    // Preserve the exact mathematical value for both +0.0 and -0.0.  The
    // sign-bit selection below would otherwise put the two zeros on opposite
    // sides of the approximation error.
    if x == 0.0 {
        return 0.5;
    }

    let z = x.abs();
    let t = 1.0 / P.mul_add(z, 1.0);
    let poly = A5
        .mul_add(t, A4)
        .mul_add(t, A3)
        .mul_add(t, A2)
        .mul_add(t, A1)
        * t;
    let pdf = INV_SQRT_2PI * (-0.5 * z * z).exp();
    let cdf_pos = pdf.mul_add(-poly, 1.0);
    // Branch-free sign handling via bit extraction.
    let sign = (x.to_bits() >> 63) as f64;
    sign.mul_add(1.0 - 2.0 * cdf_pos, cdf_pos)
}

/// Batch normal CDF approximation with runtime SIMD dispatch and scalar fallback.
pub fn normal_cdf_batch_approx(xs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; xs.len()];
    normal_cdf_batch_approx_into(xs, &mut out);
    out
}

/// Allocation-free normal CDF batch API.
pub fn normal_cdf_batch_approx_into(xs: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), out.len(), "output length must match input");
    let backend = detected_batch_simd_backend();
    if !simd_is_worthwhile(xs.len(), backend) {
        for (dst, &x) in out.iter_mut().zip(xs.iter()) {
            *dst = normal_cdf_approx(x);
        }
        return;
    }

    match backend {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx512 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_avx512(xs, out) };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx2 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_avx2(xs, out) };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        BatchSimdBackend::Neon => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_neon(xs, out) };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
        BatchSimdBackend::WasmSimd128 => {
            crate::math::simd_wasm::normal_cdf_batch_into(xs, out);
            return;
        }
        _ => {}
    }

    for (dst, &x) in out.iter_mut().zip(xs.iter()) {
        *dst = normal_cdf_approx(x);
    }
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
    bs_price_batch_into(spots, strikes, r, q, vol, t, is_call, &mut out);
    out
}

/// Allocation-free batch pricing API for caller-owned workspaces.
#[allow(clippy::too_many_arguments)]
pub fn bs_price_batch_into(
    spots: &[f64],
    strikes: &[f64],
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    out: &mut [f64],
) {
    assert_eq!(spots.len(), strikes.len(), "spots and strikes must match");
    assert_eq!(spots.len(), out.len(), "output length must match input");
    let backend = selected_price_backend(spots, strikes, r, q, vol, t, is_call);
    if matches!(backend, BatchSimdBackend::Scalar) {
        for i in 0..spots.len() {
            out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        }
        return;
    }

    match backend {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx512 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { bs_price_batch_avx512(spots, strikes, r, q, vol, t, is_call, out) };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx2 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { bs_price_batch_avx2(spots, strikes, r, q, vol, t, is_call, out) };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        BatchSimdBackend::Neon => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                crate::math::simd_neon::bs_price_neon_batch_into(
                    spots, strikes, r, q, vol, t, is_call, out,
                )
            };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
        BatchSimdBackend::WasmSimd128 => {
            bs_price_batch_wasm_simd(spots, strikes, r, q, vol, t, is_call, out);
            return;
        }
        _ => {}
    }

    for i in 0..spots.len() {
        out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
    }
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
    bs_greeks_batch_into(
        spots, strikes, r, q, vol, t, is_call, &mut delta, &mut gamma, &mut vega, &mut theta,
    );
    (delta, gamma, vega, theta)
}

/// Allocation-free batch Greeks API for caller-owned workspaces.
#[allow(clippy::too_many_arguments)]
pub fn bs_greeks_batch_into(
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
    let n = spots.len();
    assert_eq!(n, strikes.len(), "spots and strikes must match");
    for output in [&*delta, &*gamma, &*vega, &*theta] {
        assert_eq!(n, output.len(), "output length must match input");
    }

    let backend = detected_batch_simd_backend();
    let sensitive_total_vol = t > 0.0 && vol > 0.0 && vol * t.sqrt() <= ACCURATE_CDF_TOTAL_VOL;
    if t <= 0.0
        || vol <= 0.0
        || sensitive_total_vol
        || !simd_is_worthwhile(n, backend)
        || batch_requires_scalar(spots, strikes, r, q, vol, t, is_call)
    {
        for i in 0..n {
            let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            delta[i] = d;
            gamma[i] = g;
            vega[i] = v;
            theta[i] = th;
        }
        return;
    }

    match backend {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx512 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_avx512(
                    spots, strikes, r, q, vol, t, is_call, delta, gamma, vega, theta,
                )
            };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        BatchSimdBackend::Avx2 => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_avx2(
                    spots, strikes, r, q, vol, t, is_call, delta, gamma, vega, theta,
                )
            };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        BatchSimdBackend::Neon => {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_neon(
                    spots, strikes, r, q, vol, t, is_call, delta, gamma, vega, theta,
                )
            };
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
        BatchSimdBackend::WasmSimd128 => {
            bs_greeks_batch_wasm_simd(
                spots, strikes, r, q, vol, t, is_call, delta, gamma, vega, theta,
            );
            return;
        }
        _ => {}
    }

    for i in 0..n {
        let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        delta[i] = d;
        gamma[i] = g;
        vega[i] = v;
        theta[i] = th;
    }
}

#[inline]
fn bs_price_scalar(spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool) -> f64 {
    if !spot.is_finite()
        || !strike.is_finite()
        || !r.is_finite()
        || !q.is_finite()
        || !vol.is_finite()
        || !t.is_finite()
        || spot < 0.0
        || strike < 0.0
        || vol < 0.0
        || (spot == 0.0 && strike == 0.0)
    {
        return f64::NAN;
    }
    if t <= 0.0 {
        return if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
    }

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    if spot == 0.0 {
        return if is_call { 0.0 } else { strike * df_r };
    }
    if strike == 0.0 {
        return if is_call { spot * df_q } else { 0.0 };
    }
    if vol == 0.0 {
        return if is_call {
            (spot * df_q - strike * df_r).max(0.0)
        } else {
            (strike * df_r - spot * df_q).max(0.0)
        };
    }
    if super::bs_inline::requires_stable_price(spot, strike, r, q, vol, t, is_call) {
        return super::bs_inline::stable_short_total_vol_price(spot, strike, r, q, vol, t, is_call);
    }
    let (d1, d2) = super::bs_inline::stable_d1_d2(spot, strike, r, q, vol, t);

    let s_df_q = spot * df_q;
    let k_df_r = strike * df_r;
    // Only 2 CDF evaluations; put uses put-call parity.
    let call = s_df_q * normal_cdf_approx(d1) - k_df_r * normal_cdf_approx(d2);
    let (call, put) = (call, call - s_df_q + k_df_r);
    if is_call { call } else { put }
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
    if !spot.is_finite()
        || !strike.is_finite()
        || !r.is_finite()
        || !q.is_finite()
        || !vol.is_finite()
        || !t.is_finite()
        || spot < 0.0
        || strike < 0.0
        || vol < 0.0
        || (spot == 0.0 && strike == 0.0)
    {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if t <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    if spot == 0.0 {
        return if is_call {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (-df_q, 0.0, 0.0, r * strike * df_r)
        };
    }
    if strike == 0.0 {
        return if is_call {
            (df_q, 0.0, 0.0, q * spot * df_q)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
    }
    if vol == 0.0 {
        let forward_value = spot * df_q - strike * df_r;
        return match forward_value.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) if is_call => {
                (df_q, 0.0, 0.0, q * spot * df_q - r * strike * df_r)
            }
            Some(std::cmp::Ordering::Less) if !is_call => {
                (-df_q, 0.0, 0.0, -q * spot * df_q + r * strike * df_r)
            }
            Some(std::cmp::Ordering::Equal) => {
                // The deterministic payoff has a kink at the forward strike;
                // no unique Black-Scholes Greek exists there.
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            }
            Some(_) => (0.0, 0.0, 0.0, 0.0),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let (d1, d2) = super::bs_inline::stable_d1_d2(spot, strike, r, q, vol, t);

    let pdf = normal_pdf_scalar(d1);

    // Only 2 CDF evaluations; derive N(-d) = 1 - N(d) for puts.
    let cdf = if sig_sqrt_t <= ACCURATE_CDF_TOTAL_VOL
        || super::bs_inline::requires_stable_price(spot, strike, r, q, vol, t, is_call)
    {
        accurate_norm_cdf
    } else {
        normal_cdf_approx
    };
    let nd1 = cdf(d1);
    let nd2 = cdf(d2);

    let delta = if is_call {
        df_q * nd1
    } else {
        -df_q * cdf(-d1)
    };
    let gamma = super::bs_inline::stable_gamma(df_q * pdf, spot, vol, sqrt_t, d1, -q * t);
    let vega = spot * df_q * pdf * sqrt_t;

    let theta_common = super::bs_inline::stable_theta_diffusion(spot * df_q * pdf, vol, sqrt_t);
    let theta = if is_call {
        theta_common + q * spot * df_q * nd1 - r * strike * df_r * nd2
    } else {
        theta_common - q * spot * df_q * cdf(-d1) + r * strike * df_r * cdf(-d2)
    };

    (delta, gamma, vega, theta)
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd_impl {
    use std::arch::wasm32::*;

    use crate::math::simd_wasm::{load_f64x2, normal_cdf_f64x2, store_f64x2};

    use super::{bs_greeks_scalar, bs_price_scalar};

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bs_price_batch_wasm_simd(
        spots: &[f64],
        strikes: &[f64],
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
        is_call: bool,
        out: &mut [f64],
    ) {
        // Preserve the scalar contract for degenerate and non-finite uniform
        // parameters. The hot path below is only meaningful for a regular
        // Black-Scholes diffusion.
        if t <= 0.0
            || vol <= 0.0
            || !r.is_finite()
            || !q.is_finite()
            || !vol.is_finite()
            || !t.is_finite()
        {
            for index in 0..spots.len() {
                out[index] = bs_price_scalar(spots[index], strikes[index], r, q, vol, t, is_call);
            }
            return;
        }

        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let inv_sig_sqrt_t = f64x2_splat(1.0 / sig_sqrt_t);
        let sig_sqrt_t_v = f64x2_splat(sig_sqrt_t);
        let drift = f64x2_splat((0.5 * vol).mul_add(vol, r - q) * t);
        let df_r = f64x2_splat((-r * t).exp());
        let df_q = f64x2_splat((-q * t).exp());

        let mut index = 0;
        while index + 2 <= spots.len() {
            let spot_0 = spots[index];
            let spot_1 = spots[index + 1];
            let strike_0 = strikes[index];
            let strike_1 = strikes[index + 1];
            if !(spot_0.is_finite()
                && spot_1.is_finite()
                && strike_0.is_finite()
                && strike_1.is_finite()
                && spot_0 > 0.0
                && spot_1 > 0.0
                && strike_0 > 0.0
                && strike_1 > 0.0)
            {
                out[index] = bs_price_scalar(spot_0, strike_0, r, q, vol, t, is_call);
                out[index + 1] = bs_price_scalar(spot_1, strike_1, r, q, vol, t, is_call);
                index += 2;
                continue;
            }

            let spots_v = load_f64x2(spots, index);
            let strikes_v = load_f64x2(strikes, index);

            // WebAssembly SIMD has no vector logarithm. Division and all
            // downstream d1/d2, CDF-polynomial, discount, and payoff
            // arithmetic remain explicit f64x2 operations.
            let ratios = f64x2_div(spots_v, strikes_v);
            let ln_ratio = f64x2(
                f64x2_extract_lane::<0>(ratios).ln(),
                f64x2_extract_lane::<1>(ratios).ln(),
            );
            let d1 = f64x2_mul(f64x2_add(ln_ratio, drift), inv_sig_sqrt_t);
            let d2 = f64x2_sub(d1, sig_sqrt_t_v);
            let nd1 = normal_cdf_f64x2(d1);
            let nd2 = normal_cdf_f64x2(d2);
            let discounted_spot = f64x2_mul(spots_v, df_q);
            let discounted_strike = f64x2_mul(strikes_v, df_r);
            let call = f64x2_sub(
                f64x2_mul(discounted_spot, nd1),
                f64x2_mul(discounted_strike, nd2),
            );
            let result = if is_call {
                call
            } else {
                f64x2_add(f64x2_sub(call, discounted_spot), discounted_strike)
            };
            store_f64x2(out, index, result);
            index += 2;
        }

        if index < spots.len() {
            out[index] = bs_price_scalar(spots[index], strikes[index], r, q, vol, t, is_call);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bs_greeks_batch_wasm_simd(
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
        if t <= 0.0
            || vol <= 0.0
            || !r.is_finite()
            || !q.is_finite()
            || !vol.is_finite()
            || !t.is_finite()
        {
            for index in 0..spots.len() {
                let values = bs_greeks_scalar(spots[index], strikes[index], r, q, vol, t, is_call);
                delta[index] = values.0;
                gamma[index] = values.1;
                vega[index] = values.2;
                theta[index] = values.3;
            }
            return;
        }

        const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
        let one = f64x2_splat(1.0);
        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let inv_sig_sqrt_t = f64x2_splat(1.0 / sig_sqrt_t);
        let sig_sqrt_t_v = f64x2_splat(sig_sqrt_t);
        let drift = f64x2_splat((0.5 * vol).mul_add(vol, r - q) * t);
        let df_r = f64x2_splat((-r * t).exp());
        let df_q = f64x2_splat((-q * t).exp());
        let sqrt_t_v = f64x2_splat(sqrt_t);
        let gamma_denom = f64x2_splat(vol * sqrt_t);
        let theta_scale = f64x2_splat(-0.5 * vol / sqrt_t);
        let q_v = f64x2_splat(q);
        let r_v = f64x2_splat(r);

        let mut index = 0;
        while index + 2 <= spots.len() {
            let spot_0 = spots[index];
            let spot_1 = spots[index + 1];
            let strike_0 = strikes[index];
            let strike_1 = strikes[index + 1];
            if !(spot_0.is_finite()
                && spot_1.is_finite()
                && strike_0.is_finite()
                && strike_1.is_finite()
                && spot_0 > 0.0
                && spot_1 > 0.0
                && strike_0 > 0.0
                && strike_1 > 0.0)
            {
                for lane in index..index + 2 {
                    let values =
                        bs_greeks_scalar(spots[lane], strikes[lane], r, q, vol, t, is_call);
                    delta[lane] = values.0;
                    gamma[lane] = values.1;
                    vega[lane] = values.2;
                    theta[lane] = values.3;
                }
                index += 2;
                continue;
            }

            let spots_v = load_f64x2(spots, index);
            let strikes_v = load_f64x2(strikes, index);
            let ratios = f64x2_div(spots_v, strikes_v);
            let ln_ratio = f64x2(
                f64x2_extract_lane::<0>(ratios).ln(),
                f64x2_extract_lane::<1>(ratios).ln(),
            );
            let d1 = f64x2_mul(f64x2_add(ln_ratio, drift), inv_sig_sqrt_t);
            let d2 = f64x2_sub(d1, sig_sqrt_t_v);
            let nd1 = normal_cdf_f64x2(d1);
            let nd2 = normal_cdf_f64x2(d2);

            // As with CDF, WebAssembly has no vector exponential. Only the two
            // exponentials are scalar; PDF scaling and all Greek formulas are
            // vector arithmetic.
            let pdf_exponent = f64x2_mul(f64x2_splat(-0.5), f64x2_mul(d1, d1));
            let pdf = f64x2(
                INV_SQRT_2PI * f64x2_extract_lane::<0>(pdf_exponent).exp(),
                INV_SQRT_2PI * f64x2_extract_lane::<1>(pdf_exponent).exp(),
            );

            let discounted_pdf = f64x2_mul(df_q, pdf);
            let delta_v = if is_call {
                f64x2_mul(df_q, nd1)
            } else {
                f64x2_mul(df_q, f64x2_sub(nd1, one))
            };
            let gamma_v = f64x2_div(discounted_pdf, f64x2_mul(spots_v, gamma_denom));
            let vega_v = f64x2_mul(f64x2_mul(spots_v, discounted_pdf), sqrt_t_v);
            let theta_common = f64x2_mul(f64x2_mul(spots_v, discounted_pdf), theta_scale);
            let theta_v = if is_call {
                f64x2_sub(
                    f64x2_add(
                        theta_common,
                        f64x2_mul(f64x2_mul(q_v, spots_v), f64x2_mul(df_q, nd1)),
                    ),
                    f64x2_mul(f64x2_mul(r_v, strikes_v), f64x2_mul(df_r, nd2)),
                )
            } else {
                f64x2_add(
                    f64x2_sub(
                        theta_common,
                        f64x2_mul(
                            f64x2_mul(q_v, spots_v),
                            f64x2_mul(df_q, f64x2_sub(one, nd1)),
                        ),
                    ),
                    f64x2_mul(
                        f64x2_mul(r_v, strikes_v),
                        f64x2_mul(df_r, f64x2_sub(one, nd2)),
                    ),
                )
            };

            store_f64x2(delta, index, delta_v);
            store_f64x2(gamma, index, gamma_v);
            store_f64x2(vega, index, vega_v);
            store_f64x2(theta, index, theta_v);
            index += 2;
        }

        if index < spots.len() {
            let values = bs_greeks_scalar(spots[index], strikes[index], r, q, vol, t, is_call);
            delta[index] = values.0;
            gamma[index] = values.1;
            vega[index] = values.2;
            theta[index] = values.3;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(unsafe_op_in_unsafe_fn)]
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
        let drift = (0.5 * vol).mul_add(vol, r - q) * t;
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
        let denom_gamma_v = unsafe { splat_f64x2(denom_gamma) };
        // Pre-compute theta time denominator to avoid per-iteration divide.
        let neg_inv_2sqrt_t = unsafe { splat_f64x2(-0.5 / sqrt_t) };

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

            // Only 2 CDF evaluations; derive N(-d) = 1 - N(d) for puts.
            let nd1 = unsafe { norm_cdf_f64x2(d1) };
            let nd2 = unsafe { norm_cdf_f64x2(d2) };
            let pdf_d1 = unsafe { norm_pdf_f64x2(d1) };

            let delta_call = vmulq_f64(df_q_v, nd1);
            let delta_put = vmulq_f64(df_q_v, vsubq_f64(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = vdivq_f64(vmulq_f64(df_q_v, pdf_d1), vmulq_f64(s, denom_gamma_v));
            let vega_v = vmulq_f64(vmulq_f64(vmulq_f64(s, df_q_v), pdf_d1), sqrt_t_v);

            // theta_common = S * df_q * pdf(d1) * vol * (-1/(2*sqrt_t))
            let theta_common = vmulq_f64(
                vmulq_f64(vmulq_f64(vmulq_f64(s, df_q_v), pdf_d1), vol_v),
                neg_inv_2sqrt_t,
            );

            let theta_v = if is_call {
                vsubq_f64(
                    vaddq_f64(
                        theta_common,
                        vmulq_f64(vmulq_f64(q_v, s), vmulq_f64(df_q_v, nd1)),
                    ),
                    vmulq_f64(vmulq_f64(r_v, k), vmulq_f64(df_r_v, nd2)),
                )
            } else {
                let nmd1 = vsubq_f64(one, nd1);
                let nmd2 = vsubq_f64(one, nd2);
                vaddq_f64(
                    vsubq_f64(
                        theta_common,
                        vmulq_f64(vmulq_f64(q_v, s), vmulq_f64(df_q_v, nmd1)),
                    ),
                    vmulq_f64(vmulq_f64(r_v, k), vmulq_f64(df_r_v, nmd2)),
                )
            };

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
        let drift = (0.5 * vol).mul_add(vol, r - q) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let drift_v = unsafe { splat_f64x4(drift) };
        let inv_sig_sqrt_t_v = unsafe { splat_f64x4(inv_sig_sqrt_t) };
        let sig_sqrt_t_v = unsafe { splat_f64x4(sig_sqrt_t) };
        let df_r_v = unsafe { splat_f64x4(df_r) };
        let df_q_v = unsafe { splat_f64x4(df_q) };

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

            // Only 2 CDF evaluations regardless of call/put.
            // Put uses put-call parity: P = C - S*df_q + K*df_r.
            let nd1 = unsafe { norm_cdf_f64x4(d1) };
            let nd2 = unsafe { norm_cdf_f64x4(d2) };

            let s_df_q = _mm256_mul_pd(s, df_q_v);
            let k_df_r = _mm256_mul_pd(k, df_r_v);

            let call = _mm256_sub_pd(_mm256_mul_pd(s_df_q, nd1), _mm256_mul_pd(k_df_r, nd2));

            let result = if is_call {
                call
            } else {
                // Put-call parity: P = C - S*df_q + K*df_r
                _mm256_add_pd(_mm256_sub_pd(call, s_df_q), k_df_r)
            };

            // SAFETY: bounds are checked in loop condition.
            unsafe { store_f64x4(out, i, result) };
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
        let drift = (0.5 * vol).mul_add(vol, r - q) * t;
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
        let denom_gamma_v = unsafe { splat_f64x4(denom_gamma) };
        // Pre-compute the theta time denominator: -1/(2*sqrt_t).
        // This replaces a negate + divide per iteration with a single multiply.
        let neg_inv_2sqrt_t = unsafe { splat_f64x4(-0.5 / sqrt_t) };

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

            // Only 2 CDF evaluations + derive negations via 1-CDF (no extra CDF calls).
            let nd1 = unsafe { norm_cdf_f64x4(d1) };
            let nd2 = unsafe { norm_cdf_f64x4(d2) };
            let pdf_d1 = unsafe { norm_pdf_f64x4(d1) };

            let delta_call = _mm256_mul_pd(df_q_v, nd1);
            let delta_put = _mm256_mul_pd(df_q_v, _mm256_sub_pd(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = _mm256_div_pd(
                _mm256_mul_pd(df_q_v, pdf_d1),
                _mm256_mul_pd(s, denom_gamma_v),
            );

            let vega_v = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), sqrt_t_v);

            // theta_common = -S * df_q * pdf(d1) * vol / (2*sqrt_t)
            //              = S * df_q * pdf(d1) * vol * (-1/(2*sqrt_t))
            let theta_common = _mm256_mul_pd(
                _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), vol_v),
                neg_inv_2sqrt_t,
            );

            let theta_v = if is_call {
                // theta_call = common + q*S*df_q*N(d1) - r*K*df_r*N(d2)
                _mm256_sub_pd(
                    _mm256_add_pd(
                        theta_common,
                        _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nd1)),
                    ),
                    _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nd2)),
                )
            } else {
                // For put, use N(-d) = 1 - N(d) without extra CDF calls.
                let nmd1 = _mm256_sub_pd(one, nd1);
                let nmd2 = _mm256_sub_pd(one, nd2);
                // theta_put = common - q*S*df_q*N(-d1) + r*K*df_r*N(-d2)
                _mm256_add_pd(
                    _mm256_sub_pd(
                        theta_common,
                        _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nmd1)),
                    ),
                    _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nmd2)),
                )
            };

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
#[allow(unsafe_op_in_unsafe_fn)]
mod avx512_impl {
    use std::arch::x86_64::*;

    use crate::math::simd_avx512::{
        ln_f64x8, load_f64x8, norm_cdf_f64x8, norm_pdf_f64x8, splat_f64x8, store_f64x8,
    };

    use super::{bs_greeks_scalar, bs_price_scalar};

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn normal_cdf_batch_avx512(xs: &[f64], out: &mut [f64]) {
        let n = xs.len();
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: bounds are checked in loop condition.
            let x = load_f64x8(xs, i);
            // SAFETY: target feature is enabled by this function attribute.
            let y = norm_cdf_f64x8(x);
            // SAFETY: bounds are checked in loop condition.
            store_f64x8(out, i, y);
            i += 8;
        }
        while i < n {
            out[i] = super::normal_cdf_approx(xs[i]);
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bs_price_batch_avx512(
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
        let drift = (0.5 * vol).mul_add(vol, r - q) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let drift_v = splat_f64x8(drift);
        let inv_sig_sqrt_t_v = splat_f64x8(inv_sig_sqrt_t);
        let sig_sqrt_t_v = splat_f64x8(sig_sqrt_t);
        let df_r_v = splat_f64x8(df_r);
        let df_q_v = splat_f64x8(df_q);

        let n = spots.len();
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = load_f64x8(spots, i);
            // SAFETY: bounds are checked in loop condition.
            let k = load_f64x8(strikes, i);
            let ln_sk = ln_f64x8(_mm512_div_pd(s, k));

            let d1 = _mm512_mul_pd(_mm512_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm512_sub_pd(d1, sig_sqrt_t_v);

            // Only 2 CDF evaluations regardless of call/put.
            // Put uses put-call parity: P = C - S*df_q + K*df_r.
            let nd1 = norm_cdf_f64x8(d1);
            let nd2 = norm_cdf_f64x8(d2);

            let s_df_q = _mm512_mul_pd(s, df_q_v);
            let k_df_r = _mm512_mul_pd(k, df_r_v);

            let call = _mm512_sub_pd(_mm512_mul_pd(s_df_q, nd1), _mm512_mul_pd(k_df_r, nd2));

            let result = if is_call {
                call
            } else {
                // Put-call parity: P = C - S*df_q + K*df_r
                _mm512_add_pd(_mm512_sub_pd(call, s_df_q), k_df_r)
            };

            // SAFETY: bounds are checked in loop condition.
            store_f64x8(out, i, result);
            i += 8;
        }

        while i < n {
            out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bs_greeks_batch_avx512(
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
        let drift = (0.5 * vol).mul_add(vol, r - q) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let denom_gamma = vol * sqrt_t;

        let drift_v = splat_f64x8(drift);
        let inv_sig_sqrt_t_v = splat_f64x8(inv_sig_sqrt_t);
        let sig_sqrt_t_v = splat_f64x8(sig_sqrt_t);
        let df_r_v = splat_f64x8(df_r);
        let df_q_v = splat_f64x8(df_q);
        let sqrt_t_v = splat_f64x8(sqrt_t);
        let vol_v = splat_f64x8(vol);
        let q_v = splat_f64x8(q);
        let r_v = splat_f64x8(r);
        let one = splat_f64x8(1.0);
        let denom_gamma_v = splat_f64x8(denom_gamma);
        // Pre-compute the theta time denominator: -1/(2*sqrt_t).
        // This replaces a negate + divide per iteration with a single multiply.
        let neg_inv_2sqrt_t = splat_f64x8(-0.5 / sqrt_t);

        let n = spots.len();
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = load_f64x8(spots, i);
            // SAFETY: bounds are checked in loop condition.
            let k = load_f64x8(strikes, i);
            let ln_sk = ln_f64x8(_mm512_div_pd(s, k));

            let d1 = _mm512_mul_pd(_mm512_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm512_sub_pd(d1, sig_sqrt_t_v);

            // Only 2 CDF evaluations + derive negations via 1-CDF (no extra CDF calls).
            let nd1 = norm_cdf_f64x8(d1);
            let nd2 = norm_cdf_f64x8(d2);
            let pdf_d1 = norm_pdf_f64x8(d1);

            let delta_call = _mm512_mul_pd(df_q_v, nd1);
            let delta_put = _mm512_mul_pd(df_q_v, _mm512_sub_pd(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = _mm512_div_pd(
                _mm512_mul_pd(df_q_v, pdf_d1),
                _mm512_mul_pd(s, denom_gamma_v),
            );

            let vega_v = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(s, df_q_v), pdf_d1), sqrt_t_v);

            // theta_common = -S * df_q * pdf(d1) * vol / (2*sqrt_t)
            //              = S * df_q * pdf(d1) * vol * (-1/(2*sqrt_t))
            let theta_common = _mm512_mul_pd(
                _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(s, df_q_v), pdf_d1), vol_v),
                neg_inv_2sqrt_t,
            );

            let theta_v = if is_call {
                // theta_call = common + q*S*df_q*N(d1) - r*K*df_r*N(d2)
                _mm512_sub_pd(
                    _mm512_add_pd(
                        theta_common,
                        _mm512_mul_pd(_mm512_mul_pd(q_v, s), _mm512_mul_pd(df_q_v, nd1)),
                    ),
                    _mm512_mul_pd(_mm512_mul_pd(r_v, k), _mm512_mul_pd(df_r_v, nd2)),
                )
            } else {
                // For put, use N(-d) = 1 - N(d) without extra CDF calls.
                let nmd1 = _mm512_sub_pd(one, nd1);
                let nmd2 = _mm512_sub_pd(one, nd2);
                // theta_put = common - q*S*df_q*N(-d1) + r*K*df_r*N(-d2)
                _mm512_add_pd(
                    _mm512_sub_pd(
                        theta_common,
                        _mm512_mul_pd(_mm512_mul_pd(q_v, s), _mm512_mul_pd(df_q_v, nmd1)),
                    ),
                    _mm512_mul_pd(_mm512_mul_pd(r_v, k), _mm512_mul_pd(df_r_v, nmd2)),
                )
            };

            // SAFETY: bounds are checked in loop condition.
            store_f64x8(delta, i, delta_v);
            store_f64x8(gamma, i, gamma_v);
            store_f64x8(vega, i, vega_v);
            store_f64x8(theta, i, theta_v);

            i += 8;
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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use avx512_impl::{bs_greeks_batch_avx512, bs_price_batch_avx512, normal_cdf_batch_avx512};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use neon_impl::{bs_greeks_batch_neon, normal_cdf_batch_neon};
#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
use wasm_simd_impl::{bs_greeks_batch_wasm_simd, bs_price_batch_wasm_simd};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_backend_selection_routes_sensitive_prices_to_scalar() {
        let tail_spots = [5.0e18; 16];
        let tail_strikes = [1.0e18; 16];
        let ordinary_spots = [100.0; 16];
        let ordinary_strikes = [105.0; 16];

        for backend in [
            BatchSimdBackend::Scalar,
            BatchSimdBackend::Avx2,
            BatchSimdBackend::Avx512,
            BatchSimdBackend::Neon,
            BatchSimdBackend::WasmSimd128,
        ] {
            assert_eq!(
                selected_price_backend_for(
                    backend,
                    &tail_spots,
                    &tail_strikes,
                    0.0,
                    0.0,
                    0.2,
                    1.0,
                    false,
                ),
                BatchSimdBackend::Scalar,
                "tail backend={backend:?}"
            );

            let ordinary = selected_price_backend_for(
                backend,
                &ordinary_spots,
                &ordinary_strikes,
                0.02,
                0.0,
                0.2,
                1.0,
                true,
            );
            let expected = backend;
            assert_eq!(ordinary, expected, "ordinary backend={backend:?}");
        }

        assert_eq!(
            selected_price_backend_for(
                BatchSimdBackend::Avx512,
                &ordinary_spots,
                &ordinary_strikes,
                0.0,
                0.0,
                1.0e155,
                1.0e-310,
                true,
            ),
            BatchSimdBackend::Scalar
        );
    }

    #[test]
    fn deep_tail_price_is_batch_length_invariant_above_every_backend_width() {
        const EXPECTED: f64 = 22.752_884_600_977_636;
        let mut reference_bits = None;
        for len in [1, 2, 4, 8, 15, 16, 17, 32] {
            let spots = vec![5.0e18; len];
            let strikes = vec![1.0e18; len];
            let prices = bs_price_batch(&spots, &strikes, 0.0, 0.0, 0.2, 1.0, false);
            for price in prices {
                assert!(price >= 0.0);
                assert!(
                    ((price - EXPECTED) / EXPECTED).abs() <= 2.0e-12,
                    "len={len} price={price:.17e}"
                );
                if let Some(bits) = reference_bits {
                    assert_eq!(price.to_bits(), bits, "len={len}");
                } else {
                    reference_bits = Some(price.to_bits());
                }
            }
        }
    }

    #[test]
    fn gamma_underflow_is_batch_length_invariant_and_finite() {
        for len in [1, 2, 4, 8, 16, 17] {
            let spots = vec![2.0; len];
            let strikes = vec![1.0; len];
            let (_, gamma, _, _) =
                bs_greeks_batch(&spots, &strikes, 0.0, 0.0, 1.0e-300, 1.0e-100, true);
            assert!(
                gamma.iter().all(|value| value.is_finite() && *value == 0.0),
                "len={len} gamma={gamma:?}"
            );
        }
    }
}
