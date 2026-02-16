//! SIMD-accelerated Black-Scholes batch routines with scalar fallback.

/// Scalar Abramowitz & Stegun 7.1.26 normal CDF approximation.
#[inline]
pub fn normal_cdf_approx(x: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A1: f64 = 0.319_381_530;
    const A2: f64 = -0.356_563_782;
    const A3: f64 = 1.781_477_937;
    const A4: f64 = -1.821_255_978;
    const A5: f64 = 1.330_274_429;
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    let z = x.abs();
    let t = 1.0 / (1.0 + P * z);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let pdf = INV_SQRT_2PI * (-0.5 * z * z).exp();
    let approx = 1.0 - pdf * poly;
    if x < 0.0 { 1.0 - approx } else { approx }
}

/// Batch normal CDF approximation with runtime SIMD dispatch and scalar fallback.
pub fn normal_cdf_batch_approx(xs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; xs.len()];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { normal_cdf_batch_avx2(xs, &mut out) };
            return out;
        }
    }

    for (dst, &x) in out.iter_mut().zip(xs.iter()) {
        *dst = normal_cdf_approx(x);
    }
    out
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

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe { bs_price_batch_avx2(spots, strikes, r, q, vol, t, is_call, &mut out) };
            return out;
        }
    }

    for i in 0..spots.len() {
        out[i] = bs_price_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
    }
    out
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

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                bs_greeks_batch_avx2(
                    spots, strikes, r, q, vol, t, is_call, &mut delta, &mut gamma, &mut vega,
                    &mut theta,
                )
            };
            return (delta, gamma, vega, theta);
        }
    }

    for i in 0..n {
        let (d, g, v, th) = bs_greeks_scalar(spots[i], strikes[i], r, q, vol, t, is_call);
        delta[i] = d;
        gamma[i] = g;
        vega[i] = v;
        theta[i] = th;
    }

    (delta, gamma, vega, theta)
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
        spot * df_q * normal_cdf_approx(d1) - strike * df_r * normal_cdf_approx(d2)
    } else {
        strike * df_r * normal_cdf_approx(-d2) - spot * df_q * normal_cdf_approx(-d1)
    }
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
    if t <= 0.0 || vol <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sqrt_t = t.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-r * t).exp();
    let df_q = (-q * t).exp();
    let pdf = normal_pdf_scalar(d1);

    let delta = if is_call {
        df_q * normal_cdf_approx(d1)
    } else {
        df_q * (normal_cdf_approx(d1) - 1.0)
    };
    let gamma = df_q * pdf / (spot * vol * sqrt_t);
    let vega = spot * df_q * pdf * sqrt_t;

    let theta = if is_call {
        -spot * df_q * pdf * vol / (2.0 * sqrt_t) + q * spot * df_q * normal_cdf_approx(d1)
            - r * strike * df_r * normal_cdf_approx(d2)
    } else {
        -spot * df_q * pdf * vol / (2.0 * sqrt_t) - q * spot * df_q * normal_cdf_approx(-d1)
            + r * strike * df_r * normal_cdf_approx(-d2)
    };

    (delta, gamma, vega, theta)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2_impl {
    use std::arch::x86_64::*;

    use super::{bs_greeks_scalar, bs_price_scalar};

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp256_pd(x: __m256d) -> __m256d {
        let max_x = _mm256_set1_pd(709.782_712_893_384);
        let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
        let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

        let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
        let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
        let half = _mm256_set1_pd(0.5);

        let n = _mm256_floor_pd(_mm256_fmadd_pd(x, log2e, half));
        let r = _mm256_fnmadd_pd(n, ln2, x);

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
    unsafe fn ln256_pd_scalar(v: __m256d) -> __m256d {
        let mut lanes = [0.0_f64; 4];
        // SAFETY: `lanes` has room for 4 f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
        for lane in &mut lanes {
            *lane = lane.ln();
        }
        // SAFETY: `lanes` contains 4 f64 values.
        unsafe { _mm256_loadu_pd(lanes.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn normal_pdf_pd(x: __m256d) -> __m256d {
        let inv_sqrt_2pi = _mm256_set1_pd(0.398_942_280_401_432_7);
        let half = _mm256_set1_pd(-0.5);
        let x2 = _mm256_mul_pd(x, x);
        let exponent = _mm256_mul_pd(half, x2);
        _mm256_mul_pd(inv_sqrt_2pi, unsafe { exp256_pd(exponent) })
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn normal_cdf_pd(x: __m256d) -> __m256d {
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

        let approx = _mm256_fnmadd_pd(unsafe { normal_pdf_pd(z) }, poly, one);
        let reflected = _mm256_sub_pd(one, approx);
        let neg_mask = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
        _mm256_blendv_pd(approx, reflected, neg_mask)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load4(values: &[f64], i: usize) -> __m256d {
        // SAFETY: caller guarantees there are at least 4 values starting at `i`.
        unsafe { _mm256_loadu_pd(values.as_ptr().add(i)) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store4(values: &mut [f64], i: usize, v: __m256d) {
        // SAFETY: caller guarantees there are at least 4 values starting at `i`.
        unsafe { _mm256_storeu_pd(values.as_mut_ptr().add(i), v) };
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn normal_cdf_batch_avx2(xs: &[f64], out: &mut [f64]) {
        let n = xs.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let x = unsafe { load4(xs, i) };
            // SAFETY: target feature is enabled by this function attribute.
            let y = unsafe { normal_cdf_pd(x) };
            // SAFETY: bounds are checked in loop condition.
            unsafe { store4(out, i, y) };
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
        let drift = (r - q + 0.5 * vol * vol) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let drift_v = _mm256_set1_pd(drift);
        let inv_sig_sqrt_t_v = _mm256_set1_pd(inv_sig_sqrt_t);
        let sig_sqrt_t_v = _mm256_set1_pd(sig_sqrt_t);
        let df_r_v = _mm256_set1_pd(df_r);
        let df_q_v = _mm256_set1_pd(df_q);
        let zero = _mm256_setzero_pd();

        let n = spots.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = unsafe { load4(spots, i) };
            // SAFETY: bounds are checked in loop condition.
            let k = unsafe { load4(strikes, i) };
            let ln_sk = unsafe { ln256_pd_scalar(_mm256_div_pd(s, k)) };

            let d1 = _mm256_mul_pd(_mm256_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm256_sub_pd(d1, sig_sqrt_t_v);

            let nd1 = unsafe { normal_cdf_pd(d1) };
            let nd2 = unsafe { normal_cdf_pd(d2) };

            let call = _mm256_sub_pd(
                _mm256_mul_pd(_mm256_mul_pd(s, df_q_v), nd1),
                _mm256_mul_pd(_mm256_mul_pd(k, df_r_v), nd2),
            );

            let put = _mm256_sub_pd(
                _mm256_mul_pd(_mm256_mul_pd(k, df_r_v), unsafe {
                    normal_cdf_pd(_mm256_sub_pd(zero, d2))
                }),
                _mm256_mul_pd(_mm256_mul_pd(s, df_q_v), unsafe {
                    normal_cdf_pd(_mm256_sub_pd(zero, d1))
                }),
            );

            // SAFETY: bounds are checked in loop condition.
            unsafe { store4(out, i, if is_call { call } else { put }) };
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
        let drift = (r - q + 0.5 * vol * vol) * t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let denom_gamma = vol * sqrt_t;

        let drift_v = _mm256_set1_pd(drift);
        let inv_sig_sqrt_t_v = _mm256_set1_pd(inv_sig_sqrt_t);
        let sig_sqrt_t_v = _mm256_set1_pd(sig_sqrt_t);
        let df_r_v = _mm256_set1_pd(df_r);
        let df_q_v = _mm256_set1_pd(df_q);
        let sqrt_t_v = _mm256_set1_pd(sqrt_t);
        let vol_v = _mm256_set1_pd(vol);
        let q_v = _mm256_set1_pd(q);
        let r_v = _mm256_set1_pd(r);
        let one = _mm256_set1_pd(1.0);
        let zero = _mm256_setzero_pd();
        let denom_gamma_v = _mm256_set1_pd(denom_gamma);

        let n = spots.len();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: bounds are checked in loop condition.
            let s = unsafe { load4(spots, i) };
            // SAFETY: bounds are checked in loop condition.
            let k = unsafe { load4(strikes, i) };
            let ln_sk = unsafe { ln256_pd_scalar(_mm256_div_pd(s, k)) };

            let d1 = _mm256_mul_pd(_mm256_add_pd(ln_sk, drift_v), inv_sig_sqrt_t_v);
            let d2 = _mm256_sub_pd(d1, sig_sqrt_t_v);

            let nd1 = unsafe { normal_cdf_pd(d1) };
            let nd2 = unsafe { normal_cdf_pd(d2) };
            let nmd1 = unsafe { normal_cdf_pd(_mm256_sub_pd(zero, d1)) };
            let nmd2 = unsafe { normal_cdf_pd(_mm256_sub_pd(zero, d2)) };
            let pdf_d1 = unsafe { normal_pdf_pd(d1) };

            let delta_call = _mm256_mul_pd(df_q_v, nd1);
            let delta_put = _mm256_mul_pd(df_q_v, _mm256_sub_pd(nd1, one));
            let delta_v = if is_call { delta_call } else { delta_put };

            let gamma_v = _mm256_div_pd(
                _mm256_mul_pd(df_q_v, pdf_d1),
                _mm256_mul_pd(s, denom_gamma_v),
            );

            let vega_v = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), sqrt_t_v);

            let theta_common =
                _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(s, df_q_v), pdf_d1), vol_v);
            let theta_common = _mm256_div_pd(
                _mm256_sub_pd(zero, theta_common),
                _mm256_mul_pd(_mm256_set1_pd(2.0), sqrt_t_v),
            );

            let theta_call = _mm256_sub_pd(
                _mm256_add_pd(
                    theta_common,
                    _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nd1)),
                ),
                _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nd2)),
            );

            let theta_put = _mm256_add_pd(
                _mm256_sub_pd(
                    theta_common,
                    _mm256_mul_pd(_mm256_mul_pd(q_v, s), _mm256_mul_pd(df_q_v, nmd1)),
                ),
                _mm256_mul_pd(_mm256_mul_pd(r_v, k), _mm256_mul_pd(df_r_v, nmd2)),
            );
            let theta_v = if is_call { theta_call } else { theta_put };

            // SAFETY: bounds are checked in loop condition.
            unsafe {
                store4(delta, i, delta_v);
                store4(gamma, i, gamma_v);
                store4(vega, i, vega_v);
                store4(theta, i, theta_v);
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
use avx2_impl::{bs_greeks_batch_avx2, bs_price_batch_avx2, normal_cdf_batch_avx2};
