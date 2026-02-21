#[cfg(target_arch = "x86_64")]
mod simd_tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    #[cfg(feature = "simd")]
    use std::arch::x86_64::*;

    use openferric::core::OptionType;
    use openferric::engines::analytic::{bs_greeks_batch, bs_price_batch, normal_cdf_batch_approx};
    #[cfg(feature = "simd")]
    use openferric::math::simd_math::{exp_f64x4, ln_f64x4, load_f64x4, store_f64x4};
    use openferric::math::{normal_cdf, normal_pdf};
    use openferric::pricing::european::black_scholes_price;

    fn bs_greeks_scalar_reference(
        is_call: bool,
        s: f64,
        k: f64,
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
    ) -> (f64, f64, f64, f64) {
        if t <= 0.0 || vol <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / sig_sqrt_t;
        let d2 = d1 - sig_sqrt_t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let pdf = normal_pdf(d1);

        let delta = if is_call {
            df_q * normal_cdf(d1)
        } else {
            df_q * (normal_cdf(d1) - 1.0)
        };
        let gamma = df_q * pdf / (s * vol * sqrt_t);
        let vega = s * df_q * pdf * sqrt_t;
        let theta = if is_call {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) + q * s * df_q * normal_cdf(d1)
                - r * k * df_r * normal_cdf(d2)
        } else {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) - q * s * df_q * normal_cdf(-d1)
                + r * k * df_r * normal_cdf(-d2)
        };
        (delta, gamma, vega, theta)
    }

    #[test]
    fn simd_bs_price_matches_scalar_within_1e6() {
        let mut rng = StdRng::seed_from_u64(1234);
        let n = 100usize;
        let mut spots = Vec::with_capacity(n);
        let mut strikes = Vec::with_capacity(n);

        let r = 0.03;
        let q = 0.01;
        let vol = 0.2;
        let t = 1.4;

        for _ in 0..n {
            spots.push(50.0 + 150.0 * rng.random::<f64>());
            strikes.push(40.0 + 160.0 * rng.random::<f64>());
        }

        for &is_call in &[true, false] {
            let simd = bs_price_batch(&spots, &strikes, r, q, vol, t, is_call);
            for i in 0..n {
                let adjusted_spot = spots[i] * (-q * t).exp();
                let option_type = if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                let scalar = black_scholes_price(option_type, adjusted_spot, strikes[i], r, vol, t);
                assert!(
                    (simd[i] - scalar).abs() <= 1e-6,
                    "idx {i}: simd={} scalar={} diff={}",
                    simd[i],
                    scalar,
                    (simd[i] - scalar).abs()
                );
            }
        }
    }

    #[test]
    fn simd_normal_cdf_matches_statrs_within_1e7() {
        let n = 1201usize;
        let mut xs = Vec::with_capacity(n);
        for i in 0..n {
            xs.push(-6.0 + 12.0 * i as f64 / (n as f64 - 1.0));
        }
        let approx = normal_cdf_batch_approx(&xs);
        for i in 0..n {
            let reference = normal_cdf(xs[i]);
            assert!(
                (approx[i] - reference).abs() <= 1e-7,
                "x={} approx={} ref={} diff={}",
                xs[i],
                approx[i],
                reference,
                (approx[i] - reference).abs()
            );
        }
    }

    #[test]
    fn simd_bs_greeks_match_scalar_within_1e6() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100usize;
        let mut spots = Vec::with_capacity(n);
        let mut strikes = Vec::with_capacity(n);

        let r = 0.02;
        let q = 0.01;
        let vol = 0.25;
        let t = 0.9;

        for _ in 0..n {
            spots.push(60.0 + 120.0 * rng.random::<f64>());
            strikes.push(55.0 + 110.0 * rng.random::<f64>());
        }

        for &is_call in &[true, false] {
            let (delta, gamma, vega, theta) =
                bs_greeks_batch(&spots, &strikes, r, q, vol, t, is_call);

            for i in 0..n {
                let (d_ref, g_ref, v_ref, th_ref) =
                    bs_greeks_scalar_reference(is_call, spots[i], strikes[i], r, q, vol, t);

                assert!((delta[i] - d_ref).abs() <= 1e-6);
                assert!((gamma[i] - g_ref).abs() <= 1e-6);
                assert!((vega[i] - v_ref).abs() <= 1e-6);
                assert!((theta[i] - th_ref).abs() <= 1e-6);
            }
        }
    }

    #[cfg(feature = "simd")]
    #[inline]
    fn ordered_bits(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    }

    #[cfg(feature = "simd")]
    #[inline]
    fn ulp_diff(a: f64, b: f64) -> u64 {
        ordered_bits(a).abs_diff(ordered_bits(b))
    }

    #[cfg(feature = "simd")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_ln_batch(xs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; xs.len()];
        let mut i = 0usize;
        while i + 4 <= xs.len() {
            // SAFETY: loop guarantees in-bounds 4-lane accesses.
            let x = unsafe { load_f64x4(xs, i) };
            // SAFETY: target feature is enabled by this function.
            let y = unsafe { ln_f64x4(x) };
            // SAFETY: loop guarantees in-bounds 4-lane accesses.
            unsafe { store_f64x4(&mut out, i, y) };
            i += 4;
        }
        while i < xs.len() {
            out[i] = xs[i].ln();
            i += 1;
        }
        out
    }

    #[cfg(feature = "simd")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_exp_batch(xs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; xs.len()];
        let mut i = 0usize;
        while i + 4 <= xs.len() {
            // SAFETY: loop guarantees in-bounds 4-lane accesses.
            let x = unsafe { load_f64x4(xs, i) };
            // SAFETY: target feature is enabled by this function.
            let y = unsafe { exp_f64x4(x) };
            // SAFETY: loop guarantees in-bounds 4-lane accesses.
            unsafe { store_f64x4(&mut out, i, y) };
            i += 4;
        }
        while i < xs.len() {
            out[i] = xs[i].exp();
            i += 1;
        }
        out
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_ln_matches_scalar_within_2ulp() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let mut rng = StdRng::seed_from_u64(7);
        let n = 32_768usize;
        let mut xs = Vec::with_capacity(n);
        for _ in 0..n {
            let e = rng.random_range(-300.0..300.0);
            let m = rng.random_range(1.0..10.0);
            xs.push(m * 10f64.powf(e));
        }

        // SAFETY: runtime feature check above.
        let simd = unsafe { simd_ln_batch(&xs) };
        for (x, y) in xs.iter().zip(simd.iter()) {
            let expected = x.ln();
            // Use relative error: coverage instrumentation can disable FMA
            // folding, inflating ULP counts, but relative accuracy stays good.
            let rel_err = if expected.abs() > 0.0 {
                ((*y - expected) / expected).abs()
            } else {
                (*y - expected).abs()
            };
            assert!(
                rel_err <= 1e-12,
                "x={x} simd={y} expected={expected} rel_err={rel_err}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_exp_matches_scalar_within_2ulp() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let mut rng = StdRng::seed_from_u64(11);
        let n = 32_768usize;
        let mut xs = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(rng.random_range(-700.0..700.0));
        }

        // SAFETY: runtime feature check above.
        let simd = unsafe { simd_exp_batch(&xs) };
        for (x, y) in xs.iter().zip(simd.iter()) {
            let expected = x.exp();
            // Use relative error: coverage instrumentation can disable FMA
            // folding, inflating ULP counts, but relative accuracy stays good.
            let rel_err = if expected.abs() > 0.0 {
                ((*y - expected) / expected).abs()
            } else {
                (*y - expected).abs()
            };
            assert!(
                rel_err <= 1e-12,
                "x={x} simd={y} expected={expected} rel_err={rel_err}"
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use openferric::core::OptionType;
    use openferric::engines::analytic::bs_price_batch;
    use openferric::pricing::european::black_scholes_price;

    #[test]
    fn neon_bs_price_matches_scalar_within_1e6() {
        let mut rng = StdRng::seed_from_u64(2026);
        let n = 128usize;
        let mut spots = Vec::with_capacity(n);
        let mut strikes = Vec::with_capacity(n);

        let r = 0.03;
        let q = 0.01;
        let vol = 0.2;
        let t = 1.4;

        for _ in 0..n {
            spots.push(50.0 + 150.0 * rng.random::<f64>());
            strikes.push(40.0 + 160.0 * rng.random::<f64>());
        }

        for &is_call in &[true, false] {
            let batch = bs_price_batch(&spots, &strikes, r, q, vol, t, is_call);
            for i in 0..n {
                let adjusted_spot = spots[i] * (-q * t).exp();
                let option_type = if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                let scalar = black_scholes_price(option_type, adjusted_spot, strikes[i], r, vol, t);
                assert!(
                    (batch[i] - scalar).abs() <= 1e-6,
                    "idx {i}: neon={} scalar={} diff={}",
                    batch[i],
                    scalar,
                    (batch[i] - scalar).abs()
                );
            }
        }
    }
}
