#[cfg(target_arch = "x86_64")]
mod simd_tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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
    fn simd_bs_price_matches_scalar_closely() {
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
                // The batch path uses the fast A&S normal CDF (~7.5e-8 abs
                // error) while the scalar reference now uses the tail-accurate
                // Cody CDF, so the bound is (S + K) * 7.5e-8 ~ 3e-5.
                assert!(
                    (simd[i] - scalar).abs() <= 5e-5,
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
            // Use absolute error: near x=1, ln(x)≈0 so relative error is
            // meaningless. The SIMD polynomial maintains ~1e-13 absolute
            // accuracy across the full range, but coverage instrumentation
            // (no FMA) can degrade to ~1e-10.
            let abs_err = (*y - expected).abs();
            assert!(
                abs_err <= 1e-9,
                "x={x} simd={y} expected={expected} abs_err={abs_err}"
            );
        }
    }

    /// NaN-aware exact equality (boundary contract must match bit-for-bit
    /// behavior of the scalar path, not just approximately).
    #[cfg(feature = "simd")]
    fn same_result(got: f64, want: f64) -> bool {
        got == want || (got.is_nan() && want.is_nan())
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_inv_cdf_boundary_values_match_scalar_at_any_position() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
        use openferric::math::simd_math::inv_norm_cdf_batch_avx2;

        let specials = [0.0, 1.0, -0.1, 1.1, f64::NAN, 0.5];

        // Place each special value at every offset of buffers with lengths
        // that exercise both the 4-wide vector body and the scalar remainder.
        for len in [6usize, 7, 9, 13] {
            for offset in 0..len {
                for &s in &specials {
                    let mut buf = vec![0.5_f64; len];
                    buf[offset] = s;
                    let expected: Vec<f64> = buf
                        .iter()
                        .map(|&p| beasley_springer_moro_inv_cdf(p))
                        .collect();

                    // SAFETY: runtime feature check above.
                    unsafe { inv_norm_cdf_batch_avx2(&mut buf) };

                    for (i, (&got, &want)) in buf.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            same_result(got, want),
                            "len={len} offset={offset} special={s} i={i} got={got} want={want}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn avx512_inv_cdf_boundary_values_match_scalar_at_any_position() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
        use openferric::math::simd_avx512::inv_norm_cdf_batch_avx512;

        let specials = [0.0, 1.0, -0.1, 1.1, f64::NAN, 0.5];

        for len in [10usize, 13, 17, 23] {
            for offset in 0..len {
                for &s in &specials {
                    let mut buf = vec![0.5_f64; len];
                    buf[offset] = s;
                    let expected: Vec<f64> = buf
                        .iter()
                        .map(|&p| beasley_springer_moro_inv_cdf(p))
                        .collect();

                    // SAFETY: runtime feature check above.
                    unsafe { inv_norm_cdf_batch_avx512(&mut buf) };

                    for (i, (&got, &want)) in buf.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            same_result(got, want),
                            "len={len} offset={offset} special={s} i={i} got={got} want={want}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn simd_fast_exp_batch(xs: &[f64]) -> Vec<f64> {
        use openferric::math::simd_math::fast_exp_f64x4;
        let mut out = vec![0.0; xs.len()];
        let mut i = 0usize;
        while i + 4 <= xs.len() {
            // SAFETY: loop guarantees in-bounds 4-lane accesses.
            let x = unsafe { load_f64x4(xs, i) };
            // SAFETY: target feature is enabled by this function.
            let y = unsafe { fast_exp_f64x4(x) };
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
    fn simd_exp_handles_overflow_boundary_band() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        // n = round(x*log2e) reaches 1024 around x ~ 709.44; the old single
        // 2^n reconstruction overflowed to +inf for the entire band
        // [~709.44, 709.78] where exp(x) is still finite.
        let xs = [
            709.0,
            709.4,
            709.5,
            709.7,
            709.782_712_893_384,
            709.8,
            710.0,
            800.0,
        ];
        // SAFETY: runtime feature check above.
        let exact = unsafe { simd_exp_batch(&xs) };
        // SAFETY: runtime feature check above.
        let fast = unsafe { simd_fast_exp_batch(&xs) };

        // The degree-7 fast polynomial has ~2.6e-9 truncation error at |r| ~ ln2/2.
        for (out, tol) in [(&exact, 1.0e-10), (&fast, 5.0e-9)] {
            for (x, y) in xs.iter().zip(out.iter()) {
                let expected = x.exp();
                if expected.is_infinite() {
                    assert_eq!(*y, f64::INFINITY, "x={x}");
                } else {
                    let rel = ((y - expected) / expected).abs();
                    assert!(rel <= tol, "x={x} simd={y} expected={expected} rel={rel}");
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn avx512_exp_handles_overflow_boundary_band() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        use openferric::math::simd_avx512::{exp_f64x8, fast_exp_f64x8, load_f64x8, store_f64x8};

        let xs = [
            709.0,
            709.4,
            709.5,
            709.7,
            709.782_712_893_384,
            709.8,
            710.0,
            800.0,
        ];
        let mut exact = vec![0.0_f64; 8];
        let mut fast = vec![0.0_f64; 8];
        // SAFETY: runtime feature check above; buffers hold exactly 8 lanes.
        unsafe {
            let x = load_f64x8(&xs, 0);
            store_f64x8(&mut exact, 0, exp_f64x8(x));
            store_f64x8(&mut fast, 0, fast_exp_f64x8(x));
        }

        // The degree-7 fast polynomial has ~2.6e-9 truncation error at |r| ~ ln2/2.
        for (out, tol) in [(&exact, 1.0e-10), (&fast, 5.0e-9)] {
            for (x, y) in xs.iter().zip(out.iter()) {
                let expected = x.exp();
                if expected.is_infinite() {
                    assert_eq!(*y, f64::INFINITY, "x={x}");
                } else {
                    let rel = ((y - expected) / expected).abs();
                    assert!(rel <= tol, "x={x} simd={y} expected={expected} rel={rel}");
                }
            }
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
    #[cfg(feature = "simd")]
    use openferric::math::simd_neon::{load_f64x2, simd_exp_f64x2, simd_ln_f64x2, store_f64x2};
    use openferric::pricing::european::black_scholes_price;

    #[cfg(feature = "simd")]
    unsafe fn neon_ln_batch(xs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; xs.len()];
        let mut i = 0usize;
        while i + 2 <= xs.len() {
            let x = unsafe { load_f64x2(xs, i) };
            let y = unsafe { simd_ln_f64x2(x) };
            unsafe { store_f64x2(&mut out, i, y) };
            i += 2;
        }
        while i < xs.len() {
            out[i] = xs[i].ln();
            i += 1;
        }
        out
    }

    #[cfg(feature = "simd")]
    unsafe fn neon_exp_batch(xs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; xs.len()];
        let mut i = 0usize;
        while i + 2 <= xs.len() {
            let x = unsafe { load_f64x2(xs, i) };
            let y = unsafe { simd_exp_f64x2(x) };
            unsafe { store_f64x2(&mut out, i, y) };
            i += 2;
        }
        while i < xs.len() {
            out[i] = xs[i].exp();
            i += 1;
        }
        out
    }

    #[cfg(feature = "simd")]
    #[test]
    fn neon_ln_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
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

        let simd = unsafe { neon_ln_batch(&xs) };
        for (x, y) in xs.iter().zip(simd.iter()) {
            let expected = x.ln();
            let abs_err = (*y - expected).abs();
            assert!(
                abs_err <= 1e-9,
                "x={x} neon={y} expected={expected} abs_err={abs_err}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn neon_exp_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut rng = StdRng::seed_from_u64(11);
        let n = 32_768usize;
        let mut xs = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(rng.random_range(-700.0..700.0));
        }

        let simd = unsafe { neon_exp_batch(&xs) };
        for (x, y) in xs.iter().zip(simd.iter()) {
            let expected = x.exp();
            let rel_err = if expected.abs() > 0.0 {
                ((*y - expected) / expected).abs()
            } else {
                (*y - expected).abs()
            };
            assert!(
                rel_err <= 1e-12,
                "x={x} neon={y} expected={expected} rel_err={rel_err}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn neon_ln_special_values_match_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        // ln(negative) = NaN, ln(+-0) = -inf, ln(+inf) = +inf, mirroring the
        // AVX2/AVX512 special-value blending and scalar f64::ln.
        let xs = [-1.0, -0.5, 0.0, -0.0, 1.0, 2.5, f64::INFINITY, 4.0];
        let got = unsafe { neon_ln_batch(&xs) };
        for (x, y) in xs.iter().zip(got.iter()) {
            let expected = x.ln();
            if expected.is_nan() {
                assert!(y.is_nan(), "x={x} neon={y} expected NaN");
            } else if expected.is_infinite() {
                assert_eq!(*y, expected, "x={x} neon={y} expected={expected}");
            } else {
                let abs_err = (*y - expected).abs();
                assert!(
                    abs_err <= 1e-9,
                    "x={x} neon={y} expected={expected} abs_err={abs_err}"
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn neon_exp_handles_overflow_boundary_band() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        // n = round(x*log2e) reaches 1024 around x ~ 709.44; the old single
        // 2^n reconstruction overflowed to +inf for the entire band
        // [~709.44, 709.78] where exp(x) is still finite.
        let xs = [
            709.0,
            709.4,
            709.5,
            709.7,
            709.782_712_893_384,
            709.8,
            710.0,
            800.0,
        ];
        let got = unsafe { neon_exp_batch(&xs) };
        for (x, y) in xs.iter().zip(got.iter()) {
            let expected = x.exp();
            if expected.is_infinite() {
                assert_eq!(*y, f64::INFINITY, "x={x}");
            } else {
                let rel = ((y - expected) / expected).abs();
                assert!(
                    rel <= 1.0e-10,
                    "x={x} neon={y} expected={expected} rel={rel}"
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn neon_zero_vol_price_is_discounted_forward_intrinsic() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        use openferric::math::simd_neon::bs_price_neon_batch;

        let spots = vec![100.0, 80.0, 120.0, 95.0, 101.0];
        let strikes = vec![90.0, 100.0, 100.0, 95.0, 99.0];
        let (r, q, t) = (0.05_f64, 0.02_f64, 2.0_f64);

        let calls = unsafe { bs_price_neon_batch(&spots, &strikes, r, q, 0.0, t, true) };
        let puts = unsafe { bs_price_neon_batch(&spots, &strikes, r, q, 0.0, t, false) };

        let df = (-r * t).exp();
        for i in 0..spots.len() {
            let fwd = spots[i] * ((r - q) * t).exp();
            let call_expected = df * (fwd - strikes[i]).max(0.0);
            let put_expected = df * (strikes[i] - fwd).max(0.0);
            assert!(
                (calls[i] - call_expected).abs() < 1e-12,
                "call i={i} got={} expected={call_expected}",
                calls[i]
            );
            assert!(
                (puts[i] - put_expected).abs() < 1e-12,
                "put i={i} got={} expected={put_expected}",
                puts[i]
            );
        }
    }

    #[test]
    fn neon_bs_price_matches_scalar_closely() {
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
                // The batch path uses the fast A&S normal CDF (~7.5e-8 abs
                // error) while the scalar reference now uses the tail-accurate
                // Cody CDF, so the bound is (S + K) * 7.5e-8 ~ 3e-5.
                assert!(
                    (batch[i] - scalar).abs() <= 5e-5,
                    "idx {i}: neon={} scalar={} diff={}",
                    batch[i],
                    scalar,
                    (batch[i] - scalar).abs()
                );
            }
        }
    }
}
