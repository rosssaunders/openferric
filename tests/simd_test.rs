#[cfg(target_arch = "x86_64")]
mod simd_tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use statrs::distribution::{Continuous, ContinuousCDF, Normal};

    use openferric::core::OptionType;
    use openferric::engines::analytic::{bs_greeks_batch, bs_price_batch, normal_cdf_batch_approx};
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
        let n = Normal::new(0.0, 1.0).expect("normal distribution should be valid");
        let sqrt_t = t.sqrt();
        let sig_sqrt_t = vol * sqrt_t;
        let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / sig_sqrt_t;
        let d2 = d1 - sig_sqrt_t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let pdf = n.pdf(d1);

        let delta = if is_call {
            df_q * n.cdf(d1)
        } else {
            df_q * (n.cdf(d1) - 1.0)
        };
        let gamma = df_q * pdf / (s * vol * sqrt_t);
        let vega = s * df_q * pdf * sqrt_t;
        let theta = if is_call {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) + q * s * df_q * n.cdf(d1)
                - r * k * df_r * n.cdf(d2)
        } else {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) - q * s * df_q * n.cdf(-d1)
                + r * k * df_r * n.cdf(-d2)
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
        let normal = Normal::new(0.0, 1.0).expect("normal distribution should be valid");
        let n = 1201usize;
        let mut xs = Vec::with_capacity(n);
        for i in 0..n {
            xs.push(-6.0 + 12.0 * i as f64 / (n as f64 - 1.0));
        }
        let approx = normal_cdf_batch_approx(&xs);
        for i in 0..n {
            let reference = normal.cdf(xs[i]);
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
}
