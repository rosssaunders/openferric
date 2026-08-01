mod batch_workspace_tests {
    use openferric::core::OptionType;
    use openferric::engines::analytic::{
        bs_greeks_batch, bs_greeks_batch_into, bs_price_asm, bs_price_batch, bs_price_batch_into,
        normal_cdf_approx, normal_cdf_batch_approx, normal_cdf_batch_approx_into,
    };
    use openferric::greeks::black_scholes_merton_greeks;
    use openferric::math::{black_scholes_price_greeks_aad, normal_cdf, normal_pdf};
    use openferric::pricing::european::black_scholes_price;

    const GUARD: f64 = -9_876_543.25;

    #[track_caller]
    fn assert_machine_precision_eq(actual: f64, expected: f64, operation_scale: f64) {
        let scale = actual
            .abs()
            .max(expected.abs())
            .max(operation_scale.abs())
            .max(1.0);
        let tolerance = 4.0 * f64::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differs from expected {expected} by more than {tolerance}"
        );
    }

    #[track_caller]
    fn assert_vector_roundoff_close(label: &str, actual: f64, expected: f64, operation_scale: f64) {
        // The SIMD log/PDF polynomials are independently bounded at roughly
        // 2e-14 over the pricing domain.  Convert that and ordinary expression
        // grouping into an explicit binary64 operation budget; this is not an
        // economic price/Greek tolerance.
        let scale = actual
            .abs()
            .max(expected.abs())
            .max(operation_scale.abs())
            .max(1.0);
        let tolerance = 1_024.0 * f64::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17e}, reference={expected:.17e}, \
             error={:.3e}, vector-roundoff budget={tolerance:.3e}",
            (actual - expected).abs()
        );
    }

    fn bs_greeks_reference(
        is_call: bool,
        spot: f64,
        strike: f64,
        r: f64,
        q: f64,
        vol: f64,
        t: f64,
    ) -> (f64, f64, f64, f64) {
        let sqrt_t = t.sqrt();
        let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();
        let pdf = normal_pdf(d1);
        let nd1 = normal_cdf(d1);
        let nd2 = normal_cdf(d2);
        let nmd1 = normal_cdf(-d1);
        let nmd2 = normal_cdf(-d2);
        let delta = if is_call { df_q * nd1 } else { -df_q * nmd1 };
        let gamma = df_q * pdf / (spot * vol * sqrt_t);
        let vega = spot * df_q * pdf * sqrt_t;
        let theta_common = -spot * df_q * pdf * vol / (2.0 * sqrt_t);
        let theta = if is_call {
            theta_common + q * spot * df_q * nd1 - r * strike * df_r * nd2
        } else {
            theta_common - q * spot * df_q * nmd1 + r * strike * df_r * nmd2
        };
        (delta, gamma, vega, theta)
    }

    #[test]
    fn batch_workspace_apis_cover_all_vector_tails_and_unaligned_slices() {
        // 0..=17 covers empty input and more than two full AVX-512 vectors,
        // four AVX2 vectors, or eight NEON vectors, including every tail.
        for len in 0..=17 {
            let spot_storage: Vec<f64> = (0..len + 2).map(|i| 70.0 + i as f64 * 3.25).collect();
            let strike_storage: Vec<f64> = (0..len + 2).map(|i| 80.0 + i as f64 * 2.50).collect();
            let spots = &spot_storage[1..1 + len];
            let strikes = &strike_storage[1..1 + len];

            for is_call in [false, true] {
                let mut guarded = vec![GUARD; len + 2];
                bs_price_batch_into(
                    spots,
                    strikes,
                    0.03,
                    0.01,
                    0.24,
                    1.3,
                    is_call,
                    &mut guarded[1..1 + len],
                );

                assert_eq!(guarded[0], GUARD);
                assert_eq!(guarded[len + 1], GUARD);
                for index in 0..len {
                    let adjusted_spot = spots[index] * (-0.01_f64 * 1.3).exp();
                    let option_type = if is_call {
                        OptionType::Call
                    } else {
                        OptionType::Put
                    };
                    let scalar = black_scholes_price(
                        option_type,
                        adjusted_spot,
                        strikes[index],
                        0.03,
                        0.24,
                        1.3,
                    );
                    assert_vector_roundoff_close(
                        &format!("price length={len} index={index}"),
                        guarded[index + 1],
                        scalar,
                        adjusted_spot + strikes[index] * (-0.03_f64 * 1.3).exp(),
                    );
                }

                let mut delta = vec![GUARD; len + 2];
                let mut gamma = vec![GUARD; len + 2];
                let mut vega = vec![GUARD; len + 2];
                let mut theta = vec![GUARD; len + 2];
                bs_greeks_batch_into(
                    spots,
                    strikes,
                    0.03,
                    0.01,
                    0.24,
                    1.3,
                    is_call,
                    &mut delta[1..1 + len],
                    &mut gamma[1..1 + len],
                    &mut vega[1..1 + len],
                    &mut theta[1..1 + len],
                );

                for output in [&delta, &gamma, &vega, &theta] {
                    assert_eq!(output[0], GUARD);
                    assert_eq!(output[len + 1], GUARD);
                }
                for index in 0..len {
                    let reference = bs_greeks_reference(
                        is_call,
                        spots[index],
                        strikes[index],
                        0.03,
                        0.01,
                        0.24,
                        1.3,
                    );
                    assert_vector_roundoff_close(
                        &format!("delta length={len} index={index}"),
                        delta[index + 1],
                        reference.0,
                        1.0,
                    );
                    assert_vector_roundoff_close(
                        &format!("gamma length={len} index={index}"),
                        gamma[index + 1],
                        reference.1,
                        1.0,
                    );
                    assert_vector_roundoff_close(
                        &format!("vega length={len} index={index}"),
                        vega[index + 1],
                        reference.2,
                        spots[index] * 1.3_f64.sqrt(),
                    );
                    assert_vector_roundoff_close(
                        &format!("theta length={len} index={index}"),
                        theta[index + 1],
                        reference.3,
                        spots[index] + strikes[index],
                    );
                }
            }
        }
    }

    #[test]
    fn normal_cdf_workspace_covers_tails_and_special_values() {
        let values = [
            f64::NEG_INFINITY,
            -8.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            8.0,
            f64::INFINITY,
            f64::NAN,
        ];

        for len in 0..=values.len() {
            let expected: Vec<f64> = values[..len]
                .iter()
                .copied()
                .map(normal_cdf_approx)
                .collect();
            let mut guarded = vec![GUARD; len + 2];
            normal_cdf_batch_approx_into(&values[..len], &mut guarded[1..1 + len]);
            assert_eq!(guarded[0], GUARD);
            assert_eq!(guarded[len + 1], GUARD);
            for index in 0..len {
                if expected[index].is_nan() {
                    assert!(guarded[index + 1].is_nan());
                } else {
                    assert_eq!(guarded[index + 1].to_bits(), expected[index].to_bits());
                }
            }
        }

        let all = normal_cdf_batch_approx(&values);
        assert_eq!(all[0], 0.0);
        assert_eq!(all[3].to_bits(), 0.5_f64.to_bits());
        assert_eq!(all[4].to_bits(), 0.5_f64.to_bits());
        assert_eq!(all[7], 1.0);
        assert!(all[8].is_nan());

        // Long enough to enter every supported SIMD backend.
        let signed_zeros: Vec<f64> = (0..16)
            .map(|index| if index % 2 == 0 { -0.0 } else { 0.0 })
            .collect();
        for value in normal_cdf_batch_approx(&signed_zeros) {
            assert_eq!(value.to_bits(), 0.5_f64.to_bits());
        }
    }

    #[test]
    fn batch_price_edge_domains_match_scalar_contract() {
        let r = 0.05_f64;
        let q = 0.02_f64;
        let expiry = 1.7_f64;
        let strike = 100.0_f64;
        let deterministic_boundary = strike * ((q - r) * expiry).exp();
        let spots = [80.0, deterministic_boundary, 100.0, 120.0, 150.0];
        let strikes = [strike; 5];
        let df_r = (-r * expiry).exp();
        let df_q = (-q * expiry).exp();

        // Zero volatility with time remaining is the deterministic discounted
        // terminal payoff. Its Greeks are piecewise analytic away from the
        // forward-strike kink.
        for is_call in [false, true] {
            let prices = bs_price_batch(&spots, &strikes, r, q, 0.0, expiry, is_call);
            let (delta, gamma, vega, theta) =
                bs_greeks_batch(&spots, &strikes, r, q, 0.0, expiry, is_call);
            for index in 0..spots.len() {
                let forward_value = spots[index] * df_q - strikes[index] * df_r;
                let expected = if is_call {
                    forward_value.max(0.0)
                } else {
                    (-forward_value).max(0.0)
                };
                let operation_scale = (spots[index] * df_q)
                    .abs()
                    .max((strikes[index] * df_r).abs());
                assert_machine_precision_eq(prices[index], expected, operation_scale);

                let expected_greeks = if forward_value > 0.0 && is_call {
                    (
                        df_q,
                        0.0,
                        0.0,
                        q * spots[index] * df_q - r * strikes[index] * df_r,
                    )
                } else if forward_value < 0.0 && !is_call {
                    (
                        -df_q,
                        0.0,
                        0.0,
                        -q * spots[index] * df_q + r * strikes[index] * df_r,
                    )
                } else if forward_value == 0.0 {
                    (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                for (got, expected) in [
                    (delta[index], expected_greeks.0),
                    (gamma[index], expected_greeks.1),
                    (vega[index], expected_greeks.2),
                    (theta[index], expected_greeks.3),
                ] {
                    if expected.is_nan() {
                        assert!(got.is_nan());
                    } else {
                        assert_machine_precision_eq(got, expected, operation_scale);
                    }
                }
            }
        }

        // A negative volatility is invalid rather than an alias for the
        // deterministic limit.
        for is_call in [false, true] {
            assert!(
                bs_price_batch(&spots, &strikes, r, q, -0.1, expiry, is_call)
                    .iter()
                    .all(|value| value.is_nan())
            );
            let outputs = bs_greeks_batch(&spots, &strikes, r, q, -0.1, expiry, is_call);
            for output in [outputs.0, outputs.1, outputs.2, outputs.3] {
                assert!(output.iter().all(|value| value.is_nan()));
            }
        }

        // At the exact deterministic boundary, the payoff kink has no unique
        // delta/gamma/vega/theta convention; signal that explicitly.
        let boundary = [100.0; 4];
        let outputs = bs_greeks_batch(&boundary, &boundary, 0.0, 0.0, 0.0, 1.0, true);
        for output in [outputs.0, outputs.1, outputs.2, outputs.3] {
            assert!(output.iter().all(|value| value.is_nan()));
        }

        // At or past expiry, intrinsic value takes precedence even when
        // volatility is also non-positive.
        for (vol, expiry) in [(0.2, 0.0), (0.2, -1.0), (0.0, -1.0)] {
            for is_call in [false, true] {
                let prices = bs_price_batch(&spots, &strikes, r, q, vol, expiry, is_call);
                for index in 0..spots.len() {
                    let intrinsic = if is_call {
                        f64::max(spots[index] - strikes[index], 0.0)
                    } else {
                        f64::max(strikes[index] - spots[index], 0.0)
                    };
                    let operation_scale = spots[index].abs().max(strikes[index].abs());
                    assert_machine_precision_eq(prices[index], intrinsic, operation_scale);
                }

                let (delta, gamma, vega, theta) =
                    bs_greeks_batch(&spots, &strikes, r, q, vol, expiry, is_call);
                for output in [delta, gamma, vega, theta] {
                    assert!(output.iter().all(|value| *value == 0.0));
                }
            }
        }
    }

    #[test]
    fn tiny_total_volatility_preserves_time_value() {
        let spots = [100.0; 16];
        let strikes = [100.0; 16];
        for vol in [1.0e-8, 1.0e-12, 1.0e-16] {
            let expected = 100.0 * 0.398_942_280_401_432_7 * vol;
            for is_call in [false, true] {
                let option_type = if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                for (path, price) in [
                    (
                        "scalar",
                        black_scholes_price(option_type, 100.0, 100.0, 0.0, vol, 1.0),
                    ),
                    (
                        "runtime-dispatched",
                        bs_price_asm(100.0, 100.0, 0.0, 0.0, vol, 1.0, is_call),
                    ),
                ] {
                    let relative_error = ((price - expected) / expected).abs();
                    assert!(
                        relative_error <= 8.0 * f64::EPSILON,
                        "path={path} vol={vol} price={price} reference={expected} rel={relative_error}"
                    );
                }

                let prices = bs_price_batch(&spots, &strikes, 0.0, 0.0, vol, 1.0, is_call);
                for price in prices {
                    let relative_error = ((price - expected) / expected).abs();
                    assert!(
                        relative_error <= 8.0 * f64::EPSILON,
                        "vol={vol} price={price} reference={expected} rel={relative_error}"
                    );
                }
            }
        }
    }

    #[test]
    fn tiny_volatility_preserves_adjacent_float_moneyness_and_extreme_scales() {
        let strike = 100.0_f64;
        let cases = [
            (
                f64::from_bits(strike.to_bits() + 1),
                f64::from_bits(0x3d10_64e0_3188_2f35),
                f64::from_bits(0x3cb9_380c_620b_cd22),
                [
                    f64::from_bits(0x3fed_83ec_a07e_6cae),
                    f64::from_bits(0x42aa_6fea_0407_bd93),
                    f64::from_bits(0x402d_1166_786e_3549),
                    f64::from_bits(0xbcca_2e9e_148f_8ddc),
                    f64::from_bits(0x4057_0f10_dd62_c4e8),
                ],
            ),
            (
                f64::from_bits(strike.to_bits() - 1),
                f64::from_bits(0x3cb9_380c_620b_cd1e),
                f64::from_bits(0x3d10_64e0_3188_2f34),
                [
                    f64::from_bits(0x3fb3_e09a_fc0c_9a92),
                    f64::from_bits(0x42aa_6fea_0407_bd94),
                    f64::from_bits(0x402d_1166_786e_3546),
                    f64::from_bits(0xbcca_2e9e_148f_8dd9),
                    f64::from_bits(0x401f_0ef2_29d3_b182),
                ],
            ),
        ];

        for (spot, expected_call, expected_put, call_greek_reference) in cases {
            let mut prices = [0.0; 2];
            for (index, is_call) in [true, false].into_iter().enumerate() {
                let option_type = if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                let expected = if is_call { expected_call } else { expected_put };
                let (aad, aad_greeks) = black_scholes_price_greeks_aad(
                    option_type,
                    spot,
                    strike,
                    0.0,
                    0.0,
                    1.0e-16,
                    1.0,
                );
                for (path, price) in [
                    (
                        "scalar",
                        black_scholes_price(option_type, spot, strike, 0.0, 1.0e-16, 1.0),
                    ),
                    (
                        "runtime-dispatched",
                        bs_price_asm(spot, strike, 0.0, 0.0, 1.0e-16, 1.0, is_call),
                    ),
                    (
                        "batch",
                        bs_price_batch(&[spot], &[strike], 0.0, 0.0, 1.0e-16, 1.0, is_call)[0],
                    ),
                    ("aad", aad),
                ] {
                    let tolerance = 16.0 * f64::EPSILON * expected.abs();
                    assert!(
                        (price - expected).abs() <= tolerance,
                        "path={path} spot={spot:.17e} price={price:.17e} expected={expected:.17e}"
                    );
                }
                prices[index] = aad;

                let reference_greeks =
                    black_scholes_merton_greeks(option_type, spot, strike, 0.0, 0.0, 1.0e-16, 1.0);
                let expected_greeks = if is_call {
                    call_greek_reference
                } else {
                    [
                        call_greek_reference[0] - 1.0,
                        call_greek_reference[1],
                        call_greek_reference[2],
                        call_greek_reference[3],
                        call_greek_reference[4] - 100.0,
                    ]
                };
                let batch_greeks =
                    bs_greeks_batch(&[spot], &[strike], 0.0, 0.0, 1.0e-16, 1.0, is_call);
                for (path, actual, expected) in [
                    ("scalar delta", reference_greeks.delta, expected_greeks[0]),
                    ("scalar gamma", reference_greeks.gamma, expected_greeks[1]),
                    ("scalar vega", reference_greeks.vega, expected_greeks[2]),
                    ("scalar theta", reference_greeks.theta, expected_greeks[3]),
                    ("scalar rho", reference_greeks.rho, expected_greeks[4]),
                    ("aad delta", aad_greeks.delta, expected_greeks[0]),
                    ("aad gamma", aad_greeks.gamma, expected_greeks[1]),
                    ("aad vega", aad_greeks.vega, expected_greeks[2]),
                    ("aad theta", aad_greeks.theta, expected_greeks[3]),
                    ("aad rho", aad_greeks.rho, expected_greeks[4]),
                    ("batch delta", batch_greeks.0[0], expected_greeks[0]),
                    ("batch gamma", batch_greeks.1[0], expected_greeks[1]),
                    ("batch vega", batch_greeks.2[0], expected_greeks[2]),
                    ("batch theta", batch_greeks.3[0], expected_greeks[3]),
                ] {
                    let tolerance = 16.0 * f64::EPSILON * expected.abs().max(f64::MIN_POSITIVE);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{path}: actual={actual:.17e} expected={expected:.17e}"
                    );
                }
            }

            let parity = spot - strike;
            let tolerance = 16.0 * f64::EPSILON * parity.abs();
            assert!(
                ((prices[0] - prices[1]) - parity).abs() <= tolerance,
                "spot={spot:.17e} call={} put={} parity={parity}",
                prices[0],
                prices[1]
            );
        }

        for is_call in [false, true] {
            let price = bs_price_asm(1.0e308, 1.0e-308, 0.0, 0.0, 1.0e-16, 1.0, is_call);
            assert_eq!(price, if is_call { 1.0e308 } else { 0.0 });
        }
        assert_eq!(
            black_scholes_price(OptionType::Call, 100.0, 0.0, 0.0, 1.0e-16, 1.0),
            100.0
        );

        let subnormal_vol = f64::from_bits(1);
        let expected = (1.0e308 * subnormal_vol) * 0.398_942_280_401_432_7;
        let price =
            black_scholes_price(OptionType::Call, 1.0e308, 1.0e308, 0.0, subnormal_vol, 1.0);
        assert!(((price - expected) / expected).abs() <= 8.0 * f64::EPSILON);

        // The Gaussian tail itself underflows here, but the 1e308 notional
        // rescales the option value into the normal f64 range.
        let tail_spot = f64::from_bits(0x7fe1_1a45_4927_27df);
        let tail_strike = f64::from_bits(0x7fe1_ccf3_85eb_c8a0);
        let tail_expected = f64::from_bits(0x3660_589d_1bcc_e3f6);
        for (path, price) in [
            (
                "scalar log-tail",
                black_scholes_price(OptionType::Call, tail_spot, tail_strike, 0.0, 1.0e-3, 1.0),
            ),
            (
                "dispatch log-tail",
                bs_price_asm(tail_spot, tail_strike, 0.0, 0.0, 1.0e-3, 1.0, true),
            ),
            (
                "batch log-tail",
                bs_price_batch(&[tail_spot], &[tail_strike], 0.0, 0.0, 1.0e-3, 1.0, true)[0],
            ),
        ] {
            let relative_error = ((price - tail_expected) / tail_expected).abs();
            assert!(
                relative_error <= 1.0e-8,
                "path={path} price={price:.17e} expected={tail_expected:.17e} rel={relative_error}"
            );
        }

        let subnormal_tail_rate = -10.0 * subnormal_vol;
        let subnormal_tail_expected = f64::from_bits(0x37c0_15bf_4aac_4661);
        for (path, price) in [
            (
                "scalar subnormal log-tail",
                black_scholes_price(
                    OptionType::Call,
                    1.0e308,
                    1.0e308,
                    subnormal_tail_rate,
                    subnormal_vol,
                    1.0,
                ),
            ),
            (
                "dispatch subnormal log-tail",
                bs_price_asm(
                    1.0e308,
                    1.0e308,
                    subnormal_tail_rate,
                    0.0,
                    subnormal_vol,
                    1.0,
                    true,
                ),
            ),
            (
                "batch subnormal log-tail",
                bs_price_batch(
                    &[1.0e308],
                    &[1.0e308],
                    subnormal_tail_rate,
                    0.0,
                    subnormal_vol,
                    1.0,
                    true,
                )[0],
            ),
        ] {
            let relative_error =
                ((price - subnormal_tail_expected) / subnormal_tail_expected).abs();
            assert!(
                relative_error <= 1.0e-8,
                "path={path} price={price:.17e} expected={subnormal_tail_expected:.17e} rel={relative_error}"
            );
        }

        let subnormal_expiry = 0.25;
        let subnormal_sqrt_t = 0.5;
        let subnormal_atm_price =
            ((1.0e308 * subnormal_vol) * subnormal_sqrt_t) * 0.398_942_280_401_432_7;
        let subnormal_atm_greeks = [
            0.5,
            0.398_942_280_401_432_7 / ((1.0e308 * subnormal_vol) * subnormal_sqrt_t),
            (1.0e308 * 0.398_942_280_401_432_7) * subnormal_sqrt_t,
            -((1.0e308 * 0.398_942_280_401_432_7) * subnormal_vol) / (2.0 * subnormal_sqrt_t),
            1.0e308 * subnormal_expiry * 0.5,
        ];
        let (aad_price, aad_greeks) = black_scholes_price_greeks_aad(
            OptionType::Call,
            1.0e308,
            1.0e308,
            0.0,
            0.0,
            subnormal_vol,
            subnormal_expiry,
        );
        let scalar_greeks = black_scholes_merton_greeks(
            OptionType::Call,
            1.0e308,
            1.0e308,
            0.0,
            0.0,
            subnormal_vol,
            subnormal_expiry,
        );
        let batch_greeks = bs_greeks_batch(
            &[1.0e308],
            &[1.0e308],
            0.0,
            0.0,
            subnormal_vol,
            subnormal_expiry,
            true,
        );
        for (path, actual, reference) in [
            ("subnormal price", aad_price, subnormal_atm_price),
            ("scalar delta", scalar_greeks.delta, subnormal_atm_greeks[0]),
            ("scalar gamma", scalar_greeks.gamma, subnormal_atm_greeks[1]),
            ("scalar vega", scalar_greeks.vega, subnormal_atm_greeks[2]),
            ("scalar theta", scalar_greeks.theta, subnormal_atm_greeks[3]),
            ("scalar rho", scalar_greeks.rho, subnormal_atm_greeks[4]),
            ("aad delta", aad_greeks.delta, subnormal_atm_greeks[0]),
            ("aad gamma", aad_greeks.gamma, subnormal_atm_greeks[1]),
            ("aad vega", aad_greeks.vega, subnormal_atm_greeks[2]),
            ("aad theta", aad_greeks.theta, subnormal_atm_greeks[3]),
            ("aad rho", aad_greeks.rho, subnormal_atm_greeks[4]),
            ("batch delta", batch_greeks.0[0], subnormal_atm_greeks[0]),
            ("batch gamma", batch_greeks.1[0], subnormal_atm_greeks[1]),
            ("batch vega", batch_greeks.2[0], subnormal_atm_greeks[2]),
            ("batch theta", batch_greeks.3[0], subnormal_atm_greeks[3]),
        ] {
            let relative_error = ((actual - reference) / reference).abs();
            assert!(
                relative_error <= 16.0 * f64::EPSILON,
                "path={path} actual={actual:.17e} expected={reference:.17e} rel={relative_error}"
            );
        }

        for (spot, is_call, expected) in [
            (101.0, true, f64::INFINITY),
            (101.0, false, 0.0),
            (99.0, true, 0.0),
            (99.0, false, f64::INFINITY),
        ] {
            let option_type = if is_call {
                OptionType::Call
            } else {
                OptionType::Put
            };
            for (path, price) in [
                (
                    "scalar overflowed discount",
                    openferric::engines::analytic::black_scholes::bs_price(
                        option_type,
                        spot,
                        100.0,
                        -1_000.0,
                        -1_000.0,
                        1.0e-16,
                        1.0,
                    ),
                ),
                (
                    "dispatch overflowed discount",
                    bs_price_asm(spot, 100.0, -1_000.0, -1_000.0, 1.0e-16, 1.0, is_call),
                ),
                (
                    "batch overflowed discount",
                    bs_price_batch(&[spot], &[100.0], -1_000.0, -1_000.0, 1.0e-16, 1.0, is_call)[0],
                ),
                (
                    "aad overflowed discount",
                    black_scholes_price_greeks_aad(
                        option_type,
                        spot,
                        100.0,
                        -1_000.0,
                        -1_000.0,
                        1.0e-16,
                        1.0,
                    )
                    .0,
                ),
            ] {
                assert_eq!(price, expected, "path={path} spot={spot} is_call={is_call}");
            }
        }

        for (r, q, call_expected, put_expected) in [
            (-1_000.0, -1_000.0, f64::INFINITY, f64::INFINITY),
            (-1_000.0, 0.0, 0.0, f64::INFINITY),
            (0.0, -1_000.0, f64::INFINITY, 0.0),
        ] {
            for (is_call, expected) in [(true, call_expected), (false, put_expected)] {
                let option_type = if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                };
                for (path, price) in [
                    (
                        "scalar ordinary-vol overflowed discount",
                        openferric::engines::analytic::black_scholes::bs_price(
                            option_type,
                            100.0,
                            100.0,
                            r,
                            q,
                            0.2,
                            1.0,
                        ),
                    ),
                    (
                        "dispatch ordinary-vol overflowed discount",
                        bs_price_asm(100.0, 100.0, r, q, 0.2, 1.0, is_call),
                    ),
                    (
                        "batch ordinary-vol overflowed discount",
                        bs_price_batch(&[100.0], &[100.0], r, q, 0.2, 1.0, is_call)[0],
                    ),
                    (
                        "aad ordinary-vol overflowed discount",
                        black_scholes_price_greeks_aad(option_type, 100.0, 100.0, r, q, 0.2, 1.0).0,
                    ),
                ] {
                    assert_eq!(price, expected, "path={path} r={r} q={q} is_call={is_call}");
                }
            }
        }
    }

    #[test]
    fn near_routing_boundary_put_preserves_price_and_greeks() {
        // midpoint = ln(S/K)/(vol*sqrt(T)) = 7.9, just inside the ordinary
        // SIMD route's cutoff at 8.0; d1=8.4 is already far enough into the
        // upper tail that Phi(d1) rounds to 1.  Recovering either the put price
        // by parity or delta as Phi(d1)-1 therefore erased a representable
        // result.  Sixteen lanes exercise complete WASM/NEON/AVX2/AVX-512
        // vectors whenever those backends are selected.
        const EXPECTED_PRICE: f64 = 7.878_301_698_281_711e-13;
        const EXPECTED_DELTA: f64 = -2.232_393_197_288_031e-17;
        const EXPECTED_GAMMA: f64 = 7.048_137_000_654_067e-22;
        const EXPECTED_VEGA: f64 = 5.127_753_636_796_672e-11;
        const EXPECTED_THETA: f64 = -2.563_876_818_398_336e-11;

        let spots = [269_728.232_826_851; 16];
        let strikes = [100.0; 16];
        let prices = bs_price_batch(&spots, &strikes, 0.0, 0.0, 1.0, 1.0, false);
        let (delta, gamma, vega, theta) =
            bs_greeks_batch(&spots, &strikes, 0.0, 0.0, 1.0, 1.0, false);

        // Independent SciPy 1.17.1 `ndtr` values, evaluated with Phi(-d1)
        // and Phi(-d2) directly.  A relative budget is required here: an
        // ordinary absolute epsilon would allow the old zero result.
        for (lane, values) in prices
            .iter()
            .zip(delta.iter())
            .zip(gamma.iter())
            .zip(vega.iter())
            .zip(theta.iter())
            .enumerate()
        {
            let ((((&price, &delta), &gamma), &vega), &theta) = values;
            for (label, actual, expected) in [
                ("price", price, EXPECTED_PRICE),
                ("delta", delta, EXPECTED_DELTA),
                ("gamma", gamma, EXPECTED_GAMMA),
                ("vega", vega, EXPECTED_VEGA),
                ("theta", theta, EXPECTED_THETA),
            ] {
                let relative_error = ((actual - expected) / expected).abs();
                assert!(
                    relative_error <= 2.0e-12,
                    "lane={lane} {label}: actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}"
                );
            }
        }
    }

    #[test]
    fn batch_price_and_greeks_handle_zero_and_invalid_spot_strike_lanes() {
        let spots = [
            0.0,
            100.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            120.0,
            80.0,
            100.0,
        ];
        let strikes = [
            100.0,
            0.0,
            100.0,
            100.0,
            100.0,
            -1.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let (r, q, t, vol) = (0.05_f64, 0.02_f64, 1.0_f64, 0.2_f64);
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let calls = bs_price_batch(&spots, &strikes, r, q, vol, t, true);
        let puts = bs_price_batch(&spots, &strikes, r, q, vol, t, false);
        assert_eq!(calls[0], 0.0);
        assert_eq!(puts[0], 100.0 * df_r);
        assert_eq!(calls[1], 100.0 * df_q);
        assert_eq!(puts[1], 0.0);
        assert_eq!(
            black_scholes_price(OptionType::Call, 100.0 * df_q, 0.0, r, vol, t),
            100.0 * df_q
        );
        for index in 2..spots.len() {
            assert!(calls[index].is_nan(), "call index={index}");
            assert!(puts[index].is_nan(), "put index={index}");
        }

        let call_greeks = bs_greeks_batch(&spots, &strikes, r, q, vol, t, true);
        assert_eq!(
            (
                call_greeks.0[0],
                call_greeks.1[0],
                call_greeks.2[0],
                call_greeks.3[0],
            ),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(call_greeks.0[1], df_q);
        assert_eq!(call_greeks.1[1], 0.0);
        assert_eq!(call_greeks.2[1], 0.0);
        assert_eq!(call_greeks.3[1], q * 100.0 * df_q);

        let put_greeks = bs_greeks_batch(&spots, &strikes, r, q, vol, t, false);
        assert_eq!(put_greeks.0[0], -df_q);
        assert_eq!(put_greeks.1[0], 0.0);
        assert_eq!(put_greeks.2[0], 0.0);
        assert_eq!(put_greeks.3[0], r * 100.0 * df_r);
        assert_eq!(
            (
                put_greeks.0[1],
                put_greeks.1[1],
                put_greeks.2[1],
                put_greeks.3[1],
            ),
            (0.0, 0.0, 0.0, 0.0)
        );
        for index in 2..spots.len() {
            assert!(call_greeks.0[index].is_nan(), "call delta index={index}");
            assert!(put_greeks.0[index].is_nan(), "put delta index={index}");
        }

        // With both spot and strike zero there is no unique scale or Greek
        // convention. All public scalar, dispatch, and batch entry points
        // reject the point consistently instead of selecting path-dependent
        // boundary values.
        for is_call in [false, true] {
            let option_type = if is_call {
                OptionType::Call
            } else {
                OptionType::Put
            };
            assert!(black_scholes_price(option_type, 0.0, 0.0, r, vol, t).is_nan());
            assert!(bs_price_asm(0.0, 0.0, r, q, vol, t, is_call).is_nan());
            assert!(bs_price_batch(&[0.0], &[0.0], r, q, vol, t, is_call)[0].is_nan());

            let greeks = bs_greeks_batch(&[0.0], &[0.0], r, q, vol, t, is_call);
            for output in [greeks.0, greeks.1, greeks.2, greeks.3] {
                assert!(output[0].is_nan());
            }
        }
    }

    #[test]
    fn subnormal_volatility_far_from_strike_reaches_deterministic_limit() {
        let strikes = [100.0; 16];
        let tiny_vol = f64::from_bits(1);
        for (spot, expected_call, expected_put) in [(50.0, 0.0, 50.0), (150.0, 50.0, 0.0)] {
            let spots = [spot; 16];
            for price in bs_price_batch(&spots, &strikes, 0.0, 0.0, tiny_vol, 1.0, true) {
                assert_eq!(price, expected_call);
            }
            for price in bs_price_batch(&spots, &strikes, 0.0, 0.0, tiny_vol, 1.0, false) {
                assert_eq!(price, expected_put);
            }
        }
    }

    #[test]
    #[should_panic(expected = "output length must match input")]
    fn price_workspace_rejects_wrong_output_length() {
        let mut output = [0.0; 1];
        bs_price_batch_into(
            &[100.0, 101.0],
            &[100.0, 101.0],
            0.03,
            0.0,
            0.2,
            1.0,
            true,
            &mut output,
        );
    }

    #[test]
    #[should_panic(expected = "output length must match input")]
    fn cdf_workspace_rejects_wrong_output_length() {
        let mut output = [0.0; 1];
        normal_cdf_batch_approx_into(&[0.0, 1.0], &mut output);
    }
}

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

    #[track_caller]
    fn assert_vector_roundoff_close(label: &str, actual: f64, expected: f64, operation_scale: f64) {
        let tolerance = 1_024.0
            * f64::EPSILON
            * actual
                .abs()
                .max(expected.abs())
                .max(operation_scale.abs())
                .max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17e}, reference={expected:.17e}, \
             error={:.3e}, vector-roundoff budget={tolerance:.3e}",
            (actual - expected).abs()
        );
    }

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
        let nmd1 = normal_cdf(-d1);
        let nmd2 = normal_cdf(-d2);

        let delta = if is_call {
            df_q * normal_cdf(d1)
        } else {
            -df_q * nmd1
        };
        let gamma = df_q * pdf / (s * vol * sqrt_t);
        let vega = s * df_q * pdf * sqrt_t;
        let theta = if is_call {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) + q * s * df_q * normal_cdf(d1)
                - r * k * df_r * normal_cdf(d2)
        } else {
            -s * df_q * pdf * vol / (2.0 * sqrt_t) - q * s * df_q * nmd1 + r * k * df_r * nmd2
        };
        (delta, gamma, vega, theta)
    }

    #[test]
    fn simd_bs_price_matches_scalar_with_vector_roundoff_budget() {
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
                assert_vector_roundoff_close(
                    &format!("price index={i}"),
                    simd[i],
                    scalar,
                    adjusted_spot + strikes[i] * (-r * t).exp(),
                );
            }
        }
    }

    #[test]
    fn simd_normal_cdf_matches_independent_cody_reference_within_1e_minus_7() {
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
    fn simd_bs_greeks_match_independent_scalar_with_vector_roundoff_budget() {
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

                assert_vector_roundoff_close("delta", delta[i], d_ref, 1.0);
                assert_vector_roundoff_close("gamma", gamma[i], g_ref, 1.0);
                assert_vector_roundoff_close("vega", vega[i], v_ref, spots[i] * t.sqrt());
                assert_vector_roundoff_close("theta", theta[i], th_ref, spots[i] + strikes[i]);
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
    fn simd_ln_matches_scalar_absolute_error_bound() {
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
    fn avx2_subnormal_exp_ln_and_inverse_cdf_match_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
        use openferric::math::simd_math::inv_norm_cdf_batch_avx2;

        let exp_inputs = [-708.5, -710.0, -744.0, -745.0];
        let exact = unsafe { simd_exp_batch(&exp_inputs) };
        let fast = unsafe { simd_fast_exp_batch(&exp_inputs) };
        for (index, input) in exp_inputs.iter().enumerate() {
            let expected = input.exp();
            assert_eq!(exact[index].to_bits(), expected.to_bits(), "x={input}");
            assert_eq!(fast[index].to_bits(), expected.to_bits(), "x={input}");
        }

        let ln_inputs = [
            f64::from_bits(1),
            f64::from_bits((1_u64 << 52) - 1),
            f64::MIN_POSITIVE / 2.0,
            f64::MIN_POSITIVE,
        ];
        let logarithms = unsafe { simd_ln_batch(&ln_inputs) };
        for index in 0..3 {
            assert_eq!(
                logarithms[index].to_bits(),
                ln_inputs[index].ln().to_bits(),
                "x={}",
                ln_inputs[index]
            );
        }

        let mut probabilities = [f64::from_bits(1), 1.0e-310, 1.0e-300, 0.5];
        let expected = probabilities.map(beasley_springer_moro_inv_cdf);
        unsafe { inv_norm_cdf_batch_avx2(&mut probabilities) };
        for (got, want) in probabilities.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 2.0e-13,
                "inverse CDF got={got}, reference={want}"
            );
        }
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
    fn avx512_subnormal_exp_ln_and_inverse_cdf_match_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        use openferric::math::fast_norm::beasley_springer_moro_inv_cdf;
        use openferric::math::simd_avx512::{
            exp_f64x8, fast_exp_f64x8, inv_norm_cdf_batch_avx512, ln_f64x8, load_f64x8, store_f64x8,
        };

        let exp_inputs = [
            -708.5, -710.0, -744.0, -745.0, -720.0, -730.0, -740.0, -750.0,
        ];
        let mut exact = [0.0_f64; 8];
        let mut fast = [0.0_f64; 8];
        unsafe {
            let input = load_f64x8(&exp_inputs, 0);
            store_f64x8(&mut exact, 0, exp_f64x8(input));
            store_f64x8(&mut fast, 0, fast_exp_f64x8(input));
        }
        for (index, input) in exp_inputs.iter().enumerate() {
            let expected = input.exp();
            assert_eq!(exact[index].to_bits(), expected.to_bits(), "x={input}");
            assert_eq!(fast[index].to_bits(), expected.to_bits(), "x={input}");
        }

        let ln_inputs = [
            f64::from_bits(1),
            f64::from_bits((1_u64 << 52) - 1),
            f64::MIN_POSITIVE / 2.0,
            f64::MIN_POSITIVE / 4.0,
            1.0,
            2.0,
            0.5,
            f64::MIN_POSITIVE,
        ];
        let mut logarithms = [0.0_f64; 8];
        unsafe {
            let input = load_f64x8(&ln_inputs, 0);
            store_f64x8(&mut logarithms, 0, ln_f64x8(input));
        }
        for index in 0..4 {
            assert_eq!(
                logarithms[index].to_bits(),
                ln_inputs[index].ln().to_bits(),
                "x={}",
                ln_inputs[index]
            );
        }

        let mut probabilities = [
            f64::from_bits(1),
            1.0e-310,
            1.0e-300,
            0.5,
            0.25,
            0.75,
            0.1,
            0.9,
        ];
        let expected = probabilities.map(beasley_springer_moro_inv_cdf);
        unsafe { inv_norm_cdf_batch_avx512(&mut probabilities) };
        for (got, want) in probabilities.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 2.0e-13,
                "inverse CDF got={got}, reference={want}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_exp_matches_scalar_relative_error_bound() {
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

        // ln(negative) = NaN, ln(+-0) = -inf, ln(+inf) = +inf, ln(NaN) = NaN,
        // mirroring the AVX2/AVX512 special-value blending and scalar f64::ln.
        let xs = [
            -1.0,
            -0.5,
            0.0,
            -0.0,
            1.0,
            2.5,
            f64::INFINITY,
            4.0,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
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
    fn neon_exp_special_values_match_std() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        // Subnormal outputs must retain their IEEE-754 value; NaN must
        // propagate instead of being clamped into exp(max_x), and overflow
        // stays +inf.
        // Even length so every special value (incl. the NaN) goes through the
        // 2-lane SIMD body rather than the scalar remainder.
        let xs = [
            f64::NEG_INFINITY,
            -1e308,
            -710.0,
            -708.5,
            0.0,
            709.5,
            709.9,
            f64::INFINITY,
            f64::NAN,
            1.0,
        ];
        let got = unsafe { neon_exp_batch(&xs) };
        for (x, y) in xs.iter().zip(got.iter()) {
            let expected = x.exp();
            if expected.is_nan() {
                assert!(y.is_nan(), "x={x} neon={y} expected NaN");
            } else if expected.is_infinite() {
                assert_eq!(*y, expected, "x={x} neon={y} expected={expected}");
            } else if expected < f64::MIN_POSITIVE {
                assert_eq!(
                    y.to_bits(),
                    expected.to_bits(),
                    "x={x} neon={y} expected subnormal {expected}"
                );
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
    fn neon_bs_price_matches_scalar_with_vector_roundoff_budget() {
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
                let operation_scale = adjusted_spot + strikes[i] * (-r * t).exp();
                let tolerance = 1_024.0 * f64::EPSILON * operation_scale.max(1.0);
                assert!(
                    (batch[i] - scalar).abs() <= tolerance,
                    "idx {i}: neon={} scalar={} diff={} vector-roundoff budget={tolerance}",
                    batch[i],
                    scalar,
                    (batch[i] - scalar).abs()
                );
            }
        }
    }

    #[test]
    fn neon_odd_length_tail_matches_vector_lanes() {
        use openferric::engines::analytic::bs_simd::bs_price_batch;

        let (s, k, r, q, vol, t) = (100.0_f64, 105.0_f64, 0.02, 0.0, 0.2, 1.0);
        for &is_call in &[true, false] {
            for n in [9usize, 17, 33] {
                let spots = vec![s; n];
                let strikes = vec![k; n];
                let out = bs_price_batch(&spots, &strikes, r, q, vol, t, is_call);
                let lane = out[0];
                for (i, &price) in out.iter().enumerate() {
                    // Accurate per-lane CDF values are shared by the vector
                    // body and scalar tail; only ln/FMA grouping differs.
                    assert!(
                        (price - lane).abs() <= 1e-12 * lane.abs(),
                        "n={n} is_call={is_call} idx {i}: tail={price} lane={lane} diff={}",
                        (price - lane).abs()
                    );
                }
            }
        }
    }
}
