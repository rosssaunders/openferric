//! Tests executed by `wasm-bindgen-test` in Node by default, or in a browser
//! when the `browser-tests` feature is enabled.

use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(feature = "browser-tests")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use super::{dsl, pricing};

fn assert_roundoff_close(actual: f64, expected: f64, label: &str) {
    let budget = 256.0 * f64::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= budget,
        "{label}: actual={actual:.17e}, reference={expected:.17e}, budget={budget:.3e}"
    );
}

// SIMD128 has no vector ln/erfc, so those evaluations are lane-scalar and all
// remaining price/Greek arithmetic is f64x2.  This explicit operation budget
// covers ordinary expression grouping against the cached SciPy grid; it is not
// an economic-price tolerance.
const BATCH_KERNEL_ROUNDOFF_ULPS: f64 = 512.0;

fn assert_batch_row(actual: [f64; 5], analytic: [f64; 5], label: &str) {
    const FIELDS: [&str; 5] = ["price", "delta", "gamma", "vega", "theta"];
    for field in 0..FIELDS.len() {
        let roundoff_budget =
            BATCH_KERNEL_ROUNDOFF_ULPS * f64::EPSILON * analytic[field].abs().max(1.0);
        let roundoff_error = (actual[field] - analytic[field]).abs();
        assert!(
            roundoff_error <= roundoff_budget,
            "{label} {}: actual={:.17e}, SciPy={:.17e}, error={:.3e}, \
             vector-roundoff budget={:.3e}",
            FIELDS[field],
            actual[field],
            analytic[field],
            roundoff_error,
            roundoff_budget
        );
    }
}

#[wasm_bindgen_test]
fn implied_vol_export_round_trips_through_wasm() {
    let call =
        pricing::bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).expect("valid call inputs");
    let implied = pricing::bs_implied_vol(call, 100.0, 100.0, 0.05, 0.0, 1.0, true)
        .expect("price generated from Black-Scholes has an implied volatility");
    assert!((implied - 0.20).abs() <= 1.0e-12);
}

#[wasm_bindgen_test]
fn slice_and_vector_exports_preserve_shape_and_values() {
    // scipy.stats.norm 1.17.1 closed-form references. Rows are
    // [price, delta, gamma, vega, theta, rho, vanna, volga].
    const SCIPY: [[f64; 8]; 3] = [
        [
            19.256_744_296_451_643,
            -0.997_728_343_542_620_7,
            0.001_186_078_233_901_680_7,
            0.284_658_776_136_403_4,
            2.886_852_720_554_918_2,
            -24.768_752_944_965_325,
            0.138_189_849_348_209_52,
            15.685_915_156_123_569,
        ],
        [
            11.235_557_594_039_619,
            0.590_833_805_852_729_6,
            0.015_331_789_545_074_541,
            38.329_473_862_686_356,
            -6.114_263_346_632_399,
            47.847_822_991_233_33,
            0.007_665_894_772_537_278,
            -0.187_814_421_927_163_33,
        ],
        [
            10.007_353_474_594_296,
            -0.221_505_887_327_483_83,
            0.004_918_244_023_908_108,
            49.575_899_760_993_735,
            -3.040_102_360_978_294_4,
            -73.176_119_907_784_68,
            -0.202_049_710_504_194_12,
            25.272_716_549_454_152,
        ],
    ];
    let spots = [80.0, 100.0, 120.0];
    let strikes = [100.0, 100.0, 100.0];
    let rates = [0.03, 0.04, 0.05];
    let dividends = [0.0, 0.01, 0.02];
    let vols = [0.15, 0.25, 0.35];
    let expiries = [0.25, 1.0, 2.0];
    let is_calls = [0_u8, 1, 0];

    let batch = pricing::bs_price_batch_wasm(
        &spots, &strikes, &rates, &dividends, &vols, &expiries, &is_calls,
    )
    .expect("matching batch lengths");
    assert_eq!(batch.len(), spots.len());
    for index in 0..spots.len() {
        let scalar = pricing::bs_price(
            spots[index],
            strikes[index],
            rates[index],
            dividends[index],
            vols[index],
            expiries[index],
            is_calls[index] != 0,
        )
        .expect("valid scalar batch member");
        assert_roundoff_close(scalar, SCIPY[index][0], "scalar price vs SciPy");
        assert_roundoff_close(batch[index], SCIPY[index][0], "batch price vs SciPy");
        assert_eq!(batch[index].to_bits(), scalar.to_bits());
    }

    let greeks = pricing::bsm_greeks_batch_wasm(
        &spots, &strikes, &rates, &dividends, &vols, &expiries, &is_calls,
    )
    .expect("matching batch lengths");
    assert_eq!(greeks.len(), spots.len() * 7);
    for index in 0..spots.len() {
        for greek in 0..7 {
            assert_roundoff_close(
                greeks[index * 7 + greek],
                SCIPY[index][greek + 1],
                "batch Greek vs SciPy",
            );
        }
    }
}

#[wasm_bindgen_test]
fn scalar_and_batch_pricing_reject_the_same_invalid_domains() {
    let scalar_error = pricing::bs_price(-1.0, 100.0, 0.03, 0.0, 0.2, 1.0, true)
        .expect_err("negative scalar spot must be rejected")
        .as_string()
        .expect("validation error must contain a JavaScript string");
    assert!(scalar_error.contains("`spot`"));

    let batch_error = pricing::bs_price_batch_wasm(
        &[100.0, -1.0],
        &[100.0, 100.0],
        &[0.03, 0.03],
        &[0.0, 0.0],
        &[0.2, 0.2],
        &[1.0, 1.0],
        &[1, 1],
    )
    .expect_err("negative batch spot must be rejected")
    .as_string()
    .expect("validation error must contain a JavaScript string");
    assert!(batch_error.contains("`spots[1]`"));

    let uniform_error = pricing::bs_price_uniform_batch_wasm(
        &[100.0, f64::NAN],
        &[100.0, 100.0],
        0.03,
        0.0,
        0.2,
        1.0,
        true,
    )
    .expect_err("non-finite uniform-batch spot must be rejected")
    .as_string()
    .expect("validation error must contain a JavaScript string");
    assert!(uniform_error.contains("`spots[1]`"));

    let flag_error =
        pricing::bs_price_batch_wasm(&[100.0], &[100.0], &[0.03], &[0.0], &[0.2], &[1.0], &[2])
            .expect_err("batch option flags outside 0/1 must be rejected")
            .as_string()
            .expect("validation error must contain a JavaScript string");
    assert!(flag_error.contains("`is_calls[0]`"));
}

#[wasm_bindgen_test]
fn greek_and_black76_batches_report_the_invalid_element() {
    let greek_error = pricing::bsm_greeks_batch_wasm(
        &[100.0, 100.0],
        &[100.0, 100.0],
        &[0.03, 0.03],
        &[0.0, 0.0],
        &[0.2, f64::NAN],
        &[1.0, 1.0],
        &[1, 1],
    )
    .expect_err("non-finite batch volatility must be rejected")
    .as_string()
    .expect("validation error must contain a JavaScript string");
    assert!(greek_error.contains("`vols[1]`"));

    let black_price_error = pricing::black76_price_batch_wasm(
        &[100.0, -1.0],
        &[100.0, 100.0],
        &[0.03, 0.03],
        &[0.2, 0.2],
        &[1.0, 1.0],
        &[1, 1],
    )
    .expect_err("negative Black-76 forward must be rejected")
    .as_string()
    .expect("validation error must contain a JavaScript string");
    assert!(black_price_error.contains("`forwards[1]`"));

    let black_greek_error = pricing::black76_greeks_batch_wasm(
        &[100.0, 100.0],
        &[100.0, 100.0],
        &[0.03, f64::INFINITY],
        &[0.2, 0.2],
        &[1.0, 1.0],
        &[1, 1],
    )
    .expect_err("non-finite Black-76 rate must be rejected")
    .as_string()
    .expect("validation error must contain a JavaScript string");
    assert!(black_greek_error.contains("`rates[1]`"));
}

#[wasm_bindgen_test]
fn black76_deep_tail_put_greeks_remain_representable_through_wasm() {
    let actual =
        pricing::black76_greeks_batch_wasm(&[5.0e18], &[1.0e18], &[0.0], &[0.2], &[1.0], &[0])
            .expect("valid deep-tail Black-76 put");
    let expected = [
        -1.862_397_753_202_606_3e-16,
        1.539_548_175_331_966_1e-33,
        7.697_740_876_659_831e3,
        -7.697_740_876_659_832e2,
        -2.275_288_460_097_726_8e1,
        -6.117_540_594_728_432e-14,
        2.492_038_143_976_294_4e6,
    ];
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 2.0e-11,
            "Greek {index}: actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}"
        );
    }
}

#[wasm_bindgen_test]
fn empty_slices_and_invalid_dsl_are_well_defined() {
    let empty = pricing::bs_price_batch_wasm(&[], &[], &[], &[], &[], &[], &[])
        .expect("empty arrays have matching lengths");
    assert!(empty.is_empty());

    let invalid_dsl = dsl::dsl_parse_and_compile("product { payoff = @; }");
    let value: serde_json::Value =
        serde_json::from_str(&invalid_dsl).expect("DSL ABI should return valid JSON");
    assert!(value.get("err").is_some());
}

#[wasm_bindgen_test]
fn remaining_batch_exports_reject_mismatched_lengths_without_trapping() {
    assert!(
        pricing::bs_price_uniform_batch_wasm(&[100.0], &[100.0, 110.0], 0.03, 0.0, 0.2, 1.0, true,)
            .is_err()
    );
    assert!(
        pricing::bsm_greeks_uniform_batch_wasm(
            &[100.0, 110.0],
            &[100.0],
            0.03,
            0.0,
            0.2,
            1.0,
            true,
        )
        .is_err()
    );
    assert!(
        pricing::black76_price_batch_wasm(&[100.0], &[100.0], &[0.03], &[0.2], &[1.0], &[],)
            .is_err()
    );
    assert!(
        pricing::bsm_greeks_batch_wasm(&[100.0], &[100.0], &[0.03], &[0.0], &[], &[1.0], &[1],)
            .is_err()
    );
    assert!(
        pricing::black76_greeks_batch_wasm(
            &[100.0],
            &[100.0],
            &[0.03, 0.04],
            &[0.2],
            &[1.0],
            &[1],
        )
        .is_err()
    );
}

#[wasm_bindgen_test]
fn vector_return_values_remain_independent_across_calls() {
    // Independent scipy.stats.norm 1.17.1 references for all seven Greeks.
    const FIRST: [f64; 7] = [
        0.636_830_651_175_619_1,
        0.018_762_017_345_846_895,
        37.524_034_691_693_79,
        -6.414_027_546_438_197,
        53.232_481_545_376_345,
        -0.281_430_260_187_703_45,
        9.850_059_106_569_622,
    ];
    const SECOND: [f64; 7] = [
        -0.841_792_038_504_025_8,
        0.010_190_287_964_317_096,
        13.043_568_594_325_887,
        -7.566_600_243_802_140_5,
        -50.115_277_223_267_25,
        0.761_120_212_289_303,
        44.670_713_225_205_41,
    ];
    let first = pricing::bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true)
        .expect("valid first option");
    let second = pricing::bsm_greeks_wasm(80.0, 110.0, -0.01, 0.02, 0.40, 0.5, false)
        .expect("valid second option");

    assert_eq!(first.len(), 7);
    assert_eq!(second.len(), 7);
    for greek in 0..7 {
        assert_roundoff_close(first[greek], FIRST[greek], "first scalar Greek vs SciPy");
        assert_roundoff_close(second[greek], SECOND[greek], "second scalar Greek vs SciPy");
    }
}

#[wasm_bindgen_test]
fn uniform_analytic_batches_match_scalar_exports_and_cover_f64x2_tail() {
    // Five values exercise two complete f64x2 vectors and the scalar tail in
    // the opt-in SIMD128 package. The same ABI test also covers the portable
    // package's scalar fallback.
    let spots = [72.0, 91.0, 100.0, 117.0, 143.0];
    let strikes = [100.0, 95.0, 100.0, 105.0, 110.0];
    let rate = 0.035;
    let dividend = 0.012;
    let vol = 0.27;
    let expiry = 1.4;

    // scipy.stats.norm 1.17.1 closed-form references, [price, delta, gamma,
    // vega, theta], ordered as puts then calls.
    const ANALYTIC: [[[f64; 5]; 5]; 2] = [
        [
            [
                26.902_156_828_552_96,
                -0.765_711_631_470_484_9,
                0.012_701_528_416_451_334,
                24.889_305_411_514_04,
                -0.190_446_213_967_233_82,
            ],
            [
                11.911_230_862_887_805,
                -0.442_420_553_294_698_87,
                0.013_387_646_298_840_055,
                41.906_251_422_262_51,
                -2.698_080_660_328_436,
            ],
            [
                10.785_641_740_665_284,
                -0.390_611_091_035_622_64,
                0.011_869_910_877_981_915,
                44.868_263_118_771_644,
                -3.050_679_544_719_192,
            ],
            [
                7.392_265_337_949_354,
                -0.269_927_703_883_271_9,
                0.008_770_427_956_695_85,
                45.382_070_777_101_19,
                -3.391_023_515_528_074,
            ],
            [
                3.709_610_860_637_326_4,
                -0.137_348_317_187_422_21,
                0.004_783_372_691_996_976,
                36.974_141_131_528_25,
                -2.983_788_613_759_915,
            ],
        ],
        [
            [
                2.484_547_837_295_997,
                0.217_628_701_565_536_3,
                0.012_701_528_416_451_334,
                24.889_305_411_514_04,
                -2.673_474_120_168_879_4,
            ],
            [
                10.937_993_847_807_753,
                0.540_919_779_741_322_3,
                0.013_387_646_298_840_055,
                41.906_251_422_262_51,
                -4.790_275_272_900_629_5,
            ],
            [
                13.901_562_074_416_915,
                0.592_729_242_000_398_6,
                0.011_869_910_877_981_915,
                44.868_263_118_771_644,
                -5.203_305_099_020_734,
            ],
            [
                22.464_065_684_820_81,
                0.713_412_629_152_749_3,
                0.008_770_427_956_695_85,
                45.382_070_777_101_19,
                -5.509_679_339_587_506,
            ],
            [
                39.587_354_217_952_836,
                0.845_992_015_848_599,
                0.004_783_372_691_996_976,
                36.974_141_131_528_25,
                -4.962_273_951_609_347_5,
            ],
        ],
    ];
    for (option_kind, is_call) in [false, true].into_iter().enumerate() {
        let prices = pricing::bs_price_uniform_batch_wasm(
            &spots, &strikes, rate, dividend, vol, expiry, is_call,
        )
        .expect("matching uniform batch");
        let greeks = pricing::bsm_greeks_uniform_batch_wasm(
            &spots, &strikes, rate, dividend, vol, expiry, is_call,
        )
        .expect("matching uniform batch");

        assert_eq!(prices.len(), spots.len());
        assert_eq!(greeks.len(), spots.len() * 4);
        for index in 0..spots.len() {
            let scalar_price = pricing::bs_price(
                spots[index],
                strikes[index],
                rate,
                dividend,
                vol,
                expiry,
                is_call,
            )
            .expect("valid scalar option");
            let scalar_greeks = pricing::bsm_greeks_wasm(
                spots[index],
                strikes[index],
                rate,
                dividend,
                vol,
                expiry,
                is_call,
            )
            .expect("valid scalar option");

            assert_roundoff_close(
                scalar_price,
                ANALYTIC[option_kind][index][0],
                "scalar price vs SciPy",
            );
            for greek in 0..4 {
                assert_roundoff_close(
                    scalar_greeks[greek],
                    ANALYTIC[option_kind][index][greek + 1],
                    "scalar Greek vs SciPy",
                );
            }

            assert_batch_row(
                [
                    prices[index],
                    greeks[index * 4],
                    greeks[index * 4 + 1],
                    greeks[index * 4 + 2],
                    greeks[index * 4 + 3],
                ],
                ANALYTIC[option_kind][index],
                "uniform batch",
            );
        }
    }
}

#[wasm_bindgen_test]
fn uniform_analytic_batches_preserve_deterministic_limits() {
    let spots = [80.0, 100.0, 120.0];
    let strikes = [100.0; 3];

    let expiry_prices =
        pricing::bs_price_uniform_batch_wasm(&spots, &strikes, 0.03, 0.01, 0.2, 0.0, true)
            .expect("valid zero-expiry batch");
    assert_eq!(expiry_prices, [0.0, 0.0, 20.0]);

    let deterministic =
        pricing::bs_price_uniform_batch_wasm(&spots, &strikes, 0.03, 0.01, 0.0, 1.5, false)
            .expect("valid deterministic batch");
    let df_r = (-0.03_f64 * 1.5).exp();
    let df_q = (-0.01_f64 * 1.5).exp();
    for index in 0..spots.len() {
        let expected = (strikes[index] * df_r - spots[index] * df_q).max(0.0);
        assert_eq!(deterministic[index], expected);
    }

    let deterministic_greeks =
        pricing::bsm_greeks_uniform_batch_wasm(&spots, &strikes, 0.03, 0.01, 0.0, 1.5, false)
            .expect("valid deterministic batch");
    for index in 0..spots.len() {
        let scalar =
            pricing::bsm_greeks_wasm(spots[index], strikes[index], 0.03, 0.01, 0.0, 1.5, false)
                .expect("valid deterministic scalar option");
        assert_eq!(
            &deterministic_greeks[index * 4..index * 4 + 4],
            &scalar[..4]
        );
    }

    assert_eq!(deterministic_greeks[0], -df_q);
    assert_eq!(deterministic_greeks[1], 0.0);
    assert_eq!(deterministic_greeks[2], 0.0);
    assert_eq!(
        deterministic_greeks[3],
        -0.01 * spots[0] * df_q + 0.03 * strikes[0] * df_r
    );
    assert!(deterministic_greeks[4..].iter().all(|value| *value == 0.0));
}

#[cfg(all(feature = "simd", target_feature = "simd128"))]
#[wasm_bindgen_test]
fn opt_in_package_selects_explicit_wasm_simd128_pricing_backend() {
    use openferric::engines::analytic::{
        BatchSimdBackend, detected_batch_simd_backend, normal_cdf_batch_approx,
    };

    assert_eq!(detected_batch_simd_backend(), BatchSimdBackend::WasmSimd128);

    // scipy.stats.norm 1.17.1 analytic prices for the five lanes below.
    const ANALYTIC_PRICES: [f64; 5] = [
        1.413_162_170_741_385_3,
        4.106_357_300_061_326,
        8.827_321_225_352_122,
        15.453_530_883_296_693,
        23.505_867_004_864_285,
    ];
    // The named export is important: it keeps the pricing kernel reachable
    // through wasm-bindgen/LTO, so SIMD opcodes come from a real pricing path
    // rather than incidental dependency code.
    let prices = pricing::bs_price_uniform_batch_wasm(
        &[80.0, 90.0, 100.0, 110.0, 120.0],
        &[100.0; 5],
        0.03,
        0.01,
        0.2,
        1.0,
        true,
    )
    .expect("matching uniform batch");
    assert_eq!(prices.len(), 5);
    for index in 0..prices.len() {
        let roundoff_budget =
            BATCH_KERNEL_ROUNDOFF_ULPS * f64::EPSILON * ANALYTIC_PRICES[index].abs().max(1.0);
        assert!(
            (prices[index] - ANALYTIC_PRICES[index]).abs() <= roundoff_budget,
            "SIMD price {index}: actual={}, SciPy={}, vector-roundoff budget={roundoff_budget}",
            prices[index],
            ANALYTIC_PRICES[index]
        );
    }

    // Two SIMD pairs plus a scalar tail cover infinities, signed zero, and
    // NaN using the same edge semantics as the scalar CDF.
    let cdf = normal_cdf_batch_approx(&[f64::NEG_INFINITY, f64::INFINITY, -0.0, 0.0, f64::NAN]);
    assert_eq!(cdf[0], 0.0);
    assert_eq!(cdf[1], 1.0);
    assert_eq!(cdf[2].to_bits(), 0.5_f64.to_bits());
    assert_eq!(cdf[3].to_bits(), 0.5_f64.to_bits());
    assert!(cdf[4].is_nan());

    let finite_cdf = normal_cdf_batch_approx(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
    let analytic_cdf: [f64; 5] = [
        0.001_349_898_031_630_093_3,
        0.158_655_253_931_457_07,
        0.5,
        0.841_344_746_068_542_9,
        0.998_650_101_968_369_9,
    ];
    let approximation_grid_cdf: [f64; 5] = [
        0.001_349_967_222_235_237_7,
        0.158_655_259_563_131_53,
        0.5,
        0.841_344_740_436_868_5,
        0.998_650_032_777_764_8,
    ];
    for index in 0..finite_cdf.len() {
        assert!((finite_cdf[index] - analytic_cdf[index]).abs() <= 7.0e-8);
        let roundoff_budget = BATCH_KERNEL_ROUNDOFF_ULPS
            * f64::EPSILON
            * approximation_grid_cdf[index].abs().max(1.0);
        assert!(
            (finite_cdf[index] - approximation_grid_cdf[index]).abs() <= roundoff_budget,
            "SIMD CDF {index}: actual={}, A&S grid={}, budget={roundoff_budget}",
            finite_cdf[index],
            approximation_grid_cdf[index]
        );
    }
}
