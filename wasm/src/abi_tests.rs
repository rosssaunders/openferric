//! Tests executed by `wasm-bindgen-test` in Node by default, or in a browser
//! when the `browser-tests` feature is enabled.

use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(feature = "browser-tests")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use super::{dsl, pricing};

#[wasm_bindgen_test]
fn implied_vol_export_round_trips_through_wasm() {
    let call =
        pricing::bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).expect("valid call inputs");
    let implied = pricing::bs_implied_vol(call, 100.0, 100.0, 0.05, 0.0, 1.0, true)
        .expect("price generated from Black-Scholes has an implied volatility");
    assert!((implied - 0.20).abs() < 1.0e-8);
}

#[wasm_bindgen_test]
fn slice_and_vector_exports_preserve_shape_and_values() {
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
        assert!((batch[index] - scalar).abs() < 1.0e-12);
    }

    let greeks = pricing::bsm_greeks_batch_wasm(
        &spots, &strikes, &rates, &dividends, &vols, &expiries, &is_calls,
    )
    .expect("matching batch lengths");
    assert_eq!(greeks.len(), spots.len() * 7);
    assert!(greeks.iter().all(|value| value.is_finite()));
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
    let first = pricing::bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true)
        .expect("valid first option");
    let second = pricing::bsm_greeks_wasm(80.0, 110.0, -0.01, 0.02, 0.40, 0.5, false)
        .expect("valid second option");

    assert_eq!(first.len(), 7);
    assert_eq!(second.len(), 7);
    assert!(first.iter().all(|value| value.is_finite()));
    assert!(second.iter().all(|value| value.is_finite()));
    assert_ne!(first, second);
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

    for is_call in [false, true] {
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

            assert!(
                (prices[index] - scalar_price).abs() < 2.0e-5,
                "price mismatch at option {index}"
            );
            for greek in 0..4 {
                assert!(
                    (greeks[index * 4 + greek] - scalar_greeks[greek]).abs() < 2.0e-5,
                    "Greek {greek} mismatch at option {index}"
                );
            }
        }
    }
}

#[wasm_bindgen_test]
fn uniform_analytic_batches_preserve_degenerate_scalar_contract() {
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
    assert!(deterministic_greeks.iter().all(|value| *value == 0.0));
}

#[cfg(all(feature = "simd", target_feature = "simd128"))]
#[wasm_bindgen_test]
fn opt_in_package_selects_explicit_wasm_simd128_pricing_backend() {
    use openferric::engines::analytic::{
        BatchSimdBackend, detected_batch_simd_backend, normal_cdf_batch_approx,
    };

    assert_eq!(detected_batch_simd_backend(), BatchSimdBackend::WasmSimd128);

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
    assert!(prices.iter().all(|price| price.is_finite()));

    // Two SIMD pairs plus a scalar tail cover infinities, signed zero, and
    // NaN using the same edge semantics as the scalar CDF.
    let cdf = normal_cdf_batch_approx(&[f64::NEG_INFINITY, f64::INFINITY, -0.0, 0.0, f64::NAN]);
    assert_eq!(cdf[0], 0.0);
    assert_eq!(cdf[1], 1.0);
    assert!(cdf[2] < 0.5);
    assert!(cdf[3] > 0.5);
    assert!(cdf[4].is_nan());
}
