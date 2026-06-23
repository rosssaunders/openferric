use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn pricing_success_path_returns_values() {
    let price = crate::pricing::bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true)
        .expect("valid Black-Scholes input should price");
    assert!((price - 10.4506).abs() < 0.01);

    let greeks = crate::pricing::bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true)
        .expect("valid Greeks input should price");
    assert_eq!(greeks.len(), 7);
}

#[wasm_bindgen_test]
fn batch_length_mismatch_returns_js_error() {
    let result = crate::pricing::bs_price_batch_wasm(
        &[100.0, 101.0],
        &[100.0],
        &[0.05, 0.05],
        &[0.0, 0.0],
        &[0.20, 0.20],
        &[1.0, 1.0],
        &[1, 0],
    );
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn invalid_scalar_inputs_return_js_errors() {
    assert!(
        crate::pricing::barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "sideways", true,)
            .is_err()
    );
    assert!(crate::pricing::bond_price(1000.0, 0.05, 10.0, 0.05, 0).is_err());
    assert!(crate::risk::var_historical(&[], 0.95).is_err());
    assert!(crate::credit::cds_fair_spread(1_000_000.0, 5.0, 1.2, 0.02, 0.03).is_err());
}

#[wasm_bindgen_test]
fn payoff_and_heston_reject_invalid_shapes() {
    assert!(
        crate::payoff::strategy_intrinsic_pnl_wasm(
            &[90.0, 100.0, 110.0],
            &[100.0, 110.0],
            &[1.0],
            &[1, 0],
            5.0,
        )
        .is_err()
    );

    assert!(
        crate::heston::heston_price(
            100.0, 100.0, 0.01, 0.0, 0.04, 4.0, 0.25, 1.0, -1.2, 1.0, true,
        )
        .is_err()
    );
}
