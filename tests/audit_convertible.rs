//! Audit regression tests for `ConvertibleBinomialEngine`.
//!
//! Covers two verified findings:
//!
//! 1. The matured-bond (`maturity <= 0`) branch used to return
//!    `min(face, call)`, so an expired bond with a call price below face
//!    redeemed below par. A matured bond must redeem at face (best of face,
//!    conversion, and the put floor); the issuer call right is meaningless at
//!    expiry.
//! 2. Model-convention pinning for the always-callable (no call-protection
//!    schedule) convention: a straight bond callable anytime at `call < face`
//!    rationally prices to PV(call), not PV(face), and the terminal nodes no
//!    longer apply the call cap (an O(dt) change only).
//!
//! Reference values are closed-form discount factors, not tree output:
//! `PV(x) = x * exp(-r * T)` for a zero-coupon claim under continuous
//! compounding (the project-wide rate convention).

use openferric::core::PricingEngine;
use openferric::engines::tree::ConvertibleBinomialEngine;
use openferric::instruments::ConvertibleBond;
use openferric::market::Market;

fn flat_market(spot: f64, rate: f64, vol: f64) -> Market {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .flat_vol(vol)
        .build()
        .unwrap()
}

#[test]
fn expired_bond_with_call_below_face_redeems_at_face() {
    // Face 100, issuer call 90, already matured. Redemption is contractual:
    // the bond pays face = 100. The pre-fix engine returned
    // min(face, call) = 90.
    let market = flat_market(100.0, 0.05, 0.20);
    let engine = ConvertibleBinomialEngine::new(0.0);

    let bond = ConvertibleBond::new(100.0, 0.05, 0.0, 0.0, Some(90.0), None);
    let price = engine.price(&bond, &market).unwrap().price;

    assert_eq!(
        price, 100.0,
        "matured bond with call < face must redeem at face, got {price}"
    );
}

#[test]
fn expired_bond_pays_best_of_face_conversion_and_put() {
    let market = flat_market(100.0, 0.05, 0.20);
    let engine = ConvertibleBinomialEngine::new(0.0);

    // Conversion value 1.2 * 100 = 120 dominates face 100 even with a low call.
    let converting = ConvertibleBond::new(100.0, 0.0, 0.0, 1.2, Some(90.0), None);
    assert_eq!(engine.price(&converting, &market).unwrap().price, 120.0);

    // Put floor 130 dominates both face and conversion at expiry.
    let puttable = ConvertibleBond::new(100.0, 0.0, 0.0, 1.2, Some(90.0), Some(130.0));
    assert_eq!(engine.price(&puttable, &market).unwrap().price, 130.0);

    // No embedded features: plain face redemption.
    let plain = ConvertibleBond::new(100.0, 0.0, 0.0, 0.0, None, None);
    assert_eq!(engine.price(&plain, &market).unwrap().price, 100.0);
}

#[test]
fn always_callable_straight_bond_converges_to_pv_of_call() {
    // Probe from the audit verification: zero-coupon straight bond
    // (conversion ratio 0), face 100, callable anytime at 90, r = 5%, T = 5.
    //
    // Under the engine's always-callable model the rational issuer calls as
    // soon as the hold value exceeds 90, so the bond is worth
    // PV(call) = 90 * exp(-0.05 * 5) = 70.09207... (closed-form anchor),
    // NOT PV(face) = 77.88. With the terminal call cap removed the finite-step
    // price approaches PV(call) from above at O(dt); at 500 steps it must lie
    // in [70.09, 70.14] (verified: 70.127125 with the cap applied only at
    // interior steps, converging to 70.093823 by 10000 steps).
    let market = flat_market(100.0, 0.05, 0.20);
    let engine = ConvertibleBinomialEngine::new(0.0).with_steps(500);

    let bond = ConvertibleBond::new(100.0, 0.0, 5.0, 0.0, Some(90.0), None);
    let price = engine.price(&bond, &market).unwrap().price;

    let pv_call = 90.0 * (-0.05_f64 * 5.0).exp();
    assert!(
        (pv_call - 1e-9..=70.14).contains(&price),
        "always-callable straight bond: expected price in [{pv_call:.6}, 70.14], got {price:.6}"
    );
}

#[test]
fn straight_bond_with_call_at_or_above_face_prices_to_pv_of_face() {
    // With call >= face the cap never binds (hold value never exceeds face
    // for a zero-coupon bond), so the price is exactly
    // PV(face) = 100 * exp(-0.05 * 5) = 77.880078... (closed-form anchor).
    // This pins that removing the terminal call cap did not perturb the
    // call >= face case at all.
    let market = flat_market(100.0, 0.05, 0.20);
    let engine = ConvertibleBinomialEngine::new(0.0).with_steps(500);

    let pv_face = 100.0 * (-0.05_f64 * 5.0).exp();

    let callable_at_par = ConvertibleBond::new(100.0, 0.0, 5.0, 0.0, Some(100.0), None);
    let callable_above_par = ConvertibleBond::new(100.0, 0.0, 5.0, 0.0, Some(110.0), None);
    let straight = ConvertibleBond::new(100.0, 0.0, 5.0, 0.0, None, None);

    for bond in [&callable_at_par, &callable_above_par, &straight] {
        let price = engine.price(bond, &market).unwrap().price;
        assert!(
            (price - pv_face).abs() < 1e-9,
            "expected PV(face) = {pv_face:.9}, got {price:.9}"
        );
    }
}

#[test]
fn callable_convertible_never_exceeds_non_callable() {
    // Monotonicity guard for the terminal-cap change: an issuer call can only
    // remove value from the holder.
    let market = flat_market(100.0, 0.05, 0.20);
    let engine = ConvertibleBinomialEngine::new(0.02).with_steps(400);

    let no_call = ConvertibleBond::new(100.0, 0.04, 7.0, 1.0, None, None);
    let with_call = ConvertibleBond::new(100.0, 0.04, 7.0, 1.0, Some(105.0), None);

    let no_call_price = engine.price(&no_call, &market).unwrap().price;
    let with_call_price = engine.price(&with_call, &market).unwrap().price;

    assert!(
        with_call_price <= no_call_price + 1e-12,
        "callable {with_call_price} must not exceed non-callable {no_call_price}"
    );
}
