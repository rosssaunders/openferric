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
    // PV(call) = 90 * exp(-0.05 * 5) = 70.09207... (continuous-exercise
    // limit), NOT PV(face) = 77.88.  On the finite 500-step convention the
    // terminal face is capped to 90 one step before maturity, then discounted
    // through the remaining 499 intervals.  This gives an exact finite-grid
    // target rather than a broad interval around the limiting value.
    let market = flat_market(100.0, 0.05, 0.20);
    let steps = 500_usize;
    let engine = ConvertibleBinomialEngine::new(0.0).with_steps(steps);

    let bond = ConvertibleBond::new(100.0, 0.0, 5.0, 0.0, Some(90.0), None);
    let price = engine.price(&bond, &market).unwrap().price;

    let finite_grid_reference = 90.0 * (-0.05_f64 * 5.0 * (steps - 1) as f64 / steps as f64).exp();
    // Backward induction and the closed form order binary64 operations
    // differently.  The measured difference is 42 scaled epsilons; 128
    // scaled epsilons is an accumulated-roundoff budget, not a price band.
    let roundoff = 128.0 * f64::EPSILON * finite_grid_reference;
    assert!(
        (price - finite_grid_reference).abs() <= roundoff,
        "always-callable straight bond: expected finite-grid price \
         {finite_grid_reference:.17e}, got {price:.17e}, binary64 budget={roundoff:.3e}"
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

#[test]
fn fully_featured_convertible_matches_independent_decimal_crr_grid() {
    // Full non-degenerate contract: coupon-bearing, convertible, issuer
    // callable, holder puttable, dividend-paying equity, nonzero volatility,
    // and a nonzero credit spread.  Across the tree the call, put, conversion,
    // and continuation branches all bind.
    //
    // The finite-grid reference was generated independently with Python's
    // 60-digit `decimal` arithmetic.  Its recurrence materialized every tree
    // level and applied
    //
    //   max(conversion, put, min(call,
    //       exp(-(r+s)dt) * (p V_up + (1-p)V_down + face*coupon*dt)))
    //
    // at interior nodes, with max(face, conversion, put) at maturity.  No
    // OpenFerric code or output entered the calculation.
    let market = Market::builder()
        .spot(100.0)
        .rate(0.04)
        .dividend_yield(0.015)
        .flat_vol(0.25)
        .build()
        .unwrap();
    let bond = ConvertibleBond::new(100.0, 0.04, 5.0, 1.0, Some(120.0), Some(110.0));

    let finite_grid_price = ConvertibleBinomialEngine::new(0.015)
        .with_steps(512)
        .price(&bond, &market)
        .unwrap()
        .price;
    const DECIMAL_512_STEP_REFERENCE: f64 = 114.386_876_176_185_11;
    const BINARY64_ACCUMULATION_BUDGET: f64 = 5.0e-10;
    assert!(
        (finite_grid_price - DECIMAL_512_STEP_REFERENCE).abs() <= BINARY64_ACCUMULATION_BUDGET,
        "fully featured convertible finite-grid price: \
         actual={finite_grid_price:.17e}, \
         independent decimal reference={DECIMAL_512_STEP_REFERENCE:.17e}, \
         roundoff budget={BINARY64_ACCUMULATION_BUDGET:.3e}"
    );

    // CRR exercise boundaries produce the familiar even/odd grid ripple.  A
    // 512-to-1024 refinement changes this 114.39 price by 0.04297 (4.3 bp of
    // notional); the explicit 0.0431 budget is discretization error and is
    // deliberately kept separate from the exact finite-grid assertion above.
    let refined_price = ConvertibleBinomialEngine::new(0.015)
        .with_steps(1_024)
        .price(&bond, &market)
        .unwrap()
        .price;
    const GRID_512_TO_1024_BUDGET: f64 = 4.31e-2;
    assert!(
        (refined_price - finite_grid_price).abs() <= GRID_512_TO_1024_BUDGET,
        "fully featured convertible grid change: p512={finite_grid_price:.12}, \
         p1024={refined_price:.12}, budget={GRID_512_TO_1024_BUDGET:.3e}"
    );
}
