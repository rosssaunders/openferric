//! European swaption pricing reference tests derived from QuantLib's swaptions.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/swaptions.cpp — testCachedValue, testStrikeDependence
//!
//! Our API uses year-fraction tenor and Black-76 model. QuantLib tests use
//! calendar-based schedules with Actual/365(Fixed). Tolerances are adjusted
//! to account for the simplified year-fraction approach.

use approx::assert_relative_eq;

use openferric::rates::{Swaption, YieldCurve};
use statrs::distribution::{ContinuousCDF, Normal};

/// Build a flat continuous yield curve.
fn flat_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let n = max_tenor.ceil() as usize + 1;
    let points: Vec<(f64, f64)> = (1..=n)
        .map(|i| {
            let t = i as f64;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

// ── Cached value tests ──────────────────────────────────────────────────────

/// Reference: QuantLib swaptions.cpp testCachedValue.
/// Setup: 5% flat forward curve, 20% Black vol.
/// Payer swaption, 5Y exercise into 10Y swap.
///
/// QuantLib's test uses the same Black swaption methodology but a full dated
/// schedule. This test evaluates the exact Black-76 value for OpenFerric's
/// documented annual year-fraction schedule, using `statrs` as the independent
/// normal-CDF implementation.
#[test]
fn swaption_cached_value_payer_5y_into_10y() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 20.0);

    let swaption = Swaption {
        notional: 1_000_000.0,
        strike: 0.06,
        option_expiry: 5.0,
        swap_tenor: 10.0,
        is_payer: true,
    };

    let price = swaption.price(&curve, vol);

    let expected_annuity: f64 = (6..=15).map(|t| (-rate * t as f64).exp()).sum();
    let expected_forward = rate.exp() - 1.0;
    let sigma_sqrt_t = vol * swaption.option_expiry.sqrt();
    let d1 = ((expected_forward / swaption.strike).ln() + 0.5 * vol * vol * swaption.option_expiry)
        / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let expected_price = swaption.notional
        * expected_annuity
        * (expected_forward * normal.cdf(d1) - swaption.strike * normal.cdf(d2));

    let fwd = swaption.forward_swap_rate(&curve);
    assert_relative_eq!(fwd, expected_forward, epsilon = 1.0e-12);

    let annuity = swaption.annuity_factor(&curve);
    assert_relative_eq!(annuity, expected_annuity, epsilon = 1.0e-12);
    assert_relative_eq!(price, expected_price, epsilon = 1.0e-8);
    assert_relative_eq!(price, 36_279.649_346_017_74, epsilon = 1.0e-8);
}

/// ATM swaption: strike = forward rate.
#[test]
fn swaption_atm_price() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 20.0);

    let fwd = {
        let tmp = Swaption {
            notional: 1.0,
            strike: 0.05,
            option_expiry: 5.0,
            swap_tenor: 10.0,
            is_payer: true,
        };
        tmp.forward_swap_rate(&curve)
    };

    let swaption = Swaption {
        notional: 1_000_000.0,
        strike: fwd,
        option_expiry: 5.0,
        swap_tenor: 10.0,
        is_payer: true,
    };

    let price = swaption.price(&curve, vol);

    // ATM payer and receiver swaptions should have equal value (put-call parity)
    let receiver = Swaption {
        is_payer: false,
        ..swaption
    };
    let recv_price = receiver.price(&curve, vol);

    assert_relative_eq!(price, recv_price, epsilon = 1.0e-8);
}

// ── Strike dependence ───────────────────────────────────────────────────────

/// Reference: QuantLib swaptions.cpp testStrikeDependence.
/// Payer swaption value decreases as strike increases (farther OTM).
/// Receiver swaption value increases as strike increases (farther ITM).
#[test]
fn swaption_payer_value_decreases_with_strike() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 20.0);

    let strikes = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08];
    let mut prev_payer = f64::MAX;
    let mut prev_receiver = 0.0_f64;

    for &k in &strikes {
        let payer = Swaption {
            notional: 1_000_000.0,
            strike: k,
            option_expiry: 5.0,
            swap_tenor: 10.0,
            is_payer: true,
        };
        let receiver = Swaption {
            is_payer: false,
            ..payer
        };

        let payer_price = payer.price(&curve, vol);
        let recv_price = receiver.price(&curve, vol);

        assert!(
            payer_price < prev_payer,
            "Payer swaption price must decrease with strike: K={k}, price={payer_price} >= prev={prev_payer}"
        );
        assert!(
            recv_price >= prev_receiver,
            "Receiver swaption price must increase with strike: K={k}"
        );

        prev_payer = payer_price;
        prev_receiver = recv_price;
    }
}

// ── Put-call parity ─────────────────────────────────────────────────────────

/// Payer - Receiver = Notional * Annuity * (Forward - Strike)
/// This is the swaption put-call parity.
#[test]
fn swaption_put_call_parity() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 20.0);

    for strike in [0.03, 0.04, 0.05, 0.06, 0.07] {
        let payer = Swaption {
            notional: 1_000_000.0,
            strike,
            option_expiry: 5.0,
            swap_tenor: 10.0,
            is_payer: true,
        };
        let receiver = Swaption {
            is_payer: false,
            ..payer
        };

        let payer_price = payer.price(&curve, vol);
        let recv_price = receiver.price(&curve, vol);
        let annuity = payer.annuity_factor(&curve);
        let fwd = payer.forward_swap_rate(&curve);

        let parity_rhs = payer.notional * annuity * (fwd - strike);
        let parity_lhs = payer_price - recv_price;

        assert_relative_eq!(parity_lhs, parity_rhs, epsilon = 1.0e-8);
    }
}

// ── Implied vol round-trip ──────────────────────────────────────────────────

/// Price at known vol, then recover vol via implied_vol.
#[test]
fn swaption_implied_vol_round_trip() {
    let rate = 0.05;
    let curve = flat_curve(rate, 20.0);

    for (vol, strike) in [(0.10, 0.04), (0.20, 0.05), (0.30, 0.06), (0.40, 0.07)] {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike,
            option_expiry: 5.0,
            swap_tenor: 10.0,
            is_payer: true,
        };

        let price = swaption.price(&curve, vol);
        let recovered_vol = swaption.implied_vol(price, &curve);

        assert_relative_eq!(recovered_vol, vol, epsilon = 1.0e-10);
    }
}

// ── Tenor dependence ────────────────────────────────────────────────────────

/// Longer option expiry (more time value) → higher swaption price, all else equal.
#[test]
fn swaption_value_increases_with_expiry() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 35.0);

    let expiries = [1.0, 2.0, 5.0, 10.0];
    let mut prev_price = 0.0;

    for &expiry in &expiries {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.06,
            option_expiry: expiry,
            swap_tenor: 5.0,
            is_payer: true,
        };
        let price = swaption.price(&curve, vol);
        assert!(
            price > prev_price,
            "Price must increase with expiry: T={expiry}, price={price} <= prev={prev_price}"
        );
        prev_price = price;
    }
}

/// Longer swap tenor → larger annuity → higher swaption price.
#[test]
fn swaption_value_increases_with_swap_tenor() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 35.0);

    let tenors = [2.0, 5.0, 10.0, 20.0];
    let mut prev_price = 0.0;

    for &tenor in &tenors {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.06,
            option_expiry: 5.0,
            swap_tenor: tenor,
            is_payer: true,
        };
        let price = swaption.price(&curve, vol);
        assert!(
            price > prev_price,
            "Price must increase with tenor: swap_tenor={tenor}, price={price} <= prev={prev_price}"
        );
        prev_price = price;
    }
}

// ── Vol dependence ──────────────────────────────────────────────────────────

/// Higher vol → higher swaption price.
#[test]
fn swaption_value_increases_with_vol() {
    let rate = 0.05;
    let curve = flat_curve(rate, 20.0);

    let swaption = Swaption {
        notional: 1_000_000.0,
        strike: 0.06,
        option_expiry: 5.0,
        swap_tenor: 10.0,
        is_payer: true,
    };

    let vols = [0.05, 0.10, 0.20, 0.30, 0.50];
    let mut prev_price = 0.0;

    for &vol in &vols {
        let price = swaption.price(&curve, vol);
        assert!(
            price > prev_price,
            "Price must increase with vol: vol={vol}, price={price} <= prev={prev_price}"
        );
        prev_price = price;
    }
}
