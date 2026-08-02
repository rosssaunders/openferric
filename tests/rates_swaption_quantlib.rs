//! European swaption pricing reference tests derived from QuantLib's swaptions.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/swaptions.cpp — testCachedValue, testStrikeDependence
//!
//! Our API uses year-fraction tenor and Black-76 model. QuantLib tests use
//! calendar-based schedules with Actual/365(Fixed). Compatible year-fraction
//! targets below are evaluated independently; price allowances cover only
//! binary64 arithmetic and normal-CDF implementation roundoff.

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

fn ulp(value: f64) -> f64 {
    let magnitude = value.abs();
    magnitude.next_up() - magnitude
}

fn assert_scipy_reference(actual: f64, expected: f64, case: &str) {
    // SciPy 1.17.1 `scipy.special.ndtr` fixtures were generated from a separate
    // Python Black-76 implementation that explicitly builds the annual
    // fixed-leg cashflows.  512 ULP covers cross-language libm operation order;
    // it is below 2e-8 currency units even for the largest case.
    let numerical_budget = 512.0 * ulp(expected.abs().max(1.0));
    assert!(
        (actual - expected).abs() <= numerical_budget,
        "{case}: actual={actual:.17}, reference={expected:.17}, budget={numerical_budget:e}"
    );
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
    assert!((fwd - expected_forward).abs() <= 32.0 * ulp(expected_forward));

    let annuity = swaption.annuity_factor(&curve);
    assert!((annuity - expected_annuity).abs() <= 32.0 * ulp(expected_annuity));

    // The 256-ULP allowance covers the independently implemented normal CDF,
    // the ten-term annuity sum, and Black-76 arithmetic. It is roughly 2e-9
    // currency units, not an economic price tolerance.
    let expected_price_roundoff = 256.0 * ulp(expected_price);
    assert!((price - expected_price).abs() <= expected_price_roundoff);
    let cached_reference = 36_279.649_346_017_74;
    assert!((price - cached_reference).abs() <= 256.0 * ulp(cached_reference));
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

    let scipy_atm_reference = 54_219.469_523_109_04;
    assert_scipy_reference(price, scipy_atm_reference, "ATM payer");
    assert_scipy_reference(recv_price, scipy_atm_reference, "ATM receiver");

    assert!((price - recv_price).abs() <= 8.0 * ulp(price.max(recv_price)));
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

    let scipy_grid = [
        (0.03, 132_934.188_434_577_2, 5_802.285_530_339_885),
        (0.04, 89_026.765_481_541_86, 21_662.305_052_688_92),
        (0.05, 57_428.825_217_066_06, 49_831.807_263_597_43),
        (0.06, 36_279.649_346_017_74, 88_450.073_867_933_37),
        (0.07, 22_702.865_492_423_374, 134_640.732_489_723_33),
        (0.08, 14_177.144_339_561_07, 185_882.453_812_245_34),
    ];
    let mut prev_payer = f64::MAX;
    let mut prev_receiver = 0.0_f64;

    for &(k, expected_payer, expected_receiver) in &scipy_grid {
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

        assert_scipy_reference(payer_price, expected_payer, &format!("payer strike {k}"));
        assert_scipy_reference(
            recv_price,
            expected_receiver,
            &format!("receiver strike {k}"),
        );

        // Supplemental shape checks retain the QuantLib strike-dependence
        // regression after every point has been pinned to an exact oracle.
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

        // Propagate a bounded number of rounding units through the two option
        // prices and their cancellation in put-call parity.
        let operation_scale = payer_price
            .abs()
            .max(recv_price.abs())
            .max(parity_rhs.abs());
        let parity_roundoff = 64.0 * ulp(operation_scale);
        assert!((parity_lhs - parity_rhs).abs() <= parity_roundoff);
    }
}

// ── Implied vol round-trip ──────────────────────────────────────────────────

/// Recover known vols from independently generated Black-76 market prices.
#[test]
fn swaption_implied_vol_round_trip() {
    let rate = 0.05;
    let curve = flat_curve(rate, 20.0);

    let scipy_market_prices = [
        (0.10, 0.04, 71_417.756_990_846_36),
        (0.20, 0.05, 57_428.825_217_066_06),
        (0.30, 0.06, 63_556.283_455_518_01),
        (0.40, 0.07, 76_110.462_671_506_51),
    ];
    for (vol, strike, market_price) in scipy_market_prices {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike,
            option_expiry: 5.0,
            swap_tenor: 10.0,
            is_payer: true,
        };

        // Do not feed the solver a price generated by the implementation under
        // test: this is a frozen SciPy Black-76 market-price oracle.
        let recovered_vol = swaption.implied_vol(market_price, &curve);

        assert_relative_eq!(recovered_vol, vol, epsilon = 2.0e-10);
    }
}

// ── Tenor dependence ────────────────────────────────────────────────────────

/// Longer option expiry (more time value) → higher swaption price, all else equal.
#[test]
fn swaption_value_increases_with_expiry() {
    let rate = 0.05;
    let vol = 0.20;
    let curve = flat_curve(rate, 35.0);

    let scipy_grid = [
        (1.0, 5_591.330_729_456_484),
        (2.0, 11_033.052_920_139_333),
        (5.0, 20_395.566_322_707_975),
        (10.0, 25_780.413_375_191_47),
    ];
    let mut prev_price = 0.0;

    for &(expiry, expected) in &scipy_grid {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.06,
            option_expiry: expiry,
            swap_tenor: 5.0,
            is_payer: true,
        };
        let price = swaption.price(&curve, vol);
        assert_scipy_reference(price, expected, &format!("expiry {expiry}"));
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

    let scipy_grid = [
        (2.0, 8_774.419_633_294_889),
        (5.0, 20_395.566_322_707_975),
        (10.0, 36_279.649_346_017_74),
        (20.0, 58_284.368_998_000_86),
    ];
    let mut prev_price = 0.0;

    for &(tenor, expected) in &scipy_grid {
        let swaption = Swaption {
            notional: 1_000_000.0,
            strike: 0.06,
            option_expiry: 5.0,
            swap_tenor: tenor,
            is_payer: true,
        };
        let price = swaption.price(&curve, vol);
        assert_scipy_reference(price, expected, &format!("tenor {tenor}"));
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

    let scipy_grid = [
        (0.05, 1_339.083_720_997_969),
        (0.10, 10_498.226_548_693_827),
        (0.20, 36_279.649_346_017_74),
        (0.30, 63_556.283_455_518_01),
        (0.50, 116_102.086_793_826_69),
    ];
    let mut prev_price = 0.0;

    for &(vol, expected) in &scipy_grid {
        let price = swaption.price(&curve, vol);
        assert_scipy_reference(price, expected, &format!("volatility {vol}"));
        assert!(
            price > prev_price,
            "Price must increase with vol: vol={vol}, price={price} <= prev={prev_price}"
        );
        prev_price = price;
    }
}
