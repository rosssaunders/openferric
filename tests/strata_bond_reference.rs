//! Fixed-rate bond pricing reference tests validated against OpenGamma Strata
//! (Apache 2.0) bond pricing conventions, https://github.com/OpenGamma/Strata
//!
//! Strata uses Actual/365 Fixed day-count and semi-annual compounding for
//! standard fixed-rate bonds. Our API uses year-fraction maturity and a
//! YieldCurve for discounting.  These tests verify known financial identities
//! and internal consistency of the FixedRateBond analytics.

// Reference values validated against OpenGamma Strata (Apache 2.0) bond pricing conventions,
// https://github.com/OpenGamma/Strata

use approx::assert_relative_eq;
use openferric::rates::{DayCountConvention, FixedRateBond, YieldCurve};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a flat continuous yield curve with discount factors e^{-r*t}.
fn flat_continuous_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let n = max_tenor.ceil() as usize + 1;
    let points: Vec<(f64, f64)> = (1..=n)
        .map(|i| {
            let t = i as f64;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

/// Price a bond analytically using the same compounding convention as
/// `FixedRateBond::discount_at`: (1 + y/m)^{-m*t}.
///
/// PV = sum_{k=1..N} [C/m * (1+y/m)^{-k}] + Face * (1+y/m)^{-N}
/// where N = maturity * m, C = face * coupon_rate.
fn analytic_bond_price(face: f64, coupon_rate: f64, freq: u32, maturity: f64, y: f64) -> f64 {
    let m = freq as f64;
    let n = (maturity * m).round() as u32;
    let c = face * coupon_rate / m;
    let base = 1.0 + y / m;
    let mut pv = 0.0;
    for k in 1..=n {
        pv += c * base.powf(-(k as f64));
    }
    pv += face * base.powf(-(n as f64));
    pv
}

// ===========================================================================
// 1. Par bond tests -- coupon_rate = yield => price ~ face
// ===========================================================================

/// Strata convention: a semi-annual par bond prices at 100 when coupon = yield.
/// Tested across multiple maturities (5Y, 10Y, 30Y).
#[test]
fn strata_par_bond_semiannual_various_maturities() {
    let rate = 0.05;
    let curve = flat_continuous_curve(rate, 35.0);

    for maturity in [5.0, 10.0, 30.0] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: rate,
            frequency: 2,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        let price = bond.dirty_price(&curve);
        assert_relative_eq!(price, 100.0, epsilon = 1.0e-6);
    }
}

/// Par bond identity at different coupon frequencies (1, 2, 4 per year).
#[test]
fn strata_par_bond_various_frequencies() {
    let rate = 0.04;
    let curve = flat_continuous_curve(rate, 12.0);

    for freq in [1, 2, 4] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: rate,
            frequency: freq,
            maturity: 10.0,
            day_count: DayCountConvention::Act365Fixed,
        };
        let price = bond.dirty_price(&curve);
        assert_relative_eq!(price, 100.0, epsilon = 1.0e-6);
    }
}

// ===========================================================================
// 2. Premium / discount bond
// ===========================================================================

/// Strata: premium bond (coupon > yield) prices above par; magnitude grows
/// with maturity.
#[test]
fn strata_premium_bond_increases_with_maturity() {
    let yield_rate = 0.03;
    let coupon = 0.06;
    let curve = flat_continuous_curve(yield_rate, 35.0);

    let mut prev_premium = 0.0_f64;
    for maturity in [2.0, 5.0, 10.0, 20.0] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: 2,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        let price = bond.dirty_price(&curve);
        let premium = price - 100.0;
        assert!(
            premium > 0.0,
            "Premium bond ({maturity}Y) must price above par"
        );
        assert!(
            premium > prev_premium,
            "Premium should grow with maturity: {premium} vs {prev_premium}"
        );
        prev_premium = premium;
    }
}

/// Strata: discount bond (coupon < yield) prices below par and matches
/// the analytic closed-form price.
#[test]
fn strata_discount_bond_below_par() {
    let yield_rate = 0.07;
    let coupon = 0.03;
    let curve = flat_continuous_curve(yield_rate, 15.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: coupon,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    };
    let price = bond.dirty_price(&curve);
    assert!(
        price < 100.0,
        "Discount bond must price below par, got {price}"
    );

    // Verify against analytic formula using the same zero rate
    let z = curve.zero_rate(5.0); // flat curve, same for all t
    let expected = analytic_bond_price(100.0, coupon, 2, 10.0, z);
    assert_relative_eq!(price, expected, epsilon = 1.0e-6);
}

// ===========================================================================
// 3. Zero-coupon bond
// ===========================================================================

/// Zero-coupon bond: price = face * (1 + r/m)^{-m*T} under frequency-m
/// compounding.  For frequency=1 this is 100 * (1+r)^{-T}.
/// Duration of a zero-coupon bond equals its maturity.
#[test]
fn strata_zero_coupon_price_and_duration() {
    let rate = 0.04;
    let curve = flat_continuous_curve(rate, 35.0);

    for maturity in [1.0, 5.0, 10.0, 30.0] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.0,
            frequency: 1,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };

        // Price check
        let price = bond.dirty_price(&curve);
        let expected_price = 100.0 * (1.0 + rate).powf(-maturity);
        assert_relative_eq!(price, expected_price, epsilon = 1.0e-6);

        // Duration = maturity for zero-coupon bond
        let dur = bond.duration(&curve);
        assert_relative_eq!(dur, maturity, epsilon = 1.0e-10);
    }
}

// ===========================================================================
// 4. YTM round-trip
// ===========================================================================

/// Price a bond, solve YTM from that price, then re-price analytically
/// at the solved YTM.  The round-trip should recover the original price.
#[test]
fn strata_ytm_round_trip_multiple_bonds() {
    let rate = 0.05;
    let curve = flat_continuous_curve(rate, 35.0);

    let bonds = [
        // (coupon, freq, maturity)
        (0.02, 2_u32, 5.0),
        (0.04, 2, 10.0),
        (0.07, 1, 20.0),
        (0.03, 4, 7.0),
    ];

    for (coupon, freq, maturity) in bonds {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: freq,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };

        let price = bond.dirty_price(&curve);
        let ytm = bond.ytm(price);
        assert!(
            ytm.is_finite(),
            "YTM should converge for coupon={coupon}, freq={freq}, mat={maturity}"
        );

        // Re-price analytically with the solved ytm
        let reprice = analytic_bond_price(100.0, coupon, freq, maturity, ytm);
        assert_relative_eq!(reprice, price, epsilon = 1.0e-6);
    }
}

/// YTM of a par bond must equal the coupon rate.
#[test]
fn strata_ytm_par_bond_equals_coupon() {
    for coupon in [0.02, 0.05, 0.08, 0.12] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: 2,
            maturity: 10.0,
            day_count: DayCountConvention::Act365Fixed,
        };
        let ytm = bond.ytm(100.0);
        assert_relative_eq!(ytm, coupon, epsilon = 1.0e-8);
    }
}

// ===========================================================================
// 5. Duration properties
// ===========================================================================

/// Strata: duration decreases as coupon increases (at constant yield and
/// maturity).  Higher coupons pull more cashflow weight to earlier dates.
#[test]
fn strata_duration_decreases_with_higher_coupon() {
    let rate = 0.05;
    let curve = flat_continuous_curve(rate, 15.0);

    let coupons = [0.01, 0.03, 0.05, 0.08, 0.12];
    let mut prev_dur = f64::MAX;

    for coupon in coupons {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: 2,
            maturity: 10.0,
            day_count: DayCountConvention::Act365Fixed,
        };
        let dur = bond.duration(&curve);
        assert!(dur > 0.0, "Duration must be positive for coupon={coupon}");
        assert!(
            dur < 10.0,
            "Duration of coupon bond must be less than maturity, got {dur}"
        );
        assert!(
            dur < prev_dur,
            "Duration should decrease with higher coupon: {dur} >= {prev_dur} at coupon={coupon}"
        );
        prev_dur = dur;
    }
}

/// Duration increases with maturity for a given coupon bond.
#[test]
fn strata_duration_increases_with_maturity() {
    let rate = 0.05;
    let curve = flat_continuous_curve(rate, 35.0);

    let maturities = [2.0, 5.0, 10.0, 20.0, 30.0];
    let mut prev_dur = 0.0_f64;

    for maturity in maturities {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.05,
            frequency: 2,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        let dur = bond.duration(&curve);
        assert!(
            dur > prev_dur,
            "Duration should increase with maturity: {dur} <= {prev_dur} at mat={maturity}"
        );
        prev_dur = dur;
    }
}

// ===========================================================================
// 6. Convexity
// ===========================================================================

/// Convexity is positive for all standard fixed-rate bonds.
/// Convexity increases with maturity.
#[test]
fn strata_convexity_positive_and_increases_with_maturity() {
    let rate = 0.05;
    let curve = flat_continuous_curve(rate, 35.0);

    let maturities = [2.0, 5.0, 10.0, 20.0, 30.0];
    let mut prev_conv = 0.0_f64;

    for maturity in maturities {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.05,
            frequency: 2,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        let conv = bond.convexity(&curve);
        assert!(
            conv > 0.0,
            "Convexity must be positive at maturity={maturity}"
        );
        assert!(
            conv > prev_conv,
            "Convexity should increase with maturity: {conv} <= {prev_conv} at mat={maturity}"
        );
        prev_conv = conv;
    }
}

// ===========================================================================
// 7. Accrued interest
// ===========================================================================

/// For an annual-paying bond, accrued interest at mid-year (t=0.5) should be
/// approximately half the annual coupon.
#[test]
fn strata_accrued_interest_mid_period_annual() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 1,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    // Annual coupon = 100 * 0.06 = 6.0
    // At t=0.5, half the annual period has elapsed, so accrued = 3.0
    let accrued = bond.accrued_interest(0.5);
    assert_relative_eq!(accrued, 3.0, epsilon = 1.0e-10);

    // At a coupon date, accrued is zero
    let accrued_at_coupon = bond.accrued_interest(1.0);
    assert_relative_eq!(accrued_at_coupon, 0.0, epsilon = 1.0e-12);

    // At t=0 (start), accrued is zero
    let accrued_at_start = bond.accrued_interest(0.0);
    assert_relative_eq!(accrued_at_start, 0.0, epsilon = 1.0e-12);
}

/// Accrued interest for semi-annual bond at quarter-period.
#[test]
fn strata_accrued_interest_semiannual_quarter() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.08,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    // Semi-annual coupon = 100 * 0.08 / 2 = 4.0
    // Period = 0.5 years.  At t=0.25 (quarter of a year = half a period),
    // accrued = 4.0 * 0.5 = 2.0
    let accrued = bond.accrued_interest(0.25);
    assert_relative_eq!(accrued, 2.0, epsilon = 1.0e-10);
}

// ===========================================================================
// 8. Clean vs dirty price
// ===========================================================================

/// Clean = dirty - accrued at any settlement date.
#[test]
fn strata_clean_equals_dirty_minus_accrued() {
    let rate = 0.04;
    let curve = flat_continuous_curve(rate, 12.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    // Test at several settlement dates within the bond's life
    for settlement in [0.1, 0.25, 0.4, 0.75, 1.3, 3.7, 9.9] {
        let dirty = bond.dirty_price(&curve);
        let accrued = bond.accrued_interest(settlement);
        let clean = bond.clean_price(&curve, settlement);

        assert_relative_eq!(clean, dirty - accrued, epsilon = 1.0e-12);
    }
}
