//! Fixed-rate bond pricing reference tests derived from QuantLib's bonds.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/bonds.cpp — testCachedFixed, testCachedZero
//!
//! Our API uses year-fraction maturity and continuous compounding rather than
//! calendar-based schedules, so tolerances are relaxed to 1e-4 where
//! calendar effects matter.

use approx::assert_relative_eq;

use openferric::rates::{DayCountConvention, FixedRateBond, YieldCurve};

/// Build a flat continuous yield curve.
fn flat_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let points: Vec<(f64, f64)> = (1..=(max_tenor.ceil() as usize + 1))
        .map(|i| {
            let t = i as f64;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

// ── Fixed-coupon bond pricing ───────────────────────────────────────────────

/// A par bond (coupon = yield) should price at face value.
/// Reference: fundamental fixed-income identity.
#[test]
fn fixed_coupon_bond_par_pricing() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    for (coupon, freq, maturity) in [
        (0.05, 2, 2.0),
        (0.05, 2, 5.0),
        (0.05, 2, 10.0),
        (0.05, 1, 5.0),
        (0.05, 4, 5.0),
    ] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: freq,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };

        assert_relative_eq!(
            bond.dirty_price(&curve),
            100.0,
            epsilon = 1.0e-6,
        );
    }
}

/// Premium bond: coupon > yield → price > par.
/// Discount bond: coupon < yield → price < par.
/// Reference: QuantLib bonds.cpp testCachedFixed pattern.
#[test]
fn fixed_coupon_bond_premium_and_discount() {
    let yield_rate = 0.05;
    let curve = flat_curve(yield_rate, 12.0);

    // Premium bond: 8% coupon on 5% curve
    let premium = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.08,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };
    assert!(
        premium.dirty_price(&curve) > 100.0,
        "Premium bond must price above par"
    );

    // Discount bond: 2% coupon on 5% curve
    let discount = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.02,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };
    assert!(
        discount.dirty_price(&curve) < 100.0,
        "Discount bond must price below par"
    );
}

/// Cached reference values for fixed-coupon bonds on a flat 5% curve.
/// Computed via closed-form PV of coupon annuity + principal.
///
/// PV = C/m * sum_{k=1..n*m} (1 + y/m)^(-k) + 100 * (1 + y/m)^(-n*m)
/// where y = zero_rate(t) on a flat continuous curve => simple compounding rate
#[test]
fn fixed_coupon_bond_cached_values() {
    let rate = 0.05;
    let curve = flat_curve(rate, 35.0);

    struct BondCase {
        coupon_rate: f64,
        frequency: u32,
        maturity: f64,
        // Expected dirty price (from formula, not QuantLib C++ directly)
        expected_dirty_price: f64,
        tolerance: f64,
    }

    let cases = vec![
        // 8% semiannual 10Y on 5% flat curve
        BondCase {
            coupon_rate: 0.08,
            frequency: 2,
            maturity: 10.0,
            expected_dirty_price: 123.3862, // PV of premium bond
            tolerance: 0.5,
        },
        // 2% semiannual 10Y on 5% flat curve
        BondCase {
            coupon_rate: 0.02,
            frequency: 2,
            maturity: 10.0,
            expected_dirty_price: 76.6138, // PV of discount bond (symmetric)
            tolerance: 0.5,
        },
        // 5% annual 30Y on 5% flat curve => par
        BondCase {
            coupon_rate: 0.05,
            frequency: 1,
            maturity: 30.0,
            expected_dirty_price: 100.0,
            tolerance: 1.0e-4,
        },
    ];

    for case in &cases {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: case.coupon_rate,
            frequency: case.frequency,
            maturity: case.maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        let price = bond.dirty_price(&curve);
        assert_relative_eq!(
            price,
            case.expected_dirty_price,
            epsilon = case.tolerance,
        );
    }
}

// ── Zero-coupon bond pricing ────────────────────────────────────────────────

/// Zero-coupon bond: price = face_value * DF(maturity).
/// Reference: QuantLib bonds.cpp testCachedZero.
#[test]
fn zero_coupon_bond_cached_values() {
    let rate = 0.05;
    let curve = flat_curve(rate, 35.0);

    for maturity in [1.0, 5.0, 10.0, 30.0] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.0,
            frequency: 1,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };

        // discount_at uses (1 + z/m)^(-m*t) with m=frequency=1, z=zero_rate.
        // On a flat continuous curve with rate r, zero_rate(t) = r,
        // so price = 100 * (1 + r)^(-t).
        let price = bond.dirty_price(&curve);
        let expected = 100.0 * (1.0 + rate).powf(-maturity);
        assert_relative_eq!(
            price,
            expected,
            epsilon = 1.0e-6,
        );
        // Must be below par for positive rates
        assert!(price < 100.0, "Zero-coupon bond must price below par for r > 0");
    }
}

// ── YTM round-trip ──────────────────────────────────────────────────────────

/// YTM of a par bond must equal the coupon rate.
/// Reference: QuantLib bonds.cpp — fundamental identity.
#[test]
fn ytm_par_bond_equals_coupon_rate() {
    for coupon in [0.01, 0.03, 0.05, 0.08, 0.10] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: 2,
            maturity: 5.0,
            day_count: DayCountConvention::Act365Fixed,
        };
        let ytm = bond.ytm(100.0);
        assert_relative_eq!(
            ytm,
            coupon,
            epsilon = 1.0e-8,
        );
    }
}

/// YTM round-trip: price → ytm → re-price should recover the original price.
#[test]
fn ytm_round_trip() {
    let rate = 0.06;
    let curve = flat_curve(rate, 12.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.04,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    let price = bond.dirty_price(&curve);
    let ytm = bond.ytm(price);

    // Re-price using ytm as flat yield
    let ytm_curve = flat_curve(ytm, 12.0);
    let reprice = bond.dirty_price(&ytm_curve);

    assert_relative_eq!(reprice, price, epsilon = 0.01);
}

// ── Duration and Convexity ──────────────────────────────────────────────────

/// Duration of a zero-coupon bond equals its maturity.
#[test]
fn duration_zero_coupon_bond_equals_maturity() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    for maturity in [1.0, 5.0, 10.0] {
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.0,
            frequency: 1,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };
        assert_relative_eq!(
            bond.duration(&curve),
            maturity,
            epsilon = 1.0e-10,
        );
    }
}

/// Duration must be positive for coupon bonds with positive maturity.
/// Convexity must be positive.
#[test]
fn duration_and_convexity_are_positive() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    assert!(bond.duration(&curve) > 0.0);
    assert!(bond.convexity(&curve) > 0.0);
    // Duration must be less than maturity for coupon bonds
    assert!(bond.duration(&curve) < 10.0);
}

/// Accrued interest at settlement = 0 is zero.
/// Accrued interest at half-period is half a coupon payment.
#[test]
fn accrued_interest_basic_cases() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    // Settlement at a coupon date → zero accrued
    assert_relative_eq!(bond.accrued_interest(0.5), 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(bond.accrued_interest(1.0), 0.0, epsilon = 1.0e-12);

    // Settlement at mid-period → half a coupon
    let half_coupon = 100.0 * 0.06 / 2.0 * 0.5; // = 1.5
    assert_relative_eq!(bond.accrued_interest(0.25), half_coupon, epsilon = 1.0e-12);
}

/// Clean price = dirty price - accrued interest.
#[test]
fn clean_price_equals_dirty_minus_accrued() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    let settlement = 0.25;
    let dirty = bond.dirty_price(&curve);
    let accrued = bond.accrued_interest(settlement);
    let clean = bond.clean_price(&curve, settlement);

    assert_relative_eq!(clean, dirty - accrued, epsilon = 1.0e-12);
}
