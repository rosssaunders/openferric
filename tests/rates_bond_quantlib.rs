//! Fixed-rate bond pricing reference tests derived from QuantLib's bonds.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/bonds.cpp — testCachedFixed, testCachedZero
//!
//! Our API uses exact year-fraction schedules rather than QuantLib calendar
//! dates, so the primary oracles below evaluate those cash flows directly at
//! full `f64` precision. QuantLib's calendar-based cases are provenance, not a
//! reason to admit source-rounding tolerances.
//!
//! Curve-based bond pricing discounts every cashflow with the curve's own
//! discount factors (`curve.discount_factor(t)`); discrete compounding only
//! enters through the quoted-yield path (`FixedRateBond::ytm`).

use approx::assert_relative_eq;

use openferric::rates::{DayCountConvention, FixedRateBond, YieldCurve};

#[derive(Debug, Clone, Copy)]
struct FlatBondReference {
    price: f64,
    duration: f64,
    convexity: f64,
}

/// Independently evaluate the bond's regular cash flows under
/// `DF(t) = exp(-r t)`, including the principal in the last payment.
fn flat_continuous_reference(
    face: f64,
    coupon_rate: f64,
    frequency: u32,
    maturity: f64,
    rate: f64,
) -> FlatBondReference {
    let m = frequency as f64;
    let exact_periods = maturity * m;
    let periods = exact_periods.round() as usize;
    assert_eq!(exact_periods, periods as f64, "regular coupon schedule");

    let coupon = face * coupon_rate / m;
    let mut price = 0.0;
    let mut first_moment = 0.0;
    let mut second_moment = 0.0;
    for period in 1..=periods {
        let t = period as f64 / m;
        let cashflow = coupon + if period == periods { face } else { 0.0 };
        let pv = cashflow * (-rate * t).exp();
        price += pv;
        first_moment += t * pv;
        second_moment += t * t * pv;
    }

    FlatBondReference {
        price,
        duration: first_moment / price,
        convexity: second_moment / price,
    }
}

/// Independently evaluate regular cash flows using an m-times compounded
/// quoted yield, `DF(t) = (1 + y/m)^(-m t)`.
fn quoted_yield_reference(
    face: f64,
    coupon_rate: f64,
    frequency: u32,
    maturity: f64,
    yield_rate: f64,
) -> f64 {
    let m = frequency as f64;
    let periods = (maturity * m).round() as usize;
    let coupon = face * coupon_rate / m;
    let base = 1.0 + yield_rate / m;
    (1..=periods)
        .map(|period| {
            let cashflow = coupon + if period == periods { face } else { 0.0 };
            cashflow * base.powf(-(period as f64))
        })
        .sum()
}

// Measured after comparing the independently evaluated cash-flow formulas
// above with curve interpolation: price/reprice differences peak at 1.85e-13,
// cash-flow moments are bit-exact on this grid, and YTM errors peak at 2.64e-16.
// These are roundoff/solver budgets, not rounded-source-price tolerances.
const PRICE_ROUNDOFF_BUDGET: f64 = 2.0e-13;
const MOMENT_ROUNDOFF_BUDGET: f64 = 0.0;
const YTM_SOLVER_BUDGET: f64 = 8.0e-16;

/// Build a flat continuous yield curve: DF(t) = exp(-r t).
fn flat_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let points: Vec<(f64, f64)> = (1..=(max_tenor.ceil() as usize + 1))
        .map(|i| {
            let t = i as f64;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

/// Build a flat curve whose discount factors embed m-times compounding:
/// DF(t) = (1 + r/m)^(-m t).
///
/// ln DF is linear in t, so the default log-linear interpolation reproduces
/// these discount factors exactly at every coupon date.
fn flat_compounded_curve(rate: f64, m: u32, max_tenor: f64) -> YieldCurve {
    let m = m as f64;
    let points: Vec<(f64, f64)> = (1..=(max_tenor.ceil() as usize + 1))
        .map(|i| {
            let t = i as f64;
            (t, (1.0 + rate / m).powf(-m * t))
        })
        .collect();
    YieldCurve::new(points)
}

// ── Fixed-coupon bond pricing ───────────────────────────────────────────────

/// A par bond (coupon = yield) should price at face value.
/// Reference: fundamental fixed-income identity.
///
/// The par identity requires the curve to discount with the same compounding
/// as the bond's coupon frequency, i.e. DF(t) = (1 + c/m)^(-m t).
#[test]
fn fixed_coupon_bond_par_pricing() {
    let rate = 0.05;

    for (coupon, freq, maturity) in [
        (0.05, 2, 2.0),
        (0.05, 2, 5.0),
        (0.05, 2, 10.0),
        (0.05, 1, 5.0),
        (0.05, 4, 5.0),
    ] {
        let curve = flat_compounded_curve(rate, freq, 12.0);
        let bond = FixedRateBond {
            face_value: 100.0,
            coupon_rate: coupon,
            frequency: freq,
            maturity,
            day_count: DayCountConvention::Act365Fixed,
        };

        let expected = quoted_yield_reference(100.0, coupon, freq, maturity, rate);
        assert_relative_eq!(expected, 100.0, epsilon = PRICE_ROUNDOFF_BUDGET);
        assert_relative_eq!(
            bond.dirty_price(&curve),
            expected,
            epsilon = PRICE_ROUNDOFF_BUDGET,
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
    let premium_expected = flat_continuous_reference(100.0, 0.08, 2, 5.0, yield_rate).price;
    assert_relative_eq!(
        premium.dirty_price(&curve),
        premium_expected,
        epsilon = PRICE_ROUNDOFF_BUDGET,
    );
    assert!(
        premium_expected > 100.0,
        "premium reference must exceed par"
    );

    // Discount bond: 2% coupon on 5% curve
    let discount = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.02,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };
    let discount_expected = flat_continuous_reference(100.0, 0.02, 2, 5.0, yield_rate).price;
    assert_relative_eq!(
        discount.dirty_price(&curve),
        discount_expected,
        epsilon = PRICE_ROUNDOFF_BUDGET,
    );
    assert!(
        discount_expected < 100.0,
        "discount reference must be below par"
    );
}

/// Cached reference values for fixed-coupon bonds on a flat 5% continuous curve.
/// Computed via closed-form PV of coupon annuity + principal with the curve's
/// own (continuous) discount factors:
///
/// PV = C/m * sum_{k=1..n*m} exp(-r * k/m) + 100 * exp(-r * n)
///
/// With q = exp(-r/m), the coupon annuity is q (1 - q^{n m}) / (1 - q).
/// (a 5% annual bond is below par on a 5% continuous curve, since the
/// annually-compounded equivalent yield e^{0.05} - 1 = 5.127% exceeds 5%).
#[test]
fn fixed_coupon_bond_cached_values() {
    let rate = 0.05;
    let curve = flat_curve(rate, 35.0);

    struct BondCase {
        coupon_rate: f64,
        frequency: u32,
        maturity: f64,
    }

    let cases = vec![
        // 8% semiannual 10Y on 5% flat continuous curve
        BondCase {
            coupon_rate: 0.08,
            frequency: 2,
            maturity: 10.0,
        },
        // 2% semiannual 10Y on 5% flat continuous curve
        BondCase {
            coupon_rate: 0.02,
            frequency: 2,
            maturity: 10.0,
        },
        // 5% annual 30Y on 5% flat continuous curve
        BondCase {
            coupon_rate: 0.05,
            frequency: 1,
            maturity: 30.0,
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
        let expected =
            flat_continuous_reference(100.0, case.coupon_rate, case.frequency, case.maturity, rate)
                .price;
        assert_relative_eq!(
            bond.dirty_price(&curve),
            expected,
            epsilon = PRICE_ROUNDOFF_BUDGET,
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

        // Curve-based pricing discounts with the curve's own discount factor,
        // so on a flat continuous curve price = 100 * exp(-r * t).
        let price = bond.dirty_price(&curve);
        let expected = 100.0 * (-rate * maturity).exp();
        assert_relative_eq!(price, expected, epsilon = PRICE_ROUNDOFF_BUDGET,);
        // Must be below par for positive rates
        assert!(
            price < 100.0,
            "Zero-coupon bond must price below par for r > 0"
        );
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
        assert_relative_eq!(ytm, coupon, epsilon = YTM_SOLVER_BUDGET,);
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

    let expected_price = flat_continuous_reference(100.0, 0.04, 2, 10.0, rate).price;
    let expected_ytm = 2.0 * ((rate / 2.0).exp() - 1.0);
    assert_relative_eq!(price, expected_price, epsilon = PRICE_ROUNDOFF_BUDGET);
    assert_relative_eq!(ytm, expected_ytm, epsilon = YTM_SOLVER_BUDGET);

    // Re-price on a curve embedding the YTM's own compounding convention,
    // DF(t) = (1 + y/m)^(-m t): the round-trip must recover the price exactly.
    let ytm_curve = flat_compounded_curve(ytm, 2, 12.0);
    let reprice = bond.dirty_price(&ytm_curve);

    let direct_reprice = quoted_yield_reference(100.0, 0.04, 2, 10.0, ytm);
    assert_relative_eq!(direct_reprice, price, epsilon = PRICE_ROUNDOFF_BUDGET);
    assert_relative_eq!(reprice, price, epsilon = PRICE_ROUNDOFF_BUDGET);
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
            epsilon = MOMENT_ROUNDOFF_BUDGET,
        );
    }
}

/// Duration and convexity equal the first two discounted cash-flow moments.
#[test]
fn duration_and_convexity_match_discounted_cashflow_moments() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    let expected = flat_continuous_reference(100.0, 0.06, 2, 10.0, rate);
    let duration = bond.duration(&curve);
    let convexity = bond.convexity(&curve);
    assert_relative_eq!(
        duration,
        expected.duration,
        epsilon = MOMENT_ROUNDOFF_BUDGET,
    );
    assert_relative_eq!(
        convexity,
        expected.convexity,
        epsilon = MOMENT_ROUNDOFF_BUDGET,
    );
    assert!(duration > 0.0 && duration < 10.0);
    assert!(convexity > 0.0);
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
    assert_eq!(bond.accrued_interest(0.5), 0.0);
    assert_eq!(bond.accrued_interest(1.0), 0.0);

    // Settlement at mid-period → half a coupon
    let half_coupon = 100.0 * 0.06 / 2.0 * 0.5; // = 1.5
    assert_eq!(bond.accrued_interest(0.25), half_coupon);
}

/// Clean price = dirty price at settlement - accrued interest, with the
/// settlement dirty price anchored to a hand computation.
///
/// The dirty price at settlement `s` is the value AT `s` of the cashflows
/// paid strictly after `s`: sum_{t > s} CF(t) * DF(0, t) / DF(0, s).
///
/// Anchor (hand-computed): 5y 6% semiannual bond, face 100, flat 5%
/// continuously-compounded curve, s = 0.25:
///   dirty(0.25)   = sum_{t in {0.5, 1.0, .., 4.5}} 3 * exp(-0.05 (t - 0.25))
///                   + 103 * exp(-0.05 (5 - 0.25))  = 105.402903895
///   accrued(0.25) = 3 * 0.25 / 0.5                 = 1.5
///   clean(0.25)   = 105.402903895 - 1.5            = 103.902903895
/// (The pre-fix implementation discounted every cashflow to t=0 instead of
/// to settlement: 104.093568 dirty / 102.593568 clean.)
#[test]
fn clean_price_equals_settlement_dirty_minus_accrued() {
    let rate = 0.05;
    // Quarter-year node spacing so both the 0.25 settlement and the
    // half-year coupon grid lie inside the node range and log-linear
    // interpolation reproduces exp(-r t) exactly (no extrapolation).
    let points: Vec<(f64, f64)> = (1..=21)
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-rate * t).exp())
        })
        .collect();
    let curve = YieldCurve::new(points);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 5.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    let settlement = 0.25;
    let dirty = bond.dirty_price_at(&curve, settlement);
    let accrued = bond.accrued_interest(settlement);
    let clean = bond.clean_price(&curve, settlement);

    let expected_dirty =
        flat_continuous_reference(100.0, 0.06, 2, 5.0, rate).price / (-rate * settlement).exp();
    let expected_accrued = 1.5;
    let expected_clean = expected_dirty - expected_accrued;

    assert_relative_eq!(dirty, expected_dirty, epsilon = PRICE_ROUNDOFF_BUDGET);
    assert_eq!(accrued, expected_accrued);
    assert_relative_eq!(clean, expected_clean, epsilon = PRICE_ROUNDOFF_BUDGET);

    // Settlement 0 degenerates to today's dirty price (accrued(0) = 0).
    assert_relative_eq!(
        bond.clean_price(&curve, 0.0),
        bond.dirty_price(&curve),
        epsilon = PRICE_ROUNDOFF_BUDGET,
    );
}
