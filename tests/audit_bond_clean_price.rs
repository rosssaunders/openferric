//! Regression tests for `FixedRateBond` settlement-aware clean/dirty pricing.
//!
//! Audit finding: `clean_price(curve, s)` used to subtract accrued-at-`s`
//! from the t=0 present value of ALL cashflows -- including coupons already
//! paid before `s` -- and never forward-valued to the settlement date. For a
//! 10y bond at s = 3.7 that included 7 stale coupons and discounted to t=0.
//!
//! Correct semantics tested here:
//!   dirty(s) = sum over cashflows with t > s of CF(t) * DF(0, t) / DF(0, s)
//!   clean(s) = dirty(s) - accrued(s)
//!
//! All reference numbers below are hand-computed from first principles on
//! flat continuously-compounded curves (DF(t) = exp(-r t)), where the
//! forward discount factor is exp(-r (t - s)). The curves are built with
//! node spacing such that every time used is inside the node range; ln DF is
//! linear in t, so the default log-linear interpolation reproduces exp(-r t)
//! exactly and 1e-9 epsilons are safe.

use approx::assert_relative_eq;
use openferric::rates::{DayCountConvention, FixedRateBond, YieldCurve};

/// Flat continuously-compounded curve with `step`-spaced nodes out to
/// `max_tenor`: DF(t) = exp(-r t) exactly at (and, via log-linear
/// interpolation, between) every node.
fn flat_cc_curve(rate: f64, step: f64, max_tenor: f64) -> YieldCurve {
    let n = (max_tenor / step).ceil() as usize + 1;
    let points: Vec<(f64, f64)> = (1..=n)
        .map(|i| {
            let t = i as f64 * step;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

fn ten_year_semiannual_bond() -> FixedRateBond {
    FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 10.0,
        day_count: DayCountConvention::Act365Fixed,
    }
}

/// Anchor 1: 10y 6% semiannual bond, face 100, flat 4% cc curve, s = 3.7.
///
/// Hand computation (double-precision sum over the remaining cashflows):
///   dirty(3.7)   = sum_{t in {4.0, 4.5, ..., 9.5}} 3 * exp(-0.04 (t - 3.7))
///                  + 103 * exp(-0.04 (10 - 3.7))    = 111.997548830
///   accrued(3.7) = 3 * (3.7 - 3.5) / 0.5            = 1.2
///   clean(3.7)   = dirty - accrued                  = 110.797548830
///
/// The pre-fix implementation returned dirty = 115.991126156 (t=0 PV of all
/// 20 coupons, including the 7 paid at t = 0.5..3.5) and clean =
/// 114.791126156 -- an error of +3.99 per 100 face.
#[test]
fn ten_year_bond_settlement_anchor() {
    let curve = flat_cc_curve(0.04, 0.5, 12.0);
    let bond = ten_year_semiannual_bond();

    let dirty = bond.dirty_price_at(&curve, 3.7);
    let accrued = bond.accrued_interest(3.7);
    let clean = bond.clean_price(&curve, 3.7);

    assert_relative_eq!(dirty, 111.997548830, epsilon = 1.0e-9);
    assert_relative_eq!(accrued, 1.2, epsilon = 1.0e-12);
    assert_relative_eq!(clean, 110.797548830, epsilon = 1.0e-9);

    // Guard against the old bug resurfacing: the buggy values were several
    // points away from the correct ones.
    assert!((dirty - 115.991126156).abs() > 1.0);
    assert!((clean - 114.791126156).abs() > 1.0);
}

/// Anchor 2: 2y 5% annual bond, face 100, flat 4% cc curve, s = 0.5.
///
/// Hand computation:
///   dirty(0.5)   = 5 * exp(-0.04 * 0.5) + 105 * exp(-0.04 * 1.5)
///                = 4.900993347 + 98.885276046      = 103.786269393
///   accrued(0.5) = 5 * 0.5                          = 2.5
///   clean(0.5)   = dirty - accrued                  = 101.286269393
///
/// The pre-fix implementation returned clean = 99.231164 (t=0 discounting
/// understates here: error sign varies with the coupon/discount mix).
#[test]
fn two_year_annual_bond_settlement_anchor() {
    // Half-year node spacing so s = 0.5 and both coupon dates are nodes.
    let curve = flat_cc_curve(0.04, 0.5, 3.0);
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.05,
        frequency: 1,
        maturity: 2.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    let dirty = bond.dirty_price_at(&curve, 0.5);
    let accrued = bond.accrued_interest(0.5);
    let clean = bond.clean_price(&curve, 0.5);

    assert_relative_eq!(dirty, 103.786269393, epsilon = 1.0e-9);
    assert_relative_eq!(accrued, 2.5, epsilon = 1.0e-12);
    assert_relative_eq!(clean, 101.286269393, epsilon = 1.0e-9);
}

/// Settlement 0 must be bitwise-identical to the historical behavior:
/// clean(0) == dirty_price_at(0) == dirty_price() and accrued(0) == 0.
#[test]
fn settlement_zero_regression_guard() {
    let curve = flat_cc_curve(0.04, 0.5, 12.0);
    let bond = ten_year_semiannual_bond();

    let dirty0 = bond.dirty_price(&curve);
    assert_eq!(bond.dirty_price_at(&curve, 0.0), dirty0);
    assert_eq!(bond.accrued_interest(0.0), 0.0);
    assert_eq!(bond.clean_price(&curve, 0.0), dirty0);
    // Negative settlement is clamped to the same degenerate case.
    assert_eq!(bond.dirty_price_at(&curve, -1.0), dirty0);
}

/// A settlement exactly on a coupon date excludes the just-paid coupon.
///
/// Hand computation (s = 3.5): remaining cashflows are the 13 coupons of 3.0
/// at t = 4.0..10.0 plus 100 redemption at 10.0, each forward-valued by
/// exp(-0.04 (t - 3.5)):
///   dirty(3.5) = 111.105142823, accrued(3.5) = 0, clean = dirty.
/// Including the just-paid t = 3.5 coupon would add exactly 3.0.
#[test]
fn settlement_on_coupon_date_excludes_paid_coupon() {
    let curve = flat_cc_curve(0.04, 0.5, 12.0);
    let bond = ten_year_semiannual_bond();

    let dirty = bond.dirty_price_at(&curve, 3.5);
    assert_relative_eq!(dirty, 111.105142823, epsilon = 1.0e-9);
    assert_relative_eq!(bond.accrued_interest(3.5), 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(bond.clean_price(&curve, 3.5), dirty, epsilon = 1.0e-12);
}

/// The settlement dirty price obeys the forward-value identity on any curve:
/// dirty(s) * DF(0, s) == t=0 PV of the cashflows strictly after s.
/// Checked on a non-flat (upward-sloping) curve to make sure the DF-ratio
/// forward valuation is used rather than any flat-rate shortcut.
#[test]
fn forward_value_identity_on_sloped_curve() {
    // Upward-sloping zero curve: r(t) = 0.02 + 0.004 t at half-year nodes.
    let points: Vec<(f64, f64)> = (1..=24)
        .map(|i| {
            let t = i as f64 * 0.5;
            let r = 0.02 + 0.004 * t;
            (t, (-r * t).exp())
        })
        .collect();
    let curve = YieldCurve::new(points);
    let bond = ten_year_semiannual_bond();

    for settlement in [1.3, 3.7, 6.25, 9.9] {
        // t=0 PV of cashflows strictly after settlement, computed directly
        // from node discount factors (all coupon times are nodes).
        let mut pv0 = 0.0;
        let mut t = 0.5;
        while t < 10.0 - 1.0e-12 {
            if t > settlement + 1.0e-12 {
                pv0 += 3.0 * curve.discount_factor(t);
            }
            t += 0.5;
        }
        pv0 += 103.0 * curve.discount_factor(10.0);

        let dirty = bond.dirty_price_at(&curve, settlement);
        assert_relative_eq!(
            dirty * curve.discount_factor(settlement),
            pv0,
            epsilon = 1.0e-9
        );

        // clean = dirty - accrued always holds against the settlement-aware
        // dirty price (the identity the old code faked with a t=0 PV).
        assert_relative_eq!(
            bond.clean_price(&curve, settlement),
            dirty - bond.accrued_interest(settlement),
            epsilon = 1.0e-12
        );
    }
}

/// After the final cashflow there is nothing left to price.
#[test]
fn settlement_at_or_after_maturity_prices_to_zero() {
    let curve = flat_cc_curve(0.04, 0.5, 12.0);
    let bond = ten_year_semiannual_bond();

    assert_eq!(bond.dirty_price_at(&curve, 10.0), 0.0);
    assert_eq!(bond.dirty_price_at(&curve, 10.5), 0.0);
    assert_eq!(bond.clean_price(&curve, 10.0), 0.0);
    assert_eq!(bond.clean_price(&curve, 10.5), 0.0);
}
