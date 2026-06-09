use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{DayCountConvention, FixedRateBond, YieldCurve, year_fraction};

#[test]
fn day_count_known_values() {
    let start = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    let end = NaiveDate::from_ymd_opt(2024, 7, 15).unwrap();

    assert_relative_eq!(
        year_fraction(start, end, DayCountConvention::Act360),
        182.0 / 360.0,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        year_fraction(start, end, DayCountConvention::Act365Fixed),
        182.0 / 365.0,
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        year_fraction(start, end, DayCountConvention::Thirty360),
        180.0 / 360.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn yield_curve_flat_five_percent_discount_factor() {
    let r = 0.05_f64;
    let curve = YieldCurve::new(vec![
        (0.5, (-r * 0.5).exp()),
        (1.0, (-r * 1.0).exp()),
        (2.0, (-r * 2.0).exp()),
        (5.0, (-r * 5.0).exp()),
    ]);

    for t in [0.25, 0.75, 1.5, 3.0, 7.0] {
        assert_relative_eq!(curve.discount_factor(t), (-r * t).exp(), epsilon = 1.0e-12);
    }
}

#[test]
fn fixed_rate_bond_flat_curve_prices_at_par() {
    // Bond cashflows discount with the curve's own discount factors, so the
    // par identity (coupon bond at face value) holds when the curve DFs embed
    // the bond's compounding convention: DF(t) = (1 + c/m)^(-m t).
    let r = 0.05_f64;
    let m = 2.0_f64;
    let df = |t: f64| (1.0 + r / m).powf(-m * t);
    let curve = YieldCurve::new(vec![
        (0.5, df(0.5)),
        (1.0, df(1.0)),
        (1.5, df(1.5)),
        (2.0, df(2.0)),
    ]);

    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.05,
        frequency: 2,
        maturity: 2.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    assert_relative_eq!(bond.dirty_price(&curve), 100.0, epsilon = 1.0e-10);
}

#[test]
fn ytm_of_par_bond_equals_coupon_rate() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.05,
        frequency: 2,
        maturity: 2.0,
        day_count: DayCountConvention::Act365Fixed,
    };

    assert_relative_eq!(bond.ytm(100.0), 0.05, epsilon = 1.0e-10);
}
