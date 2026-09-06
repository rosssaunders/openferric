use approx::assert_relative_eq;
use openferric::instruments::{
    CallableRateNote, CouponScheduleBuilder, ExerciseSchedule, SnowballNote,
};
use openferric::models::HullWhite;
use openferric::rates::{Frequency, YieldCurve};

#[test]
fn fixed_note_notice_preserves_contractual_call_payment_date() {
    let note = CallableRateNote {
        notional: 100.0,
        redemption: 1000.0,
        call_price: 100.0,
        maturity: 2.0,
        coupon_schedule: CouponScheduleBuilder::new(0.0, 2.0, Frequency::Annual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap(),
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 0.5).unwrap(),
    };
    let curve = YieldCurve::new(vec![(2.0, (-0.1_f64).exp())]);
    let actual = note
        .price_hull_white_tree(&HullWhite::new(0.1, 0.0), &curve, 40)
        .unwrap();
    assert_relative_eq!(actual, 100.0 * (-0.05_f64).exp(), epsilon = 2.0e-8);
}

#[test]
fn noncallable_floating_note_reduces_to_par_floater() {
    let note = CallableRateNote {
        notional: 100.0,
        redemption: 100.0,
        call_price: 1.0e10,
        maturity: 1.0,
        coupon_schedule: CouponScheduleBuilder::new(0.0, 1.0, Frequency::SemiAnnual)
            .unwrap()
            .build_floating(0.0, None, None)
            .unwrap(),
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 0.0).unwrap(),
    };
    let curve = YieldCurve::new(vec![(2.0, (-0.1_f64).exp())]);
    let actual = note
        .price_hull_white_tree(&HullWhite::new(0.1, 0.0), &curve, 40)
        .unwrap();
    assert_relative_eq!(actual, 100.0, epsilon = 2.0e-8);
    let sloped = YieldCurve::new(vec![(0.5, 0.99), (1.0, 0.92), (2.0, 0.80)]);
    assert_relative_eq!(
        note.price_hull_white_tree(&HullWhite::new(0.1, 0.0), &sloped, 40)
            .unwrap(),
        100.0,
        epsilon = 1.0e-12
    );
    let with_notice = CallableRateNote {
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 0.5).unwrap(),
        ..note
    };
    assert!(
        with_notice
            .price_hull_white_tree(&HullWhite::new(0.1, 0.01), &sloped, 40)
            .is_err()
    );
}

#[test]
fn snowball_rejects_nan_rates_before_payoff_flooring() {
    let note = SnowballNote {
        notional: 100.0,
        redemption: 100.0,
        initial_coupon: 0.05,
        spread: 0.01,
        floor: None,
        cap: None,
        coupon_schedule: CouponScheduleBuilder::new(0.0, 1.0, Frequency::SemiAnnual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap(),
    };
    let curve = YieldCurve::new(vec![(1.0, 0.95)]);
    assert!(note.price(&[f64::NAN, 0.02], &curve).is_err());
    let reversed = SnowballNote {
        coupon_schedule: note.coupon_schedule.iter().rev().cloned().collect(),
        ..note
    };
    assert!(reversed.price(&[0.01, 0.02], &curve).is_err());
}

#[test]
fn quarterly_range_accrual_uses_coupon_period_not_payment_lag() {
    use openferric::instruments::{DualRangeAccrual, RangeAccrual};
    use openferric::pricing::range_accrual::{dual_range_accrual_mc_price, range_accrual_mc_price};

    let single = RangeAccrual {
        notional: 100.0,
        coupon_rate: 0.08,
        lower_bound: 0.02,
        upper_bound: 0.06,
        accrual_factor: 0.25,
        fixing_times: vec![0.25],
        payment_time: 0.5,
    };
    let dual = DualRangeAccrual {
        notional: 100.0,
        coupon_rate: 0.08,
        lower_bound: 0.01,
        upper_bound: 0.03,
        accrual_factor: 0.25,
        fixing_times: vec![0.25],
        payment_time: 0.5,
    };
    let expected = 2.0 * (-0.015_f64).exp();
    let actual = range_accrual_mc_price(&single, 0.04, 0.1, 0.04, 0.0, 0.03, 1, 7).unwrap();
    assert_relative_eq!(actual.price, expected, epsilon = 1.0e-14);
    let actual = dual_range_accrual_mc_price(
        &dual, 0.05, 0.03, 0.1, 0.05, 0.0, 0.1, 0.03, 0.0, 0.0, 0.03, 1, 7,
    )
    .unwrap();
    assert_relative_eq!(actual.price, expected, epsilon = 1.0e-14);
    for invalid_factor in [0.0, -0.25, f64::NAN, f64::INFINITY] {
        assert!(
            RangeAccrual {
                accrual_factor: invalid_factor,
                ..single.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            DualRangeAccrual {
                accrual_factor: invalid_factor,
                ..dual.clone()
            }
            .validate()
            .is_err()
        );
    }
    assert!(range_accrual_mc_price(&single, f64::NAN, 0.1, 0.04, 0.0, 0.03, 1, 7).is_err());
    assert!(
        dual_range_accrual_mc_price(
            &dual,
            0.05,
            0.03,
            0.1,
            0.05,
            0.0,
            0.1,
            0.03,
            0.0,
            f64::NAN,
            0.03,
            1,
            7
        )
        .is_err()
    );
}

#[test]
fn stochastic_tree_discounts_notice_settlement_and_intervening_fixed_coupon() {
    let note = CallableRateNote {
        notional: 100.0,
        redemption: 1.0e9,
        call_price: 100.0,
        maturity: 2.0,
        coupon_schedule: CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
            .unwrap()
            .build_fixed(0.06)
            .unwrap(),
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 1.0).unwrap(),
    };
    let curve = YieldCurve::new(vec![(2.0, (-0.1_f64).exp())]);
    let actual = note
        .price_hull_white_tree(&HullWhite::new(0.1, 0.01), &curve, 40)
        .unwrap();
    let expected = 100.0 * (-0.05_f64).exp() + 3.0 * (-0.025_f64).exp();
    assert_relative_eq!(actual, expected, epsilon = 2.0e-8);
}
