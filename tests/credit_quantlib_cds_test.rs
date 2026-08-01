use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::credit::{
    Cds, CdsDateRule, CdsIndex, DatedCds, GaussianCopula, IsdaConventions, NthToDefaultBasket,
    ProtectionSide, SurvivalCurve, first_to_default_spread_copula, hazard_from_par_spread,
    price_isda_flat, price_midpoint_flat,
};
use openferric::rates::YieldCurve;

#[test]
fn quantlib_midpoint_one_coupon_regression() {
    // QuantLib-Python 1.43 MidPointCdsEngine fixture with a NullCalendar,
    // explicit 20-Dec-2025/20-Mar-2026 schedule, continuously compounded
    // Act/360 flat curves (h=1.234%, r=6%), and same-day protection start and
    // settlement. The original QuantLib cached test uses a TARGET-generated
    // Forward schedule that this compatibility API does not claim to model.
    let evaluation_date = NaiveDate::from_ymd_opt(2025, 12, 20).unwrap();
    let cds = DatedCds {
        side: ProtectionSide::Seller,
        notional: 10_000.0,
        running_spread: 0.0120,
        recovery_rate: 0.40,
        issue_date: evaluation_date,
        maturity_date: NaiveDate::from_ymd_opt(2026, 3, 20).unwrap(),
        coupon_interval_months: 3,
        date_rule: CdsDateRule::QuarterlyImm,
    };

    let result = price_midpoint_flat(
        &cds,
        evaluation_date,
        0.01234,
        0.06,
        IsdaConventions {
            step_in_days: 0,
            cash_settle_days: 0,
        },
    );

    let leg_roundoff = 512.0 * f64::EPSILON * 30.0;
    assert!((result.clean_npv - 11.164_799_954_192_627).abs() <= leg_roundoff);
    assert!((result.dirty_npv - 11.164_799_954_192_627).abs() <= leg_roundoff);
    assert!((result.premium_leg_pv - 29.508_185_029_241_762).abs() <= leg_roundoff);
    assert!((result.protection_leg_pv - 18.343_385_075_049_135).abs() <= leg_roundoff);
    assert_eq!(result.accrued_premium_pv, 0.0);
    assert!(
        (result.fair_spread - 0.007_459_646_219_598_271).abs()
            <= 256.0 * f64::EPSILON * result.fair_spread
    );
}

#[test]
fn isda_standard_par_cds_is_near_zero() {
    // QuantLib ISDA engine style par-CDS consistency check.
    let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
    let notional = 10_000_000.0;
    let running_spread = 0.01;
    let recovery = 0.40;

    let cds = DatedCds::standard_imm(
        ProtectionSide::Buyer,
        valuation_date,
        5,
        notional,
        running_spread,
        recovery,
    );

    let hazard = hazard_from_par_spread(running_spread, recovery);
    let probe = price_isda_flat(
        &cds,
        valuation_date,
        hazard,
        0.05,
        IsdaConventions::default(),
    );
    let par_cds = DatedCds {
        running_spread: probe.fair_spread,
        ..cds
    };
    let result = price_isda_flat(
        &par_cds,
        valuation_date,
        hazard,
        0.05,
        IsdaConventions::default(),
    );

    assert_relative_eq!(result.clean_npv, 0.0, epsilon = 1.0e-8);
}

#[test]
fn cds_index_and_first_to_default_behaviour() {
    let discount_rate = 0.03;
    let discount_curve = YieldCurve::new(
        (1..=80)
            .map(|i| {
                let t = i as f64 * 0.25;
                (t, (-discount_rate * t).exp())
            })
            .collect(),
    );

    let recovery = 0.4;
    let single_spread = 0.01;
    let hazard = hazard_from_par_spread(single_spread, recovery);
    let curve = SurvivalCurve::from_piecewise_hazard(&[10.0], &[hazard]);

    let single_name = Cds {
        notional: 1.0,
        spread: single_spread,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };

    let index = CdsIndex {
        constituents: vec![single_name.clone(); 5],
        weights: vec![1.0; 5],
    };
    let curves = vec![curve.clone(); 5];

    let index_spread = index.fair_spread(&discount_curve, &curves);
    let name_spread = single_name.fair_spread(&discount_curve, &curve);

    assert_relative_eq!(index_spread, name_spread, epsilon = 1.0e-12);

    let ftd = first_to_default_spread_copula(
        1.0,
        5.0,
        recovery,
        4,
        &discount_curve,
        &curves,
        &GaussianCopula::new(0.25),
        20_000,
        42,
    );

    assert!(ftd > name_spread);

    let basket = NthToDefaultBasket {
        n: 1,
        notional: 1.0,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };
    let ntd_spread = basket.fair_spread(&discount_curve, &curves);
    assert!(ntd_spread > name_spread);
}
