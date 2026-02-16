use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::credit::{
    Cds, CdsDateRule, CdsIndex, DatedCds, GaussianCopula, IsdaConventions, NthToDefaultBasket,
    ProtectionSide, SurvivalCurve, first_to_default_spread_copula, hazard_from_par_spread,
    price_isda_flat, price_midpoint_flat,
};
use openferric::rates::YieldCurve;

#[test]
fn quantlib_cached_value_midpoint_regression() {
    // QuantLib creditdefaultswap.cpp testCachedValue() reference setup.
    // Historical QuantLib test versions use issueDate = evaluationDate - 1Y.
    let evaluation_date = NaiveDate::from_ymd_opt(2006, 6, 9).unwrap();
    let issue_date = NaiveDate::from_ymd_opt(2005, 6, 9).unwrap();
    let maturity_date = NaiveDate::from_ymd_opt(2015, 6, 9).unwrap();

    let cds = DatedCds {
        side: ProtectionSide::Seller,
        notional: 10_000.0,
        running_spread: 0.0120,
        recovery_rate: 0.40,
        issue_date,
        maturity_date,
        coupon_interval_months: 6,
        date_rule: CdsDateRule::TwentiethImm,
    };

    let result = price_midpoint_flat(
        &cds,
        evaluation_date,
        0.01234,
        0.06,
        IsdaConventions {
            step_in_days: 1,
            cash_settle_days: 1,
        },
    );

    let expected_npv = 295.0153398;
    let expected_fair_spread = 0.007517539081;

    assert_relative_eq!(result.clean_npv, expected_npv, epsilon = 6.0e-1);
    assert_relative_eq!(result.fair_spread, expected_fair_spread, epsilon = 5.0e-7);
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
    let result = price_isda_flat(
        &cds,
        valuation_date,
        hazard,
        0.05,
        IsdaConventions::default(),
    );

    assert_relative_eq!(result.clean_npv, 0.0, epsilon = 3.0e4);
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
