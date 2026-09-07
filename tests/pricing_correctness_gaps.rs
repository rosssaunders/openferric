//! Independent cashflow, Gaussian and QuantLib checks for the pricing audit.
//! QuantLib 1.43 callable fixture: FlatForward(2025-01-02,.03,Actual365Fixed),
//! HullWhite(.1,.02), 110*P(0,2)-110*discountBondOption(Call,100/110,1,2).

use chrono::NaiveDate;
use openferric::core::{DiagKey, PricingEngine};
use openferric::engines::analytic::BlackScholesEngine;
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::instruments::*;
use openferric::math::normal_cdf;
use openferric::models::HullWhite;
use openferric::rates::{
    DayCountConvention, ForwardRateAgreement, Frequency, Swaption, YieldCurve,
};

fn curve(rate: f64) -> YieldCurve {
    YieldCurve::new(vec![(30.0, (-rate * 30.0).exp())])
}

/// SciPy 1.17.1 adaptive integration of the expiry-forward Gaussian payoff,
/// split at its independently solved exercise boundary; quadrature error <8e-14.
#[test]
fn negative_rate_stub_swaption_matches_independent_gaussian_integral() {
    let discount_curve = curve(-0.015);
    let model = HullWhite::new(0.1, 0.01);
    let payer = Swaption {
        notional: 100.0,
        strike: -0.02,
        option_expiry: 1.0,
        swap_tenor: 2.5,
        is_payer: true,
    };
    let price = payer.price_hull_white(&discount_curve, &model).unwrap();
    assert!((price - 1.682_397_693_379_852).abs() < 2.0e-12);
    let receiver = Swaption {
        is_payer: false,
        ..payer
    };
    let swap_value = 100.0
        * (discount_curve.discount_factor(1.0) - discount_curve.discount_factor(3.5)
            + 0.02
                * (discount_curve.discount_factor(2.0)
                    + discount_curve.discount_factor(3.0)
                    + 0.5 * discount_curve.discount_factor(3.5)));
    assert!(
        (price - receiver.price_hull_white(&discount_curve, &model).unwrap() - swap_value).abs()
            < 2.0e-12
    );
    assert_eq!(
        Swaption {
            strike: -2.0,
            ..receiver
        }
        .price_hull_white(&discount_curve, &model)
        .unwrap(),
        0.0
    );
}

/// SciPy 1.17.1 Sobol, 2^20 points in eight Gaussian dimensions, scrambles
/// 314159/271828. Exact OU/integral covariance, four semiannual payments,
/// flat 3%, a=.1, sigma=.03; neither reference calls OpenFerric.
#[test]
fn nonlinear_rate_notes_match_independent_sobol_cashflow_references() {
    let schedule = CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
        .unwrap()
        .build_floating(0.0, None, None)
        .unwrap();
    let tarn = TargetRedemptionNote {
        notional: 100.0,
        redemption: 100.0,
        target_coupon: 5.0,
        spread: 0.01,
        floor: Some(0.0),
        cap: Some(0.12),
        coupon_schedule: schedule.clone(),
    };
    let snowball = SnowballNote {
        notional: 100.0,
        redemption: 100.0,
        initial_coupon: 0.06,
        spread: 0.02,
        floor: Some(0.0),
        cap: Some(0.12),
        coupon_schedule: schedule,
    };
    let model = HullWhite::new(0.1, 0.03);
    let results = [
        tarn.price_hull_white_mc(
            &model,
            &curve(0.03),
            &RateNoteHistory::default(),
            100_000,
            77,
        )
        .unwrap(),
        snowball
            .price_hull_white_mc(
                &model,
                &curve(0.03),
                &RateNoteHistory::default(),
                100_000,
                77,
            )
            .unwrap(),
    ];
    for (result, references) in results.iter().zip([
        [100.650_161_256_956_31_f64, 100.650_157_354_743_87],
        [102.058_189_490_471_3, 102.058_193_979_169_97],
    ]) {
        let reference = 0.5 * (references[0] + references[1]);
        let reference_error = (references[0] - references[1]).abs();
        assert!(
            (result.price - reference).abs() < 4.0 * result.stderr.unwrap() + reference_error,
            "price={}, reference={reference}, stderr={:?}",
            result.price,
            result.stderr
        );
    }
}

#[test]
fn callable_note_matches_quantlib_bond_option_not_shared_tree_primitives() {
    let note = CallableRateNote {
        notional: 100.0,
        redemption: 100.0,
        call_price: 100.0,
        maturity: 2.0,
        coupon_schedule: vec![CouponPeriod {
            start_time: 0.0,
            end_time: 2.0,
            payment_time: 2.0,
            coupon: CouponType::Fixed { rate: 0.05 },
        }],
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 0.0).unwrap(),
    };
    let model = HullWhite::new(0.1, 0.02);
    let coarse = note
        .price_hull_white_tree(&model, &curve(0.03), 300)
        .unwrap();
    let fine = note
        .price_hull_white_tree(&model, &curve(0.03), 1200)
        .unwrap();
    let reference = 97.044_483_574_357_24;
    assert!(
        (fine - reference).abs() < 1.0e-4,
        "fine={fine:.14}, reference={reference:.14}"
    );
    assert!((fine - reference).abs() < (coarse - reference).abs());
}

#[test]
fn daily_range_accrual_matches_payment_measure_gaussian_marginals() {
    let note = CallableRangeAccrualNote::new(
        100.0,
        1.0,
        Frequency::Annual,
        0.08,
        0.01,
        0.02,
        0.045,
        f64::MAX,
        ExerciseSchedule::new(vec![1.0], 0.0).unwrap(),
    )
    .unwrap();
    let reversion = 0.1;
    let volatility = 0.02;
    let response = |time: f64| -(-reversion * time).exp_m1() / reversion;
    let mut expected_coupon = 0.0;
    for day in 0..365 {
        let time = day as f64 / 365.0;
        let variance =
            volatility * volatility * -(-2.0 * reversion * time).exp_m1() / (2.0 * reversion);
        let mean = 0.03 - variance * response(1.0 - time);
        let probability = if day == 0 {
            1.0
        } else {
            normal_cdf((0.045 - mean) / variance.sqrt())
                - normal_cdf((0.02 - mean) / variance.sqrt())
        };
        expected_coupon += 100.0 / 365.0 * (0.01 + 0.07 * probability);
    }
    let reference = (100.0 + expected_coupon) * (-0.03_f64).exp();
    let price = note
        .price_hull_white_tree(&HullWhite::new(reversion, volatility), &curve(0.03), 1460)
        .unwrap();
    assert!(
        (price - reference).abs() < 0.01,
        "daily price={price:.12}, Gaussian={reference:.12}"
    );
    assert!((price - 108.0 * (-0.03_f64).exp()).abs() > 0.5);
}

#[test]
fn seasoned_coupon_is_paid_even_if_issuer_calls_at_valuation() {
    let note = CallableRateNote {
        notional: 100.0,
        redemption: 100.0,
        call_price: 90.0,
        maturity: 1.0,
        coupon_schedule: CouponScheduleBuilder::new(0.0, 1.0, Frequency::Annual)
            .unwrap()
            .build_floating(0.01, None, None)
            .unwrap(),
        exercise_schedule: ExerciseSchedule::new(vec![0.5], 0.0).unwrap(),
    };
    let history = RateNoteHistory {
        valuation_time: 0.5,
        fixings: vec![(0.0, 0.04)],
    };
    let expected = 90.0 + 5.0 * (-0.03_f64 * 0.5).exp();
    for volatility in [0.0, 0.02] {
        let actual = note
            .price_hull_white_tree_with_history(
                &HullWhite::new(0.1, volatility),
                &curve(0.03),
                200,
                &history,
            )
            .unwrap();
        assert!((actual - expected).abs() < 1.0e-9);
    }
    assert!(
        note.price_hull_white_tree_with_history(
            &HullWhite::new(0.1, 0.01),
            &curve(0.03),
            200,
            &RateNoteHistory {
                valuation_time: 0.5,
                fixings: vec![]
            }
        )
        .is_err()
    );
}

#[test]
fn seasoned_range_coupon_preserves_each_daily_observation_before_call() {
    let maturity = 60.0 / 365.0;
    let valuation = 15.0 / 365.0;
    let note = CallableRangeAccrualNote::new(
        100.0,
        maturity,
        Frequency::Annual,
        0.12,
        0.0,
        0.02,
        0.04,
        90.0,
        ExerciseSchedule::new(vec![valuation], 0.0).unwrap(),
    )
    .unwrap();
    let history = RateNoteHistory {
        valuation_time: valuation,
        fixings: (0..15)
            .map(|day| (day as f64 / 365.0, if day % 2 == 0 { 0.03 } else { 0.06 }))
            .collect(),
    };
    let price = note
        .note
        .price_hull_white_tree_with_history(&HullWhite::new(0.1, 0.02), &curve(0.03), 120, &history)
        .unwrap();
    let expected = 90.0 + 100.0 * 0.12 * 8.0 / 365.0 * (-0.03 * (maturity - valuation)).exp();
    assert!((price - expected).abs() < 1.0e-12);
}

#[test]
fn seasoned_fra_uses_fixing_and_explicit_settlement_order() {
    let date = |month, day| NaiveDate::from_ymd_opt(2025, month, day).unwrap();
    let mut fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.04,
        valuation_date: date(4, 1),
        start_date: date(1, 1),
        end_date: date(7, 1),
        day_count: DayCountConvention::Act360,
        curve_day_count: DayCountConvention::Act365Fixed,
    };
    let expected = 1_000_000.0 * (0.06 - 0.04) * 181.0 / 360.0 * (-0.03_f64 * 91.0 / 365.0).exp();
    assert!(
        (fra.npv_with_fixing(&curve(0.03), Some(0.06), false, false)
            .unwrap()
            - expected)
            .abs()
            < 1.0e-10
    );
    assert_eq!(
        fra.npv_with_fixing(&curve(0.03), None, true, false)
            .unwrap(),
        0.0
    );
    assert!(
        fra.npv_with_fixing(&curve(0.03), None, false, false)
            .is_err()
    );
    fra.valuation_date = fra.start_date;
    let amount = 1_000_000.0 * (0.06 - 0.04) * 181.0 / 360.0 / (1.0 + 0.06 * 181.0 / 360.0);
    assert!(
        (fra.npv_with_fixing(&curve(0.03), Some(0.06), true, true)
            .unwrap()
            - amount)
            .abs()
            < 1.0e-10
    );
    assert_eq!(
        fra.npv_with_fixing(&curve(0.03), Some(0.06), true, false)
            .unwrap(),
        0.0
    );
}

#[test]
fn historical_barrier_hits_cannot_be_forgotten_after_spot_recovers() {
    let market = openferric::market::Market::builder()
        .spot(100.0)
        .rate(0.03)
        .flat_vol(0.2)
        .build()
        .unwrap();
    let knock_in = BarrierOption::builder()
        .put()
        .strike(100.0)
        .expiry(1.0)
        .down_and_in(80.0)
        .build()
        .unwrap();
    let expected = BlackScholesEngine
        .price(&VanillaOption::european_put(100.0, 1.0), &market)
        .unwrap();
    assert_eq!(
        knock_in.price_with_history(&market, true).unwrap(),
        expected
    );
    let knock_out = BarrierOption::builder()
        .put()
        .strike(100.0)
        .expiry(1.0)
        .down_and_out(80.0)
        .rebate(2.0)
        .build()
        .unwrap();
    assert_eq!(
        knock_out.price_with_history(&market, true).unwrap().price,
        0.0
    );
}

#[test]
fn lsm_reports_separate_training_and_evaluation_counts() {
    let market = openferric::market::Market::builder()
        .spot(100.0)
        .rate(0.05)
        .flat_vol(0.2)
        .build()
        .unwrap();
    let engine = LongstaffSchwartzEngine::new(20_000, 50, 77);
    let result = engine
        .price(&VanillaOption::american_put(100.0, 1.0), &market)
        .unwrap();
    assert_eq!(
        result.diagnostics.get(DiagKey::TrainingPaths.as_str()),
        Some(&20_000.0)
    );
    assert_eq!(
        result.diagnostics.get(DiagKey::NumPaths.as_str()),
        Some(&20_000.0)
    );
    assert!((result.price - 6.0896).abs() < 4.0 * result.stderr.unwrap() + 0.05);
}

#[test]
fn jamshidian_price_matches_single_exercise_tree_with_stub() {
    let swaption = Swaption {
        notional: 100.0,
        strike: 0.03,
        option_expiry: 1.0,
        swap_tenor: 2.5,
        is_payer: true,
    };
    let model = HullWhite::new(0.1, 0.01);
    let exact = swaption.price_hull_white(&curve(0.03), &model).unwrap();
    let tree = openferric::engines::tree::BermudanSwaptionEngine::new(model, 1200).price(
        &swaption,
        &[1.0],
        &curve(0.03),
    );
    assert!((tree - exact).abs() < 0.0003, "tree={tree}, exact={exact}");
}

#[test]
fn stochastic_mbs_reduces_to_discounted_known_cashflows_and_recovers_oas() {
    let pool = MbsPassThrough {
        original_balance: 100.0,
        coupon_rate: 0.06,
        servicing_fee: 0.005,
        original_term: 24,
        age: 0,
        prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
    };
    let config = MbsHullWhiteConfig {
        num_paths: 100,
        seed: 21,
        refinancing_tenor: 5.0,
        refinancing_spread: 0.02,
        refinancing_floor: 0.001,
    };
    let prepayment = RateIncentivePrepayment::default();
    let rate = 12.0 * (0.03_f64 / 12.0).exp_m1() + 0.02;
    let cashflows = pool
        .cashflows_with_refinancing_rates(&prepayment, &[rate])
        .unwrap();
    let expected: f64 = cashflows
        .iter()
        .map(|cashflow| cashflow.total_cashflow * (-0.034 * cashflow.month as f64 / 12.0).exp())
        .sum();
    let deterministic = pool
        .price_hull_white_mc(
            &prepayment,
            &HullWhite::new(0.1, 0.0),
            &curve(0.03),
            &config,
            0.004,
        )
        .unwrap();
    assert!((deterministic.price - expected).abs() < 2.0e-12);
    assert_eq!(deterministic.stderr, Some(0.0));
    let model = HullWhite::new(0.1, 0.03);
    let stochastic = pool
        .price_hull_white_mc(&prepayment, &model, &curve(0.03), &config, 0.004)
        .unwrap();
    assert!(stochastic.stderr.unwrap() > 0.0);
    let spread = pool
        .oas_hull_white(stochastic.price, &prepayment, &model, &curve(0.03), &config)
        .unwrap();
    assert!((spread - 0.004).abs() < 1.0e-12);
}

/// SciPy 1.17.1 adaptive Gaussian integration (absolute quadrature error below
/// 3e-11). A three-payment pool has only one stochastic prepayment decision:
/// month-two and final cashflows integrate under their own payment measures.
#[test]
fn stochastic_mbs_matches_independent_payment_measure_integrals() {
    let pool = MbsPassThrough {
        original_balance: 100.0,
        coupon_rate: 0.06,
        servicing_fee: 0.005,
        original_term: 23,
        age: 20,
        prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
    };
    let config = MbsHullWhiteConfig {
        num_paths: 80_000,
        seed: 21,
        refinancing_tenor: 5.0,
        refinancing_spread: 0.02,
        refinancing_floor: 0.001,
    };
    let result = pool
        .price_hull_white_mc(
            &RateIncentivePrepayment::default(),
            &HullWhite::new(0.1, 0.08),
            &curve(0.03),
            &config,
            0.0,
        )
        .unwrap();
    assert!(
        (result.price - 100.407_340_883_422_08).abs() < 4.0 * result.stderr.unwrap() + 3.0e-11,
        "MBS price={}, stderr={:?}",
        result.price,
        result.stderr
    );
}
