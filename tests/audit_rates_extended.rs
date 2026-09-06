use approx::assert_relative_eq;
use chrono::NaiveDate;
use openferric::rates::multi_curve::{
    MultiCurveEnvironment, dual_curve_bootstrap, price_irs_multi_curve,
};
use openferric::rates::{
    BusinessDayConvention, CapFloor, DayCountConvention, FixedRateBond, ForwardRateAgreement,
    Frequency, InflationCurve, InflationIndexedBond, InterestRateSwap, Swaption, YieldCurve,
    YieldCurveBuilder, ZeroCouponInflationSwap,
};

fn flat_curve(rate: f64) -> YieldCurve {
    YieldCurve::new(vec![(10.0, (-rate * 10.0).exp())])
}

fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).unwrap()
}

#[test]
fn swap_dv01_preserves_curve_interpolation_convention() {
    use openferric::rates::{YieldCurveInterpolationMethod, YieldCurveInterpolationSettings};
    let settings = YieldCurveInterpolationSettings {
        method: YieldCurveInterpolationMethod::LinearZeroRate,
        ..Default::default()
    };
    let nodes = vec![(0.25, 0.995), (1.0, 0.94), (3.0, 0.80)];
    let curve = YieldCurve::new_with_settings(nodes.clone(), settings).unwrap();
    let bumped = YieldCurve::new_with_settings(
        nodes
            .iter()
            .map(|(time, discount)| (*time, discount * (-0.0001 * time).exp()))
            .collect(),
        settings,
    )
    .unwrap();
    let swap = InterestRateSwap::builder()
        .notional(1_000_000.0)
        .fixed_rate(0.04)
        .start_date(date(2025, 1, 2))
        .end_date(date(2027, 1, 2))
        .build();
    assert_relative_eq!(
        swap.dv01(&curve),
        swap.npv(&bumped) - swap.npv(&curve),
        epsilon = 1.0e-8
    );
}

#[test]
fn short_rate_zero_bonds_preserve_zero_reversion_and_zero_volatility_limits() {
    use openferric::models::{CIR, Vasicek};
    let quadrature = |mean_reversion: f64, volatility: f64| {
        let intervals = 2000;
        let dt = 2.0 / intervals as f64;
        let mut integral = 0.0;
        for index in 0..=intervals {
            let time = index as f64 * dt;
            let mean_rate = 0.05 + (0.03 - 0.05) * (-mean_reversion * time).exp();
            let response = if mean_reversion == 0.0 {
                time
            } else {
                -(-mean_reversion * time).exp_m1() / mean_reversion
            };
            let log_discount_density =
                -mean_rate + 0.5 * volatility * volatility * response * response;
            let weight = if index == 0 || index == intervals {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            integral += weight * log_discount_density;
        }
        (integral * dt / 3.0).exp()
    };
    for mean_reversion in [0.0, 1.0e-10, 0.1] {
        let vasicek = Vasicek {
            a: mean_reversion,
            b: 0.05,
            sigma: 0.02,
        };
        assert_relative_eq!(
            vasicek.bond_price(0.0, 2.0, 0.03),
            quadrature(mean_reversion, 0.02),
            epsilon = 3.0e-14
        );
        for volatility in [0.0, 1.0e-10] {
            let cir = CIR {
                a: mean_reversion,
                b: 0.05,
                sigma: volatility,
            };
            assert_relative_eq!(
                cir.bond_price(0.0, 2.0, 0.03),
                quadrature(mean_reversion, 0.0),
                epsilon = 3.0e-14
            );
        }
    }
    let long_cir = CIR {
        a: 0.1,
        b: 0.05,
        sigma: 0.2,
    };
    let expected = 1.5_f64.powf(0.25) * (-250.15_f64).exp();
    assert_relative_eq!(
        long_cir.bond_price(0.0, 10_000.0, 0.03),
        expected,
        max_relative = 2.0e-13
    );
}

#[test]
fn bermudan_swaption_at_time_zero_pays_current_swap_intrinsic() {
    use openferric::engines::tree::BermudanSwaptionEngine;
    use openferric::models::HullWhite;
    let curve = flat_curve(0.03);
    for strike in [0.0, 0.02] {
        let option = Swaption {
            notional: 100.0,
            strike,
            option_expiry: 0.0,
            swap_tenor: 2.0,
            is_payer: true,
        };
        let expected =
            100.0 * (1.0 - (-0.06_f64).exp() - strike * ((-0.03_f64).exp() + (-0.06_f64).exp()));
        let actual = BermudanSwaptionEngine::new(HullWhite::new(0.1, 0.01), 20).price(
            &option,
            &[0.0],
            &curve,
        );
        assert_relative_eq!(actual, expected, epsilon = 3.0e-13);
    }
}

#[test]
fn fixed_bond_short_back_stub_pays_only_accrued_coupon() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.06,
        frequency: 2,
        maturity: 1.1,
        day_count: DayCountConvention::Act365Fixed,
    };
    let curve = flat_curve(0.04);
    let expected = 3.0 * (-0.02_f64).exp() + 3.0 * (-0.04_f64).exp() + 100.6 * (-0.044_f64).exp();
    assert_relative_eq!(bond.dirty_price(&curve), expected, epsilon = 3.0e-13);
    assert_relative_eq!(bond.accrued_interest(1.05), 0.3, epsilon = 1.0e-14);
    assert_relative_eq!(
        bond.clean_price(&curve, 1.05),
        100.6 * (-0.002_f64).exp() - 0.3,
        epsilon = 3.0e-13
    );
    let short = FixedRateBond {
        maturity: 0.1,
        ..bond
    };
    assert_relative_eq!(
        short.dirty_price(&curve),
        100.6 * (-0.004_f64).exp(),
        epsilon = 3.0e-13
    );
}

#[test]
fn bond_yield_solver_handles_valid_near_negative_frequency_yields() {
    let bond = FixedRateBond {
        face_value: 100.0,
        coupon_rate: 0.0,
        frequency: 2,
        maturity: 1.0,
        day_count: DayCountConvention::Act365Fixed,
    };
    for yield_rate in [-1.99_f64, -1.5, -0.05, 0.0, 0.25, 10.0] {
        let market_price = 100.0 / (1.0 + yield_rate / 2.0).powi(2);
        assert_relative_eq!(bond.ytm(market_price), yield_rate, epsilon = 3.0e-12);
    }
    assert!(bond.ytm(f64::NAN).is_nan());
}

#[test]
fn mixed_accrual_swap_uses_one_curve_clock() {
    let swap = InterestRateSwap::builder()
        .notional(1_000_000.0)
        .fixed_rate(0.05)
        .start_date(date(2025, 1, 1))
        .end_date(date(2026, 1, 1))
        .fixed_freq(Frequency::Annual)
        .float_freq(Frequency::Annual)
        .business_day_convention(BusinessDayConvention::Unadjusted)
        .fixed_day_count(DayCountConvention::Thirty360)
        .float_day_count(DayCountConvention::Act360)
        .build();
    let curve = flat_curve(0.04);
    let expected_fixed = 50_000.0 * (-0.04_f64).exp();
    let expected_float = 1_000_000.0 * -(-0.04_f64).exp_m1();
    assert_relative_eq!(swap.fixed_leg_pv(&curve), expected_fixed, epsilon = 2.0e-9);
    assert_relative_eq!(swap.float_leg_pv(&curve), expected_float, epsilon = 2.0e-9);
    assert_relative_eq!(
        swap.npv(&curve),
        expected_float - expected_fixed,
        epsilon = 3.0e-9
    );
}

#[test]
fn cap_coupon_day_count_does_not_move_discount_or_expiry_dates() {
    let cap = CapFloor {
        notional: 1_000_000.0,
        strike: 0.0,
        start_date: date(2025, 1, 1),
        end_date: date(2026, 1, 1),
        frequency: Frequency::Quarterly,
        day_count: DayCountConvention::Act360,
        curve_day_count: DayCountConvention::Act365Fixed,
        is_cap: true,
    };
    assert_relative_eq!(
        cap.price(&flat_curve(0.04), 0.2),
        1_000_000.0 * -(-0.04_f64).exp_m1(),
        epsilon = 3.0e-9
    );
}

#[test]
fn replacing_forward_curve_updates_prices_instead_of_using_stale_quotes() {
    let mut environment = MultiCurveEnvironment::new(flat_curve(0.02));
    environment.add_forward_curve("3M", flat_curve(0.04));
    environment.add_forward_curve("3M", flat_curve(0.06));
    assert_eq!(environment.forward_curves.len(), 1);
    assert_relative_eq!(
        environment.forward_rate("3M", 0.0, 0.25).unwrap(),
        (0.015_f64).exp_m1() / 0.25,
        epsilon = 2.0e-14
    );
}

#[test]
fn fra_curve_dates_do_not_change_with_coupon_day_count() {
    let fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.035,
        valuation_date: date(2025, 1, 1),
        start_date: date(2025, 7, 1),
        end_date: date(2026, 1, 1),
        day_count: DayCountConvention::Act360,
        curve_day_count: DayCountConvention::Act365Fixed,
    };
    let curve = flat_curve(0.04);
    let accrual = 184.0 / 360.0;
    let expected_forward = (0.04_f64 * 184.0 / 365.0).exp_m1() / accrual;
    assert_relative_eq!(
        fra.forward_rate(&curve),
        expected_forward,
        epsilon = 1.0e-14
    );
    assert_relative_eq!(
        fra.npv(&curve),
        1_000_000.0 * (expected_forward - 0.035) * accrual * (-0.04_f64).exp(),
        epsilon = 2.0e-9
    );
    let seasoned = ForwardRateAgreement {
        valuation_date: date(2025, 8, 1),
        ..fra
    };
    assert!(seasoned.npv(&curve).is_nan());
}

#[test]
fn multi_curve_swap_retains_final_stub_and_true_maturity() {
    let mut environment = MultiCurveEnvironment::new(flat_curve(0.02));
    environment.add_forward_curve("3M", flat_curve(0.04));
    for tenor in [0.1, 1.1] {
        let mut expected = 0.0;
        let mut previous = 0.0;
        for payment in [0.25_f64, 0.5, 0.75, 1.0, 1.1] {
            let end = payment.min(tenor);
            let accrual = end - previous;
            expected += 1_000_000.0
                * ((0.04_f64 * accrual).exp_m1() - 0.03 * accrual)
                * (-0.02_f64 * end).exp();
            previous = end;
            if end == tenor {
                break;
            }
        }
        let actual =
            price_irs_multi_curve(&environment, "3M", 1_000_000.0, 0.03, tenor, 4).unwrap();
        assert_relative_eq!(actual, expected, epsilon = 3.0e-9);
    }
    assert!(price_irs_multi_curve(&environment, "3M", 1.0, 0.03, 1.0, 0).is_none());
}

#[test]
fn single_and_dual_curve_bootstraps_reprice_nonintegral_tenors() {
    let quotes = [(0.1, 0.03), (0.6, 0.035), (1.1, 0.04)];
    let single = YieldCurveBuilder::from_swap_rates(&quotes, 4);
    let discount = flat_curve(0.02);
    let dual = dual_curve_bootstrap(&quotes, &discount, 4);
    assert_eq!(single.tenors.len(), quotes.len());
    assert_eq!(dual.tenors.len(), quotes.len());
    for (tenor, rate) in quotes {
        let mut single_annuity = 0.0;
        let mut dual_pv = 0.0;
        let mut previous = 0.0;
        for payment in [0.25_f64, 0.5, 0.75, 1.0, 1.1] {
            let end = payment.min(tenor);
            let accrual = end - previous;
            single_annuity += accrual * single.discount_factor(end);
            dual_pv +=
                (dual.discount_factor(previous) / dual.discount_factor(end) - 1.0 - rate * accrual)
                    * discount.discount_factor(end);
            previous = end;
            if end == tenor {
                break;
            }
        }
        assert_relative_eq!(
            rate * single_annuity,
            1.0 - single.discount_factor(tenor),
            epsilon = 3.0e-14
        );
        assert_relative_eq!(dual_pv, 0.0, epsilon = 3.0e-14);
        assert!(dual.tenors.iter().any(|node| node.0 == tenor));
    }
}

#[test]
fn inflation_mtm_uses_forward_discount_on_original_time_axis() {
    let discount = YieldCurve::new(vec![(1.0, 0.98), (4.0, 0.86), (5.0, 0.80)]);
    let inflation = InflationCurve::new(vec![(1.0, 1.02), (5.0, 1.15)]);
    let swap = ZeroCouponInflationSwap {
        notional: 1_000_000.0,
        cpi_base: 100.0,
        fixed_rate: 0.02,
        tenor: 5.0,
        receive_inflation: true,
    };
    let expected = 1_000_000.0 * (1.03 * 1.15 / 1.02 - 1.02_f64.powi(5)) * 0.80 / 0.98;
    assert_relative_eq!(
        swap.mtm(1.0, 103.0, &discount, &inflation),
        expected,
        epsilon = 3.0e-9
    );
}

#[test]
fn tips_redemption_floor_does_not_floor_intermediate_coupons() {
    let bond = InflationIndexedBond {
        face_value: 100.0,
        coupon_rate: 0.02,
        maturity_years: 2,
        coupon_frequency: 1,
        cpi_base: 100.0,
    };
    let inflation = InflationCurve::new(vec![(1.0, 0.95), (2.0, 0.9)]);
    let expected = 1.9 * (-0.03_f64).exp() + 101.8 * (-0.06_f64).exp();
    assert_relative_eq!(
        bond.price(&flat_curve(0.03), &inflation),
        expected,
        epsilon = 3.0e-13
    );
}

#[test]
fn swaption_zero_strike_and_zero_forward_obey_black_limits() {
    let payer = Swaption {
        notional: 1_000_000.0,
        strike: 0.0,
        option_expiry: 1.0,
        swap_tenor: 2.0,
        is_payer: true,
    };
    let expected = 1_000_000.0 * ((-0.03_f64).exp() - (-0.09_f64).exp());
    assert_relative_eq!(
        payer.price(&flat_curve(0.03), 0.2),
        expected,
        epsilon = 3.0e-9
    );
    let receiver = Swaption {
        strike: 0.02,
        is_payer: false,
        ..payer
    };
    assert_relative_eq!(
        receiver.price(&flat_curve(0.0), 0.2),
        40_000.0,
        epsilon = 3.0e-9
    );
    assert!(payer.price(&flat_curve(0.03), -0.2).is_nan());
}

#[test]
fn rate_option_implied_vol_rejects_impossible_or_nonfinite_prices() {
    let curve = flat_curve(0.04);
    let swaption = Swaption {
        notional: 1_000_000.0,
        strike: 0.02,
        option_expiry: 1.0,
        swap_tenor: 3.0,
        is_payer: true,
    };
    assert!(
        swaption
            .implied_vol(swaption.price(&curve, 0.0) * 0.5, &curve)
            .is_nan()
    );
    let cap = CapFloor {
        notional: 1_000_000.0,
        strike: 0.02,
        start_date: date(2025, 1, 1),
        end_date: date(2028, 1, 1),
        frequency: Frequency::SemiAnnual,
        day_count: DayCountConvention::Act365Fixed,
        curve_day_count: DayCountConvention::Act365Fixed,
        is_cap: true,
    };
    assert!(
        cap.implied_vol(cap.price(&curve, 0.0) * 0.5, &curve)
            .is_nan()
    );
    assert!(cap.implied_vol(f64::NAN, &curve).is_nan());
    assert!(cap.implied_vol(f64::INFINITY, &curve).is_nan());
}
