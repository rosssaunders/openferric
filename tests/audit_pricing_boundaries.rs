use openferric::core::{AsianSpec, Averaging, OptionType, PricingEngine, StrikeType};
use openferric::credit::cds_option::{CdsOption, fair_spread_from_hazard, risky_annuity};
use openferric::engines::analytic::{
    Black76Engine, BlackScholesEngine, DoubleBarrierAnalyticEngine, GeometricAsianEngine,
    black76_greeks,
};
use openferric::engines::monte_carlo::{
    ArithmeticAsianMC, MonteCarloPricingEngine, VarianceReduction,
};
use openferric::engines::pde::{
    CrankNicolsonEngine, ExplicitFdEngine, HopscotchEngine, ImplicitFdEngine,
};
use openferric::instruments::double_barrier::{DoubleBarrierOption, DoubleBarrierType};
use openferric::instruments::{AsianOption, FuturesOption, VanillaOption};
use openferric::market::{DividendEvent, DividendKind, DividendSchedule, Market};
use openferric::rates::CapFloor;

fn market(rate: f64) -> Market {
    Market::builder()
        .spot(100.0)
        .rate(rate)
        .dividend_yield(0.02)
        .flat_vol(0.3)
        .build()
        .unwrap()
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

fn asian(option_type: OptionType, averaging: Averaging, expiry: f64, times: &[f64]) -> AsianOption {
    AsianOption::new(
        option_type,
        95.0,
        expiry,
        AsianSpec {
            averaging,
            strike_type: StrikeType::Fixed,
            observation_times: times.to_vec(),
        },
    )
}

fn single_fixing_reference(option: &AsianOption, market: &Market) -> f64 {
    let fixing = option.asian.observation_times[0];
    [
        (-0.05, OptionType::Call, 0.0, 5.389_420_754_423_158),
        (-0.05, OptionType::Call, 0.5, 9.587_343_768_802_722),
        (-0.05, OptionType::Call, 1.5, 12.011_489_268_451_179),
        (-0.05, OptionType::Put, 0.0, 0.0),
        (-0.05, OptionType::Put, 0.5, 7.905_260_683_603_887),
        (-0.05, OptionType::Put, 1.5, 17.365_930_247_640_364),
        (0.2, OptionType::Call, 0.0, 3.704_091_103_408_589_3),
        (0.2, OptionType::Call, 0.5, 13.096_840_337_494_951),
        (0.2, OptionType::Call, 1.5, 29.829_293_710_117_195),
        (0.2, OptionType::Put, 0.0, 0.0),
        (0.2, OptionType::Put, 0.5, 2.416_146_705_239_445),
        (0.2, OptionType::Put, 1.5, 3.162_471_320_029_557),
    ]
    .into_iter()
    .find_map(|(rate, option_type, time, reference)| {
        (market.rate == rate && option.option_type == option_type && fixing == time)
            .then_some(reference)
    })
    .expect("SciPy 1.17.1 single-fixing reference")
}

#[test]
fn geometric_asian_discounts_to_payment_not_last_fixing() {
    for rate in [-0.05, 0.2] {
        let market = market(rate);
        for option_type in [OptionType::Call, OptionType::Put] {
            for fixing in [0.0, 0.5, 1.5] {
                let option = asian(option_type, Averaging::Geometric, 1.5, &[fixing]);
                let result = GeometricAsianEngine::new().price(&option, &market).unwrap();
                assert_close(
                    "single-fixing geometric Asian",
                    result.price,
                    single_fixing_reference(&option, &market),
                    2.0e-12,
                );
            }
        }
    }
}

#[test]
fn geometric_asian_dividends_are_applied_at_each_fixing() {
    let option = asian(OptionType::Call, Averaging::Geometric, 1.5, &[0.5]);
    let mut market = market(0.2);
    let engine = GeometricAsianEngine::new();
    let expected = single_fixing_reference(&option, &market);
    market.dividend_schedule =
        DividendSchedule::new(vec![DividendEvent::cash(1.0, 20.0).unwrap()]).unwrap();
    assert_close(
        "dividend after final fixing",
        engine.price(&option, &market).unwrap().price,
        expected,
        2.0e-12,
    );

    let option = asian(OptionType::Call, Averaging::Geometric, 1.5, &[0.0, 0.5]);
    market.dividend_schedule =
        DividendSchedule::new(vec![DividendEvent::proportional(0.25, 0.2).unwrap()]).unwrap();
    let mut equivalent = market.clone();
    equivalent.dividend_schedule = DividendSchedule::default();
    equivalent.spot *= 0.8_f64.sqrt();
    assert_close(
        "proportional fixing weights",
        engine.price(&option, &market).unwrap().price,
        engine.price(&option, &equivalent).unwrap().price,
        2.0e-12,
    );

    market.dividend_schedule =
        DividendSchedule::new(vec![DividendEvent::cash(0.25, 20.0).unwrap()]).unwrap();
    assert!(engine.price(&option, &market).is_err());
}

#[test]
fn negative_rate_american_put_boundaries_preserve_european_reduction() {
    for spot in [0.1, 1.0, 10.0] {
        let market = Market::builder()
            .spot(spot)
            .rate(-0.05)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = VanillaOption::american_put(100.0, 1.0);
        let european = VanillaOption::european_put(100.0, 1.0);
        let expected = 100.0 * 0.05_f64.exp() - spot;
        let cn = CrankNicolsonEngine::new(800, 800);
        let cn_price = cn.price(&option, &market).unwrap().price;
        assert_close("negative-rate CN boundary", cn_price, expected, 4.0e-9);
        assert_close(
            "no put exercise at negative rates",
            cn_price,
            cn.price(&european, &market).unwrap().price,
            2.0e-12,
        );
        let implicit = ImplicitFdEngine::new(800, 800)
            .price(&option, &market)
            .unwrap()
            .price;
        let time_error_bound = 100.0 * 0.05_f64.exp() * 0.05_f64.powi(2) / 800.0;
        assert_close(
            "negative-rate implicit boundary",
            implicit,
            expected,
            time_error_bound,
        );
    }
}

#[test]
fn american_call_upper_boundary_preserves_no_dividend_european_reduction() {
    let market = Market::builder()
        .spot(1_000.0)
        .rate(0.05)
        .flat_vol(0.2)
        .build()
        .unwrap();
    let american = VanillaOption::american_call(100.0, 1.0);
    let european = VanillaOption::european_call(100.0, 1.0);
    let engine = CrankNicolsonEngine::new(800, 800).with_s_max_multiplier(11.0);
    let price = engine.price(&american, &market).unwrap().price;
    assert_close(
        "deep-ITM call carry",
        price,
        1_000.0 - 100.0 * (-0.05_f64).exp(),
        4.0e-9,
    );
    assert_close(
        "no call early exercise without dividends",
        price,
        engine.price(&european, &market).unwrap().price,
        2.0e-12,
    );
}

#[test]
fn pde_pricers_reject_spot_outside_the_configured_domain() {
    let market = Market::builder()
        .spot(1_000.0)
        .rate(0.05)
        .flat_vol(0.2)
        .build()
        .unwrap();
    let option = VanillaOption::european_call(100.0, 1.0);
    assert!(
        CrankNicolsonEngine::new(10, 20)
            .price(&option, &market)
            .is_err()
    );
    assert!(
        ImplicitFdEngine::new(10, 20)
            .with_s_max_multiplier(0.5)
            .price(&option, &market)
            .is_err()
    );
    assert!(
        ExplicitFdEngine::new(10, 20)
            .with_s_max_multiplier(0.5)
            .price(&option, &market)
            .is_err()
    );
    assert!(
        HopscotchEngine::new(10, 20)
            .with_s_max_multiplier(0.5)
            .price(&option, &market)
            .is_err()
    );
}

#[test]
fn arithmetic_asian_controls_use_the_cashflow_payment_date() {
    for rate in [-0.05, 0.2] {
        let market = market(rate);
        for option_type in [OptionType::Call, OptionType::Put] {
            for fixing in [0.0, 0.5, 1.5] {
                let option = asian(option_type, Averaging::Arithmetic, 1.5, &[fixing]);
                let expected = single_fixing_reference(&option, &market);
                let generic = MonteCarloPricingEngine::new(1_024, 6, 42)
                    .with_variance_reduction(VarianceReduction::ControlVariate)
                    .price(&option, &market)
                    .unwrap();
                let dedicated = ArithmeticAsianMC::new(1_024, 6, 42)
                    .price(&option, &market)
                    .unwrap();
                for result in [generic, dedicated] {
                    assert_close(
                        "single-fixing arithmetic control",
                        result.price,
                        expected,
                        2.0e-11,
                    );
                    assert!(result.stderr.unwrap().is_finite());
                }
            }
        }
    }
}

#[test]
fn arithmetic_asian_controls_match_the_simulated_fixing_grid() {
    let market = market(0.1);
    for (option_type, coarse_reference, aligned_reference) in [
        (
            OptionType::Call,
            6.642_141_347_798_106,
            7.284_814_779_332_883,
        ),
        (
            OptionType::Put,
            3.520_801_893_749_042,
            3.866_655_639_520_113_5,
        ),
    ] {
        let mut option = asian(option_type, Averaging::Arithmetic, 1.0, &[0.0, 0.37, 1.0]);
        option.strike = 100.0;
        for (steps, reference) in [(4, coarse_reference), (100, aligned_reference)] {
            let generic = MonteCarloPricingEngine::new(65_536, steps, 42)
                .with_variance_reduction(VarianceReduction::ControlVariate)
                .price(&option, &market)
                .unwrap();
            let dedicated = ArithmeticAsianMC::new(65_536, steps, 42)
                .price(&option, &market)
                .unwrap();
            for result in [generic, dedicated] {
                assert_close(
                    "conditional-lognormal quadrature",
                    result.price,
                    reference,
                    4.0 * result.stderr.unwrap() + 1.0e-10,
                );
            }
        }
    }
}

#[test]
fn black76_zero_volatility_retains_deterministic_greeks() {
    for option_type in [OptionType::Call, OptionType::Put] {
        for forward in [80.0, 120.0] {
            for rate in [-0.03, 0.05] {
                let option = FuturesOption::new(forward, 100.0, 0.0, rate, 1.5, option_type);
                let result = Black76Engine::new().price(&option, &market(rate)).unwrap();
                let sign = if option_type == OptionType::Call {
                    1.0
                } else {
                    -1.0
                };
                let discount = (-rate * option.t).exp();
                let expected_price = discount * (sign * (forward - option.strike)).max(0.0);
                let expected_delta = if expected_price > 0.0 {
                    sign * discount
                } else {
                    0.0
                };
                assert_close(
                    "deterministic Black price",
                    result.price,
                    expected_price,
                    1.0e-13,
                );
                for greeks in [
                    result.greeks.unwrap(),
                    black76_greeks(option_type, forward, 100.0, rate, 0.0, 1.5).unwrap(),
                ] {
                    assert_close("Black delta", greeks.delta, expected_delta, 1.0e-14);
                    assert_close("Black gamma", greeks.gamma, 0.0, 1.0e-14);
                    assert_close("Black vega", greeks.vega, 0.0, 1.0e-14);
                    assert_close("Black theta", greeks.theta, rate * expected_price, 1.0e-13);
                    assert_close("Black rho", greeks.rho, -option.t * expected_price, 1.0e-13);
                }
            }
        }
    }
}

#[test]
fn black76_zero_volatility_atm_marks_kinks_and_keeps_right_vega() {
    let discount = (-0.05_f64 * 1.5).exp();
    for option_type in [OptionType::Call, OptionType::Put] {
        let greeks = black76_greeks(option_type, 100.0, 100.0, 0.05, 0.0, 1.5).unwrap();
        assert!(greeks.delta.is_nan());
        assert!(greeks.gamma.is_nan());
        assert_close(
            "ATM right vega",
            greeks.vega,
            discount * 100.0 * (1.5 / std::f64::consts::TAU).sqrt(),
            1.0e-13,
        );
        assert_eq!(greeks.theta, 0.0);
        assert_eq!(greeks.rho, 0.0);
    }
}

#[test]
fn double_barrier_strikes_outside_corridor_match_killed_density() {
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.25)
        .build()
        .unwrap();
    let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(30);
    for (option_type, references) in [
        (
            OptionType::Call,
            [
                7.166_996_371_725_996,
                3.418_094_770_031_186,
                0.529_158_958_186_655_5,
                0.0,
                0.0,
            ],
        ),
        (
            OptionType::Put,
            [
                0.0,
                0.0,
                0.859_965_789_850_278_5,
                4.079_708_433_358_433,
                7.828_610_035_053_242,
            ],
        ),
    ] {
        for (strike, expected) in [60.0, 80.0, 100.0, 120.0, 140.0]
            .into_iter()
            .zip(references)
        {
            let option = DoubleBarrierOption::new(
                option_type,
                strike,
                1.0,
                80.0,
                120.0,
                DoubleBarrierType::KnockOut,
                0.0,
            );
            let result = engine.price(&option, &market).unwrap();
            assert_close(
                "killed log-Brownian density",
                result.price,
                expected,
                2.0e-12,
            );
            let knock_in = DoubleBarrierOption {
                barrier_type: DoubleBarrierType::KnockIn,
                ..option
            };
            let vanilla = VanillaOption {
                option_type,
                strike,
                expiry: 1.0,
                exercise: openferric::core::ExerciseStyle::European,
            };
            let vanilla_price = BlackScholesEngine::new()
                .price(&vanilla, &market)
                .unwrap()
                .price;
            assert_close(
                "outside-strike in/out parity",
                engine.price(&knock_in, &market).unwrap().price + result.price,
                vanilla_price,
                2.0e-12,
            );
        }
    }
}

#[test]
fn caplets_and_floorlets_preserve_zero_strike_and_forward_limits() {
    for volatility in [0.0, 0.3] {
        assert_close(
            "zero-strike caplet",
            CapFloor::black_caplet(1_000_000.0, 0.95, 0.25, 0.04, 0.0, volatility, 1.0),
            9_500.0,
            1.0e-10,
        );
        assert_close(
            "zero-strike floorlet",
            CapFloor::black_floorlet(1_000_000.0, 0.95, 0.25, 0.04, 0.0, volatility, 1.0),
            0.0,
            1.0e-10,
        );
        assert_close(
            "zero-forward floorlet",
            CapFloor::black_floorlet(1_000_000.0, 0.95, 0.25, 0.0, 0.04, volatility, 1.0),
            9_500.0,
            1.0e-10,
        );
    }
    assert!(CapFloor::black_floorlet(1_000_000.0, 0.95, 0.25, -0.01, 0.04, 0.3, 1.0).is_nan());
    assert!(CapFloor::black_caplet(1_000_000.0, 0.95, 0.25, 0.04, 0.03, -0.1, 1.0).is_nan());
}

#[test]
fn cds_option_at_expiry_pays_intrinsic() {
    for is_payer in [false, true] {
        for forward in [0.01, 0.03] {
            let option = CdsOption {
                notional: 1_000_000.0,
                strike_spread: 0.02,
                option_expiry: 0.0,
                cds_maturity: 5.0,
                is_payer,
                recovery_rate: 0.4,
            };
            let sign = if is_payer { 1.0 } else { -1.0 };
            let expected =
                option.notional * 4.0 * (sign * (forward - option.strike_spread)).max(0.0);
            assert_close(
                "expiring CDS option",
                option.black_price(forward, 0.3, 4.0),
                expected,
                1.0e-10,
            );
        }
    }
}

#[test]
fn cds_options_handle_zero_spreads_and_reject_invalid_model_inputs() {
    let option = CdsOption {
        notional: 1_000_000.0,
        strike_spread: 0.02,
        option_expiry: 1.0,
        cds_maturity: 5.0,
        is_payer: false,
        recovery_rate: 0.4,
    };
    assert_close(
        "zero-forward CDS receiver",
        option.black_price(0.0, 0.3, 4.0),
        80_000.0,
        1.0e-10,
    );
    assert!(option.black_price(-0.01, 0.3, 4.0).is_nan());
    assert!(option.black_price(0.01, -0.3, 4.0).is_nan());
    assert!(option.black_price(f64::NAN, 0.3, 4.0).is_nan());
    let expired = CdsOption {
        option_expiry: -0.1,
        ..option
    };
    assert_eq!(expired.black_price(0.01, 0.3, 4.0), 0.0);
}

#[test]
fn cds_annuity_and_spread_include_the_final_stub() {
    for tenor in [0.1_f64, 0.3, 1.1, 1.25] {
        let rate = 0.03;
        let hazard = 0.02;
        let mut previous = 0.0_f64;
        let mut annuity = 0.0;
        let mut protection = 0.0;
        for payment in [0.25_f64, 0.5, 0.75, 1.0, 1.25] {
            let end = payment.min(tenor);
            annuity += (end - previous) * (-(rate + hazard) * end).exp();
            protection += 0.6
                * ((-hazard * previous).exp() - (-hazard * end).exp())
                * (-rate * 0.5 * (previous + end)).exp();
            if end == tenor {
                break;
            }
            previous = end;
        }
        assert_close(
            "stub RPV01",
            risky_annuity(4, tenor, hazard, rate, 0.4),
            annuity,
            1.0e-14,
        );
        assert_close(
            "stub fair spread",
            fair_spread_from_hazard(4, tenor, hazard, rate, 0.4),
            protection / annuity,
            1.0e-14,
        );
    }
}

#[test]
fn discrete_dividend_greeks_differentiate_the_original_market() {
    let mut market = market(0.05);
    market.dividend_schedule = DividendSchedule::new(vec![
        DividendEvent::cash(0.4, 5.0).unwrap(),
        DividendEvent::proportional(0.7, 0.15).unwrap(),
        DividendEvent::cash(2.0, 3.0).unwrap(),
    ])
    .unwrap();
    let engine = BlackScholesEngine::new();
    for option in [
        VanillaOption::european_call(100.0, 1.2),
        VanillaOption::european_put(100.0, 1.2),
    ] {
        let price = engine.price(&option, &market).unwrap().price;
        let mut up = market.clone();
        let mut down = market.clone();
        up.spot += 0.01;
        down.spot -= 0.01;
        let price_up = engine.price(&option, &up).unwrap().price;
        let price_down = engine.price(&option, &down).unwrap().price;
        let delta = (price_up - price_down) / 0.02;
        let gamma = (price_up - 2.0 * price + price_down) / 0.0001;
        up = market.clone();
        down = market.clone();
        up.rate += 1.0e-5;
        down.rate -= 1.0e-5;
        let rho = (engine.price(&option, &up).unwrap().price
            - engine.price(&option, &down).unwrap().price)
            / 2.0e-5;
        let roll = |elapsed: f64| {
            let mut rolled_market = market.clone();
            rolled_market.dividend_schedule = DividendSchedule::new(
                market
                    .dividends()
                    .events()
                    .iter()
                    .map(|event| match event.kind {
                        DividendKind::Cash(amount) => {
                            DividendEvent::cash(event.time - elapsed, amount).unwrap()
                        }
                        DividendKind::Proportional(ratio) => {
                            DividendEvent::proportional(event.time - elapsed, ratio).unwrap()
                        }
                    })
                    .collect(),
            )
            .unwrap();
            let rolled_option = VanillaOption {
                expiry: option.expiry - elapsed,
                ..option.clone()
            };
            engine.price(&rolled_option, &rolled_market).unwrap().price
        };
        let theta = (roll(1.0e-5) - roll(-1.0e-5)) / 2.0e-5;
        for result in [
            engine.price(&option, &market).unwrap(),
            engine.price_with_greeks_aad(&option, &market).unwrap(),
        ] {
            let greeks = result.greeks.unwrap();
            assert_close("dividend delta", greeks.delta, delta, 2.0e-8);
            assert_close("dividend gamma", greeks.gamma, gamma, 2.0e-8);
            assert_close("dividend rho", greeks.rho, rho, 2.0e-6);
            assert_close("dividend calendar theta", greeks.theta, theta, 2.0e-7);
        }
    }
}
