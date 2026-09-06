use approx::assert_relative_eq;
use openferric::instruments::Autocallable;
use openferric::pricing::autocallable::price_autocallable;

fn autocall() -> Autocallable {
    Autocallable {
        underlyings: vec![0],
        notional: 100.0,
        autocall_dates: vec![0.1, 0.2],
        autocall_barrier: 0.5,
        coupon_rate: 0.1,
        ki_barrier: 0.5,
        ki_strike: 1.0,
        maturity: 1.0,
    }
}

#[test]
fn autocall_preserves_distinct_observations_sharing_one_simulation_step() {
    let note = autocall();
    let result = price_autocallable(&note, &[100.0], &[0.0], &[vec![1.0]], 0.03, 0.0, 8, 1);
    let expected = 101.0 * (-0.003_f64).exp();
    assert_relative_eq!(result.price, expected, epsilon = 2.0e-12);
}

#[test]
fn autocall_knock_in_monitoring_includes_initial_fixing() {
    let note = Autocallable {
        ki_barrier: 1.0,
        autocall_barrier: 10.0,
        autocall_dates: vec![1.0],
        ..autocall()
    };
    let result = price_autocallable(&note, &[100.0], &[0.0], &[vec![1.0]], 0.03, 0.0, 8, 4);
    assert_relative_eq!(result.price, 100.0 * (-0.03_f64).exp(), epsilon = 2.0e-12);
}

#[test]
fn comonotonic_equal_volatility_rainbows_reduce_to_ordered_vanillas() {
    use openferric::core::OptionType;
    use openferric::engines::analytic::black_scholes::bs_price;
    use openferric::engines::analytic::rainbow::{best_of_two_call_price, worst_of_two_call_price};
    use openferric::instruments::{BestOfTwoCallOption, WorstOfTwoCallOption};

    for second_spot in [50.0, 150.0] {
        for strike in [0.0, 100.0] {
            let best = BestOfTwoCallOption {
                s1: 100.0,
                s2: second_spot,
                k: strike,
                vol1: 0.25,
                vol2: 0.25,
                rho: 1.0,
                q1: 0.01,
                q2: 0.08,
                r: 0.03,
                t: 2.0,
            };
            let worst = WorstOfTwoCallOption {
                s1: best.s1,
                s2: best.s2,
                k: best.k,
                vol1: best.vol1,
                vol2: best.vol2,
                rho: best.rho,
                q1: best.q1,
                q2: best.q2,
                r: best.r,
                t: best.t,
            };
            let first_call = bs_price(
                OptionType::Call,
                best.s1,
                strike,
                best.r,
                best.q1,
                best.vol1,
                best.t,
            );
            let second_call = bs_price(
                OptionType::Call,
                best.s2,
                strike,
                best.r,
                best.q2,
                best.vol2,
                best.t,
            );
            assert_relative_eq!(
                best_of_two_call_price(&best).unwrap(),
                first_call.max(second_call),
                epsilon = 5.0e-14
            );
            assert_relative_eq!(
                worst_of_two_call_price(&worst).unwrap(),
                first_call.min(second_call),
                epsilon = 5.0e-14
            );
        }
    }
}

#[test]
fn power_put_retains_small_tail_premium() {
    use openferric::core::OptionType;
    use openferric::core::PricingEngine;
    use openferric::engines::analytic::BlackScholesEngine;
    use openferric::engines::analytic::power::power_option_price;
    use openferric::instruments::VanillaOption;
    use openferric::market::Market;
    let actual =
        power_option_price(OptionType::Put, 100.0, 1000.0, 0.03, 0.01, 0.2, 2.0, 1.0).unwrap();
    let scipy_reference = 2.577_277_057_726_424e-7;
    assert_relative_eq!(actual, scipy_reference, max_relative = 2.0e-12);
    let market = Market::builder()
        .spot(10_000.0 * 0.05_f64.exp())
        .rate(0.0)
        .flat_vol(0.4)
        .build()
        .unwrap();
    let option = VanillaOption::european_put(1000.0 * (-0.03_f64).exp(), 1.0);
    let price_and_greeks = BlackScholesEngine::new().price(&option, &market).unwrap();
    assert_relative_eq!(
        price_and_greeks.price,
        scipy_reference,
        max_relative = 2.0e-12
    );
}

#[test]
fn variance_replication_rejects_interior_duplicate_strikes() {
    use openferric::engines::analytic::variance_swap::fair_variance_strike_from_quotes;
    use openferric::instruments::VarianceOptionQuote;
    let quotes = [80.0, 90.0, 90.0, 100.0].map(|strike| VarianceOptionQuote::new(strike, 5.0, 5.0));
    assert!(fair_variance_strike_from_quotes(1.0, 0.03, 100.0, 0.0, &quotes).is_err());
}

#[test]
fn deterministic_tarf_matches_discounted_leveraged_fixings() {
    use openferric::instruments::tarf::{Tarf, TarfType};
    use openferric::pricing::tarf::tarf_mc_price;
    for direction in [TarfType::Standard, TarfType::Decumulator] {
        let contract = Tarf {
            strike: 100.0,
            notional_per_fixing: 10.0,
            ko_barrier: f64::INFINITY,
            target_profit: 1.0e8,
            downside_leverage: 2.0,
            fixing_times: vec![0.25, 0.75, 1.0],
            tarf_type: direction,
        };
        let expected = contract
            .fixing_times
            .iter()
            .map(|time| {
                let terminal = 102.0 * (-0.04 * time).exp();
                let difference = match direction {
                    TarfType::Standard => terminal - 100.0,
                    TarfType::Decumulator => 100.0 - terminal,
                };
                10.0 * difference * if difference < 0.0 { 2.0 } else { 1.0 } * (-0.01 * time).exp()
            })
            .sum::<f64>();
        let actual = tarf_mc_price(&contract, 102.0, 0.01, 0.05, 0.0, 1, 42).unwrap();
        assert_relative_eq!(actual.price, expected, epsilon = 1.0e-11);
        assert_eq!(actual.std_error, 0.0);
        assert!(tarf_mc_price(&contract, 102.0, f64::NAN, 0.05, 0.0, 1, 42).is_err());
    }
}

#[test]
fn continuous_only_engines_reject_active_discrete_dividends() {
    use openferric::core::{OptionType, PricingEngine};
    use openferric::dsl::{DslMonteCarloEngine, DslProduct, parse_and_compile};
    use openferric::engines::analytic::{
        BarrierAnalyticEngine, DoubleBarrierAnalyticEngine, ExoticAnalyticEngine,
    };
    use openferric::engines::tree::{ConvertibleBinomialEngine, SwingTreeEngine};
    use openferric::instruments::{
        BarrierOption, ConvertibleBond, DoubleBarrierOption, DoubleBarrierType, ExoticOption,
        SwingOption,
    };
    use openferric::market::{DividendEvent, DividendSchedule, Market};

    let barrier = BarrierOption::builder()
        .call()
        .strike(100.0)
        .expiry(1.0)
        .down_and_out(80.0)
        .build()
        .unwrap();
    let double = DoubleBarrierOption::new(
        OptionType::Call,
        100.0,
        1.0,
        80.0,
        120.0,
        DoubleBarrierType::KnockOut,
        0.0,
    );
    let convertible = ConvertibleBond::new(100.0, 0.05, 1.0, 1.0, None, None);
    let swing = SwingOption {
        min_exercises: 0,
        max_exercises: 1,
        exercise_dates: vec![0.5, 1.0],
        strike: 100.0,
        payoff_per_exercise: 1.0,
    };
    let product = DslProduct::new(parse_and_compile("product \"Forward\"\n    notional: 100\n    maturity: 1.0\n    underlyings\n        SPX = asset(0)\n    schedule annual from 1.0 to 1.0\n        redeem notional * SPX\n").unwrap());
    for event in [
        DividendEvent::cash(0.5, 5.0).unwrap(),
        DividendEvent::proportional(0.5, 0.05).unwrap(),
        DividendEvent::cash(2.0, 5.0).unwrap(),
    ] {
        let active = event.time <= 1.0;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .flat_vol(0.2)
            .dividend_schedule(DividendSchedule::new(vec![event]).unwrap())
            .build()
            .unwrap();
        for result in [
            BarrierAnalyticEngine::new().price(&barrier, &market),
            DoubleBarrierAnalyticEngine::new().price(&double, &market),
            ExoticAnalyticEngine::new().price(&ExoticOption::lookback_floating_put(1.0), &market),
            ConvertibleBinomialEngine::new(0.0).price(&convertible, &market),
            SwingTreeEngine::new(20).price(&swing, &market),
            DslMonteCarloEngine::new(8, 4, 42).price(&product, &market),
        ] {
            if active {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("discrete dividends")
                );
            } else {
                assert!(result.unwrap().price.is_finite());
            }
        }
    }
}

#[test]
fn forward_start_and_employee_options_reject_nonfinite_inputs() {
    use openferric::core::OptionType;
    use openferric::instruments::{EmployeeStockOption, ForwardStartOption};
    let forward = ForwardStartOption::atm_call(100.0, 0.03, 0.01, 0.2, 0.5, 1.0);
    for invalid in [
        ForwardStartOption {
            rate: f64::NAN,
            ..forward
        },
        ForwardStartOption {
            t_start: f64::INFINITY,
            ..forward
        },
        ForwardStartOption {
            spot: f64::NAN,
            ..forward
        },
    ] {
        assert!(invalid.price_rubinstein().is_err());
    }
    let employee = EmployeeStockOption::new(
        OptionType::Call,
        100.0,
        5.0,
        1.0,
        3.0,
        Some(2.0),
        0.02,
        1_000_000.0,
        10_000.0,
    );
    assert!(
        employee
            .price_binomial(100.0, f64::NAN, 0.01, 0.2, 100)
            .is_err()
    );
    let invalid = EmployeeStockOption {
        early_exercise_multiple: Some(f64::NAN),
        ..employee
    };
    assert!(invalid.validate().is_err());
}
