use approx::assert_relative_eq;

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::{BlackScholesEngine, ExoticAnalyticEngine};
use openferric::instruments::{ExoticOption, LookbackFloatingOption, VanillaOption};
use openferric::market::Market;
use openferric::vol::implied::implied_vol_newton;

#[derive(Debug, Clone, Copy)]
struct EuropeanCase {
    option_type: OptionType,
    strike: f64,
    spot: f64,
    dividend_yield: f64,
    rate: f64,
    expiry: f64,
    vol: f64,
    expected_value: f64,
}

#[derive(Debug, Clone, Copy)]
struct LookbackCase {
    option_type: OptionType,
    minmax: f64,
    spot: f64,
    dividend_yield: f64,
    rate: f64,
    expiry: f64,
    vol: f64,
    expected_value: f64,
}

fn european_cases() -> Vec<EuropeanCase> {
    vec![
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 65.0,
            spot: 60.0,
            dividend_yield: 0.00,
            rate: 0.08,
            expiry: 0.25,
            vol: 0.30,
            expected_value: 2.1334,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 95.0,
            spot: 100.0,
            dividend_yield: 0.05,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.20,
            expected_value: 2.4648,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 19.0,
            spot: 19.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.75,
            vol: 0.28,
            expected_value: 1.7011,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 19.0,
            spot: 19.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.75,
            vol: 0.28,
            expected_value: 1.7011,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 1.60,
            spot: 1.56,
            dividend_yield: 0.08,
            rate: 0.06,
            expiry: 0.50,
            vol: 0.12,
            expected_value: 0.0291,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 70.0,
            spot: 75.0,
            dividend_yield: 0.05,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.35,
            expected_value: 4.0870,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            expected_value: 0.0205,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            expected_value: 1.8734,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            expected_value: 9.9413,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            expected_value: 0.3150,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            expected_value: 3.1217,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            expected_value: 10.3556,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            expected_value: 0.8069,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            expected_value: 4.0232,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            expected_value: 10.5769,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            expected_value: 2.7026,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            expected_value: 6.6997,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            expected_value: 12.7857,
        },
    ]
}

fn lookback_cases() -> Vec<LookbackCase> {
    vec![
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 120.0,
            dividend_yield: 0.06,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            expected_value: 25.3533,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 1.00,
            vol: 0.30,
            expected_value: 23.7884,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 0.20,
            vol: 0.30,
            expected_value: 10.7190,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 110.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 0.20,
            vol: 0.30,
            expected_value: 14.4597,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            expected_value: 15.3526,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 110.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            expected_value: 16.8468,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 120.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            expected_value: 21.0645,
        },
    ]
}

fn build_vanilla_case(case: &EuropeanCase) -> VanillaOption {
    match case.option_type {
        OptionType::Call => VanillaOption::european_call(case.strike, case.expiry),
        OptionType::Put => VanillaOption::european_put(case.strike, case.expiry),
    }
}

fn to_pricing_option_type(option_type: OptionType) -> openferric::pricing::OptionType {
    match option_type {
        OptionType::Call => openferric::pricing::OptionType::Call,
        OptionType::Put => openferric::pricing::OptionType::Put,
    }
}

#[test]
fn quantlib_haug_european_reference_values() {
    // Reference: QuantLib europeanoption.cpp testValues (Haug 1998, pp. 2-8 and 24).
    let engine = BlackScholesEngine::new();

    for case in european_cases() {
        let option = build_vanilla_case(&case);
        let market = Market::builder()
            .spot(case.spot)
            .rate(case.rate)
            .dividend_yield(case.dividend_yield)
            .flat_vol(case.vol)
            .build()
            .expect("valid market");

        let price = engine.price(&option, &market).expect("pricing succeeds").price;
        assert_relative_eq!(
            price,
            case.expected_value,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }
}

#[test]
fn quantlib_haug_lookback_floating_reference_values() {
    // Reference: QuantLib test suite (Haug 1998 pp. 61-62, Broadie-Glasserman-Kou 1999 data).
    let engine = ExoticAnalyticEngine::new();

    for case in lookback_cases() {
        let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
            option_type: case.option_type,
            expiry: case.expiry,
            observed_extreme: Some(case.minmax),
        });

        let market = Market::builder()
            .spot(case.spot)
            .rate(case.rate)
            .dividend_yield(case.dividend_yield)
            .flat_vol(case.vol)
            .build()
            .expect("valid market");

        let price = engine.price(&option, &market).expect("pricing succeeds").price;
        assert_relative_eq!(
            price,
            case.expected_value,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }
}

#[test]
fn quantlib_put_call_parity_with_dividends() {
    let engine = BlackScholesEngine::new();
    let parity_inputs = [
        (60.0, 65.0, 0.00, 0.08, 0.25, 0.30),
        (100.0, 95.0, 0.05, 0.10, 0.50, 0.20),
        (100.0, 100.0, 0.10, 0.10, 0.10, 0.25),
        (90.0, 100.0, 0.10, 0.10, 0.50, 0.15),
        (110.0, 100.0, 0.10, 0.10, 0.50, 0.25),
        (75.0, 70.0, 0.05, 0.10, 0.50, 0.35),
    ];

    for (spot, strike, q, r, expiry, vol) in parity_inputs {
        let market = Market::builder()
            .spot(spot)
            .rate(r)
            .dividend_yield(q)
            .flat_vol(vol)
            .build()
            .expect("valid market");

        let call = engine
            .price(&VanillaOption::european_call(strike, expiry), &market)
            .expect("call pricing succeeds")
            .price;
        let put = engine
            .price(&VanillaOption::european_put(strike, expiry), &market)
            .expect("put pricing succeeds")
            .price;

        let rhs = spot * (-q * expiry).exp() - strike * (-r * expiry).exp();
        assert_relative_eq!(call - put, rhs, epsilon = 1e-4, max_relative = 1e-4);
    }
}

#[test]
fn quantlib_european_implied_vol_round_trip() {
    let engine = BlackScholesEngine::new();

    for case in european_cases() {
        let option = build_vanilla_case(&case);
        let market = Market::builder()
            .spot(case.spot)
            .rate(case.rate)
            .dividend_yield(case.dividend_yield)
            .flat_vol(case.vol)
            .build()
            .expect("valid market");

        let price = engine.price(&option, &market).expect("pricing succeeds").price;

        let spot_no_dividend = case.spot * (-case.dividend_yield * case.expiry).exp();
        let recovered = implied_vol_newton(
            to_pricing_option_type(case.option_type),
            spot_no_dividend,
            case.strike,
            case.rate,
            case.expiry,
            price,
            1e-12,
            100,
        )
        .expect("implied vol converges");

        assert_relative_eq!(recovered, case.vol, epsilon = 1e-6, max_relative = 1e-6);
    }
}
