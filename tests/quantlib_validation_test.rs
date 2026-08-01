use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::black_scholes::bs_vega;
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
    published_value_4dp: f64,
    exact_value: f64,
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
    published_value_4dp: f64,
    exact_value: f64,
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
            published_value_4dp: 2.1334,
            exact_value: 2.133_368_444_916_198_5,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 95.0,
            spot: 100.0,
            dividend_yield: 0.05,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.20,
            published_value_4dp: 2.4648,
            exact_value: 2.464_787_646_755_826,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 19.0,
            spot: 19.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.75,
            vol: 0.28,
            published_value_4dp: 1.7011,
            exact_value: 1.701_050_725_236_268,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 19.0,
            spot: 19.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.75,
            vol: 0.28,
            published_value_4dp: 1.7011,
            exact_value: 1.701_050_725_236_268,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 1.60,
            spot: 1.56,
            dividend_yield: 0.08,
            rate: 0.06,
            expiry: 0.50,
            vol: 0.12,
            published_value_4dp: 0.0291,
            exact_value: 0.029_099_253_149_439_65,
        },
        EuropeanCase {
            option_type: OptionType::Put,
            strike: 70.0,
            spot: 75.0,
            dividend_yield: 0.05,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.35,
            published_value_4dp: 4.0870,
            exact_value: 4.086_953_828_635_357,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            published_value_4dp: 0.0205,
            exact_value: 0.020_490_148_536_478_306,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            published_value_4dp: 1.8734,
            exact_value: 1.873_344_572_764_941_6,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.15,
            published_value_4dp: 9.9413,
            exact_value: 9.941_277_395_489_541,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            published_value_4dp: 0.3150,
            exact_value: 0.315_045_800_777_366_8,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            published_value_4dp: 3.1217,
            exact_value: 3.121_720_698_181_133,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.10,
            vol: 0.25,
            published_value_4dp: 10.3556,
            exact_value: 10.355_552_136_523_041,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            published_value_4dp: 0.8069,
            exact_value: 0.806_892_413_675_934_4,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            published_value_4dp: 4.0232,
            exact_value: 4.023_167_048_617_123,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.15,
            published_value_4dp: 10.5769,
            exact_value: 10.576_857_786_819_106,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 90.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            published_value_4dp: 2.7026,
            exact_value: 2.702_593_730_354_701_4,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 100.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            published_value_4dp: 6.6997,
            exact_value: 6.699_696_963_531_342,
        },
        EuropeanCase {
            option_type: OptionType::Call,
            strike: 100.0,
            spot: 110.0,
            dividend_yield: 0.10,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.25,
            published_value_4dp: 12.7857,
            exact_value: 12.785_678_929_758_692,
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
            published_value_4dp: 25.3533,
            exact_value: 25.353_355_271_810_194,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 1.00,
            vol: 0.30,
            published_value_4dp: 23.7884,
            exact_value: 23.788_436_501_680_813,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 0.20,
            vol: 0.30,
            published_value_4dp: 10.7190,
            exact_value: 10.719_018_364_058_12,
        },
        LookbackCase {
            option_type: OptionType::Call,
            minmax: 100.0,
            spot: 110.0,
            dividend_yield: 0.00,
            rate: 0.05,
            expiry: 0.20,
            vol: 0.30,
            published_value_4dp: 14.4597,
            exact_value: 14.459_698_511_705_63,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 100.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            published_value_4dp: 15.3526,
            exact_value: 15.352_555_467_893_225,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 110.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            published_value_4dp: 16.8468,
            exact_value: 16.846_772_680_120_75,
        },
        LookbackCase {
            option_type: OptionType::Put,
            minmax: 120.0,
            spot: 100.0,
            dividend_yield: 0.00,
            rate: 0.10,
            expiry: 0.50,
            vol: 0.30,
            published_value_4dp: 21.0645,
            exact_value: 21.064_537_607_933_733,
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

/// Compare two independently generated binary64 closed-form results.  The
/// allowance is expressed in accumulated floating-point roundoff, not in
/// economic price units.
fn assert_binary64_close(label: &str, actual: f64, expected: f64) {
    // The stability-preserving log-domain Black-Scholes path differs from
    // QuantLib's operation ordering by at most 26 scaled epsilons on this grid.
    let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label}: expected={expected:.17e}, actual={actual:.17e}, error={error:.3e}, binary64 budget={tolerance:.3e}"
    );
}

fn assert_within_published_precision(label: &str, exact: f64, published_4dp: f64) {
    let source_precision = 1.0e-4;
    let error = (exact - published_4dp).abs();
    assert!(
        error < source_precision,
        "{label}: exact={exact:.17e}, published={published_4dp:.4}, error={error:.3e}"
    );
}

#[test]
fn quantlib_haug_european_reference_values() {
    // Published fixtures: QuantLib europeanoption.cpp testValues (Haug 1998,
    // pp. 2-8 and 24), quoted to four decimal places.  `exact_value` was
    // independently regenerated with QuantLib 1.43 BlackCalculator from the
    // same spot, forward, discount factor and standard deviation.  The
    // published number is retained only as a source-provenance check.
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

        let price = engine
            .price(&option, &market)
            .expect("pricing succeeds")
            .price;
        assert_within_published_precision(
            "QuantLib/Haug European published fixture",
            case.exact_value,
            case.published_value_4dp,
        );
        assert_binary64_close(
            "QuantLib BlackCalculator European price",
            price,
            case.exact_value,
        );
    }
}

#[test]
fn quantlib_haug_lookback_floating_reference_values() {
    // Published fixtures: QuantLib lookbackoptions.cpp (Haug 1998 pp. 61-62,
    // Broadie-Glasserman-Kou 1999 data), quoted to four decimal places.
    // `exact_value` is the Goldman-Sosin-Gatto closed form evaluated
    // independently with SciPy 1.17.1's double-precision ndtr.  This avoids
    // treating a rounded book fixture as an exact economic target.
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

        let price = engine
            .price(&option, &market)
            .expect("pricing succeeds")
            .price;
        assert_within_published_precision(
            "QuantLib/Haug lookback published fixture",
            case.exact_value,
            case.published_value_4dp,
        );
        assert_binary64_close(
            "Goldman-Sosin-Gatto floating-lookback price",
            price,
            case.exact_value,
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

        let lhs = call - put;
        let rhs = spot * (-q * expiry).exp() - strike * (-r * expiry).exp();
        let tolerance = 64.0 * f64::EPSILON * (call.abs() + put.abs() + rhs.abs()).max(1.0);
        let error = (lhs - rhs).abs();
        assert!(
            error <= tolerance,
            "put-call parity: lhs={lhs:.17e}, rhs={rhs:.17e}, error={error:.3e}, roundoff budget={tolerance:.3e}"
        );
    }
}

#[test]
fn quantlib_european_implied_vol_round_trip() {
    let engine = BlackScholesEngine::new();
    const PRICE_TOLERANCE: f64 = 1.0e-12;

    for case in european_cases() {
        let option = build_vanilla_case(&case);
        let market = Market::builder()
            .spot(case.spot)
            .rate(case.rate)
            .dividend_yield(case.dividend_yield)
            .flat_vol(case.vol)
            .build()
            .expect("valid market");

        let price = engine
            .price(&option, &market)
            .expect("pricing succeeds")
            .price;

        let spot_no_dividend = case.spot * (-case.dividend_yield * case.expiry).exp();
        let recovered = implied_vol_newton(
            to_pricing_option_type(case.option_type),
            spot_no_dividend,
            case.strike,
            case.rate,
            case.expiry,
            price,
            PRICE_TOLERANCE,
            100,
        )
        .expect("implied vol converges");

        // Newton terminates on price error.  Convert that stopping criterion
        // into a volatility error budget with the local analytic vega, and
        // add a binary64 price-evaluation allowance.  This measures the
        // solver's actual conditioning instead of accepting a blanket IV band.
        let vega = bs_vega(
            spot_no_dividend,
            case.strike,
            case.rate,
            0.0,
            case.vol,
            case.expiry,
        );
        let price_roundoff =
            256.0 * f64::EPSILON * (price.abs() + spot_no_dividend + case.strike).max(1.0);
        let tolerance = 1.25 * (PRICE_TOLERANCE + price_roundoff) / vega.abs()
            + 32.0 * f64::EPSILON * case.vol.abs().max(1.0);
        let error = (recovered - case.vol).abs();
        assert!(
            error <= tolerance,
            "implied vol: expected={:.17e}, recovered={recovered:.17e}, error={error:.3e}, vega={vega:.3e}, conditioned budget={tolerance:.3e}",
            case.vol,
        );
    }
}
