use std::fs;

use openferric::core::{AsianSpec, Averaging, OptionType, PricingEngine, StrikeType};
use openferric::engines::analytic::GeometricAsianEngine;
use openferric::engines::monte_carlo::MonteCarloPricingEngine;
use openferric::instruments::AsianOption;
use openferric::market::Market;

#[derive(Debug, Clone)]
struct ArithmeticCase {
    option_type: OptionType,
    underlying: f64,
    strike: f64,
    dividend_yield: f64,
    risk_free_rate: f64,
    first: f64,
    length: f64,
    fixings: usize,
    volatility: f64,
    control_variate: bool,
    expected: f64,
    line: usize,
}

fn fixture_file(path: &str) -> String {
    let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
    fs::read_to_string(full).expect("failed to read fixture")
}

fn parse_option_type(raw: &str) -> OptionType {
    match raw {
        "Call" => OptionType::Call,
        "Put" => OptionType::Put,
        "Option::Call" => OptionType::Call,
        "Option::Put" => OptionType::Put,
        other => panic!("unsupported option type token: {other}"),
    }
}

fn parse_f64_expr(raw: &str) -> f64 {
    let token = raw.trim();
    if let Some((lhs, rhs)) = token.split_once('/') {
        let a: f64 = lhs.trim().parse().expect("invalid lhs fraction");
        let b: f64 = rhs.trim().parse().expect("invalid rhs fraction");
        return a / b;
    }
    token.parse().expect("invalid float token")
}

fn parse_arithmetic_cases4() -> Vec<ArithmeticCase> {
    let source = fixture_file("tests/fixtures/asian_arithmetic_cases4.csv");
    let mut out = Vec::new();

    for (idx, line) in source.lines().enumerate() {
        let fixture_line = idx + 1;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("source_line,") {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').map(|p| p.trim()).collect();
        assert_eq!(
            parts.len(),
            12,
            "unexpected arithmetic fixture fields at line {fixture_line}"
        );

        out.push(ArithmeticCase {
            line: parts[0].parse().expect("invalid source line"),
            option_type: parse_option_type(parts[1]),
            underlying: parse_f64_expr(parts[2]),
            strike: parse_f64_expr(parts[3]),
            dividend_yield: parse_f64_expr(parts[4]),
            risk_free_rate: parse_f64_expr(parts[5]),
            first: parse_f64_expr(parts[6]),
            length: parse_f64_expr(parts[7]),
            fixings: parse_f64_expr(parts[8]) as usize,
            volatility: parse_f64_expr(parts[9]),
            control_variate: match parts[10] {
                "true" => true,
                "false" => false,
                other => panic!("invalid bool token {other}"),
            },
            expected: parse_f64_expr(parts[11]),
        });
    }

    assert_eq!(out.len(), 30, "expected 30 rows in cases4 fixture");
    out
}

fn build_observation_times(first: f64, length: f64, fixings: usize) -> Vec<f64> {
    if fixings == 0 {
        return Vec::new();
    }
    if fixings == 1 {
        return vec![first];
    }

    let dt = length / (fixings as f64 - 1.0);
    (0..fixings).map(|i| first + i as f64 * dt).collect()
}

#[test]
fn asian_geometric_quantlib_discrete_reference_value() {
    // Source: vendor/QuantLib/test-suite/asianoptions.cpp:376-436.
    // Reference: Clewlow & Strickland, "Implementing Derivatives Models", pp. 118-123.
    let strike = 100.0;
    let expiry = 1.0;
    let option = AsianOption::new(
        OptionType::Call,
        strike,
        expiry,
        AsianSpec {
            averaging: Averaging::Geometric,
            strike_type: StrikeType::Fixed,
            observation_times: (1..=10).map(|i| i as f64 / 10.0).collect(),
        },
    );

    let market = Market::builder()
        .spot(100.0)
        .rate(0.06)
        .dividend_yield(0.03)
        .flat_vol(0.20)
        .build()
        .expect("valid market");

    let price = GeometricAsianEngine::new()
        .price(&option, &market)
        .expect("pricing succeeds")
        .price;

    let expected = 5.3425606635;
    let tolerance = 1.0e-4;
    assert!(
        (price - expected).abs() <= tolerance,
        "discrete geometric mismatch: expected={expected} got={price}"
    );
}

#[test]
fn asian_geometric_quantlib_continuous_haug_value_via_dense_discrete_schedule() {
    // Source: vendor/QuantLib/test-suite/asianoptions.cpp:153-229.
    // Reference: E.G. Haug, "Option Pricing Formulas" (1998), pp. 96-97.
    // QuantLib itself checks a dense discrete approximation against the continuous value.
    let strike = 85.0;
    let expiry = 0.25;
    let option = AsianOption::new(
        OptionType::Put,
        strike,
        expiry,
        AsianSpec {
            averaging: Averaging::Geometric,
            strike_type: StrikeType::Fixed,
            observation_times: (1..=90).map(|d| d as f64 / 360.0).collect(),
        },
    );

    let market = Market::builder()
        .spot(80.0)
        .rate(0.05)
        .dividend_yield(-0.03)
        .flat_vol(0.20)
        .build()
        .expect("valid market");

    let price = GeometricAsianEngine::new()
        .price(&option, &market)
        .expect("pricing succeeds")
        .price;

    let expected = 4.6922;
    let tolerance = 4.0e-3;
    assert!(
        (price - expected).abs() <= tolerance,
        "continuous geometric approximation mismatch: expected={expected} got={price}"
    );
}

#[test]
fn asian_arithmetic_mc_converges_to_quantlib_reference_within_two_stderr() {
    // Fixture: tests/fixtures/asian_arithmetic_cases4.csv
    // Original source: vendor/QuantLib/test-suite/asianoptions.cpp:685-746 (cases4 table).
    // Reference: Levy (1997), in Clewlow & Strickland (eds.), "Exotic Options: The State of the Art".
    let cases = parse_arithmetic_cases4();

    let case = cases
        .iter()
        .find(|c| (c.first - (1.0 / 12.0)).abs() <= 1e-15 && c.fixings == 12)
        .expect("expected 12-fixing case from QuantLib table")
        .clone();

    assert_eq!(case.option_type, OptionType::Put);
    assert!(
        case.control_variate,
        "expected control variate row from QuantLib"
    );

    let observation_times = build_observation_times(case.first, case.length, case.fixings);
    let expiry = *observation_times
        .last()
        .expect("non-empty observation schedule required");

    let option = AsianOption::new(
        case.option_type,
        case.strike,
        expiry,
        AsianSpec {
            averaging: Averaging::Arithmetic,
            strike_type: StrikeType::Fixed,
            observation_times,
        },
    );

    let market = Market::builder()
        .spot(case.underlying)
        .rate(case.risk_free_rate)
        .dividend_yield(case.dividend_yield)
        .flat_vol(case.volatility)
        .build()
        .expect("valid market");

    let coarse = MonteCarloPricingEngine::new(20_000, 128, 42)
        .price(&option, &market)
        .expect("coarse MC succeeds");
    let fine = MonteCarloPricingEngine::new(80_000, 128, 42)
        .price(&option, &market)
        .expect("fine MC succeeds");

    let coarse_err = (coarse.price - case.expected).abs();
    let fine_err = (fine.price - case.expected).abs();
    let fine_stderr = fine.stderr.expect("MC stderr must be present");

    assert!(
        fine_err <= 2.0 * fine_stderr,
        "line {}: expected={} fine_price={} abs_err={} stderr={}",
        case.line,
        case.expected,
        fine.price,
        fine_err,
        fine_stderr
    );

    assert!(
        fine_err <= coarse_err + 5.0e-3,
        "line {}: convergence check failed (coarse_err={} fine_err={})",
        case.line,
        coarse_err,
        fine_err
    );
}
