use std::fs;

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::{BarrierAnalyticEngine, BlackScholesEngine};
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::Market;

#[derive(Debug, Clone)]
struct BarrierCase {
    barrier_type: String,
    barrier: f64,
    rebate: f64,
    option_type: OptionType,
    exercise_type: String,
    strike: f64,
    spot: f64,
    dividend: f64,
    rate: f64,
    expiry: f64,
    vol: f64,
    expected: f64,
    tolerance: f64,
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

fn parse_barrier_haug_values() -> Vec<BarrierCase> {
    let source = fixture_file("tests/fixtures/barrier_haug_values.csv");
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
            14,
            "unexpected barrier fixture fields at line {fixture_line}"
        );

        out.push(BarrierCase {
            line: parts[0].parse().expect("invalid source line"),
            barrier_type: parts[1].to_string(),
            barrier: parts[2].parse().expect("invalid barrier"),
            rebate: parts[3].parse().expect("invalid rebate"),
            option_type: parse_option_type(parts[4]),
            exercise_type: parts[5].to_string(),
            strike: parts[6].parse().expect("invalid strike"),
            spot: parts[7].parse().expect("invalid spot"),
            dividend: parts[8].parse().expect("invalid dividend"),
            rate: parts[9].parse().expect("invalid rate"),
            expiry: parts[10].parse().expect("invalid expiry"),
            vol: parts[11].parse().expect("invalid vol"),
            expected: parts[12].parse().expect("invalid expected"),
            tolerance: parts[13].parse().expect("invalid tolerance"),
        });
    }

    assert_eq!(
        out.len(),
        96,
        "expected 96 barrier rows in QuantLib Haug fixture"
    );
    out
}

fn build_barrier_option(case: &BarrierCase) -> BarrierOption {
    let mut builder = BarrierOption::builder()
        .strike(case.strike)
        .expiry(case.expiry)
        .rebate(case.rebate);

    builder = match case.option_type {
        OptionType::Call => builder.call(),
        OptionType::Put => builder.put(),
    };

    builder = match case.barrier_type.as_str() {
        "Barrier::DownOut" => builder.down_and_out(case.barrier),
        "Barrier::DownIn" => builder.down_and_in(case.barrier),
        "Barrier::UpOut" => builder.up_and_out(case.barrier),
        "Barrier::UpIn" => builder.up_and_in(case.barrier),
        other => panic!("unsupported barrier type token: {other}"),
    };

    builder.build().expect("valid barrier option")
}

#[test]
fn barrier_quantlib_haug_reference_values() {
    // Fixture: tests/fixtures/barrier_haug_values.csv
    // Original source: vendor/QuantLib/test-suite/barrieroption.cpp:333-449
    // Reference: E.G. Haug, "Option Pricing Formulas" (1998), p. 72.
    let cases = parse_barrier_haug_values();
    let engine = BarrierAnalyticEngine::new();

    let mut checked = 0usize;

    for case in &cases {
        if case.exercise_type != "european" {
            continue;
        }

        let option = build_barrier_option(case);
        let market = Market::builder()
            .spot(case.spot)
            .rate(case.rate)
            .dividend_yield(case.dividend)
            .flat_vol(case.vol)
            .build()
            .expect("valid market");

        let price = engine
            .price(&option, &market)
            .expect("pricing succeeds")
            .price;
        let err = (price - case.expected).abs();

        assert!(
            err <= case.tolerance,
            "line {}: {:?} {} S={} K={} H={} rebate={} q={} r={} t={} vol={} expected={} got={} err={} tol={}",
            case.line,
            case.option_type,
            case.barrier_type,
            case.spot,
            case.strike,
            case.barrier,
            case.rebate,
            case.dividend,
            case.rate,
            case.expiry,
            case.vol,
            case.expected,
            price,
            err,
            case.tolerance
        );
        checked += 1;
    }

    assert_eq!(checked, 72, "expected all 72 European Haug barrier rows");
}

#[test]
fn barrier_quantlib_knock_in_out_parity_equals_vanilla() {
    // Source: vendor/QuantLib/test-suite/barrieroption.cpp:159-231 (testParity).
    let expiry = 0.5;
    let spot = 100.0;
    let strike = 100.0;
    let barrier = 90.0;
    let rate = 0.01;
    let dividend = 0.0;
    let vol = 0.20;

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend)
        .flat_vol(vol)
        .build()
        .expect("valid market");

    let knock_in = BarrierOption::builder()
        .call()
        .strike(strike)
        .expiry(expiry)
        .down_and_in(barrier)
        .rebate(0.0)
        .build()
        .expect("valid knock-in");

    let knock_out = BarrierOption::builder()
        .call()
        .strike(strike)
        .expiry(expiry)
        .down_and_out(barrier)
        .rebate(0.0)
        .build()
        .expect("valid knock-out");

    let vanilla = VanillaOption::european_call(strike, expiry);

    let barrier_engine = BarrierAnalyticEngine::new();
    let vanilla_engine = BlackScholesEngine::new();

    let in_px = barrier_engine
        .price(&knock_in, &market)
        .expect("knock-in prices")
        .price;
    let out_px = barrier_engine
        .price(&knock_out, &market)
        .expect("knock-out prices")
        .price;
    let vanilla_px = vanilla_engine
        .price(&vanilla, &market)
        .expect("vanilla prices")
        .price;

    assert!(
        ((in_px + out_px) - vanilla_px).abs() <= 1e-7,
        "in+out parity failed: in={} out={} vanilla={}",
        in_px,
        out_px,
        vanilla_px
    );
}
