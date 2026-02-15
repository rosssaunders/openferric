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

fn quantlib_file(path: &str) -> String {
    let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
    fs::read_to_string(full).expect("failed to read QuantLib fixture")
}

fn parse_option_type(raw: &str) -> OptionType {
    match raw {
        "Option::Call" => OptionType::Call,
        "Option::Put" => OptionType::Put,
        other => panic!("unsupported option type token: {other}"),
    }
}

fn parse_barrier_haug_values() -> Vec<BarrierCase> {
    let source = quantlib_file("tests/quantlib_data/barrieroption.cpp");

    let mut in_haug_case = false;
    let mut in_values_array = false;
    let mut out = Vec::new();

    for (idx, line) in source.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();

        if trimmed.contains("BOOST_AUTO_TEST_CASE(testHaugValues)") {
            in_haug_case = true;
            continue;
        }
        if !in_haug_case {
            continue;
        }

        if trimmed.starts_with("NewBarrierOptionData values[] = {") {
            in_values_array = true;
            continue;
        }

        if in_values_array && trimmed.starts_with("};") {
            break;
        }

        if !in_values_array || !trimmed.starts_with('{') || !trimmed.contains("Barrier::") {
            continue;
        }

        let inner = trimmed
            .trim_start_matches('{')
            .trim_end_matches(',')
            .trim_end_matches('}')
            .trim();

        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        assert_eq!(
            parts.len(),
            13,
            "unexpected barrier fields at line {line_no}"
        );

        out.push(BarrierCase {
            barrier_type: parts[0].to_string(),
            barrier: parts[1].parse().expect("invalid barrier"),
            rebate: parts[2].parse().expect("invalid rebate"),
            option_type: parse_option_type(parts[3]),
            exercise_type: parts[4].to_string(),
            strike: parts[5].parse().expect("invalid strike"),
            spot: parts[6].parse().expect("invalid spot"),
            dividend: parts[7].parse().expect("invalid dividend"),
            rate: parts[8].parse().expect("invalid rate"),
            expiry: parts[9].parse().expect("invalid expiry"),
            vol: parts[10].parse().expect("invalid vol"),
            expected: parts[11].parse().expect("invalid expected"),
            tolerance: parts[12].parse().expect("invalid tolerance"),
            line: line_no,
        });
    }

    assert!(
        !out.is_empty(),
        "failed to parse barrieroption.cpp Haug table"
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
    // Source: tests/quantlib_data/barrieroption.cpp:333-418
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
    // Source: tests/quantlib_data/barrieroption.cpp:159-231 (testParity).
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
