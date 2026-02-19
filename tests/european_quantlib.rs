use std::collections::HashMap;
use std::fs;

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::BlackScholesEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

#[derive(Debug, Clone)]
struct EuropeanCase {
    option_type: OptionType,
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
        other => panic!("unsupported option type token: {other}"),
    }
}

fn parse_european_haug_values() -> Vec<EuropeanCase> {
    let source = fixture_file("tests/fixtures/european_haug_values.csv");
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
            10,
            "unexpected european fixture fields at line {fixture_line}"
        );

        out.push(EuropeanCase {
            line: parts[0].parse().expect("invalid source line"),
            option_type: parse_option_type(parts[1]),
            strike: parts[2].parse().expect("invalid strike"),
            spot: parts[3].parse().expect("invalid spot"),
            dividend: parts[4].parse().expect("invalid dividend"),
            rate: parts[5].parse().expect("invalid rate"),
            expiry: parts[6].parse().expect("invalid expiry"),
            vol: parts[7].parse().expect("invalid vol"),
            expected: parts[8].parse().expect("invalid expected value"),
            tolerance: parts[9].parse().expect("invalid tolerance"),
        });
    }

    assert!(!out.is_empty(), "failed to parse european_haug_values.csv");
    out
}

#[test]
fn european_quantlib_haug_reference_values() {
    // Fixture: tests/fixtures/european_haug_values.csv
    // Original source: vendor/QuantLib/test-suite/europeanoption.cpp:288-336
    // Reference: E.G. Haug, "Option Pricing Formulas" (1998), pp. 2-8, 24, 27.
    let cases = parse_european_haug_values();
    let engine = BlackScholesEngine::new();

    for case in &cases {
        let option = match case.option_type {
            OptionType::Call => VanillaOption::european_call(case.strike, case.expiry),
            OptionType::Put => VanillaOption::european_put(case.strike, case.expiry),
        };

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
            "line {}: {:?} S={} K={} q={} r={} t={} vol={} expected={} got={} err={} tol={}",
            case.line,
            case.option_type,
            case.spot,
            case.strike,
            case.dividend,
            case.rate,
            case.expiry,
            case.vol,
            case.expected,
            price,
            err,
            case.tolerance
        );
    }
}

#[test]
fn european_quantlib_put_call_parity() {
    // Fixture source dataset: tests/fixtures/european_haug_values.csv.
    let cases = parse_european_haug_values();
    let engine = BlackScholesEngine::new();

    let mut grouped: HashMap<(u64, u64, u64, u64, u64, u64), (Option<f64>, Option<f64>)> =
        HashMap::new();

    for case in &cases {
        let key = (
            case.spot.to_bits(),
            case.strike.to_bits(),
            case.dividend.to_bits(),
            case.rate.to_bits(),
            case.expiry.to_bits(),
            case.vol.to_bits(),
        );

        let option = match case.option_type {
            OptionType::Call => VanillaOption::european_call(case.strike, case.expiry),
            OptionType::Put => VanillaOption::european_put(case.strike, case.expiry),
        };

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

        let entry = grouped.entry(key).or_insert((None, None));
        match case.option_type {
            OptionType::Call => entry.0 = Some(price),
            OptionType::Put => entry.1 = Some(price),
        }
    }

    let mut checked = 0usize;
    for (key, (call, put)) in grouped {
        if let (Some(c), Some(p)) = (call, put) {
            let spot = f64::from_bits(key.0);
            let strike = f64::from_bits(key.1);
            let q = f64::from_bits(key.2);
            let r = f64::from_bits(key.3);
            let t = f64::from_bits(key.4);

            let rhs = spot * (-q * t).exp() - strike * (-r * t).exp();
            assert!(
                ((c - p) - rhs).abs() <= 1e-10,
                "parity failed for S={spot} K={strike} q={q} r={r} t={t}: C-P={} rhs={}",
                c - p,
                rhs
            );
            checked += 1;
        }
    }

    assert!(checked > 0, "no call/put parity pairs found");
}
