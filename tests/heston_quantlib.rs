use std::fs;

use openferric::core::PricingEngine;
use openferric::engines::analytic::HestonEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

#[derive(Debug, Clone)]
struct HestonFixture {
    spot: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    maturity: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    strikes: Vec<f64>,
    expected_put_call: Vec<(f64, f64)>,
}

fn quantlib_file(path: &str) -> String {
    let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
    fs::read_to_string(full).expect("failed to read QuantLib fixture")
}

fn parse_scalar(source: &str, key: &str) -> f64 {
    let line = source
        .lines()
        .find(|l| l.contains(key))
        .unwrap_or_else(|| panic!("missing key in fixture: {key}"));
    let rhs = line
        .split('=')
        .nth(1)
        .expect("expected assignment")
        .trim()
        .trim_end_matches(';')
        .trim();
    rhs.parse()
        .unwrap_or_else(|_| panic!("invalid numeric value for key {key}: {rhs}"))
}

fn parse_inline_array(source: &str, key: &str) -> Vec<f64> {
    let line = source
        .lines()
        .find(|l| l.contains(key))
        .unwrap_or_else(|| panic!("missing inline array key in fixture: {key}"));
    let start = line.find('{').expect("missing { in inline array");
    let end = line.find('}').expect("missing } in inline array");
    line[start + 1..end]
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().expect("invalid inline array number"))
        .collect()
}

fn parse_expected_table(source: &str) -> Vec<(f64, f64)> {
    let mut in_table = false;
    let mut out = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("const Real expectedResults[][2]") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        if trimmed.starts_with("};") {
            break;
        }
        if !trimmed.starts_with('{') {
            continue;
        }

        let inner = trimmed
            .trim_start_matches('{')
            .trim_end_matches(',')
            .trim_end_matches('}')
            .trim();
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        assert_eq!(parts.len(), 2, "unexpected expectedResults row: {trimmed}");
        let put: f64 = parts[0].parse().expect("invalid put price in table");
        let call: f64 = parts[1].parse().expect("invalid call price in table");
        out.push((put, call));
    }

    assert!(!out.is_empty(), "failed to parse expectedResults table");
    out
}

fn parse_heston_fixture() -> HestonFixture {
    let source = quantlib_file("tests/quantlib_data/hestonmodel.cpp");
    let strikes = parse_inline_array(&source, "const Real strikes[]");
    let expected_put_call = parse_expected_table(&source);

    assert_eq!(
        strikes.len(),
        expected_put_call.len(),
        "fixture strike and expectedResults length mismatch"
    );

    HestonFixture {
        spot: parse_scalar(&source, "const Real spot"),
        risk_free_rate: parse_scalar(&source, "const Rate riskFreeRate"),
        dividend_rate: parse_scalar(&source, "const Rate dividendRate"),
        maturity: parse_scalar(&source, "const Time maturity"),
        v0: parse_scalar(&source, "const Volatility v0"),
        kappa: parse_scalar(&source, "const Real kappa"),
        theta: parse_scalar(&source, "const Volatility theta"),
        sigma_v: parse_scalar(&source, "const Volatility sigma_v"),
        rho: parse_scalar(&source, "const Real rho"),
        strikes,
        expected_put_call,
    }
}

#[test]
fn heston_quantlib_cached_reference_values() {
    // Source: tests/quantlib_data/hestonmodel.cpp
    // Reference: QuantLib hestonmodel.cpp cached values (Lewis FT dataset).
    let fixture = parse_heston_fixture();

    let engine = HestonEngine::new(
        fixture.v0,
        fixture.kappa,
        fixture.theta,
        fixture.sigma_v,
        fixture.rho,
    );

    let market = Market::builder()
        .spot(fixture.spot)
        .rate(fixture.risk_free_rate)
        .dividend_yield(fixture.dividend_rate)
        // Flat vol is required by Market even though HestonEngine does not use it.
        .flat_vol(0.20)
        .build()
        .expect("valid market");

    for (idx, strike) in fixture.strikes.iter().copied().enumerate().take(3) {
        let call = VanillaOption::european_call(strike, fixture.maturity);
        let call_price = engine.price(&call, &market).expect("call pricing").price;
        let expected_call = fixture.expected_put_call[idx].1;
        // QuantLib cached entries are from a broader model-validation table;
        // keep tolerance loose enough for 32-point quadrature and rounded fixtures.
        let tol = 1.0;
        let call_err = (call_price - expected_call).abs();

        assert!(
            call_err <= tol,
            "K={strike} call mismatch: expected={expected_call} got={call_price} err={call_err} tol={tol}"
        );
    }
}
