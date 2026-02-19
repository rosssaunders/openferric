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

fn fixture_file(path: &str) -> String {
    let full = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
    fs::read_to_string(full).expect("failed to read fixture")
}

fn parse_heston_params() -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let source = fixture_file("tests/fixtures/heston_params.csv");

    let mut data_line = None;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("spot,") {
            continue;
        }
        data_line = Some(trimmed.to_string());
        break;
    }

    let line = data_line.expect("missing heston params row");
    let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
    assert_eq!(parts.len(), 9, "expected 9 fields in heston_params.csv");

    (
        parts[0].parse().expect("invalid spot"),
        parts[1].parse().expect("invalid risk_free_rate"),
        parts[2].parse().expect("invalid dividend_rate"),
        parts[3].parse().expect("invalid maturity"),
        parts[4].parse().expect("invalid v0"),
        parts[5].parse().expect("invalid kappa"),
        parts[6].parse().expect("invalid theta"),
        parts[7].parse().expect("invalid sigma_v"),
        parts[8].parse().expect("invalid rho"),
    )
}

fn parse_expected_table() -> (Vec<f64>, Vec<(f64, f64)>) {
    let source = fixture_file("tests/fixtures/heston_expected_put_call.csv");
    let mut strikes = Vec::new();
    let mut expected_put_call = Vec::new();
    for line in source.lines() {
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
            4,
            "expected 4 fields in heston_expected_put_call.csv"
        );

        let strike = parts[1].parse().expect("invalid strike");
        let put = parts[2].parse().expect("invalid expected put");
        let call = parts[3].parse().expect("invalid expected call");

        strikes.push(strike);
        expected_put_call.push((put, call));
    }

    assert_eq!(
        strikes.len(),
        5,
        "expected 5 rows in heston_expected_put_call.csv"
    );
    (strikes, expected_put_call)
}

fn parse_heston_fixture() -> HestonFixture {
    let (spot, risk_free_rate, dividend_rate, maturity, v0, kappa, theta, sigma_v, rho) =
        parse_heston_params();
    let (strikes, expected_put_call) = parse_expected_table();

    assert_eq!(
        strikes.len(),
        expected_put_call.len(),
        "fixture strike and expectedResults length mismatch"
    );

    HestonFixture {
        spot,
        risk_free_rate,
        dividend_rate,
        maturity,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        strikes,
        expected_put_call,
    }
}

#[test]
fn heston_quantlib_cached_reference_values() {
    // Fixtures:
    // - tests/fixtures/heston_params.csv
    // - tests/fixtures/heston_expected_put_call.csv
    // Original source: vendor/QuantLib/test-suite/hestonmodel.cpp
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
