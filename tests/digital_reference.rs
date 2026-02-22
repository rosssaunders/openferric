//! Digital option reference tests.
//!
//! Sources:
//! - Fang & Oosterlee (2008) Table 3: cash-or-nothing call via COS method
//! - Haug "Option Pricing Formulas" (1998): cash-or-nothing, asset-or-nothing
//! - Black-Scholes decomposition: vanilla = asset-or-nothing - K * cash-or-nothing
//!
//! These test the digital option analytic engine against known reference values.

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::digital::DigitalAnalyticEngine;
use openferric::instruments::digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
use openferric::market::Market;

fn make_market(spot: f64, rate: f64, dividend_yield: f64, vol: f64) -> Market {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .expect("market build failed")
}

// =======================================================================
// Cash-or-nothing call: analytical verification
// S=100, K=100, r=0.05, q=0, sigma=0.25, T=1.0, cash=1
// d2 = (ln(1) + (0.05 - 0 - 0.5*0.0625)*1) / (0.25) = 0.01875/0.25 = 0.075
// N(0.075) = 0.52989
// Price = exp(-0.05) * 0.52989 = 0.95123 * 0.52989 = 0.50407
// =======================================================================
#[test]
fn cash_or_nothing_call_analytical() {
    let instrument = CashOrNothingOption::new(OptionType::Call, 100.0, 1.0, 1.0);
    let market = make_market(100.0, 0.05, 0.0, 0.25);
    let engine = DigitalAnalyticEngine::new();
    let result = engine.price(&instrument, &market).expect("pricing failed");
    let expected = 0.5041;
    let err = (result.price - expected).abs();
    assert!(
        err < 0.01,
        "CoN call analytical: got {}, expected {expected}, err={err}",
        result.price
    );
}

// =======================================================================
// Haug reference: cash-or-nothing call
// S=100, K=80, r=0.06, q=0.06, sigma=0.35, T=0.75, cash=10
// Reference: ~6.9404 (from existing unit tests in the codebase)
// =======================================================================
#[test]
fn haug_cash_or_nothing_call() {
    let instrument = CashOrNothingOption::new(OptionType::Call, 80.0, 10.0, 0.75);
    let market = make_market(100.0, 0.06, 0.06, 0.35);
    let engine = DigitalAnalyticEngine::new();
    let result = engine.price(&instrument, &market).expect("pricing failed");
    // N(d2) with d2 = (ln(100/80) + (0.06-0.06-0.5*0.35^2)*0.75) / (0.35*sqrt(0.75))
    // d2 = (0.22314 - 0.045938) / 0.30311 = 0.5848
    // N(0.5848) = 0.7206
    // Price = 10 * exp(-0.06*0.75) * N(d2) = 10 * 0.9560 * 0.7206 = 6.889
    assert!(
        result.price > 6.0 && result.price < 8.0,
        "Haug CoN call: got {}, expected ~6.9",
        result.price
    );
}

// =======================================================================
// Haug reference: cash-or-nothing put
// Same params but put => price = cash * exp(-rT) * N(-d2)
// =======================================================================
#[test]
fn haug_cash_or_nothing_put() {
    let instrument = CashOrNothingOption::new(OptionType::Put, 80.0, 10.0, 0.75);
    let market = make_market(100.0, 0.06, 0.06, 0.35);
    let engine = DigitalAnalyticEngine::new();
    let result = engine.price(&instrument, &market).expect("pricing failed");
    // N(-d2) = 1 - N(d2) ≈ 0.2794
    // Price = 10 * 0.9560 * 0.2794 ≈ 2.671
    assert!(
        result.price > 2.0 && result.price < 4.0,
        "Haug CoN put: got {}, expected ~2.7",
        result.price
    );
}

// =======================================================================
// Asset-or-nothing: call + put parity = S * exp(-qT)
// =======================================================================
#[test]
fn asset_or_nothing_call_put_parity() {
    let spot = 100.0;
    let rate = 0.05;
    let q = 0.02;
    let vol = 0.25;
    let t = 1.0;
    let strike = 100.0;

    let call = AssetOrNothingOption::new(OptionType::Call, strike, t);
    let put = AssetOrNothingOption::new(OptionType::Put, strike, t);
    let market = make_market(spot, rate, q, vol);
    let engine = DigitalAnalyticEngine::new();

    let call_price = engine.price(&call, &market).expect("call failed").price;
    let put_price = engine.price(&put, &market).expect("put failed").price;

    let expected_sum = spot * (-q * t).exp();
    let err = (call_price + put_price - expected_sum).abs();
    assert!(
        err < 0.001,
        "AoN parity: call={call_price} + put={put_price} = {}, expected {expected_sum}, err={err}",
        call_price + put_price
    );
}

// =======================================================================
// Cash-or-nothing: call + put parity = cash * exp(-rT)
// =======================================================================
#[test]
fn cash_or_nothing_call_put_parity() {
    let cash = 15.0;
    let rate = 0.05;
    let t = 1.0;

    let call = CashOrNothingOption::new(OptionType::Call, 100.0, cash, t);
    let put = CashOrNothingOption::new(OptionType::Put, 100.0, cash, t);
    let market = make_market(100.0, rate, 0.02, 0.30);
    let engine = DigitalAnalyticEngine::new();

    let call_price = engine.price(&call, &market).expect("call failed").price;
    let put_price = engine.price(&put, &market).expect("put failed").price;

    let expected_sum = cash * (-rate * t).exp();
    let err = (call_price + put_price - expected_sum).abs();
    assert!(
        err < 0.001,
        "CoN parity: call={call_price} + put={put_price} = {}, expected {expected_sum}, err={err}",
        call_price + put_price
    );
}

// =======================================================================
// Gap option: equal trigger and payoff strikes = vanilla
// gap_call(K,K) = BS_call(K)
// =======================================================================
#[test]
fn gap_option_equals_vanilla_when_strikes_equal() {
    let strike = 100.0;
    let t = 0.5;
    let spot = 105.0;
    let rate = 0.05;
    let q = 0.0;
    let vol = 0.25;

    let gap = GapOption::new(OptionType::Call, strike, strike, t);
    let market = make_market(spot, rate, q, vol);
    let engine = DigitalAnalyticEngine::new();
    let gap_price = engine.price(&gap, &market).expect("gap failed").price;

    // Compare with Black-Scholes vanilla call
    let bs = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        spot,
        strike,
        rate,
        vol,
        t,
    );
    let err = (gap_price - bs).abs();
    assert!(
        err < 0.01,
        "Gap=vanilla: gap={gap_price}, bs={bs}, err={err}"
    );
}

// =======================================================================
// BS decomposition: vanilla_call = AoN_call - K * CoN_call
// Uses q=0 since the legacy black_scholes_price doesn't take dividend_yield
// =======================================================================
#[test]
fn bs_decomposition_into_digitals() {
    let spot = 110.0;
    let strike = 100.0;
    let rate = 0.04;
    let q = 0.0; // no dividend so legacy BS function matches
    let vol = 0.20;
    let t = 0.75;

    let market = make_market(spot, rate, q, vol);
    let engine = DigitalAnalyticEngine::new();

    let aon = AssetOrNothingOption::new(OptionType::Call, strike, t);
    let con = CashOrNothingOption::new(OptionType::Call, strike, 1.0, t);

    let aon_price = engine.price(&aon, &market).expect("aon failed").price;
    let con_price = engine.price(&con, &market).expect("con failed").price;

    let digital_decomp = aon_price - strike * con_price;

    let bs = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        spot,
        strike,
        rate,
        vol,
        t,
    );

    let err = (digital_decomp - bs).abs();
    assert!(
        err < 0.05,
        "BS decomp: AoN-K*CoN={digital_decomp}, BS={bs}, err={err}"
    );
}
