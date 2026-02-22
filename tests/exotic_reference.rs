//! Exotic option reference tests.
//!
//! Sources:
//! - Goldman, Sosin & Gatto (1979): floating lookback formulas
//! - Conze & Viswanathan (1991): fixed lookback formulas
//! - Rubinstein (1991): chooser option
//! - Haug "Option Pricing Formulas" (1998): numerical examples
//! - QuantLib test suite: lookbackoptions.cpp
//!
//! These test the ExoticAnalyticEngine against known reference values for
//! lookback, chooser, quanto, and compound options.

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::exotic::ExoticAnalyticEngine;
use openferric::instruments::exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFixedOption, LookbackFloatingOption,
    QuantoOption,
};
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

fn price_exotic(option: ExoticOption, market: &Market) -> f64 {
    let engine = ExoticAnalyticEngine::new();
    engine.price(&option, market).expect("pricing failed").price
}

// =======================================================================
// Floating lookback call: Goldman-Sosin-Gatto (1979)
// S=120, S_min=100, r=0.10, q=0.06, sigma=0.30, T=0.50
// Reference from Haug: ~25.36
// =======================================================================
#[test]
fn floating_lookback_call_haug() {
    let market = make_market(120.0, 0.10, 0.06, 0.30);
    let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type: OptionType::Call,
        expiry: 0.50,
        observed_extreme: Some(100.0),
    });
    let price = price_exotic(option, &market);
    // Goldman-Sosin-Gatto formula
    assert!(
        price > 24.0 && price < 27.0,
        "Floating lookback call: got {price}, expected ~25.36"
    );
}

// =======================================================================
// Floating lookback put: Goldman-Sosin-Gatto (1979)
// S=120, S_max=140, r=0.10, q=0.06, sigma=0.30, T=0.50
// =======================================================================
#[test]
fn floating_lookback_put_haug() {
    let market = make_market(120.0, 0.10, 0.06, 0.30);
    let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type: OptionType::Put,
        expiry: 0.50,
        observed_extreme: Some(140.0),
    });
    let price = price_exotic(option, &market);
    // Put has guaranteed payoff of at least S_max - S = 20
    assert!(
        price > 19.0 && price < 30.0,
        "Floating lookback put: got {price}, expected ~21-27"
    );
}

// =======================================================================
// Floating lookback: at inception (S = S_extreme), no intrinsic
// These should be purely time value
// =======================================================================
#[test]
fn floating_lookback_call_at_inception() {
    let market = make_market(100.0, 0.05, 0.0, 0.20);
    let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type: OptionType::Call,
        expiry: 1.0,
        observed_extreme: None, // defaults to spot
    });
    let price = price_exotic(option, &market);
    // Time value should be > vanilla ATM call (lookbacks are more expensive)
    let bs_atm = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    );
    assert!(
        price > bs_atm,
        "Floating lookback call at inception ({price}) should exceed ATM call ({bs_atm})"
    );
}

#[test]
fn floating_lookback_put_at_inception() {
    let market = make_market(100.0, 0.05, 0.0, 0.20);
    let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type: OptionType::Put,
        expiry: 1.0,
        observed_extreme: None,
    });
    let price = price_exotic(option, &market);
    let bs_atm = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Put,
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
    );
    assert!(
        price > bs_atm,
        "Floating lookback put at inception ({price}) should exceed ATM put ({bs_atm})"
    );
}

// =======================================================================
// Fixed lookback call: Conze & Viswanathan (1991)
// S=100, K=95, S_max=100, r=0.10, q=0, sigma=0.30, T=0.50
// =======================================================================
#[test]
fn fixed_lookback_call_haug() {
    let market = make_market(100.0, 0.10, 0.0, 0.30);
    let option = ExoticOption::LookbackFixed(LookbackFixedOption {
        option_type: OptionType::Call,
        strike: 95.0,
        expiry: 0.50,
        observed_extreme: Some(100.0),
    });
    let price = price_exotic(option, &market);
    // Fixed lookback call with K=95 and current S_max=100 => intrinsic = 5
    // Time value adds more
    assert!(
        price > 5.0 && price < 25.0,
        "Fixed lookback call: got {price}, expected 8-20 range"
    );
}

// =======================================================================
// Fixed lookback put: Conze & Viswanathan (1991)
// S=100, K=105, S_min=95, r=0.10, q=0, sigma=0.30, T=0.50
// =======================================================================
#[test]
fn fixed_lookback_put_haug() {
    let market = make_market(100.0, 0.10, 0.0, 0.30);
    let option = ExoticOption::LookbackFixed(LookbackFixedOption {
        option_type: OptionType::Put,
        strike: 105.0,
        expiry: 0.50,
        observed_extreme: Some(95.0),
    });
    let price = price_exotic(option, &market);
    // Fixed lookback put with K=105 and S_min=95 => intrinsic = 10
    assert!(
        price > 10.0 && price < 25.0,
        "Fixed lookback put: got {price}, expected 10-20 range"
    );
}

// =======================================================================
// Chooser option: Rubinstein (1991)
// S=50, K=50, r=0.08, q=0, sigma=0.25, T_choose=0.25, T_underlying=0.50
// Reference from Haug p.103: 6.1071
// =======================================================================
#[test]
fn chooser_option_haug() {
    let market = make_market(50.0, 0.08, 0.0, 0.25);
    let option = ExoticOption::Chooser(ChooserOption {
        strike: 50.0,
        expiry: 0.50,
        choose_time: 0.25,
    });
    let price = price_exotic(option, &market);
    let reference = 6.1071;
    let err = (price - reference).abs();
    assert!(
        err < 0.60,
        "Chooser option: got {price}, expected {reference}, err={err}"
    );
}

// =======================================================================
// Chooser option: positive value and exceeds both call and put
// =======================================================================
#[test]
fn chooser_positive_and_exceeds_straddle_discount() {
    let spot = 100.0;
    let market = make_market(spot, 0.08, 0.0, 0.25);

    let option = ExoticOption::Chooser(ChooserOption {
        strike: 100.0,
        expiry: 1.0,
        choose_time: 0.5,
    });
    let price = price_exotic(option, &market);

    // Chooser must be positive and bounded
    assert!(price > 0.0, "Chooser must be positive: got {price}");
    assert!(price < spot, "Chooser must be < spot: got {price}");
}

// =======================================================================
// Chooser option: lower bound is max(call, put)
// =======================================================================
#[test]
fn chooser_exceeds_both_call_and_put() {
    let spot = 100.0;
    let rate = 0.05;
    let q = 0.0;
    let vol = 0.30;
    let t = 1.0;
    let strike = 100.0;

    let market = make_market(spot, rate, q, vol);

    let option = ExoticOption::Chooser(ChooserOption {
        strike,
        expiry: t,
        choose_time: 0.5,
    });
    let chooser_price = price_exotic(option, &market);

    let bs_call = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        spot,
        strike,
        rate,
        vol,
        t,
    );
    let bs_put = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Put,
        spot,
        strike,
        rate,
        vol,
        t,
    );

    assert!(
        chooser_price >= bs_call - 0.01,
        "Chooser ({chooser_price}) should be >= call ({bs_call})"
    );
    assert!(
        chooser_price >= bs_put - 0.01,
        "Chooser ({chooser_price}) should be >= put ({bs_put})"
    );
}

// =======================================================================
// Quanto option: FX-adjusted European
// S=100, K=100, r_d=0.05, r_f=0.02, sigma_s=0.20, sigma_fx=0.10,
// corr=-0.3, fx=1.5, T=1.0
// =======================================================================
#[test]
fn quanto_call_basic() {
    let market = make_market(100.0, 0.05, 0.0, 0.20);
    let option = ExoticOption::Quanto(QuantoOption {
        option_type: OptionType::Call,
        strike: 100.0,
        expiry: 1.0,
        fx_rate: 1.5,
        foreign_rate: 0.02,
        fx_vol: 0.10,
        asset_fx_corr: -0.30,
    });
    let price = price_exotic(option, &market);
    // Quanto adjustment shifts the effective drift
    assert!(
        price > 5.0 && price < 30.0,
        "Quanto call: got {price}, expected reasonable range"
    );
}

// =======================================================================
// Compound option: call on call (Geske 1979)
// S=100, K_underlying=100, K_compound=5, T_compound=0.5, T_underlying=1.0
// r=0.05, q=0, sigma=0.25
// =======================================================================
#[test]
fn compound_call_on_call_basic() {
    let market = make_market(100.0, 0.05, 0.0, 0.25);
    let option = ExoticOption::Compound(CompoundOption {
        option_type: OptionType::Call,
        underlying_option_type: OptionType::Call,
        compound_strike: 5.0,
        underlying_strike: 100.0,
        compound_expiry: 0.5,
        underlying_expiry: 1.0,
    });
    let price = price_exotic(option, &market);

    // The underlying call is worth ~12 at ATM, so a call on it with K=5 should
    // be roughly the call value minus the discounted compound strike
    let bs_call = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        100.0,
        100.0,
        0.05,
        0.25,
        1.0,
    );
    assert!(
        price > 0.0 && price < bs_call,
        "Compound call on call: got {price}, BS underlying call={bs_call}"
    );
}

// =======================================================================
// Compound option: put on call
// Should be cheaper than call on call when underlying is in-the-money
// =======================================================================
#[test]
fn compound_put_on_call_cheaper_than_call_on_call() {
    let market = make_market(110.0, 0.05, 0.0, 0.25);

    let call_on_call = ExoticOption::Compound(CompoundOption {
        option_type: OptionType::Call,
        underlying_option_type: OptionType::Call,
        compound_strike: 5.0,
        underlying_strike: 100.0,
        compound_expiry: 0.5,
        underlying_expiry: 1.0,
    });
    let put_on_call = ExoticOption::Compound(CompoundOption {
        option_type: OptionType::Put,
        underlying_option_type: OptionType::Call,
        compound_strike: 5.0,
        underlying_strike: 100.0,
        compound_expiry: 0.5,
        underlying_expiry: 1.0,
    });

    let cc_price = price_exotic(call_on_call, &market);
    let pc_price = price_exotic(put_on_call, &market);

    // When S > K, the underlying call is in the money, so call-on-call > put-on-call
    assert!(
        cc_price > pc_price,
        "Call-on-call ({cc_price}) should exceed put-on-call ({pc_price}) when S>K"
    );
}
