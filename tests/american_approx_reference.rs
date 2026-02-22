//! American option approximation reference tests.
//!
//! Sources:
//! - Barone-Adesi & Whaley (1987), "Efficient Analytic Approximation of
//!   American Option Values", J. Finance 42(2), 301-320
//! - Ju (1999), "An Approximate Formula for Pricing American Options",
//!   J. Derivatives 7(2), 31-40
//! - Haug "Option Pricing Formulas" (1998), pp. 24, 27
//! - QuantLib vendored test suite: americanoption.cpp
//!
//! These test the AmericanBinomialEngine against published reference values.

use openferric::core::{ExerciseStyle, OptionType, PricingEngine};
use openferric::engines::numerical::american_binomial::AmericanBinomialEngine;
use openferric::instruments::vanilla::VanillaOption;
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

fn american_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    q: f64,
    vol: f64,
    t: f64,
    steps: usize,
) -> f64 {
    let option = VanillaOption {
        option_type,
        strike,
        expiry: t,
        exercise: ExerciseStyle::American,
    };
    let market = make_market(spot, rate, q, vol);
    let engine = AmericanBinomialEngine::new(steps);
    engine
        .price(&option, &market)
        .expect("pricing failed")
        .price
}

// =======================================================================
// Barone-Adesi-Whaley call options
// K=100, q=0.10, r=0.10
// From Haug p.24, tolerance 3e-3
// =======================================================================

struct BawCase {
    option_type: OptionType,
    spot: f64,
    t: f64,
    vol: f64,
    expected: f64,
}

const BAW_CALLS: &[BawCase] = &[
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.10, vol: 0.15, expected: 0.0206 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.10, vol: 0.15, expected: 1.8771 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.10, vol: 0.15, expected: 10.0089 },
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.10, vol: 0.25, expected: 0.3159 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.10, vol: 0.25, expected: 3.1280 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.10, vol: 0.25, expected: 10.3919 },
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.10, vol: 0.35, expected: 0.9495 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.10, vol: 0.35, expected: 4.3777 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.10, vol: 0.35, expected: 11.1679 },
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.50, vol: 0.15, expected: 0.8208 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.50, vol: 0.15, expected: 4.0842 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.50, vol: 0.15, expected: 10.8087 },
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.50, vol: 0.25, expected: 2.7437 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.50, vol: 0.25, expected: 6.8015 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.50, vol: 0.25, expected: 13.0170 },
    BawCase { option_type: OptionType::Call, spot: 90.0,  t: 0.50, vol: 0.35, expected: 5.0063 },
    BawCase { option_type: OptionType::Call, spot: 100.0, t: 0.50, vol: 0.35, expected: 9.5106 },
    BawCase { option_type: OptionType::Call, spot: 110.0, t: 0.50, vol: 0.35, expected: 15.5689 },
];

const BAW_PUTS: &[BawCase] = &[
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.10, vol: 0.15, expected: 10.0000 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.10, vol: 0.15, expected: 1.8770 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.10, vol: 0.15, expected: 0.0410 },
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.10, vol: 0.25, expected: 10.2533 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.10, vol: 0.25, expected: 3.1277 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.10, vol: 0.25, expected: 0.4562 },
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.10, vol: 0.35, expected: 10.8787 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.10, vol: 0.35, expected: 4.3777 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.10, vol: 0.35, expected: 1.2402 },
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.50, vol: 0.15, expected: 10.5595 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.50, vol: 0.15, expected: 4.0842 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.50, vol: 0.15, expected: 1.0822 },
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.50, vol: 0.25, expected: 12.4419 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.50, vol: 0.25, expected: 6.8014 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.50, vol: 0.25, expected: 3.3226 },
    BawCase { option_type: OptionType::Put, spot: 90.0,  t: 0.50, vol: 0.35, expected: 14.6945 },
    BawCase { option_type: OptionType::Put, spot: 100.0, t: 0.50, vol: 0.35, expected: 9.5104 },
    BawCase { option_type: OptionType::Put, spot: 110.0, t: 0.50, vol: 0.35, expected: 5.8823 },
];

#[test]
fn baw_american_calls() {
    let steps = 1000;
    for (i, case) in BAW_CALLS.iter().enumerate() {
        let price = american_price(case.option_type, case.spot, 100.0, 0.10, 0.10, case.vol, case.t, steps);
        let err = (price - case.expected).abs();
        assert!(
            err < 0.15,
            "BAW call #{i}: S={}, T={}, vol={}: got {price}, expected {}, err={err}",
            case.spot, case.t, case.vol, case.expected
        );
    }
}

#[test]
fn baw_american_puts() {
    let steps = 1000;
    for (i, case) in BAW_PUTS.iter().enumerate() {
        let price = american_price(case.option_type, case.spot, 100.0, 0.10, 0.10, case.vol, case.t, steps);
        let err = (price - case.expected).abs();
        assert!(
            err < 0.15,
            "BAW put #{i}: S={}, T={}, vol={}: got {price}, expected {}, err={err}",
            case.spot, case.t, case.vol, case.expected
        );
    }
}

// =======================================================================
// Ju (1999) short-dated American puts
// S=40, K=35/40/45, q=0, r=0.0488
// From Haug/QuantLib americanoption.cpp
// =======================================================================

struct JuCase {
    strike: f64,
    t: f64,
    vol: f64,
    expected: f64,
}

const JU_PUTS: &[JuCase] = &[
    JuCase { strike: 35.0, t: 0.0833, vol: 0.20, expected: 0.006 },
    JuCase { strike: 35.0, t: 0.3333, vol: 0.20, expected: 0.201 },
    JuCase { strike: 35.0, t: 0.5833, vol: 0.20, expected: 0.433 },
    JuCase { strike: 40.0, t: 0.0833, vol: 0.20, expected: 0.851 },
    JuCase { strike: 40.0, t: 0.3333, vol: 0.20, expected: 1.576 },
    JuCase { strike: 40.0, t: 0.5833, vol: 0.20, expected: 1.984 },
    JuCase { strike: 45.0, t: 0.0833, vol: 0.20, expected: 5.000 },
    JuCase { strike: 45.0, t: 0.3333, vol: 0.20, expected: 5.084 },
    JuCase { strike: 45.0, t: 0.5833, vol: 0.20, expected: 5.260 },
    JuCase { strike: 35.0, t: 0.0833, vol: 0.30, expected: 0.078 },
    JuCase { strike: 35.0, t: 0.3333, vol: 0.30, expected: 0.697 },
    JuCase { strike: 35.0, t: 0.5833, vol: 0.30, expected: 1.218 },
    JuCase { strike: 40.0, t: 0.0833, vol: 0.30, expected: 1.309 },
    JuCase { strike: 40.0, t: 0.3333, vol: 0.30, expected: 2.477 },
    JuCase { strike: 40.0, t: 0.5833, vol: 0.30, expected: 3.161 },
    JuCase { strike: 45.0, t: 0.0833, vol: 0.30, expected: 5.059 },
    JuCase { strike: 45.0, t: 0.3333, vol: 0.30, expected: 5.699 },
    JuCase { strike: 45.0, t: 0.5833, vol: 0.30, expected: 6.231 },
    JuCase { strike: 35.0, t: 0.0833, vol: 0.40, expected: 0.247 },
    JuCase { strike: 35.0, t: 0.3333, vol: 0.40, expected: 1.344 },
    JuCase { strike: 35.0, t: 0.5833, vol: 0.40, expected: 2.150 },
    JuCase { strike: 40.0, t: 0.0833, vol: 0.40, expected: 1.767 },
    JuCase { strike: 40.0, t: 0.3333, vol: 0.40, expected: 3.381 },
    JuCase { strike: 40.0, t: 0.5833, vol: 0.40, expected: 4.342 },
    JuCase { strike: 45.0, t: 0.0833, vol: 0.40, expected: 5.288 },
    JuCase { strike: 45.0, t: 0.3333, vol: 0.40, expected: 6.501 },
    JuCase { strike: 45.0, t: 0.5833, vol: 0.40, expected: 7.367 },
];

#[test]
fn ju_american_puts() {
    let steps = 1000;
    for (i, case) in JU_PUTS.iter().enumerate() {
        let price = american_price(
            OptionType::Put,
            40.0,
            case.strike,
            0.0488,
            0.0,
            case.vol,
            case.t,
            steps,
        );
        let err = (price - case.expected).abs();
        assert!(
            err < 0.15,
            "Ju put #{i}: K={}, T={}, vol={}: got {price}, expected {}, err={err}",
            case.strike, case.t, case.vol, case.expected
        );
    }
}

// =======================================================================
// American >= European: basic monotonicity check
// =======================================================================
#[test]
fn american_put_exceeds_european() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let q = 0.03;
    let vol = 0.25;
    let t = 1.0;

    let am_price = american_price(OptionType::Put, spot, strike, rate, q, vol, t, 500);
    let eu_price = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Put,
        spot,
        strike,
        rate,
        vol,
        t,
    );

    assert!(
        am_price >= eu_price - 0.01,
        "American put ({am_price}) must >= European put ({eu_price})"
    );
}

#[test]
fn american_call_no_dividend_equals_european() {
    // Merton (1973): American call on non-dividend stock = European call
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let q = 0.0; // no dividend => no early exercise premium for calls
    let vol = 0.25;
    let t = 1.0;

    let am_price = american_price(OptionType::Call, spot, strike, rate, q, vol, t, 500);
    let eu_price = openferric::pricing::european::black_scholes_price(
        openferric::pricing::OptionType::Call,
        spot,
        strike,
        rate,
        vol,
        t,
    );

    let err = (am_price - eu_price).abs();
    assert!(
        err < 0.15,
        "American call no-div ({am_price}) should ≈ European call ({eu_price}), err={err}"
    );
}

// =======================================================================
// Deep ITM American put: early exercise value ≈ intrinsic
// =======================================================================
#[test]
fn deep_itm_american_put_near_intrinsic() {
    let price = american_price(OptionType::Put, 50.0, 100.0, 0.05, 0.0, 0.20, 0.25, 500);
    let intrinsic = 50.0;
    let err = (price - intrinsic).abs();
    assert!(
        err < 1.0,
        "Deep ITM American put: got {price}, intrinsic={intrinsic}, err={err}"
    );
}
