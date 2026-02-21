//! Longstaff-Schwartz LSM Reference Tests
//!
//! Reference values from Longstaff & Schwartz (2001) "Valuing American Options by Simulation",
//! QuantLib test suite (BSD 3-Clause), and Barone-Adesi & Whaley (1987)
//!
//! These tests validate the LSM engine for American put pricing and barrier options.
//! Because the LSM engine is Monte Carlo based, tolerances are wider than analytic tests.

use openferric::core::PricingEngine;
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::Market;

// ============================================================================
// Longstaff-Schwartz (2001) Table 1 -- American Put
// Common parameters: K=40, r=0.06, q=0.0
// Source: Longstaff & Schwartz (2001), "Valuing American Options by Simulation:
//         A Simple Least-Squares Approach", Review of Financial Studies 14(1).
// ============================================================================

struct LsmAmericanPutCase {
    spot: f64,
    sigma: f64,
    expiry: f64,
    expected: f64,
    tolerance: f64,
}

fn longstaff_schwartz_table1_cases() -> Vec<LsmAmericanPutCase> {
    vec![
        // S=36
        LsmAmericanPutCase { spot: 36.0, sigma: 0.20, expiry: 1.0, expected: 4.474, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 36.0, sigma: 0.20, expiry: 2.0, expected: 4.831, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 36.0, sigma: 0.40, expiry: 1.0, expected: 7.081, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 36.0, sigma: 0.40, expiry: 2.0, expected: 8.480, tolerance: 0.10 },
        // S=38
        LsmAmericanPutCase { spot: 38.0, sigma: 0.20, expiry: 1.0, expected: 3.242, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 38.0, sigma: 0.20, expiry: 2.0, expected: 3.746, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 38.0, sigma: 0.40, expiry: 1.0, expected: 6.141, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 38.0, sigma: 0.40, expiry: 2.0, expected: 7.647, tolerance: 0.10 },
        // S=40 (ATM)
        LsmAmericanPutCase { spot: 40.0, sigma: 0.20, expiry: 1.0, expected: 2.319, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 40.0, sigma: 0.20, expiry: 2.0, expected: 2.875, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 40.0, sigma: 0.40, expiry: 1.0, expected: 5.318, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 40.0, sigma: 0.40, expiry: 2.0, expected: 6.912, tolerance: 0.10 },
        // S=42
        LsmAmericanPutCase { spot: 42.0, sigma: 0.20, expiry: 1.0, expected: 1.615, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 42.0, sigma: 0.20, expiry: 2.0, expected: 2.216, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 42.0, sigma: 0.40, expiry: 1.0, expected: 4.586, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 42.0, sigma: 0.40, expiry: 2.0, expected: 6.241, tolerance: 0.10 },
        // S=44
        LsmAmericanPutCase { spot: 44.0, sigma: 0.20, expiry: 1.0, expected: 1.100, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 44.0, sigma: 0.20, expiry: 2.0, expected: 1.678, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 44.0, sigma: 0.40, expiry: 1.0, expected: 3.942, tolerance: 0.10 },
        LsmAmericanPutCase { spot: 44.0, sigma: 0.40, expiry: 2.0, expected: 5.642, tolerance: 0.10 },
    ]
}

// ============================================================================
// Test: Longstaff-Schwartz Table 1 American Puts
// ============================================================================

#[test]
fn test_lsm_longstaff_schwartz_table1_american_put() {
    let strike = 40.0;
    let rate = 0.06;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);
    let cases = longstaff_schwartz_table1_cases();

    for (i, c) in cases.iter().enumerate() {
        let option = VanillaOption::american_put(strike, c.expiry);

        let market = Market::builder()
            .spot(c.spot)
            .rate(rate)
            .flat_vol(c.sigma)
            .build()
            .unwrap();

        let result = engine.price(&option, &market).unwrap();
        let error = (result.price - c.expected).abs();

        assert!(
            error <= c.tolerance,
            "LSM Table 1 case {i}: S={} sigma={} T={} expected={} got={:.4} err={:.4}",
            c.spot, c.sigma, c.expiry, c.expected, result.price, error
        );
    }

    println!(
        "All {} Longstaff-Schwartz Table 1 American put cases passed",
        cases.len()
    );
}

// ============================================================================
// Test: American put >= European put (early exercise premium > 0)
// ============================================================================

#[test]
fn test_lsm_american_put_geq_european_put() {
    let strike = 40.0;
    let rate = 0.06;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    let test_params: Vec<(f64, f64, f64)> = vec![
        (36.0, 0.20, 1.0),
        (36.0, 0.40, 2.0),
        (40.0, 0.20, 1.0),
        (40.0, 0.40, 2.0),
        (44.0, 0.20, 1.0),
        (44.0, 0.40, 2.0),
    ];

    for (spot, sigma, expiry) in &test_params {
        let market = Market::builder()
            .spot(*spot)
            .rate(rate)
            .flat_vol(*sigma)
            .build()
            .unwrap();

        let american = VanillaOption::american_put(strike, *expiry);
        let european = VanillaOption::european_put(strike, *expiry);

        let american_result = engine.price(&american, &market).unwrap();
        let european_result = engine.price(&european, &market).unwrap();

        // Allow a small margin for MC noise: american price should not be
        // substantially below european price.
        assert!(
            american_result.price >= european_result.price - 0.15,
            "American put should be >= European put: S={} sigma={} T={} \
             american={:.4} european={:.4}",
            spot, sigma, expiry, american_result.price, european_result.price
        );
    }

    println!("All American >= European put tests passed");
}

// ============================================================================
// Test: Put price increases with volatility
// ============================================================================

#[test]
fn test_lsm_put_price_increases_with_volatility() {
    let strike = 40.0;
    let rate = 0.06;
    let expiry = 1.0;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    for &spot in &[36.0, 40.0, 44.0] {
        let market_low_vol = Market::builder()
            .spot(spot)
            .rate(rate)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let market_high_vol = Market::builder()
            .spot(spot)
            .rate(rate)
            .flat_vol(0.40)
            .build()
            .unwrap();

        let option = VanillaOption::american_put(strike, expiry);

        let price_low = engine.price(&option, &market_low_vol).unwrap().price;
        let price_high = engine.price(&option, &market_high_vol).unwrap().price;

        assert!(
            price_high > price_low,
            "Put price should increase with vol: S={} low_vol_price={:.4} high_vol_price={:.4}",
            spot, price_low, price_high
        );
    }

    println!("All volatility monotonicity tests passed");
}

// ============================================================================
// Test: Put price increases as spot decreases (deeper ITM)
// ============================================================================

#[test]
fn test_lsm_put_price_increases_as_spot_decreases() {
    let strike = 40.0;
    let rate = 0.06;
    let expiry = 1.0;
    let sigma = 0.20;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);
    let spots = [44.0, 42.0, 40.0, 38.0, 36.0];

    let mut prev_price = 0.0_f64;
    for &spot in &spots {
        let market = Market::builder()
            .spot(spot)
            .rate(rate)
            .flat_vol(sigma)
            .build()
            .unwrap();

        let option = VanillaOption::american_put(strike, expiry);
        let price = engine.price(&option, &market).unwrap().price;

        if prev_price > 0.0 {
            assert!(
                price > prev_price - 0.05, // small MC tolerance
                "Put price should increase as spot decreases: S={} price={:.4} prev_price={:.4}",
                spot, price, prev_price
            );
        }
        prev_price = price;
    }

    println!("All spot-monotonicity tests passed");
}

// ============================================================================
// Test: American put price increases with time to maturity
// ============================================================================

#[test]
fn test_lsm_american_put_price_increases_with_maturity() {
    let strike = 40.0;
    let rate = 0.06;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    for &(spot, sigma) in &[(36.0, 0.20), (40.0, 0.40), (44.0, 0.20)] {
        let market = Market::builder()
            .spot(spot)
            .rate(rate)
            .flat_vol(sigma)
            .build()
            .unwrap();

        let option_1y = VanillaOption::american_put(strike, 1.0);
        let option_2y = VanillaOption::american_put(strike, 2.0);

        let price_1y = engine.price(&option_1y, &market).unwrap().price;
        let price_2y = engine.price(&option_2y, &market).unwrap().price;

        assert!(
            price_2y > price_1y - 0.10, // MC noise margin
            "American put price should increase with maturity: S={} sigma={} \
             price_1y={:.4} price_2y={:.4}",
            spot, sigma, price_1y, price_2y
        );
    }

    println!("All maturity-monotonicity tests passed");
}

// ============================================================================
// Test: Convergence toward Bjerksund-Stensland reference with high paths
// S=36, K=40, r=0.06, q=0, T=1, sigma=0.20 -> 4.4531
// ============================================================================

#[test]
fn test_lsm_convergence_toward_bjerksund_stensland() {
    let spot = 36.0;
    let strike = 40.0;
    let rate = 0.06;
    let sigma = 0.20;
    let expiry = 1.0;
    let reference = 4.4531;
    let tolerance = 0.10;

    let engine = LongstaffSchwartzEngine::new(200_000, 100, 42);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .flat_vol(sigma)
        .build()
        .unwrap();

    let option = VanillaOption::american_put(strike, expiry);
    let result = engine.price(&option, &market).unwrap();
    let error = (result.price - reference).abs();

    assert!(
        error <= tolerance,
        "LSM should converge toward Bjerksund-Stensland: expected={} got={:.4} err={:.4}",
        reference, result.price, error
    );

    // Verify stderr is reported
    assert!(
        result.stderr.is_some(),
        "LSM engine should report standard error"
    );

    println!(
        "Bjerksund-Stensland convergence test passed: reference={} lsm={:.4} stderr={:.4}",
        reference,
        result.price,
        result.stderr.unwrap()
    );
}

// ============================================================================
// Test: Barrier option -- knock-out American should be <= plain American
// ============================================================================

#[test]
fn test_lsm_barrier_knockout_leq_plain() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let sigma = 0.25;
    let expiry = 1.0;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .flat_vol(sigma)
        .build()
        .unwrap();

    // Plain American put
    let vanilla = VanillaOption::american_put(strike, expiry);
    let vanilla_price = engine.price(&vanilla, &market).unwrap().price;

    // Down-and-out put with barrier at 80 (knock-out reduces value)
    let barrier_option = BarrierOption::builder()
        .put()
        .strike(strike)
        .expiry(expiry)
        .down_and_out(80.0)
        .build()
        .unwrap();
    let barrier_price = engine.price(&barrier_option, &market).unwrap().price;

    assert!(
        barrier_price <= vanilla_price + 0.15, // small MC noise margin
        "Knock-out should be <= plain: vanilla={:.4} barrier={:.4}",
        vanilla_price, barrier_price
    );

    println!(
        "Barrier knock-out <= plain test passed: vanilla={:.4} barrier_do={:.4}",
        vanilla_price, barrier_price
    );
}

// ============================================================================
// Test: Barrier option -- knock-in + knock-out ~ plain (European payoff in LSM barrier)
// ============================================================================

#[test]
fn test_lsm_barrier_knockin_plus_knockout_approx_plain() {
    let spot = 100.0;
    let strike = 105.0;
    let rate = 0.05;
    let sigma = 0.25;
    let expiry = 0.5;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .flat_vol(sigma)
        .build()
        .unwrap();

    // Down-and-out put, barrier at 90
    let ko = BarrierOption::builder()
        .put()
        .strike(strike)
        .expiry(expiry)
        .down_and_out(90.0)
        .build()
        .unwrap();

    // Down-and-in put, barrier at 90
    let ki = BarrierOption::builder()
        .put()
        .strike(strike)
        .expiry(expiry)
        .down_and_in(90.0)
        .build()
        .unwrap();

    let ko_price = engine.price(&ko, &market).unwrap().price;
    let ki_price = engine.price(&ki, &market).unwrap().price;

    // For European-exercise barrier options (which LSM barrier uses):
    // knock-in + knock-out = plain European
    // Use a European put via LSM for the reference
    let plain = VanillaOption::european_put(strike, expiry);
    let plain_price = engine.price(&plain, &market).unwrap().price;

    let combined = ko_price + ki_price;
    let error = (combined - plain_price).abs();

    // MC noise on sum of two estimates can be larger; use generous tolerance
    assert!(
        error <= 0.50,
        "KI + KO should approximate plain: ki={:.4} + ko={:.4} = {:.4} vs plain={:.4} err={:.4}",
        ki_price, ko_price, combined, plain_price, error
    );

    println!(
        "Barrier KI+KO ~ plain test passed: ki={:.4} + ko={:.4} = {:.4} vs plain={:.4} err={:.4}",
        ki_price, ko_price, combined, plain_price, error
    );
}

// ============================================================================
// Test: Diagnostics are populated
// ============================================================================

#[test]
fn test_lsm_diagnostics_populated() {
    let engine = LongstaffSchwartzEngine::new(10_000, 20, 42);

    let market = Market::builder()
        .spot(40.0)
        .rate(0.06)
        .flat_vol(0.20)
        .build()
        .unwrap();

    let option = VanillaOption::american_put(40.0, 1.0);
    let result = engine.price(&option, &market).unwrap();

    // Verify diagnostics contain expected keys
    let diag = &result.diagnostics;
    assert!(
        diag.get("num_paths").is_some(),
        "diagnostics should contain num_paths"
    );
    assert!(
        diag.get("num_steps").is_some(),
        "diagnostics should contain num_steps"
    );
    assert!(
        diag.get("vol").is_some(),
        "diagnostics should contain vol"
    );

    assert_eq!(*diag.get("num_paths").unwrap(), 10_000.0);
    assert_eq!(*diag.get("num_steps").unwrap(), 20.0);
    assert!((*diag.get("vol").unwrap() - 0.20).abs() < 1e-10);

    println!("Diagnostics test passed");
}

// ============================================================================
// Test: Seed reproducibility -- same seed produces same price
// ============================================================================

#[test]
fn test_lsm_seed_reproducibility() {
    let engine = LongstaffSchwartzEngine::new(50_000, 50, 12345);

    let market = Market::builder()
        .spot(40.0)
        .rate(0.06)
        .flat_vol(0.20)
        .build()
        .unwrap();

    let option = VanillaOption::american_put(40.0, 1.0);

    let result1 = engine.price(&option, &market).unwrap();
    let result2 = engine.price(&option, &market).unwrap();

    assert!(
        (result1.price - result2.price).abs() < 1e-12,
        "Same seed should produce identical prices: {} vs {}",
        result1.price, result2.price
    );

    println!("Seed reproducibility test passed");
}

// ============================================================================
// Test: American call on non-dividend-paying stock ~ European call
// For q=0 there is no early exercise benefit for calls.
// ============================================================================

#[test]
fn test_lsm_american_call_no_dividend_approx_european() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let sigma = 0.30;
    let expiry = 1.0;
    let num_paths = 100_000;
    let num_steps = 50;
    let seed = 42;

    let engine = LongstaffSchwartzEngine::new(num_paths, num_steps, seed);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .flat_vol(sigma)
        .build()
        .unwrap();

    let american_call = VanillaOption::american_call(strike, expiry);
    let european_call = VanillaOption::european_call(strike, expiry);

    let am_price = engine.price(&american_call, &market).unwrap().price;
    let eu_price = engine.price(&european_call, &market).unwrap().price;

    // For non-dividend calls, American ~ European (within MC noise)
    let diff = (am_price - eu_price).abs();
    assert!(
        diff < 0.50,
        "American call (q=0) should be close to European: am={:.4} eu={:.4} diff={:.4}",
        am_price, eu_price, diff
    );

    println!(
        "American call ~ European call (no dividend) test passed: am={:.4} eu={:.4}",
        am_price, eu_price
    );
}
