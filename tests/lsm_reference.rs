//! Longstaff-Schwartz LSM Reference Tests
//!
//! Reference values from Longstaff & Schwartz (2001) "Valuing American Options by Simulation",
//! QuantLib test suite (BSD 3-Clause), and Barone-Adesi & Whaley (1987)
//!
//! These tests validate the LSM engine for American put pricing and barrier options.
//! Stochastic assertions use reported or replicate-estimated standard errors.

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::Market;
use openferric::pricing::european::black_scholes_price;

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
}

fn longstaff_schwartz_table1_cases() -> Vec<LsmAmericanPutCase> {
    vec![
        // S=36
        LsmAmericanPutCase {
            spot: 36.0,
            sigma: 0.20,
            expiry: 1.0,
            expected: 4.486_693_114_646_285,
        },
        LsmAmericanPutCase {
            spot: 36.0,
            sigma: 0.20,
            expiry: 2.0,
            expected: 4.848_315_697_580_784,
        },
        LsmAmericanPutCase {
            spot: 36.0,
            sigma: 0.40,
            expiry: 1.0,
            expected: 7.109_023_651_525_109,
        },
        LsmAmericanPutCase {
            spot: 36.0,
            sigma: 0.40,
            expiry: 2.0,
            expected: 8.514_294_765_551_25,
        },
        // S=38
        LsmAmericanPutCase {
            spot: 38.0,
            sigma: 0.20,
            expiry: 1.0,
            expected: 3.257_207_614_108_937,
        },
        LsmAmericanPutCase {
            spot: 38.0,
            sigma: 0.20,
            expiry: 2.0,
            expected: 3.751_361_696_554_655,
        },
        LsmAmericanPutCase {
            spot: 38.0,
            sigma: 0.40,
            expiry: 1.0,
            expected: 6.154_718_699_114_115,
        },
        LsmAmericanPutCase {
            spot: 38.0,
            sigma: 0.40,
            expiry: 2.0,
            expected: 7.675_053_485_814_676,
        },
        // S=40 (ATM)
        LsmAmericanPutCase {
            spot: 40.0,
            sigma: 0.20,
            expiry: 1.0,
            expected: 2.319_547_063_106_665,
        },
        LsmAmericanPutCase {
            spot: 40.0,
            sigma: 0.20,
            expiry: 2.0,
            expected: 2.889_913_925_641_943,
        },
        LsmAmericanPutCase {
            spot: 40.0,
            sigma: 0.40,
            expiry: 1.0,
            expected: 5.318_221_267_864_815,
        },
        LsmAmericanPutCase {
            spot: 40.0,
            sigma: 0.40,
            expiry: 2.0,
            expected: 6.923_369_771_907_185,
        },
        // S=42
        LsmAmericanPutCase {
            spot: 42.0,
            sigma: 0.20,
            expiry: 1.0,
            expected: 1.621_176_005_773_276,
        },
        LsmAmericanPutCase {
            spot: 42.0,
            sigma: 0.20,
            expiry: 2.0,
            expected: 2.216_770_711_147_334,
        },
        LsmAmericanPutCase {
            spot: 42.0,
            sigma: 0.40,
            expiry: 1.0,
            expected: 4.588_155_734_065_328,
        },
        LsmAmericanPutCase {
            spot: 42.0,
            sigma: 0.40,
            expiry: 2.0,
            expected: 6.250_359_492_586_553,
        },
        // S=44
        LsmAmericanPutCase {
            spot: 44.0,
            sigma: 0.20,
            expiry: 1.0,
            expected: 1.112_978_968_983_765,
        },
        LsmAmericanPutCase {
            spot: 44.0,
            sigma: 0.20,
            expiry: 2.0,
            expected: 1.693_341_191_929_454,
        },
        LsmAmericanPutCase {
            spot: 44.0,
            sigma: 0.40,
            expiry: 1.0,
            expected: 3.952_776_596_660_82,
        },
        LsmAmericanPutCase {
            spot: 44.0,
            sigma: 0.40,
            expiry: 2.0,
            expected: 5.646_890_690_597_949,
        },
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
        let stderr = result.stderr.expect("LSM reports standard error");
        // QuantLib 1.43 CRR values use 10,000 steps. Across this grid the
        // largest observed 5k-to-10k refinement is 2.80e-4.
        let tolerance = 4.0 * stderr + 3.0e-4;

        assert!(
            error <= tolerance,
            "LSM/QuantLib case {i}: S={} sigma={} T={} expected={} got={:.6} err={:.6} stderr={stderr} tolerance={tolerance}",
            c.spot,
            c.sigma,
            c.expiry,
            c.expected,
            result.price,
            error
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
            spot,
            sigma,
            expiry,
            american_result.price,
            european_result.price
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
            spot,
            price_low,
            price_high
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
                spot,
                price,
                prev_price
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
            spot,
            sigma,
            price_1y,
            price_2y
        );
    }

    println!("All maturity-monotonicity tests passed");
}

// ============================================================================
// Test: Convergence toward a QuantLib CRR reference with high paths
// S=36, K=40, r=0.06, q=0, T=1, sigma=0.20
// ============================================================================

#[test]
fn test_lsm_matches_quantlib_crr_with_reported_stderr() {
    let spot = 36.0;
    let strike = 40.0;
    let rate = 0.06;
    let sigma = 0.20;
    let expiry = 1.0;
    // QuantLib 1.43 BinomialVanillaEngine("crr"), 10,000 steps.
    let reference = 4.486_693_114_646_285;

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
    let stderr = result.stderr.expect("LSM reports standard error");
    // QuantLib's 10k-to-5k CRR refinement is 1.91e-5; LSM sampling gets a
    // four-standard-error budget on top of that external-grid uncertainty.
    let tolerance = 4.0 * stderr + 2.0e-5;

    assert!(
        error <= tolerance,
        "LSM should converge toward QuantLib CRR: expected={} got={:.6} err={:.6} stderr={stderr} tolerance={tolerance}",
        reference,
        result.price,
        error
    );

    // Verify stderr is reported
    assert!(
        result.stderr.is_some(),
        "LSM engine should report standard error"
    );

    println!(
        "QuantLib CRR convergence test passed: reference={} lsm={:.4} stderr={:.4}",
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
        vanilla_price,
        barrier_price
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
fn test_lsm_barrier_knockin_plus_knockout_within_reported_stderr() {
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

    let ko_result = engine.price(&ko, &market).unwrap();
    let ki_result = engine.price(&ki, &market).unwrap();

    // For European-exercise barrier options (which LSM barrier uses):
    // knock-in + knock-out = plain European
    // Use a European put via LSM for the reference
    let plain = VanillaOption::european_put(strike, expiry);
    let plain_result = engine.price(&plain, &market).unwrap();

    let combined = ko_result.price + ki_result.price;
    let error = (combined - plain_result.price).abs();
    let combined_stderr = (ko_result.stderr.unwrap().powi(2)
        + ki_result.stderr.unwrap().powi(2)
        + plain_result.stderr.unwrap().powi(2))
    .sqrt();

    assert!(
        error <= 4.0 * combined_stderr,
        "KI + KO parity: ki={:.4} + ko={:.4} = {:.4} vs plain={:.4} err={:.4} combined_se={combined_stderr:.4}",
        ki_result.price,
        ko_result.price,
        combined,
        plain_result.price,
        error
    );

    println!(
        "Barrier KI+KO ~ plain test passed: ki={:.4} + ko={:.4} = {:.4} vs plain={:.4} err={:.4}",
        ki_result.price, ko_result.price, combined, plain_result.price, error
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
    assert!(diag.get("vol").is_some(), "diagnostics should contain vol");

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
        result1.price,
        result2.price
    );

    println!("Seed reproducibility test passed");
}

#[cfg(feature = "parallel")]
#[test]
fn test_lsm_reproducible_across_rayon_thread_counts() {
    use rayon::ThreadPoolBuilder;

    let engine = LongstaffSchwartzEngine::new(20_000, 40, 12345);
    let market = Market::builder()
        .spot(40.0)
        .rate(0.06)
        .flat_vol(0.20)
        .build()
        .unwrap();
    let option = VanillaOption::american_put(40.0, 1.0);

    let price_with_threads = |threads| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| engine.price(&option, &market).unwrap().price)
    };

    assert_eq!(
        price_with_threads(1).to_bits(),
        price_with_threads(4).to_bits()
    );
}

// ============================================================================
// Test: American call on non-dividend-paying stock ~ European call
// For q=0 there is no early exercise benefit for calls.
// ============================================================================

#[test]
fn test_lsm_american_call_no_dividend_matches_black_scholes_with_reported_stderr() {
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

    let am_result = engine.price(&american_call, &market).unwrap();
    let eu_result = engine.price(&european_call, &market).unwrap();
    let exact = black_scholes_price(OptionType::Call, spot, strike, rate, sigma, expiry);
    let am_stderr = am_result.stderr.expect("American LSM reports stderr");
    let eu_stderr = eu_result.stderr.expect("European MC reports stderr");
    let am_tolerance = 4.0 * am_stderr + 1.0e-12;
    let eu_tolerance = 4.0 * eu_stderr + 1.0e-12;
    assert!(
        (am_result.price - exact).abs() <= am_tolerance,
        "American q=0 call: price={} exact={exact} stderr={am_stderr} tolerance={am_tolerance}",
        am_result.price
    );
    assert!(
        (eu_result.price - exact).abs() <= eu_tolerance,
        "European call: price={} exact={exact} stderr={eu_stderr} tolerance={eu_tolerance}",
        eu_result.price
    );

    println!(
        "American call ~ European call (no dividend) test passed: am={:.4} eu={:.4}",
        am_result.price, eu_result.price
    );
}
