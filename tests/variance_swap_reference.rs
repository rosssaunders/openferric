// Reference values from QuantLib variance swap tests (BSD 3-Clause),
// Demeterfi/Derman/Kamal/Zou (1999), and Heston model closed-form.

use openferric::core::{OptionType, PricingEngine, PricingError};
use openferric::engines::analytic::black_scholes::bs_price;
use openferric::engines::analytic::{
    VarianceSwapEngine, fair_variance_strike_from_quotes, fair_volatility_strike_from_variance,
    variance_swap_mtm, volatility_swap_mtm,
};
use openferric::instruments::variance_swap::{VarianceOptionQuote, VarianceSwap, VolatilitySwap};
use openferric::market::Market;

// ---------------------------------------------------------------------------
// Helper: generate OTM option quotes from flat Black-Scholes vol
// ---------------------------------------------------------------------------

/// Generates a dense strip of option quotes (strikes 1..=max_strike) using BS
/// with given spot, rate, dividend_yield, vol, and expiry. This lets us test
/// that the discrete replication converges to sigma^2 under flat vol.
fn flat_vol_quotes(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    max_strike: u32,
) -> Vec<VarianceOptionQuote> {
    (1..=max_strike)
        .map(|k| {
            let strike = k as f64;
            let call = bs_price(OptionType::Call, spot, strike, rate, dividend_yield, vol, expiry);
            let put = bs_price(OptionType::Put, spot, strike, rate, dividend_yield, vol, expiry);
            VarianceOptionQuote::new(strike, call, put)
        })
        .collect()
}

/// Generates a strip of quotes with a skewed vol surface (higher vol for
/// lower strikes, lower vol for higher strikes). Uses per-strike BS prices.
fn skewed_vol_quotes(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    strikes_and_vols: &[(f64, f64)],
) -> Vec<VarianceOptionQuote> {
    strikes_and_vols
        .iter()
        .map(|&(strike, vol)| {
            let call =
                bs_price(OptionType::Call, spot, strike, rate, dividend_yield, vol, expiry);
            let put =
                bs_price(OptionType::Put, spot, strike, rate, dividend_yield, vol, expiry);
            VarianceOptionQuote::new(strike, call, put)
        })
        .collect()
}

// ===========================================================================
// Section 1: Flat vol -- fair variance = sigma^2
// ===========================================================================

#[test]
fn flat_vol_020_fair_variance() {
    let spot = 100.0;
    let rate = 0.05;
    let dividend_yield = 0.0;
    let vol = 0.20;
    let expiry = 1.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    // Under flat BS vol, fair variance should equal sigma^2 = 0.04
    assert!(
        (fair_var - 0.04).abs() < 1e-4,
        "flat vol 0.20: expected fair_var ~ 0.04, got {fair_var}"
    );
}

#[test]
fn flat_vol_030_fair_variance() {
    let spot = 100.0;
    let rate = 0.05;
    let dividend_yield = 0.0;
    let vol = 0.30;
    let expiry = 1.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    assert!(
        (fair_var - 0.09).abs() < 1e-4,
        "flat vol 0.30: expected fair_var ~ 0.09, got {fair_var}"
    );
}

#[test]
fn flat_vol_015_fair_variance() {
    let spot = 100.0;
    let rate = 0.05;
    let dividend_yield = 0.0;
    let vol = 0.15;
    let expiry = 1.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    assert!(
        (fair_var - 0.0225).abs() < 1e-4,
        "flat vol 0.15: expected fair_var ~ 0.0225, got {fair_var}"
    );
}

// ===========================================================================
// Section 2: Fair volatility strike from variance (zero var-of-var)
// ===========================================================================

#[test]
fn fair_volatility_zero_var_of_var() {
    // With zero var-of-var the convexity adjustment vanishes:
    // K_vol = sqrt(K_var)
    let cases = [(0.04, 0.20), (0.09, 0.30), (0.0225, 0.15)];

    for (k_var, expected_k_vol) in cases {
        let k_vol = fair_volatility_strike_from_variance(k_var, 0.0).unwrap();
        assert!(
            (k_vol - expected_k_vol).abs() < 1e-10,
            "K_var={k_var}: expected K_vol={expected_k_vol}, got {k_vol}"
        );
    }
}

#[test]
fn fair_volatility_with_convexity_adjustment() {
    // K_vol = sqrt(K_var) - var_of_var / (8 * K_var^{3/2})
    let k_var = 0.04;
    let var_of_var = 0.001;
    let expected = 0.04_f64.sqrt() - 0.001 / (8.0 * 0.04_f64.powf(1.5));
    let k_vol = fair_volatility_strike_from_variance(k_var, var_of_var).unwrap();
    assert!(
        (k_vol - expected).abs() < 1e-10,
        "convexity adjustment: expected {expected}, got {k_vol}"
    );
}

// ===========================================================================
// Section 3: Heston continuous fair variance (closed-form reference)
// ===========================================================================
//
// Under the Heston model the continuous fair variance strike is:
//   K_var = theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T)
//
// The replicating engine does not implement Heston directly. We test
// the closed-form formula here and verify the engine converges when
// fed prices from a Heston-parameterized smile. Below we only test the
// closed-form values since producing an accurate Heston smile strip for
// the replicating engine would require a separate semi-analytic pricer.

fn heston_fair_variance(kappa: f64, theta: f64, v0: f64, t: f64) -> f64 {
    theta + (v0 - theta) * (1.0 - (-kappa * t).exp()) / (kappa * t)
}

#[test]
fn heston_fair_variance_case_1() {
    // kappa=2.0, theta=0.04, v0=0.09, T=1.0
    let k_var = heston_fair_variance(2.0, 0.04, 0.09, 1.0);
    let expected = 0.04 + (0.09 - 0.04) * (1.0 - (-2.0_f64).exp()) / 2.0;
    assert!(
        (k_var - expected).abs() < 1e-12,
        "Heston case 1: got {k_var}, expected {expected}"
    );
    // Numerical reference: ~0.06162
    assert!(
        (k_var - 0.06162).abs() < 1e-4,
        "Heston case 1: expected ~0.06162, got {k_var}"
    );
}

#[test]
fn heston_fair_variance_case_2() {
    // kappa=0.5, theta=0.04, v0=0.04, T=1.0
    // When v0 == theta, fair variance = theta regardless of kappa/T
    let k_var = heston_fair_variance(0.5, 0.04, 0.04, 1.0);
    assert!(
        (k_var - 0.04).abs() < 1e-12,
        "Heston case 2: expected 0.04, got {k_var}"
    );
}

#[test]
fn heston_fair_variance_case_3() {
    // kappa=1.0, theta=0.0225, v0=0.0625, T=0.5
    let k_var = heston_fair_variance(1.0, 0.0225, 0.0625, 0.5);
    let expected = 0.0225 + (0.0625 - 0.0225) * (1.0 - (-0.5_f64).exp()) / 0.5;
    assert!(
        (k_var - expected).abs() < 1e-12,
        "Heston case 3: got {k_var}, expected {expected}"
    );
    // Numerical reference: ~0.05398
    assert!(
        (k_var - 0.05398).abs() < 1e-4,
        "Heston case 3: expected ~0.05398, got {k_var}"
    );
}

// ===========================================================================
// Section 4: Replicating variance swap with skewed vol surface
// ===========================================================================

#[test]
fn replicating_skewed_surface_fair_variance() {
    // QuantLib-inspired test: S=100, r=5%, q=0%, T=90/365
    // 19 options with a realistic skew (higher vol for lower strikes).
    let spot = 100.0;
    let rate = 0.05;
    let dividend_yield = 0.0;
    let expiry = 90.0 / 365.0;

    // Strikes and implied vols (puts below forward, calls above)
    let strikes_and_vols: Vec<(f64, f64)> = vec![
        (50.0, 0.30),
        (55.0, 0.29),
        (60.0, 0.28),
        (65.0, 0.27),
        (70.0, 0.26),
        (75.0, 0.25),
        (80.0, 0.24),
        (85.0, 0.23),
        (90.0, 0.22),
        (95.0, 0.21),
        (100.0, 0.20),
        (105.0, 0.19),
        (110.0, 0.18),
        (115.0, 0.17),
        (120.0, 0.16),
        (125.0, 0.15),
        (130.0, 0.14),
        (135.0, 0.13),
        (140.0, 0.13),
    ];

    let quotes = skewed_vol_quotes(spot, rate, dividend_yield, expiry, &strikes_and_vols);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    // Expected: ~0.04189. Tolerance is wider due to coarse strike spacing.
    assert!(
        (fair_var - 0.04189).abs() < 1e-2,
        "skewed surface: expected fair_var ~ 0.04189, got {fair_var}"
    );
}

// ===========================================================================
// Section 5: PricingEngine<VarianceSwap> integration test
// ===========================================================================

#[test]
fn engine_variance_swap_flat_vol() {
    let spot = 100.0;
    let rate = 0.01;
    let dividend_yield = 0.0;
    let vol = 0.20;
    let expiry = 1.0;
    let strike_vol = 0.20;
    let notional_vega = 100_000.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);

    let instrument = VarianceSwap::new(notional_vega, strike_vol, expiry, quotes);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .unwrap();

    let engine = VarianceSwapEngine::new();
    let result = engine.price(&instrument, &market).unwrap();

    // When fair variance matches strike variance, the MtM should be ~0
    // (modulo small discretization error in the replication).
    assert!(
        result.price.abs() < 500.0,
        "at-the-money variance swap price should be near zero, got {}",
        result.price
    );

    // Verify diagnostics contain fair_variance
    let diag_var = result.diagnostics.get("fair_variance");
    assert!(diag_var.is_some(), "diagnostics should contain fair_variance");
    let fair_var = diag_var.unwrap();
    assert!(
        (fair_var - 0.04).abs() < 1e-3,
        "fair_variance diagnostic should be ~0.04, got {fair_var}"
    );
}

#[test]
fn engine_volatility_swap_flat_vol() {
    let spot = 100.0;
    let rate = 0.01;
    let dividend_yield = 0.0;
    let vol = 0.20;
    let expiry = 1.0;
    let strike_vol = 0.20;
    let notional_vega = 100_000.0;
    let var_of_var = 0.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);

    let instrument =
        VolatilitySwap::new(notional_vega, strike_vol, expiry, quotes, var_of_var);

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .unwrap();

    let engine = VarianceSwapEngine::new();
    let result = engine.price(&instrument, &market).unwrap();

    // At-the-money vol swap: price should be near zero.
    assert!(
        result.price.abs() < 500.0,
        "at-the-money volatility swap price should be near zero, got {}",
        result.price
    );
}

// ===========================================================================
// Section 6: Mark-to-market via variance_swap_mtm
// ===========================================================================

#[test]
fn mtm_at_inception_zero() {
    // At inception, if fair_variance == strike_vol^2, MtM should be zero.
    let strike_vol = 0.20;
    let fair_var = 0.04; // = 0.20^2
    let notional_vega = 50_000.0;
    let expiry = 1.0;
    let rate = 0.05;

    // Need at least 2 quotes for validation
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];

    let instrument = VarianceSwap::new(notional_vega, strike_vol, expiry, quotes);
    let mtm = variance_swap_mtm(&instrument, fair_var, rate).unwrap();

    // N_var = 50000 / (2*0.20) = 125000
    // payoff = 125000 * (0.04 - 0.04) = 0
    assert!(
        mtm.abs() < 1e-10,
        "MtM at inception with matching fair var should be zero, got {mtm}"
    );
}

#[test]
fn mtm_long_variance_swap_losing() {
    // Mark-to-market test inspired by trader scenario:
    // N_vega = 50000, K_vol = 0.22 (variance strike = 0.0484)
    // 6 months elapsed of 1 year, realized vol = 0.16 (var = 0.0256)
    // New 6-month implied fair K_var = 0.0361 (vol = 0.19)
    // r = 2.75%
    //
    // The API convention:
    //   variance_swap_mtm uses observed_realized_var as expected realized variance
    //   if provided, otherwise uses the passed fair_variance.
    //   payoff = N_var * (expected_realized_var - K_vol^2)
    //   mtm = DF * payoff
    //
    // For a mid-life MtM with a blend of realized + forward fair:
    //   expected_var = (t/T)*realized + ((T-t)/T)*new_fair_var
    //   N_var = N_vega / (2*K_vol)
    //   mtm = DF * N_var * (expected_var - K_var)
    //
    // With 6 months done, 6 months remaining at original 1-year horizon:
    let notional_vega: f64 = 50_000.0;
    let strike_vol: f64 = 0.22;
    let original_expiry: f64 = 1.0;
    let rate: f64 = 0.0275;
    let realized_var: f64 = 0.0256; // 0.16^2
    let new_fair_var: f64 = 0.0361; // 0.19^2
    let t_elapsed: f64 = 0.5;

    // Blended expected variance over the full period
    let expected_var =
        (t_elapsed / original_expiry) * realized_var
            + ((original_expiry - t_elapsed) / original_expiry) * new_fair_var;

    let n_var = notional_vega / (2.0 * strike_vol);
    let k_var = strike_vol * strike_vol;
    let df = (-rate * original_expiry).exp();
    let expected_mtm = df * n_var * (expected_var - k_var);

    // Now compute via the API. We set observed_realized_var to the blended
    // expected variance, since the API uses it as the single expected realized
    // variance figure.
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];

    let instrument = VarianceSwap::new(notional_vega, strike_vol, original_expiry, quotes)
        .with_observed_realized_var(expected_var);

    let mtm = variance_swap_mtm(&instrument, new_fair_var, rate).unwrap();

    assert!(
        (mtm - expected_mtm).abs() < 1.0,
        "MtM mismatch: expected {expected_mtm:.2}, got {mtm:.2}"
    );

    // The position should be losing (negative MtM for a long variance position
    // when realized + implied are below the strike).
    assert!(
        mtm < 0.0,
        "Long variance swap losing when realized+implied < strike: got {mtm:.2}"
    );
}

#[test]
fn mtm_vega_notional_to_variance_notional_conversion() {
    // Verify N_var = N_vega / (2 * K_vol) relationship.
    // With N_vega = 50000 and K_vol = 0.22:
    //   N_var = 50000 / (2 * 0.22) = 113636.36...
    let notional_vega = 50_000.0;
    let strike_vol = 0.22;
    let expected_n_var: f64 = notional_vega / (2.0 * strike_vol);

    assert!(
        (expected_n_var - 113636.36363636_f64).abs() < 0.01,
        "N_var conversion: expected ~113636.36, got {expected_n_var}"
    );

    // Verify via MtM: if expected_realized_var - K_var = 1 unit,
    // then payoff = N_var * 1, and MtM = DF * N_var
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];

    let k_var = strike_vol * strike_vol;
    let instrument = VarianceSwap::new(notional_vega, strike_vol, 1.0, quotes)
        .with_observed_realized_var(k_var + 1.0);

    let rate = 0.0;
    let mtm = variance_swap_mtm(&instrument, 0.0, rate).unwrap();

    // mtm = exp(0) * N_var * 1.0 = N_var
    assert!(
        (mtm - expected_n_var).abs() < 0.01,
        "MtM unit test: expected {expected_n_var:.2}, got {mtm:.2}"
    );
}

// ===========================================================================
// Section 7: Volatility swap MtM
// ===========================================================================

#[test]
fn vol_swap_mtm_at_the_money() {
    // When observed realized vol equals strike vol, MtM should be zero.
    let strike_vol = 0.20;
    let notional_vega = 100_000.0;
    let expiry = 1.0;
    let rate = 0.05;
    let fair_var = 0.04;
    let fair_vol = 0.20;

    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];

    let instrument = VolatilitySwap::new(notional_vega, strike_vol, expiry, quotes, 0.0)
        .with_observed_realized_var(0.04); // realized vol = 0.20

    let mtm = volatility_swap_mtm(&instrument, fair_var, fair_vol, rate).unwrap();

    assert!(
        mtm.abs() < 1e-10,
        "At-the-money vol swap MtM should be zero, got {mtm}"
    );
}

#[test]
fn vol_swap_mtm_in_the_money() {
    // Realized vol > strike vol => long position profits.
    let strike_vol = 0.20;
    let notional_vega = 100_000.0;
    let expiry = 1.0;
    let rate = 0.05;

    let realized_var = 0.0625; // realized vol = 0.25
    let fair_var = 0.0625;
    let fair_vol = 0.25;

    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];

    let instrument = VolatilitySwap::new(notional_vega, strike_vol, expiry, quotes, 0.0)
        .with_observed_realized_var(realized_var);

    let mtm = volatility_swap_mtm(&instrument, fair_var, fair_vol, rate).unwrap();

    // payoff = N_vega * (realized_vol - strike_vol) = 100000 * (0.25 - 0.20) = 5000
    // mtm = exp(-0.05) * 5000
    let df = (-0.05_f64).exp();
    let expected = df * 100_000.0 * (0.25 - 0.20);

    assert!(
        (mtm - expected).abs() < 1e-6,
        "In-the-money vol swap: expected {expected:.2}, got {mtm:.2}"
    );
    assert!(mtm > 0.0, "Long vol swap should profit when realized > strike");
}

// ===========================================================================
// Section 8: Realized variance from log returns
// ===========================================================================

#[test]
fn realized_variance_log_returns() {
    // Verify the standard formula: sigma^2 = (AF/N) * sum[ln(S_i/S_{i-1})]^2
    // where AF = annualization factor.
    let prices: Vec<f64> = vec![100.0, 102.0, 99.0, 101.0, 103.0, 100.5];
    let n = prices.len() - 1;
    let af: f64 = 252.0; // daily annualization

    let sum_sq: f64 = prices
        .windows(2)
        .map(|w| {
            let lr = (w[1] / w[0]).ln();
            lr * lr
        })
        .sum();

    let realized_var = (af / n as f64) * sum_sq;

    // Just verify the formula is self-consistent (positive, finite).
    assert!(realized_var > 0.0);
    assert!(realized_var.is_finite());

    // Realized vol should be reasonable for these small moves
    let realized_vol = realized_var.sqrt();
    assert!(
        realized_vol > 0.05 && realized_vol < 1.0,
        "realized vol should be reasonable, got {realized_vol}"
    );
}

// ===========================================================================
// Section 9: Edge cases and validation
// ===========================================================================

#[test]
fn validation_too_few_quotes() {
    let result = fair_variance_strike_from_quotes(
        1.0,
        0.05,
        100.0,
        0.0,
        &[VarianceOptionQuote::new(100.0, 5.0, 5.0)],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        PricingError::InvalidInput(msg) => {
            assert!(msg.contains("at least two"), "unexpected error: {msg}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn validation_negative_strike() {
    let quotes = vec![
        VarianceOptionQuote::new(-10.0, 5.0, 5.0),
        VarianceOptionQuote::new(100.0, 5.0, 5.0),
    ];
    let result = fair_variance_strike_from_quotes(1.0, 0.05, 100.0, 0.0, &quotes);
    assert!(result.is_err());
}

#[test]
fn validation_zero_expiry() {
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 5.0, 5.0),
        VarianceOptionQuote::new(110.0, 5.0, 5.0),
    ];
    let result = fair_variance_strike_from_quotes(0.0, 0.05, 100.0, 0.0, &quotes);
    assert!(result.is_err());
}

#[test]
fn validation_negative_spot() {
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 5.0, 5.0),
        VarianceOptionQuote::new(110.0, 5.0, 5.0),
    ];
    let result = fair_variance_strike_from_quotes(1.0, 0.05, -100.0, 0.0, &quotes);
    assert!(result.is_err());
}

#[test]
fn fair_volatility_negative_variance_rejected() {
    let result = fair_volatility_strike_from_variance(-0.01, 0.0);
    assert!(result.is_err());
}

#[test]
fn fair_volatility_negative_var_of_var_rejected() {
    let result = fair_volatility_strike_from_variance(0.04, -0.001);
    assert!(result.is_err());
}

// ===========================================================================
// Section 10: Different expiries under flat vol
// ===========================================================================

#[test]
fn flat_vol_short_expiry() {
    // T=0.25 (3 months), vol=0.25 => K_var=0.0625
    let spot = 100.0;
    let rate = 0.02;
    let dividend_yield = 0.0;
    let vol = 0.25;
    let expiry = 0.25;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 400);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    assert!(
        (fair_var - 0.0625).abs() < 1e-3,
        "short expiry flat vol 0.25: expected ~0.0625, got {fair_var}"
    );
}

#[test]
fn flat_vol_long_expiry() {
    // T=5.0 (5 years), vol=0.20 => K_var=0.04
    let spot = 100.0;
    let rate = 0.03;
    let dividend_yield = 0.01;
    let vol = 0.20;
    let expiry = 5.0;

    let quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 600);
    let fair_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();

    assert!(
        (fair_var - 0.04).abs() < 1e-3,
        "long expiry flat vol 0.20: expected ~0.04, got {fair_var}"
    );
}

// ===========================================================================
// Section 11: Settlement payoff computation
// ===========================================================================

#[test]
fn settlement_payoff_long_variance() {
    // Settlement = N_var * (realized_var - K_var)
    // N_vega = 100000, K_vol = 0.20, so N_var = 100000/(2*0.20) = 250000
    // realized_var = 0.05, K_var = 0.04
    // Settlement = 250000 * (0.05 - 0.04) = 2500
    let n_vega = 100_000.0;
    let k_vol = 0.20;
    let n_var = n_vega / (2.0 * k_vol);
    let realized_var = 0.05;
    let k_var = k_vol * k_vol;

    let settlement: f64 = n_var * (realized_var - k_var);
    assert!(
        (settlement - 2500.0).abs() < 1e-6,
        "settlement: expected 2500, got {settlement}"
    );

    // Verify via API (at expiry, rate=0 so DF=1):
    let quotes = vec![
        VarianceOptionQuote::new(90.0, 12.0, 2.0),
        VarianceOptionQuote::new(110.0, 2.0, 12.0),
    ];
    let instrument = VarianceSwap::new(n_vega, k_vol, 1.0, quotes)
        .with_observed_realized_var(realized_var);

    let mtm = variance_swap_mtm(&instrument, 0.0, 0.0).unwrap();
    assert!(
        (mtm - 2500.0).abs() < 1e-6,
        "API settlement: expected 2500, got {mtm}"
    );
}

// ===========================================================================
// Section 12: Dense strike grid convergence
// ===========================================================================

#[test]
fn dense_strikes_improve_replication_accuracy() {
    // With more strikes, replication should converge more closely to sigma^2.
    let spot = 100.0;
    let rate = 0.05;
    let dividend_yield = 0.0;
    let vol = 0.25;
    let expiry = 0.5;
    let expected_var = vol * vol; // 0.0625

    let coarse_quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 50);
    let fine_quotes = flat_vol_quotes(spot, rate, dividend_yield, vol, expiry, 500);

    let coarse_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &coarse_quotes)
            .unwrap();
    let fine_var =
        fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &fine_quotes).unwrap();

    let coarse_err = (coarse_var - expected_var).abs();
    let fine_err = (fine_var - expected_var).abs();

    assert!(
        fine_err < coarse_err,
        "finer strike grid should be more accurate: coarse_err={coarse_err}, fine_err={fine_err}"
    );
    assert!(
        fine_err < 1e-3,
        "fine grid should be within 1e-3 of sigma^2: err={fine_err}"
    );
}
