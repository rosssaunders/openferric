//! Regression tests for Carr-Madan damping admissibility and non-finite propagation.
//!
//! Audit finding: the FFT transform layer masked NaN integrands to 0.0
//! (`f64::max(0.0)` returns the non-NaN operand) and never verified the
//! Carr-Madan admissibility condition `E[S_T^(alpha+1)] < inf` (Carr & Madan
//! 1999, Section 2; default alpha = 1.5 requires a finite 2.5-th spot moment).
//! Moment-exploding Heston parameterizations silently returned all-zero calls
//! (parity puts near -50), and VG/CGMY/NIG sets that pass their risk-neutral
//! constructors but violate the damping moment bound crossed principal branch
//! cuts and returned finite calls far below the no-arbitrage lower bound
//! `S*exp(-q*T) - K*exp(-r*T)` (e.g. a K=50 call of 8.79 against a bound of
//! ~51).
//!
//! After the fix every FFT pricing entry point must either return arbitrage-
//! free prices (possibly after automatically reducing alpha to an admissible
//! value) or a clean error - never a silent zero or below-intrinsic price.
//!
//! Reference numbers used below:
//! - No-arbitrage bounds are the standard European call bounds
//!   `max(0, S*exp(-q*T) - K*exp(-r*T)) <= C <= S*exp(-q*T)` (model-free).
//! - The Black-Scholes anchor 8.916037278572539 for S=K=100, r=0.02, q=0, sigma=0.2,
//!   T=1 is the closed-form value: d1=0.2, d2=0.0,
//!   C = 100*N(0.2) - 100*exp(-0.02)*N(0.0).
//! - The Heston anchor 16.070154917028834 (K=100) is QuantLib's cached Lewis
//!   dataset value from vendor/QuantLib/test-suite/hestonmodel.cpp (same
//!   source as tests/fixtures/heston_expected_put_call.csv).

use openferric::engines::fft::{
    CarrMadanContext, CarrMadanParams, CgmyCharFn, DEFAULT_ALPHA, NigCharFn, VarianceGammaCharFn,
    carr_madan_price_at_strikes, heston_price_fft, try_heston_price_fft,
};

const SPOT: f64 = 100.0;
const RATE: f64 = 0.02;
const DIV: f64 = 0.0;
const STRIKES: [f64; 3] = [50.0, 100.0, 150.0];

/// Binary64/numerical tolerance on the model-free call bounds.  This is a
/// supplemental invariant, not a pricing oracle; successful fallback prices
/// must nevertheless respect it to well below a cent.
const BOUND_TOL: f64 = 1.0e-8;

fn assert_arbitrage_free(prices: &[(f64, f64)], maturity: f64, label: &str) {
    assert_eq!(prices.len(), STRIKES.len(), "{label}: wrong output length");
    let forward_spot = SPOT * (-DIV * maturity).exp();
    for (strike, call) in prices {
        let lower = (forward_spot - strike * (-RATE * maturity).exp()).max(0.0);
        assert!(
            call.is_finite(),
            "{label}: non-finite call at K={strike}: {call}"
        );
        assert!(
            *call >= lower - BOUND_TOL,
            "{label}: call {call} at K={strike} is below the no-arbitrage lower bound {lower}"
        );
        assert!(
            *call <= forward_spot + BOUND_TOL,
            "{label}: call {call} at K={strike} exceeds the upper bound {forward_spot}"
        );
    }
}

/// Audit case 1 (NaN-masked-to-zero regime): Heston v0=0.04, kappa=0.5,
/// theta=0.04, xi=1.5, rho=+0.7, T=5. The 2.5-th moment explodes and the
/// characteristic function is non-finite on the damping contour; before the
/// fix every strike returned exactly 0.0 (parity put ~ -54.8).
#[test]
fn heston_nan_regime_is_not_masked_to_zero() {
    let result = try_heston_price_fft(SPOT, &STRIKES, RATE, DIV, 0.04, 0.5, 0.04, 1.5, 0.7, 5.0);
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, 5.0, "heston kappa=0.5 xi=1.5 rho=0.7 T=5"),
    }

    // The infallible convenience API documents an empty vector on failure;
    // it must never return confident garbage.
    let prices = heston_price_fft(SPOT, &STRIKES, RATE, DIV, 0.04, 0.5, 0.04, 1.5, 0.7, 5.0);
    if !prices.is_empty() {
        assert_arbitrage_free(
            &prices,
            5.0,
            "heston_price_fft kappa=0.5 xi=1.5 rho=0.7 T=5",
        );
    }
}

/// Audit case 2 (finite-but-wrong-branch regime): same as case 1 but with
/// kappa=2.0 the characteristic function stays finite while the moment still
/// explodes. Before the fix the K=50 call was 37.72 against a lower bound of
/// 54.76 - no NaN ever appeared, so NaN propagation alone cannot catch this.
#[test]
fn heston_finite_branch_garbage_is_not_below_intrinsic() {
    let result = try_heston_price_fft(SPOT, &STRIKES, RATE, DIV, 0.04, 2.0, 0.04, 1.5, 0.7, 5.0);
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, 5.0, "heston kappa=2.0 xi=1.5 rho=0.7 T=5"),
    }
}

/// Audit case 3: Heston rho=0, xi=3, T=3 returned a K=50 call of ~50.1
/// against a lower bound of 52.91 before the fix.
#[test]
fn heston_high_volvol_zero_corr_is_not_below_intrinsic() {
    let result = try_heston_price_fft(SPOT, &STRIKES, RATE, DIV, 0.04, 1.0, 0.04, 3.0, 0.0, 3.0);
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, 3.0, "heston kappa=1.0 xi=3.0 rho=0 T=3"),
    }
}

/// Audit case 4: VG sigma=0.3, theta=+0.3, nu=1.5 passes the risk-neutral
/// constructor (martingale term 1 - 0.45 - 0.0675 = 0.4825 > 0) but the
/// Carr-Madan denominator at the default damping point u = -2.5i is
/// 1 - 0.45*2.5 - 0.0675*6.25 = -0.546875, so powf continued on the principal
/// branch and the K=50 call came back ~9.0 against a lower bound of 50.99.
#[test]
fn vg_positive_theta_high_nu_is_not_below_intrinsic() {
    let maturity = 1.0;
    let cf = VarianceGammaCharFn::risk_neutral(SPOT, RATE, DIV, maturity, 0.3, 0.3, 1.5)
        .expect("VG constructor accepts this set");
    let result = carr_madan_price_at_strikes(
        &cf,
        RATE,
        maturity,
        SPOT,
        &STRIKES,
        CarrMadanParams::default(),
    );
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, maturity, "vg sigma=0.3 theta=0.3 nu=1.5"),
    }
}

/// Audit case 5: CGMY with M=1.2 passes the constructor (which only requires
/// M > 1) but E[S^2.5] needs M > 2.5; the base of (M - i*u)^Y turns negative
/// real on the contour and powc crosses the branch cut. Before the fix the
/// K=50 call was 3.67 against a lower bound of 50.99.
#[test]
fn cgmy_m_slightly_above_one_is_not_below_intrinsic() {
    let maturity = 1.0;
    let cf = CgmyCharFn::risk_neutral(SPOT, RATE, DIV, maturity, 0.5, 5.0, 1.2, 0.5)
        .expect("CGMY constructor accepts M=1.2");
    let result = carr_madan_price_at_strikes(
        &cf,
        RATE,
        maturity,
        SPOT,
        &STRIKES,
        CarrMadanParams::default(),
    );
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, maturity, "cgmy M=1.2"),
    }
}

/// CGMY with M=1.05 admits no damping alpha at or above the fallback floor
/// (0.1 requires a finite 1.1-th moment, but moments explode at 1.05), so the
/// pricing call must fail cleanly instead of returning garbage.
#[test]
fn cgmy_without_admissible_alpha_errors_cleanly() {
    let maturity = 1.0;
    let cf = CgmyCharFn::risk_neutral(SPOT, RATE, DIV, maturity, 0.5, 5.0, 1.05, 0.5)
        .expect("CGMY constructor accepts M=1.05");
    let result = carr_madan_price_at_strikes(
        &cf,
        RATE,
        maturity,
        SPOT,
        &STRIKES,
        CarrMadanParams::default(),
    );
    let err = result.expect_err("no admissible damping alpha exists for M=1.05");
    assert!(
        err.contains("inadmissible"),
        "error should explain the admissibility failure, got: {err}"
    );
}

/// NIG with alpha_nig=2, beta=0.5 passes the constructor (|beta+1| < alpha)
/// but the default damping moment 2.5 violates |beta + 2.5| < alpha_nig, so
/// the CF square root turns imaginary on the contour.
#[test]
fn nig_damping_beyond_martingale_bound_is_not_below_intrinsic() {
    let maturity = 1.0;
    let cf = NigCharFn::risk_neutral(SPOT, RATE, DIV, maturity, 2.0, 0.5, 0.5)
        .expect("NIG constructor accepts alpha=2, beta=0.5");
    let result = carr_madan_price_at_strikes(
        &cf,
        RATE,
        maturity,
        SPOT,
        &STRIKES,
        CarrMadanParams::default(),
    );
    match result {
        Err(msg) => assert!(!msg.is_empty(), "error message must be descriptive"),
        Ok(prices) => assert_arbitrage_free(&prices, maturity, "nig alpha=2 beta=0.5"),
    }
}

/// The automatic fallback must be observable: a VG model whose admissible
/// moment bound (m < 1.7584) rejects the default alpha=1.5 gets a reduced
/// alpha in the reusable context, and the context still prices sanely.
#[test]
fn context_reports_reduced_alpha_when_fallback_engages() {
    let maturity = 1.0;
    let cf = VarianceGammaCharFn::risk_neutral(SPOT, RATE, DIV, maturity, 0.3, 0.3, 1.5)
        .expect("VG constructor accepts this set");
    let ctx = CarrMadanContext::new(&cf, RATE, maturity, SPOT, CarrMadanParams::default())
        .expect("an admissible fallback alpha exists (any alpha < 0.7584)");
    assert!(
        ctx.params().alpha < DEFAULT_ALPHA,
        "context must expose the reduced admissible alpha, got {}",
        ctx.params().alpha
    );
    let prices = ctx
        .price_strikes(&STRIKES)
        .expect("context pricing succeeds");
    assert_arbitrage_free(&prices, maturity, "vg fallback context");
}

/// Deterministic finite-grid lock for the automatic small-alpha Heston path.
///
/// The QuantLib and finer-grid assertions below test provenance and
/// convergence. This test separately pins the production fallback decision
/// and its binary64 outputs: default alpha 1.5 is inadmissible, so the context
/// selects alpha 0.1875 and refines the grid by eight in both resolution and
/// frequency spacing.
#[test]
fn heston_small_alpha_fallback_finite_grid_binary64_lock() {
    use openferric::engines::fft::HestonCharFn;

    let (v0, kappa, theta, xi, rho, maturity) = (0.04, 2.0, 0.04, 1.5, 0.7, 5.0);
    let cf = HestonCharFn::new(SPOT, RATE, DIV, maturity, v0, kappa, theta, xi, rho);
    let context = CarrMadanContext::new(&cf, RATE, maturity, SPOT, CarrMadanParams::default())
        .expect("small-alpha Heston fallback context builds");
    let effective = context.params();
    assert_eq!(effective.alpha, 0.1875);
    assert_eq!(effective.n, 32_768);
    assert_eq!(effective.eta, 0.03125);

    let expected: [f64; 3] = [
        54.860_113_436_677_324,
        18.623_918_174_295_15,
        9.818_703_664_151_924,
    ];
    let prices = context
        .price_strikes(&STRIKES)
        .expect("small-alpha Heston fallback prices");
    for (((actual_strike, actual_price), expected_strike), expected_price) in
        prices.iter().zip(STRIKES).zip(expected)
    {
        assert_eq!(*actual_strike, expected_strike);
        let binary64_budget = 256.0 * f64::EPSILON * expected_price.abs().max(1.0);
        assert!(
            (actual_price - expected_price).abs() <= binary64_budget,
            "Heston small-alpha finite-grid lock drifted at K={expected_strike}: actual={actual_price}, expected={expected_price}, budget={binary64_budget}"
        );
    }
}

/// The automatic fallback must refine the frequency grid alongside the
/// reduced alpha. A small alpha sharpens the damped integrand's peak near
/// v = 0 (peak width scales with alpha), and the default eta = 0.25 that was
/// tuned for alpha = 1.5 under-resolves it: before the grid refinement the
/// Heston kappa=2, xi=1.5, rho=+0.7, T=5 fallback (alpha 1.5 -> 0.1875)
/// returned calls with a ~0.9 currency-unit strike-independent error - within
/// the model-free bounds, so only visible on the parity put (K=50 put of 1.03
/// instead of ~0.102).
///
/// Anchors:
/// - Parity put at K=50 is 0.10216768794637598 from QuantLib-Python 1.43's
///   `AnalyticHestonEngine` with order-192 Gauss-Laguerre integration. The
///   auto-refined FFT is required to be within 3e-4 of that independent
///   reference; its residual discretization error is deliberately explicit.
/// - Self-consistency: an explicitly finer admissible grid must agree with
///   the auto-refined fallback grid to quadrature accuracy.
#[test]
fn fallback_refines_quadrature_grid() {
    use openferric::engines::fft::{CarrMadanContext, HestonCharFn};

    let (v0, kappa, theta, xi, rho, t) = (0.04, 2.0, 0.04, 1.5, 0.7, 5.0);
    let prices = try_heston_price_fft(SPOT, &STRIKES, RATE, DIV, v0, kappa, theta, xi, rho, t)
        .expect("fallback alpha exists for this set");
    let put50 = prices[0].1 - SPOT + STRIKES[0] * (-RATE * t).exp();
    let quantlib_put50 = 0.102_167_687_946_375_98;
    assert!(
        (put50 - quantlib_put50).abs() < 3e-4,
        "K=50 parity put {put50} deviates from the QuantLib reference \
         {quantlib_put50}: \
         the fallback frequency grid under-resolves the small-alpha integrand peak"
    );

    // Self-consistency against an explicitly refined grid (the context also
    // resolves admissibility, exercising the n-cap branch of the refinement).
    let cf = HestonCharFn::new(SPOT, RATE, DIV, t, v0, kappa, theta, xi, rho);
    let fine = CarrMadanParams {
        n: 65536,
        eta: 0.015625,
        alpha: DEFAULT_ALPHA,
    };
    let refined = CarrMadanContext::new(&cf, RATE, t, SPOT, fine)
        .expect("refined context builds")
        .price_strikes(&STRIKES)
        .expect("refined context prices");
    for ((k_a, c_a), (k_b, c_b)) in prices.iter().zip(refined.iter()) {
        assert_eq!(k_a, k_b);
        assert!(
            (c_a - c_b).abs() < 1e-4,
            "fallback grid call {c_a} at K={k_a} disagrees with refined-grid call {c_b}"
        );
    }
}

/// Healthy-regime anchor: the QuantLib Lewis dataset (cached values from
/// vendor/QuantLib/test-suite/hestonmodel.cpp, same source as
/// tests/fixtures/heston_expected_put_call.csv) must still price through the
/// default path with alpha untouched - the admissibility layer must not
/// perturb valid parameterizations.
#[test]
fn healthy_heston_reference_price_is_unchanged() {
    // spot=100, r=0.01, q=0.02, T=1, v0=0.04, kappa=4.0, theta=0.25,
    // sigma_v=1.0, rho=-0.5; expected K=100 call = 16.070154917028834.
    let prices = try_heston_price_fft(100.0, &[100.0], 0.01, 0.02, 0.04, 4.0, 0.25, 1.0, -0.5, 1.0)
        .expect("healthy Heston set must price without error");
    let expected = 16.070154917028834;
    assert!(
        (prices[0].1 - expected).abs() < 5e-12,
        "QuantLib Lewis anchor drifted: got {} expected {expected}",
        prices[0].1
    );
}

/// Healthy-regime anchor: Carr-Madan under Black-Scholes matches the closed
/// form C = S*N(d1) - K*exp(-r*T)*N(d2) = 8.916037278572539 for S=K=100,
/// r=0.02, q=0, sigma=0.2, T=1 (d1=0.2, d2=0).
#[test]
fn healthy_black_scholes_anchor_is_unchanged() {
    use openferric::engines::fft::BlackScholesCharFn;

    let cf = BlackScholesCharFn::new(100.0, 0.02, 0.0, 0.2, 1.0);
    let prices =
        carr_madan_price_at_strikes(&cf, 0.02, 1.0, 100.0, &[100.0], CarrMadanParams::default())
            .expect("BS pricing succeeds");
    let expected = 8.916_037_278_572_539;
    assert!(
        (prices[0].1 - expected).abs() < 3e-12,
        "BS closed-form anchor drifted: got {} expected {expected}",
        prices[0].1
    );
}
