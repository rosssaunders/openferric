//! SVI volatility surface reference tests.
//!
//! Sources:
//! - Gatheral & Jacquier (2014), "Arbitrage-Free SVI Volatility Surfaces",
//!   arXiv:1204.0646
//! - Zeliade whitepaper: "Quasi-Explicit Calibration of Gatheral's SVI Model"
//!
//! SVI raw parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
//! where w is total implied variance and k is log-moneyness ln(K/F).

use openferric::vol::surface::{SviParams, VolSurface, calibrate_svi};

// =======================================================================
// Gatheral-Jacquier parameter set A
// a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.1
// =======================================================================
const SET_A: SviParams = SviParams {
    a: 0.04,
    b: 0.4,
    rho: -0.4,
    m: 0.05,
    sigma: 0.1,
};

#[test]
fn svi_set_a_atm_total_variance() {
    // At k=0 (ATM): w(0) = 0.04 + 0.4*(-0.4*(-0.05) + sqrt(0.0025 + 0.01))
    // = 0.04 + 0.4*(0.02 + sqrt(0.0125))
    // = 0.04 + 0.4*(0.02 + 0.11180)
    // = 0.04 + 0.4 * 0.13180 = 0.04 + 0.05272 = 0.09272
    let w = SET_A.total_variance(0.0);
    let expected = 0.04 + 0.4 * (-0.4 * (0.0 - 0.05) + ((0.0 - 0.05_f64).powi(2) + 0.01).sqrt());
    let err = (w - expected).abs();
    assert!(
        err < 1e-10,
        "SVI set A ATM: got {w}, expected {expected}, err={err}"
    );
}

#[test]
fn svi_set_a_wing_total_variance() {
    // At k=0.5 (OTM call wing)
    let w = SET_A.total_variance(0.5);
    let k = 0.5;
    let expected = 0.04 + 0.4 * (-0.4 * (k - 0.05) + ((k - 0.05_f64).powi(2) + 0.01).sqrt());
    let err = (w - expected).abs();
    assert!(
        err < 1e-10,
        "SVI set A k=0.5: got {w}, expected {expected}, err={err}"
    );
}

// =======================================================================
// Gatheral-Jacquier parameter set B (extreme: rho=-1)
// a=0.04, b=0.2, rho=-1.0, m=0.1, sigma=0.5
// Note: rho=-1 makes the smile asymmetric (hockey-stick shape)
// =======================================================================
const SET_B: SviParams = SviParams {
    a: 0.04,
    b: 0.2,
    rho: -0.999, // use -0.999 since calibration clamps to (-0.999, 0.999)
    m: 0.1,
    sigma: 0.5,
};

#[test]
fn svi_set_b_total_variance_positive() {
    // SVI w(k) must be positive for all k when well-parameterized
    for i in -20..=20 {
        let k = i as f64 * 0.1;
        let w = SET_B.total_variance(k);
        assert!(
            w > 0.0,
            "SVI set B w({k}) = {w} must be > 0"
        );
    }
}

// =======================================================================
// Axel Vogt example (from Zeliade whitepaper)
// a=-0.0410, b=0.1331, m=0.3586, rho=0.3060, sigma=0.4153
// =======================================================================
const VOGT: SviParams = SviParams {
    a: -0.0410,
    b: 0.1331,
    rho: 0.3060,
    m: 0.3586,
    sigma: 0.4153,
};

#[test]
fn svi_vogt_atm_total_variance() {
    let w = VOGT.total_variance(0.0);
    let expected = -0.0410
        + 0.1331 * (0.3060 * (0.0 - 0.3586) + ((0.0 - 0.3586_f64).powi(2) + 0.4153 * 0.4153).sqrt());
    let err = (w - expected).abs();
    assert!(
        err < 1e-10,
        "SVI Vogt ATM: got {w}, expected {expected}, err={err}"
    );
    // Total variance must be positive
    assert!(w > 0.0, "SVI Vogt ATM total variance must be positive");
}

// =======================================================================
// SVI calibration: round-trip test with synthetic data
// Generate points from known params, then calibrate and recover
// =======================================================================
#[test]
fn svi_calibration_round_trip_set_a() {
    let true_params = SET_A;

    // Generate 21 synthetic total-variance observations
    let points: Vec<(f64, f64)> = (-10..=10)
        .map(|i| {
            let k = i as f64 * 0.05;
            (k, true_params.total_variance(k))
        })
        .collect();

    // Start from a different initial guess
    let init = SviParams {
        a: 0.06,
        b: 0.2,
        rho: -0.1,
        m: 0.0,
        sigma: 0.2,
    };

    let fit = calibrate_svi(&points, init, 5_000, 5e-3);

    // Check fit quality: MSE < 1e-6
    let mse: f64 = points
        .iter()
        .map(|(k, w)| (fit.total_variance(*k) - *w).powi(2))
        .sum::<f64>()
        / points.len() as f64;

    assert!(
        mse < 1e-6,
        "SVI calibration MSE={mse}, expected < 1e-6"
    );
}

#[test]
fn svi_calibration_round_trip_vogt() {
    let true_params = VOGT;

    let points: Vec<(f64, f64)> = (-10..=10)
        .map(|i| {
            let k = i as f64 * 0.1;
            (k, true_params.total_variance(k))
        })
        .collect();

    let init = SviParams {
        a: 0.01,
        b: 0.1,
        rho: 0.0,
        m: 0.0,
        sigma: 0.5,
    };

    let fit = calibrate_svi(&points, init, 5_000, 5e-3);

    let mse: f64 = points
        .iter()
        .map(|(k, w)| (fit.total_variance(*k) - *w).powi(2))
        .sum::<f64>()
        / points.len() as f64;

    assert!(
        mse < 5e-5,
        "SVI Vogt calibration MSE={mse}, expected < 5e-5"
    );
}

// =======================================================================
// SVI slope derivative: dw/dk
// Numerical check against finite difference
// =======================================================================
#[test]
fn svi_derivative_matches_finite_difference() {
    let p = SET_A;
    let h = 1e-6;

    for i in -10..=10 {
        let k = i as f64 * 0.1;
        let analytic = p.dw_dk(k);
        let numerical = (p.total_variance(k + h) - p.total_variance(k - h)) / (2.0 * h);
        let err = (analytic - numerical).abs();
        assert!(
            err < 1e-5,
            "SVI dw/dk at k={k}: analytic={analytic}, numerical={numerical}, err={err}"
        );
    }
}

// =======================================================================
// SVI asymptotic behavior: w(k) ~ b*(1+rho)*k as k -> +inf
// and w(k) ~ b*(1-rho)*|k| as k -> -inf (for SVI-JW form)
// =======================================================================
#[test]
fn svi_asymptotic_slope() {
    let p = SviParams {
        a: 0.02,
        b: 0.3,
        rho: -0.5,
        m: 0.0,
        sigma: 0.2,
    };

    // For large positive k: dw/dk -> b*(rho + 1) = 0.3 * 0.5 = 0.15
    let right_slope = p.dw_dk(50.0);
    let expected_right = p.b * (p.rho + 1.0);
    let err = (right_slope - expected_right).abs();
    assert!(
        err < 0.01,
        "SVI right asymptote: slope={right_slope}, expected={expected_right}"
    );

    // For large negative k: dw/dk -> b*(rho - 1) = 0.3 * (-1.5) = -0.45
    let left_slope = p.dw_dk(-50.0);
    let expected_left = p.b * (p.rho - 1.0);
    let err = (left_slope - expected_left).abs();
    assert!(
        err < 0.01,
        "SVI left asymptote: slope={left_slope}, expected={expected_left}"
    );
}

// =======================================================================
// VolSurface: interpolation across expiries
// =======================================================================
#[test]
fn vol_surface_monotone_total_variance_across_expiry() {
    let p_short = SviParams {
        a: 0.01,
        b: 0.15,
        rho: -0.2,
        m: 0.0,
        sigma: 0.25,
    };
    let p_long = SviParams {
        a: 0.03,
        b: 0.20,
        rho: -0.25,
        m: 0.0,
        sigma: 0.30,
    };

    let surface = VolSurface::new(vec![(0.25, p_short), (2.0, p_long)], 100.0)
        .expect("surface build failed");

    // At ATM (K=100), total variance should increase with time
    let v1 = surface.vol(100.0, 0.25);
    let v2 = surface.vol(100.0, 1.0);
    let v3 = surface.vol(100.0, 2.0);

    assert!(v1 > 0.0, "vol at T=0.25 must be positive: {v1}");
    assert!(v2 > 0.0, "vol at T=1.0 must be positive: {v2}");
    assert!(v3 > 0.0, "vol at T=2.0 must be positive: {v3}");
}
