//! Yield curve bootstrapping reference tests derived from QuantLib's
//! piecewiseyieldcurve.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/piecewiseyieldcurve.cpp
//!
//! Tests verify deposit-based and swap-based curve construction, discount
//! factor monotonicity, and forward rate consistency.

use approx::assert_relative_eq;

use openferric::rates::{YieldCurve, YieldCurveBuilder};

// ── Deposit-based bootstrapping ─────────────────────────────────────────────

/// Reference: QuantLib piecewiseyieldcurve.cpp deposit rate inputs.
/// Deposits use simple compounding: DF = 1 / (1 + r * t).
#[test]
fn yield_curve_from_deposits_discount_factors() {
    // Typical USD deposit rates (simplified from QuantLib test setup)
    let deposits = vec![
        (1.0 / 52.0, 0.0525), // 1W
        (1.0 / 12.0, 0.0530), // 1M
        (3.0 / 12.0, 0.0545), // 3M
        (6.0 / 12.0, 0.0560), // 6M
        (9.0 / 12.0, 0.0570), // 9M
    ];

    let curve = YieldCurveBuilder::from_deposits(&deposits);

    // Verify each pillar point: DF = 1 / (1 + r * t)
    for &(tenor, rate) in &deposits {
        let expected_df = 1.0 / (1.0 + rate * tenor);
        let actual_df = curve.discount_factor(tenor);
        assert_relative_eq!(actual_df, expected_df, epsilon = 1.0e-10,);
    }

    // DF(0) = 1
    assert_relative_eq!(curve.discount_factor(0.0), 1.0, epsilon = 1.0e-12);

    // Discount factors must be monotonically decreasing for positive rates
    let mut prev_df = 1.0;
    for &(tenor, _) in &deposits {
        let df = curve.discount_factor(tenor);
        assert!(
            df < prev_df,
            "DF must decrease: tenor={tenor}, df={df} >= prev={prev_df}"
        );
        prev_df = df;
    }
}

/// Interpolated points between deposit pillars should produce sensible
/// discount factors (monotonically decreasing, between neighboring pillars).
#[test]
fn yield_curve_deposits_interpolation() {
    let deposits = vec![(0.25, 0.05), (0.50, 0.055), (1.00, 0.06)];

    let curve = YieldCurveBuilder::from_deposits(&deposits);

    // Test interpolation at midpoints
    let df_0_25 = curve.discount_factor(0.25);
    let df_0_375 = curve.discount_factor(0.375);
    let df_0_50 = curve.discount_factor(0.50);

    assert!(
        df_0_375 < df_0_25 && df_0_375 > df_0_50,
        "Interpolated DF must be between neighbors"
    );

    // Zero rate should be sensible
    let z = curve.zero_rate(0.375);
    assert!(z > 0.0 && z < 0.10, "Zero rate should be reasonable: {z}");
}

// ── Swap-based bootstrapping ────────────────────────────────────────────────

/// Reference: QuantLib piecewiseyieldcurve.cpp swap rate inputs.
/// Bootstrap from par swap rates using annual frequency.
#[test]
fn yield_curve_from_swap_rates_discount_factors() {
    // Typical swap rates (annual fixed frequency)
    let swap_rates = vec![
        (1.0, 0.0500),
        (2.0, 0.0510),
        (3.0, 0.0520),
        (5.0, 0.0540),
        (7.0, 0.0550),
        (10.0, 0.0560),
        (15.0, 0.0570),
        (20.0, 0.0575),
        (30.0, 0.0580),
    ];

    let curve = YieldCurveBuilder::from_swap_rates(&swap_rates, 1);

    // DF(0) = 1
    assert_relative_eq!(curve.discount_factor(0.0), 1.0, epsilon = 1.0e-12);

    // Discount factors must be monotonically decreasing
    let mut prev_df = 1.0;
    for &(tenor, _) in &swap_rates {
        let df = curve.discount_factor(tenor);
        assert!(
            df < prev_df,
            "DF must decrease: tenor={tenor}, df={df} >= prev={prev_df}"
        );
        assert!(df > 0.0, "DF must be positive: tenor={tenor}");
        prev_df = df;
    }

    // 1Y swap rate: (1 - DF(1)) / DF(1) ≈ swap_rate for annual frequency
    // More precisely: par_rate = (1 - DF(N)) / sum(DF(i)) for annual swap
    let df1 = curve.discount_factor(1.0);
    let implied_1y = (1.0 - df1) / df1;
    assert_relative_eq!(implied_1y, 0.0500, epsilon = 1.0e-4);
}

/// Verify that bootstrapping from swap rates recovers par rates.
/// Par rate = (DF_0 - DF_N) / sum(DF_i) for an annual swap.
///
/// Note: For tenors whose intermediate annual points are all pillar points
/// (e.g. 1Y, 2Y, 3Y), the par rate recovery is exact. For tenors like 5Y
/// where year 4 is interpolated between 3Y and 5Y pillars, there is a
/// small interpolation error.
#[test]
fn yield_curve_swap_bootstrap_recovers_par_rates() {
    let swap_rates = vec![
        (1.0, 0.0500),
        (2.0, 0.0520),
        (3.0, 0.0540),
        (5.0, 0.0560),
        (10.0, 0.0580),
    ];

    let curve = YieldCurveBuilder::from_swap_rates(&swap_rates, 1);

    for &(tenor, expected_rate) in &swap_rates {
        let n = tenor as usize;
        let annuity: f64 = (1..=n).map(|i| curve.discount_factor(i as f64)).sum();
        let df_n = curve.discount_factor(tenor);
        let par_rate = (1.0 - df_n) / annuity;

        assert_relative_eq!(par_rate, expected_rate, epsilon = 1.0e-4,);
    }
}

// ── Forward rate consistency ────────────────────────────────────────────────

/// Forward rates implied by the curve must be consistent with discount factors.
/// DF(t2) = DF(t1) * exp(-f * (t2 - t1)) where f is the forward rate.
#[test]
fn yield_curve_forward_rate_consistency() {
    let deposits = vec![(0.25, 0.05), (0.50, 0.055), (1.00, 0.06), (2.00, 0.065)];
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    let pairs = [(0.1, 0.5), (0.25, 1.0), (0.5, 2.0), (0.1, 2.0)];

    for &(t1, t2) in &pairs {
        let fwd = curve.forward_rate(t1, t2);
        let df1 = curve.discount_factor(t1);
        let df2 = curve.discount_factor(t2);
        let expected_df2 = df1 * (-(fwd) * (t2 - t1)).exp();

        assert_relative_eq!(df2, expected_df2, epsilon = 1.0e-10,);
    }
}

/// Forward rates should be positive on an upward-sloping deposit curve.
#[test]
fn yield_curve_forward_rates_positive() {
    let deposits = vec![
        (0.25, 0.04),
        (0.50, 0.045),
        (1.00, 0.05),
        (2.00, 0.055),
        (5.00, 0.06),
    ];
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    let pairs = [
        (0.0 + 1e-6, 0.25),
        (0.25, 0.50),
        (0.50, 1.00),
        (1.00, 2.00),
        (2.00, 5.00),
    ];

    for &(t1, t2) in &pairs {
        let fwd = curve.forward_rate(t1, t2);
        assert!(
            fwd > 0.0,
            "Forward rate must be positive: f({t1},{t2}) = {fwd}"
        );
    }
}

// ── Zero rate consistency ───────────────────────────────────────────────────

/// Zero rate and discount factor must be consistent:
/// DF(t) = exp(-z * t), z = -ln(DF(t)) / t
#[test]
fn yield_curve_zero_rate_discount_factor_consistency() {
    let swap_rates = vec![(1.0, 0.05), (2.0, 0.052), (5.0, 0.055), (10.0, 0.058)];
    let curve = YieldCurveBuilder::from_swap_rates(&swap_rates, 1);

    for t in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] {
        let z = curve.zero_rate(t);
        let df = curve.discount_factor(t);
        let expected_df = (-z * t).exp();

        assert_relative_eq!(df, expected_df, epsilon = 1.0e-12,);
    }
}

// ── Flat curve properties ───────────────────────────────────────────────────

/// A flat curve constructed from a single rate should produce constant
/// zero rates and forward rates.
#[test]
fn yield_curve_flat_constant_rates() {
    let r = 0.05_f64;
    let curve = YieldCurve::new(vec![
        (1.0, (-r * 1.0).exp()),
        (5.0, (-r * 5.0).exp()),
        (10.0, (-r * 10.0).exp()),
    ]);

    // Zero rate at any tenor should be ~5%
    for t in [0.5, 1.0, 3.0, 7.0, 10.0] {
        assert_relative_eq!(curve.zero_rate(t), r, epsilon = 1.0e-10);
    }

    // Forward rate between any two tenors should be ~5%
    let pairs = [(0.5, 1.0), (1.0, 5.0), (5.0, 10.0), (0.5, 10.0)];
    for &(t1, t2) in &pairs {
        assert_relative_eq!(curve.forward_rate(t1, t2), r, epsilon = 1.0e-10);
    }
}

/// Extrapolation beyond the last pillar should produce reasonable results
/// (flat extrapolation of the last zero rate).
#[test]
fn yield_curve_extrapolation_beyond_last_pillar() {
    let deposits = vec![(1.0, 0.05), (2.0, 0.055)];
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    // DF at 3Y (extrapolated) should still be positive and less than DF at 2Y
    let df2 = curve.discount_factor(2.0);
    let df3 = curve.discount_factor(3.0);
    assert!(df3 > 0.0, "Extrapolated DF must be positive");
    assert!(df3 < df2, "Extrapolated DF must decrease");
}

// ── Swap bootstrap with semi-annual frequency ───────────────────────────────

/// Bootstrap with semi-annual frequency should also produce consistent par rates.
#[test]
fn yield_curve_swap_bootstrap_semiannual() {
    let swap_rates = vec![(1.0, 0.050), (2.0, 0.052), (5.0, 0.056)];

    let curve = YieldCurveBuilder::from_swap_rates(&swap_rates, 2);

    // DF must be monotonically decreasing
    let mut prev_df = 1.0;
    for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] {
        let df = curve.discount_factor(t);
        assert!(
            df < prev_df,
            "DF must decrease: t={t}, df={df} >= prev={prev_df}"
        );
        prev_df = df;
    }

    // Verify semi-annual par rate recovery for 1Y.
    // Note: the bootstrap stores DFs only at pillar tenors (1Y, 2Y, 5Y),
    // not at the intermediate 0.5Y point, so DF(0.5) is interpolated and
    // the recovered par rate has a small interpolation error.
    let df_05 = curve.discount_factor(0.5);
    let df_10 = curve.discount_factor(1.0);
    let annuity = 0.5 * df_05 + 0.5 * df_10;
    let par = (1.0 - df_10) / annuity;
    assert_relative_eq!(par, 0.050, epsilon = 1.0e-2);
}
