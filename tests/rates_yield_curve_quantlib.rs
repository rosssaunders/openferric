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

// Measured against direct formulas and SciPy 1.17.1 `optimize.brentq`
// solutions of the same par equations. The semiannual 5Y pillar has the
// largest cross-solver difference, 1.110e-15.
const CURVE_ROUNDOFF_BUDGET: f64 = 2.0e-15;
const BOOTSTRAP_DF_BUDGET: f64 = 1.2e-15;
const BOOTSTRAP_RATE_BUDGET: f64 = 2.0e-15;

fn deposit_nodes(deposits: &[(f64, f64)]) -> Vec<(f64, f64)> {
    deposits
        .iter()
        .map(|&(tenor, rate)| (tenor, 1.0 / (1.0 + rate * tenor)))
        .collect()
}

/// Independent reference for the default interpolation: linear in log-DF,
/// with the implicit origin `(0, 1)` and linear log-DF extrapolation.
fn log_linear_discount(nodes: &[(f64, f64)], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }

    let mut augmented = Vec::with_capacity(nodes.len() + 1);
    augmented.push((0.0, 1.0));
    augmented.extend_from_slice(nodes);
    let segment = augmented
        .windows(2)
        .find(|pair| t <= pair[1].0)
        .unwrap_or_else(|| &augmented[augmented.len() - 2..]);
    let (t0, df0) = segment[0];
    let (t1, df1) = segment[1];
    let weight = (t - t0) / (t1 - t0);
    ((1.0 - weight) * df0.ln() + weight * df1.ln()).exp()
}

fn par_swap_rate(curve: &YieldCurve, tenor: f64, frequency: usize) -> f64 {
    let dt = 1.0 / frequency as f64;
    let periods = (tenor * frequency as f64).round() as usize;
    let annuity: f64 = (1..=periods)
        .map(|i| dt * curve.discount_factor(i as f64 * dt))
        .sum();
    (1.0 - curve.discount_factor(tenor)) / annuity
}

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
        assert_eq!(actual_df, expected_df, "deposit pillar {tenor}");
    }

    // DF(0) = 1
    assert_eq!(curve.discount_factor(0.0), 1.0);

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

/// Interpolated points reproduce the exact log-linear discount formula.
#[test]
fn yield_curve_deposits_interpolation() {
    let deposits = vec![(0.25, 0.05), (0.50, 0.055), (1.00, 0.06)];

    let curve = YieldCurveBuilder::from_deposits(&deposits);

    let nodes = deposit_nodes(&deposits);
    let df_0_25 = curve.discount_factor(0.25);
    let df_0_375 = curve.discount_factor(0.375);
    let df_0_50 = curve.discount_factor(0.50);
    let expected_mid = log_linear_discount(&nodes, 0.375);
    assert_relative_eq!(df_0_375, expected_mid, epsilon = CURVE_ROUNDOFF_BUDGET,);

    assert!(
        df_0_375 < df_0_25 && df_0_375 > df_0_50,
        "Interpolated DF must be between neighbors"
    );

    let z = curve.zero_rate(0.375);
    let expected_zero = -expected_mid.ln() / 0.375;
    assert_relative_eq!(z, expected_zero, epsilon = CURVE_ROUNDOFF_BUDGET);
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

    // Independent SciPy 1.17.1 `optimize.brentq` solutions of the annual par
    // equations under log-linear discount interpolation.
    const EXPECTED_DFS: [f64; 9] = [
        0.952_380_952_380_952_2,
        0.905_260_296_316_433_3,
        0.858_747_770_976_935_2,
        0.767_990_021_739_035_4,
        0.686_039_145_310_956_6,
        0.577_468_717_616_923_4,
        0.431_309_266_094_378_6,
        0.321_923_258_938_187_46,
        0.178_924_020_934_392_77,
    ];

    // DF(0) = 1
    assert_eq!(curve.discount_factor(0.0), 1.0);

    // Discount factors must be monotonically decreasing
    let mut prev_df = 1.0;
    for ((tenor, quote), expected_df) in swap_rates.iter().zip(EXPECTED_DFS) {
        let df = curve.discount_factor(*tenor);
        assert_relative_eq!(df, expected_df, epsilon = BOOTSTRAP_DF_BUDGET);
        assert_relative_eq!(
            par_swap_rate(&curve, *tenor, 1),
            *quote,
            epsilon = BOOTSTRAP_RATE_BUDGET,
        );
        assert!(
            df < prev_df,
            "DF must decrease: tenor={tenor}, df={df} >= prev={prev_df}"
        );
        assert!(df > 0.0, "DF must be positive: tenor={tenor}");
        prev_df = df;
    }

    // 1Y swap rate: (1 - DF(1)) / DF(1) = swap_rate for annual frequency.
    // The bootstrap solves each pillar so the par swap reprices exactly.
    let df1 = curve.discount_factor(1.0);
    let implied_1y = (1.0 - df1) / df1;
    assert_relative_eq!(implied_1y, 0.0500, epsilon = BOOTSTRAP_RATE_BUDGET);
}

/// Verify that bootstrapping from swap rates recovers par rates.
/// Par rate = (DF_0 - DF_N) / sum(DF_i) for an annual swap.
///
/// The bootstrap solves each pillar DF with a root-finder so the par swap
/// reprices exactly under the curve's interpolation, including tenors like
/// 5Y where year 4 is interpolated between the 3Y and 5Y pillars.
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

        assert_relative_eq!(par_rate, expected_rate, epsilon = BOOTSTRAP_RATE_BUDGET,);
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

        assert_relative_eq!(df2, expected_df2, epsilon = CURVE_ROUNDOFF_BUDGET,);
    }
}

/// Forward rates match the full-precision log-linear DF formula on an
/// upward-sloping deposit curve.
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
    let nodes = deposit_nodes(&deposits);

    let pairs = [
        (0.0 + 1e-6, 0.25),
        (0.25, 0.50),
        (0.50, 1.00),
        (1.00, 2.00),
        (2.00, 5.00),
    ];

    for &(t1, t2) in &pairs {
        let fwd = curve.forward_rate(t1, t2);
        let expected =
            (log_linear_discount(&nodes, t1) / log_linear_discount(&nodes, t2)).ln() / (t2 - t1);
        assert_relative_eq!(fwd, expected, epsilon = CURVE_ROUNDOFF_BUDGET);
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

        assert_relative_eq!(df, expected_df, epsilon = CURVE_ROUNDOFF_BUDGET,);
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
        assert_relative_eq!(curve.zero_rate(t), r, epsilon = CURVE_ROUNDOFF_BUDGET);
    }

    // Forward rate between any two tenors should be ~5%
    let pairs = [(0.5, 1.0), (1.0, 5.0), (5.0, 10.0), (0.5, 10.0)];
    for &(t1, t2) in &pairs {
        assert_relative_eq!(
            curve.forward_rate(t1, t2),
            r,
            epsilon = CURVE_ROUNDOFF_BUDGET,
        );
    }
}

/// Extrapolation beyond the last pillar continues the final log-DF slope.
#[test]
fn yield_curve_extrapolation_beyond_last_pillar() {
    let deposits = vec![(1.0, 0.05), (2.0, 0.055)];
    let curve = YieldCurveBuilder::from_deposits(&deposits);
    let nodes = deposit_nodes(&deposits);

    // DF at 3Y (extrapolated) should still be positive and less than DF at 2Y
    let df2 = curve.discount_factor(2.0);
    let df3 = curve.discount_factor(3.0);
    let expected_df3 = log_linear_discount(&nodes, 3.0);
    assert_relative_eq!(df3, expected_df3, epsilon = CURVE_ROUNDOFF_BUDGET);
    assert!(df3 > 0.0, "Extrapolated DF must be positive");
    assert!(df3 < df2, "Extrapolated DF must decrease");
}

// ── Swap bootstrap with semi-annual frequency ───────────────────────────────

/// Bootstrap with semi-annual frequency should also produce consistent par rates.
#[test]
fn yield_curve_swap_bootstrap_semiannual() {
    let swap_rates = vec![(1.0, 0.050), (2.0, 0.052), (5.0, 0.056)];

    let curve = YieldCurveBuilder::from_swap_rates(&swap_rates, 2);

    // Independent SciPy 1.17.1 `optimize.brentq` solutions of the semiannual
    // par equations under log-linear discount interpolation.
    const EXPECTED_DFS: [f64; 3] = [
        0.951_814_396_192_742_6,
        0.902_331_059_071_83,
        0.757_739_901_815_504_9,
    ];
    for ((tenor, _), expected_df) in swap_rates.iter().zip(EXPECTED_DFS) {
        assert_relative_eq!(
            curve.discount_factor(*tenor),
            expected_df,
            epsilon = BOOTSTRAP_DF_BUDGET,
        );
    }

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

    // Verify semi-annual par rate recovery for every input tenor.
    // The bootstrap stores DFs only at pillar tenors (1Y, 2Y, 5Y); the
    // intermediate coupon DFs (0.5Y, 1.5Y, ...) are interpolated, and each
    // pillar is solved so the par swap reprices exactly under that same
    // interpolation.
    for &(tenor, expected_rate) in &swap_rates {
        let periods = (tenor * 2.0).round() as usize;
        let annuity: f64 = (1..=periods)
            .map(|i| 0.5 * curve.discount_factor(i as f64 * 0.5))
            .sum();
        let df_n = curve.discount_factor(tenor);
        let par = (1.0 - df_n) / annuity;
        assert_relative_eq!(par, expected_rate, epsilon = BOOTSTRAP_RATE_BUDGET);
    }
}
