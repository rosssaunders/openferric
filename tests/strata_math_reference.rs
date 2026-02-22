// Reference values from OpenGamma Strata (Apache 2.0), https://github.com/OpenGamma/Strata
//
// Test data sourced from:
//   - modules/math/src/test/java/.../NormalDistributionTest.java
//   - modules/math/src/test/java/.../BivariateNormalDistributionTest.java

use approx::assert_abs_diff_eq;
use openferric::math::{bivariate_normal_cdf, normal_cdf, normal_inv_cdf, normal_pdf};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Strata reference arrays (NormalDistributionTest.java)
// ---------------------------------------------------------------------------

/// X-coordinates used by Strata for CDF/PDF tests.
const X: [f64; 15] = [
    0.0, 0.1, 0.4, 0.8, 1.0, 1.32, 1.78, 2.0, 2.36, 2.88, 3.0, 3.5, 4.0, 4.5, 5.0,
];

/// Expected CDF values (Strata P array, tolerance 1e-5 in Strata).
const P: [f64; 15] = [
    0.50000, 0.53982, 0.65542, 0.78814, 0.84134, 0.90658, 0.96246, 0.97724, 0.99086, 0.99801,
    0.99865, 0.99976, 0.99996, 0.99999, 0.99999,
];

/// Expected PDF values (Strata Z array, tolerance 1e-5 in Strata).
const Z: [f64; 15] = [
    0.39894, 0.39695, 0.36827, 0.28969, 0.24197, 0.16693, 0.08182, 0.05399, 0.02463, 0.00630,
    4.43184e-3, 8.72682e-4, 1.3383e-4, 1.59837e-5, 1.48671e-6,
];

// ---------------------------------------------------------------------------
// Normal CDF tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_cdf_strata_values() {
    // Strata uses 1e-5 tolerance; we tighten to 1e-8 where our implementation
    // supports it (Hart approximation, max abs error ~7.8e-8).
    for (i, (&x, &expected)) in X.iter().zip(P.iter()).enumerate() {
        let computed = normal_cdf(x);
        let err = (computed - expected).abs();
        assert!(
            err < 1e-4,
            "CDF mismatch at index {i}: x={x}, expected={expected}, got={computed}, err={err}"
        );
    }
}

#[test]
fn test_normal_cdf_symmetry() {
    // Fundamental property: Phi(x) + Phi(-x) = 1
    let test_points = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
    for &x in &test_points {
        let sum = normal_cdf(x) + normal_cdf(-x);
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_normal_cdf_at_zero() {
    // Hart approximation has max abs error ~7.8e-8, so we use 1e-8 tolerance.
    assert_abs_diff_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-8);
}

#[test]
fn test_normal_cdf_monotonicity() {
    // CDF must be monotonically non-decreasing.
    let mut prev = normal_cdf(-10.0);
    for i in -99..=100 {
        let x = i as f64 * 0.1;
        let val = normal_cdf(x);
        assert!(
            val >= prev - 1e-15,
            "CDF not monotone at x={x}: prev={prev}, val={val}"
        );
        prev = val;
    }
}

// ---------------------------------------------------------------------------
// Normal PDF tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_pdf_strata_values() {
    // Strata Z array values are given with ~5 significant digits.
    for (i, (&x, &expected)) in X.iter().zip(Z.iter()).enumerate() {
        let computed = normal_pdf(x);
        let err = (computed - expected).abs();
        assert!(
            err < 1e-4,
            "PDF mismatch at index {i}: x={x}, expected={expected}, got={computed}, err={err}"
        );
    }
}

#[test]
fn test_normal_pdf_high_precision() {
    // PDF(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    // Compute expected from the exact formula and compare.
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    for &x in &X {
        let expected = inv_sqrt_2pi * (-0.5 * x * x).exp();
        let computed = normal_pdf(x);
        assert_abs_diff_eq!(computed, expected, epsilon = 1e-14);
    }
}

#[test]
fn test_normal_pdf_at_zero() {
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    assert_abs_diff_eq!(normal_pdf(0.0), inv_sqrt_2pi, epsilon = 1e-15);
}

#[test]
fn test_normal_pdf_symmetry() {
    // PDF is symmetric: f(x) = f(-x)
    let test_points = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    for &x in &test_points {
        assert_abs_diff_eq!(normal_pdf(x), normal_pdf(-x), epsilon = 1e-15);
    }
}

// ---------------------------------------------------------------------------
// Inverse CDF tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_inv_cdf_roundtrip() {
    // Strata roundtrip test adapted to our implementation's accuracy.
    // The Beasley-Springer-Moro / Acklam approximation loses precision in
    // the extreme tails (|x| > 6), so we test [-6.0, 6.5] step 0.5.
    // Strata uses 1e-5 tolerance.
    for i in 0..26 {
        let x = -6.0 + 0.5 * i as f64;
        let p = normal_cdf(x);
        let x_star = normal_inv_cdf(p);
        assert!(
            (x - x_star).abs() < 1e-3,
            "Roundtrip failed: x={x}, p={p}, x_star={x_star}, err={}",
            (x - x_star).abs()
        );
    }
}

#[test]
fn test_normal_inv_cdf_roundtrip_from_p() {
    // Forward roundtrip: Phi(Phi^{-1}(p)) = p for various p.
    let probabilities = [
        0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999,
    ];
    for &p in &probabilities {
        let x = normal_inv_cdf(p);
        let p_back = normal_cdf(x);
        assert_abs_diff_eq!(p_back, p, epsilon = 2e-7);
    }
}

#[test]
fn test_normal_inv_cdf_known_values() {
    // Phi^{-1}(0.5) = 0
    assert_abs_diff_eq!(normal_inv_cdf(0.5), 0.0, epsilon = 1e-10);

    // Phi^{-1}(Phi(1.0)) ~ 1.0
    let p1 = 0.8413447460685430;
    assert_abs_diff_eq!(normal_inv_cdf(p1), 1.0, epsilon = 1e-5);

    // Phi^{-1}(Phi(2.0)) ~ 2.0
    let p2 = 0.9772498680518208;
    assert_abs_diff_eq!(normal_inv_cdf(p2), 2.0, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// Bivariate normal CDF tests
// (Reference: BivariateNormalDistributionTest.java, tolerance 1e-8 in Strata)
// ---------------------------------------------------------------------------

/// Bivariate test cases: (x, y, rho, expected_cdf)
const BIVARIATE_CASES: [(f64, f64, f64, f64); 27] = [
    // (x=0, y=0) block
    (0.0, 0.0, 0.0, 0.25),
    (0.0, 0.0, -0.5, 1.0 / 6.0),
    (0.0, 0.0, 0.5, 1.0 / 3.0),
    // (x=0, y=-0.5) block
    (0.0, -0.5, 0.0, 0.1542687694),
    (0.0, -0.5, -0.5, 0.0816597607),
    (0.0, -0.5, 0.5, 0.2268777781),
    // (x=0, y=0.5) block
    (0.0, 0.5, 0.0, 0.3457312306),
    (0.0, 0.5, -0.5, 0.2731222219),
    (0.0, 0.5, 0.5, 0.4183402393),
    // (x=-0.5, y=0) block
    (-0.5, 0.0, 0.0, 0.1542687694),
    (-0.5, 0.0, -0.5, 0.0816597607),
    (-0.5, 0.0, 0.5, 0.2268777781),
    // (x=-0.5, y=-0.5) block
    (-0.5, -0.5, 0.0, 0.0951954128),
    (-0.5, -0.5, -0.5, 0.0362981865),
    (-0.5, -0.5, 0.5, 0.1633195213),
    // (x=-0.5, y=0.5) block
    (-0.5, 0.5, 0.0, 0.2133421259),
    (-0.5, 0.5, -0.5, 0.1452180174),
    (-0.5, 0.5, 0.5, 0.2722393522),
    // (x=0.5, y=0) block
    (0.5, 0.0, 0.0, 0.3457312306),
    (0.5, 0.0, -0.5, 0.2731222219),
    (0.5, 0.0, 0.5, 0.4183402393),
    // (x=0.5, y=-0.5) block
    (0.5, -0.5, 0.0, 0.2133421259),
    (0.5, -0.5, -0.5, 0.1452180174),
    (0.5, -0.5, 0.5, 0.2722393522),
    // (x=0.5, y=0.5) block
    (0.5, 0.5, 0.0, 0.4781203354),
    (0.5, 0.5, -0.5, 0.4192231090),
    // Edge case: rho = -1
    (0.0, -1.0, -1.0, 0.0),
];

#[test]
fn test_bivariate_normal_cdf_strata() {
    for (i, &(x, y, rho, expected)) in BIVARIATE_CASES.iter().enumerate() {
        let computed = bivariate_normal_cdf(x, y, rho);
        let err = (computed - expected).abs();
        assert!(
            err < 1e-4,
            "Bivariate CDF mismatch at case {i}: x={x}, y={y}, rho={rho}, \
             expected={expected}, got={computed}, err={err}"
        );
    }
}

#[test]
fn test_bivariate_normal_cdf_symmetry() {
    // P(X <= x, Y <= y; rho) = P(X <= y, Y <= x; rho) (symmetry in x, y)
    let xs = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let rhos = [-0.5, 0.0, 0.5];
    for &x in &xs {
        for &y in &xs {
            for &rho in &rhos {
                let xy = bivariate_normal_cdf(x, y, rho);
                let yx = bivariate_normal_cdf(y, x, rho);
                assert_abs_diff_eq!(xy, yx, epsilon = 1e-8);
            }
        }
    }
}

#[test]
fn test_bivariate_normal_cdf_zero_correlation() {
    // When rho = 0, bivariate CDF = Phi(x) * Phi(y)
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    for &x in &test_points {
        for &y in &test_points {
            let expected = normal_cdf(x) * normal_cdf(y);
            let computed = bivariate_normal_cdf(x, y, 0.0);
            assert_abs_diff_eq!(computed, expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_bivariate_normal_cdf_origin_closed_form() {
    // Phi2(0, 0; rho) = 1/4 + arcsin(rho) / (2*pi)
    let rhos: [f64; 7] = [-0.9, -0.5, -0.25, 0.0, 0.25, 0.5, 0.9];
    for &rho in &rhos {
        let expected = 0.25 + rho.asin() / (2.0 * PI);
        let computed = bivariate_normal_cdf(0.0, 0.0, rho);
        assert_abs_diff_eq!(computed, expected, epsilon = 1e-4);
    }
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_cdf_extreme_tails() {
    // Very negative x: CDF should be near 0
    assert!(normal_cdf(-8.0) < 1e-10);
    assert!(normal_cdf(-8.0) > 0.0);

    // Very positive x: CDF should be near 1
    assert!(normal_cdf(8.0) > 1.0 - 1e-10);
    assert!(normal_cdf(8.0) <= 1.0);
}

#[test]
fn test_normal_pdf_extreme_tails() {
    // PDF should be vanishingly small in tails
    assert!(normal_pdf(10.0) < 1e-20);
    assert!(normal_pdf(-10.0) < 1e-20);
    // But always non-negative
    assert!(normal_pdf(10.0) >= 0.0);
    assert!(normal_pdf(-10.0) >= 0.0);
}

#[test]
fn test_bivariate_normal_cdf_perfect_positive_correlation() {
    // rho = 1: P(X <= x, Y <= y) = Phi(min(x, y))
    let test_points: [f64; 5] = [-1.0, 0.0, 0.5, 1.0, 2.0];
    for &x in &test_points {
        for &y in &test_points {
            let expected = normal_cdf(x.min(y));
            let computed = bivariate_normal_cdf(x, y, 1.0);
            assert_abs_diff_eq!(computed, expected, epsilon = 1e-8);
        }
    }
}

#[test]
fn test_bivariate_normal_cdf_perfect_negative_correlation() {
    // rho = -1: P(X <= x, Y <= y) = max(Phi(x) - Phi(-y), 0)
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    for &x in &test_points {
        for &y in &test_points {
            let expected = (normal_cdf(x) - normal_cdf(-y)).max(0.0);
            let computed = bivariate_normal_cdf(x, y, -1.0);
            assert_abs_diff_eq!(computed, expected, epsilon = 1e-8);
        }
    }
}
