// Test grids from OpenGamma Strata (Apache 2.0), https://github.com/OpenGamma/Strata
//
// Coordinates were sourced from:
//   - modules/math/src/test/java/.../NormalDistributionTest.java
//   - modules/math/src/test/java/.../BivariateNormalDistributionTest.java
// Rounded Strata expectations have been replaced by full-precision SciPy 1.17.1
// `special.ndtr` / adaptive-quadrature values so source-data rounding does not
// determine the error budget.

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

/// Full-precision CDF values at the Strata coordinates (SciPy `special.ndtr`).
const P: [f64; 15] = [
    0.5,
    0.539_827_837_277_029,
    0.655_421_741_610_324_2,
    0.788_144_601_416_603_4,
    0.841_344_746_068_542_9,
    0.906_582_491_006_528_2,
    0.962_462_019_651_483_3,
    0.977_249_868_051_820_8,
    0.990_862_532_469_427_4,
    0.998_011_624_145_105_7,
    0.998_650_101_968_369_9,
    0.999_767_370_920_964_5,
    0.999_968_328_758_166_9,
    0.999_996_602_326_875_3,
    0.999_999_713_348_428_1,
];

/// Full-precision PDF values at the Strata coordinates (analytical formula).
const Z: [f64; 15] = [
    0.398_942_280_401_432_7,
    0.396_952_547_477_011_8,
    0.368_270_140_303_323_3,
    0.289_691_552_761_482_7,
    0.241_970_724_519_143_37,
    0.166_937_041_741_713_8,
    0.081_827_775_992_142_82,
    0.053_990_966_513_188_06,
    0.024_631_269_306_382_507,
    0.006_306_726_396_265_927_5,
    0.004_431_848_411_938_007_5,
    0.000_872_682_695_045_760_2,
    0.000_133_830_225_764_885_37,
    0.000_015_983_741_106_905_478,
    0.000_001_486_719_514_734_298,
];

// ---------------------------------------------------------------------------
// Normal CDF tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_cdf_strata_values() {
    for (i, (&x, &expected)) in X.iter().zip(P.iter()).enumerate() {
        let computed = normal_cdf(x);
        let err = (computed - expected).abs();
        assert!(
            err <= 2e-15,
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
        assert_abs_diff_eq!(sum, 1.0, epsilon = 2e-16);
    }
}

#[test]
fn test_normal_cdf_at_zero() {
    assert_eq!(normal_cdf(0.0), 0.5);
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
    for (i, (&x, &expected)) in X.iter().zip(Z.iter()).enumerate() {
        let computed = normal_pdf(x);
        let err = (computed - expected).abs();
        assert!(
            err <= 5e-16,
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
        assert_abs_diff_eq!(computed, expected, epsilon = 5e-16);
    }
}

#[test]
fn test_normal_pdf_at_zero() {
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    assert_abs_diff_eq!(normal_pdf(0.0), inv_sqrt_2pi, epsilon = 1e-16);
}

#[test]
fn test_normal_pdf_symmetry() {
    // PDF is symmetric: f(x) = f(-x)
    let test_points = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    for &x in &test_points {
        assert_eq!(normal_pdf(x), normal_pdf(-x));
    }
}

// ---------------------------------------------------------------------------
// Inverse CDF tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_inv_cdf_scipy_reference() {
    // SciPy 1.17.1 `special.ndtri` references. Acklam's rational approximation
    // has a measured maximum absolute error below 5.1e-9 over this grid.
    let cases = [
        (1e-10, -6.361_340_902_404_056),
        (1e-8, -5.612_001_244_174_789),
        (1e-6, -4.753_424_308_822_899),
        (1e-3, -3.090_232_306_167_813),
        (0.01, -2.326_347_874_040_840_8),
        (0.1, -1.281_551_565_544_600_4),
        (0.5, 0.0),
        (0.9, 1.281_551_565_544_600_4),
        (0.99, 2.326_347_874_040_840_8),
        (0.999, 3.090_232_306_167_813),
        (1.0 - 1e-6, 4.753_424_308_817_087),
        (1.0 - 1e-8, 5.612_001_243_305_505),
        (1.0 - 1e-10, 6.361_340_889_697_422),
    ];
    for (p, expected) in cases {
        let actual = normal_inv_cdf(p);
        let error = (actual - expected).abs();
        assert!(
            error <= 5.1e-9,
            "inverse CDF p={p}: expected={expected}, got={actual}, error={error}"
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
        assert_abs_diff_eq!(p_back, p, epsilon = 6e-10);
    }
}

#[test]
fn test_normal_inv_cdf_known_values() {
    // Phi^{-1}(0.5) = 0
    assert_eq!(normal_inv_cdf(0.5), 0.0);

    // Phi^{-1}(Phi(1.0)) ~ 1.0
    let p1 = 0.841_344_746_068_543;
    assert_abs_diff_eq!(normal_inv_cdf(p1), 1.0, epsilon = 5e-9);

    // Phi^{-1}(Phi(2.0)) ~ 2.0
    let p2 = 0.9772498680518208;
    assert_abs_diff_eq!(normal_inv_cdf(p2), 2.0, epsilon = 5e-9);
}

// ---------------------------------------------------------------------------
// Bivariate normal CDF tests
// (Reference: BivariateNormalDistributionTest.java, tolerance 1e-8 in Strata)
// ---------------------------------------------------------------------------

/// Bivariate test cases: `(x, y, rho, expected_cdf)`. General-case values
/// come from independent adaptive integration of the bivariate density.
const BIVARIATE_CASES: [(f64, f64, f64, f64); 28] = [
    // (x=0, y=0) block
    (0.0, 0.0, 0.0, 0.25),
    (0.0, 0.0, -0.5, 1.0 / 6.0),
    (0.0, 0.0, 0.5, 1.0 / 3.0),
    // (x=0, y=-0.5) block
    (0.0, -0.5, 0.0, 0.154_268_769_362_993_44),
    (0.0, -0.5, -0.5, 0.081_659_760_654_531_68),
    (0.0, -0.5, 0.5, 0.226_877_778_071_455_2),
    // (x=0, y=0.5) block
    (0.0, 0.5, 0.0, 0.345_731_230_637_006_56),
    (0.0, 0.5, -0.5, 0.273_122_221_928_544_8),
    (0.0, 0.5, 0.5, 0.418_340_239_345_468_3),
    // (x=-0.5, y=0) block
    (-0.5, 0.0, 0.0, 0.154_268_769_362_993_44),
    (-0.5, 0.0, -0.5, 0.081_659_760_654_531_68),
    (-0.5, 0.0, 0.5, 0.226_877_778_071_455_2),
    // (x=-0.5, y=-0.5) block
    (-0.5, -0.5, 0.0, 0.095_195_412_803_089_85),
    (-0.5, -0.5, -0.5, 0.036_298_186_488_576_495),
    (-0.5, -0.5, 0.5, 0.163_319_521_309_063_35),
    // (x=-0.5, y=0.5) block
    (-0.5, 0.5, 0.0, 0.213_342_125_922_897_03),
    (-0.5, 0.5, -0.5, 0.145_218_017_416_923_53),
    (-0.5, 0.5, 0.5, 0.272_239_352_237_410_36),
    // (x=0.5, y=0) block
    (0.5, 0.0, 0.0, 0.345_731_230_637_006_56),
    (0.5, 0.0, -0.5, 0.273_122_221_928_544_8),
    (0.5, 0.0, 0.5, 0.418_340_239_345_468_3),
    // (x=0.5, y=-0.5) block
    (0.5, -0.5, 0.0, 0.213_342_125_922_897_03),
    (0.5, -0.5, -0.5, 0.145_218_017_416_923_53),
    (0.5, -0.5, 0.5, 0.272_239_352_237_410_36),
    // (x=0.5, y=0.5) block
    (0.5, 0.5, 0.0, 0.478_120_335_351_116_1),
    (0.5, 0.5, -0.5, 0.419_223_109_036_602_76),
    (0.5, 0.5, 0.5, 0.546_244_443_857_089_6),
    // Edge case: rho = -1
    (0.0, -1.0, -1.0, 0.0),
];

#[test]
fn test_bivariate_normal_cdf_strata() {
    for (i, &(x, y, rho, expected)) in BIVARIATE_CASES.iter().enumerate() {
        let computed = bivariate_normal_cdf(x, y, rho);
        let err = (computed - expected).abs();
        assert!(
            err <= 2e-15,
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
                assert_abs_diff_eq!(xy, yx, epsilon = 2e-15);
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
            assert_eq!(computed, expected);
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
        assert_abs_diff_eq!(computed, expected, epsilon = 2e-15);
    }
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_normal_cdf_extreme_tails() {
    // SciPy `special.ndtr` references, including the representable upper-tail value.
    assert_abs_diff_eq!(normal_cdf(-8.0), 6.220_960_574_271_74e-16, epsilon = 2e-29);
    assert_eq!(normal_cdf(8.0), 0.999_999_999_999_999_3);
}

#[test]
fn test_normal_pdf_extreme_tails() {
    let expected = 7.694_598_626_706_42e-23;
    assert_abs_diff_eq!(normal_pdf(10.0), expected, epsilon = 2e-37);
    assert_abs_diff_eq!(normal_pdf(-10.0), expected, epsilon = 2e-37);
}

#[test]
fn test_bivariate_normal_cdf_perfect_positive_correlation() {
    // rho = 1: P(X <= x, Y <= y) = Phi(min(x, y))
    let test_points: [f64; 5] = [-1.0, 0.0, 0.5, 1.0, 2.0];
    for &x in &test_points {
        for &y in &test_points {
            let expected = normal_cdf(x.min(y));
            let computed = bivariate_normal_cdf(x, y, 1.0);
            assert_eq!(computed, expected);
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
            assert_eq!(computed, expected);
        }
    }
}
