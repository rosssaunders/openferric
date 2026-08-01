//! VaR / ES Reference Tests
//!
//! Reference values computed from analytical closed-form formulas.
//! Cross-validated against QuantLib gaussianstatistics.hpp, McNeil/Frey/Embrechts (2005),
//! and HandWiki Expected Shortfall article.
//!
//! Standard normal: VaR_alpha = Phi^{-1}(alpha), ES_alpha = phi(Phi^{-1}(alpha)) / (1 - alpha)
//! Scaled N(mu, sigma): VaR = mu + sigma * Phi^{-1}(alpha), ES = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
//! Cornish-Fisher: z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36
//! Delta-Normal: VaR = |delta| * sigma_annual * sqrt(h/252) * Phi^{-1}(alpha)
//! Delta-Gamma Normal: mean_loss = -0.5*gamma*sigma^2, var_loss = delta^2*sigma^2 + 0.5*gamma^2*sigma^4,
//!                     VaR = mean_loss + z * sqrt(var_loss)

use approx::{assert_abs_diff_eq, assert_relative_eq};
use openferric::math::normal_inv_cdf;
use openferric::risk::{
    cornish_fisher_var, delta_gamma_normal_var, delta_normal_var, historical_expected_shortfall,
    historical_var, normal_expected_shortfall,
};

/// Measured absolute error of the Acklam inverse-normal approximation used by
/// `normal_inv_cdf`, against SciPy 1.17.1 `special.ndtri` over the grids below.
const ACKLAM_ABS_ERROR: f64 = 5.1e-9;

// ============================================================================
// Normal VaR (standard normal): VaR_alpha = Phi^{-1}(alpha)
// ============================================================================

struct NormalVarCase {
    alpha: f64,
    expected_var: f64,
    expected_es: f64,
}

fn standard_normal_cases() -> Vec<NormalVarCase> {
    vec![
        NormalVarCase {
            alpha: 0.90,
            expected_var: 1.281_551_565_544_600_4,
            expected_es: 1.754_983_319_324_869,
        },
        NormalVarCase {
            alpha: 0.95,
            expected_var: 1.644_853_626_951_472_2,
            expected_es: 2.062_712_807_507_425_7,
        },
        NormalVarCase {
            alpha: 0.975,
            expected_var: 1.959_963_984_540_054,
            expected_es: 2.337_802_792_201_413_3,
        },
        NormalVarCase {
            alpha: 0.99,
            expected_var: 2.326_347_874_040_840_8,
            expected_es: 2.665_214_220_345_806,
        },
        NormalVarCase {
            alpha: 0.995,
            expected_var: 2.575_829_303_548_900_4,
            expected_es: 2.891_948_605_383_481_3,
        },
        NormalVarCase {
            alpha: 0.999,
            expected_var: 3.090_232_306_167_813,
            expected_es: 3.367_090_077_063_993,
        },
    ]
}

#[test]
fn standard_normal_var_matches_analytical() {
    for case in standard_normal_cases() {
        // VaR for standard normal N(0,1) is just Phi^{-1}(alpha)
        let computed_var = normal_inv_cdf(case.alpha);
        assert_abs_diff_eq!(computed_var, case.expected_var, epsilon = ACKLAM_ABS_ERROR);
    }
}

#[test]
fn standard_normal_es_matches_analytical() {
    for case in standard_normal_cases() {
        // ES for standard normal N(0,1): phi(Phi^{-1}(alpha)) / (1 - alpha)
        // Using the library's normal_expected_shortfall with mean=0, std=1
        let computed_es = normal_expected_shortfall(0.0, 1.0, case.alpha);
        // d[phi(z)/(1-alpha)]/dz = -z*ES, so this is the measured
        // inverse-CDF budget propagated through the analytical ES formula.
        let propagated_error =
            case.expected_var.abs() * case.expected_es * ACKLAM_ABS_ERROR + 2e-14;
        assert_abs_diff_eq!(computed_es, case.expected_es, epsilon = propagated_error);
    }
}

// ============================================================================
// Normal VaR/ES (scaled): VaR = mu + sigma * Phi^{-1}(alpha)
// ============================================================================

struct ScaledNormalCase {
    mu: f64,
    sigma: f64,
    alpha: f64,
    expected_var: f64,
    expected_es: f64,
}

fn scaled_normal_cases() -> Vec<ScaledNormalCase> {
    vec![
        ScaledNormalCase {
            mu: 0.01,
            sigma: 0.02,
            alpha: 0.95,
            expected_var: 0.042_897_072_539_029_45,
            expected_es: 0.051_254_256_150_148_524,
        },
        ScaledNormalCase {
            mu: 0.01,
            sigma: 0.02,
            alpha: 0.99,
            expected_var: 0.056_526_957_480_816_82,
            expected_es: 0.063_304_284_406_916_11,
        },
        ScaledNormalCase {
            mu: 0.0,
            sigma: 0.20,
            alpha: 0.95,
            expected_var: 0.328_970_725_390_294_5,
            expected_es: 0.412_542_561_501_485_2,
        },
        ScaledNormalCase {
            mu: 0.0,
            sigma: 0.20,
            alpha: 0.99,
            expected_var: 0.465_269_574_808_168_16,
            expected_es: 0.533_042_844_069_161_1,
        },
    ]
}

#[test]
fn scaled_normal_var_matches_analytical() {
    for case in scaled_normal_cases() {
        // VaR = mu + sigma * Phi^{-1}(alpha)
        let computed_var = case.mu + case.sigma * normal_inv_cdf(case.alpha);
        assert_abs_diff_eq!(
            computed_var,
            case.expected_var,
            epsilon = case.sigma * ACKLAM_ABS_ERROR + 2e-15
        );
    }
}

#[test]
fn scaled_normal_es_matches_analytical() {
    for case in scaled_normal_cases() {
        // ES = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
        let computed_es = normal_expected_shortfall(case.mu, case.sigma, case.alpha);
        let z = (case.expected_var - case.mu) / case.sigma;
        let propagated_error =
            z.abs() * (case.expected_es - case.mu).abs() * ACKLAM_ABS_ERROR + 2e-15;
        assert_abs_diff_eq!(computed_es, case.expected_es, epsilon = propagated_error);
    }
}

// ============================================================================
// Cornish-Fisher VaR
// z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36
// VaR = mu + sigma * z_cf
// ============================================================================

struct CornishFisherCase {
    skewness: f64,
    excess_kurtosis: f64,
    alpha: f64,
    expected_z_cf: f64,
}

fn cornish_fisher_cases() -> Vec<CornishFisherCase> {
    vec![
        CornishFisherCase {
            skewness: 0.0,
            excess_kurtosis: 0.0,
            alpha: 0.95,
            expected_z_cf: 1.644_853_626_951_472_2,
        },
        CornishFisherCase {
            skewness: 0.0,
            excess_kurtosis: 0.0,
            alpha: 0.99,
            expected_z_cf: 2.326_347_874_040_840_8,
        },
        CornishFisherCase {
            skewness: 0.5,
            excess_kurtosis: 0.0,
            alpha: 0.95,
            expected_z_cf: 1.782_286_569_015_465_4,
        },
        CornishFisherCase {
            skewness: -0.5,
            excess_kurtosis: 0.0,
            alpha: 0.95,
            expected_z_cf: 1.498_029_326_666_23,
        },
        CornishFisherCase {
            skewness: 0.0,
            excess_kurtosis: 3.0,
            alpha: 0.95,
            expected_z_cf: 1.584_311_387_262_627,
        },
        CornishFisherCase {
            skewness: 0.0,
            excess_kurtosis: 3.0,
            alpha: 0.99,
            expected_z_cf: 3.027_711_059_302_616_6,
        },
        CornishFisherCase {
            skewness: -1.0,
            excess_kurtosis: 5.0,
            alpha: 0.95,
            expected_z_cf: 1.240_909_935_344_996,
        },
        CornishFisherCase {
            skewness: 0.5,
            excess_kurtosis: 2.0,
            alpha: 0.95,
            expected_z_cf: 1.741_925_075_889_568_6,
        },
    ]
}

#[test]
fn cornish_fisher_var_matches_analytical() {
    // cornish_fisher_var(mean_loss, std_dev_loss, skewness, excess_kurtosis, confidence)
    // With mean=0, std=1 the result equals z_cf directly
    for case in cornish_fisher_cases() {
        let computed =
            cornish_fisher_var(0.0, 1.0, case.skewness, case.excess_kurtosis, case.alpha);
        // The polynomial propagates the Acklam input error by at most 2.2x
        // over this case grid.
        assert_abs_diff_eq!(computed, case.expected_z_cf, epsilon = 1.1e-8);
    }
}

#[test]
fn cornish_fisher_zero_moments_reduces_to_gaussian() {
    // With S=0, K=0 the Cornish-Fisher expansion reduces to the plain Gaussian quantile
    for alpha in &[0.90, 0.95, 0.99, 0.995] {
        let cf = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, *alpha);
        let gaussian = normal_inv_cdf(*alpha);
        assert_relative_eq!(cf, gaussian, epsilon = 1e-12);
    }
}

// ============================================================================
// Delta-Normal VaR
// VaR = |delta| * sigma_annual * sqrt(h/252) * Phi^{-1}(alpha)
// ============================================================================

struct DeltaNormalCase {
    delta: f64,
    annual_vol: f64,
    horizon_days: f64,
    alpha: f64,
    expected_var: f64,
}

fn delta_normal_cases() -> Vec<DeltaNormalCase> {
    vec![
        DeltaNormalCase {
            delta: 1.0,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.95,
            expected_var: 0.020_723_207_809_600_977,
        },
        DeltaNormalCase {
            delta: 1.0,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.029_309_228_274_932_753,
        },
        DeltaNormalCase {
            delta: 1.0,
            annual_vol: 0.20,
            horizon_days: 10.0,
            alpha: 0.95,
            expected_var: 0.065_532_537_103_328_06,
        },
        DeltaNormalCase {
            delta: 1.0,
            annual_vol: 0.20,
            horizon_days: 10.0,
            alpha: 0.99,
            expected_var: 0.092_683_917_810_595_23,
        },
        DeltaNormalCase {
            delta: 100.0,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 2.930_922_827_493_275,
        },
        DeltaNormalCase {
            delta: 0.5,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.014_654_614_137_466_377,
        },
    ]
}

#[test]
fn delta_normal_var_matches_analytical() {
    for case in delta_normal_cases() {
        let computed = delta_normal_var(case.delta, case.annual_vol, case.alpha, case.horizon_days);
        let quantile_scale =
            case.delta.abs() * case.annual_vol * (case.horizon_days / 252.0).sqrt();
        assert_abs_diff_eq!(
            computed,
            case.expected_var,
            epsilon = quantile_scale * ACKLAM_ABS_ERROR + 2e-15
        );
    }
}

// ============================================================================
// Delta-Gamma Normal VaR
// mean_loss = -0.5 * gamma * sigma^2
// var_loss = delta^2 * sigma^2 + 0.5 * gamma^2 * sigma^4
// VaR = mean_loss + z * sqrt(var_loss)
// ============================================================================

struct DeltaGammaNormalCase {
    delta: f64,
    gamma: f64,
    annual_vol: f64,
    horizon_days: f64,
    alpha: f64,
    expected_var: f64,
}

fn delta_gamma_normal_cases() -> Vec<DeltaGammaNormalCase> {
    vec![
        DeltaGammaNormalCase {
            delta: 1.0,
            gamma: 0.0,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.029_309_228_274_932_753,
        },
        DeltaGammaNormalCase {
            delta: 1.0,
            gamma: 0.5,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.029_269_836_499_961_45,
        },
        DeltaGammaNormalCase {
            delta: 0.0,
            gamma: 1.0,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.000_181_742_278_910_029_7,
        },
        DeltaGammaNormalCase {
            delta: 1.0,
            gamma: -0.5,
            annual_vol: 0.20,
            horizon_days: 1.0,
            alpha: 0.99,
            expected_var: 0.029_349_201_579_326_533,
        },
    ]
}

#[test]
fn delta_gamma_normal_var_matches_analytical() {
    for case in delta_gamma_normal_cases() {
        let computed = delta_gamma_normal_var(
            case.delta,
            case.gamma,
            case.annual_vol,
            case.alpha,
            case.horizon_days,
        );
        // The inverse-normal error is multiplied by the moment-matched loss standard deviation.
        let sigma = case.annual_vol * (case.horizon_days / 252.0).sqrt();
        let loss_std =
            (case.delta.powi(2) * sigma.powi(2) + 0.5 * case.gamma.powi(2) * sigma.powi(4)).sqrt();
        assert_abs_diff_eq!(
            computed,
            case.expected_var,
            epsilon = loss_std * ACKLAM_ABS_ERROR + 2e-15
        );
    }
}

#[test]
fn delta_gamma_with_zero_gamma_equals_delta_normal() {
    // When gamma=0, delta-gamma VaR should match delta-normal VaR
    for alpha in &[0.95, 0.99] {
        for horizon in &[1.0, 10.0] {
            let dn = delta_normal_var(1.0, 0.20, *alpha, *horizon);
            let dg = delta_gamma_normal_var(1.0, 0.0, 0.20, *alpha, *horizon);
            assert_relative_eq!(dn, dg, epsilon = 1e-12);
        }
    }
}

// ============================================================================
// Historical VaR / ES
// Sample: losses = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
//   i.e. PnL = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
// At 90%: rank=8.1, VaR=4+0.1*(5-4)=4.1, ES=5.0
// At 95%: rank=8.55, VaR=4+0.55*(5-4)=4.55, ES=5.0
// ============================================================================

#[test]
fn historical_var_matches_reference() {
    // PnL such that losses = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
    let pnl = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];

    let var_90 = historical_var(&pnl, 0.90);
    assert_abs_diff_eq!(var_90, 4.1, epsilon = 1e-15);

    let var_95 = historical_var(&pnl, 0.95);
    assert_abs_diff_eq!(var_95, 4.55, epsilon = 1e-15);
}

#[test]
fn historical_es_matches_reference() {
    let pnl = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];

    let es_90 = historical_expected_shortfall(&pnl, 0.90);
    assert_eq!(es_90, 5.0);

    let es_95 = historical_expected_shortfall(&pnl, 0.95);
    assert_eq!(es_95, 5.0);
}

// ============================================================================
// ES >= VaR property
// ============================================================================

#[test]
fn es_always_greater_or_equal_to_var() {
    // For standard normal across multiple confidence levels
    for alpha in &[0.90, 0.95, 0.975, 0.99, 0.995, 0.999] {
        let var = normal_inv_cdf(*alpha);
        let es = normal_expected_shortfall(0.0, 1.0, *alpha);
        assert!(
            es >= var,
            "ES ({}) should be >= VaR ({}) at alpha={}",
            es,
            var,
            alpha
        );
    }

    // For scaled normal
    for case in scaled_normal_cases() {
        assert!(
            case.expected_es >= case.expected_var,
            "ES ({}) should be >= VaR ({}) for mu={}, sigma={}, alpha={}",
            case.expected_es,
            case.expected_var,
            case.mu,
            case.sigma,
            case.alpha
        );
    }
}

// ============================================================================
// Portfolio VaR (2-asset)
// sigma_p = sqrt(w1^2*s1^2 + w2^2*s2^2 + 2*w1*w2*s1*s2*rho)
// portfolio_var = sigma_p * Phi^{-1}(alpha)
// ============================================================================

struct PortfolioVarCase {
    w1: f64,
    w2: f64,
    s1: f64,
    s2: f64,
    rho: f64,
    alpha: f64,
    expected_var: f64,
}

fn portfolio_var_cases() -> Vec<PortfolioVarCase> {
    vec![
        PortfolioVarCase {
            w1: 0.5,
            w2: 0.5,
            s1: 0.20,
            s2: 0.30,
            rho: 0.0,
            alpha: 0.99,
            expected_var: 0.419_388_327_221_044_7,
        },
        PortfolioVarCase {
            w1: 0.5,
            w2: 0.5,
            s1: 0.20,
            s2: 0.30,
            rho: 0.5,
            alpha: 0.99,
            expected_var: 0.507_015_764_523_235_6,
        },
        PortfolioVarCase {
            w1: 0.5,
            w2: 0.5,
            s1: 0.20,
            s2: 0.30,
            rho: 1.0,
            alpha: 0.99,
            expected_var: 0.581_586_968_510_210_2,
        },
        PortfolioVarCase {
            w1: 0.5,
            w2: 0.5,
            s1: 0.20,
            s2: 0.30,
            rho: -1.0,
            alpha: 0.99,
            expected_var: 0.116_317_393_702_042_1,
        },
    ]
}

/// Portfolio VaR computed manually from component volatilities and correlation,
/// using the library's normal_inv_cdf for the quantile.
fn compute_portfolio_var(w1: f64, w2: f64, s1: f64, s2: f64, rho: f64, alpha: f64) -> f64 {
    let sigma_p = (w1 * w1 * s1 * s1 + w2 * w2 * s2 * s2 + 2.0 * w1 * w2 * s1 * s2 * rho).sqrt();
    sigma_p * normal_inv_cdf(alpha)
}

#[test]
fn portfolio_var_two_asset_matches_analytical() {
    for case in portfolio_var_cases() {
        let computed =
            compute_portfolio_var(case.w1, case.w2, case.s1, case.s2, case.rho, case.alpha);
        let sigma_p = (case.w1.powi(2) * case.s1.powi(2)
            + case.w2.powi(2) * case.s2.powi(2)
            + 2.0 * case.w1 * case.w2 * case.s1 * case.s2 * case.rho)
            .sqrt();
        assert_abs_diff_eq!(
            computed,
            case.expected_var,
            epsilon = sigma_p * ACKLAM_ABS_ERROR + 2e-15
        );
    }
}

#[test]
fn portfolio_var_perfect_correlation_equals_undiversified() {
    // At rho=1, portfolio vol = w1*s1 + w2*s2 (no diversification benefit)
    let w1 = 0.5;
    let w2 = 0.5;
    let s1 = 0.20;
    let s2 = 0.30;
    let alpha = 0.99;

    let portfolio = compute_portfolio_var(w1, w2, s1, s2, 1.0, alpha);
    let undiversified = (w1 * s1 + w2 * s2) * normal_inv_cdf(alpha);
    assert_eq!(portfolio, undiversified);
}

#[test]
fn portfolio_var_n_identical_uncorrelated_scales_as_inverse_sqrt_n() {
    // N identical uncorrelated assets with equal weights: VaR scales as 1/sqrt(N)
    let sigma = 0.20;
    let alpha = 0.99;

    // Single asset VaR
    let single_var = sigma * normal_inv_cdf(alpha);

    for n in [2, 4, 10, 25, 100] {
        // Portfolio variance = N * (1/N)^2 * sigma^2 = sigma^2 / N
        let portfolio_sigma = sigma / (n as f64).sqrt();
        let portfolio_var = portfolio_sigma * normal_inv_cdf(alpha);
        let expected_ratio = 1.0 / (n as f64).sqrt();
        assert_relative_eq!(portfolio_var / single_var, expected_ratio, epsilon = 2e-16,);
    }
}
