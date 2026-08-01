//! Module `rates::cms`.
//!
//! Implements cms workflows with concrete routines such as `cms_convexity_adjustment`, `cms_spread_option_mc`, `sabr_cms_convexity_adjustment`.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `CmsConvexityParams`, `CmsSpreadOptionType`, `CmsSpreadOption`, `CmsSpreadResult` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
/// CMS (Constant Maturity Swap) spread options.
///
/// References:
/// - Hagan, "Convexity Conundrums" (2003)
/// - Pelsser, "Efficient Methods for Valuing Interest Rate Derivatives" (2000)
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// CMS convexity adjustment parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmsConvexityParams {
    /// Swap rate level.
    pub swap_rate: f64,
    /// Swap annuity (PV01).
    pub annuity: f64,
    /// Swap tenor in years.
    pub tenor: f64,
    /// Option expiry in years.
    pub expiry: f64,
    /// SABR/Black vol of the swap rate.
    pub vol: f64,
}

/// Hull (Ch. 30) CMS convexity adjustment kernel.
///
/// `CA = -0.5 * S^2 * sigma^2 * T * G''(S) / G'(S)`
///
/// where `G(y)` is the price of a bond paying annual coupons at the fixed
/// swap-rate level `S` over the swap tenor of `n` years (the module models the
/// underlying swap with annual periods, consistent with the `(1+S)^{-n}`
/// annuity used elsewhere in this file):
///
/// `G(y) = S * (1 - (1+y)^{-n}) / y + (1+y)^{-n}`
///
/// With `v(y) = (1+y)^{-n}`, `v' = -n (1+y)^{-n-1}`, `v'' = n(n+1)(1+y)^{-n-2}`:
///
/// `G'(y)  = S * (-y v' - 1 + v) / y^2 + v'`
/// `G''(y) = S * (-v'' y^2 + 2 v' y + 2(1 - v)) / y^3 + v''`
///
/// evaluated at `y = S`. `G' < 0` and `G'' > 0`, so the adjustment is positive.
fn hull_cms_convexity_adjustment(swap_rate: f64, tenor: f64, expiry: f64, vol: f64) -> f64 {
    if expiry <= 0.0 || tenor <= 0.0 {
        return 0.0;
    }

    let s = swap_rate;
    if s.abs() <= 1e-10 || s <= -1.0 {
        // CA carries an S^2 prefactor; at S ~ 0 the adjustment vanishes.
        return 0.0;
    }

    let n = tenor;
    let y = s;
    let one_plus = 1.0 + y;
    let v = one_plus.powf(-n);
    let vp = -n * one_plus.powf(-n - 1.0);
    let vpp = n * (n + 1.0) * one_plus.powf(-n - 2.0);

    let g_prime = s * (-y * vp - 1.0 + v) / (y * y) + vp;
    let g_double_prime = s * (-vpp * y * y + 2.0 * vp * y + 2.0 * (1.0 - v)) / (y * y * y) + vpp;

    if g_prime.abs() <= 1e-14 {
        return 0.0;
    }

    -0.5 * s * s * vol * vol * expiry * g_double_prime / g_prime
}

/// CMS convexity adjustment (Hull, Ch. 30).
///
/// Adjusted CMS rate ≈ S + convexity_adjustment, with
/// `CA = -0.5 * S^2 * sigma^2 * T * G''(S)/G'(S)` where `G(y)` is the
/// annuity-based bond-price function of the swap yield (see
/// [`hull_cms_convexity_adjustment`]). The `annuity` field of
/// [`CmsConvexityParams`] is retained for API compatibility but is not used by
/// this formula.
pub fn cms_convexity_adjustment(params: &CmsConvexityParams) -> f64 {
    hull_cms_convexity_adjustment(params.swap_rate, params.tenor, params.expiry, params.vol)
}

/// CMS spread option type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmsSpreadOptionType {
    Call,
    Put,
}

/// CMS spread option: payoff on S(T₁) - S(T₂) - K.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmsSpreadOption {
    /// Strike on the spread.
    pub strike: f64,
    /// Option type (call/put on the spread).
    pub option_type: CmsSpreadOptionType,
    /// Notional.
    pub notional: f64,
    /// Option expiry.
    pub expiry: f64,
}

/// CMS spread option pricing result.
#[derive(Debug, Clone)]
pub struct CmsSpreadResult {
    /// Present value.
    pub price: f64,
    /// Standard error of MC estimate.
    pub std_error: f64,
    /// Expected CMS1 rate.
    pub expected_cms1: f64,
    /// Expected CMS2 rate.
    pub expected_cms2: f64,
}

/// Price CMS spread option via Monte Carlo with correlated lognormal CMS rates.
///
/// CMS rates include convexity adjustments, then simulate as correlated
/// lognormal processes.
///
/// # Arguments
/// * `option` - CMS spread option definition
/// * `cms1_fwd` - Forward CMS rate 1 (e.g., 10Y swap rate)
/// * `cms2_fwd` - Forward CMS rate 2 (e.g., 2Y swap rate)
/// * `vol1` - Volatility of CMS rate 1
/// * `vol2` - Volatility of CMS rate 2
/// * `rho` - Correlation between CMS rates
/// * `ca1` - Convexity adjustment for CMS1
/// * `ca2` - Convexity adjustment for CMS2
/// * `discount_rate` - Risk-free rate for discounting
/// * `num_paths` - Number of MC paths
/// * `seed` - RNG seed
#[allow(clippy::too_many_arguments)]
pub fn cms_spread_option_mc(
    option: &CmsSpreadOption,
    cms1_fwd: f64,
    cms2_fwd: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    ca1: f64,
    ca2: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> Result<CmsSpreadResult, String> {
    if num_paths == 0 {
        return Err("num_paths must be > 0".to_string());
    }
    if vol1 <= 0.0 || vol2 <= 0.0 {
        return Err("volatilities must be > 0".to_string());
    }
    if rho.abs() > 1.0 {
        return Err("rho must be in [-1, 1]".to_string());
    }
    if option.expiry <= 0.0 {
        return Err("expiry must be > 0".to_string());
    }

    let adj_cms1 = cms1_fwd + ca1;
    let adj_cms2 = cms2_fwd + ca2;
    let t = option.expiry;
    let df = (-discount_rate * t).exp();
    let rho_comp = (1.0 - rho * rho).sqrt();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum_pv = 0.0;
    let mut sum_pv2 = 0.0;
    let mut sum_cms1 = 0.0;
    let mut sum_cms2 = 0.0;

    for _ in 0..num_paths {
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z2: f64 = StandardNormal.sample(&mut rng);
        let w2 = rho * z1 + rho_comp * z2;

        let cms1 = adj_cms1 * (-0.5 * vol1 * vol1 * t + vol1 * t.sqrt() * z1).exp();
        let cms2 = adj_cms2 * (-0.5 * vol2 * vol2 * t + vol2 * t.sqrt() * w2).exp();

        let spread = cms1 - cms2;
        let payoff = match option.option_type {
            CmsSpreadOptionType::Call => (spread - option.strike).max(0.0),
            CmsSpreadOptionType::Put => (option.strike - spread).max(0.0),
        };

        let pv = option.notional * payoff * df;
        sum_pv += pv;
        sum_pv2 += pv * pv;
        sum_cms1 += cms1;
        sum_cms2 += cms2;
    }

    let n = num_paths as f64;
    let mean = sum_pv / n;
    let variance = (sum_pv2 / n - mean * mean).max(0.0);

    Ok(CmsSpreadResult {
        price: mean,
        std_error: (variance / n).sqrt(),
        expected_cms1: sum_cms1 / n,
        expected_cms2: sum_cms2 / n,
    })
}

/// SABR-based CMS convexity adjustment.
///
/// Computes the SABR ATM lognormal vol from the SABR parameters and feeds it
/// into the Hull convexity adjustment
/// `CA = -0.5 * S^2 * sigma_ATM^2 * T * G''(S)/G'(S)`
/// (see [`hull_cms_convexity_adjustment`]). The `annuity` argument is retained
/// for API compatibility but is not used by this formula.
pub fn sabr_cms_convexity_adjustment(
    swap_rate: f64,
    _annuity: f64,
    tenor: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    if expiry <= 0.0 || tenor <= 0.0 {
        return 0.0;
    }

    let s = swap_rate;
    let t = expiry;

    // SABR ATM vol approximation (Hagan et al. 2002, ATM expansion):
    // sigma_ATM = alpha / f^(1-beta) * [1 + ((1-beta)^2/24 * alpha^2/f^(2-2beta)
    //             + rho*beta*nu*alpha/(4 f^(1-beta)) + (2-3rho^2)/24 * nu^2) * T]
    let f_pow = s.powf(1.0 - beta);
    let atm_vol = alpha / f_pow
        * (1.0
            + ((1.0 - beta).powi(2) / 24.0 * alpha * alpha / f_pow.powi(2)
                + 0.25 * rho * beta * nu * alpha / f_pow
                + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
                * t);

    hull_cms_convexity_adjustment(s, tenor, t, atm_vol)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use statrs::distribution::{ContinuousCDF, Normal};

    use super::*;

    #[test]
    fn convexity_adjustment_is_positive() {
        let params = CmsConvexityParams {
            swap_rate: 0.04,
            annuity: 8.5,
            tenor: 10.0,
            expiry: 1.0,
            vol: 0.20,
        };
        let ca = cms_convexity_adjustment(&params);
        // Hand-computed Hull value for S=4%, sigma=20%, T=1y, 10y annual tenor:
        // v = 1.04^-10, v' = -10*1.04^-11, v'' = 110*1.04^-12
        // G'(S)  = S(-S v' - 1 + v)/S^2 + v'  = -8.110896
        // G''(S) = S(-v'' S^2 + 2 v' S + 2(1-v))/S^3 + v'' = 80.754323
        // CA = -0.5 * 0.04^2 * 0.20^2 * 1.0 * G''/G' = 3.186008564594e-4 (~3.2bp)
        assert_relative_eq!(ca, 3.186_008_564_594e-4, epsilon = 1.0e-15);
    }

    #[test]
    fn general_rho_cms_spread_call_matches_scipy_and_quantlib() {
        let option = CmsSpreadOption {
            strike: 0.005,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option, 0.04, 0.025, // 10Y=4%, 2Y=2.5%
            0.20, 0.25, // vols
            0.85, // correlation
            0.001, 0.0005, // convexity adjustments
            0.03,   // discount rate
            200_000, 42,
        )
        .unwrap();

        // Independent SciPy 1.17.1 reference.  Condition on the first normal
        // factor, evaluate the second lognormal's truncated zeroth/first
        // moments analytically, then integrate the remaining standard-normal
        // factor with scipy.integrate.quad (epsabs=1e-14, epsrel=1e-13).
        const SCIPY_REFERENCE: f64 = 10_195.447_873_475_201;
        let scipy_error = (result.price - SCIPY_REFERENCE).abs();
        assert!(
            scipy_error <= 4.0 * result.std_error,
            "general-rho CMS spread: MC={} +/- {}, SciPy={}, error={scipy_error}",
            result.price,
            result.std_error,
            SCIPY_REFERENCE
        );

        // Second-library oracle generated with QuantLib-Python 1.43.  The two
        // adjusted forwards are modelled as correlated Black-Scholes assets
        // under a zero external rate and priced with SpreadBasketPayoff plus
        // MCLDEuropeanBasketEngine (one time step, seed 42).  QuantLib's raw
        // payoff expectation is then multiplied by exp(-.03)*1,000,000 to
        // apply this function's discounting and notional conventions.
        //
        // Required Sobol samples         discounted PV
        //             2^22       10,195.442862148004
        //             2^23       10,195.445455123237
        //             2^24       10,195.446638787185
        const QUANTLIB_2_POW_22: f64 = 10_195.442_862_148_004;
        const QUANTLIB_2_POW_23: f64 = 10_195.445_455_123_237;
        const QUANTLIB_REFERENCE: f64 = 10_195.446_638_787_185;
        let previous_increment = (QUANTLIB_2_POW_23 - QUANTLIB_2_POW_22).abs();
        let final_increment = (QUANTLIB_REFERENCE - QUANTLIB_2_POW_23).abs();
        assert!(final_increment < previous_increment);

        // Two final QMC increments conservatively cover the unresolved Sobol
        // discretisation; in particular they contain the independent SciPy
        // value, so this is an observed convergence budget rather than a wide
        // economic price band.
        let quantlib_reference_error = 2.0 * final_increment;
        assert!(
            (QUANTLIB_REFERENCE - SCIPY_REFERENCE).abs() <= quantlib_reference_error,
            "QuantLib reference has not converged to the conditional-quadrature value"
        );
        let quantlib_error = (result.price - QUANTLIB_REFERENCE).abs();
        assert!(
            quantlib_error <= 4.0 * result.std_error + quantlib_reference_error,
            "general-rho CMS spread: MC={} +/- {}, QuantLib={} +/- {}, error={quantlib_error}",
            result.price,
            result.std_error,
            QUANTLIB_REFERENCE,
            quantlib_reference_error
        );
    }

    #[test]
    fn cms_spread_mc_matches_black76_in_comonotonic_reduction() {
        let option = CmsSpreadOption {
            strike: 0.012,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 2.0,
        };
        let cms1_fwd = 0.04;
        let cms2_fwd = 0.025;
        let ca1 = 0.001;
        let ca2 = 0.0005;
        let vol = 0.20;
        let discount_rate = 0.03;

        // With rho=1 and equal volatilities, both rates are multiplied by the
        // same unit-mean lognormal variate.  Their difference is therefore a
        // single lognormal forward and the spread call reduces exactly to
        // Black-76.  `statrs` supplies an independent normal CDF.
        let spread_forward = (cms1_fwd + ca1) - (cms2_fwd + ca2);
        let sigma_sqrt_t = vol * option.expiry.sqrt();
        let d1 = ((spread_forward / option.strike).ln() + 0.5 * vol * vol * option.expiry)
            / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let expected = option.notional
            * (-discount_rate * option.expiry).exp()
            * (spread_forward * normal.cdf(d1) - option.strike * normal.cdf(d2));

        let result = cms_spread_option_mc(
            &option,
            cms1_fwd,
            cms2_fwd,
            vol,
            vol,
            1.0,
            ca1,
            ca2,
            discount_rate,
            200_000,
            7,
        )
        .unwrap();

        let error = (result.price - expected).abs();
        assert!(
            error <= 4.0 * result.std_error,
            "MC price {} differs from Black-76 reference {} by {} (> 4 stderr = {})",
            result.price,
            expected,
            error,
            4.0 * result.std_error
        );
    }

    #[test]
    fn cms_spread_put_is_positive_when_itm() {
        let option = CmsSpreadOption {
            strike: 0.05, // Deep ITM put (spread ≈ 0.015)
            option_type: CmsSpreadOptionType::Put,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.85, 0.001, 0.0005, 0.03, 10000, 42,
        )
        .unwrap();
        // Reproducibility regression only; the independent pricing reduction
        // is exercised by `cms_spread_mc_matches_black76_in_comonotonic_reduction`.
        assert_relative_eq!(result.price, 33_490.071_949_875_67, epsilon = 1.0e-9);
        assert_relative_eq!(result.std_error, 43.443_033_905_648_065, epsilon = 1.0e-10);
    }

    #[test]
    fn higher_correlation_reduces_spread_vol() {
        let option = CmsSpreadOption {
            strike: 0.01,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let high_rho = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.95, 0.001, 0.0005, 0.03, 10000, 42,
        )
        .unwrap();
        let low_rho = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.3, 0.001, 0.0005, 0.03, 10000, 42,
        )
        .unwrap();
        // Seeded implementation regressions, supplemented by the analytic
        // Black-76 reduction and lognormal-martingale checks in this module.
        assert_relative_eq!(low_rho.price, 6_671.306_015_827_68, epsilon = 1.0e-9);
        assert_relative_eq!(high_rho.price, 5_342.627_391_370_247, epsilon = 1.0e-9);
    }

    #[test]
    fn sabr_convexity_adjustment_is_positive() {
        let ca = sabr_cms_convexity_adjustment(0.04, 8.5, 10.0, 1.0, 0.03, 0.5, -0.3, 0.4);
        assert_relative_eq!(ca, 0.000_182_640_609_863_79, epsilon = 1.0e-15);
    }

    #[test]
    fn simulated_cms_means_match_lognormal_martingales() {
        let option = CmsSpreadOption {
            strike: 0.01,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.85, 0.001, 0.0005, 0.03, 50000, 42,
        )
        .unwrap();
        let n = 50_000.0_f64;
        let expected_cms1 = 0.041;
        let expected_cms2 = 0.0255;
        let cms1_std_error = expected_cms1 * ((0.20_f64.powi(2)).exp() - 1.0).sqrt() / n.sqrt();
        let cms2_std_error = expected_cms2 * ((0.25_f64.powi(2)).exp() - 1.0).sqrt() / n.sqrt();

        assert!((result.expected_cms1 - expected_cms1).abs() <= 4.0 * cms1_std_error);
        assert!((result.expected_cms2 - expected_cms2).abs() <= 4.0 * cms2_std_error);
    }
}
