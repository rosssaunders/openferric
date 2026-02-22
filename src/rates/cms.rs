//! Rates analytics for Cms.
//!
//! Module openferric::rates::cms contains pricing and conventions for fixed-income instruments.

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

/// Hagan-Woodward linear CMS convexity adjustment.
///
/// Adjusted CMS rate ≈ S + convexity_adjustment
/// where the adjustment comes from the correlation between
/// the swap rate and the discount factor.
pub fn cms_convexity_adjustment(params: &CmsConvexityParams) -> f64 {
    if params.expiry <= 0.0 || params.tenor <= 0.0 {
        return 0.0;
    }

    let s = params.swap_rate;
    let t = params.expiry;
    let sigma = params.vol;

    // Hagan (2003) linear TSR approximation:
    // CA ≈ S^2 * sigma^2 * T * duration / annuity
    // where duration ≈ (1 - (1+S)^{-n}) / S for n periods
    let n = params.tenor;
    let duration = if s.abs() > 1e-10 {
        (1.0 - (1.0 + s).powf(-n)) / s
    } else {
        n
    };

    s * s * sigma * sigma * t * duration / params.annuity.max(1e-10)
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

/// SABR-based CMS convexity adjustment (Hagan & Woodward).
///
/// Uses SABR parameters to compute a more accurate convexity adjustment
/// that accounts for smile effects.
pub fn sabr_cms_convexity_adjustment(
    swap_rate: f64,
    annuity: f64,
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
    let n = tenor;
    let t = expiry;

    // Duration
    let duration = if s.abs() > 1e-10 {
        (1.0 - (1.0 + s).powf(-n)) / s
    } else {
        n
    };

    // SABR ATM vol approximation
    let f_beta = s.powf(beta);
    let atm_vol = alpha / f_beta * (1.0 + (
        (1.0 - beta).powi(2) / 24.0 * alpha * alpha / f_beta.powi(2)
        + 0.25 * rho * beta * nu * alpha / f_beta
        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu
    ) * t);

    // Apply same Hagan formula with SABR vol
    s * s * atm_vol * atm_vol * t * duration / annuity.max(1e-10)
}

#[cfg(test)]
mod tests {
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
        assert!(ca > 0.0);
        assert!(ca.is_finite());
        // Typical CA is a few bps
        assert!(ca < 0.01);
    }

    #[test]
    fn cms_spread_call_is_positive() {
        let option = CmsSpreadOption {
            strike: 0.005,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option,
            0.04, 0.025,         // 10Y=4%, 2Y=2.5%
            0.20, 0.25,          // vols
            0.85,                // correlation
            0.001, 0.0005,       // convexity adjustments
            0.03,                // discount rate
            10000, 42,
        ).unwrap();
        assert!(result.price > 0.0);
        assert!(result.price.is_finite());
    }

    #[test]
    fn cms_spread_put_is_positive_when_itm() {
        let option = CmsSpreadOption {
            strike: 0.05,  // Deep ITM put (spread ≈ 0.015)
            option_type: CmsSpreadOptionType::Put,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option,
            0.04, 0.025,
            0.20, 0.25,
            0.85,
            0.001, 0.0005,
            0.03,
            10000, 42,
        ).unwrap();
        assert!(result.price > 0.0);
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
        ).unwrap();
        let low_rho = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.3, 0.001, 0.0005, 0.03, 10000, 42,
        ).unwrap();
        // Lower correlation → higher spread vol → higher option value
        assert!(low_rho.price > high_rho.price);
    }

    #[test]
    fn sabr_convexity_adjustment_is_positive() {
        let ca = sabr_cms_convexity_adjustment(
            0.04, 8.5, 10.0, 1.0, 0.03, 0.5, -0.3, 0.4,
        );
        assert!(ca > 0.0);
        assert!(ca.is_finite());
    }

    #[test]
    fn expected_cms_rates_near_forward() {
        let option = CmsSpreadOption {
            strike: 0.01,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        };
        let result = cms_spread_option_mc(
            &option, 0.04, 0.025, 0.20, 0.25, 0.85, 0.001, 0.0005, 0.03, 50000, 42,
        ).unwrap();
        // Expected rates should be near the adjusted forwards
        assert!((result.expected_cms1 - 0.041).abs() < 0.005);
        assert!((result.expected_cms2 - 0.0255).abs() < 0.005);
    }
}
