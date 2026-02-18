/// Funding Valuation Adjustment (FVA).
///
/// FVA captures the cost/benefit of funding uncollateralized derivative exposures.
/// FVA = -∫ FundingSpread(t) * FundingExposure(t) * DF(t) dt
///
/// Reference: Green, "XVA: Credit, Funding and Capital Valuation Adjustments" (2015)
use crate::rates::YieldCurve;

/// Collateral agreement (CSA) terms.
#[derive(Debug, Clone, PartialEq)]
pub struct CsaTerms {
    /// Minimum transfer amount.
    pub mta: f64,
    /// Independent amount (initial margin threshold).
    pub threshold: f64,
    /// Margin period of risk in years (typically 10 business days ≈ 2 weeks).
    pub margin_period_of_risk: f64,
    /// Collateral posting frequency in years (e.g., 1/252 for daily).
    pub posting_frequency: f64,
    /// Whether collateral is posted (true = collateralized).
    pub collateralized: bool,
}

impl Default for CsaTerms {
    fn default() -> Self {
        Self {
            mta: 0.0,
            threshold: 0.0,
            margin_period_of_risk: 10.0 / 252.0,
            posting_frequency: 1.0 / 252.0,
            collateralized: false,
        }
    }
}

/// Compute FVA from exposure profile and funding spread curve.
///
/// # Arguments
/// * `times` - Time grid
/// * `funding_exposure` - Unsigned funding exposure at each time (E[max(V,0)] - collateral)
/// * `funding_spread` - Funding spread at each time point (e.g., bank's CDS spread)
/// * `discount_curve` - Risk-free discount curve
pub fn fva_from_profile(
    times: &[f64],
    funding_exposure: &[f64],
    funding_spread: &[f64],
    discount_curve: &YieldCurve,
) -> f64 {
    assert_eq!(times.len(), funding_exposure.len());
    assert_eq!(times.len(), funding_spread.len());

    let mut fva = 0.0;
    let mut prev_t = 0.0;

    for i in 0..times.len() {
        let t = times[i];
        let dt = t - prev_t;
        if dt <= 0.0 {
            prev_t = t;
            continue;
        }
        let df = discount_curve.discount_factor(t);
        fva -= funding_spread[i] * funding_exposure[i] * df * dt;
        prev_t = t;
    }

    fva
}

/// Compute funding exposure from raw exposure paths with CSA collateral.
///
/// Returns the average uncollateralized exposure at each time step.
pub fn funding_exposure_profile(
    exposure_paths: &[Vec<f64>],
    csa: &CsaTerms,
) -> Vec<f64> {
    if exposure_paths.is_empty() {
        return Vec::new();
    }

    let n_steps = exposure_paths[0].len();
    let mut profile = vec![0.0; n_steps];

    for path in exposure_paths {
        for (i, &mtm) in path.iter().enumerate() {
            let exposure = mtm.max(0.0);
            let collateral = if csa.collateralized {
                (exposure - csa.threshold).max(0.0)
            } else {
                0.0
            };
            let uncollateralized = (exposure - collateral).max(0.0);
            profile[i] += uncollateralized;
        }
    }

    let n = exposure_paths.len() as f64;
    for v in &mut profile {
        *v /= n;
    }
    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fva_is_negative_for_positive_exposure_and_spread() {
        let times = vec![0.25, 0.5, 0.75, 1.0];
        let funding_exposure = vec![100.0, 80.0, 60.0, 40.0];
        let funding_spread = vec![0.01; 4];
        let discount_curve = YieldCurve::new(
            times.iter().map(|&t| (t, (-0.03_f64 * t).exp())).collect(),
        );

        let fva = fva_from_profile(&times, &funding_exposure, &funding_spread, &discount_curve);
        assert!(fva < 0.0, "FVA should be negative (cost) for positive exposure");
        assert!(fva.is_finite());
    }

    #[test]
    fn funding_exposure_uncollateralized_equals_positive_mtm() {
        let paths = vec![
            vec![10.0, -5.0, 20.0],
            vec![-10.0, 5.0, 40.0],
        ];
        let csa = CsaTerms { collateralized: false, ..Default::default() };
        let profile = funding_exposure_profile(&paths, &csa);
        assert_eq!(profile.len(), 3);
        // t=0: avg(max(10,0), max(-10,0)) = avg(10, 0) = 5
        assert!((profile[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn collateralized_reduces_funding_exposure() {
        let paths = vec![vec![100.0; 4]];
        let uncoll = CsaTerms { collateralized: false, ..Default::default() };
        let coll = CsaTerms { collateralized: true, threshold: 0.0, ..Default::default() };

        let p_uncoll = funding_exposure_profile(&paths, &uncoll);
        let p_coll = funding_exposure_profile(&paths, &coll);

        assert!(p_uncoll[0] > p_coll[0]);
    }
}
