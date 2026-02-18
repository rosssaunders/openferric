/// Margin Valuation Adjustment (MVA).
///
/// MVA = cost of posting initial margin over the life of the trade.
/// MVA = -∫ FundingSpread(t) * E[IM(t)] * DF(t) dt
///
/// Initial margin can be computed via:
/// - ISDA SIMM (sensitivity-based)
/// - Schedule-based approach
/// - Historical VaR
///
/// Reference: Andersen, Pykhtin, Sokol (2017) "Rethinking Margin Period of Risk"
use crate::rates::YieldCurve;

/// SIMM risk weight buckets (simplified).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimmRiskClass {
    InterestRate,
    CreditQualifying,
    CreditNonQualifying,
    Equity,
    Commodity,
    Fx,
}

/// Simplified SIMM initial margin calculation.
///
/// In practice SIMM is far more complex with bucket-level aggregation,
/// intra/inter-bucket correlations, and concentration thresholds.
/// This provides a first-order approximation.
#[derive(Debug, Clone)]
pub struct SimmMargin {
    /// Risk class.
    pub risk_class: SimmRiskClass,
    /// Net sensitivities (delta) per bucket.
    pub sensitivities: Vec<f64>,
    /// Risk weights per bucket.
    pub risk_weights: Vec<f64>,
    /// Intra-bucket correlation.
    pub intra_corr: f64,
}

impl SimmMargin {
    /// Compute SIMM-like initial margin for a single risk class.
    ///
    /// IM = sqrt(sum_i sum_j rho_ij * RW_i * s_i * RW_j * s_j)
    pub fn compute(&self) -> f64 {
        if self.sensitivities.len() != self.risk_weights.len() {
            return 0.0;
        }

        let n = self.sensitivities.len();
        let mut im2 = 0.0;

        for i in 0..n {
            for j in 0..n {
                let corr = if i == j { 1.0 } else { self.intra_corr };
                let ws_i = self.risk_weights[i] * self.sensitivities[i];
                let ws_j = self.risk_weights[j] * self.sensitivities[j];
                im2 += corr * ws_i * ws_j;
            }
        }

        im2.max(0.0).sqrt()
    }
}

/// Compute MVA from time-series of initial margin and funding spread.
///
/// # Arguments
/// * `times` - Time grid
/// * `expected_im` - Expected initial margin at each time
/// * `funding_spread` - Funding spread (cost of funding the margin)
/// * `discount_curve` - Discount curve
pub fn mva_from_profile(
    times: &[f64],
    expected_im: &[f64],
    funding_spread: &[f64],
    discount_curve: &YieldCurve,
) -> f64 {
    assert_eq!(times.len(), expected_im.len());
    assert_eq!(times.len(), funding_spread.len());

    let mut mva = 0.0;
    let mut prev_t = 0.0;

    for i in 0..times.len() {
        let t = times[i];
        let dt = t - prev_t;
        if dt <= 0.0 {
            prev_t = t;
            continue;
        }
        let df = discount_curve.discount_factor(t);
        mva -= funding_spread[i] * expected_im[i] * df * dt;
        prev_t = t;
    }

    mva
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simm_margin_single_bucket() {
        let simm = SimmMargin {
            risk_class: SimmRiskClass::InterestRate,
            sensitivities: vec![1_000_000.0],
            risk_weights: vec![0.017], // ~1.7% risk weight
            intra_corr: 1.0,
        };
        let im = simm.compute();
        assert!((im - 17_000.0).abs() < 0.01);
    }

    #[test]
    fn simm_margin_two_buckets_with_correlation() {
        let simm = SimmMargin {
            risk_class: SimmRiskClass::InterestRate,
            sensitivities: vec![1_000_000.0, 500_000.0],
            risk_weights: vec![0.02, 0.02],
            intra_corr: 0.5,
        };
        let im = simm.compute();
        assert!(im > 0.0);
        assert!(im.is_finite());
        // Should be less than sum of individual IMs (diversification)
        let im_1 = 0.02 * 1_000_000.0;
        let im_2 = 0.02 * 500_000.0;
        assert!(im < im_1 + im_2);
    }

    #[test]
    fn mva_is_negative_for_positive_im() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let expected_im = vec![50_000.0; 4];
        let funding_spread = vec![0.005; 4];
        let discount_curve = YieldCurve::new(
            times.iter().map(|&t| (t, (-0.03_f64 * t).exp())).collect(),
        );
        let mva = mva_from_profile(&times, &expected_im, &funding_spread, &discount_curve);
        assert!(mva < 0.0);
        assert!(mva.is_finite());
    }
}
