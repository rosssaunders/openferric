/// Capital Valuation Adjustment (KVA).
///
/// KVA = cost of holding regulatory capital against derivative exposures.
/// KVA = -∫ HurdleRate * E[RegulatoryCapital(t)] * DF(t) dt
///
/// Regulatory capital under SA-CCR / IMM:
///   K = 12.5 * RiskWeight * EAD * LGD * MaturityAdjustment
///
/// Reference: Green (2015), Basel III SA-CCR framework

use crate::rates::YieldCurve;

/// SA-CCR asset class for add-on factors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SaCcrAssetClass {
    InterestRate,
    ForeignExchange,
    Credit,
    Equity,
    Commodity,
}

impl SaCcrAssetClass {
    /// Supervisory factor for the asset class (Basel III SA-CCR Table 2).
    pub fn supervisory_factor(&self) -> f64 {
        match self {
            SaCcrAssetClass::InterestRate => 0.005,
            SaCcrAssetClass::ForeignExchange => 0.04,
            SaCcrAssetClass::Credit => 0.05,
            SaCcrAssetClass::Equity => 0.32,
            SaCcrAssetClass::Commodity => 0.18,
        }
    }
}

/// Simplified SA-CCR EAD calculation.
///
/// EAD = alpha * (RC + PFE)
/// where alpha = 1.4, RC = replacement cost, PFE = potential future exposure
pub fn sa_ccr_ead(
    replacement_cost: f64,
    notional: f64,
    maturity: f64,
    asset_class: SaCcrAssetClass,
) -> f64 {
    let alpha = 1.4;
    let sf = asset_class.supervisory_factor();
    let maturity_factor = (maturity.min(1.0).max(10.0 / 252.0)).sqrt();
    let pfe = sf * notional * maturity_factor;
    let rc = replacement_cost.max(0.0);
    alpha * (rc + pfe)
}

/// Compute regulatory capital from EAD.
///
/// Under standardised approach:
///   K = RiskWeight * EAD * 8%
pub fn regulatory_capital(ead: f64, risk_weight: f64) -> f64 {
    risk_weight * ead * 0.08
}

/// Compute KVA from time-series of regulatory capital.
///
/// # Arguments
/// * `times` - Time grid
/// * `expected_capital` - Expected regulatory capital at each time
/// * `hurdle_rate` - Bank's cost of equity / hurdle rate
/// * `discount_curve` - Discount curve
pub fn kva_from_profile(
    times: &[f64],
    expected_capital: &[f64],
    hurdle_rate: f64,
    discount_curve: &YieldCurve,
) -> f64 {
    assert_eq!(times.len(), expected_capital.len());

    let mut kva = 0.0;
    let mut prev_t = 0.0;

    for i in 0..times.len() {
        let t = times[i];
        let dt = t - prev_t;
        if dt <= 0.0 {
            prev_t = t;
            continue;
        }
        let df = discount_curve.discount_factor(t);
        kva -= hurdle_rate * expected_capital[i] * df * dt;
        prev_t = t;
    }

    kva
}

/// Netting set aggregation: compute net exposure from a set of trade-level exposures.
///
/// With netting: E_net(t) = max(sum_i V_i(t), 0)
/// Without netting: E_gross(t) = sum_i max(V_i(t), 0)
pub fn netting_set_exposure(trade_mtms: &[f64], netting: bool) -> f64 {
    if netting {
        trade_mtms.iter().sum::<f64>().max(0.0)
    } else {
        trade_mtms.iter().map(|v| v.max(0.0)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sa_ccr_ead_is_positive() {
        let ead = sa_ccr_ead(50_000.0, 10_000_000.0, 5.0, SaCcrAssetClass::InterestRate);
        assert!(ead > 0.0);
        assert!(ead.is_finite());
    }

    #[test]
    fn regulatory_capital_scales_with_risk_weight() {
        let ead = 1_000_000.0;
        let k_low = regulatory_capital(ead, 0.20);
        let k_high = regulatory_capital(ead, 1.0);
        assert!(k_high > k_low);
        assert!((k_low - 16_000.0).abs() < 0.01); // 0.20 * 1M * 0.08
    }

    #[test]
    fn kva_is_negative_for_positive_capital() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_capital = vec![100_000.0; 5];
        let discount_curve = YieldCurve::new(
            times.iter().map(|&t| (t, (-0.03_f64 * t).exp())).collect(),
        );
        let kva = kva_from_profile(&times, &expected_capital, 0.10, &discount_curve);
        assert!(kva < 0.0);
        assert!(kva.is_finite());
    }

    #[test]
    fn netting_reduces_exposure() {
        let mtms = vec![100.0, -80.0, 50.0, -30.0];
        let net = netting_set_exposure(&mtms, true);
        let gross = netting_set_exposure(&mtms, false);
        assert!(net < gross);
        assert!((net - 40.0).abs() < 1e-10); // sum = 40
        assert!((gross - 150.0).abs() < 1e-10); // 100 + 0 + 50 + 0
    }

    #[test]
    fn supervisory_factors_are_ordered() {
        // IR < FX < Credit < Commodity < Equity
        assert!(SaCcrAssetClass::InterestRate.supervisory_factor()
            < SaCcrAssetClass::ForeignExchange.supervisory_factor());
        assert!(SaCcrAssetClass::ForeignExchange.supervisory_factor()
            < SaCcrAssetClass::Equity.supervisory_factor());
    }
}
