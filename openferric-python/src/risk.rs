use pyo3::prelude::*;

use openferric_core::credit::SurvivalCurve;
use openferric_core::rates::YieldCurve;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_cva(
    times: Vec<f64>,
    ee_profile: Vec<f64>,
    discount_rate: f64,
    hazard_rate: f64,
    lgd: f64,
) -> f64 {
    use openferric_core::risk::XvaCalculator;
    let discount_curve = YieldCurve::new(
        times
            .iter()
            .map(|t| (*t, (-discount_rate * *t).exp()))
            .collect(),
    );
    let hazards = vec![hazard_rate; times.len()];
    let survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);
    let own_survival = SurvivalCurve::from_piecewise_hazard(&times, &vec![0.0; times.len()]);
    let calc = XvaCalculator::new(discount_curve, survival, own_survival, lgd, 0.0);
    calc.cva_from_expected_exposure(&times, &ee_profile)
}

#[pyfunction]
pub fn py_sa_ccr_ead(
    replacement_cost: f64,
    notional: f64,
    maturity: f64,
    asset_class: &str,
) -> f64 {
    use openferric_core::risk::kva::{SaCcrAssetClass, sa_ccr_ead};
    let ac = match asset_class.to_ascii_lowercase().as_str() {
        "ir" | "interest_rate" => SaCcrAssetClass::InterestRate,
        "fx" | "foreign_exchange" => SaCcrAssetClass::ForeignExchange,
        "credit" => SaCcrAssetClass::Credit,
        "equity" => SaCcrAssetClass::Equity,
        "commodity" => SaCcrAssetClass::Commodity,
        _ => return f64::NAN,
    };
    sa_ccr_ead(replacement_cost, notional, maturity, ac)
}

#[pyfunction]
pub fn py_historical_var(returns: Vec<f64>, confidence: f64) -> f64 {
    use openferric_core::risk::var::historical_var;
    historical_var(&returns, confidence)
}

#[pyfunction]
pub fn py_historical_es(returns: Vec<f64>, confidence: f64) -> f64 {
    use openferric_core::risk::var::historical_expected_shortfall;
    historical_expected_shortfall(&returns, confidence)
}
