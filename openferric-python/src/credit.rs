use pyo3::prelude::*;

use openferric_core::credit::{Cds, SurvivalCurve};
use openferric_core::rates::YieldCurve;

use crate::helpers::tenor_grid;

#[pyfunction]
pub fn py_cds_npv(
    notional: f64,
    spread: f64,
    maturity: f64,
    recovery_rate: f64,
    payment_freq: usize,
    discount_rate: f64,
    hazard_rate: f64,
) -> f64 {
    if payment_freq == 0 {
        return f64::NAN;
    }

    let cds = Cds {
        notional,
        spread,
        maturity,
        recovery_rate,
        payment_freq,
    };

    let tenors = tenor_grid(maturity, payment_freq);
    let discount_curve = YieldCurve::new(
        tenors
            .iter()
            .map(|t| (*t, (-discount_rate * *t).exp()))
            .collect(),
    );

    let hazards = vec![hazard_rate.max(0.0); tenors.len()];
    let survival_curve = SurvivalCurve::from_piecewise_hazard(&tenors, &hazards);

    cds.npv(&discount_curve, &survival_curve)
}

#[pyfunction]
pub fn py_survival_prob(hazard_rate: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }

    let tt = t.max(1e-8);
    let tenors = vec![tt];
    let hazards = vec![hazard_rate.max(0.0)];
    let curve = SurvivalCurve::from_piecewise_hazard(&tenors, &hazards);
    curve.survival_prob(tt)
}
