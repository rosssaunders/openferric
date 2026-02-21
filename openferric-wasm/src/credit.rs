use wasm_bindgen::prelude::*;

/// Approximate CDS fair spread from flat hazard rate and flat discount rate.
#[wasm_bindgen]
pub fn cds_fair_spread(
    _notional: f64,
    maturity: f64,
    recovery_rate: f64,
    hazard_rate: f64,
    discount_rate: f64,
) -> f64 {
    use openferric::credit::Cds;
    use openferric::credit::survival_curve::SurvivalCurve;
    use openferric::rates::YieldCurve;

    let tenors: Vec<(f64, f64)> = (1..=((maturity * 4.0).ceil() as u32))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-discount_rate * t).exp())
        })
        .collect();
    let discount_curve = YieldCurve::new(tenors);

    let surv_nodes: Vec<(f64, f64)> = (1..=((maturity * 4.0).ceil() as u32))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-hazard_rate * t).exp())
        })
        .collect();
    let survival_curve = SurvivalCurve::new(surv_nodes);

    let cds = Cds {
        notional: 1.0,
        spread: 0.01, // dummy
        maturity,
        recovery_rate,
        payment_freq: 4,
    };
    cds.fair_spread(&discount_curve, &survival_curve)
}
