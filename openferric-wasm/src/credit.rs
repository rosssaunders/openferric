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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cds_fair_spread_positive() {
        let spread = cds_fair_spread(1_000_000.0, 5.0, 0.40, 0.02, 0.03);
        assert!(spread > 0.0);
    }

    #[test]
    fn cds_fair_spread_approx_lgd_times_hazard() {
        // Fair spread ≈ (1-R)*λ for flat curves
        let hazard = 0.02;
        let recovery = 0.40;
        let spread = cds_fair_spread(1_000_000.0, 5.0, recovery, hazard, 0.03);
        let approx = (1.0 - recovery) * hazard;
        assert!((spread - approx).abs() < 0.005);
    }

    #[test]
    fn cds_fair_spread_higher_hazard() {
        let spread_low = cds_fair_spread(1_000_000.0, 5.0, 0.40, 0.01, 0.03);
        let spread_high = cds_fair_spread(1_000_000.0, 5.0, 0.40, 0.05, 0.03);
        assert!(spread_high > spread_low);
    }

    #[test]
    fn cds_fair_spread_lower_recovery() {
        // Lower recovery → higher spread
        let spread_high_r = cds_fair_spread(1_000_000.0, 5.0, 0.60, 0.02, 0.03);
        let spread_low_r = cds_fair_spread(1_000_000.0, 5.0, 0.20, 0.02, 0.03);
        assert!(spread_low_r > spread_high_r);
    }
}
