use pyo3::prelude::*;

use openferric_core::rates::YieldCurve;

#[pyfunction]
pub fn py_swaption_price(
    notional: f64,
    strike: f64,
    swap_tenor: f64,
    option_expiry: f64,
    vol: f64,
    discount_rate: f64,
    option_type: &str,
) -> f64 {
    use openferric_core::rates::swaption::Swaption;
    let is_payer = match option_type.to_ascii_lowercase().as_str() {
        "payer" | "call" => true,
        "receiver" | "put" => false,
        _ => return f64::NAN,
    };
    let swaption = Swaption {
        notional,
        strike,
        option_expiry,
        swap_tenor,
        is_payer,
    };
    let tenors: Vec<(f64, f64)> = (1..=((option_expiry + swap_tenor).ceil() as usize * 4))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-discount_rate * t).exp())
        })
        .collect();
    let curve = YieldCurve::new(tenors);
    swaption.price(&curve, vol)
}
