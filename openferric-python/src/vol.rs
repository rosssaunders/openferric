use pyo3::prelude::*;

use openferric_core::vol::implied::implied_vol;
use openferric_core::vol::sabr::SabrParams;

use crate::helpers::parse_option_type;

#[pyfunction]
pub fn py_implied_vol(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    market_price: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    implied_vol(
        option_type,
        spot,
        strike,
        rate,
        expiry,
        market_price,
        1e-10,
        100,
    )
    .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_sabr_vol(
    forward: f64,
    strike: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    SabrParams {
        alpha,
        beta,
        rho,
        nu,
    }
    .implied_vol(forward, strike, expiry)
}

#[pyfunction]
pub fn py_svi_vol(
    strike: f64,
    forward: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
) -> f64 {
    let k = (strike / forward).ln();
    let total_var = a + b * (rho * (k - m) + ((k - m).powi(2) + sigma * sigma).sqrt());
    if total_var > 0.0 {
        total_var.sqrt()
    } else {
        f64::NAN
    }
}
