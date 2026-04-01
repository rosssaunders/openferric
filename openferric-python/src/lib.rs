#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod calibration;
mod core;
mod credit;
mod engines;
mod fft;
mod funding;
mod helpers;
mod instruments;
mod market;
mod math_bindings;
mod mc;
mod models;
mod pricing;
mod rates;
mod risk;
mod vol;

#[pymodule]
pub fn openferric(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("MODEL_SABR", openferric_core::vol::slice::MODEL_SABR)?;
    module.add("MODEL_SVI", openferric_core::vol::slice::MODEL_SVI)?;
    module.add("MODEL_VV", openferric_core::vol::slice::MODEL_VV)?;

    core::register(module)?;
    credit::register(module)?;
    engines::register(module)?;
    fft::register(module)?;
    funding::register(module)?;
    instruments::register(module)?;
    market::register(module)?;
    math_bindings::register(module)?;
    mc::register(module)?;
    models::register(module)?;
    calibration::register(module)?;
    pricing::register(module)?;
    rates::register(module)?;
    risk::register(module)?;
    vol::register(module)?;

    Ok(())
}
