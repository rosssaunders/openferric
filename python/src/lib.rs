#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod calibration;
mod core;
mod credit;
mod data;
mod dsl;
mod dsl_data;
mod engines;
mod enums;
mod fft;
mod fft_extra;
mod functions_extra;
mod funding;
mod greeks_extra;
mod helpers;
mod instruments;
mod market;
mod math_bindings;
mod mc;
mod mc_extra;
mod model_extra;
mod models;
mod namespaces;
mod native_engines;
mod numerical;
mod pricing;
mod rates;
mod risk;
mod timeseries;
mod vol;

type Registrar = for<'py> fn(&Bound<'py, PyModule>) -> PyResult<()>;

fn aliases(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let entries = module.dict().items();
    for entry in entries.iter() {
        let name: String = entry.get_item(0)?.extract()?;
        if let Some(alias) = name
            .strip_prefix("py_")
            .or_else(|| name.strip_suffix("_py"))
            && !module.hasattr(alias)?
        {
            module.add(alias, entry.get_item(1)?)?;
        }
    }
    Ok(())
}

fn domain<'py>(
    root: &Bound<'py, PyModule>,
    name: &str,
    registrars: &[Registrar],
) -> PyResult<Bound<'py, PyModule>> {
    let full_name = format!("openferric.{name}");
    let child = PyModule::new(root.py(), &full_name)?;
    for register in registrars {
        register(&child)?;
    }
    aliases(&child)?;
    root.add(name, &child)?;
    root.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(&full_name, &child)?;
    Ok(child)
}

#[pyfunction]
fn build_features(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let features = PyDict::new(py);
    features.set_item("parallel", cfg!(feature = "parallel"))?;
    features.set_item("simd", cfg!(feature = "simd"))?;
    features.set_item("jit", cfg!(feature = "jit"))?;
    features.set_item("gpu", cfg!(feature = "gpu"))?;
    Ok(features.unbind())
}

#[pymodule]
pub fn openferric(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("MODEL_SABR", openferric_core::vol::slice::MODEL_SABR)?;
    module.add("MODEL_SVI", openferric_core::vol::slice::MODEL_SVI)?;
    module.add("MODEL_VV", openferric_core::vol::slice::MODEL_VV)?;

    core::register(module)?;
    credit::register(module)?;
    dsl::register(module)?;
    dsl_data::register(module)?;
    engines::register(module)?;
    native_engines::register(module)?;
    fft::register(module)?;
    fft_extra::register(module)?;
    funding::register(module)?;
    instruments::register(module)?;
    market::register(module)?;
    math_bindings::register(module)?;
    numerical::register(module)?;
    mc::register(module)?;
    mc_extra::register(module)?;
    models::register(module)?;
    model_extra::register(module)?;
    calibration::register(module)?;
    pricing::register(module)?;
    rates::register(module)?;
    risk::register(module)?;
    timeseries::register(module)?;
    vol::register(module)?;
    functions_extra::register(module)?;
    greeks_extra::register(module)?;
    enums::register(module)?;

    aliases(module)?;
    module.add_function(wrap_pyfunction!(build_features, module)?)?;
    module.add("__path__", Vec::<String>::new())?;
    domain(module, "core", &[core::register, dsl::register])?;
    domain(module, "credit", &[credit::register])?;
    domain(module, "dsl", &[dsl::register, dsl_data::register])?;
    domain(
        module,
        "engines",
        &[
            engines::register,
            native_engines::register,
            mc_extra::register,
        ],
    )?;
    domain(module, "fft", &[fft::register, fft_extra::register])?;
    domain(module, "funding", &[funding::register])?;
    let instruments = domain(module, "instruments", &[instruments::register])?;
    instruments.add("Frequency", module.getattr("Frequency")?)?;
    instruments.add("Portfolio", module.getattr("InstrumentPortfolio")?)?;
    domain(module, "market", &[market::register])?;
    let math = domain(
        module,
        "math",
        &[
            math_bindings::register,
            numerical::register,
            timeseries::register,
        ],
    )?;
    domain(module, "mc", &[mc::register, mc_extra::register])?;
    domain(module, "models", &[models::register, model_extra::register])?;
    domain(module, "calibration", &[calibration::register])?;
    domain(module, "pricing", &[pricing::register])?;
    domain(module, "rates", &[rates::register, funding::register])?;
    domain(module, "risk", &[risk::register])?;
    domain(module, "greeks", &[greeks_extra::register])?;
    let timeseries = domain(module, "timeseries", &[timeseries::register])?;
    math.add("timeseries", &timeseries)?;
    module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("openferric.math.timeseries", &timeseries)?;
    domain(module, "vol", &[vol::register])?;
    namespaces::register(module)?;

    let public_names = module
        .dict()
        .keys()
        .iter()
        .filter_map(|key| {
            let name = key.extract::<String>().ok()?;
            (!name.starts_with('_') || name == "__version__").then_some(name)
        })
        .collect::<Vec<_>>();
    module.add("__all__", public_names)?;

    Ok(())
}
