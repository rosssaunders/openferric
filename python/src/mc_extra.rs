use crate::core::{OptionType, PricingResult};
use crate::helpers::catch_unwind_py;
use crate::instruments::VanillaOption;
use crate::market::Market;
use crate::math_bindings::{FactorCorrelationModel, FastRng};
use crate::models::Heston;
use crate::native_engines::MonteCarloPricingEngine;
use openferric_core::engines::monte_carlo as native;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyclass(module = "openferric", from_py_object, get_all, set_all)]
#[derive(Clone, Copy)]
pub struct HestonAadConfig {
    num_paths: usize,
    num_steps: usize,
    seed: u64,
}
#[pymethods]
impl HestonAadConfig {
    #[new]
    #[pyo3(signature = (num_paths, num_steps, seed=42))]
    fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
        }
    }
}
#[pyfunction]
fn heston_price_delta_aad(
    py: Python<'_>,
    option_type: OptionType,
    strike: f64,
    maturity: f64,
    spot: f64,
    rate: f64,
    model: Heston,
    config: HestonAadConfig,
) -> PyResult<(f64, f64)> {
    py.detach(|| {
        catch_unwind_py(|| {
            native::heston_price_delta_aad(
                option_type.to_core(),
                strike,
                maturity,
                spot,
                rate,
                model.to_core(),
                native::HestonAadConfig::new(config.num_paths, config.num_steps, config.seed),
            )
        })?
        .map_err(error)
    })
}
#[pyfunction]
fn mc_european_pathwise_aad(
    py: Python<'_>,
    engine: &MonteCarloPricingEngine,
    instrument: &VanillaOption,
    market: &Market,
) -> PyResult<PricingResult> {
    let instrument = instrument.to_core()?;
    let market = market.to_core()?;
    py.detach(|| {
        native::mc_european_pathwise_aad(&engine.inner, &instrument, &market)
            .map(Into::into)
            .map_err(error)
    })
}
#[pyfunction]
fn mc_european_qmc(
    py: Python<'_>,
    instrument: &VanillaOption,
    market: &Market,
    num_paths: usize,
    num_steps: usize,
) -> PyResult<PricingResult> {
    let instrument = instrument.to_core()?;
    let market = market.to_core()?;
    py.detach(|| {
        catch_unwind_py(|| native::mc_european_qmc(&instrument, &market, num_paths, num_steps))
            .map(Into::into)
    })
}
#[pyfunction]
fn mc_european_qmc_with_seed(
    py: Python<'_>,
    instrument: &VanillaOption,
    market: &Market,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> PyResult<PricingResult> {
    let instrument = instrument.to_core()?;
    let market = market.to_core()?;
    py.detach(|| {
        catch_unwind_py(|| {
            native::mc_european_qmc_with_seed(&instrument, &market, num_paths, num_steps, seed)
        })
        .map(Into::into)
    })
}

#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone)]
pub struct SoaPaths {
    num_steps: usize,
    num_paths: usize,
    levels: Vec<Vec<f64>>,
}
impl From<native::SoaPaths> for SoaPaths {
    fn from(value: native::SoaPaths) -> Self {
        Self {
            num_steps: value.num_steps,
            num_paths: value.num_paths,
            levels: value.levels,
        }
    }
}
#[pymethods]
impl SoaPaths {
    fn terminal(&self) -> Vec<f64> {
        self.levels.last().cloned().unwrap_or_default()
    }
}

macro_rules! paths_function {
    ($name:ident, $output:ty) => {
        #[pyfunction]
        fn $name(
            py: Python<'_>,
            spot: f64,
            rate: f64,
            dividend_yield: f64,
            vol: f64,
            expiry: f64,
            num_paths: usize,
            num_steps: usize,
            seed: u64,
        ) -> PyResult<$output> {
            py.detach(|| {
                catch_unwind_py(|| {
                    native::$name(
                        spot,
                        rate,
                        dividend_yield,
                        vol,
                        expiry,
                        num_paths,
                        num_steps,
                        seed,
                    )
                })
                .map(Into::into)
            })
        }
    };
}
paths_function!(simulate_gbm_paths_soa, SoaPaths);
paths_function!(simulate_gbm_paths_soa_scalar, SoaPaths);
paths_function!(simulate_gbm_terminal_soa, Vec<f64>);
paths_function!(simulate_gbm_terminal_soa_scalar, Vec<f64>);
macro_rules! call_function {
    ($name:ident) => {
        #[pyfunction]
        fn $name(
            py: Python<'_>,
            spot: f64,
            strike: f64,
            rate: f64,
            dividend_yield: f64,
            vol: f64,
            expiry: f64,
            num_paths: usize,
            num_steps: usize,
            seed: u64,
        ) -> PyResult<f64> {
            py.detach(|| {
                catch_unwind_py(|| {
                    native::$name(
                        spot,
                        strike,
                        rate,
                        dividend_yield,
                        vol,
                        expiry,
                        num_paths,
                        num_steps,
                        seed,
                    )
                })
            })
        }
    };
}
call_function!(mc_european_call_soa);
call_function!(mc_european_call_soa_scalar);

#[pyfunction]
fn cholesky_for_correlation(correlation: Vec<Vec<f64>>) -> PyResult<(Vec<Vec<f64>>, bool)> {
    native::cholesky_for_correlation(&correlation).map_err(error)
}
#[pyfunction]
fn sample_correlated_normals_cholesky(
    cholesky: Vec<Vec<f64>>,
    rng: &mut FastRng,
) -> PyResult<Vec<f64>> {
    let mut output = vec![0.0; cholesky.len()];
    native::sample_correlated_normals_cholesky(&cholesky, &mut rng.inner, &mut output)
        .map_err(error)?;
    Ok(output)
}
#[pyfunction]
fn sample_correlated_normals_cholesky_with_scratch(
    cholesky: Vec<Vec<f64>>,
    rng: &mut FastRng,
    mut scratch: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mut output = vec![0.0; cholesky.len()];
    native::correlated_mc::sample_correlated_normals_cholesky_with_scratch(
        &cholesky,
        &mut rng.inner,
        &mut scratch,
        &mut output,
    )
    .map_err(error)?;
    Ok((output, scratch))
}
#[pyfunction]
fn sample_correlated_normals_factor(
    model: &FactorCorrelationModel,
    rng: &mut FastRng,
) -> PyResult<Vec<f64>> {
    let model = model.to_core()?;
    let mut output = vec![0.0; model.n_assets()];
    native::sample_correlated_normals_factor(&model, &mut rng.inner, &mut output).map_err(error)?;
    Ok(output)
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    #[pyclass(module = "openferric", from_py_object, get_all)]
    #[derive(Clone, Copy)]
    pub struct GreeksGridPoint {
        spot: f64,
        vol: f64,
        delta: f64,
        gamma: f64,
        vega: f64,
    }
    macro_rules! price {
        ($name:ident) => {
            #[pyfunction]
            fn $name(
                py: Python<'_>,
                instrument: &VanillaOption,
                market: &Market,
                num_paths: usize,
                num_steps: usize,
            ) -> PyResult<PricingResult> {
                let instrument = instrument.to_core()?;
                let market = market.to_core()?;
                py.detach(|| {
                    catch_unwind_py(|| native::$name(&instrument, &market, num_paths, num_steps))
                        .map(Into::into)
                })
            }
        };
    }
    price!(mc_european_parallel);
    price!(mc_european_sequential);
    macro_rules! grid {
        ($name:ident) => {
            #[pyfunction]
            fn $name(
                py: Python<'_>,
                option_type: OptionType,
                strike: f64,
                rate: f64,
                dividend_yield: f64,
                expiry: f64,
                spots: Vec<f64>,
                vols: Vec<f64>,
            ) -> PyResult<Vec<GreeksGridPoint>> {
                py.detach(|| {
                    catch_unwind_py(|| {
                        native::$name(
                            option_type.to_core(),
                            strike,
                            rate,
                            dividend_yield,
                            expiry,
                            &spots,
                            &vols,
                        )
                    })
                    .map(|values| {
                        values
                            .into_iter()
                            .map(|value| GreeksGridPoint {
                                spot: value.spot,
                                vol: value.vol,
                                delta: value.delta,
                                gamma: value.gamma,
                                vega: value.vega,
                            })
                            .collect()
                    })
                })
            }
        };
    }
    grid!(mc_greeks_grid_parallel);
    grid!(mc_greeks_grid_sequential);
    pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<GreeksGridPoint>()?;
        module.add_function(wrap_pyfunction!(mc_european_parallel, module)?)?;
        module.add_function(wrap_pyfunction!(mc_european_sequential, module)?)?;
        module.add_function(wrap_pyfunction!(mc_greeks_grid_parallel, module)?)?;
        module.add_function(wrap_pyfunction!(mc_greeks_grid_sequential, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use openferric_core::engines::gpu as native_gpu;
    #[pyclass(module = "openferric", from_py_object, get_all)]
    #[derive(Clone, Copy)]
    pub struct GpuMcResult {
        price: f64,
        stderr: f64,
    }
    #[pyfunction]
    fn gpu_is_ready() -> bool {
        native_gpu::gpu_is_ready()
    }
    #[pyfunction]
    fn prewarm_gpu(py: Python<'_>) -> PyResult<()> {
        py.detach(native_gpu::prewarm_gpu)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    #[pyfunction]
    fn mc_european_gpu(
        py: Python<'_>,
        spot: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
        is_call: bool,
    ) -> PyResult<GpuMcResult> {
        let result = py
            .detach(|| {
                native_gpu::mc_european_gpu(
                    spot, strike, rate, vol, expiry, num_paths, num_steps, seed, is_call,
                )
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(GpuMcResult {
            price: result.price,
            stderr: result.stderr,
        })
    }
    pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<GpuMcResult>()?;
        module.add_function(wrap_pyfunction!(gpu_is_ready, module)?)?;
        module.add_function(wrap_pyfunction!(prewarm_gpu, module)?)?;
        module.add_function(wrap_pyfunction!(mc_european_gpu, module)?)?;
        Ok(())
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HestonAadConfig>()?;
    module.add_class::<SoaPaths>()?;
    macro_rules! functions { ($($name:ident),+) => { $(module.add_function(wrap_pyfunction!($name, module)?)?;)+ } }
    functions!(
        heston_price_delta_aad,
        mc_european_pathwise_aad,
        mc_european_qmc,
        mc_european_qmc_with_seed,
        simulate_gbm_paths_soa,
        simulate_gbm_paths_soa_scalar,
        simulate_gbm_terminal_soa,
        simulate_gbm_terminal_soa_scalar,
        mc_european_call_soa,
        mc_european_call_soa_scalar,
        cholesky_for_correlation,
        sample_correlated_normals_cholesky,
        sample_correlated_normals_cholesky_with_scratch,
        sample_correlated_normals_factor
    );
    #[cfg(feature = "parallel")]
    parallel::register(module)?;
    #[cfg(feature = "gpu")]
    gpu::register(module)?;
    Ok(())
}
