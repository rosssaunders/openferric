use super::*;
use crate::helpers::catch_unwind_py;
use std::sync::{Arc, Mutex};

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PricingArena {
    pub(crate) inner: Arc<Mutex<core_math::PricingArena>>,
}
impl PricingArena {
    pub(crate) fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, core_math::PricingArena>> {
        self.inner
            .lock()
            .map_err(|_| PyValueError::new_err("pricing arena is poisoned"))
    }
}
#[pymethods]
impl PricingArena {
    #[new]
    #[pyo3(signature = (max_paths=0, max_steps=0))]
    fn new(max_paths: usize, max_steps: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(core_math::PricingArena::with_capacity(
                max_paths, max_steps,
            ))),
        }
    }
    #[staticmethod]
    fn with_capacity(max_paths: usize, max_steps: usize) -> Self {
        Self::new(max_paths, max_steps)
    }
    #[getter]
    fn path_buffer(&self) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.path_buffer.clone())
    }
    #[setter]
    fn set_path_buffer(&self, values: Vec<f64>) -> PyResult<()> {
        self.lock()?.path_buffer = values;
        Ok(())
    }
    #[getter]
    fn payoff_buffer(&self) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.payoff_buffer.clone())
    }
    #[setter]
    fn set_payoff_buffer(&self, values: Vec<f64>) -> PyResult<()> {
        self.lock()?.payoff_buffer = values;
        Ok(())
    }
    #[getter]
    fn tree_buffer(&self) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.tree_buffer.clone())
    }
    #[setter]
    fn set_tree_buffer(&self, values: Vec<f64>) -> PyResult<()> {
        self.lock()?.tree_buffer = values;
        Ok(())
    }
    fn path_slice(&self, count: usize) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.path_slice(count).to_vec())
    }
    fn payoff_slice(&self, count: usize) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.payoff_slice(count).to_vec())
    }
    fn tree_slice(&self, count: usize) -> PyResult<Vec<f64>> {
        Ok(self.lock()?.tree_slice(count).to_vec())
    }
}

macro_rules! rng {
    ($name:ident, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name { inner: core_math::fast_rng::$name }
        #[pymethods]
        impl $name {
            #[new]
            fn new(seed: u64) -> Self { Self { inner: core_math::fast_rng::$name::seed_from_u64(seed) } }
            #[staticmethod]
            fn seed_from_u64(seed: u64) -> Self { Self::new(seed) }
            fn next_u64(&mut self) -> u64 { self.inner.next_u64() }
            fn next_f64(&mut self) -> f64 { self.inner.next_f64() }
            $($methods)*
        }
    }
}
rng!(Pcg64, {});
rng!(Xoshiro256PlusPlus, {
    fn next_f64_pair(&mut self) -> (f64, f64) {
        self.inner.next_f64_pair()
    }
});

#[pyfunction]
fn fill_normals(rng: &mut Xoshiro256PlusPlus, count: usize) -> Vec<f64> {
    let mut values = vec![0.0; count];
    core_math::fast_rng::fill_normals(&mut rng.inner, &mut values);
    values
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AnyInterpolator {
    inner: core_math::AnyInterpolator,
}
#[pymethods]
impl AnyInterpolator {
    #[new]
    fn new(interpolator: &Bound<'_, PyAny>) -> PyResult<Self> {
        macro_rules! select { ($($name:ty => $variant:ident),+) => { $(if let Ok(value) = interpolator.extract::<PyRef<$name>>() { return Ok(Self { inner: core_math::AnyInterpolator::$variant(value.inner.clone()) }); })+ } }
        select!(LinearInterpolator => Linear, LogLinearInterpolator => LogLinear, MonotoneConvexInterpolator => MonotoneConvex,
            TensionSplineInterpolator => TensionSpline, HermiteMonotoneInterpolator => HermiteMonotone,
            LogCubicMonotoneInterpolator => LogCubicMonotone, NelsonSiegelInterpolator => NelsonSiegel,
            NelsonSiegelSvenssonInterpolator => NelsonSiegelSvensson, SmithWilsonInterpolator => SmithWilson);
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a supported native interpolator",
        ))
    }
    fn value(&self, point: f64) -> PyResult<f64> {
        self.inner.value(point).map_err(interp_err)
    }
    fn derivative(&self, point: f64) -> PyResult<f64> {
        self.inner.derivative(point).map_err(interp_err)
    }
    fn jacobian(&self, point: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(point).map_err(interp_err)
    }
    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }
    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyfunction]
fn mc_european_with_arena(
    py: Python<'_>,
    instrument: &crate::instruments::VanillaOption,
    market: &crate::market::Market,
    num_paths: usize,
    num_steps: usize,
    arena: &PricingArena,
) -> PyResult<crate::core::PricingResult> {
    let instrument = instrument.to_core()?;
    let market = market.to_core()?;
    py.detach(|| {
        let mut arena = arena.lock()?;
        catch_unwind_py(std::panic::AssertUnwindSafe(|| {
            openferric_core::engines::monte_carlo::mc_european_with_arena(
                &instrument,
                &market,
                num_paths,
                num_steps,
                &mut arena,
            )
        }))
        .map(Into::into)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PricingArena>()?;
    module.add_class::<Pcg64>()?;
    module.add_class::<Xoshiro256PlusPlus>()?;
    module.add_class::<AnyInterpolator>()?;
    module.add("Var", module.py().get_type::<AadVar>())?;
    module.add_function(wrap_pyfunction!(mc_european_with_arena, module)?)?;
    module.add_function(wrap_pyfunction!(fill_normals, module)?)?;
    Ok(())
}
