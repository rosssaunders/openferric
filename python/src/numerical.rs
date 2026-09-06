use std::cell::RefCell;

use openferric_core::math::{self as native, Interpolator};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::helpers::catch_unwind_py;

fn math_error(error: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{error:?}"))
}

macro_rules! scalar_functions {
    ($($name:ident($($argument:ident: $argument_type:ty),*) -> $output:ty => $body:expr);* $(;)?) => {
        $(
            #[pyfunction]
            fn $name($($argument: $argument_type),*) -> $output {
                $body
            }
        )*

        fn register_scalars(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_function(wrap_pyfunction!($name, module)?)?;)*
            Ok(())
        }
    };
}

scalar_functions! {
    normal_pdf(value: f64) -> f64 => native::normal_pdf(value);
    normal_cdf(value: f64) -> f64 => native::normal_cdf(value);
    branch_free_normal_cdf(value: f64) -> f64 => native::branch_free_normal_cdf(value);
    normal_inv_cdf(probability: f64) -> f64 => native::normal_inv_cdf(probability);
    bivariate_normal_cdf(first: f64, second: f64, correlation: f64) -> f64 => native::bivariate_normal_cdf(first, second, correlation);
    accurate_norm_cdf(value: f64) -> f64 => native::fast_norm::accurate_norm_cdf(value);
    hart_norm_cdf(value: f64) -> f64 => native::fast_norm::hart_norm_cdf(value);
    fast_norm_cdf(value: f64) -> f64 => native::fast_norm::fast_norm_cdf(value);
    fast_norm_inv_cdf(probability: f64) -> f64 => native::fast_norm::fast_norm_inv_cdf(probability);
    beasley_springer_moro_inv_cdf(probability: f64) -> f64 => native::fast_norm::beasley_springer_moro_inv_cdf(probability);
    fast_norm_pdf(value: f64) -> f64 => native::fast_norm::fast_norm_pdf(value);
    erfc_cody(value: f64) -> f64 => native::fast_norm::erfc_cody(value);
    gamma(value: f64) -> f64 => native::gamma::gamma(value);
    stream_seed(base_seed: u64, stream_index: usize) -> u64 => native::fast_rng::stream_seed(base_seed, stream_index);
    resolve_stream_seed(base_seed: u64, stream_index: usize, reproducible: bool) -> u64 => native::fast_rng::resolve_stream_seed(base_seed, stream_index, reproducible);
    uniform_open01(value: f64) -> f64 => native::fast_rng::uniform_open01(value);
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AccuracyTier {
    High,
    Fast,
}

impl AccuracyTier {
    pub(crate) fn to_core(self) -> native::AccuracyTier {
        match self {
            Self::High => native::AccuracyTier::High,
            Self::Fast => native::AccuracyTier::Fast,
        }
    }
}

#[pymethods]
impl AccuracyTier {
    #[staticmethod]
    fn for_mc(num_paths: usize, num_steps: usize) -> Self {
        match native::AccuracyTier::for_mc(num_paths, num_steps) {
            native::AccuracyTier::High => Self::High,
            native::AccuracyTier::Fast => Self::Fast,
        }
    }

    #[staticmethod]
    fn for_analytic() -> Self {
        match native::AccuracyTier::for_analytic() {
            native::AccuracyTier::High => Self::High,
            native::AccuracyTier::Fast => Self::Fast,
        }
    }

    fn uses_fast_exp(&self) -> bool {
        self.to_core().uses_fast_exp()
    }
}

#[pyfunction]
#[pyo3(signature = (value, tier=AccuracyTier::High))]
fn tiered_exp(value: f64, tier: AccuracyTier) -> f64 {
    native::approx_tier::tiered_exp(value, tier.to_core())
}

#[pyfunction]
fn gauss_legendre_nodes_weights(order: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    native::gauss_legendre_nodes_weights(order).map_err(math_error)
}

fn scalar_callback(
    function: &Bound<'_, PyAny>,
    argument: f64,
    failure: &RefCell<Option<PyErr>>,
) -> f64 {
    if failure.borrow().is_some() {
        return f64::NAN;
    }
    match function
        .call1((argument,))
        .and_then(|value| value.extract::<f64>())
    {
        Ok(value) if value.is_finite() => value,
        Ok(_) => {
            *failure.borrow_mut() = Some(PyValueError::new_err(
                "callback returned a non-finite value",
            ));
            f64::NAN
        }
        Err(error) => {
            *failure.borrow_mut() = Some(error);
            f64::NAN
        }
    }
}

#[pyfunction]
#[pyo3(signature = (function, derivative, initial, tolerance=1e-12, max_iterations=100))]
fn newton_raphson(
    function: &Bound<'_, PyAny>,
    derivative: &Bound<'_, PyAny>,
    initial: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    if !initial.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PyValueError::new_err(
            "initial must be finite and tolerance must be positive and finite",
        ));
    }
    let failure = RefCell::new(None);
    let result = native::newton_raphson(
        |argument| scalar_callback(function, argument, &failure),
        |argument| scalar_callback(derivative, argument, &failure),
        initial,
        tolerance,
        max_iterations,
    );
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    result.map_err(math_error)
}

#[pyfunction]
fn gauss_legendre_integrate(
    function: &Bound<'_, PyAny>,
    lower: f64,
    upper: f64,
    order: usize,
) -> PyResult<f64> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(PyValueError::new_err("integration bounds must be finite"));
    }
    let failure = RefCell::new(None);
    let result = native::gauss_legendre_integrate(
        |argument| scalar_callback(function, argument, &failure),
        lower,
        upper,
        order,
    );
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    result.map_err(math_error)
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CubicSpline {
    inner: native::CubicSpline,
}

#[pymethods]
impl CubicSpline {
    #[new]
    fn new(nodes: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: native::CubicSpline::new(nodes, values).map_err(math_error)?,
        })
    }

    fn interpolate(&self, value: f64) -> f64 {
        self.inner.interpolate(value)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PiecewiseConstantInterpolator {
    inner: native::PiecewiseConstantInterpolator,
}

#[pymethods]
impl PiecewiseConstantInterpolator {
    #[new]
    #[pyo3(signature = (nodes, values, extrapolation="flat"))]
    fn new(nodes: Vec<f64>, values: Vec<f64>, extrapolation: &str) -> PyResult<Self> {
        let mode = match extrapolation.to_ascii_lowercase().as_str() {
            "flat" => native::ExtrapolationMode::Flat,
            "linear" => native::ExtrapolationMode::Linear,
            "none" | "disabled" | "error" => native::ExtrapolationMode::Error,
            _ => {
                return Err(PyValueError::new_err(
                    "extrapolation must be flat, linear, or none",
                ));
            }
        };
        Ok(Self {
            inner: native::PiecewiseConstantInterpolator::new(nodes, values, mode)
                .map_err(math_error)?,
        })
    }

    fn value(&self, value: f64) -> PyResult<f64> {
        self.inner.value(value).map_err(math_error)
    }

    fn derivative(&self, value: f64) -> PyResult<f64> {
        self.inner.derivative(value).map_err(math_error)
    }

    fn jacobian(&self, value: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(value).map_err(math_error)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct SobolSequence {
    inner: native::SobolSequence,
}

#[pymethods]
impl SobolSequence {
    #[new]
    #[pyo3(signature = (dimensions, seed=0))]
    fn new(dimensions: usize, seed: u64) -> PyResult<Self> {
        catch_unwind_py(|| Self {
            inner: native::SobolSequence::new(dimensions, seed),
        })
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn next_into(&mut self, point: &Bound<'_, numpy::PyArray1<f64>>) -> PyResult<bool> {
        use numpy::PyArrayMethods;
        let mut point = point
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let point = point
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("point must be contiguous float64"))?;
        if point.len() != self.inner.dimensions() {
            return Err(PyValueError::new_err(
                "point length must equal Sobol dimensions",
            ));
        }
        Ok(self.inner.next_into(point))
    }

    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__(&mut self) -> Option<Vec<f64>> {
        self.inner.next()
    }

    fn fill_points(&mut self, py: Python<'_>, count: usize) -> PyResult<Vec<Vec<f64>>> {
        let dimensions = self.inner.dimensions();
        let size = count
            .checked_mul(dimensions)
            .ok_or_else(|| PyMemoryError::new_err("point count is too large"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(size)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
        values.resize(size, 0.0);
        let generated = py.detach(|| self.inner.fill_points(&mut values, count));
        Ok(values
            .chunks_exact(dimensions)
            .take(generated)
            .map(<[f64]>::to_vec)
            .collect())
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_scalars(module)?;
    module.add_class::<AccuracyTier>()?;
    module.add_class::<CubicSpline>()?;
    module.add_class::<PiecewiseConstantInterpolator>()?;
    module.add_class::<SobolSequence>()?;
    module.add("SOBOL_MAX_DIMENSIONS", native::sobol::SOBOL_MAX_DIMENSIONS)?;
    module.add_function(wrap_pyfunction!(tiered_exp, module)?)?;
    module.add_function(wrap_pyfunction!(gauss_legendre_nodes_weights, module)?)?;
    module.add_function(wrap_pyfunction!(gauss_legendre_integrate, module)?)?;
    module.add_function(wrap_pyfunction!(newton_raphson, module)?)?;
    Ok(())
}
