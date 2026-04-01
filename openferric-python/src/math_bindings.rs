use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use openferric_core::math as core_math;
use openferric_core::math::Interpolator;
use openferric_core::math::fast_rng::{fill_standard_normals, sample_standard_normal};

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn interp_err(err: core_math::InterpolationError) -> PyErr {
    let message = match err {
        core_math::InterpolationError::InvalidInput(message) => message.to_string(),
        core_math::InterpolationError::ExtrapolationDisabled => {
            "extrapolation is disabled".to_string()
        }
        core_math::InterpolationError::SingularSystem => {
            "singular interpolation system".to_string()
        }
        core_math::InterpolationError::NonConvergence => {
            "interpolator fit did not converge".to_string()
        }
    };
    py_value_error(message)
}

fn rng_kind_from_str(value: &str) -> PyResult<core_math::FastRngKind> {
    match value.to_ascii_lowercase().as_str() {
        "xoshiro256plusplus" | "xoshiro256_plus_plus" | "xoshiro" => {
            Ok(core_math::FastRngKind::Xoshiro256PlusPlus)
        }
        "pcg64" | "pcg" => Ok(core_math::FastRngKind::Pcg64),
        "threadrng" | "thread_rng" | "thread" => Ok(core_math::FastRngKind::ThreadRng),
        "stdrng" | "std_rng" | "std" => Ok(core_math::FastRngKind::StdRng),
        _ => Err(py_value_error(format!("unsupported rng kind '{value}'"))),
    }
}

fn rng_kind_name(kind: core_math::FastRngKind) -> &'static str {
    match kind {
        core_math::FastRngKind::Xoshiro256PlusPlus => "xoshiro256plusplus",
        core_math::FastRngKind::Pcg64 => "pcg64",
        core_math::FastRngKind::ThreadRng => "thread_rng",
        core_math::FastRngKind::StdRng => "std_rng",
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct Dual {
    inner: core_math::Dual,
}

#[pymethods]
impl Dual {
    #[new]
    fn new(value: f64, derivative: f64) -> Self {
        Self {
            inner: core_math::Dual { value, derivative },
        }
    }

    #[staticmethod]
    fn variable(value: f64) -> Self {
        Self {
            inner: core_math::Dual::variable(value),
        }
    }

    #[staticmethod]
    fn constant(value: f64) -> Self {
        Self {
            inner: core_math::Dual::constant(value),
        }
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    fn derivative(&self) -> f64 {
        self.inner.derivative
    }

    fn add(&self, other: &Dual) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }

    fn sub(&self, other: &Dual) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }

    fn mul(&self, other: &Dual) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }

    fn div(&self, other: &Dual) -> Self {
        Self {
            inner: self.inner / other.inner,
        }
    }

    fn add_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner + rhs,
        }
    }

    fn sub_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner - rhs,
        }
    }

    fn mul_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner * rhs,
        }
    }

    fn div_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner / rhs,
        }
    }

    fn neg(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
        }
    }

    fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
        }
    }

    fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt(),
        }
    }

    fn normal_cdf(&self) -> Self {
        Self {
            inner: self.inner.normal_cdf(),
        }
    }

    fn positive_part(&self) -> Self {
        Self {
            inner: self.inner.positive_part(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Dual(value={}, derivative={})",
            self.inner.value, self.inner.derivative
        )
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct Dual2 {
    inner: core_math::Dual2,
}

#[pymethods]
impl Dual2 {
    #[new]
    fn new(value: f64, first: f64, second: f64) -> Self {
        Self {
            inner: core_math::Dual2 {
                value,
                first,
                second,
            },
        }
    }

    #[staticmethod]
    fn variable(value: f64) -> Self {
        Self {
            inner: core_math::Dual2::variable(value),
        }
    }

    #[staticmethod]
    fn constant(value: f64) -> Self {
        Self {
            inner: core_math::Dual2::constant(value),
        }
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    fn first(&self) -> f64 {
        self.inner.first
    }

    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    fn add(&self, other: &Dual2) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }

    fn sub(&self, other: &Dual2) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }

    fn mul(&self, other: &Dual2) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }

    fn div(&self, other: &Dual2) -> Self {
        Self {
            inner: self.inner / other.inner,
        }
    }

    fn add_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner + rhs,
        }
    }

    fn sub_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner - rhs,
        }
    }

    fn mul_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner * rhs,
        }
    }

    fn div_scalar(&self, rhs: f64) -> Self {
        Self {
            inner: self.inner / rhs,
        }
    }

    fn neg(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
        }
    }

    fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
        }
    }

    fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt(),
        }
    }

    fn normal_cdf(&self) -> Self {
        Self {
            inner: self.inner.normal_cdf(),
        }
    }

    fn positive_part(&self) -> Self {
        Self {
            inner: self.inner.positive_part(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Dual2(value={}, first={}, second={})",
            self.inner.value, self.inner.first, self.inner.second
        )
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct AadVar {
    inner: core_math::Var,
}

#[pymethods]
impl AadVar {
    fn __repr__(&self) -> String {
        format!("AadVar({:?})", self.inner)
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct TapeCheckpoint {
    inner: core_math::TapeCheckpoint,
}

#[pymethods]
impl TapeCheckpoint {
    fn __repr__(&self) -> String {
        format!("TapeCheckpoint({:?})", self.inner)
    }
}

#[pyclass(module = "openferric")]
pub struct AadTape {
    inner: core_math::AadTape,
}

#[pymethods]
impl AadTape {
    #[new]
    fn new(nodes: Option<usize>) -> Self {
        Self {
            inner: core_math::AadTape::with_capacity(nodes.unwrap_or_default()),
        }
    }

    #[staticmethod]
    fn with_capacity(nodes: usize) -> Self {
        Self {
            inner: core_math::AadTape::with_capacity(nodes),
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn checkpoint(&self) -> TapeCheckpoint {
        TapeCheckpoint {
            inner: self.inner.checkpoint(),
        }
    }

    fn rewind(&mut self, checkpoint: &TapeCheckpoint) {
        self.inner.rewind(checkpoint.inner);
    }

    fn variable(&mut self, value: f64) -> AadVar {
        AadVar {
            inner: self.inner.variable(value),
        }
    }

    fn constant(&mut self, value: f64) -> AadVar {
        AadVar {
            inner: self.inner.constant(value),
        }
    }

    fn value(&self, var: &AadVar) -> f64 {
        self.inner.value(var.inner)
    }

    fn add(&mut self, a: &AadVar, b: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.add(a.inner, b.inner),
        }
    }

    fn sub(&mut self, a: &AadVar, b: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.sub(a.inner, b.inner),
        }
    }

    fn mul(&mut self, a: &AadVar, b: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.mul(a.inner, b.inner),
        }
    }

    fn div(&mut self, a: &AadVar, b: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.div(a.inner, b.inner),
        }
    }

    fn neg(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.neg(a.inner),
        }
    }

    fn exp(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.exp(a.inner),
        }
    }

    fn ln(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.ln(a.inner),
        }
    }

    fn sqrt(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.sqrt(a.inner),
        }
    }

    fn normal_cdf(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.normal_cdf(a.inner),
        }
    }

    fn positive_part(&mut self, a: &AadVar) -> AadVar {
        AadVar {
            inner: self.inner.positive_part(a.inner),
        }
    }

    fn reverse(&mut self, output: &AadVar) {
        self.inner.reverse(output.inner);
    }

    fn adjoint(&self, var: &AadVar) -> f64 {
        self.inner.adjoint(var.inner)
    }

    fn gradient(&mut self, output: &AadVar, inputs: Vec<PyRef<'_, AadVar>>) -> Vec<f64> {
        let vars = inputs.iter().map(|item| item.inner).collect::<Vec<_>>();
        self.inner.gradient_vec(output.inner, &vars)
    }

    fn __repr__(&self) -> String {
        format!("AadTape(len={})", self.inner.len())
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_black_scholes_price_greeks_aad(
    option_type: &str,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let option_type = crate::helpers::parse_option_type(option_type)
        .ok_or_else(|| py_value_error("option_type must be 'call' or 'put'"))?;
    let (price, greeks) = core_math::black_scholes_price_greeks_aad(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        vol,
        expiry,
    );
    Ok((
        price,
        greeks.delta,
        greeks.gamma,
        greeks.vega,
        greeks.theta,
        greeks.rho,
    ))
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct ExtrapolationMode {
    inner: core_math::ExtrapolationMode,
}

#[pymethods]
impl ExtrapolationMode {
    #[staticmethod]
    fn flat() -> Self {
        Self {
            inner: core_math::ExtrapolationMode::Flat,
        }
    }

    #[staticmethod]
    fn linear() -> Self {
        Self {
            inner: core_math::ExtrapolationMode::Linear,
        }
    }

    #[staticmethod]
    fn error() -> Self {
        Self {
            inner: core_math::ExtrapolationMode::Error,
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core_math::ExtrapolationMode::Flat => "flat",
            core_math::ExtrapolationMode::Linear => "linear",
            core_math::ExtrapolationMode::Error => "error",
        }
    }

    fn __repr__(&self) -> String {
        format!("ExtrapolationMode.{}", self.kind())
    }
}

#[pyclass(module = "openferric")]
pub struct LinearInterpolator {
    inner: core_math::LinearInterpolator,
}

#[pymethods]
impl LinearInterpolator {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::LinearInterpolator::new(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct LogLinearInterpolator {
    inner: core_math::LogLinearInterpolator,
}

#[pymethods]
impl LogLinearInterpolator {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::LogLinearInterpolator::new(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct MonotoneConvexInterpolator {
    inner: core_math::MonotoneConvexInterpolator,
}

#[pymethods]
impl MonotoneConvexInterpolator {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::MonotoneConvexInterpolator::new(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct TensionSplineInterpolator {
    inner: core_math::TensionSplineInterpolator,
}

#[pymethods]
impl TensionSplineInterpolator {
    #[new]
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        tension: f64,
        extrapolation: &ExtrapolationMode,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::TensionSplineInterpolator::new(x, y, tension, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct HermiteMonotoneInterpolator {
    inner: core_math::HermiteMonotoneInterpolator,
}

#[pymethods]
impl HermiteMonotoneInterpolator {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::HermiteMonotoneInterpolator::new(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct LogCubicMonotoneInterpolator {
    inner: core_math::LogCubicMonotoneInterpolator,
}

#[pymethods]
impl LogCubicMonotoneInterpolator {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::LogCubicMonotoneInterpolator::new(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct NelsonSiegelInterpolator {
    inner: core_math::NelsonSiegelInterpolator,
}

#[pymethods]
impl NelsonSiegelInterpolator {
    #[staticmethod]
    fn fit(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::NelsonSiegelInterpolator::fit(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_params(
        x: Vec<f64>,
        y: Vec<f64>,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        tau: f64,
        extrapolation: &ExtrapolationMode,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::NelsonSiegelInterpolator::from_params(
                x,
                y,
                beta0,
                beta1,
                beta2,
                tau,
                extrapolation.inner,
            )
            .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct NelsonSiegelSvenssonInterpolator {
    inner: core_math::NelsonSiegelSvenssonInterpolator,
}

#[pymethods]
impl NelsonSiegelSvenssonInterpolator {
    #[staticmethod]
    fn fit(x: Vec<f64>, y: Vec<f64>, extrapolation: &ExtrapolationMode) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::NelsonSiegelSvenssonInterpolator::fit(x, y, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
pub struct SmithWilsonInterpolator {
    inner: core_math::SmithWilsonInterpolator,
}

#[pymethods]
impl SmithWilsonInterpolator {
    #[new]
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        ufr: f64,
        alpha: f64,
        extrapolation: &ExtrapolationMode,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core_math::SmithWilsonInterpolator::new(x, y, ufr, alpha, extrapolation.inner)
                .map_err(interp_err)?,
        })
    }

    fn value(&self, x: f64) -> PyResult<f64> {
        self.inner.value(x).map_err(interp_err)
    }

    fn derivative(&self, x: f64) -> PyResult<f64> {
        self.inner.derivative(x).map_err(interp_err)
    }

    fn jacobian(&self, x: f64) -> PyResult<Vec<f64>> {
        self.inner.jacobian(x).map_err(interp_err)
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn y(&self) -> Vec<f64> {
        self.inner.y().to_vec()
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct PsdProjectionConfig {
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub max_iterations: usize,
}

impl From<PsdProjectionConfig> for core_math::PsdProjectionConfig {
    fn from(value: PsdProjectionConfig) -> Self {
        Self {
            tol: value.tol,
            max_iterations: value.max_iterations,
        }
    }
}

#[pymethods]
impl PsdProjectionConfig {
    #[new]
    fn new(tol: Option<f64>, max_iterations: Option<usize>) -> Self {
        let default = core_math::PsdProjectionConfig::default();
        Self {
            tol: tol.unwrap_or(default.tol),
            max_iterations: max_iterations.unwrap_or(default.max_iterations),
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self::new(None, None)
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct CopulaFamily {
    inner: core_math::CopulaFamily,
}

#[pymethods]
impl CopulaFamily {
    #[staticmethod]
    fn gaussian() -> Self {
        Self {
            inner: core_math::CopulaFamily::Gaussian,
        }
    }

    #[staticmethod]
    fn student_t(degrees_of_freedom: u32) -> Self {
        Self {
            inner: core_math::CopulaFamily::StudentT { degrees_of_freedom },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core_math::CopulaFamily::Gaussian => "gaussian",
            core_math::CopulaFamily::StudentT { .. } => "student_t",
        }
    }

    #[getter]
    fn degrees_of_freedom(&self) -> Option<u32> {
        match self.inner {
            core_math::CopulaFamily::Gaussian => None,
            core_math::CopulaFamily::StudentT { degrees_of_freedom } => Some(degrees_of_freedom),
        }
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct CorrelationStressScenario {
    inner: core_math::CorrelationStressScenario,
}

#[pymethods]
impl CorrelationStressScenario {
    #[staticmethod]
    fn scale_off_diagonal(factor: f64) -> Self {
        Self {
            inner: core_math::CorrelationStressScenario::ScaleOffDiagonal { factor },
        }
    }

    #[staticmethod]
    fn additive_shift(shift: f64) -> Self {
        Self {
            inner: core_math::CorrelationStressScenario::AdditiveShift { shift },
        }
    }

    #[staticmethod]
    fn floor_off_diagonal(floor: f64) -> Self {
        Self {
            inner: core_math::CorrelationStressScenario::FloorOffDiagonal { floor },
        }
    }

    #[staticmethod]
    fn cap_off_diagonal(cap: f64) -> Self {
        Self {
            inner: core_math::CorrelationStressScenario::CapOffDiagonal { cap },
        }
    }

    #[staticmethod]
    fn override_pair(i: usize, j: usize, value: f64) -> Self {
        Self {
            inner: core_math::CorrelationStressScenario::OverridePair { i, j, value },
        }
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct FactorCorrelationModel {
    inner: core_math::FactorCorrelationModel,
}

#[pymethods]
impl FactorCorrelationModel {
    #[staticmethod]
    fn one_factor(loadings: Vec<f64>) -> Self {
        Self {
            inner: core_math::FactorCorrelationModel::OneFactor { loadings },
        }
    }

    #[staticmethod]
    fn multi_factor(loadings: Vec<Vec<f64>>) -> Self {
        Self {
            inner: core_math::FactorCorrelationModel::MultiFactor { loadings },
        }
    }

    fn n_assets(&self) -> usize {
        self.inner.n_assets()
    }

    fn n_factors(&self) -> usize {
        self.inner.n_factors()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(py_value_error)
    }

    fn correlation_matrix(&self) -> PyResult<Vec<Vec<f64>>> {
        self.inner.correlation_matrix().map_err(py_value_error)
    }

    fn sample_correlated_normals(&self, rng: &mut FastRng) -> PyResult<Vec<f64>> {
        let mut out = vec![0.0; self.inner.n_assets()];
        self.inner
            .sample_correlated_normals(&mut rng.inner, &mut out)
            .map_err(py_value_error)?;
        Ok(out)
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone, Copy)]
pub struct FastRngKind {
    inner: core_math::FastRngKind,
}

#[pymethods]
impl FastRngKind {
    #[staticmethod]
    fn xoshiro256plusplus() -> Self {
        Self {
            inner: core_math::FastRngKind::Xoshiro256PlusPlus,
        }
    }

    #[staticmethod]
    fn pcg64() -> Self {
        Self {
            inner: core_math::FastRngKind::Pcg64,
        }
    }

    #[staticmethod]
    fn thread_rng() -> Self {
        Self {
            inner: core_math::FastRngKind::ThreadRng,
        }
    }

    #[staticmethod]
    fn std_rng() -> Self {
        Self {
            inner: core_math::FastRngKind::StdRng,
        }
    }

    #[staticmethod]
    fn from_str(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: rng_kind_from_str(value)?,
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        rng_kind_name(self.inner)
    }
}

#[pyclass(module = "openferric", unsendable)]
pub struct FastRng {
    pub(crate) inner: core_math::FastRng,
    kind: core_math::FastRngKind,
}

#[pymethods]
impl FastRng {
    #[new]
    fn new(kind: Option<&str>, seed: Option<u64>) -> PyResult<Self> {
        let kind = rng_kind_from_str(kind.unwrap_or("xoshiro256plusplus"))?;
        let seed = seed.unwrap_or_default();
        Ok(Self {
            inner: core_math::FastRng::from_seed(kind, seed),
            kind,
        })
    }

    #[staticmethod]
    fn from_kind(kind: &FastRngKind, seed: u64) -> Self {
        Self {
            inner: core_math::FastRng::from_seed(kind.inner, seed),
            kind: kind.inner,
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        rng_kind_name(self.kind)
    }

    fn random_f64(&mut self) -> f64 {
        self.inner.random_f64()
    }

    fn random_u64(&mut self) -> u64 {
        self.inner.random_u64()
    }

    fn sample_standard_normal(&mut self) -> f64 {
        sample_standard_normal(&mut self.inner)
    }

    fn fill_standard_normals(&mut self, count: usize) -> Vec<f64> {
        let mut out = vec![0.0; count];
        fill_standard_normals(&mut self.inner, &mut out);
        out
    }
}

#[pyfunction]
pub fn py_validate_correlation_matrix(corr_matrix: Vec<Vec<f64>>, n_assets: usize) -> PyResult<()> {
    core_math::validate_correlation_matrix(&corr_matrix, n_assets).map_err(py_value_error)
}

#[pyfunction]
pub fn py_min_eigenvalue_symmetric(matrix: Vec<Vec<f64>>) -> Option<f64> {
    core_math::min_eigenvalue_symmetric(&matrix)
}

#[pyfunction]
pub fn py_is_positive_semidefinite(matrix: Vec<Vec<f64>>, tol: f64) -> bool {
    core_math::is_positive_semidefinite(&matrix, tol)
}

#[pyfunction]
pub fn py_nearest_correlation_matrix_higham(
    matrix: Vec<Vec<f64>>,
    cfg: &PsdProjectionConfig,
) -> PyResult<Vec<Vec<f64>>> {
    core_math::nearest_correlation_matrix_higham(&matrix, (*cfg).to_owned().into())
        .map_err(py_value_error)
}

#[pyfunction]
pub fn py_validate_or_repair_correlation_matrix(
    corr_matrix: Vec<Vec<f64>>,
    n_assets: usize,
    cfg: &PsdProjectionConfig,
) -> PyResult<(Vec<Vec<f64>>, bool)> {
    core_math::validate_or_repair_correlation_matrix(
        &corr_matrix,
        n_assets,
        (*cfg).to_owned().into(),
    )
    .map_err(py_value_error)
}

#[pyfunction]
pub fn py_cholesky_lower_psd(matrix: Vec<Vec<f64>>, tol: f64) -> Option<Vec<Vec<f64>>> {
    core_math::cholesky_lower_psd(&matrix, tol)
}

#[pyfunction]
pub fn py_correlate_normals(chol: Vec<Vec<f64>>, indep: Vec<f64>) -> Vec<f64> {
    let mut out = vec![0.0; chol.len()];
    core_math::correlate_normals(&chol, &indep, &mut out);
    out
}

#[pyfunction]
pub fn py_sample_copula_uniforms_from_cholesky(
    chol: Vec<Vec<f64>>,
    copula: &CopulaFamily,
    rng: &mut FastRng,
) -> PyResult<Vec<f64>> {
    let mut out = vec![0.0; chol.len()];
    core_math::sample_copula_uniforms_from_cholesky(&chol, copula.inner, &mut rng.inner, &mut out)
        .map_err(py_value_error)?;
    Ok(out)
}

#[pyfunction]
pub fn py_sample_copula_uniforms_from_factor_model(
    model: &FactorCorrelationModel,
    copula: &CopulaFamily,
    rng: &mut FastRng,
) -> PyResult<Vec<f64>> {
    let mut out = vec![0.0; model.inner.n_assets()];
    core_math::sample_copula_uniforms_from_factor_model(
        &model.inner,
        copula.inner,
        &mut rng.inner,
        &mut out,
    )
    .map_err(py_value_error)?;
    Ok(out)
}

#[pyfunction]
pub fn py_apply_correlation_stress(
    base: Vec<Vec<f64>>,
    scenarios: Vec<PyRef<'_, CorrelationStressScenario>>,
    repair_to_psd: bool,
    cfg: &PsdProjectionConfig,
) -> PyResult<Vec<Vec<f64>>> {
    let scenarios = scenarios
        .iter()
        .map(|item| item.inner.clone())
        .collect::<Vec<_>>();
    core_math::apply_correlation_stress(&base, &scenarios, repair_to_psd, (*cfg).to_owned().into())
        .map_err(py_value_error)
}

#[pyfunction]
pub fn py_copula_uniforms_to_normals(uniforms: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut out = vec![0.0; uniforms.len()];
    core_math::copula_uniforms_to_normals(&uniforms, &mut out).map_err(py_value_error)?;
    Ok(out)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_black_scholes_price_greeks_aad, module)?)?;
    module.add_function(wrap_pyfunction!(py_validate_correlation_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(py_min_eigenvalue_symmetric, module)?)?;
    module.add_function(wrap_pyfunction!(py_is_positive_semidefinite, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_nearest_correlation_matrix_higham,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_validate_or_repair_correlation_matrix,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_cholesky_lower_psd, module)?)?;
    module.add_function(wrap_pyfunction!(py_correlate_normals, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_sample_copula_uniforms_from_cholesky,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_sample_copula_uniforms_from_factor_model,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_apply_correlation_stress, module)?)?;
    module.add_function(wrap_pyfunction!(py_copula_uniforms_to_normals, module)?)?;

    module.add_class::<Dual>()?;
    module.add_class::<Dual2>()?;
    module.add_class::<AadVar>()?;
    module.add_class::<TapeCheckpoint>()?;
    module.add_class::<AadTape>()?;
    module.add_class::<ExtrapolationMode>()?;
    module.add_class::<LinearInterpolator>()?;
    module.add_class::<LogLinearInterpolator>()?;
    module.add_class::<MonotoneConvexInterpolator>()?;
    module.add_class::<TensionSplineInterpolator>()?;
    module.add_class::<HermiteMonotoneInterpolator>()?;
    module.add_class::<LogCubicMonotoneInterpolator>()?;
    module.add_class::<NelsonSiegelInterpolator>()?;
    module.add_class::<NelsonSiegelSvenssonInterpolator>()?;
    module.add_class::<SmithWilsonInterpolator>()?;
    module.add_class::<PsdProjectionConfig>()?;
    module.add_class::<CopulaFamily>()?;
    module.add_class::<CorrelationStressScenario>()?;
    module.add_class::<FactorCorrelationModel>()?;
    module.add_class::<FastRngKind>()?;
    module.add_class::<FastRng>()?;
    Ok(())
}
