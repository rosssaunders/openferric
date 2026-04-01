use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use openferric_core::math::FastRngKind;
use openferric_core::mc as core_mc;
use openferric_core::models::{Gbm, Heston};

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn rng_kind_from_str(value: &str) -> PyResult<FastRngKind> {
    match value.to_ascii_lowercase().as_str() {
        "xoshiro256plusplus" | "xoshiro256_plus_plus" | "xoshiro" => {
            Ok(FastRngKind::Xoshiro256PlusPlus)
        }
        "pcg64" | "pcg" => Ok(FastRngKind::Pcg64),
        "threadrng" | "thread_rng" | "thread" => Ok(FastRngKind::ThreadRng),
        "stdrng" | "std_rng" | "std" => Ok(FastRngKind::StdRng),
        _ => Err(py_value_error(format!("unsupported rng kind '{value}'"))),
    }
}

fn rng_kind_name(kind: FastRngKind) -> &'static str {
    match kind {
        FastRngKind::Xoshiro256PlusPlus => "xoshiro256plusplus",
        FastRngKind::Pcg64 => "pcg64",
        FastRngKind::ThreadRng => "thread_rng",
        FastRngKind::StdRng => "std_rng",
    }
}

fn python_path_evaluator(
    callable: Py<PyAny>,
) -> (core_mc::PathEvaluator, Arc<Mutex<Option<String>>>) {
    let error = Arc::new(Mutex::new(None::<String>));
    let error_slot = Arc::clone(&error);

    let evaluator = Arc::new(move |path: &[f64]| -> f64 {
        Python::attach(|py| {
            let bound = callable.bind(py);
            match bound.call1((path.to_vec(),)) {
                Ok(result) => match result.extract::<f64>() {
                    Ok(value) => value,
                    Err(err) => {
                        if let Ok(mut slot) = error_slot.lock()
                            && slot.is_none()
                        {
                            *slot = Some(err.to_string());
                        }
                        f64::NAN
                    }
                },
                Err(err) => {
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err.to_string());
                    }
                    f64::NAN
                }
            }
        })
    });

    (evaluator, error)
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct GbmPathGenerator {
    inner: core_mc::GbmPathGenerator,
}

#[pymethods]
impl GbmPathGenerator {
    #[new]
    fn new(mu: f64, sigma: f64, s0: f64, maturity: f64, steps: usize) -> Self {
        Self {
            inner: core_mc::GbmPathGenerator {
                model: Gbm { mu, sigma },
                s0,
                maturity,
                steps,
            },
        }
    }

    #[getter]
    fn mu(&self) -> f64 {
        self.inner.model.mu
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.model.sigma
    }

    #[getter]
    fn s0(&self) -> f64 {
        self.inner.s0
    }

    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }

    #[getter]
    fn steps(&self) -> usize {
        core_mc::PathGenerator::steps(&self.inner)
    }

    fn num_normal_streams(&self) -> usize {
        core_mc::PathGenerator::num_normal_streams(&self.inner)
    }

    fn generate_from_normals(&self, normals_1: Vec<f64>, normals_2: Option<Vec<f64>>) -> Vec<f64> {
        let normals_2 = normals_2.unwrap_or_default();
        core_mc::PathGenerator::generate_from_normals(&self.inner, &normals_1, &normals_2)
    }

    fn __repr__(&self) -> String {
        format!(
            "GbmPathGenerator(mu={}, sigma={}, s0={}, maturity={}, steps={})",
            self.inner.model.mu,
            self.inner.model.sigma,
            self.inner.s0,
            self.inner.maturity,
            self.inner.steps
        )
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct HestonPathGenerator {
    inner: core_mc::HestonPathGenerator,
}

#[pymethods]
impl HestonPathGenerator {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        v0: f64,
        s0: f64,
        maturity: f64,
        steps: usize,
    ) -> Self {
        Self {
            inner: core_mc::HestonPathGenerator {
                model: Heston {
                    mu,
                    kappa,
                    theta,
                    xi,
                    rho,
                    v0,
                },
                s0,
                maturity,
                steps,
            },
        }
    }

    #[getter]
    fn s0(&self) -> f64 {
        self.inner.s0
    }

    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }

    #[getter]
    fn steps(&self) -> usize {
        core_mc::PathGenerator::steps(&self.inner)
    }

    fn validate_model(&self) -> bool {
        self.inner.model.validate()
    }

    fn num_normal_streams(&self) -> usize {
        core_mc::PathGenerator::num_normal_streams(&self.inner)
    }

    fn generate_from_normals(&self, normals_1: Vec<f64>, normals_2: Vec<f64>) -> Vec<f64> {
        core_mc::PathGenerator::generate_from_normals(&self.inner, &normals_1, &normals_2)
    }

    fn __repr__(&self) -> String {
        format!(
            "HestonPathGenerator(mu={}, kappa={}, theta={}, xi={}, rho={}, v0={}, s0={}, maturity={}, steps={})",
            self.inner.model.mu,
            self.inner.model.kappa,
            self.inner.model.theta,
            self.inner.model.xi,
            self.inner.model.rho,
            self.inner.model.v0,
            self.inner.s0,
            self.inner.maturity,
            self.inner.steps
        )
    }
}

#[pyclass(module = "openferric")]
pub struct ControlVariate {
    expected: f64,
    evaluator: Py<PyAny>,
}

impl Clone for ControlVariate {
    fn clone(&self) -> Self {
        let evaluator = Python::attach(|py| self.evaluator.clone_ref(py));
        Self {
            expected: self.expected,
            evaluator,
        }
    }
}

#[pymethods]
impl ControlVariate {
    #[new]
    fn new(py: Python<'_>, expected: f64, evaluator: Py<PyAny>) -> PyResult<Self> {
        if !evaluator.bind(py).is_callable() {
            return Err(py_value_error("control variate evaluator must be callable"));
        }
        Ok(Self {
            expected,
            evaluator,
        })
    }

    #[getter]
    fn expected(&self) -> f64 {
        self.expected
    }

    fn evaluate(&self, py: Python<'_>, path: Vec<f64>) -> PyResult<f64> {
        self.evaluator.bind(py).call1((path,))?.extract()
    }

    fn __repr__(&self) -> String {
        format!("ControlVariate(expected={})", self.expected)
    }
}

#[pyclass(module = "openferric")]
#[derive(Clone)]
pub struct MonteCarloEngine {
    num_paths: usize,
    antithetic: bool,
    control_variate: Option<ControlVariate>,
    seed: u64,
    rng_kind: FastRngKind,
    reproducible: bool,
}

impl MonteCarloEngine {
    fn to_core(
        &self,
    ) -> (
        core_mc::MonteCarloEngine,
        Option<Arc<Mutex<Option<String>>>>,
    ) {
        let mut inner = core_mc::MonteCarloEngine::new(self.num_paths, self.seed)
            .with_antithetic(self.antithetic)
            .with_rng_kind(self.rng_kind);

        if !self.reproducible {
            inner = inner.with_randomized_streams();
        }

        let mut control_error = None;
        if let Some(control) = &self.control_variate {
            let callable = Python::attach(|py| control.evaluator.clone_ref(py));
            let (evaluator, error) = python_path_evaluator(callable);
            inner = inner.with_control_variate(core_mc::ControlVariate {
                expected: control.expected,
                evaluator,
            });
            control_error = Some(error);
        }

        (inner, control_error)
    }
}

#[pymethods]
impl MonteCarloEngine {
    #[new]
    fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            num_paths,
            antithetic: false,
            control_variate: None,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
        }
    }

    #[getter]
    fn num_paths(&self) -> usize {
        self.num_paths
    }

    #[getter]
    fn antithetic(&self) -> bool {
        self.antithetic
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn rng_kind(&self) -> &'static str {
        rng_kind_name(self.rng_kind)
    }

    #[getter]
    fn reproducible(&self) -> bool {
        self.reproducible
    }

    fn with_antithetic(&self, antithetic: bool) -> Self {
        let mut next = self.clone();
        next.antithetic = antithetic;
        next
    }

    fn with_control_variate(&self, control_variate: &ControlVariate) -> Self {
        let mut next = self.clone();
        next.control_variate = Some(control_variate.clone());
        next
    }

    fn with_rng_kind(&self, rng_kind: &str) -> PyResult<Self> {
        let mut next = self.clone();
        next.rng_kind = rng_kind_from_str(rng_kind)?;
        if matches!(next.rng_kind, FastRngKind::ThreadRng) {
            next.reproducible = false;
        }
        Ok(next)
    }

    fn with_seed(&self, seed: u64) -> Self {
        let mut next = self.clone();
        next.seed = seed;
        next.reproducible = true;
        next
    }

    fn with_randomized_streams(&self) -> Self {
        let mut next = self.clone();
        next.reproducible = false;
        next
    }

    fn with_thread_rng(&self) -> Self {
        let mut next = self.clone();
        next.rng_kind = FastRngKind::ThreadRng;
        next.reproducible = false;
        next
    }

    fn run_gbm(
        &self,
        py: Python<'_>,
        generator: &GbmPathGenerator,
        payoff: Py<PyAny>,
        discount_factor: f64,
    ) -> PyResult<(f64, f64)> {
        if !payoff.bind(py).is_callable() {
            return Err(py_value_error("payoff must be callable"));
        }
        let (core_engine, control_error) = self.to_core();
        let (payoff_fn, payoff_error) = python_path_evaluator(payoff);
        let result = core_engine.run(
            &generator.inner,
            move |path| payoff_fn(path),
            discount_factor,
        );

        if let Some(message) = payoff_error.lock().ok().and_then(|slot| slot.clone()) {
            return Err(py_value_error(message));
        }
        if let Some(message) =
            control_error.and_then(|slot| slot.lock().ok().and_then(|guard| guard.clone()))
        {
            return Err(py_value_error(message));
        }
        Ok(result)
    }

    fn run_heston(
        &self,
        py: Python<'_>,
        generator: &HestonPathGenerator,
        payoff: Py<PyAny>,
        discount_factor: f64,
    ) -> PyResult<(f64, f64)> {
        if !payoff.bind(py).is_callable() {
            return Err(py_value_error("payoff must be callable"));
        }
        let (core_engine, control_error) = self.to_core();
        let (payoff_fn, payoff_error) = python_path_evaluator(payoff);
        let result = core_engine.run(
            &generator.inner,
            move |path| payoff_fn(path),
            discount_factor,
        );

        if let Some(message) = payoff_error.lock().ok().and_then(|slot| slot.clone()) {
            return Err(py_value_error(message));
        }
        if let Some(message) =
            control_error.and_then(|slot| slot.lock().ok().and_then(|guard| guard.clone()))
        {
            return Err(py_value_error(message));
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "MonteCarloEngine(num_paths={}, antithetic={}, seed={}, rng_kind='{}', reproducible={})",
            self.num_paths,
            self.antithetic,
            self.seed,
            rng_kind_name(self.rng_kind),
            self.reproducible
        )
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GbmPathGenerator>()?;
    module.add_class::<HestonPathGenerator>()?;
    module.add_class::<ControlVariate>()?;
    module.add_class::<MonteCarloEngine>()?;
    Ok(())
}
