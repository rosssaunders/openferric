use super::*;
use openferric_core::calibration::{self as native, optimizers};
use pyo3::types::PyAny;
use std::panic::AssertUnwindSafe;

macro_rules! optimizer_options {
    ($name:ident { $($field:ident: $field_type:ty),* $(,)? }) => {
        #[pyclass(module = "openferric", get_all, set_all, from_py_object)]
        #[derive(Clone, Copy)]
        pub struct $name { $(pub $field: $field_type),* }

        impl $name {
            pub(crate) fn to_core(self) -> optimizers::$name {
                optimizers::$name { $($field: self.$field),* }
            }
        }

        impl From<optimizers::$name> for $name {
            fn from(value: optimizers::$name) -> Self {
                Self { $($field: value.$field),* }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = ($($field=None),*))]
            fn new($($field: Option<$field_type>),*) -> Self {
                let defaults = optimizers::$name::default();
                Self { $($field: $field.unwrap_or(defaults.$field)),* }
            }
        }
    };
}

optimizer_options!(LmOptions {
    max_iterations: usize,
    initial_lambda: f64,
    lambda_up: f64,
    lambda_down: f64,
    gradient_tolerance: f64,
    step_tolerance: f64,
    objective_tolerance: f64,
    finite_diff_epsilon: f64,
    max_stagnation: usize,
});
optimizer_options!(DifferentialEvolutionOptions {
    max_generations: usize,
    population_size: usize,
    mutation_factor: f64,
    crossover_probability: f64,
    seed: u64,
    max_stagnation: usize,
});
optimizer_options!(NelderMeadOptions {
    max_iterations: usize,
    initial_step: f64,
    reflection: f64,
    expansion: f64,
    contraction: f64,
    shrink: f64,
    tolerance: f64,
});

macro_rules! parameters {
    ($name:ident { $($field:ident),* $(,)? }) => {
        #[pyclass(module = "openferric", get_all, set_all, from_py_object)]
        #[derive(Clone, Copy)]
        pub struct $name { $(pub $field: f64),* }

        impl From<native::$name> for $name {
            fn from(value: native::$name) -> Self {
                Self { $($field: value.$field),* }
            }
        }

        impl $name {
            fn to_core(self) -> native::$name {
                native::$name { $($field: self.$field),* }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new($($field: f64),*) -> Self {
                Self { $($field),* }
            }

            #[allow(clippy::wrong_self_convention)]
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                crate::helpers::to_python(py, &self.to_core())
            }
        }
    };
}

parameters!(SabrCalibrationParams {
    alpha,
    beta,
    rho,
    nu
});
parameters!(SviRawCalibrationParams {
    a,
    b,
    rho,
    m,
    sigma
});
parameters!(SviJumpWingsCalibrationParams {
    v,
    psi,
    p,
    c,
    vt,
    maturity
});

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SviParameterization {
    Raw,
    JumpWings,
}

impl SviParameterization {
    fn to_core(self) -> native::SviParameterization {
        match self {
            Self::Raw => native::SviParameterization::Raw,
            Self::JumpWings => native::SviParameterization::JumpWings,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SviCalibrationParams {
    pub(crate) inner: native::SviCalibrationParams,
}

#[pymethods]
impl SviCalibrationParams {
    #[staticmethod]
    fn raw(params: SviRawCalibrationParams) -> Self {
        Self {
            inner: native::SviCalibrationParams::Raw(params.to_core()),
        }
    }

    #[staticmethod]
    fn jump_wings(params: SviJumpWingsCalibrationParams) -> Self {
        Self {
            inner: native::SviCalibrationParams::JumpWings(params.to_core()),
        }
    }

    #[getter]
    fn parameterization(&self) -> SviParameterization {
        match self.inner {
            native::SviCalibrationParams::Raw(_) => SviParameterization::Raw,
            native::SviCalibrationParams::JumpWings(_) => SviParameterization::JumpWings,
        }
    }

    fn raw_params(&self) -> Option<SviRawCalibrationParams> {
        match self.inner {
            native::SviCalibrationParams::Raw(value) => Some(value.into()),
            _ => None,
        }
    }

    fn jump_wings_params(&self) -> Option<SviJumpWingsCalibrationParams> {
        match self.inner {
            native::SviCalibrationParams::JumpWings(value) => Some(value.into()),
            _ => None,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::helpers::to_python(py, &self.inner)
    }
}

fn calibration_result<Parameters>(
    value: native::CalibrationResult<Parameters>,
    convert: impl FnOnce(Parameters) -> CalibrationParamsKind,
) -> CalibrationResult {
    CalibrationResult {
        params: convert(value.params),
        objective: value.objective,
        per_instrument_error: value
            .per_instrument_error
            .into_iter()
            .map(InstrumentError::from_core)
            .collect(),
        jacobian: value.jacobian,
        condition_number: value.condition_number,
        convergence: ConvergenceInfo::from_core(value.convergence),
        diagnostics: CalibrationDiagnostics::from_core(value.diagnostics),
    }
}

#[pyclass(module = "openferric", get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct SabrCalibrator {
    forward: f64,
    maturity: f64,
    beta_pin: Option<f64>,
    lm_options: LmOptions,
    de_options: DifferentialEvolutionOptions,
    nm_options: NelderMeadOptions,
    use_global_search: bool,
    use_nelder_mead_fallback: bool,
}

#[pymethods]
impl SabrCalibrator {
    #[new]
    #[pyo3(signature = (forward=100.0, maturity=1.0, beta_pin=Some(0.5)))]
    fn new(forward: f64, maturity: f64, beta_pin: Option<f64>) -> Self {
        let defaults = native::SabrCalibrator::default();
        Self {
            forward,
            maturity,
            beta_pin,
            lm_options: defaults.lm_options.into(),
            de_options: defaults.de_options.into(),
            nm_options: defaults.nm_options.into(),
            use_global_search: defaults.use_global_search,
            use_nelder_mead_fallback: defaults.use_nelder_mead_fallback,
        }
    }

    fn calibrate(
        &self,
        py: Python<'_>,
        quotes: Vec<OptionVolQuote>,
    ) -> PyResult<CalibrationResult> {
        let calibrator = native::SabrCalibrator {
            forward: self.forward,
            maturity: self.maturity,
            beta_pin: self.beta_pin,
            lm_options: self.lm_options.to_core(),
            de_options: self.de_options.to_core(),
            nm_options: self.nm_options.to_core(),
            use_global_search: self.use_global_search,
            use_nelder_mead_fallback: self.use_nelder_mead_fallback,
        };
        let quotes = quotes
            .iter()
            .map(OptionVolQuote::to_core)
            .collect::<Vec<_>>();
        let result = py
            .detach(|| calibrator.calibrate(&quotes))
            .map_err(string_err)?;
        Ok(calibration_result(result, CalibrationParamsKind::Sabr))
    }

    fn name(&self) -> &'static str {
        native::SabrCalibrator::default().name()
    }
}

#[pyclass(module = "openferric", get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct SviCalibrator {
    forward: f64,
    maturity: f64,
    parameterization: SviParameterization,
    lm_options: LmOptions,
    de_options: DifferentialEvolutionOptions,
    nm_options: NelderMeadOptions,
    use_global_search: bool,
    use_nelder_mead_fallback: bool,
}

#[pymethods]
impl SviCalibrator {
    #[new]
    #[pyo3(signature = (forward=100.0, maturity=1.0, parameterization=SviParameterization::Raw))]
    fn new(forward: f64, maturity: f64, parameterization: SviParameterization) -> Self {
        let defaults = native::SviCalibrator::default();
        Self {
            forward,
            maturity,
            parameterization,
            lm_options: defaults.lm_options.into(),
            de_options: defaults.de_options.into(),
            nm_options: defaults.nm_options.into(),
            use_global_search: defaults.use_global_search,
            use_nelder_mead_fallback: defaults.use_nelder_mead_fallback,
        }
    }

    fn calibrate(
        &self,
        py: Python<'_>,
        quotes: Vec<OptionVolQuote>,
    ) -> PyResult<CalibrationResult> {
        let calibrator = native::SviCalibrator {
            forward: self.forward,
            maturity: self.maturity,
            parameterization: self.parameterization.to_core(),
            lm_options: self.lm_options.to_core(),
            de_options: self.de_options.to_core(),
            nm_options: self.nm_options.to_core(),
            use_global_search: self.use_global_search,
            use_nelder_mead_fallback: self.use_nelder_mead_fallback,
        };
        let quotes = quotes
            .iter()
            .map(OptionVolQuote::to_core)
            .collect::<Vec<_>>();
        let result = py
            .detach(|| calibrator.calibrate(&quotes))
            .map_err(string_err)?;
        Ok(calibration_result(result, CalibrationParamsKind::Svi))
    }

    fn name(&self) -> &'static str {
        native::SviCalibrator::default().name()
    }
}

#[pyclass(module = "openferric", get_all, from_py_object)]
#[derive(Clone)]
pub struct OptimisationResult {
    x: Vec<f64>,
    objective: f64,
    residuals: Vec<f64>,
    jacobian: Vec<Vec<f64>>,
    convergence: ConvergenceInfo,
}

impl From<optimizers::OptimisationResult> for OptimisationResult {
    fn from(value: optimizers::OptimisationResult) -> Self {
        Self {
            x: value.x,
            objective: value.objective,
            residuals: value.residuals,
            jacobian: native::core::matrix_to_rows(&value.jacobian),
            convergence: ConvergenceInfo::from_core(value.convergence),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (initial, bounds, residual_function, options=None))]
fn levenberg_marquardt(
    initial: Vec<f64>,
    bounds: &BoxConstraints,
    residual_function: &Bound<'_, PyAny>,
    options: Option<LmOptions>,
) -> PyResult<OptimisationResult> {
    let bounds = bounds.to_core()?;
    let options = options.map(LmOptions::to_core).unwrap_or_default();
    let mut failure = None;
    let mut residual_count = None;
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        optimizers::levenberg_marquardt(&initial, &bounds, options, |parameters| {
            if failure.is_some() {
                return vec![f64::NAN; residual_count.unwrap_or(1)];
            }
            let values = residual_function
                .call1((parameters.to_vec(),))
                .and_then(|value| value.extract::<Vec<f64>>());
            match values {
                Ok(values)
                    if !values.is_empty()
                        && values.iter().all(|value| value.is_finite())
                        && residual_count.is_none_or(|count| count == values.len()) =>
                {
                    residual_count = Some(values.len());
                    values
                }
                Ok(_) => {
                    failure = Some(PyValueError::new_err(
                        "residuals must be finite, nonempty, and keep the same length",
                    ));
                    vec![f64::NAN; residual_count.unwrap_or(1)]
                }
                Err(error) => {
                    failure = Some(error);
                    vec![f64::NAN; residual_count.unwrap_or(1)]
                }
            }
        })
    }))
    .map_err(crate::helpers::panic_to_pyerr)?;
    if let Some(error) = failure {
        return Err(error);
    }
    result.map(Into::into).map_err(string_err)
}

fn scalar_objective(
    function: &Bound<'_, PyAny>,
    parameters: &[f64],
    failure: &mut Option<PyErr>,
) -> f64 {
    if failure.is_some() {
        return f64::NAN;
    }
    match function
        .call1((parameters.to_vec(),))
        .and_then(|value| value.extract::<f64>())
    {
        Ok(value) if value.is_finite() => value,
        Ok(_) => {
            *failure = Some(PyValueError::new_err("objective must be finite"));
            f64::NAN
        }
        Err(error) => {
            *failure = Some(error);
            f64::NAN
        }
    }
}

#[pyfunction]
#[pyo3(signature = (bounds, objective_function, options=None))]
fn differential_evolution(
    bounds: &BoxConstraints,
    objective_function: &Bound<'_, PyAny>,
    options: Option<DifferentialEvolutionOptions>,
) -> PyResult<OptimisationResult> {
    let bounds = bounds.to_core()?;
    let options = options
        .map(DifferentialEvolutionOptions::to_core)
        .unwrap_or_default();
    let mut failure = None;
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        optimizers::differential_evolution(&bounds, options, |parameters| {
            scalar_objective(objective_function, parameters, &mut failure)
        })
    }))
    .map_err(crate::helpers::panic_to_pyerr)?;
    if let Some(error) = failure {
        return Err(error);
    }
    result.map(Into::into).map_err(string_err)
}

#[pyfunction]
#[pyo3(signature = (initial, bounds, objective_function, options=None))]
fn nelder_mead(
    initial: Vec<f64>,
    bounds: &BoxConstraints,
    objective_function: &Bound<'_, PyAny>,
    options: Option<NelderMeadOptions>,
) -> PyResult<OptimisationResult> {
    let bounds = bounds.to_core()?;
    let options = options.map(NelderMeadOptions::to_core).unwrap_or_default();
    let mut failure = None;
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        optimizers::nelder_mead(&initial, &bounds, options, |parameters| {
            scalar_objective(objective_function, parameters, &mut failure)
        })
    }))
    .map_err(crate::helpers::panic_to_pyerr)?;
    if let Some(error) = failure {
        return Err(error);
    }
    result.map(Into::into).map_err(string_err)
}

fn native_error(value: &InstrumentError) -> native::InstrumentError {
    native::InstrumentError {
        id: value.id.clone(),
        market_mid: value.market_mid,
        market_bid: value.market_bid,
        market_ask: value.market_ask,
        model: value.model,
        signed_error: value.signed_error,
        effective_error: value.effective_error,
        abs_error: value.abs_error,
        weight: value.weight,
        within_bid_ask: value.within_bid_ask,
        liquid: value.liquid,
    }
}

fn native_convergence(value: &ConvergenceInfo) -> PyResult<native::ConvergenceInfo> {
    let reason = match value.reason.as_str() {
        "gradient_tolerance" => TerminationReason::GradientTolerance,
        "step_tolerance" => TerminationReason::StepTolerance,
        "objective_tolerance" => TerminationReason::ObjectiveTolerance,
        "stagnation" => TerminationReason::Stagnation,
        "max_iterations" => TerminationReason::MaxIterations,
        "numerical_failure" => TerminationReason::NumericalFailure,
        _ => return Err(string_err("unknown termination reason")),
    };
    Ok(native::ConvergenceInfo {
        iterations: value.iterations,
        objective_evaluations: value.objective_evaluations,
        gradient_norm: value.gradient_norm,
        step_norm: value.step_norm,
        converged: value.converged,
        reason,
    })
}

fn native_stability(value: &ParameterStability) -> native::ParameterStability {
    native::ParameterStability {
        parameter_names: value.parameter_names.clone(),
        relative_changes: value.relative_changes.clone(),
        max_relative_change: value.max_relative_change,
        stable: value.stable,
    }
}

#[pyfunction]
fn bid_ask_aware_error(quote: &Bound<'_, PyAny>, model_vol: f64) -> PyResult<(f64, f64, bool)> {
    if let Ok(quote) = quote.extract::<PyRef<'_, OptionVolQuote>>() {
        Ok(native::instruments::bid_ask_aware_error(
            &quote.to_core(),
            model_vol,
        ))
    } else {
        let quote = quote.extract::<PyRef<'_, SwaptionVolQuote>>()?;
        Ok(native::instruments::bid_ask_aware_error(
            &quote.to_core(),
            model_vol,
        ))
    }
}

#[pyfunction]
fn make_error_record(quote: &Bound<'_, PyAny>, model_vol: f64) -> PyResult<InstrumentError> {
    if let Ok(quote) = quote.extract::<PyRef<'_, OptionVolQuote>>() {
        Ok(InstrumentError::from_core(
            native::instruments::make_error_record(&quote.to_core(), model_vol),
        ))
    } else {
        let quote = quote.extract::<PyRef<'_, SwaptionVolQuote>>()?;
        Ok(InstrumentError::from_core(
            native::instruments::make_error_record(&quote.to_core(), model_vol),
        ))
    }
}

#[pyfunction]
fn fit_quality(errors: Vec<InstrumentError>) -> FitQuality {
    FitQuality::from_core(native::fit_quality(
        &errors.iter().map(native_error).collect::<Vec<_>>(),
    ))
}

#[pyfunction]
fn parameter_stability(
    names: Vec<String>,
    previous: Vec<f64>,
    current: Vec<f64>,
    threshold: f64,
) -> ParameterStability {
    ParameterStability::from_core(native::parameter_stability(
        names, &previous, &current, threshold,
    ))
}

#[pyfunction]
#[pyo3(signature = (errors, convergence, condition_number, bounds=None, params=None, stability=None))]
fn calibration_diagnostics(
    errors: Vec<InstrumentError>,
    convergence: &ConvergenceInfo,
    condition_number: f64,
    bounds: Option<&BoxConstraints>,
    params: Option<Vec<f64>>,
    stability: Option<&ParameterStability>,
) -> PyResult<CalibrationDiagnostics> {
    let errors = errors.iter().map(native_error).collect::<Vec<_>>();
    let bounds = bounds.map(BoxConstraints::to_core).transpose()?;
    Ok(CalibrationDiagnostics::from_core(native::diagnostics(
        &errors,
        &native_convergence(convergence)?,
        condition_number,
        bounds.as_ref(),
        params.as_deref(),
        stability.map(native_stability),
    )))
}

#[pyfunction]
fn finite_metric(value: f64) -> f64 {
    native::core::finite_metric(value)
}

#[pyfunction]
#[pyo3(signature = (convergence, condition_number, fit, bounds=None, params=None, stability=None))]
fn warning_flags(
    convergence: &ConvergenceInfo,
    condition_number: f64,
    fit: &FitQuality,
    bounds: Option<&BoxConstraints>,
    params: Option<Vec<f64>>,
    stability: Option<&ParameterStability>,
) -> PyResult<Vec<String>> {
    let bounds = bounds.map(BoxConstraints::to_core).transpose()?;
    let stability = stability.map(native_stability);
    let fit = native::FitQuality {
        rmse: fit.rmse,
        mae: fit.mae,
        max_abs_error: fit.max_abs_error,
        liquid_rmse: fit.liquid_rmse,
    };
    Ok(native::warning_flags(
        &native_convergence(convergence)?,
        condition_number,
        &fit,
        bounds.as_ref(),
        params.as_deref(),
        stability.as_ref(),
    )
    .into_iter()
    .map(|flag| warning_flag_name(flag).to_owned())
    .collect())
}

#[pyfunction]
fn sanitize_convergence(value: &ConvergenceInfo) -> PyResult<ConvergenceInfo> {
    Ok(ConvergenceInfo::from_core(
        native::core::sanitize_convergence(native_convergence(value)?),
    ))
}

#[pyfunction]
fn matrix_condition_number(rows: Vec<Vec<f64>>) -> PyResult<f64> {
    let columns = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != columns) {
        return Err(string_err("matrix rows must have equal lengths"));
    }
    let matrix = nalgebra::DMatrix::from_row_slice(
        rows.len(),
        columns,
        &rows.into_iter().flatten().collect::<Vec<_>>(),
    );
    Ok(native::core::matrix_condition_number(&matrix))
}

#[pyfunction]
fn matrix_to_rows(rows: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let columns = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != columns) {
        return Err(string_err("matrix rows must have equal lengths"));
    }
    let matrix = nalgebra::DMatrix::from_row_slice(
        rows.len(),
        columns,
        &rows.into_iter().flatten().collect::<Vec<_>>(),
    );
    Ok(native::core::matrix_to_rows(&matrix))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(warning_flags, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_to_rows, module)?)?;
    module.add_class::<LmOptions>()?;
    module.add_class::<DifferentialEvolutionOptions>()?;
    module.add_class::<NelderMeadOptions>()?;
    module.add_class::<OptimisationResult>()?;
    module.add_class::<SabrCalibrationParams>()?;
    module.add_class::<SviRawCalibrationParams>()?;
    module.add_class::<SviJumpWingsCalibrationParams>()?;
    module.add_class::<SviCalibrationParams>()?;
    module.add_class::<SviParameterization>()?;
    module.add_class::<SabrCalibrator>()?;
    module.add_class::<SviCalibrator>()?;
    module.add_function(wrap_pyfunction!(levenberg_marquardt, module)?)?;
    module.add_function(wrap_pyfunction!(differential_evolution, module)?)?;
    module.add_function(wrap_pyfunction!(nelder_mead, module)?)?;
    module.add_function(wrap_pyfunction!(bid_ask_aware_error, module)?)?;
    module.add_function(wrap_pyfunction!(make_error_record, module)?)?;
    module.add_function(wrap_pyfunction!(fit_quality, module)?)?;
    module.add_function(wrap_pyfunction!(parameter_stability, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_diagnostics, module)?)?;
    module.add_function(wrap_pyfunction!(finite_metric, module)?)?;
    module.add_function(wrap_pyfunction!(sanitize_convergence, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_condition_number, module)?)?;
    Ok(())
}
