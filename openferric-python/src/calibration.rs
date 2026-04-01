use openferric_core::calibration::{
    BoxConstraints as CoreBoxConstraints, CalibrationDiagnostics as CoreCalibrationDiagnostics,
    CalibrationResult as CoreCalibrationResult, CalibrationWarningFlag, Calibrator,
    ConvergenceInfo as CoreConvergenceInfo, FitQuality as CoreFitQuality,
    HestonCalibrationParams as CoreHestonCalibrationParams,
    HestonCalibrator as CoreHestonCalibrator, HullWhiteCalibrationParams as CoreHullWhiteCalibrationParams,
    HullWhiteCalibrator as CoreHullWhiteCalibrator, InstrumentError as CoreInstrumentError,
    OptionVolQuote as CoreOptionVolQuote, ParameterStability as CoreParameterStability,
    SwaptionVolQuote as CoreSwaptionVolQuote, TerminationReason,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn string_err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn termination_reason_name(value: TerminationReason) -> &'static str {
    match value {
        TerminationReason::GradientTolerance => "gradient_tolerance",
        TerminationReason::StepTolerance => "step_tolerance",
        TerminationReason::ObjectiveTolerance => "objective_tolerance",
        TerminationReason::Stagnation => "stagnation",
        TerminationReason::MaxIterations => "max_iterations",
        TerminationReason::NumericalFailure => "numerical_failure",
    }
}

fn warning_flag_name(value: CalibrationWarningFlag) -> &'static str {
    match value {
        CalibrationWarningFlag::IllConditioned => "ill_conditioned",
        CalibrationWarningFlag::HitBoundary => "hit_boundary",
        CalibrationWarningFlag::PoorFit => "poor_fit",
        CalibrationWarningFlag::NonConvergent => "non_convergent",
        CalibrationWarningFlag::UnstableParameters => "unstable_parameters",
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BoxConstraints {
    #[pyo3(get, set)]
    pub lower: Vec<f64>,
    #[pyo3(get, set)]
    pub upper: Vec<f64>,
}

impl BoxConstraints {
    fn to_core(&self) -> PyResult<CoreBoxConstraints> {
        CoreBoxConstraints::new(self.lower.clone(), self.upper.clone()).map_err(string_err)
    }

    fn from_core(value: CoreBoxConstraints) -> Self {
        Self {
            lower: value.lower,
            upper: value.upper,
        }
    }
}

#[pymethods]
impl BoxConstraints {
    #[new]
    fn new(lower: Vec<f64>, upper: Vec<f64>) -> PyResult<Self> {
        CoreBoxConstraints::new(lower.clone(), upper.clone()).map_err(string_err)?;
        Ok(Self { lower, upper })
    }

    fn dimension(&self) -> usize {
        self.lower.len()
    }

    fn clamp(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.to_core()?.clamp(&x))
    }

    fn hits_boundary(&self, x: Vec<f64>, eps: f64) -> PyResult<bool> {
        Ok(self.to_core()?.hits_boundary(&x, eps))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct InstrumentError {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub market_mid: f64,
    #[pyo3(get, set)]
    pub market_bid: Option<f64>,
    #[pyo3(get, set)]
    pub market_ask: Option<f64>,
    #[pyo3(get, set)]
    pub model: f64,
    #[pyo3(get, set)]
    pub signed_error: f64,
    #[pyo3(get, set)]
    pub effective_error: f64,
    #[pyo3(get, set)]
    pub abs_error: f64,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub within_bid_ask: bool,
    #[pyo3(get, set)]
    pub liquid: bool,
}

impl InstrumentError {
    fn from_core(value: CoreInstrumentError) -> Self {
        Self {
            id: value.id,
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
}

#[pymethods]
impl InstrumentError {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        market_mid: f64,
        market_bid: Option<f64>,
        market_ask: Option<f64>,
        model: f64,
        signed_error: f64,
        effective_error: f64,
        abs_error: f64,
        weight: f64,
        within_bid_ask: bool,
        liquid: bool,
    ) -> Self {
        Self {
            id,
            market_mid,
            market_bid,
            market_ask,
            model,
            signed_error,
            effective_error,
            abs_error,
            weight,
            within_bid_ask,
            liquid,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ConvergenceInfo {
    #[pyo3(get, set)]
    pub iterations: usize,
    #[pyo3(get, set)]
    pub objective_evaluations: usize,
    #[pyo3(get, set)]
    pub gradient_norm: f64,
    #[pyo3(get, set)]
    pub step_norm: f64,
    #[pyo3(get, set)]
    pub converged: bool,
    #[pyo3(get, set)]
    pub reason: String,
}

impl ConvergenceInfo {
    fn from_core(value: CoreConvergenceInfo) -> Self {
        Self {
            iterations: value.iterations,
            objective_evaluations: value.objective_evaluations,
            gradient_norm: value.gradient_norm,
            step_norm: value.step_norm,
            converged: value.converged,
            reason: termination_reason_name(value.reason).to_string(),
        }
    }
}

#[pymethods]
impl ConvergenceInfo {
    #[new]
    fn new(
        iterations: usize,
        objective_evaluations: usize,
        gradient_norm: f64,
        step_norm: f64,
        converged: bool,
        reason: String,
    ) -> Self {
        Self {
            iterations,
            objective_evaluations,
            gradient_norm,
            step_norm,
            converged,
            reason,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FitQuality {
    #[pyo3(get, set)]
    pub rmse: f64,
    #[pyo3(get, set)]
    pub mae: f64,
    #[pyo3(get, set)]
    pub max_abs_error: f64,
    #[pyo3(get, set)]
    pub liquid_rmse: f64,
}

impl FitQuality {
    fn from_core(value: CoreFitQuality) -> Self {
        Self {
            rmse: value.rmse,
            mae: value.mae,
            max_abs_error: value.max_abs_error,
            liquid_rmse: value.liquid_rmse,
        }
    }
}

#[pymethods]
impl FitQuality {
    #[new]
    fn new(rmse: f64, mae: f64, max_abs_error: f64, liquid_rmse: f64) -> Self {
        Self {
            rmse,
            mae,
            max_abs_error,
            liquid_rmse,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ParameterStability {
    #[pyo3(get, set)]
    pub parameter_names: Vec<String>,
    #[pyo3(get, set)]
    pub relative_changes: Vec<f64>,
    #[pyo3(get, set)]
    pub max_relative_change: f64,
    #[pyo3(get, set)]
    pub stable: bool,
}

impl ParameterStability {
    fn from_core(value: CoreParameterStability) -> Self {
        Self {
            parameter_names: value.parameter_names,
            relative_changes: value.relative_changes,
            max_relative_change: value.max_relative_change,
            stable: value.stable,
        }
    }
}

#[pymethods]
impl ParameterStability {
    #[new]
    fn new(
        parameter_names: Vec<String>,
        relative_changes: Vec<f64>,
        max_relative_change: f64,
        stable: bool,
    ) -> Self {
        Self {
            parameter_names,
            relative_changes,
            max_relative_change,
            stable,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CalibrationDiagnostics {
    #[pyo3(get, set)]
    pub fit_quality: FitQuality,
    #[pyo3(get, set)]
    pub parameter_stability: Option<ParameterStability>,
    #[pyo3(get, set)]
    pub warning_flags: Vec<String>,
}

impl CalibrationDiagnostics {
    fn from_core(value: CoreCalibrationDiagnostics) -> Self {
        Self {
            fit_quality: FitQuality::from_core(value.fit_quality),
            parameter_stability: value.parameter_stability.map(ParameterStability::from_core),
            warning_flags: value
                .warning_flags
                .into_iter()
                .map(|flag| warning_flag_name(flag).to_string())
                .collect(),
        }
    }
}

#[pymethods]
impl CalibrationDiagnostics {
    #[new]
    fn new(
        fit_quality: FitQuality,
        parameter_stability: Option<ParameterStability>,
        warning_flags: Vec<String>,
    ) -> Self {
        Self {
            fit_quality,
            parameter_stability,
            warning_flags,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct HestonCalibrationParams {
    #[pyo3(get, set)]
    pub v0: f64,
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub sigma_v: f64,
    #[pyo3(get, set)]
    pub rho: f64,
}

impl HestonCalibrationParams {
    fn to_core(self) -> CoreHestonCalibrationParams {
        CoreHestonCalibrationParams {
            v0: self.v0,
            kappa: self.kappa,
            theta: self.theta,
            sigma_v: self.sigma_v,
            rho: self.rho,
        }
    }

    fn from_core(value: CoreHestonCalibrationParams) -> Self {
        Self {
            v0: value.v0,
            kappa: value.kappa,
            theta: value.theta,
            sigma_v: value.sigma_v,
            rho: value.rho,
        }
    }
}

#[pymethods]
impl HestonCalibrationParams {
    #[new]
    fn new(v0: f64, kappa: f64, theta: f64, sigma_v: f64, rho: f64) -> Self {
        Self {
            v0,
            kappa,
            theta,
            sigma_v,
            rho,
        }
    }

    fn to_vec(&self) -> Vec<f64> {
        self.to_core().to_vec()
    }

    #[staticmethod]
    fn from_slice(values: Vec<f64>) -> PyResult<Self> {
        CoreHestonCalibrationParams::from_slice(&values)
            .map(HestonCalibrationParams::from_core)
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct HullWhiteCalibrationParams {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl HullWhiteCalibrationParams {
    fn from_core(value: CoreHullWhiteCalibrationParams) -> Self {
        Self {
            a: value.a,
            sigma: value.sigma,
        }
    }
}

#[pymethods]
impl HullWhiteCalibrationParams {
    #[new]
    fn new(a: f64, sigma: f64) -> Self {
        Self { a, sigma }
    }

    fn to_vec(&self) -> Vec<f64> {
        vec![self.a, self.sigma]
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct OptionVolQuote {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub market_vol: f64,
    #[pyo3(get, set)]
    pub bid_vol: Option<f64>,
    #[pyo3(get, set)]
    pub ask_vol: Option<f64>,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub liquid: bool,
}

impl OptionVolQuote {
    fn to_core(&self) -> CoreOptionVolQuote {
        CoreOptionVolQuote {
            id: self.id.clone(),
            strike: self.strike,
            maturity: self.maturity,
            market_vol: self.market_vol,
            bid_vol: self.bid_vol,
            ask_vol: self.ask_vol,
            weight: self.weight,
            liquid: self.liquid,
        }
    }
}

#[pymethods]
impl OptionVolQuote {
    #[new]
    fn new(id: String, strike: f64, maturity: f64, market_vol: f64) -> Self {
        Self {
            id,
            strike,
            maturity,
            market_vol,
            bid_vol: None,
            ask_vol: None,
            weight: 1.0,
            liquid: true,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SwaptionVolQuote {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub tenor: f64,
    #[pyo3(get, set)]
    pub market_vol: f64,
    #[pyo3(get, set)]
    pub bid_vol: Option<f64>,
    #[pyo3(get, set)]
    pub ask_vol: Option<f64>,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub liquid: bool,
}

impl SwaptionVolQuote {
    fn to_core(&self) -> CoreSwaptionVolQuote {
        CoreSwaptionVolQuote {
            id: self.id.clone(),
            expiry: self.expiry,
            tenor: self.tenor,
            market_vol: self.market_vol,
            bid_vol: self.bid_vol,
            ask_vol: self.ask_vol,
            weight: self.weight,
            liquid: self.liquid,
        }
    }
}

#[pymethods]
impl SwaptionVolQuote {
    #[new]
    fn new(id: String, expiry: f64, tenor: f64, market_vol: f64) -> Self {
        Self {
            id,
            expiry,
            tenor,
            market_vol,
            bid_vol: None,
            ask_vol: None,
            weight: 1.0,
            liquid: true,
        }
    }
}

#[derive(Clone)]
enum CalibrationParamsKind {
    Heston(CoreHestonCalibrationParams),
    HullWhite(CoreHullWhiteCalibrationParams),
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CalibrationResult {
    params: CalibrationParamsKind,
    #[pyo3(get)]
    pub objective: f64,
    #[pyo3(get)]
    pub per_instrument_error: Vec<InstrumentError>,
    #[pyo3(get)]
    pub jacobian: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub condition_number: f64,
    #[pyo3(get)]
    pub convergence: ConvergenceInfo,
    #[pyo3(get)]
    pub diagnostics: CalibrationDiagnostics,
}

impl CalibrationResult {
    fn from_heston(value: CoreCalibrationResult<CoreHestonCalibrationParams>) -> Self {
        Self {
            params: CalibrationParamsKind::Heston(value.params),
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

    fn from_hull_white(value: CoreCalibrationResult<CoreHullWhiteCalibrationParams>) -> Self {
        Self {
            params: CalibrationParamsKind::HullWhite(value.params),
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
}

#[pymethods]
impl CalibrationResult {
    #[getter]
    fn params_type(&self) -> &'static str {
        match self.params {
            CalibrationParamsKind::Heston(_) => "heston",
            CalibrationParamsKind::HullWhite(_) => "hull_white",
        }
    }

    fn heston_params(&self) -> Option<HestonCalibrationParams> {
        match self.params {
            CalibrationParamsKind::Heston(value) => Some(HestonCalibrationParams::from_core(value)),
            CalibrationParamsKind::HullWhite(_) => None,
        }
    }

    fn hull_white_params(&self) -> Option<HullWhiteCalibrationParams> {
        match self.params {
            CalibrationParamsKind::Heston(_) => None,
            CalibrationParamsKind::HullWhite(value) => Some(HullWhiteCalibrationParams::from_core(value)),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HestonCalibrator {
    inner: CoreHestonCalibrator,
}

#[pymethods]
impl HestonCalibrator {
    #[new]
    #[pyo3(signature=(spot=100.0, rate=0.02, dividend_yield=0.0))]
    fn new(spot: f64, rate: f64, dividend_yield: f64) -> Self {
        let mut inner = CoreHestonCalibrator::default();
        inner.spot = spot;
        inner.rate = rate;
        inner.dividend_yield = dividend_yield;
        Self { inner }
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }

    #[setter]
    fn set_spot(&mut self, value: f64) {
        self.inner.spot = value;
    }

    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }

    #[setter]
    fn set_rate(&mut self, value: f64) {
        self.inner.rate = value;
    }

    #[getter]
    fn dividend_yield(&self) -> f64 {
        self.inner.dividend_yield
    }

    #[setter]
    fn set_dividend_yield(&mut self, value: f64) {
        self.inner.dividend_yield = value;
    }

    #[getter]
    fn bounds(&self) -> BoxConstraints {
        BoxConstraints::from_core(self.inner.bounds.clone())
    }

    #[setter]
    fn set_bounds(&mut self, value: &BoxConstraints) -> PyResult<()> {
        self.inner.bounds = value.to_core()?;
        Ok(())
    }

    fn calibrate(&self, quotes: Vec<OptionVolQuote>) -> PyResult<CalibrationResult> {
        let quotes: Vec<_> = quotes.into_iter().map(|quote| quote.to_core()).collect();
        self.inner
            .calibrate(&quotes)
            .map(CalibrationResult::from_heston)
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HullWhiteCalibrator {
    inner: CoreHullWhiteCalibrator,
}

#[pymethods]
impl HullWhiteCalibrator {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreHullWhiteCalibrator::default(),
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    fn bounds(&self) -> BoxConstraints {
        BoxConstraints::from_core(self.inner.bounds.clone())
    }

    #[setter]
    fn set_bounds(&mut self, value: &BoxConstraints) -> PyResult<()> {
        self.inner.bounds = value.to_core()?;
        Ok(())
    }

    #[getter]
    fn use_nelder_mead_fallback(&self) -> bool {
        self.inner.use_nelder_mead_fallback
    }

    #[setter]
    fn set_use_nelder_mead_fallback(&mut self, value: bool) {
        self.inner.use_nelder_mead_fallback = value;
    }

    fn calibrate(&self, quotes: Vec<SwaptionVolQuote>) -> PyResult<CalibrationResult> {
        let quotes: Vec<_> = quotes.into_iter().map(|quote| quote.to_core()).collect();
        self.inner
            .calibrate(&quotes)
            .map(CalibrationResult::from_hull_white)
            .map_err(string_err)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BoxConstraints>()?;
    module.add_class::<InstrumentError>()?;
    module.add_class::<ConvergenceInfo>()?;
    module.add_class::<FitQuality>()?;
    module.add_class::<ParameterStability>()?;
    module.add_class::<CalibrationDiagnostics>()?;
    module.add_class::<HestonCalibrationParams>()?;
    module.add_class::<HullWhiteCalibrationParams>()?;
    module.add_class::<OptionVolQuote>()?;
    module.add_class::<SwaptionVolQuote>()?;
    module.add_class::<CalibrationResult>()?;
    module.add_class::<HestonCalibrator>()?;
    module.add_class::<HullWhiteCalibrator>()?;
    Ok(())
}
