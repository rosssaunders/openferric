use openferric_core::core::{
    AsianSpec as CoreAsianSpec, Averaging as CoreAveraging,
    BarrierDirection as CoreBarrierDirection, BarrierSpec as CoreBarrierSpec,
    BarrierStyle as CoreBarrierStyle, DiagKey as CoreDiagKey, Diagnostics as CoreDiagnostics,
    ExerciseStyle as CoreExerciseStyle, Greeks as CoreGreeks, OptionType as CoreOptionType,
    PricingError as CorePricingError, PricingResult as CorePricingResult,
    StrikeType as CoreStrikeType,
};
use pyo3::prelude::*;

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub(crate) fn to_core(self) -> CoreOptionType {
        match self {
            Self::Call => CoreOptionType::Call,
            Self::Put => CoreOptionType::Put,
        }
    }
}

#[pymethods]
impl OptionType {
    fn sign(&self) -> f64 {
        self.to_core().sign()
    }

    fn __repr__(&self) -> String {
        format!("OptionType.{}", self.name())
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Call => "Call",
            Self::Put => "Put",
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct ExerciseStyle {
    inner: CoreExerciseStyle,
}

impl ExerciseStyle {
    pub(crate) fn to_core(&self) -> CoreExerciseStyle {
        self.inner.clone()
    }

    pub(crate) fn from_core(value: CoreExerciseStyle) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl ExerciseStyle {
    #[staticmethod]
    fn european() -> Self {
        Self {
            inner: CoreExerciseStyle::European,
        }
    }

    #[staticmethod]
    fn american() -> Self {
        Self {
            inner: CoreExerciseStyle::American,
        }
    }

    #[staticmethod]
    fn bermudan(dates: Vec<f64>) -> Self {
        Self {
            inner: CoreExerciseStyle::Bermudan { dates },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            CoreExerciseStyle::European => "European",
            CoreExerciseStyle::American => "American",
            CoreExerciseStyle::Bermudan { .. } => "Bermudan",
        }
    }

    #[getter]
    fn dates(&self) -> Option<Vec<f64>> {
        match &self.inner {
            CoreExerciseStyle::Bermudan { dates } => Some(dates.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            CoreExerciseStyle::European => "ExerciseStyle.european()".to_string(),
            CoreExerciseStyle::American => "ExerciseStyle.american()".to_string(),
            CoreExerciseStyle::Bermudan { dates } => {
                format!("ExerciseStyle.bermudan(dates={dates:?})")
            }
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BarrierDirection {
    Up,
    Down,
}

impl BarrierDirection {
    pub(crate) fn to_core(self) -> CoreBarrierDirection {
        match self {
            Self::Up => CoreBarrierDirection::Up,
            Self::Down => CoreBarrierDirection::Down,
        }
    }

    pub(crate) fn from_core(value: CoreBarrierDirection) -> Self {
        match value {
            CoreBarrierDirection::Up => Self::Up,
            CoreBarrierDirection::Down => Self::Down,
        }
    }
}

#[pymethods]
impl BarrierDirection {
    fn __repr__(&self) -> String {
        format!("BarrierDirection.{}", self.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Up => "Up",
            Self::Down => "Down",
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BarrierStyle {
    In,
    Out,
}

impl BarrierStyle {
    pub(crate) fn to_core(self) -> CoreBarrierStyle {
        match self {
            Self::In => CoreBarrierStyle::In,
            Self::Out => CoreBarrierStyle::Out,
        }
    }

    pub(crate) fn from_core(value: CoreBarrierStyle) -> Self {
        match value {
            CoreBarrierStyle::In => Self::In,
            CoreBarrierStyle::Out => Self::Out,
        }
    }
}

#[pymethods]
impl BarrierStyle {
    fn __repr__(&self) -> String {
        format!("BarrierStyle.{}", self.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::In => "In",
            Self::Out => "Out",
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct BarrierSpec {
    inner: CoreBarrierSpec,
}

impl BarrierSpec {
    pub(crate) fn to_core(&self) -> CoreBarrierSpec {
        self.inner.clone()
    }

    pub(crate) fn from_core(value: CoreBarrierSpec) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl BarrierSpec {
    #[new]
    fn new(direction: &BarrierDirection, style: &BarrierStyle, level: f64, rebate: f64) -> Self {
        Self {
            inner: CoreBarrierSpec {
                direction: direction.to_core(),
                style: style.to_core(),
                level,
                rebate,
            },
        }
    }

    #[getter]
    fn direction(&self) -> BarrierDirection {
        BarrierDirection::from_core(self.inner.direction)
    }

    #[setter]
    fn set_direction(&mut self, direction: &BarrierDirection) {
        self.inner.direction = direction.to_core();
    }

    #[getter]
    fn style(&self) -> BarrierStyle {
        BarrierStyle::from_core(self.inner.style)
    }

    #[setter]
    fn set_style(&mut self, style: &BarrierStyle) {
        self.inner.style = style.to_core();
    }

    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    #[setter]
    fn set_level(&mut self, level: f64) {
        self.inner.level = level;
    }

    #[getter]
    fn rebate(&self) -> f64 {
        self.inner.rebate
    }

    #[setter]
    fn set_rebate(&mut self, rebate: f64) {
        self.inner.rebate = rebate;
    }

    fn __repr__(&self) -> String {
        format!(
            "BarrierSpec(direction={:?}, style={:?}, level={}, rebate={})",
            self.inner.direction, self.inner.style, self.inner.level, self.inner.rebate
        )
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Averaging {
    Arithmetic,
    Geometric,
}

impl Averaging {
    pub(crate) fn to_core(self) -> CoreAveraging {
        match self {
            Self::Arithmetic => CoreAveraging::Arithmetic,
            Self::Geometric => CoreAveraging::Geometric,
        }
    }

    pub(crate) fn from_core(value: CoreAveraging) -> Self {
        match value {
            CoreAveraging::Arithmetic => Self::Arithmetic,
            CoreAveraging::Geometric => Self::Geometric,
        }
    }
}

#[pymethods]
impl Averaging {
    fn __repr__(&self) -> String {
        format!("Averaging.{}", self.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Arithmetic => "Arithmetic",
            Self::Geometric => "Geometric",
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StrikeType {
    Fixed,
    Floating,
}

impl StrikeType {
    pub(crate) fn to_core(self) -> CoreStrikeType {
        match self {
            Self::Fixed => CoreStrikeType::Fixed,
            Self::Floating => CoreStrikeType::Floating,
        }
    }

    pub(crate) fn from_core(value: CoreStrikeType) -> Self {
        match value {
            CoreStrikeType::Fixed => Self::Fixed,
            CoreStrikeType::Floating => Self::Floating,
        }
    }
}

#[pymethods]
impl StrikeType {
    fn __repr__(&self) -> String {
        format!("StrikeType.{}", self.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Fixed => "Fixed",
            Self::Floating => "Floating",
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct AsianSpec {
    inner: CoreAsianSpec,
}

impl AsianSpec {
    pub(crate) fn to_core(&self) -> CoreAsianSpec {
        self.inner.clone()
    }
}

#[pymethods]
impl AsianSpec {
    #[new]
    fn new(averaging: &Averaging, strike_type: &StrikeType, observation_times: Vec<f64>) -> Self {
        Self {
            inner: CoreAsianSpec {
                averaging: averaging.to_core(),
                strike_type: strike_type.to_core(),
                observation_times,
            },
        }
    }

    #[getter]
    fn averaging(&self) -> Averaging {
        Averaging::from_core(self.inner.averaging)
    }

    #[setter]
    fn set_averaging(&mut self, averaging: &Averaging) {
        self.inner.averaging = averaging.to_core();
    }

    #[getter]
    fn strike_type(&self) -> StrikeType {
        StrikeType::from_core(self.inner.strike_type)
    }

    #[setter]
    fn set_strike_type(&mut self, strike_type: &StrikeType) {
        self.inner.strike_type = strike_type.to_core();
    }

    #[getter]
    fn observation_times(&self) -> Vec<f64> {
        self.inner.observation_times.clone()
    }

    #[setter]
    fn set_observation_times(&mut self, observation_times: Vec<f64>) {
        self.inner.observation_times = observation_times;
    }

    fn __repr__(&self) -> String {
        format!(
            "AsianSpec(averaging={:?}, strike_type={:?}, observation_times={:?})",
            self.inner.averaging, self.inner.strike_type, self.inner.observation_times
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct Greeks {
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub rho: f64,
}

impl Greeks {
    pub(crate) fn to_core(self) -> CoreGreeks {
        CoreGreeks {
            delta: self.delta,
            gamma: self.gamma,
            vega: self.vega,
            theta: self.theta,
            rho: self.rho,
        }
    }

    pub(crate) fn from_core(value: CoreGreeks) -> Self {
        Self {
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            theta: value.theta,
            rho: value.rho,
        }
    }
}

#[pymethods]
impl Greeks {
    #[new]
    fn new(delta: f64, gamma: f64, vega: f64, theta: f64, rho: f64) -> Self {
        Self {
            delta,
            gamma,
            vega,
            theta,
            rho,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Greeks(delta={}, gamma={}, vega={}, theta={}, rho={})",
            self.delta, self.gamma, self.vega, self.theta, self.rho
        )
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DiagKey {
    BarrierLevel,
    ConversionValue,
    CreditSpread,
    D,
    D1,
    D2,
    Delta,
    DiscountFactor,
    DoubleKnockoutBase,
    EffectiveVol,
    ExerciseDates,
    FairVariance,
    FairVolatility,
    InsideBarriers,
    Integral,
    MaxExercises,
    MinExercises,
    Npv,
    NumPaths,
    NumThreads,
    ObservationCount,
    NumSpaceSteps,
    NumSteps,
    NumTimeSteps,
    Pd,
    Pm,
    Pu,
    PvForward,
    Rho,
    RhoDomestic,
    RhoForeign,
    SMax,
    SeriesTerms,
    SurvivalDigital,
    U,
    VarOfVar,
    Vol,
    VolAdj,
}

impl DiagKey {
    pub(crate) fn to_core(self) -> CoreDiagKey {
        match self {
            Self::BarrierLevel => CoreDiagKey::BarrierLevel,
            Self::ConversionValue => CoreDiagKey::ConversionValue,
            Self::CreditSpread => CoreDiagKey::CreditSpread,
            Self::D => CoreDiagKey::D,
            Self::D1 => CoreDiagKey::D1,
            Self::D2 => CoreDiagKey::D2,
            Self::Delta => CoreDiagKey::Delta,
            Self::DiscountFactor => CoreDiagKey::DiscountFactor,
            Self::DoubleKnockoutBase => CoreDiagKey::DoubleKnockoutBase,
            Self::EffectiveVol => CoreDiagKey::EffectiveVol,
            Self::ExerciseDates => CoreDiagKey::ExerciseDates,
            Self::FairVariance => CoreDiagKey::FairVariance,
            Self::FairVolatility => CoreDiagKey::FairVolatility,
            Self::InsideBarriers => CoreDiagKey::InsideBarriers,
            Self::Integral => CoreDiagKey::Integral,
            Self::MaxExercises => CoreDiagKey::MaxExercises,
            Self::MinExercises => CoreDiagKey::MinExercises,
            Self::Npv => CoreDiagKey::Npv,
            Self::NumPaths => CoreDiagKey::NumPaths,
            Self::NumThreads => CoreDiagKey::NumThreads,
            Self::ObservationCount => CoreDiagKey::ObservationCount,
            Self::NumSpaceSteps => CoreDiagKey::NumSpaceSteps,
            Self::NumSteps => CoreDiagKey::NumSteps,
            Self::NumTimeSteps => CoreDiagKey::NumTimeSteps,
            Self::Pd => CoreDiagKey::Pd,
            Self::Pm => CoreDiagKey::Pm,
            Self::Pu => CoreDiagKey::Pu,
            Self::PvForward => CoreDiagKey::PvForward,
            Self::Rho => CoreDiagKey::Rho,
            Self::RhoDomestic => CoreDiagKey::RhoDomestic,
            Self::RhoForeign => CoreDiagKey::RhoForeign,
            Self::SMax => CoreDiagKey::SMax,
            Self::SeriesTerms => CoreDiagKey::SeriesTerms,
            Self::SurvivalDigital => CoreDiagKey::SurvivalDigital,
            Self::U => CoreDiagKey::U,
            Self::VarOfVar => CoreDiagKey::VarOfVar,
            Self::Vol => CoreDiagKey::Vol,
            Self::VolAdj => CoreDiagKey::VolAdj,
        }
    }
}

#[pymethods]
impl DiagKey {
    #[getter]
    fn as_str(&self) -> &'static str {
        self.to_core().as_str()
    }

    fn __repr__(&self) -> String {
        format!("DiagKey.{}", self.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::BarrierLevel => "BarrierLevel",
            Self::ConversionValue => "ConversionValue",
            Self::CreditSpread => "CreditSpread",
            Self::D => "D",
            Self::D1 => "D1",
            Self::D2 => "D2",
            Self::Delta => "Delta",
            Self::DiscountFactor => "DiscountFactor",
            Self::DoubleKnockoutBase => "DoubleKnockoutBase",
            Self::EffectiveVol => "EffectiveVol",
            Self::ExerciseDates => "ExerciseDates",
            Self::FairVariance => "FairVariance",
            Self::FairVolatility => "FairVolatility",
            Self::InsideBarriers => "InsideBarriers",
            Self::Integral => "Integral",
            Self::MaxExercises => "MaxExercises",
            Self::MinExercises => "MinExercises",
            Self::Npv => "Npv",
            Self::NumPaths => "NumPaths",
            Self::NumThreads => "NumThreads",
            Self::ObservationCount => "ObservationCount",
            Self::NumSpaceSteps => "NumSpaceSteps",
            Self::NumSteps => "NumSteps",
            Self::NumTimeSteps => "NumTimeSteps",
            Self::Pd => "Pd",
            Self::Pm => "Pm",
            Self::Pu => "Pu",
            Self::PvForward => "PvForward",
            Self::Rho => "Rho",
            Self::RhoDomestic => "RhoDomestic",
            Self::RhoForeign => "RhoForeign",
            Self::SMax => "SMax",
            Self::SeriesTerms => "SeriesTerms",
            Self::SurvivalDigital => "SurvivalDigital",
            Self::U => "U",
            Self::VarOfVar => "VarOfVar",
            Self::Vol => "Vol",
            Self::VolAdj => "VolAdj",
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Default)]
pub struct Diagnostics {
    inner: CoreDiagnostics,
}

impl Diagnostics {
    pub(crate) fn to_core(&self) -> CoreDiagnostics {
        self.inner.clone()
    }

    pub(crate) fn from_core(value: CoreDiagnostics) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl Diagnostics {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    fn capacity() -> usize {
        CoreDiagnostics::CAPACITY
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn insert_key(&mut self, key: &DiagKey, value: f64) -> Option<f64> {
        self.inner.insert_key(key.to_core(), value)
    }

    fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    fn get(&self, key: &str) -> Option<f64> {
        self.inner.get(key).copied()
    }

    fn items(&self) -> Vec<(String, f64)> {
        self.inner
            .iter()
            .map(|(key, value)| (key.to_string(), *value))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Diagnostics({:?})", self.items())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PricingResult {
    inner: CorePricingResult,
}

impl PricingResult {
    pub(crate) fn from_core(value: CorePricingResult) -> Self {
        Self { inner: value }
    }
}

impl From<CorePricingResult> for PricingResult {
    fn from(value: CorePricingResult) -> Self {
        Self::from_core(value)
    }
}

#[pymethods]
impl PricingResult {
    #[new]
    fn new(
        price: f64,
        stderr: Option<f64>,
        greeks: Option<Greeks>,
        diagnostics: Option<&Diagnostics>,
    ) -> Self {
        Self {
            inner: CorePricingResult {
                price,
                stderr,
                greeks: greeks.map(Greeks::to_core),
                diagnostics: diagnostics.map(Diagnostics::to_core).unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn price(&self) -> f64 {
        self.inner.price
    }

    #[setter]
    fn set_price(&mut self, price: f64) {
        self.inner.price = price;
    }

    #[getter]
    fn stderr(&self) -> Option<f64> {
        self.inner.stderr
    }

    #[setter]
    fn set_stderr(&mut self, stderr: Option<f64>) {
        self.inner.stderr = stderr;
    }

    #[getter]
    fn greeks(&self) -> Option<Greeks> {
        self.inner.greeks.map(Greeks::from_core)
    }

    #[setter]
    fn set_greeks(&mut self, greeks: Option<Greeks>) {
        self.inner.greeks = greeks.map(Greeks::to_core);
    }

    #[getter]
    fn diagnostics(&self) -> Diagnostics {
        Diagnostics::from_core(self.inner.diagnostics.clone())
    }

    #[setter]
    fn set_diagnostics(&mut self, diagnostics: &Diagnostics) {
        self.inner.diagnostics = diagnostics.to_core();
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingResult(price={}, stderr={:?}, greeks={:?}, diagnostics={:?})",
            self.inner.price,
            self.inner.stderr,
            self.inner.greeks,
            self.inner.diagnostics.iter().collect::<Vec<_>>()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub struct PricingError {
    inner: CorePricingError,
}

#[pymethods]
impl PricingError {
    #[staticmethod]
    fn invalid_input(message: String) -> Self {
        Self {
            inner: CorePricingError::InvalidInput(message),
        }
    }

    #[staticmethod]
    fn convergence_failure(message: String) -> Self {
        Self {
            inner: CorePricingError::ConvergenceFailure(message),
        }
    }

    #[staticmethod]
    fn market_data_missing(message: String) -> Self {
        Self {
            inner: CorePricingError::MarketDataMissing(message),
        }
    }

    #[staticmethod]
    fn numerical_error(message: String) -> Self {
        Self {
            inner: CorePricingError::NumericalError(message),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CorePricingError::InvalidInput(_) => "InvalidInput",
            CorePricingError::ConvergenceFailure(_) => "ConvergenceFailure",
            CorePricingError::MarketDataMissing(_) => "MarketDataMissing",
            CorePricingError::NumericalError(_) => "NumericalError",
        }
    }

    #[getter]
    fn message(&self) -> String {
        match &self.inner {
            CorePricingError::InvalidInput(msg)
            | CorePricingError::ConvergenceFailure(msg)
            | CorePricingError::MarketDataMissing(msg)
            | CorePricingError::NumericalError(msg) => msg.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingError(kind={:?}, message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Greeks>()?;
    module.add_class::<PricingResult>()?;
    module.add_class::<PricingError>()?;
    module.add_class::<OptionType>()?;
    module.add_class::<ExerciseStyle>()?;
    module.add_class::<BarrierDirection>()?;
    module.add_class::<BarrierStyle>()?;
    module.add_class::<BarrierSpec>()?;
    module.add_class::<Averaging>()?;
    module.add_class::<StrikeType>()?;
    module.add_class::<AsianSpec>()?;
    module.add_class::<DiagKey>()?;
    module.add_class::<Diagnostics>()?;
    Ok(())
}
