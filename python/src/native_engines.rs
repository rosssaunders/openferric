use crate::core::PricingResult;
use crate::helpers::catch_unwind_py;
use crate::instruments::*;
use crate::market::Market;
use openferric_core::core::PricingEngine as _;
use openferric_core::engines::{analytic, lsm, monte_carlo, numerical, pde, tree};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

macro_rules! engine {
    ($name:ident, $core:ty, {$($methods:tt)*}, [$($instrument:ty => $convert:expr),+ $(,)?]) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name { pub(crate) inner: $core }
        #[pymethods]
        impl $name {
            $($methods)*
            fn price_with_greeks_aad(&self, py: Python<'_>, instrument: &Bound<'_, PyAny>, market: &Market) -> PyResult<PricingResult> {
                let market = market.to_core()?;
                $(if let Ok(value) = instrument.extract::<PyRef<$instrument>>() {
                    let contract = ($convert)(&*value)?;
                    return py.detach(|| {
                        catch_unwind_py(|| self.inner.price_with_greeks_aad(&contract, &market))?
                            .map(Into::into).map_err(|error| PyValueError::new_err(error.to_string()))
                    });
                })+
                Err(PyTypeError::new_err(concat!(stringify!($name), " does not support this instrument type")))
            }
            fn price(&self, py: Python<'_>, instrument: &Bound<'_, PyAny>, market: &Market) -> PyResult<PricingResult> {
                let market = market.to_core()?;
                $(if let Ok(value) = instrument.extract::<PyRef<$instrument>>() {
                    let contract = ($convert)(&*value)?;
                    return py.detach(|| {
                        catch_unwind_py(|| self.inner.price(&contract, &market))?
                            .map(Into::into).map_err(|error| PyValueError::new_err(error.to_string()))
                    });
                })+
                Err(PyTypeError::new_err(concat!(stringify!($name), " does not support this instrument type")))
            }
        }
    }
}

macro_rules! analytic_engine {
    ($name:ident, [$($instrument:ty => $convert:expr),+ $(,)?]) => {
        engine!($name, analytic::$name, {
            #[new]
            fn new() -> Self { Self { inner: analytic::$name::new() } }
        }, [$($instrument => $convert),+]);
    }
}

analytic_engine!(BlackScholesEngine, [VanillaOption => |value: &VanillaOption| value.to_core()]);
analytic_engine!(Black76Engine, [FuturesOption => |value: &FuturesOption| value.to_core()]);
analytic_engine!(GeometricAsianEngine, [AsianOption => |value: &AsianOption| value.to_core()]);
analytic_engine!(BarrierAnalyticEngine, [BarrierOption => |value: &BarrierOption| value.to_core()]);
analytic_engine!(DigitalAnalyticEngine, [
    CashOrNothingOption => |value: &CashOrNothingOption| value.to_core(),
    AssetOrNothingOption => |value: &AssetOrNothingOption| value.to_core(),
    GapOption => |value: &GapOption| value.to_core()
]);
analytic_engine!(GarmanKohlhagenEngine, [FxOption => |value: &FxOption| value.to_core()]);
analytic_engine!(PowerOptionEngine, [PowerOption => |value: &PowerOption| value.to_core()]);
analytic_engine!(RainbowAnalyticEngine, [
    BestOfTwoCallOption => |value: &BestOfTwoCallOption| Ok::<_, PyErr>(value.to_core()),
    WorstOfTwoCallOption => |value: &WorstOfTwoCallOption| Ok::<_, PyErr>(value.to_core()),
    TwoAssetCorrelationOption => |value: &TwoAssetCorrelationOption| value.to_core()
]);
analytic_engine!(VarianceSwapEngine, [
    VarianceSwap => |value: &VarianceSwap| Ok::<_, PyErr>(value.to_core()),
    VolatilitySwap => |value: &VolatilitySwap| Ok::<_, PyErr>(value.to_core())
]);
analytic_engine!(ExoticAnalyticEngine, [
    ExoticOption => |value: &ExoticOption| Ok::<_, PyErr>(value.to_core()),
    LookbackFloatingOption => |value: &LookbackFloatingOption| Ok::<_, PyErr>(openferric_core::instruments::ExoticOption::LookbackFloating(value.to_core())),
    LookbackFixedOption => |value: &LookbackFixedOption| Ok::<_, PyErr>(openferric_core::instruments::ExoticOption::LookbackFixed(value.to_core())),
    ChooserOption => |value: &ChooserOption| Ok::<_, PyErr>(openferric_core::instruments::ExoticOption::Chooser(value.to_core())),
    QuantoOption => |value: &QuantoOption| Ok::<_, PyErr>(openferric_core::instruments::ExoticOption::Quanto(value.to_core())),
    CompoundOption => |value: &CompoundOption| Ok::<_, PyErr>(openferric_core::instruments::ExoticOption::Compound(value.to_core()))
]);

engine!(DoubleBarrierAnalyticEngine, analytic::DoubleBarrierAnalyticEngine, {
    #[new]
    #[pyo3(signature = (series_terms=None))]
    fn new(series_terms: Option<usize>) -> Self {
        let mut inner = analytic::DoubleBarrierAnalyticEngine::new();
        if let Some(terms) = series_terms { inner = inner.with_series_terms(terms); }
        Self { inner }
    }
    #[getter]
    fn series_terms(&self) -> usize { self.inner.series_terms }
    fn with_series_terms(&self, series_terms: usize) -> Self { Self { inner: self.inner.clone().with_series_terms(series_terms) } }
}, [DoubleBarrierOption => |value: &DoubleBarrierOption| value.to_core()]);

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpreadAnalyticMethod {
    Margrabe,
    Kirk,
}

engine!(SpreadAnalyticEngine, analytic::SpreadAnalyticEngine, {
    #[new]
    #[pyo3(signature = (method=SpreadAnalyticMethod::Kirk))]
    fn new(method: SpreadAnalyticMethod) -> Self {
        Self { inner: analytic::SpreadAnalyticEngine::new(match method { SpreadAnalyticMethod::Margrabe => analytic::SpreadAnalyticMethod::Margrabe, SpreadAnalyticMethod::Kirk => analytic::SpreadAnalyticMethod::Kirk }) }
    }
}, [SpreadOption => |value: &SpreadOption| Ok::<_, PyErr>(value.to_core())]);

macro_rules! step_engine {
    ($name:ident, $module:ident, [$($instrument:ty => $convert:expr),+]) => {
        engine!($name, $module::$name, {
            #[new]
            fn new(steps: usize) -> Self { Self { inner: $module::$name::new(steps) } }
            #[getter]
            fn steps(&self) -> usize { self.inner.steps }
        }, [$($instrument => $convert),+]);
    }
}
step_engine!(BinomialTreeEngine, tree, [VanillaOption => |value: &VanillaOption| value.to_core()]);
step_engine!(TrinomialTreeEngine, tree, [VanillaOption => |value: &VanillaOption| value.to_core()]);
engine!(AmericanBinomialEngine, numerical::AmericanBinomialEngine, {
    #[new]
    fn new(steps: usize) -> Self { Self { inner: numerical::AmericanBinomialEngine::new(steps) } }
    #[staticmethod]
    fn with_arena(steps: usize, arena: &crate::math_bindings::PricingArena) -> Self { Self { inner: numerical::AmericanBinomialEngine::with_arena(steps, std::sync::Arc::clone(&arena.inner)) } }
    #[getter]
    fn steps(&self) -> usize { self.inner.steps }
}, [VanillaOption => |value: &VanillaOption| value.to_core()]);
step_engine!(SwingTreeEngine, tree, [SwingOption => |value: &SwingOption| Ok::<_, PyErr>(value.to_core())]);
step_engine!(TwoAssetBinomialEngine, tree, [
    SpreadOption => |value: &SpreadOption| Ok::<_, PyErr>(value.to_core()),
    BestOfTwoCallOption => |value: &BestOfTwoCallOption| Ok::<_, PyErr>(value.to_core()),
    WorstOfTwoCallOption => |value: &WorstOfTwoCallOption| Ok::<_, PyErr>(value.to_core())
]);

engine!(GeneralizedBinomialEngine, tree::GeneralizedBinomialEngine, {
    #[new]
    fn new(steps: usize, cost_of_carry: f64) -> Self { Self { inner: tree::GeneralizedBinomialEngine::new(steps, cost_of_carry) } }
    #[staticmethod]
    fn futures(steps: usize) -> Self { Self { inner: tree::GeneralizedBinomialEngine::futures(steps) } }
    #[staticmethod]
    fn currency(steps: usize, domestic_rate: f64, foreign_rate: f64) -> Self { Self { inner: tree::GeneralizedBinomialEngine::currency(steps, domestic_rate, foreign_rate) } }
    #[getter]
    fn steps(&self) -> usize { self.inner.steps }
    #[getter]
    fn cost_of_carry(&self) -> f64 { self.inner.cost_of_carry }
}, [VanillaOption => |value: &VanillaOption| value.to_core()]);

engine!(ConvertibleBinomialEngine, tree::ConvertibleBinomialEngine, {
    #[new]
    #[pyo3(signature = (credit_spread, steps=None))]
    fn new(credit_spread: f64, steps: Option<usize>) -> Self {
        let mut inner = tree::ConvertibleBinomialEngine::new(credit_spread);
        if let Some(steps) = steps { inner = inner.with_steps(steps); }
        Self { inner }
    }
    #[getter]
    fn steps(&self) -> usize { self.inner.steps }
    #[getter]
    fn credit_spread(&self) -> f64 { self.inner.credit_spread }
    fn with_steps(&self, steps: usize) -> Self { Self { inner: self.inner.clone().with_steps(steps) } }
}, [ConvertibleBond => |value: &ConvertibleBond| Ok::<_, PyErr>(value.to_core())]);

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BermudanSwaptionEngine {
    inner: tree::BermudanSwaptionEngine,
}
#[pymethods]
impl BermudanSwaptionEngine {
    #[getter]
    fn hw_model(&self) -> crate::models::HullWhite {
        crate::models::HullWhite::from_core(self.inner.hw_model.clone())
    }
    #[new]
    fn new(hw_model: &crate::models::HullWhite, steps: usize) -> Self {
        Self {
            inner: tree::BermudanSwaptionEngine::new(hw_model.to_core(), steps),
        }
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }
    fn price(
        &self,
        py: Python<'_>,
        swaption: &crate::rates::Swaption,
        exercise_dates: Vec<f64>,
        curve: &crate::rates::YieldCurve,
    ) -> PyResult<f64> {
        let swaption = swaption.to_core();
        let curve = curve.to_core();
        py.detach(|| catch_unwind_py(|| self.inner.price(&swaption, &exercise_dates, &curve)))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LsmDynamics {
    inner: lsm::LsmDynamics,
}
#[pymethods]
impl LsmDynamics {
    #[staticmethod]
    fn gbm() -> Self {
        Self {
            inner: lsm::LsmDynamics::Gbm,
        }
    }
    #[staticmethod]
    fn local_vol_euler() -> Self {
        Self {
            inner: lsm::LsmDynamics::LocalVolEuler,
        }
    }
    #[staticmethod]
    fn heston_euler(model: &crate::models::Heston) -> Self {
        let model = model.to_core();
        Self {
            inner: lsm::LsmDynamics::HestonEuler {
                kappa: model.kappa,
                theta: model.theta,
                xi: model.xi,
                rho: model.rho,
                v0: model.v0,
            },
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            lsm::LsmDynamics::Gbm => "Gbm",
            lsm::LsmDynamics::LocalVolEuler => "LocalVolEuler",
            lsm::LsmDynamics::HestonEuler { .. } => "HestonEuler",
        }
    }
}

#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone)]
pub struct ExerciseBoundaryPoint {
    time: f64,
    strike: f64,
    boundary_spot: Option<f64>,
    itm_paths: usize,
    exercised_paths: usize,
}
#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone)]
pub struct BermudanLsmOutput {
    result: PricingResult,
    exercise_boundary: Vec<ExerciseBoundaryPoint>,
}

engine!(LongstaffSchwartzEngine, lsm::LongstaffSchwartzEngine, {
    #[new]
    #[pyo3(signature = (num_paths, num_steps, seed=42, dynamics=None))]
    fn new(num_paths: usize, num_steps: usize, seed: u64, dynamics: Option<LsmDynamics>) -> Self {
        let mut inner = lsm::LongstaffSchwartzEngine::new(num_paths, num_steps, seed);
        if let Some(dynamics) = dynamics { inner.dynamics = dynamics.inner; }
        Self { inner }
    }
    #[getter]
    fn num_paths(&self) -> usize { self.inner.num_paths }
    #[getter]
    fn num_steps(&self) -> usize { self.inner.num_steps }
    #[getter]
    fn seed(&self) -> u64 { self.inner.seed }
    #[getter]
    fn dynamics(&self) -> LsmDynamics { LsmDynamics { inner: self.inner.dynamics } }
    fn with_local_vol_dynamics(&self) -> Self { Self { inner: self.inner.clone().with_local_vol_dynamics() } }
    fn with_heston_dynamics(&self, model: &crate::models::Heston) -> Self { Self { inner: self.inner.clone().with_heston_dynamics(model.to_core()) } }
    fn price_bermudan_with_boundary(&self, py: Python<'_>, instrument: &BermudanOption, market: &Market) -> PyResult<BermudanLsmOutput> {
        let instrument = instrument.to_core()?; let market = market.to_core()?;
        let result = py.detach(|| catch_unwind_py(|| self.inner.price_bermudan_with_boundary(&instrument, &market)))?.map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(BermudanLsmOutput { result: result.result.into(), exercise_boundary: result.exercise_boundary.into_iter().map(|point| ExerciseBoundaryPoint { time: point.time, strike: point.strike, boundary_spot: point.boundary_spot, itm_paths: point.itm_paths, exercised_paths: point.exercised_paths }).collect() })
    }
}, [VanillaOption => |value: &VanillaOption| value.to_core(), BermudanOption => |value: &BermudanOption| value.to_core(), BarrierOption => |value: &BarrierOption| value.to_core()]);

engine!(CrankNicolsonEngine, pde::CrankNicolsonEngine, {
    #[new]
    #[pyo3(signature = (time_steps, space_steps, s_max_multiplier=None))]
    fn new(time_steps: usize, space_steps: usize, s_max_multiplier: Option<f64>) -> Self {
        let mut inner = pde::CrankNicolsonEngine::new(time_steps, space_steps);
        if let Some(value) = s_max_multiplier { inner = inner.with_s_max_multiplier(value); }
        Self { inner }
    }
    #[getter]
    fn time_steps(&self) -> usize { self.inner.time_steps }
    #[getter]
    fn space_steps(&self) -> usize { self.inner.space_steps }
    #[getter]
    fn s_max_multiplier(&self) -> f64 { self.inner.s_max_multiplier }
    fn with_s_max_multiplier(&self, s_max_multiplier: f64) -> Self { Self { inner: self.inner.clone().with_s_max_multiplier(s_max_multiplier) } }
    fn price_bermudan_with_boundary(&self, py: Python<'_>, instrument: &BermudanOption, market: &Market) -> PyResult<crate::engines::BermudanPdeOutput> {
        let instrument = instrument.to_core()?; let market = market.to_core()?;
        py.detach(|| catch_unwind_py(|| self.inner.price_bermudan_with_boundary(&instrument, &market))?.map(Into::into).map_err(|error| PyValueError::new_err(error.to_string())))
    }
}, [VanillaOption => |value: &VanillaOption| value.to_core(), BermudanOption => |value: &BermudanOption| value.to_core()]);

macro_rules! fd_engine {
    ($name:ident, {$($extra:ident: $extra_type:ty => $builder:ident),*}) => {
        engine!($name, pde::$name, {
            #[new]
            #[pyo3(signature = (time_steps, space_steps, s_max_multiplier=None, grid_stretch=None, $($extra=None),*))]
            fn new(time_steps: usize, space_steps: usize, s_max_multiplier: Option<f64>, grid_stretch: Option<f64>, $($extra: Option<$extra_type>),*) -> Self {
                let mut inner = pde::$name::new(time_steps, space_steps);
                if let Some(value) = s_max_multiplier { inner = inner.with_s_max_multiplier(value); }
                if let Some(value) = grid_stretch { inner = inner.with_grid_stretch(value); }
                $(if let Some(value) = $extra { inner = inner.$builder(value); })*
                Self { inner }
            }
            #[getter]
            fn time_steps(&self) -> usize { self.inner.time_steps }
            #[getter]
            fn space_steps(&self) -> usize { self.inner.space_steps }
            #[getter]
            fn s_max_multiplier(&self) -> f64 { self.inner.s_max_multiplier }
            #[getter]
            fn grid_stretch(&self) -> f64 { self.inner.grid_stretch }
            fn with_s_max_multiplier(&self, value: f64) -> Self { Self { inner: self.inner.clone().with_s_max_multiplier(value) } }
            fn with_grid_stretch(&self, value: f64) -> Self { Self { inner: self.inner.clone().with_grid_stretch(value) } }
            $(#[getter]
            fn $extra(&self) -> $extra_type { self.inner.$extra }
            fn $builder(&self, value: $extra_type) -> Self { Self { inner: self.inner.clone().$builder(value) } })*
        }, [VanillaOption => |value: &VanillaOption| value.to_core()]);
    }
}
fd_engine!(ExplicitFdEngine, {cfl_safety_factor: f64 => with_cfl_safety_factor, enforce_cfl: bool => with_enforce_cfl});
fd_engine!(ImplicitFdEngine, {});
fd_engine!(HopscotchEngine, {});

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AdiScheme {
    DouglasRachford,
    CraigSneyd,
}
impl AdiScheme {
    fn to_core(self) -> pde::AdiScheme {
        match self {
            Self::DouglasRachford => pde::AdiScheme::DouglasRachford,
            Self::CraigSneyd => pde::AdiScheme::CraigSneyd,
        }
    }
}

engine!(AdiHestonEngine, pde::AdiHestonEngine, {
    #[getter]
    fn model(&self) -> crate::models::Heston { crate::models::Heston::from_core(self.inner.model) }
    #[getter]
    fn scheme(&self) -> AdiScheme { match self.inner.scheme { pde::AdiScheme::DouglasRachford => AdiScheme::DouglasRachford, pde::AdiScheme::CraigSneyd => AdiScheme::CraigSneyd } }
    #[getter]
    fn time_steps(&self) -> usize { self.inner.time_steps }
    #[getter]
    fn spot_steps(&self) -> usize { self.inner.spot_steps }
    #[getter]
    fn variance_steps(&self) -> usize { self.inner.variance_steps }
    #[getter]
    fn s_max_multiplier(&self) -> f64 { self.inner.s_max_multiplier }
    #[getter]
    fn v_max_multiplier(&self) -> f64 { self.inner.v_max_multiplier }
    #[getter]
    fn theta_adi(&self) -> f64 { self.inner.theta_adi }
    #[getter]
    fn enforce_feller(&self) -> bool { self.inner.enforce_feller }
    #[new]
    #[pyo3(signature = (model, time_steps, spot_steps, variance_steps, scheme=None, s_max_multiplier=None, v_max_multiplier=None, theta_adi=None, enforce_feller=None))]
    fn new(model: &crate::models::Heston, time_steps: usize, spot_steps: usize, variance_steps: usize, scheme: Option<AdiScheme>, s_max_multiplier: Option<f64>, v_max_multiplier: Option<f64>, theta_adi: Option<f64>, enforce_feller: Option<bool>) -> Self {
        let mut inner = pde::AdiHestonEngine::new(model.to_core(), time_steps, spot_steps, variance_steps);
        if let Some(value) = scheme { inner = inner.with_scheme(value.to_core()); }
        if let Some(value) = s_max_multiplier { inner = inner.with_s_max_multiplier(value); }
        if let Some(value) = v_max_multiplier { inner = inner.with_v_max_multiplier(value); }
        if let Some(value) = theta_adi { inner = inner.with_theta_adi(value); }
        if let Some(value) = enforce_feller { inner = inner.with_enforce_feller(value); }
        Self { inner }
    }
    fn with_scheme(&self, scheme: AdiScheme) -> Self { Self { inner: self.inner.clone().with_scheme(scheme.to_core()) } }
    fn with_s_max_multiplier(&self, value: f64) -> Self { Self { inner: self.inner.clone().with_s_max_multiplier(value) } }
    fn with_v_max_multiplier(&self, value: f64) -> Self { Self { inner: self.inner.clone().with_v_max_multiplier(value) } }
    fn with_theta_adi(&self, value: f64) -> Self { Self { inner: self.inner.clone().with_theta_adi(value) } }
    fn with_enforce_feller(&self, value: bool) -> Self { Self { inner: self.inner.clone().with_enforce_feller(value) } }
}, [VanillaOption => |value: &VanillaOption| value.to_core()]);

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VarianceReduction {
    None_,
    Antithetic,
    ControlVariate,
}
impl VarianceReduction {
    fn to_core(self) -> monte_carlo::VarianceReduction {
        match self {
            Self::None_ => monte_carlo::VarianceReduction::None,
            Self::Antithetic => monte_carlo::VarianceReduction::Antithetic,
            Self::ControlVariate => monte_carlo::VarianceReduction::ControlVariate,
        }
    }
}

macro_rules! random_engine {
    ($name:ident, [$($count:ident),+], {$($methods:tt)*}, [$($instrument:ty => $convert:expr),+]) => {
        engine!($name, monte_carlo::$name, {
            #[new]
            #[pyo3(signature = ($($count,)+ seed=42))]
            fn new($($count: usize,)+ seed: u64) -> Self { Self { inner: monte_carlo::$name::new($($count,)+ seed) } }
            $(#[getter]
            fn $count(&self) -> usize { self.inner.$count })+
            #[getter]
            fn seed(&self) -> u64 { self.inner.seed }
            fn rng_kind(&self) -> crate::math_bindings::FastRngKind { crate::math_bindings::FastRngKind { inner: self.inner.rng_kind() } }
            fn is_reproducible(&self) -> bool { self.inner.is_reproducible() }
            fn with_rng_kind(&self, kind: crate::math_bindings::FastRngKind) -> Self { Self { inner: self.inner.clone().with_rng_kind(kind.inner) } }
            fn with_seed(&self, seed: u64) -> Self { Self { inner: self.inner.clone().with_seed(seed) } }
            fn with_randomized_streams(&self) -> Self { Self { inner: self.inner.clone().with_randomized_streams() } }
            fn with_thread_rng(&self) -> Self { Self { inner: self.inner.clone().with_thread_rng() } }
            $($methods)*
        }, [$($instrument => $convert),+]);
    }
}

random_engine!(MonteCarloPricingEngine, [num_paths, num_steps], {
    #[getter]
    fn variance_reduction(&self) -> VarianceReduction { match self.inner.variance_reduction { monte_carlo::VarianceReduction::None => VarianceReduction::None_, monte_carlo::VarianceReduction::Antithetic => VarianceReduction::Antithetic, monte_carlo::VarianceReduction::ControlVariate => VarianceReduction::ControlVariate } }
    #[getter]
    fn accuracy_tier(&self) -> Option<crate::numerical::AccuracyTier> { self.inner.accuracy_tier.map(|tier| match tier { openferric_core::math::AccuracyTier::High => crate::numerical::AccuracyTier::High, openferric_core::math::AccuracyTier::Fast => crate::numerical::AccuracyTier::Fast }) }
    #[getter]
    fn execution_policy(&self) -> crate::dsl::ExecutionPolicy { crate::dsl::ExecutionPolicy::from_core(self.inner.execution_policy) }
    fn with_variance_reduction(&self, reduction: VarianceReduction) -> Self { Self { inner: self.inner.clone().with_variance_reduction(reduction.to_core()) } }
    fn with_accuracy_tier(&self, tier: crate::numerical::AccuracyTier) -> Self { Self { inner: self.inner.clone().with_accuracy_tier(tier.to_core()) } }
    fn with_execution_policy(&self, policy: crate::dsl::ExecutionPolicy) -> Self { Self { inner: self.inner.clone().with_execution_policy(policy.to_core()) } }
    fn effective_accuracy_tier(&self) -> crate::numerical::AccuracyTier {
        match self.inner.effective_accuracy_tier() { openferric_core::math::AccuracyTier::High => crate::numerical::AccuracyTier::High, openferric_core::math::AccuracyTier::Fast => crate::numerical::AccuracyTier::Fast }
    }
    fn resolve_execution_backend(&self, instrument: &Bound<'_, PyAny>, market: &Market) -> PyResult<crate::dsl::ExecutionBackend> {
        let market = market.to_core()?;
        let result = if let Ok(value) = instrument.extract::<PyRef<VanillaOption>>() {
            self.inner.resolve_execution_backend(&value.to_core()?, &market)
        } else if let Ok(value) = instrument.extract::<PyRef<AsianOption>>() {
            self.inner.resolve_execution_backend(&value.to_core()?, &market)
        } else if let Ok(value) = instrument.extract::<PyRef<BarrierOption>>() {
            self.inner.resolve_execution_backend(&value.to_core()?, &market)
        } else { return Err(PyTypeError::new_err("unsupported Monte Carlo instrument")); };
        result.map(Into::into).map_err(|error| PyValueError::new_err(error.to_string()))
    }
}, [VanillaOption => |value: &VanillaOption| value.to_core(), AsianOption => |value: &AsianOption| value.to_core(), BarrierOption => |value: &BarrierOption| value.to_core()]);

random_engine!(ArithmeticAsianMC, [paths, steps], {
    #[getter]
    fn control_variate(&self) -> bool { self.inner.control_variate }
    fn with_control_variate(&self, value: bool) -> Self { Self { inner: self.inner.with_control_variate(value) } }
}, [AsianOption => |value: &AsianOption| value.to_core()]);

random_engine!(SpreadMonteCarloEngine, [num_paths], {
    #[getter]
    fn antithetic(&self) -> bool { self.inner.antithetic }
    fn with_antithetic(&self, value: bool) -> Self { Self { inner: self.inner.clone().with_antithetic(value) } }
}, [SpreadOption => |value: &SpreadOption| Ok::<_, PyErr>(value.to_core())]);

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MonteCarloGreeksEngine {
    inner: monte_carlo::MonteCarloGreeksEngine,
}
#[pymethods]
impl MonteCarloGreeksEngine {
    #[getter]
    fn num_paths(&self) -> usize {
        self.inner.num_paths
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn antithetic(&self) -> bool {
        self.inner.antithetic
    }
    #[getter]
    fn spot_bump_rel(&self) -> f64 {
        self.inner.spot_bump_rel
    }
    #[new]
    #[pyo3(signature = (num_paths, seed=42))]
    fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            inner: monte_carlo::MonteCarloGreeksEngine::new(num_paths, seed),
        }
    }
    fn with_antithetic(&self, value: bool) -> Self {
        Self {
            inner: self.inner.clone().with_antithetic(value),
        }
    }
    fn with_spot_bump_rel(&self, value: f64) -> Self {
        Self {
            inner: self.inner.clone().with_spot_bump_rel(value),
        }
    }
    fn rng_kind(&self) -> crate::math_bindings::FastRngKind {
        crate::math_bindings::FastRngKind {
            inner: self.inner.rng_kind(),
        }
    }
    fn is_reproducible(&self) -> bool {
        self.inner.is_reproducible()
    }
    fn with_rng_kind(&self, kind: crate::math_bindings::FastRngKind) -> Self {
        Self {
            inner: self.inner.clone().with_rng_kind(kind.inner),
        }
    }
    fn with_seed(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().with_seed(seed),
        }
    }
    fn with_randomized_streams(&self) -> Self {
        Self {
            inner: self.inner.clone().with_randomized_streams(),
        }
    }
    fn with_thread_rng(&self) -> Self {
        Self {
            inner: self.inner.clone().with_thread_rng(),
        }
    }
    fn estimate_pathwise(
        &self,
        py: Python<'_>,
        instrument: &VanillaOption,
        market: &Market,
    ) -> PyResult<crate::core::Greeks> {
        let instrument = instrument.to_core()?;
        let market = market.to_core()?;
        py.detach(|| {
            catch_unwind_py(|| self.inner.estimate_pathwise(&instrument, &market))?
                .map(crate::core::Greeks::from_core)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }
    fn estimate_likelihood_ratio(
        &self,
        py: Python<'_>,
        instrument: &VanillaOption,
        market: &Market,
    ) -> PyResult<crate::core::Greeks> {
        let instrument = instrument.to_core()?;
        let market = market.to_core()?;
        py.detach(|| {
            catch_unwind_py(|| self.inner.estimate_likelihood_ratio(&instrument, &market))?
                .map(crate::core::Greeks::from_core)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add { ($($name:ty),* $(,)?) => { $(module.add_class::<$name>()?;)* } }
    add!(
        BlackScholesEngine,
        Black76Engine,
        GeometricAsianEngine,
        BarrierAnalyticEngine,
        DigitalAnalyticEngine,
        GarmanKohlhagenEngine,
        PowerOptionEngine,
        RainbowAnalyticEngine,
        VarianceSwapEngine,
        ExoticAnalyticEngine,
        DoubleBarrierAnalyticEngine,
        SpreadAnalyticMethod,
        SpreadAnalyticEngine,
        BinomialTreeEngine,
        TrinomialTreeEngine,
        AmericanBinomialEngine,
        SwingTreeEngine,
        TwoAssetBinomialEngine,
        GeneralizedBinomialEngine,
        ConvertibleBinomialEngine,
        BermudanSwaptionEngine,
        LsmDynamics,
        ExerciseBoundaryPoint,
        BermudanLsmOutput,
        LongstaffSchwartzEngine,
        CrankNicolsonEngine,
        ExplicitFdEngine,
        ImplicitFdEngine,
        HopscotchEngine,
        AdiScheme,
        AdiHestonEngine,
        VarianceReduction,
        MonteCarloPricingEngine,
        ArithmeticAsianMC,
        SpreadMonteCarloEngine,
        MonteCarloGreeksEngine
    );
    Ok(())
}
