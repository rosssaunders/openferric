use openferric_core::credit::SurvivalCurve;
use openferric_core::models::short_rate::{CIR, Vasicek as CoreVasicek};
use openferric_core::rates::YieldCurve;
use openferric_core::risk::{
    FundingRateModel as CoreFundingRateModel, InherentLeverage as CoreInherentLeverage,
    LiquidationPosition as CoreLiquidationPosition, LiquidationRisk as CoreLiquidationRisk,
    LiquidationSimulator as CoreLiquidationSimulator, MarginCalculator as CoreMarginCalculator,
    MarginParams as CoreMarginParams, StressScenario as CoreStressScenario,
    StressTestResult as CoreStressTestResult,
};
use pyo3::prelude::*;

use crate::helpers::catch_unwind_py;

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct MarginParams {
    #[pyo3(get, set)]
    pub initial_margin_ratio: f64,
    #[pyo3(get, set)]
    pub maintenance_margin_ratio: f64,
    #[pyo3(get, set)]
    pub funding_rate_vol: f64,
    #[pyo3(get, set)]
    pub time_to_maturity: f64,
    #[pyo3(get, set)]
    pub tick_size: f64,
}

impl MarginParams {
    fn to_core(self) -> CoreMarginParams {
        CoreMarginParams {
            initial_margin_ratio: self.initial_margin_ratio,
            maintenance_margin_ratio: self.maintenance_margin_ratio,
            funding_rate_vol: self.funding_rate_vol,
            time_to_maturity: self.time_to_maturity,
            tick_size: self.tick_size,
        }
    }

    fn from_core(params: CoreMarginParams) -> Self {
        Self {
            initial_margin_ratio: params.initial_margin_ratio,
            maintenance_margin_ratio: params.maintenance_margin_ratio,
            funding_rate_vol: params.funding_rate_vol,
            time_to_maturity: params.time_to_maturity,
            tick_size: params.tick_size,
        }
    }
}

#[pymethods]
impl MarginParams {
    #[new]
    fn new(
        initial_margin_ratio: f64,
        maintenance_margin_ratio: f64,
        funding_rate_vol: f64,
        time_to_maturity: f64,
        tick_size: f64,
    ) -> Self {
        Self {
            initial_margin_ratio,
            maintenance_margin_ratio,
            funding_rate_vol,
            time_to_maturity,
            tick_size,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MarginParams(initial_margin_ratio={}, maintenance_margin_ratio={}, funding_rate_vol={}, time_to_maturity={}, tick_size={})",
            self.initial_margin_ratio,
            self.maintenance_margin_ratio,
            self.funding_rate_vol,
            self.time_to_maturity,
            self.tick_size
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct MarginCalculator;

#[pymethods]
impl MarginCalculator {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn initial_margin(notional: f64, params: &MarginParams) -> PyResult<f64> {
        catch_unwind_py(|| CoreMarginCalculator::initial_margin(notional, &params.to_core()))
    }

    #[staticmethod]
    fn maintenance_margin(notional: f64, params: &MarginParams) -> PyResult<f64> {
        catch_unwind_py(|| CoreMarginCalculator::maintenance_margin(notional, &params.to_core()))
    }

    #[staticmethod]
    fn health_ratio(
        collateral: f64,
        notional: f64,
        unrealized_pnl: f64,
        params: &MarginParams,
    ) -> PyResult<f64> {
        catch_unwind_py(|| {
            CoreMarginCalculator::health_ratio(
                collateral,
                notional,
                unrealized_pnl,
                &params.to_core(),
            )
        })
    }

    #[staticmethod]
    fn liquidation_rate(
        entry_rate: f64,
        collateral: f64,
        notional: f64,
        params: &MarginParams,
    ) -> PyResult<f64> {
        catch_unwind_py(|| {
            CoreMarginCalculator::liquidation_rate(
                entry_rate,
                collateral,
                notional,
                &params.to_core(),
            )
        })
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct InherentLeverage;

#[pymethods]
impl InherentLeverage {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn leverage(notional: f64, yu_cost: f64) -> PyResult<f64> {
        catch_unwind_py(|| CoreInherentLeverage::leverage(notional, yu_cost))
    }

    #[staticmethod]
    fn leveraged_return(rate_move: f64, leverage: f64) -> PyResult<f64> {
        catch_unwind_py(|| CoreInherentLeverage::leveraged_return(rate_move, leverage))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Vasicek {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl Vasicek {
    fn to_core(self) -> CoreVasicek {
        CoreVasicek {
            a: self.a,
            b: self.b,
            sigma: self.sigma,
        }
    }
}

#[pymethods]
impl Vasicek {
    #[new]
    fn new(a: f64, b: f64, sigma: f64) -> Self {
        Self { a, b, sigma }
    }

    fn __repr__(&self) -> String {
        format!("Vasicek(a={}, b={}, sigma={})", self.a, self.b, self.sigma)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FundingRateModel {
    inner: CoreFundingRateModel,
}

#[pymethods]
impl FundingRateModel {
    #[staticmethod]
    fn vasicek(model: &Vasicek) -> Self {
        Self {
            inner: CoreFundingRateModel::Vasicek(model.to_core()),
        }
    }

    #[staticmethod]
    fn cir(a: f64, b: f64, sigma: f64) -> Self {
        Self {
            inner: CoreFundingRateModel::CIR(CIR { a, b, sigma }),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreFundingRateModel::Vasicek(_) => "vasicek",
            CoreFundingRateModel::CIR(_) => "cir",
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CoreFundingRateModel::Vasicek(model) => format!(
                "FundingRateModel.vasicek(Vasicek(a={}, b={}, sigma={}))",
                model.a, model.b, model.sigma
            ),
            CoreFundingRateModel::CIR(model) => format!(
                "FundingRateModel.cir(a={}, b={}, sigma={})",
                model.a, model.b, model.sigma
            ),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct LiquidationPosition {
    #[pyo3(get, set)]
    pub size: f64,
    #[pyo3(get, set)]
    pub entry_rate: f64,
    #[pyo3(get, set)]
    pub collateral: f64,
    margin_params: MarginParams,
}

impl LiquidationPosition {
    fn to_core(self) -> CoreLiquidationPosition {
        CoreLiquidationPosition {
            size: self.size,
            entry_rate: self.entry_rate,
            collateral: self.collateral,
            margin_params: self.margin_params.to_core(),
        }
    }

    fn from_core(position: CoreLiquidationPosition) -> Self {
        Self {
            size: position.size,
            entry_rate: position.entry_rate,
            collateral: position.collateral,
            margin_params: MarginParams::from_core(position.margin_params),
        }
    }
}

#[pymethods]
impl LiquidationPosition {
    #[new]
    fn new(size: f64, entry_rate: f64, collateral: f64, margin_params: MarginParams) -> Self {
        Self {
            size,
            entry_rate,
            collateral,
            margin_params,
        }
    }

    #[getter]
    fn margin_params(&self) -> MarginParams {
        self.margin_params
    }

    #[setter]
    fn set_margin_params(&mut self, value: MarginParams) {
        self.margin_params = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "LiquidationPosition(size={}, entry_rate={}, collateral={}, margin_params={})",
            self.size,
            self.entry_rate,
            self.collateral,
            self.margin_params.__repr__()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct StressScenario {
    inner: CoreStressScenario,
}

#[pymethods]
impl StressScenario {
    #[staticmethod]
    fn baseline() -> Self {
        Self {
            inner: CoreStressScenario::Baseline,
        }
    }

    #[staticmethod]
    fn liquidation_cascade(vol_multiplier: f64) -> Self {
        Self {
            inner: CoreStressScenario::LiquidationCascade { vol_multiplier },
        }
    }

    #[staticmethod]
    fn mean_shift(shift: f64) -> Self {
        Self {
            inner: CoreStressScenario::MeanShift { shift },
        }
    }

    #[staticmethod]
    fn cascade_suite() -> Vec<Self> {
        CoreStressScenario::cascade_suite()
            .into_iter()
            .map(|scenario| Self { inner: scenario })
            .collect()
    }

    #[staticmethod]
    fn mean_shift_suite(shift: f64) -> PyResult<Vec<Self>> {
        catch_unwind_py(|| {
            CoreStressScenario::mean_shift_suite(shift)
                .into_iter()
                .map(|scenario| Self { inner: scenario })
                .collect()
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreStressScenario::Baseline => "baseline",
            CoreStressScenario::LiquidationCascade { .. } => "liquidation_cascade",
            CoreStressScenario::MeanShift { .. } => "mean_shift",
        }
    }

    #[getter]
    fn vol_multiplier(&self) -> Option<f64> {
        match self.inner {
            CoreStressScenario::LiquidationCascade { vol_multiplier } => Some(vol_multiplier),
            _ => None,
        }
    }

    #[getter]
    fn shift(&self) -> Option<f64> {
        match self.inner {
            CoreStressScenario::MeanShift { shift } => Some(shift),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CoreStressScenario::Baseline => "StressScenario.baseline()".to_string(),
            CoreStressScenario::LiquidationCascade { vol_multiplier } => {
                format!("StressScenario.liquidation_cascade(vol_multiplier={vol_multiplier})")
            }
            CoreStressScenario::MeanShift { shift } => {
                format!("StressScenario.mean_shift(shift={shift})")
            }
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct LiquidationRisk {
    #[pyo3(get, set)]
    pub prob_liquidation: f64,
    #[pyo3(get, set)]
    pub expected_time_to_liquidation: Option<f64>,
    #[pyo3(get, set)]
    pub worst_case_funding_rate: f64,
}

impl LiquidationRisk {
    fn from_core(risk: CoreLiquidationRisk) -> Self {
        Self {
            prob_liquidation: risk.prob_liquidation,
            expected_time_to_liquidation: risk.expected_time_to_liquidation,
            worst_case_funding_rate: risk.worst_case_funding_rate,
        }
    }
}

#[pymethods]
impl LiquidationRisk {
    #[new]
    fn new(
        prob_liquidation: f64,
        expected_time_to_liquidation: Option<f64>,
        worst_case_funding_rate: f64,
    ) -> Self {
        Self {
            prob_liquidation,
            expected_time_to_liquidation,
            worst_case_funding_rate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LiquidationRisk(prob_liquidation={}, expected_time_to_liquidation={:?}, worst_case_funding_rate={})",
            self.prob_liquidation, self.expected_time_to_liquidation, self.worst_case_funding_rate
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct StressTestResult {
    scenario: StressScenario,
    risk: LiquidationRisk,
}

impl StressTestResult {
    fn from_core(result: CoreStressTestResult) -> Self {
        Self {
            scenario: StressScenario {
                inner: result.scenario,
            },
            risk: LiquidationRisk::from_core(result.risk),
        }
    }
}

#[pymethods]
impl StressTestResult {
    #[new]
    fn new(scenario: StressScenario, risk: LiquidationRisk) -> Self {
        Self { scenario, risk }
    }

    #[getter]
    fn scenario(&self) -> StressScenario {
        self.scenario
    }

    #[setter]
    fn set_scenario(&mut self, scenario: StressScenario) {
        self.scenario = scenario;
    }

    #[getter]
    fn risk(&self) -> LiquidationRisk {
        self.risk
    }

    #[setter]
    fn set_risk(&mut self, risk: LiquidationRisk) {
        self.risk = risk;
    }

    fn __repr__(&self) -> String {
        format!(
            "StressTestResult(scenario={}, risk={})",
            self.scenario.__repr__(),
            self.risk.__repr__()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct LiquidationSimulator {
    inner: CoreLiquidationSimulator,
}

#[pymethods]
impl LiquidationSimulator {
    #[new]
    fn new(
        position: &LiquidationPosition,
        model: &FundingRateModel,
        initial_funding_rate: f64,
        num_paths: usize,
        steps: usize,
        seed: u64,
    ) -> PyResult<Self> {
        catch_unwind_py(|| Self {
            inner: CoreLiquidationSimulator::new(
                position.to_core(),
                model.inner,
                initial_funding_rate,
                num_paths,
                steps,
                seed,
            ),
        })
    }

    #[getter]
    fn position(&self) -> LiquidationPosition {
        LiquidationPosition::from_core(self.inner.position)
    }

    #[getter]
    fn model(&self) -> FundingRateModel {
        FundingRateModel {
            inner: self.inner.model,
        }
    }

    #[getter]
    fn initial_funding_rate(&self) -> f64 {
        self.inner.initial_funding_rate
    }

    #[getter]
    fn num_paths(&self) -> usize {
        self.inner.num_paths
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn simulate(&self) -> PyResult<LiquidationRisk> {
        catch_unwind_py(|| LiquidationRisk::from_core(self.inner.simulate()))
    }

    fn simulate_stress(&self, scenario: &StressScenario) -> PyResult<LiquidationRisk> {
        catch_unwind_py(|| LiquidationRisk::from_core(self.inner.simulate_stress(scenario.inner)))
    }

    fn run_stress_scenarios(
        &self,
        py: Python<'_>,
        scenarios: Vec<Py<StressScenario>>,
    ) -> PyResult<Vec<StressTestResult>> {
        let scenarios = scenarios
            .into_iter()
            .map(|scenario| scenario.borrow(py).inner)
            .collect::<Vec<_>>();
        catch_unwind_py(|| {
            self.inner
                .run_stress_scenarios(&scenarios)
                .into_iter()
                .map(StressTestResult::from_core)
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LiquidationSimulator(initial_funding_rate={}, num_paths={}, steps={}, seed={})",
            self.inner.initial_funding_rate,
            self.inner.num_paths,
            self.inner.steps,
            self.inner.seed
        )
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_cva(
    times: Vec<f64>,
    ee_profile: Vec<f64>,
    discount_rate: f64,
    hazard_rate: f64,
    lgd: f64,
) -> f64 {
    use openferric_core::risk::XvaCalculator;
    let discount_curve = YieldCurve::new(
        times
            .iter()
            .map(|t| (*t, (-discount_rate * *t).exp()))
            .collect(),
    );
    let hazards = vec![hazard_rate; times.len()];
    let survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);
    let own_survival = SurvivalCurve::from_piecewise_hazard(&times, &vec![0.0; times.len()]);
    let calc = XvaCalculator::new(discount_curve, survival, own_survival, lgd, 0.0);
    calc.cva_from_expected_exposure(&times, &ee_profile)
}

#[pyfunction]
pub fn py_sa_ccr_ead(
    replacement_cost: f64,
    notional: f64,
    maturity: f64,
    asset_class: &str,
) -> f64 {
    use openferric_core::risk::kva::{SaCcrAssetClass, sa_ccr_ead};
    let ac = match asset_class.to_ascii_lowercase().as_str() {
        "ir" | "interest_rate" => SaCcrAssetClass::InterestRate,
        "fx" | "foreign_exchange" => SaCcrAssetClass::ForeignExchange,
        "credit" => SaCcrAssetClass::Credit,
        "equity" => SaCcrAssetClass::Equity,
        "commodity" => SaCcrAssetClass::Commodity,
        _ => return f64::NAN,
    };
    sa_ccr_ead(replacement_cost, notional, maturity, ac)
}

#[pyfunction]
pub fn py_historical_var(returns: Vec<f64>, confidence: f64) -> f64 {
    use openferric_core::risk::var::historical_var;
    historical_var(&returns, confidence)
}

#[pyfunction]
pub fn py_historical_es(returns: Vec<f64>, confidence: f64) -> f64 {
    use openferric_core::risk::var::historical_expected_shortfall;
    historical_expected_shortfall(&returns, confidence)
}
