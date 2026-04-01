use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use openferric_core::core::{Greeks as CoreGreeks, PricingError as CorePricingError};
use openferric_core::credit::SurvivalCurve as CoreSurvivalCurve;
use openferric_core::market::{
    CreditCurveSnapshot as CoreCreditCurveSnapshot,
    ForwardCurveSnapshot as CoreForwardCurveSnapshot, Market as CoreMarket,
    MarketSnapshot as CoreMarketSnapshot, SampledVolSurface as CoreSampledVolSurface,
    VolSource as CoreVolSource,
};
use openferric_core::math::VarBacktestResult as CoreVarBacktestResult;
use openferric_core::models::short_rate::{CIR, Vasicek as CoreVasicek};
use openferric_core::rates::YieldCurve as CoreYieldCurve;
use openferric_core::risk::fva as core_fva;
use openferric_core::risk::kva as core_kva;
use openferric_core::risk::mva as core_mva;
use openferric_core::risk::portfolio as core_portfolio;
use openferric_core::risk::scenarios as core_scenarios;
use openferric_core::risk::sensitivities as core_sens;
use openferric_core::risk::var as core_var;
use openferric_core::risk::wrong_way_risk as core_wwr;
use openferric_core::risk::xva as core_xva;
use openferric_core::risk::{
    FundingRateModel as CoreFundingRateModel, InherentLeverage as CoreInherentLeverage,
    LiquidationPosition as CoreLiquidationPosition, LiquidationRisk as CoreLiquidationRisk,
    LiquidationSimulator as CoreLiquidationSimulator, MarginCalculator as CoreMarginCalculator,
    MarginParams as CoreMarginParams, StressScenario as CoreStressScenario,
    StressTestResult as CoreStressTestResult,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::helpers::{catch_unwind_py, panic_to_pyerr};

fn pricing_error_to_pyerr(err: CorePricingError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn store_callback_error(slot: &Arc<Mutex<Option<PyErr>>>, err: PyErr) {
    if let Ok(mut guard) = slot.lock()
        && guard.is_none()
    {
        *guard = Some(err);
    }
}

fn take_callback_error(slot: &Arc<Mutex<Option<PyErr>>>) -> Option<PyErr> {
    slot.lock().ok().and_then(|mut guard| guard.take())
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
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
    fn to_core(self) -> CoreGreeks {
        CoreGreeks {
            delta: self.delta,
            gamma: self.gamma,
            vega: self.vega,
            theta: self.theta,
            rho: self.rho,
        }
    }

    fn from_core(greeks: CoreGreeks) -> Self {
        Self {
            delta: greeks.delta,
            gamma: greeks.gamma,
            vega: greeks.vega,
            theta: greeks.theta,
            rho: greeks.rho,
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

#[pyclass(module = "openferric", name = "YieldCurve", from_py_object)]
#[derive(Clone)]
pub struct YieldCurve {
    #[pyo3(get, set)]
    pub tenors: Vec<(f64, f64)>,
}

impl YieldCurve {
    fn to_core(&self) -> CoreYieldCurve {
        CoreYieldCurve::new(self.tenors.clone())
    }

    fn from_core(curve: CoreYieldCurve) -> Self {
        Self {
            tenors: curve.tenors,
        }
    }
}

#[pymethods]
impl YieldCurve {
    #[new]
    fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self { tenors }
    }

    fn discount_factor(&self, t: f64) -> f64 {
        self.to_core().discount_factor(t)
    }

    fn zero_rate(&self, t: f64) -> f64 {
        self.to_core().zero_rate(t)
    }

    fn __repr__(&self) -> String {
        format!("YieldCurve(tenors={:?})", self.tenors)
    }
}

#[pyclass(module = "openferric", name = "SurvivalCurve", from_py_object)]
#[derive(Clone)]
pub struct SurvivalCurve {
    #[pyo3(get, set)]
    pub tenors: Vec<(f64, f64)>,
}

impl SurvivalCurve {
    fn to_core(&self) -> CoreSurvivalCurve {
        CoreSurvivalCurve::new(self.tenors.clone())
    }

    fn from_core(curve: CoreSurvivalCurve) -> Self {
        Self {
            tenors: curve.tenors,
        }
    }
}

#[pymethods]
impl SurvivalCurve {
    #[new]
    fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self { tenors }
    }

    #[staticmethod]
    fn from_piecewise_hazard(tenors: Vec<f64>, hazards: Vec<f64>) -> PyResult<Self> {
        catch_unwind_py(|| {
            Self::from_core(CoreSurvivalCurve::from_piecewise_hazard(&tenors, &hazards))
        })
    }

    fn survival_prob(&self, t: f64) -> f64 {
        self.to_core().survival_prob(t)
    }

    fn hazard_rate(&self, t: f64) -> f64 {
        self.to_core().hazard_rate(t)
    }

    fn default_prob(&self, t1: f64, t2: f64) -> f64 {
        self.to_core().default_prob(t1, t2)
    }

    fn inverse_survival_prob(&self, p: f64) -> f64 {
        self.to_core().inverse_survival_prob(p)
    }

    fn __repr__(&self) -> String {
        format!("SurvivalCurve(tenors={:?})", self.tenors)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct KupiecBacktestResult {
    #[pyo3(get, set)]
    pub exceptions: usize,
    #[pyo3(get, set)]
    pub expected_exceptions: f64,
    #[pyo3(get, set)]
    pub lr_statistic: f64,
    #[pyo3(get, set)]
    pub p_value: f64,
}

impl KupiecBacktestResult {
    fn from_core(result: openferric_core::math::KupiecBacktestResult) -> Self {
        Self {
            exceptions: result.exceptions,
            expected_exceptions: result.expected_exceptions,
            lr_statistic: result.lr_statistic,
            p_value: result.p_value,
        }
    }
}

#[pymethods]
impl KupiecBacktestResult {
    #[new]
    fn new(exceptions: usize, expected_exceptions: f64, lr_statistic: f64, p_value: f64) -> Self {
        Self {
            exceptions,
            expected_exceptions,
            lr_statistic,
            p_value,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct ChristoffersenBacktestResult {
    #[pyo3(get, set)]
    pub n00: usize,
    #[pyo3(get, set)]
    pub n01: usize,
    #[pyo3(get, set)]
    pub n10: usize,
    #[pyo3(get, set)]
    pub n11: usize,
    #[pyo3(get, set)]
    pub lr_independence: f64,
    #[pyo3(get, set)]
    pub lr_conditional_coverage: f64,
    #[pyo3(get, set)]
    pub p_value_independence: f64,
    #[pyo3(get, set)]
    pub p_value_conditional_coverage: f64,
}

impl ChristoffersenBacktestResult {
    fn from_core(result: openferric_core::math::ChristoffersenBacktestResult) -> Self {
        Self {
            n00: result.n00,
            n01: result.n01,
            n10: result.n10,
            n11: result.n11,
            lr_independence: result.lr_independence,
            lr_conditional_coverage: result.lr_conditional_coverage,
            p_value_independence: result.p_value_independence,
            p_value_conditional_coverage: result.p_value_conditional_coverage,
        }
    }
}

#[pymethods]
impl ChristoffersenBacktestResult {
    #[new]
    fn new(
        n00: usize,
        n01: usize,
        n10: usize,
        n11: usize,
        lr_independence: f64,
        lr_conditional_coverage: f64,
        p_value_independence: f64,
        p_value_conditional_coverage: f64,
    ) -> Self {
        Self {
            n00,
            n01,
            n10,
            n11,
            lr_independence,
            lr_conditional_coverage,
            p_value_independence,
            p_value_conditional_coverage,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct VarBacktestResult {
    kupiec: KupiecBacktestResult,
    christoffersen: ChristoffersenBacktestResult,
    #[pyo3(get, set)]
    pub exception_rate: f64,
}

impl VarBacktestResult {
    fn from_core(result: CoreVarBacktestResult) -> Self {
        Self {
            kupiec: KupiecBacktestResult::from_core(result.kupiec),
            christoffersen: ChristoffersenBacktestResult::from_core(result.christoffersen),
            exception_rate: result.exception_rate,
        }
    }
}

#[pymethods]
impl VarBacktestResult {
    #[new]
    fn new(
        kupiec: KupiecBacktestResult,
        christoffersen: ChristoffersenBacktestResult,
        exception_rate: f64,
    ) -> Self {
        Self {
            kupiec,
            christoffersen,
            exception_rate,
        }
    }

    #[getter]
    fn kupiec(&self) -> KupiecBacktestResult {
        self.kupiec
    }

    #[getter]
    fn christoffersen(&self) -> ChristoffersenBacktestResult {
        self.christoffersen
    }
}

#[pyclass(module = "openferric", name = "SampledVolSurface", from_py_object)]
#[derive(Clone)]
pub struct SampledVolSurface {
    inner: CoreSampledVolSurface,
}

impl SampledVolSurface {
    fn to_core(&self) -> CoreSampledVolSurface {
        self.inner.clone()
    }

    fn from_core(surface: CoreSampledVolSurface) -> Self {
        Self { inner: surface }
    }
}

#[pymethods]
impl SampledVolSurface {
    #[new]
    fn new(strikes: Vec<f64>, expiries: Vec<f64>, vols: Vec<Vec<f64>>) -> PyResult<Self> {
        CoreSampledVolSurface::new(strikes, expiries, vols)
            .map(Self::from_core)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn strikes(&self) -> Vec<f64> {
        self.inner.strikes.clone()
    }

    #[getter]
    fn expiries(&self) -> Vec<f64> {
        self.inner.expiries.clone()
    }

    #[getter]
    fn vols(&self) -> Vec<Vec<f64>> {
        self.inner.vols.clone()
    }

    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.vol(strike, expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VolSource {
    inner: CoreVolSource,
}

impl VolSource {
    fn to_core(&self) -> CoreVolSource {
        self.inner.clone()
    }

    fn from_core(source: CoreVolSource, spot: f64) -> Self {
        let inner = match source {
            CoreVolSource::Flat(vol) => CoreVolSource::Flat(vol),
            CoreVolSource::Sampled(surface) => CoreVolSource::Sampled(surface),
            CoreVolSource::Parametric(surface) => {
                CoreVolSource::Sampled(CoreSampledVolSurface::from_surface(&surface, spot))
            }
        };
        Self { inner }
    }
}

#[pymethods]
impl VolSource {
    #[staticmethod]
    fn flat(vol: f64) -> Self {
        Self {
            inner: CoreVolSource::Flat(vol),
        }
    }

    #[staticmethod]
    fn sampled(surface: &SampledVolSurface) -> Self {
        Self {
            inner: CoreVolSource::Sampled(surface.to_core()),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreVolSource::Flat(_) => "flat",
            CoreVolSource::Parametric(_) => "parametric",
            CoreVolSource::Sampled(_) => "sampled",
        }
    }

    #[getter]
    fn flat_vol(&self) -> Option<f64> {
        match self.inner {
            CoreVolSource::Flat(vol) => Some(vol),
            _ => None,
        }
    }

    #[getter]
    fn sampled_surface(&self) -> Option<SampledVolSurface> {
        match &self.inner {
            CoreVolSource::Sampled(surface) => Some(SampledVolSurface::from_core(surface.clone())),
            _ => None,
        }
    }

    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.vol(strike, expiry)
    }
}

#[pyclass(module = "openferric", name = "Market", from_py_object)]
#[derive(Clone)]
pub struct Market {
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub rate: f64,
    #[pyo3(get, set)]
    pub dividend_yield: f64,
    vol: VolSource,
    #[pyo3(get, set)]
    pub reference_date: Option<String>,
}

impl Market {
    fn to_core(&self) -> CoreMarket {
        CoreMarket {
            spot: self.spot,
            rate: self.rate,
            dividend_yield: self.dividend_yield,
            dividend_schedule: Default::default(),
            vol: self.vol.to_core(),
            reference_date: self.reference_date.clone(),
        }
    }

    fn from_core(market: CoreMarket) -> Self {
        let spot = market.spot;
        Self {
            spot,
            rate: market.rate,
            dividend_yield: market.dividend_yield,
            vol: VolSource::from_core(market.vol, spot),
            reference_date: market.reference_date,
        }
    }
}

#[pymethods]
impl Market {
    #[new]
    fn new(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: VolSource,
        reference_date: Option<String>,
    ) -> Self {
        Self {
            spot,
            rate,
            dividend_yield,
            vol,
            reference_date,
        }
    }

    #[getter]
    fn vol_source(&self) -> VolSource {
        self.vol.clone()
    }

    #[setter]
    fn set_vol_source(&mut self, value: VolSource) {
        self.vol = value;
    }

    fn vol_for(&self, strike: f64, expiry: f64) -> f64 {
        self.to_core().vol_for(strike, expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ForwardCurveSnapshot {
    #[pyo3(get, set)]
    pub asset_id: String,
    #[pyo3(get, set)]
    pub points: Vec<(f64, f64)>,
}

impl ForwardCurveSnapshot {
    fn to_core(&self) -> CoreForwardCurveSnapshot {
        CoreForwardCurveSnapshot {
            asset_id: self.asset_id.clone(),
            points: self.points.clone(),
        }
    }

    fn from_core(snapshot: CoreForwardCurveSnapshot) -> Self {
        Self {
            asset_id: snapshot.asset_id,
            points: snapshot.points,
        }
    }
}

#[pymethods]
impl ForwardCurveSnapshot {
    #[new]
    fn new(asset_id: String, points: Vec<(f64, f64)>) -> Self {
        Self { asset_id, points }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CreditCurveSnapshot {
    #[pyo3(get, set)]
    pub curve_id: String,
    survival_curve: SurvivalCurve,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
}

impl CreditCurveSnapshot {
    fn to_core(&self) -> CoreCreditCurveSnapshot {
        CoreCreditCurveSnapshot {
            curve_id: self.curve_id.clone(),
            survival_curve: self.survival_curve.to_core(),
            recovery_rate: self.recovery_rate,
        }
    }

    fn from_core(snapshot: CoreCreditCurveSnapshot) -> Self {
        Self {
            curve_id: snapshot.curve_id,
            survival_curve: SurvivalCurve::from_core(snapshot.survival_curve),
            recovery_rate: snapshot.recovery_rate,
        }
    }
}

#[pymethods]
impl CreditCurveSnapshot {
    #[new]
    fn new(curve_id: String, survival_curve: SurvivalCurve, recovery_rate: f64) -> Self {
        Self {
            curve_id,
            survival_curve,
            recovery_rate,
        }
    }

    #[getter]
    fn survival_curve(&self) -> SurvivalCurve {
        self.survival_curve.clone()
    }

    #[setter]
    fn set_survival_curve(&mut self, value: SurvivalCurve) {
        self.survival_curve = value;
    }
}

#[pyclass(module = "openferric", name = "MarketSnapshot", from_py_object)]
#[derive(Clone)]
pub struct MarketSnapshot {
    #[pyo3(get, set)]
    pub snapshot_id: String,
    #[pyo3(get, set)]
    pub timestamp_unix_ms: i64,
    #[pyo3(get, set)]
    pub markets: Vec<(String, Market)>,
    #[pyo3(get, set)]
    pub yield_curves: Vec<(String, YieldCurve)>,
    #[pyo3(get, set)]
    pub credit_curves: Vec<CreditCurveSnapshot>,
    #[pyo3(get, set)]
    pub spot_prices: Vec<(String, f64)>,
    #[pyo3(get, set)]
    pub forward_curves: Vec<ForwardCurveSnapshot>,
}

impl MarketSnapshot {
    fn to_core(&self) -> CoreMarketSnapshot {
        CoreMarketSnapshot {
            snapshot_id: self.snapshot_id.clone(),
            timestamp_unix_ms: self.timestamp_unix_ms,
            markets: self
                .markets
                .iter()
                .map(|(id, market)| (id.clone(), market.to_core()))
                .collect(),
            yield_curves: self
                .yield_curves
                .iter()
                .map(|(id, curve)| (id.clone(), curve.to_core()))
                .collect(),
            vol_surfaces: Vec::new(),
            credit_curves: self
                .credit_curves
                .iter()
                .map(CreditCurveSnapshot::to_core)
                .collect(),
            spot_prices: self.spot_prices.clone(),
            forward_curves: self
                .forward_curves
                .iter()
                .map(ForwardCurveSnapshot::to_core)
                .collect(),
        }
    }

    fn from_core(snapshot: CoreMarketSnapshot) -> Self {
        Self {
            snapshot_id: snapshot.snapshot_id,
            timestamp_unix_ms: snapshot.timestamp_unix_ms,
            markets: snapshot
                .markets
                .into_iter()
                .map(|(id, market)| (id, Market::from_core(market)))
                .collect(),
            yield_curves: snapshot
                .yield_curves
                .into_iter()
                .map(|(id, curve)| (id, YieldCurve::from_core(curve)))
                .collect(),
            credit_curves: snapshot
                .credit_curves
                .into_iter()
                .map(CreditCurveSnapshot::from_core)
                .collect(),
            spot_prices: snapshot.spot_prices,
            forward_curves: snapshot
                .forward_curves
                .into_iter()
                .map(ForwardCurveSnapshot::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl MarketSnapshot {
    #[new]
    fn new(
        snapshot_id: String,
        timestamp_unix_ms: i64,
        markets: Vec<(String, Market)>,
        yield_curves: Vec<(String, YieldCurve)>,
        credit_curves: Vec<CreditCurveSnapshot>,
        spot_prices: Vec<(String, f64)>,
        forward_curves: Vec<ForwardCurveSnapshot>,
    ) -> Self {
        Self {
            snapshot_id,
            timestamp_unix_ms,
            markets,
            yield_curves,
            credit_curves,
            spot_prices,
            forward_curves,
        }
    }
}

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
    let discount_curve = CoreYieldCurve::new(
        times
            .iter()
            .map(|t| (*t, (-discount_rate * *t).exp()))
            .collect(),
    );
    let hazards = vec![hazard_rate; times.len()];
    let survival = CoreSurvivalCurve::from_piecewise_hazard(&times, &hazards);
    let own_survival = CoreSurvivalCurve::from_piecewise_hazard(&times, &vec![0.0; times.len()]);
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

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct XvaCalculator {
    inner: core_xva::XvaCalculator,
}

#[pymethods]
impl XvaCalculator {
    #[new]
    fn new(
        discount_curve: &YieldCurve,
        counterparty_survival: &SurvivalCurve,
        own_survival: &SurvivalCurve,
        lgd: f64,
        lgd_own: f64,
    ) -> PyResult<Self> {
        catch_unwind_py(|| Self {
            inner: core_xva::XvaCalculator::new(
                discount_curve.to_core(),
                counterparty_survival.to_core(),
                own_survival.to_core(),
                lgd,
                lgd_own,
            ),
        })
    }

    #[getter]
    fn discount_curve(&self) -> YieldCurve {
        YieldCurve::from_core(self.inner.discount_curve.clone())
    }

    #[getter]
    fn counterparty_survival(&self) -> SurvivalCurve {
        SurvivalCurve::from_core(self.inner.counterparty_survival.clone())
    }

    #[getter]
    fn own_survival(&self) -> SurvivalCurve {
        SurvivalCurve::from_core(self.inner.own_survival.clone())
    }

    #[getter]
    fn lgd(&self) -> f64 {
        self.inner.lgd
    }

    #[getter]
    fn lgd_own(&self) -> f64 {
        self.inner.lgd_own
    }

    #[staticmethod]
    fn expected_exposure_profile(exposure_paths: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        catch_unwind_py(|| core_xva::XvaCalculator::expected_exposure_profile(&exposure_paths))
    }

    #[staticmethod]
    fn negative_expected_exposure_profile(exposure_paths: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        catch_unwind_py(|| {
            core_xva::XvaCalculator::negative_expected_exposure_profile(&exposure_paths)
        })
    }

    fn cva_from_expected_exposure(&self, times: Vec<f64>, ee_profile: Vec<f64>) -> PyResult<f64> {
        catch_unwind_py(|| self.inner.cva_from_expected_exposure(&times, &ee_profile))
    }

    fn dva_from_negative_expected_exposure(
        &self,
        times: Vec<f64>,
        nee_profile: Vec<f64>,
    ) -> PyResult<f64> {
        catch_unwind_py(|| {
            self.inner
                .dva_from_negative_expected_exposure(&times, &nee_profile)
        })
    }

    fn cva_from_paths(&self, times: Vec<f64>, exposure_paths: Vec<Vec<f64>>) -> PyResult<f64> {
        catch_unwind_py(|| self.inner.cva_from_paths(&times, &exposure_paths))
    }

    fn dva_from_paths(&self, times: Vec<f64>, exposure_paths: Vec<Vec<f64>>) -> PyResult<f64> {
        catch_unwind_py(|| self.inner.dva_from_paths(&times, &exposure_paths))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CsaTerms {
    #[pyo3(get, set)]
    pub mta: f64,
    #[pyo3(get, set)]
    pub threshold: f64,
    #[pyo3(get, set)]
    pub margin_period_of_risk: f64,
    #[pyo3(get, set)]
    pub posting_frequency: f64,
    #[pyo3(get, set)]
    pub collateralized: bool,
}

impl CsaTerms {
    fn to_core(self) -> core_fva::CsaTerms {
        core_fva::CsaTerms {
            mta: self.mta,
            threshold: self.threshold,
            margin_period_of_risk: self.margin_period_of_risk,
            posting_frequency: self.posting_frequency,
            collateralized: self.collateralized,
        }
    }

    fn from_core(terms: core_fva::CsaTerms) -> Self {
        Self {
            mta: terms.mta,
            threshold: terms.threshold,
            margin_period_of_risk: terms.margin_period_of_risk,
            posting_frequency: terms.posting_frequency,
            collateralized: terms.collateralized,
        }
    }
}

#[pymethods]
impl CsaTerms {
    #[new]
    fn new(
        mta: f64,
        threshold: f64,
        margin_period_of_risk: f64,
        posting_frequency: f64,
        collateralized: bool,
    ) -> Self {
        Self {
            mta,
            threshold,
            margin_period_of_risk,
            posting_frequency,
            collateralized,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self::from_core(core_fva::CsaTerms::default())
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SimmRiskClass {
    InterestRate,
    CreditQualifying,
    CreditNonQualifying,
    Equity,
    Commodity,
    Fx,
}

impl SimmRiskClass {
    fn to_core(self) -> core_mva::SimmRiskClass {
        match self {
            Self::InterestRate => core_mva::SimmRiskClass::InterestRate,
            Self::CreditQualifying => core_mva::SimmRiskClass::CreditQualifying,
            Self::CreditNonQualifying => core_mva::SimmRiskClass::CreditNonQualifying,
            Self::Equity => core_mva::SimmRiskClass::Equity,
            Self::Commodity => core_mva::SimmRiskClass::Commodity,
            Self::Fx => core_mva::SimmRiskClass::Fx,
        }
    }

    fn from_core(value: core_mva::SimmRiskClass) -> Self {
        match value {
            core_mva::SimmRiskClass::InterestRate => Self::InterestRate,
            core_mva::SimmRiskClass::CreditQualifying => Self::CreditQualifying,
            core_mva::SimmRiskClass::CreditNonQualifying => Self::CreditNonQualifying,
            core_mva::SimmRiskClass::Equity => Self::Equity,
            core_mva::SimmRiskClass::Commodity => Self::Commodity,
            core_mva::SimmRiskClass::Fx => Self::Fx,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SimmMargin {
    #[pyo3(get, set)]
    pub risk_class: SimmRiskClass,
    #[pyo3(get, set)]
    pub sensitivities: Vec<f64>,
    #[pyo3(get, set)]
    pub risk_weights: Vec<f64>,
    #[pyo3(get, set)]
    pub intra_corr: f64,
}

impl SimmMargin {
    fn to_core(&self) -> core_mva::SimmMargin {
        core_mva::SimmMargin {
            risk_class: self.risk_class.to_core(),
            sensitivities: self.sensitivities.clone(),
            risk_weights: self.risk_weights.clone(),
            intra_corr: self.intra_corr,
        }
    }

    fn from_core(value: core_mva::SimmMargin) -> Self {
        Self {
            risk_class: SimmRiskClass::from_core(value.risk_class),
            sensitivities: value.sensitivities,
            risk_weights: value.risk_weights,
            intra_corr: value.intra_corr,
        }
    }
}

#[pymethods]
impl SimmMargin {
    #[new]
    fn new(
        risk_class: SimmRiskClass,
        sensitivities: Vec<f64>,
        risk_weights: Vec<f64>,
        intra_corr: f64,
    ) -> Self {
        Self {
            risk_class,
            sensitivities,
            risk_weights,
            intra_corr,
        }
    }

    fn compute(&self) -> f64 {
        self.to_core().compute()
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SaCcrAssetClass {
    InterestRate,
    ForeignExchange,
    Credit,
    Equity,
    Commodity,
}

impl SaCcrAssetClass {
    fn to_core(self) -> core_kva::SaCcrAssetClass {
        match self {
            Self::InterestRate => core_kva::SaCcrAssetClass::InterestRate,
            Self::ForeignExchange => core_kva::SaCcrAssetClass::ForeignExchange,
            Self::Credit => core_kva::SaCcrAssetClass::Credit,
            Self::Equity => core_kva::SaCcrAssetClass::Equity,
            Self::Commodity => core_kva::SaCcrAssetClass::Commodity,
        }
    }
}

#[pymethods]
impl SaCcrAssetClass {
    fn supervisory_factor(&self) -> f64 {
        self.to_core().supervisory_factor()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct AggregatedGreeks {
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub theta: f64,
}

impl AggregatedGreeks {
    fn from_core(value: core_portfolio::AggregatedGreeks) -> Self {
        Self {
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            theta: value.theta,
        }
    }
}

#[pymethods]
impl AggregatedGreeks {
    #[new]
    fn new(delta: f64, gamma: f64, vega: f64, theta: f64) -> Self {
        Self {
            delta,
            gamma,
            vega,
            theta,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct Position {
    #[pyo3(get, set)]
    pub instrument: String,
    #[pyo3(get, set)]
    pub quantity: f64,
    greeks: Greeks,
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub implied_vol: f64,
}

impl Position {
    fn to_core(&self) -> core_portfolio::Position<String> {
        core_portfolio::Position::new(
            self.instrument.clone(),
            self.quantity,
            self.greeks.to_core(),
            self.spot,
            self.implied_vol,
        )
    }

    fn from_core(position: core_portfolio::Position<String>) -> Self {
        Self {
            instrument: position.instrument,
            quantity: position.quantity,
            greeks: Greeks::from_core(position.greeks),
            spot: position.spot,
            implied_vol: position.implied_vol,
        }
    }
}

#[pymethods]
impl Position {
    #[new]
    fn new(
        instrument: String,
        quantity: f64,
        greeks: Greeks,
        spot: f64,
        implied_vol: f64,
    ) -> PyResult<Self> {
        catch_unwind_py(|| {
            Self::from_core(core_portfolio::Position::new(
                instrument,
                quantity,
                greeks.to_core(),
                spot,
                implied_vol,
            ))
        })
    }

    #[getter]
    fn greeks(&self) -> Greeks {
        self.greeks
    }

    #[setter]
    fn set_greeks(&mut self, value: Greeks) {
        self.greeks = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Default)]
pub struct Portfolio {
    #[pyo3(get, set)]
    pub positions: Vec<Position>,
}

impl Portfolio {
    fn to_core(&self) -> core_portfolio::Portfolio<String> {
        core_portfolio::Portfolio::new(self.positions.iter().map(Position::to_core).collect())
    }

    fn from_core(portfolio: core_portfolio::Portfolio<String>) -> Self {
        Self {
            positions: portfolio
                .positions
                .into_iter()
                .map(Position::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl Portfolio {
    #[new]
    fn new(positions: Vec<Position>) -> Self {
        Self { positions }
    }

    fn add_position(&mut self, position: Position) {
        self.positions.push(position);
    }

    fn total_delta(&self) -> f64 {
        self.to_core().total_delta()
    }

    fn total_gamma(&self) -> f64 {
        self.to_core().total_gamma()
    }

    fn total_vega(&self) -> f64 {
        self.to_core().total_vega()
    }

    fn total_theta(&self) -> f64 {
        self.to_core().total_theta()
    }

    fn aggregate_greeks(&self) -> AggregatedGreeks {
        AggregatedGreeks::from_core(self.to_core().aggregate_greeks())
    }

    fn scenario_pnl(&self, spot_shock_pct: f64, vol_shock_pct: f64) -> f64 {
        self.to_core().scenario_pnl(spot_shock_pct, vol_shock_pct)
    }

    fn scenario_pnl_with_horizon(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        horizon_years: f64,
    ) -> f64 {
        self.to_core()
            .scenario_pnl_with_horizon(spot_shock_pct, vol_shock_pct, horizon_years)
    }
}

#[pyfunction]
pub fn py_delta_normal_var(
    delta: f64,
    annual_volatility: f64,
    confidence: f64,
    horizon_days: f64,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_var::delta_normal_var(delta, annual_volatility, confidence, horizon_days)
    })
}

#[pyfunction]
pub fn py_delta_gamma_normal_var(
    delta: f64,
    gamma: f64,
    annual_volatility: f64,
    confidence: f64,
    horizon_days: f64,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_var::delta_gamma_normal_var(delta, gamma, annual_volatility, confidence, horizon_days)
    })
}

#[pyfunction]
pub fn py_normal_expected_shortfall(
    mean_loss: f64,
    std_dev_loss: f64,
    confidence: f64,
) -> PyResult<f64> {
    catch_unwind_py(|| core_var::normal_expected_shortfall(mean_loss, std_dev_loss, confidence))
}

#[pyfunction]
pub fn py_cornish_fisher_var(
    mean_loss: f64,
    std_dev_loss: f64,
    skewness: f64,
    excess_kurtosis: f64,
    confidence: f64,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_var::cornish_fisher_var(
            mean_loss,
            std_dev_loss,
            skewness,
            excess_kurtosis,
            confidence,
        )
    })
}

#[pyfunction]
pub fn py_cornish_fisher_var_from_pnl(pnl: Vec<f64>, confidence: f64) -> PyResult<f64> {
    catch_unwind_py(|| core_var::cornish_fisher_var_from_pnl(&pnl, confidence))
}

#[pyfunction]
pub fn py_historical_var_from_prices(
    prices: Vec<f64>,
    confidence: f64,
    use_log_returns: bool,
) -> PyResult<f64> {
    catch_unwind_py(|| core_var::historical_var_from_prices(&prices, confidence, use_log_returns))
}

#[pyfunction]
pub fn py_historical_expected_shortfall_from_prices(
    prices: Vec<f64>,
    confidence: f64,
    use_log_returns: bool,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_var::historical_expected_shortfall_from_prices(&prices, confidence, use_log_returns)
    })
}

#[pyfunction]
pub fn py_rolling_historical_var_from_prices(
    prices: Vec<f64>,
    window: usize,
    confidence: f64,
    use_log_returns: bool,
) -> PyResult<Vec<f64>> {
    catch_unwind_py(|| {
        core_var::rolling_historical_var_from_prices(&prices, window, confidence, use_log_returns)
    })
}

#[pyfunction]
pub fn py_backtest_historical_var_from_prices(
    prices: Vec<f64>,
    window: usize,
    confidence: f64,
    use_log_returns: bool,
) -> PyResult<VarBacktestResult> {
    catch_unwind_py(|| {
        VarBacktestResult::from_core(core_var::backtest_historical_var_from_prices(
            &prices,
            window,
            confidence,
            use_log_returns,
        ))
    })
}

#[pyfunction]
pub fn py_funding_exposure_profile(exposure_paths: Vec<Vec<f64>>, csa: &CsaTerms) -> Vec<f64> {
    core_fva::funding_exposure_profile(&exposure_paths, &csa.to_core())
}

#[pyfunction]
pub fn py_fva_from_profile(
    times: Vec<f64>,
    funding_exposure: Vec<f64>,
    funding_spread: Vec<f64>,
    discount_curve: &YieldCurve,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_fva::fva_from_profile(
            &times,
            &funding_exposure,
            &funding_spread,
            &discount_curve.to_core(),
        )
    })
}

#[pyfunction]
pub fn py_mva_from_profile(
    times: Vec<f64>,
    expected_im: Vec<f64>,
    funding_spread: Vec<f64>,
    discount_curve: &YieldCurve,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_mva::mva_from_profile(
            &times,
            &expected_im,
            &funding_spread,
            &discount_curve.to_core(),
        )
    })
}

#[pyfunction]
pub fn py_regulatory_capital(ead: f64, risk_weight: f64) -> f64 {
    core_kva::regulatory_capital(ead, risk_weight)
}

#[pyfunction]
pub fn py_kva_from_profile(
    times: Vec<f64>,
    expected_capital: Vec<f64>,
    hurdle_rate: f64,
    discount_curve: &YieldCurve,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        core_kva::kva_from_profile(
            &times,
            &expected_capital,
            hurdle_rate,
            &discount_curve.to_core(),
        )
    })
}

#[pyfunction]
pub fn py_netting_set_exposure(trade_mtms: Vec<f64>, netting: bool) -> f64 {
    core_kva::netting_set_exposure(&trade_mtms, netting)
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct WwrResult {
    #[pyo3(get, set)]
    pub cva_independent: f64,
    #[pyo3(get, set)]
    pub cva_wwr: f64,
    #[pyo3(get, set)]
    pub wwr_ratio: f64,
}

impl WwrResult {
    fn from_core(result: core_wwr::WwrResult) -> Self {
        Self {
            cva_independent: result.cva_independent,
            cva_wwr: result.cva_wwr,
            wwr_ratio: result.wwr_ratio,
        }
    }
}

#[pymethods]
impl WwrResult {
    #[new]
    fn new(cva_independent: f64, cva_wwr: f64, wwr_ratio: f64) -> Self {
        Self {
            cva_independent,
            cva_wwr,
            wwr_ratio,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct AlphaWWR {
    #[pyo3(get, set)]
    pub alpha: f64,
}

impl AlphaWWR {
    fn to_core(self) -> core_wwr::AlphaWWR {
        core_wwr::AlphaWWR { alpha: self.alpha }
    }
}

#[pymethods]
impl AlphaWWR {
    #[new]
    fn new(alpha: f64) -> PyResult<Self> {
        catch_unwind_py(|| {
            let core = core_wwr::AlphaWWR::new(alpha);
            Self { alpha: core.alpha }
        })
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            alpha: core_wwr::AlphaWWR::default().alpha,
        }
    }

    fn adjust_cva(&self, independent_cva: f64) -> f64 {
        self.to_core().adjust_cva(independent_cva)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CopulaWWR {
    #[pyo3(get, set)]
    pub correlation: f64,
    #[pyo3(get, set)]
    pub num_paths: usize,
    #[pyo3(get, set)]
    pub seed: u64,
}

impl CopulaWWR {
    fn to_core(self) -> core_wwr::CopulaWWR {
        core_wwr::CopulaWWR::new(self.correlation, self.num_paths, self.seed)
    }
}

#[pymethods]
impl CopulaWWR {
    #[new]
    fn new(correlation: f64, num_paths: usize, seed: u64) -> PyResult<Self> {
        catch_unwind_py(|| {
            let core = core_wwr::CopulaWWR::new(correlation, num_paths, seed);
            Self {
                correlation: core.correlation,
                num_paths: core.num_paths,
                seed: core.seed,
            }
        })
    }

    fn cva_with_wwr(
        &self,
        exposure_paths: Vec<Vec<f64>>,
        time_grid: Vec<f64>,
        hazard_rate: f64,
        recovery: f64,
        risk_free_rate: f64,
    ) -> PyResult<WwrResult> {
        catch_unwind_py(|| {
            WwrResult::from_core(self.to_core().cva_with_wwr(
                &exposure_paths,
                &time_grid,
                hazard_rate,
                recovery,
                risk_free_rate,
            ))
        })
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct HullWhiteWWR {
    #[pyo3(get, set)]
    pub base_hazard: f64,
    #[pyo3(get, set)]
    pub beta: f64,
    #[pyo3(get, set)]
    pub num_paths: usize,
    #[pyo3(get, set)]
    pub seed: u64,
}

impl HullWhiteWWR {
    fn to_core(self) -> core_wwr::HullWhiteWWR {
        core_wwr::HullWhiteWWR::new(self.base_hazard, self.beta, self.num_paths, self.seed)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct BumpSize {
    inner: core_sens::BumpSize,
}

impl BumpSize {
    fn to_core(self) -> core_sens::BumpSize {
        self.inner
    }
}

#[pymethods]
impl BumpSize {
    #[staticmethod]
    fn absolute(value: f64) -> Self {
        Self {
            inner: core_sens::BumpSize::Absolute(value),
        }
    }

    #[staticmethod]
    fn relative(value: f64) -> Self {
        Self {
            inner: core_sens::BumpSize::Relative(value),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core_sens::BumpSize::Absolute(_) => "absolute",
            core_sens::BumpSize::Relative(_) => "relative",
        }
    }

    #[getter]
    fn value(&self) -> f64 {
        match self.inner {
            core_sens::BumpSize::Absolute(value) | core_sens::BumpSize::Relative(value) => value,
        }
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DifferencingScheme {
    Forward,
    Central,
}

impl DifferencingScheme {
    fn to_core(self) -> core_sens::DifferencingScheme {
        match self {
            Self::Forward => core_sens::DifferencingScheme::Forward,
            Self::Central => core_sens::DifferencingScheme::Central,
        }
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CurveBumpMode {
    ZeroRate,
    ParRate,
    LogDiscount,
}

impl CurveBumpMode {
    fn to_core(self) -> core_sens::CurveBumpMode {
        match self {
            Self::ZeroRate => core_sens::CurveBumpMode::ZeroRate,
            Self::ParRate => core_sens::CurveBumpMode::ParRate,
            Self::LogDiscount => core_sens::CurveBumpMode::LogDiscount,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CurveBumpConfig {
    bump_size: BumpSize,
    #[pyo3(get, set)]
    pub differencing: DifferencingScheme,
    #[pyo3(get, set)]
    pub mode: CurveBumpMode,
}

impl CurveBumpConfig {
    fn to_core(self) -> core_sens::CurveBumpConfig {
        core_sens::CurveBumpConfig {
            bump_size: self.bump_size.to_core(),
            differencing: self.differencing.to_core(),
            mode: self.mode.to_core(),
        }
    }
}

#[pymethods]
impl CurveBumpConfig {
    #[new]
    fn new(bump_size: BumpSize, differencing: DifferencingScheme, mode: CurveBumpMode) -> Self {
        Self {
            bump_size,
            differencing,
            mode,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            bump_size: BumpSize {
                inner: core_sens::CurveBumpConfig::default().bump_size,
            },
            differencing: DifferencingScheme::Central,
            mode: CurveBumpMode::ZeroRate,
        }
    }

    #[getter]
    fn bump_size(&self) -> BumpSize {
        self.bump_size
    }

    #[setter]
    fn set_bump_size(&mut self, value: BumpSize) {
        self.bump_size = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SurfaceBumpMode {
    inner: core_sens::SurfaceBumpMode,
}

impl SurfaceBumpMode {
    fn to_core(self) -> core_sens::SurfaceBumpMode {
        self.inner
    }
}

#[pymethods]
impl SurfaceBumpMode {
    #[staticmethod]
    fn flat() -> Self {
        Self {
            inner: core_sens::SurfaceBumpMode::Flat,
        }
    }

    #[staticmethod]
    fn per_expiry(expiry_index: usize) -> Self {
        Self {
            inner: core_sens::SurfaceBumpMode::PerExpiry { expiry_index },
        }
    }

    #[staticmethod]
    fn per_strike_expiry(expiry_index: usize, strike_index: usize) -> Self {
        Self {
            inner: core_sens::SurfaceBumpMode::PerStrikeExpiry {
                expiry_index,
                strike_index,
            },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core_sens::SurfaceBumpMode::Flat => "flat",
            core_sens::SurfaceBumpMode::PerExpiry { .. } => "per_expiry",
            core_sens::SurfaceBumpMode::PerStrikeExpiry { .. } => "per_strike_expiry",
        }
    }

    #[getter]
    fn expiry_index(&self) -> Option<usize> {
        match self.inner {
            core_sens::SurfaceBumpMode::PerExpiry { expiry_index }
            | core_sens::SurfaceBumpMode::PerStrikeExpiry { expiry_index, .. } => {
                Some(expiry_index)
            }
            core_sens::SurfaceBumpMode::Flat => None,
        }
    }

    #[getter]
    fn strike_index(&self) -> Option<usize> {
        match self.inner {
            core_sens::SurfaceBumpMode::PerStrikeExpiry { strike_index, .. } => Some(strike_index),
            _ => None,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SurfaceBumpConfig {
    bump_size: BumpSize,
    #[pyo3(get, set)]
    pub differencing: DifferencingScheme,
}

impl SurfaceBumpConfig {
    fn to_core(self) -> core_sens::SurfaceBumpConfig {
        core_sens::SurfaceBumpConfig {
            bump_size: self.bump_size.to_core(),
            differencing: self.differencing.to_core(),
        }
    }
}

#[pymethods]
impl SurfaceBumpConfig {
    #[new]
    fn new(bump_size: BumpSize, differencing: DifferencingScheme) -> Self {
        Self {
            bump_size,
            differencing,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            bump_size: BumpSize {
                inner: core_sens::SurfaceBumpConfig::default().bump_size,
            },
            differencing: DifferencingScheme::Central,
        }
    }

    #[getter]
    fn bump_size(&self) -> BumpSize {
        self.bump_size
    }

    #[setter]
    fn set_bump_size(&mut self, value: BumpSize) {
        self.bump_size = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SpotBumpConfig {
    bump_size: BumpSize,
    #[pyo3(get, set)]
    pub differencing: DifferencingScheme,
}

impl SpotBumpConfig {
    fn to_core(self) -> core_sens::SpotBumpConfig {
        core_sens::SpotBumpConfig {
            bump_size: self.bump_size.to_core(),
            differencing: self.differencing.to_core(),
        }
    }
}

#[pymethods]
impl SpotBumpConfig {
    #[new]
    fn new(bump_size: BumpSize, differencing: DifferencingScheme) -> Self {
        Self {
            bump_size,
            differencing,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self {
            bump_size: BumpSize {
                inner: core_sens::SpotBumpConfig::default().bump_size,
            },
            differencing: DifferencingScheme::Central,
        }
    }

    #[getter]
    fn bump_size(&self) -> BumpSize {
        self.bump_size
    }

    #[setter]
    fn set_bump_size(&mut self, value: BumpSize) {
        self.bump_size = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BucketSensitivity {
    #[pyo3(get, set)]
    pub pillar: f64,
    #[pyo3(get, set)]
    pub bump: f64,
    #[pyo3(get, set)]
    pub value: f64,
}

impl BucketSensitivity {
    fn from_core(value: core_sens::BucketSensitivity) -> Self {
        Self {
            pillar: value.pillar,
            bump: value.bump,
            value: value.value,
        }
    }
}

#[pymethods]
impl BucketSensitivity {
    #[new]
    fn new(pillar: f64, bump: f64, value: f64) -> Self {
        Self {
            pillar,
            bump,
            value,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct KeyRateDurationPoint {
    #[pyo3(get, set)]
    pub pillar: f64,
    #[pyo3(get, set)]
    pub bump: f64,
    #[pyo3(get, set)]
    pub duration: f64,
}

impl KeyRateDurationPoint {
    fn from_core(value: core_sens::KeyRateDurationPoint) -> Self {
        Self {
            pillar: value.pillar,
            bump: value.bump,
            duration: value.duration,
        }
    }
}

#[pymethods]
impl KeyRateDurationPoint {
    #[new]
    fn new(pillar: f64, bump: f64, duration: f64) -> Self {
        Self {
            pillar,
            bump,
            duration,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct GammaLadderPoint {
    #[pyo3(get, set)]
    pub pillar: f64,
    #[pyo3(get, set)]
    pub bump: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
}

impl GammaLadderPoint {
    fn from_core(value: core_sens::GammaLadderPoint) -> Self {
        Self {
            pillar: value.pillar,
            bump: value.bump,
            gamma: value.gamma,
        }
    }
}

#[pymethods]
impl GammaLadderPoint {
    #[new]
    fn new(pillar: f64, bump: f64, gamma: f64) -> Self {
        Self {
            pillar,
            bump,
            gamma,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VegaExpiryPoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub bump: f64,
    #[pyo3(get, set)]
    pub vega: f64,
}

impl VegaExpiryPoint {
    fn from_core(value: core_sens::VegaExpiryPoint) -> Self {
        Self {
            expiry: value.expiry,
            bump: value.bump,
            vega: value.vega,
        }
    }
}

#[pymethods]
impl VegaExpiryPoint {
    #[new]
    fn new(expiry: f64, bump: f64, vega: f64) -> Self {
        Self { expiry, bump, vega }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VegaStrikeExpiryPoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub bump: f64,
    #[pyo3(get, set)]
    pub vega: f64,
}

impl VegaStrikeExpiryPoint {
    fn from_core(value: core_sens::VegaStrikeExpiryPoint) -> Self {
        Self {
            expiry: value.expiry,
            strike: value.strike,
            bump: value.bump,
            vega: value.vega,
        }
    }
}

#[pymethods]
impl VegaStrikeExpiryPoint {
    #[new]
    fn new(expiry: f64, strike: f64, bump: f64, vega: f64) -> Self {
        Self {
            expiry,
            strike,
            bump,
            vega,
        }
    }
}

#[pyclass(module = "openferric", name = "QuoteVolSurface", from_py_object)]
#[derive(Clone)]
pub struct QuoteVolSurface {
    inner: core_sens::QuoteVolSurface,
}

impl QuoteVolSurface {
    fn from_core(surface: core_sens::QuoteVolSurface) -> Self {
        Self { inner: surface }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ChainRuleJacobian {
    #[pyo3(get, set)]
    pub d_pv_d_state: Vec<f64>,
    #[pyo3(get, set)]
    pub d_state_d_quote: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub d_pv_d_quote: Vec<f64>,
}

impl ChainRuleJacobian {
    fn from_core(value: core_sens::ChainRuleJacobian) -> Self {
        Self {
            d_pv_d_state: value.d_pv_d_state,
            d_state_d_quote: value.d_state_d_quote,
            d_pv_d_quote: value.d_pv_d_quote,
        }
    }
}

#[pymethods]
impl ChainRuleJacobian {
    #[new]
    fn new(d_pv_d_state: Vec<f64>, d_state_d_quote: Vec<Vec<f64>>, d_pv_d_quote: Vec<f64>) -> Self {
        Self {
            d_pv_d_state,
            d_state_d_quote,
            d_pv_d_quote,
        }
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegulatoryRiskClass {
    IR,
    FX,
    EQ,
    COMM,
    Credit,
}

impl RegulatoryRiskClass {
    fn to_core(self) -> core_sens::RegulatoryRiskClass {
        match self {
            Self::IR => core_sens::RegulatoryRiskClass::IR,
            Self::FX => core_sens::RegulatoryRiskClass::FX,
            Self::EQ => core_sens::RegulatoryRiskClass::EQ,
            Self::COMM => core_sens::RegulatoryRiskClass::COMM,
            Self::Credit => core_sens::RegulatoryRiskClass::Credit,
        }
    }

    fn from_core(value: core_sens::RegulatoryRiskClass) -> Self {
        match value {
            core_sens::RegulatoryRiskClass::IR => Self::IR,
            core_sens::RegulatoryRiskClass::FX => Self::FX,
            core_sens::RegulatoryRiskClass::EQ => Self::EQ,
            core_sens::RegulatoryRiskClass::COMM => Self::COMM,
            core_sens::RegulatoryRiskClass::Credit => Self::Credit,
        }
    }
}

#[pymethods]
impl RegulatoryRiskClass {
    fn as_str(&self) -> &'static str {
        self.to_core().as_str()
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensitivityMeasure {
    Delta,
    Vega,
    Curvature,
}

impl SensitivityMeasure {
    fn to_core(self) -> core_sens::SensitivityMeasure {
        match self {
            Self::Delta => core_sens::SensitivityMeasure::Delta,
            Self::Vega => core_sens::SensitivityMeasure::Vega,
            Self::Curvature => core_sens::SensitivityMeasure::Curvature,
        }
    }

    fn from_core(value: core_sens::SensitivityMeasure) -> Self {
        match value {
            core_sens::SensitivityMeasure::Delta => Self::Delta,
            core_sens::SensitivityMeasure::Vega => Self::Vega,
            core_sens::SensitivityMeasure::Curvature => Self::Curvature,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SensitivityRecord {
    #[pyo3(get, set)]
    pub portfolio_id: String,
    #[pyo3(get, set)]
    pub trade_id: String,
    #[pyo3(get, set)]
    pub risk_class: RegulatoryRiskClass,
    #[pyo3(get, set)]
    pub measure: SensitivityMeasure,
    #[pyo3(get, set)]
    pub qualifier: String,
    #[pyo3(get, set)]
    pub bucket: String,
    #[pyo3(get, set)]
    pub label1: String,
    #[pyo3(get, set)]
    pub label2: String,
    #[pyo3(get, set)]
    pub amount: f64,
    #[pyo3(get, set)]
    pub amount_currency: String,
}

impl SensitivityRecord {
    fn to_core(&self) -> core_sens::SensitivityRecord {
        core_sens::SensitivityRecord {
            portfolio_id: self.portfolio_id.clone(),
            trade_id: self.trade_id.clone(),
            risk_class: self.risk_class.to_core(),
            measure: self.measure.to_core(),
            qualifier: self.qualifier.clone(),
            bucket: self.bucket.clone(),
            label1: self.label1.clone(),
            label2: self.label2.clone(),
            amount: self.amount,
            amount_currency: self.amount_currency.clone(),
        }
    }

    fn from_core(value: core_sens::SensitivityRecord) -> Self {
        Self {
            portfolio_id: value.portfolio_id,
            trade_id: value.trade_id,
            risk_class: RegulatoryRiskClass::from_core(value.risk_class),
            measure: SensitivityMeasure::from_core(value.measure),
            qualifier: value.qualifier,
            bucket: value.bucket,
            label1: value.label1,
            label2: value.label2,
            amount: value.amount,
            amount_currency: value.amount_currency,
        }
    }
}

#[pymethods]
impl SensitivityRecord {
    #[new]
    fn new(
        portfolio_id: String,
        trade_id: String,
        risk_class: RegulatoryRiskClass,
        measure: SensitivityMeasure,
        qualifier: String,
        bucket: String,
        label1: String,
        label2: String,
        amount: f64,
        amount_currency: String,
    ) -> Self {
        Self {
            portfolio_id,
            trade_id,
            risk_class,
            measure,
            qualifier,
            bucket,
            label1,
            label2,
            amount,
            amount_currency,
        }
    }

    fn to_crif(&self) -> CrifRecord {
        CrifRecord::from_core(self.to_core().to_crif())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CrifRecord {
    #[pyo3(get, set)]
    pub portfolio_id: String,
    #[pyo3(get, set)]
    pub trade_id: String,
    #[pyo3(get, set)]
    pub risk_type: String,
    #[pyo3(get, set)]
    pub qualifier: String,
    #[pyo3(get, set)]
    pub bucket: String,
    #[pyo3(get, set)]
    pub label1: String,
    #[pyo3(get, set)]
    pub label2: String,
    #[pyo3(get, set)]
    pub amount: f64,
    #[pyo3(get, set)]
    pub amount_currency: String,
}

impl CrifRecord {
    fn from_core(value: core_sens::CrifRecord) -> Self {
        Self {
            portfolio_id: value.portfolio_id,
            trade_id: value.trade_id,
            risk_type: value.risk_type,
            qualifier: value.qualifier,
            bucket: value.bucket,
            label1: value.label1,
            label2: value.label2,
            amount: value.amount,
            amount_currency: value.amount_currency,
        }
    }
}

#[pymethods]
impl CrifRecord {
    #[new]
    fn new(
        portfolio_id: String,
        trade_id: String,
        risk_type: String,
        qualifier: String,
        bucket: String,
        label1: String,
        label2: String,
        amount: f64,
        amount_currency: String,
    ) -> Self {
        Self {
            portfolio_id,
            trade_id,
            risk_type,
            qualifier,
            bucket,
            label1,
            label2,
            amount,
            amount_currency,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct RiskClassChargeConfig {
    #[pyo3(get, set)]
    pub risk_class: RegulatoryRiskClass,
    #[pyo3(get, set)]
    pub delta_weight: f64,
    #[pyo3(get, set)]
    pub vega_weight: f64,
    #[pyo3(get, set)]
    pub curvature_weight: f64,
    #[pyo3(get, set)]
    pub intra_bucket_corr: f64,
    #[pyo3(get, set)]
    pub inter_bucket_corr: f64,
    #[pyo3(get, set)]
    pub concentration_threshold: f64,
}

impl RiskClassChargeConfig {
    fn to_core(self) -> core_sens::RiskClassChargeConfig {
        core_sens::RiskClassChargeConfig {
            risk_class: self.risk_class.to_core(),
            delta_weight: self.delta_weight,
            vega_weight: self.vega_weight,
            curvature_weight: self.curvature_weight,
            intra_bucket_corr: self.intra_bucket_corr,
            inter_bucket_corr: self.inter_bucket_corr,
            concentration_threshold: self.concentration_threshold,
        }
    }

    fn from_core(value: core_sens::RiskClassChargeConfig) -> Self {
        Self {
            risk_class: RegulatoryRiskClass::from_core(value.risk_class),
            delta_weight: value.delta_weight,
            vega_weight: value.vega_weight,
            curvature_weight: value.curvature_weight,
            intra_bucket_corr: value.intra_bucket_corr,
            inter_bucket_corr: value.inter_bucket_corr,
            concentration_threshold: value.concentration_threshold,
        }
    }
}

#[pymethods]
impl RiskClassChargeConfig {
    #[new]
    fn new(
        risk_class: RegulatoryRiskClass,
        delta_weight: f64,
        vega_weight: f64,
        curvature_weight: f64,
        intra_bucket_corr: f64,
        inter_bucket_corr: f64,
        concentration_threshold: f64,
    ) -> Self {
        Self {
            risk_class,
            delta_weight,
            vega_weight,
            curvature_weight,
            intra_bucket_corr,
            inter_bucket_corr,
            concentration_threshold,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RiskChargeConfig {
    #[pyo3(get, set)]
    pub class_configs: Vec<RiskClassChargeConfig>,
}

impl RiskChargeConfig {
    fn to_core(&self) -> core_sens::RiskChargeConfig {
        core_sens::RiskChargeConfig {
            class_configs: self
                .class_configs
                .iter()
                .copied()
                .map(RiskClassChargeConfig::to_core)
                .collect(),
        }
    }

    fn from_core(value: core_sens::RiskChargeConfig) -> Self {
        Self {
            class_configs: value
                .class_configs
                .into_iter()
                .map(RiskClassChargeConfig::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl RiskChargeConfig {
    #[new]
    fn new(class_configs: Vec<RiskClassChargeConfig>) -> Self {
        Self { class_configs }
    }

    #[staticmethod]
    fn baseline() -> Self {
        Self::from_core(core_sens::RiskChargeConfig::baseline())
    }

    fn for_class(&self, risk_class: RegulatoryRiskClass) -> Option<RiskClassChargeConfig> {
        self.to_core()
            .for_class(risk_class.to_core())
            .map(RiskClassChargeConfig::from_core)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ClassRiskCharge {
    #[pyo3(get, set)]
    pub risk_class: RegulatoryRiskClass,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub curvature: f64,
    #[pyo3(get, set)]
    pub total: f64,
}

impl ClassRiskCharge {
    fn from_core(value: core_sens::ClassRiskCharge) -> Self {
        Self {
            risk_class: RegulatoryRiskClass::from_core(value.risk_class),
            delta: value.delta,
            vega: value.vega,
            curvature: value.curvature,
            total: value.total,
        }
    }
}

#[pymethods]
impl ClassRiskCharge {
    #[new]
    fn new(
        risk_class: RegulatoryRiskClass,
        delta: f64,
        vega: f64,
        curvature: f64,
        total: f64,
    ) -> Self {
        Self {
            risk_class,
            delta,
            vega,
            curvature,
            total,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RiskChargeSummary {
    #[pyo3(get, set)]
    pub by_class: Vec<ClassRiskCharge>,
    #[pyo3(get, set)]
    pub delta_total: f64,
    #[pyo3(get, set)]
    pub vega_total: f64,
    #[pyo3(get, set)]
    pub curvature_total: f64,
    #[pyo3(get, set)]
    pub total: f64,
}

impl RiskChargeSummary {
    fn from_core(value: core_sens::RiskChargeSummary) -> Self {
        Self {
            by_class: value
                .by_class
                .into_iter()
                .map(ClassRiskCharge::from_core)
                .collect(),
            delta_total: value.delta_total,
            vega_total: value.vega_total,
            curvature_total: value.curvature_total,
            total: value.total,
        }
    }
}

#[pymethods]
impl RiskChargeSummary {
    #[new]
    fn new(
        by_class: Vec<ClassRiskCharge>,
        delta_total: f64,
        vega_total: f64,
        curvature_total: f64,
        total: f64,
    ) -> Self {
        Self {
            by_class,
            delta_total,
            vega_total,
            curvature_total,
            total,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioShock {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub as_of: Option<String>,
    #[pyo3(get, set)]
    pub spot_shock_pct: f64,
    #[pyo3(get, set)]
    pub vol_shock_pct: f64,
    #[pyo3(get, set)]
    pub rate_shock_abs: f64,
    #[pyo3(get, set)]
    pub horizon_years: f64,
}

impl ScenarioShock {
    fn to_core(&self) -> core_sens::ScenarioShock {
        core_sens::ScenarioShock {
            name: self.name.clone(),
            as_of: self.as_of.clone(),
            spot_shock_pct: self.spot_shock_pct,
            vol_shock_pct: self.vol_shock_pct,
            rate_shock_abs: self.rate_shock_abs,
            horizon_years: self.horizon_years,
        }
    }

    fn from_core(value: core_sens::ScenarioShock) -> Self {
        Self {
            name: value.name,
            as_of: value.as_of,
            spot_shock_pct: value.spot_shock_pct,
            vol_shock_pct: value.vol_shock_pct,
            rate_shock_abs: value.rate_shock_abs,
            horizon_years: value.horizon_years,
        }
    }
}

#[pymethods]
impl ScenarioShock {
    #[new]
    fn new(name: String) -> Self {
        Self::from_core(core_sens::ScenarioShock::new(name))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PnlExplain {
    #[pyo3(get, set)]
    pub observed_pnl: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub explained: f64,
    #[pyo3(get, set)]
    pub unexplained: f64,
    #[pyo3(get, set)]
    pub unexplained_ratio: f64,
}

impl PnlExplain {
    fn from_core(value: core_sens::PnlExplain) -> Self {
        Self {
            observed_pnl: value.observed_pnl,
            theta: value.theta,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            rho: value.rho,
            explained: value.explained,
            unexplained: value.unexplained,
            unexplained_ratio: value.unexplained_ratio,
        }
    }
}

#[pymethods]
impl PnlExplain {
    #[new]
    fn new(
        observed_pnl: f64,
        theta: f64,
        delta: f64,
        gamma: f64,
        vega: f64,
        rho: f64,
        explained: f64,
        unexplained: f64,
        unexplained_ratio: f64,
    ) -> Self {
        Self {
            observed_pnl,
            theta,
            delta,
            gamma,
            vega,
            rho,
            explained,
            unexplained,
            unexplained_ratio,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioPnlRow {
    #[pyo3(get, set)]
    pub scenario: String,
    #[pyo3(get, set)]
    pub as_of: Option<String>,
    #[pyo3(get, set)]
    pub pnl: f64,
}

impl ScenarioPnlRow {
    fn from_core(value: core_sens::ScenarioPnlRow) -> Self {
        Self {
            scenario: value.scenario,
            as_of: value.as_of,
            pnl: value.pnl,
        }
    }
}

#[pymethods]
impl ScenarioPnlRow {
    #[new]
    fn new(scenario: String, as_of: Option<String>, pnl: f64) -> Self {
        Self {
            scenario,
            as_of,
            pnl,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TradeRiskContribution {
    #[pyo3(get, set)]
    pub trade_index: usize,
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
    #[pyo3(get, set)]
    pub total: f64,
    #[pyo3(get, set)]
    pub share_of_total: f64,
}

impl TradeRiskContribution {
    fn from_core(value: core_sens::TradeRiskContribution) -> Self {
        Self {
            trade_index: value.trade_index,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            theta: value.theta,
            rho: value.rho,
            total: value.total,
            share_of_total: value.share_of_total,
        }
    }
}

#[pyfunction]
pub fn py_parallel_dv01(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<f64> {
    let curve_core = curve.to_core();
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::parallel_dv01(&curve_core, config.to_core(), |bumped_curve| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((YieldCurve::from_core(bumped_curve.clone()),)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_bucket_dv01(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<Vec<BucketSensitivity>> {
    let curve_core = curve.to_core();
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::bucket_dv01(&curve_core, config.to_core(), |bumped_curve| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((YieldCurve::from_core(bumped_curve.clone()),)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
        .into_iter()
        .map(BucketSensitivity::from_core)
        .collect()
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_key_rate_duration(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<Vec<KeyRateDurationPoint>> {
    let curve_core = curve.to_core();
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::key_rate_duration(&curve_core, config.to_core(), |bumped_curve| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((YieldCurve::from_core(bumped_curve.clone()),)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
        .into_iter()
        .map(KeyRateDurationPoint::from_core)
        .collect()
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_gamma_ladder(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<Vec<GammaLadderPoint>> {
    let curve_core = curve.to_core();
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::gamma_ladder(&curve_core, config.to_core(), |bumped_curve| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((YieldCurve::from_core(bumped_curve.clone()),)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
        .into_iter()
        .map(GammaLadderPoint::from_core)
        .collect()
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_cross_gamma(
    curve: &YieldCurve,
    config: CurveBumpConfig,
    pillar_i: usize,
    pillar_j: usize,
    pricer: Py<PyAny>,
) -> PyResult<f64> {
    let curve_core = curve.to_core();
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::cross_gamma(
            &curve_core,
            config.to_core(),
            pillar_i,
            pillar_j,
            |bumped_curve| {
                Python::attach(|py| {
                    let callable = pricer.bind(py);
                    match callable.call1((YieldCurve::from_core(bumped_curve.clone()),)) {
                        Ok(value) => match value.extract::<f64>() {
                            Ok(result) => result,
                            Err(err) => {
                                store_callback_error(&errors_cb, err);
                                f64::NAN
                            }
                        },
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    }
                })
            },
        )
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_vega_by_expiry_bucket(
    surface: &QuoteVolSurface,
    config: SurfaceBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<Vec<VegaExpiryPoint>> {
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::vega_by_expiry_bucket(&surface.inner, config.to_core(), |bumped_surface| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((QuoteVolSurface::from_core(bumped_surface.clone()),)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
        .into_iter()
        .map(VegaExpiryPoint::from_core)
        .collect()
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_vega_by_strike_expiry_bucket(
    surface: &QuoteVolSurface,
    config: SurfaceBumpConfig,
    pricer: Py<PyAny>,
) -> PyResult<Vec<VegaStrikeExpiryPoint>> {
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::vega_by_strike_expiry_bucket(
            &surface.inner,
            config.to_core(),
            |bumped_surface| {
                Python::attach(|py| {
                    let callable = pricer.bind(py);
                    match callable.call1((QuoteVolSurface::from_core(bumped_surface.clone()),)) {
                        Ok(value) => match value.extract::<f64>() {
                            Ok(result) => result,
                            Err(err) => {
                                store_callback_error(&errors_cb, err);
                                f64::NAN
                            }
                        },
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    }
                })
            },
        )
        .into_iter()
        .map(VegaStrikeExpiryPoint::from_core)
        .collect()
    }))
    .map_err(panic_to_pyerr)?;

    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_fx_delta(spot: f64, config: SpotBumpConfig, pricer: Py<PyAny>) -> PyResult<f64> {
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::fx_delta(spot, config.to_core(), |bumped_spot| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((bumped_spot,)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
    }))
    .map_err(panic_to_pyerr)?;
    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_commodity_delta(spot: f64, config: SpotBumpConfig, pricer: Py<PyAny>) -> PyResult<f64> {
    let errors = Arc::new(Mutex::new(None));
    let errors_cb = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::commodity_delta(spot, config.to_core(), |bumped_spot| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                match callable.call1((bumped_spot,)) {
                    Ok(value) => match value.extract::<f64>() {
                        Ok(result) => result,
                        Err(err) => {
                            store_callback_error(&errors_cb, err);
                            f64::NAN
                        }
                    },
                    Err(err) => {
                        store_callback_error(&errors_cb, err);
                        f64::NAN
                    }
                }
            })
        })
    }))
    .map_err(panic_to_pyerr)?;
    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(result)
}

#[pyfunction]
pub fn py_jacobian_via_bootstrap(
    market_quotes: Vec<f64>,
    bump_size: BumpSize,
    differencing: DifferencingScheme,
    bootstrap: Py<PyAny>,
    pv_from_state: Py<PyAny>,
) -> PyResult<ChainRuleJacobian> {
    let errors = Arc::new(Mutex::new(None));
    let errors_bootstrap = Arc::clone(&errors);
    let errors_pv = Arc::clone(&errors);
    let result = catch_unwind(AssertUnwindSafe(|| {
        core_sens::jacobian_via_bootstrap(
            &market_quotes,
            bump_size.to_core(),
            differencing.to_core(),
            |quotes| {
                Python::attach(|py| {
                    let callable = bootstrap.bind(py);
                    match callable.call1((quotes.to_vec(),)) {
                        Ok(value) => match value.extract::<Vec<f64>>() {
                            Ok(result) => result,
                            Err(err) => {
                                store_callback_error(&errors_bootstrap, err);
                                Vec::new()
                            }
                        },
                        Err(err) => {
                            store_callback_error(&errors_bootstrap, err);
                            Vec::new()
                        }
                    }
                })
            },
            |state| {
                Python::attach(|py| {
                    let callable = pv_from_state.bind(py);
                    match callable.call1((state.to_vec(),)) {
                        Ok(value) => match value.extract::<f64>() {
                            Ok(result) => result,
                            Err(err) => {
                                store_callback_error(&errors_pv, err);
                                f64::NAN
                            }
                        },
                        Err(err) => {
                            store_callback_error(&errors_pv, err);
                            f64::NAN
                        }
                    }
                })
            },
        )
    }))
    .map_err(panic_to_pyerr)?;
    if let Some(err) = take_callback_error(&errors) {
        return Err(err);
    }
    Ok(ChainRuleJacobian::from_core(result))
}

#[pyfunction]
pub fn py_map_risk_class(label: &str) -> RegulatoryRiskClass {
    RegulatoryRiskClass::from_core(core_sens::map_risk_class(label))
}

#[pyfunction]
pub fn py_to_crif_csv(records: Vec<SensitivityRecord>) -> String {
    let core_records = records
        .iter()
        .map(SensitivityRecord::to_core)
        .collect::<Vec<_>>();
    core_sens::to_crif_csv(&core_records)
}

#[pyfunction]
pub fn py_compute_risk_charges(
    sensitivities: Vec<SensitivityRecord>,
    config: &RiskChargeConfig,
) -> RiskChargeSummary {
    let core_sensitivities = sensitivities
        .iter()
        .map(SensitivityRecord::to_core)
        .collect::<Vec<_>>();
    RiskChargeSummary::from_core(core_sens::compute_risk_charges(
        &core_sensitivities,
        &config.to_core(),
    ))
}

#[pyfunction]
pub fn py_pnl_explain(
    portfolio: &Portfolio,
    observed_pnl: f64,
    scenario: &ScenarioShock,
) -> PnlExplain {
    PnlExplain::from_core(core_sens::pnl_explain(
        &portfolio.to_core(),
        observed_pnl,
        &scenario.to_core(),
    ))
}

#[pyfunction]
pub fn py_scenario_pnl_report(
    portfolio: &Portfolio,
    scenarios: Vec<ScenarioShock>,
) -> Vec<ScenarioPnlRow> {
    let core_scenarios = scenarios
        .iter()
        .map(ScenarioShock::to_core)
        .collect::<Vec<_>>();
    core_sens::scenario_pnl_report(&portfolio.to_core(), &core_scenarios)
        .into_iter()
        .map(ScenarioPnlRow::from_core)
        .collect()
}

#[pyfunction]
pub fn py_risk_contribution_per_trade(
    portfolio: &Portfolio,
    scenario: &ScenarioShock,
) -> Vec<TradeRiskContribution> {
    core_sens::risk_contribution_per_trade(&portfolio.to_core(), &scenario.to_core())
        .into_iter()
        .map(TradeRiskContribution::from_core)
        .collect()
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScenarioKind {
    HistoricalReplay,
    Hypothetical,
    ParametricStress2d,
    ReverseStress,
}

impl ScenarioKind {
    fn to_core(self) -> core_scenarios::ScenarioKind {
        match self {
            Self::HistoricalReplay => core_scenarios::ScenarioKind::HistoricalReplay,
            Self::Hypothetical => core_scenarios::ScenarioKind::Hypothetical,
            Self::ParametricStress2d => core_scenarios::ScenarioKind::ParametricStress2d,
            Self::ReverseStress => core_scenarios::ScenarioKind::ReverseStress,
        }
    }

    fn from_core(value: core_scenarios::ScenarioKind) -> Self {
        match value {
            core_scenarios::ScenarioKind::HistoricalReplay => Self::HistoricalReplay,
            core_scenarios::ScenarioKind::Hypothetical => Self::Hypothetical,
            core_scenarios::ScenarioKind::ParametricStress2d => Self::ParametricStress2d,
            core_scenarios::ScenarioKind::ReverseStress => Self::ReverseStress,
        }
    }
}

#[pyclass(module = "openferric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShockFactor {
    Spot,
    Vol,
    Rate,
    CreditSpread,
}

impl ShockFactor {
    fn to_core(self) -> core_scenarios::ShockFactor {
        match self {
            Self::Spot => core_scenarios::ShockFactor::Spot,
            Self::Vol => core_scenarios::ShockFactor::Vol,
            Self::Rate => core_scenarios::ShockFactor::Rate,
            Self::CreditSpread => core_scenarios::ShockFactor::CreditSpread,
        }
    }

    fn from_core(value: core_scenarios::ShockFactor) -> Self {
        match value {
            core_scenarios::ShockFactor::Spot => Self::Spot,
            core_scenarios::ShockFactor::Vol => Self::Vol,
            core_scenarios::ShockFactor::Rate => Self::Rate,
            core_scenarios::ShockFactor::CreditSpread => Self::CreditSpread,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct MarketShock {
    #[pyo3(get, set)]
    pub spot_shock_pct: f64,
    #[pyo3(get, set)]
    pub vol_shock_pct: f64,
    #[pyo3(get, set)]
    pub rate_shock_abs: f64,
    #[pyo3(get, set)]
    pub credit_spread_shock_abs: f64,
    #[pyo3(get, set)]
    pub horizon_years: f64,
}

impl MarketShock {
    fn to_core(self) -> core_scenarios::MarketShock {
        core_scenarios::MarketShock {
            spot_shock_pct: self.spot_shock_pct,
            vol_shock_pct: self.vol_shock_pct,
            rate_shock_abs: self.rate_shock_abs,
            credit_spread_shock_abs: self.credit_spread_shock_abs,
            horizon_years: self.horizon_years,
        }
    }

    fn from_core(value: core_scenarios::MarketShock) -> Self {
        Self {
            spot_shock_pct: value.spot_shock_pct,
            vol_shock_pct: value.vol_shock_pct,
            rate_shock_abs: value.rate_shock_abs,
            credit_spread_shock_abs: value.credit_spread_shock_abs,
            horizon_years: value.horizon_years,
        }
    }
}

#[pymethods]
impl MarketShock {
    #[new]
    fn new(
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        rate_shock_abs: f64,
        credit_spread_shock_abs: f64,
        horizon_years: f64,
    ) -> Self {
        Self {
            spot_shock_pct,
            vol_shock_pct,
            rate_shock_abs,
            credit_spread_shock_abs,
            horizon_years,
        }
    }

    fn scaled(&self, multiplier: f64) -> Self {
        Self::from_core(self.to_core().scaled(multiplier))
    }

    fn add_factor(&mut self, factor: ShockFactor, amount: f64) {
        let mut shock = self.to_core();
        shock.add_factor(factor.to_core(), amount);
        *self = Self::from_core(shock);
    }

    fn factor_value(&self, factor: ShockFactor) -> f64 {
        self.to_core().factor_value(factor.to_core())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HistoricalReplayDefinition {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub replay_date: Option<String>,
    shock: MarketShock,
}

impl HistoricalReplayDefinition {
    fn to_core(&self) -> core_scenarios::HistoricalReplayDefinition {
        core_scenarios::HistoricalReplayDefinition {
            scenario_id: self.scenario_id.clone(),
            replay_date: self.replay_date.clone(),
            shock: self.shock.to_core(),
        }
    }

    fn from_core(value: core_scenarios::HistoricalReplayDefinition) -> Self {
        Self {
            scenario_id: value.scenario_id,
            replay_date: value.replay_date,
            shock: MarketShock::from_core(value.shock),
        }
    }
}

#[pymethods]
impl HistoricalReplayDefinition {
    #[new]
    fn new(scenario_id: String, replay_date: Option<String>, shock: MarketShock) -> Self {
        Self {
            scenario_id,
            replay_date,
            shock,
        }
    }

    #[getter]
    fn shock(&self) -> MarketShock {
        self.shock
    }

    #[setter]
    fn set_shock(&mut self, value: MarketShock) {
        self.shock = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HypotheticalScenarioDefinition {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub description: Option<String>,
    shock: MarketShock,
}

impl HypotheticalScenarioDefinition {
    fn to_core(&self) -> core_scenarios::HypotheticalScenarioDefinition {
        core_scenarios::HypotheticalScenarioDefinition {
            scenario_id: self.scenario_id.clone(),
            description: self.description.clone(),
            shock: self.shock.to_core(),
        }
    }

    fn from_core(value: core_scenarios::HypotheticalScenarioDefinition) -> Self {
        Self {
            scenario_id: value.scenario_id,
            description: value.description,
            shock: MarketShock::from_core(value.shock),
        }
    }
}

#[pymethods]
impl HypotheticalScenarioDefinition {
    #[new]
    fn new(scenario_id: String, description: Option<String>, shock: MarketShock) -> Self {
        Self {
            scenario_id,
            description,
            shock,
        }
    }

    #[getter]
    fn shock(&self) -> MarketShock {
        self.shock
    }

    #[setter]
    fn set_shock(&mut self, value: MarketShock) {
        self.shock = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct StressAxis {
    #[pyo3(get, set)]
    pub factor: ShockFactor,
    #[pyo3(get, set)]
    pub shocks: Vec<f64>,
}

impl StressAxis {
    fn to_core(&self) -> core_scenarios::StressAxis {
        core_scenarios::StressAxis {
            factor: self.factor.to_core(),
            shocks: self.shocks.clone(),
        }
    }

    fn from_core(value: core_scenarios::StressAxis) -> Self {
        Self {
            factor: ShockFactor::from_core(value.factor),
            shocks: value.shocks,
        }
    }
}

#[pymethods]
impl StressAxis {
    #[new]
    fn new(factor: ShockFactor, shocks: Vec<f64>) -> Self {
        Self { factor, shocks }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ParametricStress2dDefinition {
    #[pyo3(get, set)]
    pub scenario_id: String,
    x_axis: StressAxis,
    y_axis: StressAxis,
    base_shock: MarketShock,
}

impl ParametricStress2dDefinition {
    fn to_core(&self) -> core_scenarios::ParametricStress2dDefinition {
        core_scenarios::ParametricStress2dDefinition {
            scenario_id: self.scenario_id.clone(),
            x_axis: self.x_axis.to_core(),
            y_axis: self.y_axis.to_core(),
            base_shock: self.base_shock.to_core(),
        }
    }

    fn from_core(value: core_scenarios::ParametricStress2dDefinition) -> Self {
        Self {
            scenario_id: value.scenario_id,
            x_axis: StressAxis::from_core(value.x_axis),
            y_axis: StressAxis::from_core(value.y_axis),
            base_shock: MarketShock::from_core(value.base_shock),
        }
    }
}

#[pymethods]
impl ParametricStress2dDefinition {
    #[new]
    fn new(
        scenario_id: String,
        x_axis: StressAxis,
        y_axis: StressAxis,
        base_shock: Option<MarketShock>,
    ) -> Self {
        Self {
            scenario_id,
            x_axis,
            y_axis,
            base_shock: base_shock.unwrap_or_default(),
        }
    }

    #[getter]
    fn x_axis(&self) -> StressAxis {
        self.x_axis.clone()
    }

    #[setter]
    fn set_x_axis(&mut self, value: StressAxis) {
        self.x_axis = value;
    }

    #[getter]
    fn y_axis(&self) -> StressAxis {
        self.y_axis.clone()
    }

    #[setter]
    fn set_y_axis(&mut self, value: StressAxis) {
        self.y_axis = value;
    }

    #[getter]
    fn base_shock(&self) -> MarketShock {
        self.base_shock
    }

    #[setter]
    fn set_base_shock(&mut self, value: MarketShock) {
        self.base_shock = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ReverseStressDefinition {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub target_loss: f64,
    seed_shock: MarketShock,
    #[pyo3(get, set)]
    pub max_scale: f64,
    #[pyo3(get, set)]
    pub tolerance: f64,
    #[pyo3(get, set)]
    pub max_iterations: u32,
}

impl ReverseStressDefinition {
    fn to_core(&self) -> core_scenarios::ReverseStressDefinition {
        core_scenarios::ReverseStressDefinition {
            scenario_id: self.scenario_id.clone(),
            target_loss: self.target_loss,
            seed_shock: self.seed_shock.to_core(),
            max_scale: self.max_scale,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
        }
    }

    fn from_core(value: core_scenarios::ReverseStressDefinition) -> Self {
        Self {
            scenario_id: value.scenario_id,
            target_loss: value.target_loss,
            seed_shock: MarketShock::from_core(value.seed_shock),
            max_scale: value.max_scale,
            tolerance: value.tolerance,
            max_iterations: value.max_iterations,
        }
    }
}

#[pymethods]
impl ReverseStressDefinition {
    #[new]
    fn new(
        scenario_id: String,
        target_loss: f64,
        seed_shock: MarketShock,
        max_scale: Option<f64>,
        tolerance: Option<f64>,
        max_iterations: Option<u32>,
    ) -> Self {
        let core = core_scenarios::ReverseStressDefinition {
            scenario_id,
            target_loss,
            seed_shock: seed_shock.to_core(),
            max_scale: max_scale.unwrap_or(10.0),
            tolerance: tolerance.unwrap_or(1.0e-4),
            max_iterations: max_iterations.unwrap_or(64),
        };
        Self::from_core(core)
    }

    #[getter]
    fn seed_shock(&self) -> MarketShock {
        self.seed_shock
    }

    #[setter]
    fn set_seed_shock(&mut self, value: MarketShock) {
        self.seed_shock = value;
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioDefinition {
    inner: core_scenarios::ScenarioDefinition,
}

impl ScenarioDefinition {
    fn to_core(&self) -> core_scenarios::ScenarioDefinition {
        self.inner.clone()
    }

    fn from_core(value: core_scenarios::ScenarioDefinition) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl ScenarioDefinition {
    #[staticmethod]
    fn historical_replay(definition: &HistoricalReplayDefinition) -> Self {
        Self::from_core(core_scenarios::ScenarioDefinition::HistoricalReplay(
            definition.to_core(),
        ))
    }

    #[staticmethod]
    fn hypothetical(definition: &HypotheticalScenarioDefinition) -> Self {
        Self::from_core(core_scenarios::ScenarioDefinition::Hypothetical(
            definition.to_core(),
        ))
    }

    #[staticmethod]
    fn parametric_stress_2d(definition: &ParametricStress2dDefinition) -> Self {
        Self::from_core(core_scenarios::ScenarioDefinition::ParametricStress2d(
            definition.to_core(),
        ))
    }

    #[staticmethod]
    fn reverse_stress(definition: &ReverseStressDefinition) -> Self {
        Self::from_core(core_scenarios::ScenarioDefinition::ReverseStress(
            definition.to_core(),
        ))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core_scenarios::ScenarioDefinition::HistoricalReplay(_) => "historical_replay",
            core_scenarios::ScenarioDefinition::Hypothetical(_) => "hypothetical",
            core_scenarios::ScenarioDefinition::ParametricStress2d(_) => "parametric_stress_2d",
            core_scenarios::ScenarioDefinition::ReverseStress(_) => "reverse_stress",
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct StressGridPoint {
    #[pyo3(get, set)]
    pub x_factor: ShockFactor,
    #[pyo3(get, set)]
    pub y_factor: ShockFactor,
    #[pyo3(get, set)]
    pub x_value: f64,
    #[pyo3(get, set)]
    pub y_value: f64,
    #[pyo3(get, set)]
    pub x_index: usize,
    #[pyo3(get, set)]
    pub y_index: usize,
}

impl StressGridPoint {
    fn from_core(value: core_scenarios::StressGridPoint) -> Self {
        Self {
            x_factor: ShockFactor::from_core(value.x_factor),
            y_factor: ShockFactor::from_core(value.y_factor),
            x_value: value.x_value,
            y_value: value.y_value,
            x_index: value.x_index,
            y_index: value.y_index,
        }
    }
}

#[pymethods]
impl StressGridPoint {
    #[new]
    fn new(
        x_factor: ShockFactor,
        y_factor: ShockFactor,
        x_value: f64,
        y_value: f64,
        x_index: usize,
        y_index: usize,
    ) -> Self {
        Self {
            x_factor,
            y_factor,
            x_value,
            y_value,
            x_index,
            y_index,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ResolvedScenario {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub parent_scenario_id: String,
    #[pyo3(get, set)]
    pub kind: ScenarioKind,
    shock: MarketShock,
    grid_point: Option<StressGridPoint>,
    #[pyo3(get, set)]
    pub reverse_stress_scale: Option<f64>,
}

impl ResolvedScenario {
    fn from_core(value: core_scenarios::ResolvedScenario) -> Self {
        Self {
            scenario_id: value.scenario_id,
            parent_scenario_id: value.parent_scenario_id,
            kind: ScenarioKind::from_core(value.kind),
            shock: MarketShock::from_core(value.shock),
            grid_point: value.grid_point.map(StressGridPoint::from_core),
            reverse_stress_scale: value.reverse_stress_scale,
        }
    }
}

#[pymethods]
impl ResolvedScenario {
    #[new]
    fn new(
        scenario_id: String,
        parent_scenario_id: String,
        kind: ScenarioKind,
        shock: MarketShock,
        grid_point: Option<StressGridPoint>,
        reverse_stress_scale: Option<f64>,
    ) -> Self {
        Self {
            scenario_id,
            parent_scenario_id,
            kind,
            shock,
            grid_point,
            reverse_stress_scale,
        }
    }

    #[getter]
    fn shock(&self) -> MarketShock {
        self.shock
    }

    #[getter]
    fn grid_point(&self) -> Option<StressGridPoint> {
        self.grid_point.clone()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioTrade {
    #[pyo3(get, set)]
    pub trade_id: String,
    #[pyo3(get, set)]
    pub instrument: String,
    #[pyo3(get, set)]
    pub quantity: f64,
    greeks: Greeks,
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub implied_vol: f64,
    #[pyo3(get, set)]
    pub vanna: f64,
    #[pyo3(get, set)]
    pub cross_gamma: f64,
    #[pyo3(get, set)]
    pub credit_delta: f64,
}

impl ScenarioTrade {
    fn to_core(&self) -> core_scenarios::ScenarioTrade<String> {
        core_scenarios::ScenarioTrade {
            trade_id: self.trade_id.clone(),
            instrument: self.instrument.clone(),
            quantity: self.quantity,
            greeks: self.greeks.to_core(),
            spot: self.spot,
            implied_vol: self.implied_vol,
            vanna: self.vanna,
            cross_gamma: self.cross_gamma,
            credit_delta: self.credit_delta,
        }
    }

    fn from_core(value: core_scenarios::ScenarioTrade<String>) -> Self {
        Self {
            trade_id: value.trade_id,
            instrument: value.instrument,
            quantity: value.quantity,
            greeks: Greeks::from_core(value.greeks),
            spot: value.spot,
            implied_vol: value.implied_vol,
            vanna: value.vanna,
            cross_gamma: value.cross_gamma,
            credit_delta: value.credit_delta,
        }
    }
}

#[pymethods]
impl ScenarioTrade {
    #[new]
    fn new(
        trade_id: String,
        instrument: String,
        quantity: f64,
        greeks: Greeks,
        spot: f64,
        implied_vol: f64,
    ) -> PyResult<Self> {
        catch_unwind_py(|| {
            Self::from_core(core_scenarios::ScenarioTrade::new(
                trade_id,
                instrument,
                quantity,
                greeks.to_core(),
                spot,
                implied_vol,
            ))
        })
    }

    #[getter]
    fn greeks(&self) -> Greeks {
        self.greeks
    }

    #[setter]
    fn set_greeks(&mut self, value: Greeks) {
        self.greeks = value;
    }

    fn with_cross_terms(&self, vanna: f64, cross_gamma: f64) -> Self {
        Self::from_core(self.to_core().with_cross_terms(vanna, cross_gamma))
    }

    fn with_credit_delta(&self, credit_delta: f64) -> Self {
        Self::from_core(self.to_core().with_credit_delta(credit_delta))
    }

    #[staticmethod]
    fn from_position(trade_id: String, position: &Position) -> Self {
        Self::from_core(core_scenarios::ScenarioTrade::from_position(
            trade_id,
            position.to_core(),
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct ExplainedPnlComponents {
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub cross_gamma_vanna: f64,
}

impl ExplainedPnlComponents {
    fn from_core(value: core_scenarios::ExplainedPnlComponents) -> Self {
        Self {
            theta: value.theta,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            cross_gamma_vanna: value.cross_gamma_vanna,
        }
    }
}

#[pymethods]
impl ExplainedPnlComponents {
    #[new]
    fn new(theta: f64, delta: f64, gamma: f64, vega: f64, cross_gamma_vanna: f64) -> Self {
        Self {
            theta,
            delta,
            gamma,
            vega,
            cross_gamma_vanna,
        }
    }

    fn explained(&self) -> f64 {
        self.theta + self.delta + self.gamma + self.vega + self.cross_gamma_vanna
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioTradePnlRow {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub parent_scenario_id: String,
    #[pyo3(get, set)]
    pub scenario_kind: ScenarioKind,
    #[pyo3(get, set)]
    pub trade_id: String,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub cross_gamma_vanna: f64,
    #[pyo3(get, set)]
    pub explained_pnl: f64,
    #[pyo3(get, set)]
    pub observed_pnl: f64,
    #[pyo3(get, set)]
    pub unexplained_pnl: f64,
    #[pyo3(get, set)]
    pub unexplained_ratio: f64,
}

impl ScenarioTradePnlRow {
    fn to_core(&self) -> core_scenarios::ScenarioTradePnlRow {
        core_scenarios::ScenarioTradePnlRow {
            scenario_id: self.scenario_id.clone(),
            parent_scenario_id: self.parent_scenario_id.clone(),
            scenario_kind: self.scenario_kind.to_core(),
            trade_id: self.trade_id.clone(),
            theta: self.theta,
            delta: self.delta,
            gamma: self.gamma,
            vega: self.vega,
            cross_gamma_vanna: self.cross_gamma_vanna,
            explained_pnl: self.explained_pnl,
            observed_pnl: self.observed_pnl,
            unexplained_pnl: self.unexplained_pnl,
            unexplained_ratio: self.unexplained_ratio,
        }
    }

    fn from_core(value: core_scenarios::ScenarioTradePnlRow) -> Self {
        Self {
            scenario_id: value.scenario_id,
            parent_scenario_id: value.parent_scenario_id,
            scenario_kind: ScenarioKind::from_core(value.scenario_kind),
            trade_id: value.trade_id,
            theta: value.theta,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            cross_gamma_vanna: value.cross_gamma_vanna,
            explained_pnl: value.explained_pnl,
            observed_pnl: value.observed_pnl,
            unexplained_pnl: value.unexplained_pnl,
            unexplained_ratio: value.unexplained_ratio,
        }
    }
}

#[pymethods]
impl ScenarioTradePnlRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        scenario_id: String,
        parent_scenario_id: String,
        scenario_kind: ScenarioKind,
        trade_id: String,
        theta: f64,
        delta: f64,
        gamma: f64,
        vega: f64,
        cross_gamma_vanna: f64,
        explained_pnl: f64,
        observed_pnl: f64,
        unexplained_pnl: f64,
        unexplained_ratio: f64,
    ) -> Self {
        Self {
            scenario_id,
            parent_scenario_id,
            scenario_kind,
            trade_id,
            theta,
            delta,
            gamma,
            vega,
            cross_gamma_vanna,
            explained_pnl,
            observed_pnl,
            unexplained_pnl,
            unexplained_ratio,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioPortfolioPnlRow {
    #[pyo3(get, set)]
    pub scenario_id: String,
    #[pyo3(get, set)]
    pub parent_scenario_id: String,
    #[pyo3(get, set)]
    pub scenario_kind: ScenarioKind,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub cross_gamma_vanna: f64,
    #[pyo3(get, set)]
    pub explained_pnl: f64,
    #[pyo3(get, set)]
    pub observed_pnl: f64,
    #[pyo3(get, set)]
    pub unexplained_pnl: f64,
    #[pyo3(get, set)]
    pub unexplained_ratio: f64,
}

impl ScenarioPortfolioPnlRow {
    fn from_core(value: core_scenarios::ScenarioPortfolioPnlRow) -> Self {
        Self {
            scenario_id: value.scenario_id,
            parent_scenario_id: value.parent_scenario_id,
            scenario_kind: ScenarioKind::from_core(value.scenario_kind),
            theta: value.theta,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            cross_gamma_vanna: value.cross_gamma_vanna,
            explained_pnl: value.explained_pnl,
            observed_pnl: value.observed_pnl,
            unexplained_pnl: value.unexplained_pnl,
            unexplained_ratio: value.unexplained_ratio,
        }
    }
}

#[pymethods]
impl ScenarioPortfolioPnlRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        scenario_id: String,
        parent_scenario_id: String,
        scenario_kind: ScenarioKind,
        theta: f64,
        delta: f64,
        gamma: f64,
        vega: f64,
        cross_gamma_vanna: f64,
        explained_pnl: f64,
        observed_pnl: f64,
        unexplained_pnl: f64,
        unexplained_ratio: f64,
    ) -> Self {
        Self {
            scenario_id,
            parent_scenario_id,
            scenario_kind,
            theta,
            delta,
            gamma,
            vega,
            cross_gamma_vanna,
            explained_pnl,
            observed_pnl,
            unexplained_pnl,
            unexplained_ratio,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Default)]
pub struct ScenarioResultTable {
    #[pyo3(get, set)]
    pub rows: Vec<ScenarioTradePnlRow>,
}

impl ScenarioResultTable {
    fn to_core(&self) -> core_scenarios::ScenarioResultTable {
        core_scenarios::ScenarioResultTable {
            rows: self.rows.iter().map(ScenarioTradePnlRow::to_core).collect(),
        }
    }

    fn from_core(value: core_scenarios::ScenarioResultTable) -> Self {
        Self {
            rows: value
                .rows
                .into_iter()
                .map(ScenarioTradePnlRow::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl ScenarioResultTable {
    #[new]
    fn new(rows: Vec<ScenarioTradePnlRow>) -> Self {
        Self { rows }
    }

    fn portfolio_rows(&self) -> Vec<ScenarioPortfolioPnlRow> {
        self.to_core()
            .portfolio_rows()
            .into_iter()
            .map(ScenarioPortfolioPnlRow::from_core)
            .collect()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct StressHeatmap2d {
    #[pyo3(get, set)]
    pub parent_scenario_id: String,
    #[pyo3(get, set)]
    pub x_factor: ShockFactor,
    #[pyo3(get, set)]
    pub y_factor: ShockFactor,
    #[pyo3(get, set)]
    pub x_values: Vec<f64>,
    #[pyo3(get, set)]
    pub y_values: Vec<f64>,
    #[pyo3(get, set)]
    pub pnl: Vec<Vec<f64>>,
}

impl StressHeatmap2d {
    fn from_core(value: core_scenarios::StressHeatmap2d) -> Self {
        Self {
            parent_scenario_id: value.parent_scenario_id,
            x_factor: ShockFactor::from_core(value.x_factor),
            y_factor: ShockFactor::from_core(value.y_factor),
            x_values: value.x_values,
            y_values: value.y_values,
            pnl: value.pnl,
        }
    }
}

#[pymethods]
impl StressHeatmap2d {
    #[new]
    fn new(
        parent_scenario_id: String,
        x_factor: ShockFactor,
        y_factor: ShockFactor,
        x_values: Vec<f64>,
        y_values: Vec<f64>,
        pnl: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            parent_scenario_id,
            x_factor,
            y_factor,
            x_values,
            y_values,
            pnl,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScenarioRunResult {
    #[pyo3(get, set)]
    pub resolved_scenarios: Vec<ResolvedScenario>,
    table: ScenarioResultTable,
}

impl ScenarioRunResult {
    fn from_core(value: core_scenarios::ScenarioRunResult) -> Self {
        Self {
            resolved_scenarios: value
                .resolved_scenarios
                .into_iter()
                .map(ResolvedScenario::from_core)
                .collect(),
            table: ScenarioResultTable::from_core(value.table),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MarketLevelDiff {
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub spot_pct_change: f64,
    #[pyo3(get, set)]
    pub rate_abs_change: f64,
    #[pyo3(get, set)]
    pub dividend_yield_abs_change: f64,
    #[pyo3(get, set)]
    pub atm_vol_pct_change: f64,
}

impl MarketLevelDiff {
    fn from_core(value: core_scenarios::MarketLevelDiff) -> Self {
        Self {
            market_id: value.market_id,
            spot_pct_change: value.spot_pct_change,
            rate_abs_change: value.rate_abs_change,
            dividend_yield_abs_change: value.dividend_yield_abs_change,
            atm_vol_pct_change: value.atm_vol_pct_change,
        }
    }
}

#[pymethods]
impl MarketLevelDiff {
    #[new]
    fn new(
        market_id: String,
        spot_pct_change: f64,
        rate_abs_change: f64,
        dividend_yield_abs_change: f64,
        atm_vol_pct_change: f64,
    ) -> Self {
        Self {
            market_id,
            spot_pct_change,
            rate_abs_change,
            dividend_yield_abs_change,
            atm_vol_pct_change,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SpotPriceDiff {
    #[pyo3(get, set)]
    pub asset_id: String,
    #[pyo3(get, set)]
    pub spot_pct_change: f64,
}

impl SpotPriceDiff {
    fn from_core(value: core_scenarios::SpotPriceDiff) -> Self {
        Self {
            asset_id: value.asset_id,
            spot_pct_change: value.spot_pct_change,
        }
    }
}

#[pymethods]
impl SpotPriceDiff {
    #[new]
    fn new(asset_id: String, spot_pct_change: f64) -> Self {
        Self {
            asset_id,
            spot_pct_change,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct YieldCurveDiff {
    #[pyo3(get, set)]
    pub curve_id: String,
    #[pyo3(get, set)]
    pub parallel_shift_abs: f64,
    #[pyo3(get, set)]
    pub max_abs_shift: f64,
}

impl YieldCurveDiff {
    fn from_core(value: core_scenarios::YieldCurveDiff) -> Self {
        Self {
            curve_id: value.curve_id,
            parallel_shift_abs: value.parallel_shift_abs,
            max_abs_shift: value.max_abs_shift,
        }
    }
}

#[pymethods]
impl YieldCurveDiff {
    #[new]
    fn new(curve_id: String, parallel_shift_abs: f64, max_abs_shift: f64) -> Self {
        Self {
            curve_id,
            parallel_shift_abs,
            max_abs_shift,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CreditCurveDiff {
    #[pyo3(get, set)]
    pub curve_id: String,
    #[pyo3(get, set)]
    pub avg_hazard_shift_abs: f64,
    #[pyo3(get, set)]
    pub max_hazard_shift_abs: f64,
}

impl CreditCurveDiff {
    fn from_core(value: core_scenarios::CreditCurveDiff) -> Self {
        Self {
            curve_id: value.curve_id,
            avg_hazard_shift_abs: value.avg_hazard_shift_abs,
            max_hazard_shift_abs: value.max_hazard_shift_abs,
        }
    }
}

#[pymethods]
impl CreditCurveDiff {
    #[new]
    fn new(curve_id: String, avg_hazard_shift_abs: f64, max_hazard_shift_abs: f64) -> Self {
        Self {
            curve_id,
            avg_hazard_shift_abs,
            max_hazard_shift_abs,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MarketSnapshotDiff {
    #[pyo3(get, set)]
    pub from_snapshot_id: String,
    #[pyo3(get, set)]
    pub to_snapshot_id: String,
    #[pyo3(get, set)]
    pub market_diffs: Vec<MarketLevelDiff>,
    #[pyo3(get, set)]
    pub spot_price_diffs: Vec<SpotPriceDiff>,
    #[pyo3(get, set)]
    pub yield_curve_diffs: Vec<YieldCurveDiff>,
    #[pyo3(get, set)]
    pub credit_curve_diffs: Vec<CreditCurveDiff>,
}

impl MarketSnapshotDiff {
    fn to_core(&self) -> core_scenarios::MarketSnapshotDiff {
        core_scenarios::MarketSnapshotDiff {
            from_snapshot_id: self.from_snapshot_id.clone(),
            to_snapshot_id: self.to_snapshot_id.clone(),
            market_diffs: self
                .market_diffs
                .iter()
                .cloned()
                .map(|value| core_scenarios::MarketLevelDiff {
                    market_id: value.market_id,
                    spot_pct_change: value.spot_pct_change,
                    rate_abs_change: value.rate_abs_change,
                    dividend_yield_abs_change: value.dividend_yield_abs_change,
                    atm_vol_pct_change: value.atm_vol_pct_change,
                })
                .collect(),
            spot_price_diffs: self
                .spot_price_diffs
                .iter()
                .cloned()
                .map(|value| core_scenarios::SpotPriceDiff {
                    asset_id: value.asset_id,
                    spot_pct_change: value.spot_pct_change,
                })
                .collect(),
            yield_curve_diffs: self
                .yield_curve_diffs
                .iter()
                .cloned()
                .map(|value| core_scenarios::YieldCurveDiff {
                    curve_id: value.curve_id,
                    parallel_shift_abs: value.parallel_shift_abs,
                    max_abs_shift: value.max_abs_shift,
                })
                .collect(),
            credit_curve_diffs: self
                .credit_curve_diffs
                .iter()
                .cloned()
                .map(|value| core_scenarios::CreditCurveDiff {
                    curve_id: value.curve_id,
                    avg_hazard_shift_abs: value.avg_hazard_shift_abs,
                    max_hazard_shift_abs: value.max_hazard_shift_abs,
                })
                .collect(),
        }
    }

    fn from_core(value: core_scenarios::MarketSnapshotDiff) -> Self {
        Self {
            from_snapshot_id: value.from_snapshot_id,
            to_snapshot_id: value.to_snapshot_id,
            market_diffs: value
                .market_diffs
                .into_iter()
                .map(MarketLevelDiff::from_core)
                .collect(),
            spot_price_diffs: value
                .spot_price_diffs
                .into_iter()
                .map(SpotPriceDiff::from_core)
                .collect(),
            yield_curve_diffs: value
                .yield_curve_diffs
                .into_iter()
                .map(YieldCurveDiff::from_core)
                .collect(),
            credit_curve_diffs: value
                .credit_curve_diffs
                .into_iter()
                .map(CreditCurveDiff::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl MarketSnapshotDiff {
    #[new]
    fn new(
        from_snapshot_id: String,
        to_snapshot_id: String,
        market_diffs: Vec<MarketLevelDiff>,
        spot_price_diffs: Vec<SpotPriceDiff>,
        yield_curve_diffs: Vec<YieldCurveDiff>,
        credit_curve_diffs: Vec<CreditCurveDiff>,
    ) -> Self {
        Self {
            from_snapshot_id,
            to_snapshot_id,
            market_diffs,
            spot_price_diffs,
            yield_curve_diffs,
            credit_curve_diffs,
        }
    }

    fn to_market_shock(&self, horizon_years: f64) -> MarketShock {
        MarketShock::from_core(self.to_core().to_market_shock(horizon_years))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DayOverDayAttribution {
    market_diff: MarketSnapshotDiff,
    scenario: ResolvedScenario,
    table: ScenarioResultTable,
    portfolio: ScenarioPortfolioPnlRow,
}

impl DayOverDayAttribution {
    fn from_core(value: core_scenarios::DayOverDayAttribution) -> Self {
        Self {
            market_diff: MarketSnapshotDiff::from_core(value.market_diff),
            scenario: ResolvedScenario::from_core(value.scenario),
            table: ScenarioResultTable::from_core(value.table),
            portfolio: ScenarioPortfolioPnlRow::from_core(value.portfolio),
        }
    }
}

#[pymethods]
impl DayOverDayAttribution {
    #[new]
    fn new(
        market_diff: MarketSnapshotDiff,
        scenario: ResolvedScenario,
        table: ScenarioResultTable,
        portfolio: ScenarioPortfolioPnlRow,
    ) -> Self {
        Self {
            market_diff,
            scenario,
            table,
            portfolio,
        }
    }

    #[getter]
    fn market_diff(&self) -> MarketSnapshotDiff {
        self.market_diff.clone()
    }

    #[getter]
    fn scenario(&self) -> ResolvedScenario {
        self.scenario.clone()
    }

    #[getter]
    fn table(&self) -> ScenarioResultTable {
        self.table.clone()
    }

    #[getter]
    fn portfolio(&self) -> ScenarioPortfolioPnlRow {
        self.portfolio.clone()
    }
}

#[pyfunction]
pub fn py_explained_pnl_components(
    trade: &ScenarioTrade,
    shock: &MarketShock,
) -> ExplainedPnlComponents {
    ExplainedPnlComponents::from_core(core_scenarios::explained_pnl_components(
        &trade.to_core(),
        &shock.to_core(),
    ))
}

#[pyfunction]
pub fn py_apply_market_shock(market: &Market, shock: &MarketShock) -> Market {
    Market::from_core(core_scenarios::apply_market_shock(
        &market.to_core(),
        &shock.to_core(),
    ))
}

#[pyfunction]
pub fn py_diff_market_snapshots(
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
) -> MarketSnapshotDiff {
    MarketSnapshotDiff::from_core(core_scenarios::diff_market_snapshots(
        &previous.to_core(),
        &current.to_core(),
    ))
}

#[pyfunction]
pub fn py_historical_replay_from_diff(
    scenario_id: String,
    replay_date: Option<String>,
    market_diff: &MarketSnapshotDiff,
    horizon_years: f64,
) -> ScenarioDefinition {
    ScenarioDefinition::from_core(core_scenarios::historical_replay_from_diff(
        scenario_id,
        replay_date,
        &market_diff.to_core(),
        horizon_years,
    ))
}

#[pyfunction]
pub fn py_run_scenario_batch(
    trades: Vec<ScenarioTrade>,
    definitions: Vec<ScenarioDefinition>,
) -> PyResult<ScenarioRunResult> {
    let core_trades = trades
        .iter()
        .map(ScenarioTrade::to_core)
        .collect::<Vec<_>>();
    let core_defs = definitions
        .iter()
        .map(ScenarioDefinition::to_core)
        .collect::<Vec<_>>();
    core_scenarios::run_scenario_batch(&core_trades, &core_defs)
        .map(ScenarioRunResult::from_core)
        .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_run_scenario_batch_with_pricer(
    trades: Vec<ScenarioTrade>,
    definitions: Vec<ScenarioDefinition>,
    pricer: Py<PyAny>,
) -> PyResult<ScenarioRunResult> {
    let core_trades = trades
        .iter()
        .map(ScenarioTrade::to_core)
        .collect::<Vec<_>>();
    let core_defs = definitions
        .iter()
        .map(ScenarioDefinition::to_core)
        .collect::<Vec<_>>();
    core_scenarios::run_scenario_batch_with_pricer(&core_trades, &core_defs, |trade, shock| {
        Python::attach(|py| {
            let callable = pricer.bind(py);
            let value = callable
                .call1((
                    ScenarioTrade::from_core(trade.clone()),
                    MarketShock::from_core(*shock),
                ))
                .map_err(|err| CorePricingError::InvalidInput(err.to_string()))?;
            value
                .extract::<f64>()
                .map_err(|err| CorePricingError::InvalidInput(err.to_string()))
        })
    })
    .map(ScenarioRunResult::from_core)
    .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_day_over_day_attribution(
    trades: Vec<ScenarioTrade>,
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
) -> PyResult<DayOverDayAttribution> {
    let core_trades = trades
        .iter()
        .map(ScenarioTrade::to_core)
        .collect::<Vec<_>>();
    core_scenarios::day_over_day_attribution(&core_trades, &previous.to_core(), &current.to_core())
        .map(DayOverDayAttribution::from_core)
        .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_day_over_day_attribution_with_pricer(
    trades: Vec<ScenarioTrade>,
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
    pricer: Py<PyAny>,
) -> PyResult<DayOverDayAttribution> {
    let core_trades = trades
        .iter()
        .map(ScenarioTrade::to_core)
        .collect::<Vec<_>>();
    core_scenarios::day_over_day_attribution_with_pricer(
        &core_trades,
        &previous.to_core(),
        &current.to_core(),
        |trade, shock| {
            Python::attach(|py| {
                let callable = pricer.bind(py);
                let value = callable
                    .call1((
                        ScenarioTrade::from_core(trade.clone()),
                        MarketShock::from_core(*shock),
                    ))
                    .map_err(|err| CorePricingError::InvalidInput(err.to_string()))?;
                value
                    .extract::<f64>()
                    .map_err(|err| CorePricingError::InvalidInput(err.to_string()))
            })
        },
    )
    .map(DayOverDayAttribution::from_core)
    .map_err(pricing_error_to_pyerr)
}

#[pymethods]
impl ScenarioRunResult {
    #[new]
    fn new(resolved_scenarios: Vec<ResolvedScenario>, table: ScenarioResultTable) -> Self {
        Self {
            resolved_scenarios,
            table,
        }
    }

    #[getter]
    fn table(&self) -> ScenarioResultTable {
        self.table.clone()
    }

    fn stress_heatmap(&self, parent_scenario_id: &str) -> Option<StressHeatmap2d> {
        let core = core_scenarios::ScenarioRunResult {
            resolved_scenarios: self
                .resolved_scenarios
                .iter()
                .cloned()
                .map(|value| core_scenarios::ResolvedScenario {
                    scenario_id: value.scenario_id,
                    parent_scenario_id: value.parent_scenario_id,
                    kind: value.kind.to_core(),
                    shock: value.shock.to_core(),
                    grid_point: value
                        .grid_point
                        .map(|point| core_scenarios::StressGridPoint {
                            x_factor: point.x_factor.to_core(),
                            y_factor: point.y_factor.to_core(),
                            x_value: point.x_value,
                            y_value: point.y_value,
                            x_index: point.x_index,
                            y_index: point.y_index,
                        }),
                    reverse_stress_scale: value.reverse_stress_scale,
                })
                .collect(),
            table: self.table.to_core(),
        };
        core.stress_heatmap(parent_scenario_id)
            .map(StressHeatmap2d::from_core)
    }
}

#[pymethods]
impl TradeRiskContribution {
    #[new]
    fn new(
        trade_index: usize,
        delta: f64,
        gamma: f64,
        vega: f64,
        theta: f64,
        rho: f64,
        total: f64,
        share_of_total: f64,
    ) -> Self {
        Self {
            trade_index,
            delta,
            gamma,
            vega,
            theta,
            rho,
            total,
            share_of_total,
        }
    }
}

#[pymethods]
impl QuoteVolSurface {
    #[new]
    fn new(expiries: Vec<f64>, strikes: Vec<f64>, quotes: Vec<Vec<f64>>) -> PyResult<Self> {
        core_sens::QuoteVolSurface::new(expiries, strikes, quotes)
            .map(Self::from_core)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn expiries(&self) -> Vec<f64> {
        self.inner.expiries().to_vec()
    }

    fn strikes(&self) -> Vec<f64> {
        self.inner.strikes().to_vec()
    }

    fn quote(&self, expiry_index: usize, strike_index: usize) -> f64 {
        self.inner.quote(expiry_index, strike_index)
    }

    fn set_quote(&mut self, expiry_index: usize, strike_index: usize, quote: f64) {
        self.inner.set_quote(expiry_index, strike_index, quote);
    }

    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        use openferric_core::market::VolSurface;
        self.inner.vol(strike, expiry)
    }
}

#[pymethods]
impl HullWhiteWWR {
    #[new]
    fn new(base_hazard: f64, beta: f64, num_paths: usize, seed: u64) -> PyResult<Self> {
        catch_unwind_py(|| {
            let core = core_wwr::HullWhiteWWR::new(base_hazard, beta, num_paths, seed);
            Self {
                base_hazard: core.base_hazard,
                beta: core.beta,
                num_paths: core.num_paths,
                seed: core.seed,
            }
        })
    }

    fn cva_with_wwr(
        &self,
        asset_paths: Vec<Vec<f64>>,
        exposure_paths: Vec<Vec<f64>>,
        time_grid: Vec<f64>,
        recovery: f64,
        risk_free_rate: f64,
    ) -> PyResult<WwrResult> {
        catch_unwind_py(|| {
            WwrResult::from_core(self.to_core().cva_with_wwr(
                &asset_paths,
                &exposure_paths,
                &time_grid,
                recovery,
                risk_free_rate,
            ))
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(py_cva, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_sa_ccr_ead, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_historical_var, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_historical_es, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_delta_normal_var, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_delta_gamma_normal_var, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_normal_expected_shortfall,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_cornish_fisher_var, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_cornish_fisher_var_from_pnl,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_historical_var_from_prices,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_historical_expected_shortfall_from_prices,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_rolling_historical_var_from_prices,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_backtest_historical_var_from_prices,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_funding_exposure_profile, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_fva_from_profile, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_mva_from_profile, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_regulatory_capital, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_kva_from_profile, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_netting_set_exposure, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_parallel_dv01, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_bucket_dv01, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_key_rate_duration, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_gamma_ladder, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_cross_gamma, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_vega_by_expiry_bucket, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_vega_by_strike_expiry_bucket,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_fx_delta, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_commodity_delta, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_jacobian_via_bootstrap, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_map_risk_class, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_to_crif_csv, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_compute_risk_charges, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_pnl_explain, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_scenario_pnl_report, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_risk_contribution_per_trade,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_explained_pnl_components, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_apply_market_shock, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_diff_market_snapshots, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_historical_replay_from_diff,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_run_scenario_batch, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_run_scenario_batch_with_pricer,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_day_over_day_attribution, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_day_over_day_attribution_with_pricer,
        module
    )?)?;
    module.add_class::<Greeks>()?;
    module.add_class::<YieldCurve>()?;
    module.add_class::<SurvivalCurve>()?;
    module.add_class::<KupiecBacktestResult>()?;
    module.add_class::<ChristoffersenBacktestResult>()?;
    module.add_class::<VarBacktestResult>()?;
    module.add_class::<SampledVolSurface>()?;
    module.add_class::<VolSource>()?;
    module.add_class::<Market>()?;
    module.add_class::<ForwardCurveSnapshot>()?;
    module.add_class::<CreditCurveSnapshot>()?;
    module.add_class::<MarketSnapshot>()?;
    module.add_class::<MarginParams>()?;
    module.add_class::<MarginCalculator>()?;
    module.add_class::<InherentLeverage>()?;
    module.add_class::<Vasicek>()?;
    module.add_class::<FundingRateModel>()?;
    module.add_class::<LiquidationPosition>()?;
    module.add_class::<StressScenario>()?;
    module.add_class::<LiquidationRisk>()?;
    module.add_class::<StressTestResult>()?;
    module.add_class::<LiquidationSimulator>()?;
    module.add_class::<XvaCalculator>()?;
    module.add_class::<CsaTerms>()?;
    module.add_class::<SimmRiskClass>()?;
    module.add_class::<SimmMargin>()?;
    module.add_class::<SaCcrAssetClass>()?;
    module.add_class::<AggregatedGreeks>()?;
    module.add_class::<Position>()?;
    module.add_class::<Portfolio>()?;
    module.add_class::<WwrResult>()?;
    module.add_class::<AlphaWWR>()?;
    module.add_class::<CopulaWWR>()?;
    module.add_class::<HullWhiteWWR>()?;
    module.add_class::<BumpSize>()?;
    module.add_class::<DifferencingScheme>()?;
    module.add_class::<CurveBumpMode>()?;
    module.add_class::<CurveBumpConfig>()?;
    module.add_class::<SurfaceBumpMode>()?;
    module.add_class::<SurfaceBumpConfig>()?;
    module.add_class::<SpotBumpConfig>()?;
    module.add_class::<BucketSensitivity>()?;
    module.add_class::<KeyRateDurationPoint>()?;
    module.add_class::<GammaLadderPoint>()?;
    module.add_class::<VegaExpiryPoint>()?;
    module.add_class::<VegaStrikeExpiryPoint>()?;
    module.add_class::<QuoteVolSurface>()?;
    module.add_class::<ChainRuleJacobian>()?;
    module.add_class::<RegulatoryRiskClass>()?;
    module.add_class::<SensitivityMeasure>()?;
    module.add_class::<SensitivityRecord>()?;
    module.add_class::<CrifRecord>()?;
    module.add_class::<RiskClassChargeConfig>()?;
    module.add_class::<RiskChargeConfig>()?;
    module.add_class::<ClassRiskCharge>()?;
    module.add_class::<RiskChargeSummary>()?;
    module.add_class::<ScenarioShock>()?;
    module.add_class::<PnlExplain>()?;
    module.add_class::<ScenarioPnlRow>()?;
    module.add_class::<TradeRiskContribution>()?;
    module.add_class::<ScenarioKind>()?;
    module.add_class::<ShockFactor>()?;
    module.add_class::<MarketShock>()?;
    module.add_class::<HistoricalReplayDefinition>()?;
    module.add_class::<HypotheticalScenarioDefinition>()?;
    module.add_class::<StressAxis>()?;
    module.add_class::<ParametricStress2dDefinition>()?;
    module.add_class::<ReverseStressDefinition>()?;
    module.add_class::<ScenarioDefinition>()?;
    module.add_class::<StressGridPoint>()?;
    module.add_class::<ResolvedScenario>()?;
    module.add_class::<ScenarioTrade>()?;
    module.add_class::<ExplainedPnlComponents>()?;
    module.add_class::<ScenarioTradePnlRow>()?;
    module.add_class::<ScenarioPortfolioPnlRow>()?;
    module.add_class::<ScenarioResultTable>()?;
    module.add_class::<StressHeatmap2d>()?;
    module.add_class::<ScenarioRunResult>()?;
    module.add_class::<MarketLevelDiff>()?;
    module.add_class::<SpotPriceDiff>()?;
    module.add_class::<YieldCurveDiff>()?;
    module.add_class::<CreditCurveDiff>()?;
    module.add_class::<MarketSnapshotDiff>()?;
    module.add_class::<DayOverDayAttribution>()?;
    Ok(())
}
