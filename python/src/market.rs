use chrono::NaiveDate;
use openferric_core::market::{
    BootstrappedDividendPoint as CoreBootstrappedDividendPoint,
    CreditCurveSnapshot as CoreCreditCurveSnapshot,
    DividendCurveBootstrap as CoreDividendCurveBootstrap, DividendEvent as CoreDividendEvent,
    DividendKind as CoreDividendKind, DividendSchedule as CoreDividendSchedule,
    ForwardCurveSnapshot as CoreForwardCurveSnapshot, FxAtmConvention as CoreFxAtmConvention,
    FxDeltaConvention as CoreFxDeltaConvention, FxForwardCurve as CoreFxForwardCurve,
    FxPair as CoreFxPair, FxRrBfPillar as CoreFxRrBfPillar,
    FxSmileMarketQuote as CoreFxSmileMarketQuote, FxSmileSlice as CoreFxSmileSlice,
    FxVolExpiryQuote as CoreFxVolExpiryQuote, FxVolSurface as CoreFxVolSurface,
    MalzInterpolator as CoreMalzInterpolator, Market as CoreMarket,
    MarketSnapshot as CoreMarketSnapshot, NdfContract as CoreNdfContract,
    NdfSettlementCurrency as CoreNdfSettlementCurrency, PremiumCurrency as CorePremiumCurrency,
    PutCallParityQuote as CorePutCallParityQuote, SampledVolSurface as CoreSampledVolSurface,
    VolSource as CoreVolSource,
};
use openferric_core::rates::YieldCurve as CoreYieldCurve;
use openferric_core::vol::surface::{SviParams, VolSurface as CoreParametricVolSurface};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::credit::SurvivalCurve;

fn map_string_err(err: String) -> PyErr {
    PyValueError::new_err(err)
}

fn parse_date(value: &str) -> PyResult<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|err| {
        PyValueError::new_err(format!(
            "invalid date '{value}'; expected YYYY-MM-DD ({err})"
        ))
    })
}

fn format_date(value: NaiveDate) -> String {
    value.format("%Y-%m-%d").to_string()
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct DividendKind {
    inner: CoreDividendKind,
}

impl DividendKind {
    pub(crate) fn to_core(&self) -> CoreDividendKind {
        self.inner
    }

    pub(crate) fn from_core(value: CoreDividendKind) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl DividendKind {
    #[staticmethod]
    fn cash(amount: f64) -> Self {
        Self {
            inner: CoreDividendKind::Cash(amount),
        }
    }

    #[staticmethod]
    fn proportional(ratio: f64) -> Self {
        Self {
            inner: CoreDividendKind::Proportional(ratio),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreDividendKind::Cash(_) => "Cash",
            CoreDividendKind::Proportional(_) => "Proportional",
        }
    }

    #[getter]
    fn value(&self) -> f64 {
        match self.inner {
            CoreDividendKind::Cash(value) | CoreDividendKind::Proportional(value) => value,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DividendKind(kind={:?}, value={})",
            self.kind(),
            self.value()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct DividendEvent {
    inner: CoreDividendEvent,
}

impl DividendEvent {
    pub(crate) fn to_core(&self) -> CoreDividendEvent {
        self.inner
    }

    pub(crate) fn from_core(value: CoreDividendEvent) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl DividendEvent {
    #[new]
    fn new(time: f64, kind: &DividendKind) -> Self {
        Self {
            inner: CoreDividendEvent {
                time,
                kind: kind.to_core(),
            },
        }
    }

    #[staticmethod]
    fn cash(time: f64, amount: f64) -> PyResult<Self> {
        CoreDividendEvent::cash(time, amount)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[staticmethod]
    fn proportional(time: f64, ratio: f64) -> PyResult<Self> {
        CoreDividendEvent::proportional(time, ratio)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }

    #[setter]
    fn set_time(&mut self, time: f64) {
        self.inner.time = time;
    }

    #[getter]
    fn kind(&self) -> DividendKind {
        DividendKind::from_core(self.inner.kind)
    }

    #[setter]
    fn set_kind(&mut self, kind: &DividendKind) {
        self.inner.kind = kind.to_core();
    }

    fn apply_jump(&self, pre_div_spot: f64) -> f64 {
        self.inner.apply_jump(pre_div_spot)
    }

    fn __repr__(&self) -> String {
        format!(
            "DividendEvent(time={}, kind={:?})",
            self.inner.time, self.inner.kind
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Default)]
pub struct DividendSchedule {
    inner: CoreDividendSchedule,
}

impl DividendSchedule {
    pub(crate) fn to_core(&self) -> CoreDividendSchedule {
        self.inner.clone()
    }

    pub(crate) fn from_core(value: CoreDividendSchedule) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl DividendSchedule {
    #[new]
    fn new(events: Vec<DividendEvent>) -> PyResult<Self> {
        let events = events.into_iter().map(|event| event.to_core()).collect();
        CoreDividendSchedule::new(events)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[staticmethod]
    fn empty() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn events(&self) -> Vec<DividendEvent> {
        self.inner
            .events()
            .iter()
            .copied()
            .map(DividendEvent::from_core)
            .collect()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(map_string_err)
    }

    fn forward_price(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        self.inner
            .forward_price(spot, rate, continuous_dividend_yield, maturity)
    }

    fn prepaid_forward_spot(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        self.inner
            .prepaid_forward_spot(spot, rate, continuous_dividend_yield, maturity)
    }

    fn escrowed_spot_adjustment(&self, spot: f64, rate: f64, maturity: f64) -> f64 {
        self.inner.escrowed_spot_adjustment(spot, rate, maturity)
    }

    fn effective_dividend_yield(
        &self,
        spot: f64,
        rate: f64,
        continuous_dividend_yield: f64,
        maturity: f64,
    ) -> f64 {
        self.inner
            .effective_dividend_yield(spot, rate, continuous_dividend_yield, maturity)
    }

    fn apply_jumps_until(&self, spot: f64, time: f64) -> f64 {
        self.inner.apply_jumps_until(spot, time)
    }

    fn __repr__(&self) -> String {
        format!("DividendSchedule(len={})", self.inner.events().len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PutCallParityQuote {
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub call_price: f64,
    #[pyo3(get, set)]
    pub put_price: f64,
}

impl PutCallParityQuote {
    pub(crate) fn to_core(self) -> CorePutCallParityQuote {
        CorePutCallParityQuote {
            maturity: self.maturity,
            strike: self.strike,
            call_price: self.call_price,
            put_price: self.put_price,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_core(value: CorePutCallParityQuote) -> Self {
        Self {
            maturity: value.maturity,
            strike: value.strike,
            call_price: value.call_price,
            put_price: value.put_price,
        }
    }
}

#[pymethods]
impl PutCallParityQuote {
    #[new]
    fn new(maturity: f64, strike: f64, call_price: f64, put_price: f64) -> Self {
        Self {
            maturity,
            strike,
            call_price,
            put_price,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_string_err)
    }

    fn implied_forward(&self, rate: f64) -> f64 {
        self.to_core().implied_forward(rate)
    }

    fn implied_prepaid_forward(&self, rate: f64) -> f64 {
        self.to_core().implied_prepaid_forward(rate)
    }

    fn __repr__(&self) -> String {
        format!(
            "PutCallParityQuote(maturity={}, strike={}, call_price={}, put_price={})",
            self.maturity, self.strike, self.call_price, self.put_price
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct BootstrappedDividendPoint {
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub forward: f64,
    #[pyo3(get, set)]
    pub prepaid_forward: f64,
    #[pyo3(get, set)]
    pub implied_dividend_yield: f64,
    #[pyo3(get, set)]
    pub cumulative_pv_dividends: f64,
}

impl BootstrappedDividendPoint {
    pub(crate) fn from_core(value: CoreBootstrappedDividendPoint) -> Self {
        Self {
            maturity: value.maturity,
            forward: value.forward,
            prepaid_forward: value.prepaid_forward,
            implied_dividend_yield: value.implied_dividend_yield,
            cumulative_pv_dividends: value.cumulative_pv_dividends,
        }
    }
}

#[pymethods]
impl BootstrappedDividendPoint {
    #[new]
    fn new(
        maturity: f64,
        forward: f64,
        prepaid_forward: f64,
        implied_dividend_yield: f64,
        cumulative_pv_dividends: f64,
    ) -> Self {
        Self {
            maturity,
            forward,
            prepaid_forward,
            implied_dividend_yield,
            cumulative_pv_dividends,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DividendCurveBootstrap {
    inner: CoreDividendCurveBootstrap,
}

impl DividendCurveBootstrap {
    pub(crate) fn from_core(value: CoreDividendCurveBootstrap) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl DividendCurveBootstrap {
    #[staticmethod]
    fn from_put_call_parity(
        spot: f64,
        rate: f64,
        quotes: Vec<PutCallParityQuote>,
    ) -> PyResult<Self> {
        let quotes = quotes
            .into_iter()
            .map(PutCallParityQuote::to_core)
            .collect::<Vec<_>>();
        CoreDividendCurveBootstrap::from_put_call_parity(spot, rate, &quotes)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }

    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }

    #[getter]
    fn points(&self) -> Vec<BootstrappedDividendPoint> {
        self.inner
            .points
            .iter()
            .copied()
            .map(BootstrappedDividendPoint::from_core)
            .collect()
    }

    fn prepaid_forward_spot(&self, maturity: f64) -> f64 {
        self.inner.prepaid_forward_spot(maturity)
    }

    fn forward(&self, maturity: f64) -> f64 {
        self.inner.forward(maturity)
    }

    fn to_cash_dividend_schedule(&self) -> PyResult<DividendSchedule> {
        self.inner
            .to_cash_dividend_schedule()
            .map(DividendSchedule::from_core)
            .map_err(map_string_err)
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PremiumCurrency {
    Domestic,
    Foreign,
}

impl PremiumCurrency {
    fn to_core(self) -> CorePremiumCurrency {
        match self {
            Self::Domestic => CorePremiumCurrency::Domestic,
            Self::Foreign => CorePremiumCurrency::Foreign,
        }
    }

    fn from_core(value: CorePremiumCurrency) -> Self {
        match value {
            CorePremiumCurrency::Domestic => Self::Domestic,
            CorePremiumCurrency::Foreign => Self::Foreign,
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FxDeltaConvention {
    Spot,
    Forward,
    PremiumAdjustedSpot,
    PremiumAdjustedForward,
}

impl FxDeltaConvention {
    fn to_core(self) -> CoreFxDeltaConvention {
        match self {
            Self::Spot => CoreFxDeltaConvention::Spot,
            Self::Forward => CoreFxDeltaConvention::Forward,
            Self::PremiumAdjustedSpot => CoreFxDeltaConvention::PremiumAdjustedSpot,
            Self::PremiumAdjustedForward => CoreFxDeltaConvention::PremiumAdjustedForward,
        }
    }

    fn from_core(value: CoreFxDeltaConvention) -> Self {
        match value {
            CoreFxDeltaConvention::Spot => Self::Spot,
            CoreFxDeltaConvention::Forward => Self::Forward,
            CoreFxDeltaConvention::PremiumAdjustedSpot => Self::PremiumAdjustedSpot,
            CoreFxDeltaConvention::PremiumAdjustedForward => Self::PremiumAdjustedForward,
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FxAtmConvention {
    Spot,
    Forward,
    DeltaNeutralStraddle,
}

impl FxAtmConvention {
    fn to_core(self) -> CoreFxAtmConvention {
        match self {
            Self::Spot => CoreFxAtmConvention::Spot,
            Self::Forward => CoreFxAtmConvention::Forward,
            Self::DeltaNeutralStraddle => CoreFxAtmConvention::DeltaNeutralStraddle,
        }
    }

    fn from_core(value: CoreFxAtmConvention) -> Self {
        match value {
            CoreFxAtmConvention::Spot => Self::Spot,
            CoreFxAtmConvention::Forward => Self::Forward,
            CoreFxAtmConvention::DeltaNeutralStraddle => Self::DeltaNeutralStraddle,
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NdfSettlementCurrency {
    Domestic,
    Foreign,
}

impl NdfSettlementCurrency {
    fn to_core(self) -> CoreNdfSettlementCurrency {
        match self {
            Self::Domestic => CoreNdfSettlementCurrency::Domestic,
            Self::Foreign => CoreNdfSettlementCurrency::Foreign,
        }
    }

    #[allow(dead_code)]
    fn from_core(value: CoreNdfSettlementCurrency) -> Self {
        match value {
            CoreNdfSettlementCurrency::Domestic => Self::Domestic,
            CoreNdfSettlementCurrency::Foreign => Self::Foreign,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct FxPair {
    #[pyo3(get, set)]
    pub base: String,
    #[pyo3(get, set)]
    pub quote: String,
    #[pyo3(get, set)]
    pub pip_size: f64,
    #[pyo3(get, set)]
    pub spot_lag_business_days: u32,
    default_premium_currency: PremiumCurrency,
}

impl FxPair {
    fn to_core(&self) -> CoreFxPair {
        CoreFxPair {
            base: self.base.clone(),
            quote: self.quote.clone(),
            pip_size: self.pip_size,
            spot_lag_business_days: self.spot_lag_business_days,
            default_premium_currency: self.default_premium_currency.to_core(),
        }
    }

    fn from_core(value: CoreFxPair) -> Self {
        Self {
            base: value.base,
            quote: value.quote,
            pip_size: value.pip_size,
            spot_lag_business_days: value.spot_lag_business_days,
            default_premium_currency: PremiumCurrency::from_core(value.default_premium_currency),
        }
    }
}

#[pymethods]
impl FxPair {
    #[new]
    fn new(base: String, quote: String) -> PyResult<Self> {
        CoreFxPair::new(&base, &quote)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[staticmethod]
    fn from_code(code: String) -> PyResult<Self> {
        CoreFxPair::from_code(&code)
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[getter]
    fn default_premium_currency(&self) -> PremiumCurrency {
        self.default_premium_currency
    }

    #[setter]
    fn set_default_premium_currency(&mut self, value: PremiumCurrency) {
        self.default_premium_currency = value;
    }

    fn code(&self) -> String {
        self.to_core().code()
    }

    fn outright_from_forward_points(&self, spot: f64, forward_points_pips: f64) -> f64 {
        self.to_core()
            .outright_from_forward_points(spot, forward_points_pips)
    }

    fn forward_points_from_outright(&self, spot: f64, outright_forward: f64) -> f64 {
        self.to_core()
            .forward_points_from_outright(spot, outright_forward)
    }

    fn spot_settlement_date(&self, trade_date: String) -> PyResult<String> {
        Ok(format_date(
            self.to_core()
                .spot_settlement_date(parse_date(&trade_date)?),
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct FxForwardCurve {
    #[pyo3(get, set)]
    pub tenors: Vec<f64>,
    #[pyo3(get, set)]
    pub domestic_rates: Vec<f64>,
    #[pyo3(get, set)]
    pub foreign_rates: Vec<f64>,
    #[pyo3(get, set)]
    pub basis_spreads: Vec<f64>,
}

impl FxForwardCurve {
    pub(crate) fn to_core(&self) -> PyResult<CoreFxForwardCurve> {
        CoreFxForwardCurve::from_deposit_rates(
            self.tenors.clone(),
            self.domestic_rates.clone(),
            self.foreign_rates.clone(),
            self.basis_spreads.clone(),
        )
        .map_err(map_string_err)
    }
}

#[pymethods]
impl FxForwardCurve {
    #[new]
    fn new(
        tenors: Vec<f64>,
        domestic_rates: Vec<f64>,
        foreign_rates: Vec<f64>,
        basis_spreads: Vec<f64>,
    ) -> PyResult<Self> {
        let this = Self {
            tenors,
            domestic_rates,
            foreign_rates,
            basis_spreads,
        };
        let _ = this.to_core()?;
        Ok(this)
    }

    fn domestic_rate(&self, tenor: f64) -> PyResult<f64> {
        Ok(self.to_core()?.domestic_rate(tenor))
    }

    fn foreign_rate(&self, tenor: f64) -> PyResult<f64> {
        Ok(self.to_core()?.foreign_rate(tenor))
    }

    fn basis_spread(&self, tenor: f64) -> PyResult<f64> {
        Ok(self.to_core()?.basis_spread(tenor))
    }

    fn outright_forward(&self, spot: f64, tenor: f64) -> PyResult<f64> {
        Ok(self.to_core()?.outright_forward(spot, tenor))
    }

    fn forward_points(&self, pair: &FxPair, spot: f64, tenor: f64) -> PyResult<f64> {
        Ok(self.to_core()?.forward_points(&pair.to_core(), spot, tenor))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct FxRrBfPillar {
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub risk_reversal: f64,
    #[pyo3(get, set)]
    pub butterfly: f64,
}

impl FxRrBfPillar {
    fn to_core(self) -> CoreFxRrBfPillar {
        CoreFxRrBfPillar {
            delta: self.delta,
            risk_reversal: self.risk_reversal,
            butterfly: self.butterfly,
        }
    }

    fn from_core(value: CoreFxRrBfPillar) -> Self {
        Self {
            delta: value.delta,
            risk_reversal: value.risk_reversal,
            butterfly: value.butterfly,
        }
    }
}

#[pymethods]
impl FxRrBfPillar {
    #[new]
    fn new(delta: f64, risk_reversal: f64, butterfly: f64) -> Self {
        Self {
            delta,
            risk_reversal,
            butterfly,
        }
    }

    fn put_call_vols(&self, atm_vol: f64) -> (f64, f64) {
        self.to_core().put_call_vols(atm_vol)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct FxSmileMarketQuote {
    #[pyo3(get, set)]
    pub atm_vol: f64,
    #[pyo3(get, set)]
    pub pillars: Vec<FxRrBfPillar>,
    #[pyo3(get, set)]
    pub atm_convention: FxAtmConvention,
    #[pyo3(get, set)]
    pub delta_convention: FxDeltaConvention,
    #[pyo3(get, set)]
    pub premium_currency: PremiumCurrency,
}

impl FxSmileMarketQuote {
    fn to_core(&self) -> CoreFxSmileMarketQuote {
        CoreFxSmileMarketQuote {
            atm_vol: self.atm_vol,
            pillars: self
                .pillars
                .iter()
                .copied()
                .map(FxRrBfPillar::to_core)
                .collect(),
            atm_convention: self.atm_convention.to_core(),
            delta_convention: self.delta_convention.to_core(),
            premium_currency: self.premium_currency.to_core(),
        }
    }

    fn from_core(value: CoreFxSmileMarketQuote) -> Self {
        Self {
            atm_vol: value.atm_vol,
            pillars: value
                .pillars
                .into_iter()
                .map(FxRrBfPillar::from_core)
                .collect(),
            atm_convention: FxAtmConvention::from_core(value.atm_convention),
            delta_convention: FxDeltaConvention::from_core(value.delta_convention),
            premium_currency: PremiumCurrency::from_core(value.premium_currency),
        }
    }
}

#[pymethods]
impl FxSmileMarketQuote {
    #[new]
    fn new(
        atm_vol: f64,
        pillars: Vec<FxRrBfPillar>,
        atm_convention: FxAtmConvention,
        delta_convention: FxDeltaConvention,
        premium_currency: PremiumCurrency,
    ) -> Self {
        Self {
            atm_vol,
            pillars,
            atm_convention,
            delta_convention,
            premium_currency,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_string_err)
    }

    fn sorted_pillars(&self) -> Vec<FxRrBfPillar> {
        self.to_core()
            .sorted_pillars()
            .into_iter()
            .map(FxRrBfPillar::from_core)
            .collect()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct FxVolExpiryQuote {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub smile: FxSmileMarketQuote,
}

impl FxVolExpiryQuote {
    fn to_core(&self) -> CoreFxVolExpiryQuote {
        CoreFxVolExpiryQuote {
            expiry: self.expiry,
            smile: self.smile.to_core(),
        }
    }
}

#[pymethods]
impl FxVolExpiryQuote {
    #[new]
    fn new(expiry: f64, smile: FxSmileMarketQuote) -> Self {
        Self { expiry, smile }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MalzInterpolator {
    inner: CoreMalzInterpolator,
}

#[pymethods]
impl MalzInterpolator {
    #[new]
    fn new(atm_vol: f64, pillars: Vec<FxRrBfPillar>) -> PyResult<Self> {
        let pillars = pillars
            .into_iter()
            .map(FxRrBfPillar::to_core)
            .collect::<Vec<_>>();
        CoreMalzInterpolator::new(atm_vol, &pillars)
            .map(|inner| Self { inner })
            .map_err(map_string_err)
    }

    #[staticmethod]
    fn quadratic_single_pillar(
        atm_vol: f64,
        risk_reversal: f64,
        butterfly: f64,
        pillar_delta: f64,
        signed_delta: f64,
    ) -> PyResult<f64> {
        CoreMalzInterpolator::quadratic_single_pillar(
            atm_vol,
            risk_reversal,
            butterfly,
            pillar_delta,
            signed_delta,
        )
        .map_err(map_string_err)
    }

    fn vol_at_signed_delta(&self, signed_delta: f64) -> f64 {
        self.inner.vol_at_signed_delta(signed_delta)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FxSmileSlice {
    inner: CoreFxSmileSlice,
}

impl FxSmileSlice {
    fn from_core(value: CoreFxSmileSlice) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl FxSmileSlice {
    #[new]
    fn new(
        expiry: f64,
        spot: f64,
        domestic_rate: f64,
        foreign_rate: f64,
        quote: FxSmileMarketQuote,
    ) -> PyResult<Self> {
        CoreFxSmileSlice::new(expiry, spot, domestic_rate, foreign_rate, quote.to_core())
            .map(Self::from_core)
            .map_err(map_string_err)
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }

    #[getter]
    fn domestic_rate(&self) -> f64 {
        self.inner.domestic_rate
    }

    #[getter]
    fn foreign_rate(&self) -> f64 {
        self.inner.foreign_rate
    }

    #[getter]
    fn quote(&self) -> FxSmileMarketQuote {
        FxSmileMarketQuote::from_core(self.inner.quote.clone())
    }

    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    fn atm_strike(&self) -> PyResult<f64> {
        self.inner.atm_strike().map_err(map_string_err)
    }

    fn vol_at_signed_delta(&self, signed_delta: f64) -> f64 {
        self.inner.vol_at_signed_delta(signed_delta)
    }

    fn strike_from_signed_delta(&self, signed_delta: f64) -> PyResult<f64> {
        self.inner
            .strike_from_signed_delta(signed_delta)
            .map_err(map_string_err)
    }

    fn vol_at_strike(&self, strike: f64) -> PyResult<f64> {
        self.inner.vol_at_strike(strike).map_err(map_string_err)
    }

    fn pillar_strikes(&self) -> PyResult<Vec<(f64, f64, f64)>> {
        self.inner.pillar_strikes().map_err(map_string_err)
    }

    fn reconstruct_pillars(&self) -> Vec<FxRrBfPillar> {
        self.inner
            .reconstruct_pillars()
            .into_iter()
            .map(FxRrBfPillar::from_core)
            .collect()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FxVolSurface {
    inner: CoreFxVolSurface,
}

impl FxVolSurface {
    #[allow(dead_code)]
    pub(crate) fn to_core(&self) -> CoreFxVolSurface {
        self.inner.clone()
    }
}

#[pymethods]
impl FxVolSurface {
    #[staticmethod]
    fn from_market_quotes(
        spot: f64,
        forward_curve: &FxForwardCurve,
        quotes: Vec<FxVolExpiryQuote>,
    ) -> PyResult<Self> {
        let quotes = quotes.into_iter().map(|quote| quote.to_core()).collect();
        CoreFxVolSurface::from_market_quotes(spot, forward_curve.to_core()?, quotes)
            .map(|inner| Self { inner })
            .map_err(map_string_err)
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }

    #[getter]
    fn slices(&self) -> Vec<FxSmileSlice> {
        self.inner
            .slices
            .iter()
            .cloned()
            .map(FxSmileSlice::from_core)
            .collect()
    }

    fn vol_at_signed_delta(&self, expiry: f64, signed_delta: f64) -> PyResult<f64> {
        self.inner
            .vol_at_signed_delta(expiry, signed_delta)
            .map_err(map_string_err)
    }

    fn vol(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        self.inner.vol(strike, expiry).map_err(map_string_err)
    }

    fn quote_from_surface(
        &self,
        expiry: f64,
        pillar_deltas: Vec<f64>,
    ) -> PyResult<FxSmileMarketQuote> {
        self.inner
            .quote_from_surface(expiry, &pillar_deltas)
            .map(FxSmileMarketQuote::from_core)
            .map_err(map_string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct NdfContract {
    #[pyo3(get, set)]
    pub pair: FxPair,
    #[pyo3(get, set)]
    pub notional_base: f64,
    #[pyo3(get, set)]
    pub agreed_forward: f64,
    #[pyo3(get, set)]
    pub settlement_currency: NdfSettlementCurrency,
    #[pyo3(get, set)]
    pub is_long_base: bool,
}

impl NdfContract {
    fn to_core(&self) -> CoreNdfContract {
        CoreNdfContract {
            pair: self.pair.to_core(),
            notional_base: self.notional_base,
            agreed_forward: self.agreed_forward,
            settlement_currency: self.settlement_currency.to_core(),
            is_long_base: self.is_long_base,
        }
    }
}

#[pymethods]
impl NdfContract {
    #[new]
    fn new(
        pair: FxPair,
        notional_base: f64,
        agreed_forward: f64,
        settlement_currency: NdfSettlementCurrency,
        is_long_base: bool,
    ) -> Self {
        Self {
            pair,
            notional_base,
            agreed_forward,
            settlement_currency,
            is_long_base,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_string_err)
    }

    fn settlement_amount(&self, fixing_rate: f64) -> PyResult<f64> {
        self.to_core()
            .settlement_amount(fixing_rate)
            .map_err(map_string_err)
    }

    fn present_value(
        &self,
        market_forward: f64,
        domestic_discount_rate: f64,
        foreign_discount_rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        self.to_core()
            .present_value(
                market_forward,
                domestic_discount_rate,
                foreign_discount_rate,
                expiry,
            )
            .map_err(map_string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct SampledVolSurface {
    #[pyo3(get, set)]
    pub strikes: Vec<f64>,
    #[pyo3(get, set)]
    pub expiries: Vec<f64>,
    #[pyo3(get, set)]
    pub vols: Vec<Vec<f64>>,
}

impl SampledVolSurface {
    pub(crate) fn to_core(&self) -> PyResult<CoreSampledVolSurface> {
        CoreSampledVolSurface::new(
            self.strikes.clone(),
            self.expiries.clone(),
            self.vols.clone(),
        )
        .map_err(map_string_err)
    }

    pub(crate) fn from_core(value: CoreSampledVolSurface) -> Self {
        Self {
            strikes: value.strikes,
            expiries: value.expiries,
            vols: value.vols,
        }
    }
}

#[pymethods]
impl SampledVolSurface {
    #[new]
    fn new(strikes: Vec<f64>, expiries: Vec<f64>, vols: Vec<Vec<f64>>) -> PyResult<Self> {
        let this = Self {
            strikes,
            expiries,
            vols,
        };
        let _ = this.to_core()?;
        Ok(this)
    }

    fn vol(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        Ok(self.to_core()?.vol(strike, expiry))
    }
}

#[derive(Clone)]
enum VolSourceInner {
    Flat(f64),
    Sampled(SampledVolSurface),
    Parametric {
        surface: CoreParametricVolSurface,
        slices: Vec<(f64, (f64, f64, f64, f64, f64))>,
        forward: f64,
    },
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VolSource {
    inner: VolSourceInner,
}

impl VolSource {
    pub(crate) fn to_core(&self) -> PyResult<CoreVolSource> {
        match &self.inner {
            VolSourceInner::Flat(vol) => Ok(CoreVolSource::Flat(*vol)),
            VolSourceInner::Sampled(surface) => Ok(CoreVolSource::Sampled(surface.to_core()?)),
            VolSourceInner::Parametric { surface, .. } => {
                Ok(CoreVolSource::Parametric(surface.clone()))
            }
        }
    }

    pub(crate) fn from_core(source: CoreVolSource, spot: f64) -> Self {
        let inner = match source {
            CoreVolSource::Flat(vol) => VolSourceInner::Flat(vol),
            CoreVolSource::Sampled(surface) => {
                VolSourceInner::Sampled(SampledVolSurface::from_core(surface))
            }
            CoreVolSource::Parametric(surface) => VolSourceInner::Parametric {
                slices: Vec::new(),
                forward: spot,
                surface,
            },
        };
        Self { inner }
    }
}

#[pymethods]
impl VolSource {
    #[staticmethod]
    fn flat(vol: f64) -> Self {
        Self {
            inner: VolSourceInner::Flat(vol),
        }
    }

    #[staticmethod]
    fn sampled(surface: &SampledVolSurface) -> Self {
        Self {
            inner: VolSourceInner::Sampled(surface.clone()),
        }
    }

    #[staticmethod]
    fn parametric(slices: Vec<(f64, (f64, f64, f64, f64, f64))>, forward: f64) -> PyResult<Self> {
        let core_slices = slices
            .iter()
            .map(|(expiry, (a, b, rho, m, sigma))| {
                (
                    *expiry,
                    SviParams {
                        a: *a,
                        b: *b,
                        rho: *rho,
                        m: *m,
                        sigma: *sigma,
                    },
                )
            })
            .collect::<Vec<_>>();
        let surface =
            CoreParametricVolSurface::new(core_slices, forward).map_err(map_string_err)?;
        Ok(Self {
            inner: VolSourceInner::Parametric {
                surface,
                slices,
                forward,
            },
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            VolSourceInner::Flat(_) => "Flat",
            VolSourceInner::Sampled(_) => "Sampled",
            VolSourceInner::Parametric { .. } => "Parametric",
        }
    }

    fn flat_vol(&self) -> Option<f64> {
        match self.inner {
            VolSourceInner::Flat(vol) => Some(vol),
            _ => None,
        }
    }

    fn sampled_surface(&self) -> Option<SampledVolSurface> {
        match &self.inner {
            VolSourceInner::Sampled(surface) => Some(surface.clone()),
            _ => None,
        }
    }

    fn parametric_spec(&self) -> Option<(Vec<(f64, (f64, f64, f64, f64, f64))>, f64)> {
        match &self.inner {
            VolSourceInner::Parametric {
                slices, forward, ..
            } => Some((slices.clone(), *forward)),
            _ => None,
        }
    }

    fn vol(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        Ok(self.to_core()?.vol(strike, expiry))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct Market {
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    dividend_schedule: DividendSchedule,
    vol: VolSource,
    reference_date: Option<String>,
}

impl Market {
    pub(crate) fn to_core(&self) -> PyResult<CoreMarket> {
        Ok(CoreMarket {
            spot: self.spot,
            rate: self.rate,
            dividend_yield: self.dividend_yield,
            dividend_schedule: self.dividend_schedule.to_core(),
            vol: self.vol.to_core()?,
            reference_date: self.reference_date.clone(),
        })
    }

    pub(crate) fn from_core(value: CoreMarket) -> Self {
        let spot = value.spot;
        Self {
            spot,
            rate: value.rate,
            dividend_yield: value.dividend_yield,
            dividend_schedule: DividendSchedule::from_core(value.dividend_schedule),
            vol: VolSource::from_core(value.vol, spot),
            reference_date: value.reference_date,
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
        dividend_schedule: Option<&DividendSchedule>,
        vol: &VolSource,
        reference_date: Option<String>,
    ) -> PyResult<Self> {
        let market = Self {
            spot,
            rate,
            dividend_yield,
            dividend_schedule: dividend_schedule.cloned().unwrap_or_default(),
            vol: vol.clone(),
            reference_date,
        };
        let _ = market.to_core()?;
        Ok(market)
    }

    #[staticmethod]
    fn builder() -> MarketBuilder {
        MarketBuilder::default()
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.spot
    }

    #[setter]
    fn set_spot(&mut self, spot: f64) {
        self.spot = spot;
    }

    #[getter]
    fn rate(&self) -> f64 {
        self.rate
    }

    #[setter]
    fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
    }

    #[getter]
    fn dividend_yield(&self) -> f64 {
        self.dividend_yield
    }

    #[setter]
    fn set_dividend_yield(&mut self, dividend_yield: f64) {
        self.dividend_yield = dividend_yield;
    }

    #[getter]
    fn dividend_schedule(&self) -> DividendSchedule {
        self.dividend_schedule.clone()
    }

    #[setter]
    fn set_dividend_schedule(&mut self, dividend_schedule: &DividendSchedule) {
        self.dividend_schedule = dividend_schedule.clone();
    }

    #[getter]
    fn vol_source(&self) -> VolSource {
        self.vol.clone()
    }

    #[setter]
    fn set_vol_source(&mut self, vol: &VolSource) {
        self.vol = vol.clone();
    }

    #[getter]
    fn reference_date(&self) -> Option<String> {
        self.reference_date.clone()
    }

    #[setter]
    fn set_reference_date(&mut self, reference_date: Option<String>) {
        self.reference_date = reference_date;
    }

    fn dividend(&self) -> f64 {
        self.dividend_yield
    }

    fn dividends(&self) -> DividendSchedule {
        self.dividend_schedule.clone()
    }

    fn has_discrete_dividends(&self) -> bool {
        !self.dividend_schedule.is_empty()
    }

    fn forward_price(&self, maturity: f64) -> PyResult<f64> {
        Ok(self.to_core()?.forward_price(maturity))
    }

    fn prepaid_forward_spot(&self, maturity: f64) -> PyResult<f64> {
        Ok(self.to_core()?.prepaid_forward_spot(maturity))
    }

    fn effective_dividend_yield(&self, maturity: f64) -> PyResult<f64> {
        Ok(self.to_core()?.effective_dividend_yield(maturity))
    }

    fn escrowed_spot_adjustment(&self, maturity: f64) -> PyResult<f64> {
        Ok(self.to_core()?.escrowed_spot_adjustment(maturity))
    }

    fn vol(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        Ok(self.to_core()?.vol(strike, expiry))
    }

    fn vol_for(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        self.vol(strike, expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Default)]
pub struct MarketBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    dividend_schedule: Option<DividendSchedule>,
    vol: Option<VolSource>,
    reference_date: Option<String>,
}

#[pymethods]
impl MarketBuilder {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn spot(&mut self, spot: f64) {
        self.spot = Some(spot);
    }

    fn rate(&mut self, rate: f64) {
        self.rate = Some(rate);
    }

    fn dividend_yield(&mut self, dividend_yield: f64) {
        self.dividend_yield = Some(dividend_yield);
    }

    fn dividend_schedule(&mut self, dividend_schedule: &DividendSchedule) {
        self.dividend_schedule = Some(dividend_schedule.clone());
    }

    fn flat_vol(&mut self, vol: f64) {
        self.vol = Some(VolSource::flat(vol));
    }

    fn sampled_vol_surface(&mut self, surface: &SampledVolSurface) {
        self.vol = Some(VolSource::sampled(surface));
    }

    fn parametric_vol_surface(
        &mut self,
        slices: Vec<(f64, (f64, f64, f64, f64, f64))>,
        forward: f64,
    ) -> PyResult<()> {
        self.vol = Some(VolSource::parametric(slices, forward)?);
        Ok(())
    }

    fn reference_date(&mut self, reference_date: String) {
        self.reference_date = Some(reference_date);
    }

    fn build(&self) -> PyResult<Market> {
        let market = Market {
            spot: self
                .spot
                .ok_or_else(|| PyValueError::new_err("market spot is required"))?,
            rate: self.rate.unwrap_or(0.0),
            dividend_yield: self.dividend_yield.unwrap_or(0.0),
            dividend_schedule: self.dividend_schedule.clone().unwrap_or_default(),
            vol: self
                .vol
                .clone()
                .ok_or_else(|| PyValueError::new_err("market volatility source is required"))?,
            reference_date: self.reference_date.clone(),
        };
        let _ = market.to_core()?;
        Ok(market)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct ForwardCurveSnapshot {
    #[pyo3(get, set)]
    pub asset_id: String,
    #[pyo3(get, set)]
    pub points: Vec<(f64, f64)>,
}

impl ForwardCurveSnapshot {
    pub(crate) fn to_core(&self) -> CoreForwardCurveSnapshot {
        CoreForwardCurveSnapshot {
            asset_id: self.asset_id.clone(),
            points: self.points.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_core(value: CoreForwardCurveSnapshot) -> Self {
        Self {
            asset_id: value.asset_id,
            points: value.points,
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
#[derive(Clone, PartialEq)]
pub struct CreditCurveSnapshot {
    #[pyo3(get, set)]
    pub curve_id: String,
    #[pyo3(get, set)]
    pub survival_curve: SurvivalCurve,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
}

impl CreditCurveSnapshot {
    pub(crate) fn to_core(&self) -> CoreCreditCurveSnapshot {
        CoreCreditCurveSnapshot {
            curve_id: self.curve_id.clone(),
            survival_curve: self.survival_curve.to_core(),
            recovery_rate: self.recovery_rate,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_core(value: CoreCreditCurveSnapshot) -> Self {
        Self {
            curve_id: value.curve_id,
            survival_curve: SurvivalCurve::from_core(value.survival_curve),
            recovery_rate: value.recovery_rate,
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
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MarketSnapshot {
    #[pyo3(get, set)]
    pub snapshot_id: String,
    #[pyo3(get, set)]
    pub timestamp_unix_ms: i64,
    #[pyo3(get, set)]
    pub markets: Vec<(String, Market)>,
    #[pyo3(get, set)]
    pub yield_curves: Vec<(String, Vec<(f64, f64)>)>,
    #[pyo3(get, set)]
    pub vol_surfaces: Vec<(String, SampledVolSurface)>,
    #[pyo3(get, set)]
    pub credit_curves: Vec<CreditCurveSnapshot>,
    #[pyo3(get, set)]
    pub spot_prices: Vec<(String, f64)>,
    #[pyo3(get, set)]
    pub forward_curves: Vec<ForwardCurveSnapshot>,
}

impl MarketSnapshot {
    pub(crate) fn to_core(&self) -> PyResult<CoreMarketSnapshot> {
        Ok(CoreMarketSnapshot {
            snapshot_id: self.snapshot_id.clone(),
            timestamp_unix_ms: self.timestamp_unix_ms,
            markets: self
                .markets
                .iter()
                .map(|(id, market)| Ok((id.clone(), market.to_core()?)))
                .collect::<PyResult<_>>()?,
            yield_curves: self
                .yield_curves
                .iter()
                .map(|(id, points)| (id.clone(), CoreYieldCurve::new(points.clone())))
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
        })
    }

    #[allow(dead_code)]
    pub(crate) fn from_core(value: CoreMarketSnapshot) -> Self {
        let CoreMarketSnapshot {
            snapshot_id,
            timestamp_unix_ms,
            markets,
            yield_curves,
            vol_surfaces,
            credit_curves,
            spot_prices,
            forward_curves,
        } = value;
        let market_spots = markets
            .iter()
            .map(|(asset_id, market)| (asset_id.clone(), market.spot))
            .collect::<Vec<_>>();
        Self {
            snapshot_id,
            timestamp_unix_ms,
            markets: markets
                .into_iter()
                .map(|(id, market)| (id, Market::from_core(market)))
                .collect(),
            yield_curves: yield_curves
                .into_iter()
                .map(|(id, curve)| (id, curve.tenors))
                .collect(),
            vol_surfaces: vol_surfaces
                .into_iter()
                .map(|(id, surface)| {
                    let spot = spot_prices
                        .iter()
                        .find(|(asset_id, _)| asset_id == &id)
                        .map(|(_, spot)| *spot)
                        .or_else(|| {
                            market_spots
                                .iter()
                                .find(|(asset_id, _)| asset_id == &id)
                                .map(|(_, spot)| *spot)
                        })
                        .unwrap_or(1.0);
                    (
                        id,
                        SampledVolSurface::from_core(CoreSampledVolSurface::from_surface(
                            &surface, spot,
                        )),
                    )
                })
                .collect(),
            credit_curves: credit_curves
                .into_iter()
                .map(CreditCurveSnapshot::from_core)
                .collect(),
            spot_prices,
            forward_curves: forward_curves
                .into_iter()
                .map(ForwardCurveSnapshot::from_core)
                .collect(),
        }
    }
}

#[pymethods]
impl MarketSnapshot {
    #[new]
    fn new(snapshot_id: String, timestamp_unix_ms: i64) -> Self {
        Self {
            snapshot_id,
            timestamp_unix_ms,
            markets: Vec::new(),
            yield_curves: Vec::new(),
            vol_surfaces: Vec::new(),
            credit_curves: Vec::new(),
            spot_prices: Vec::new(),
            forward_curves: Vec::new(),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DividendKind>()?;
    module.add_class::<DividendEvent>()?;
    module.add_class::<DividendSchedule>()?;
    module.add_class::<PutCallParityQuote>()?;
    module.add_class::<BootstrappedDividendPoint>()?;
    module.add_class::<DividendCurveBootstrap>()?;
    module.add_class::<PremiumCurrency>()?;
    module.add_class::<FxDeltaConvention>()?;
    module.add_class::<FxAtmConvention>()?;
    module.add_class::<NdfSettlementCurrency>()?;
    module.add_class::<FxPair>()?;
    module.add_class::<FxForwardCurve>()?;
    module.add_class::<FxRrBfPillar>()?;
    module.add_class::<FxSmileMarketQuote>()?;
    module.add_class::<FxVolExpiryQuote>()?;
    module.add_class::<MalzInterpolator>()?;
    module.add_class::<FxSmileSlice>()?;
    module.add_class::<FxVolSurface>()?;
    module.add_class::<NdfContract>()?;
    module.add_class::<SampledVolSurface>()?;
    module.add_class::<VolSource>()?;
    module.add_class::<Market>()?;
    module.add_class::<MarketBuilder>()?;
    module.add_class::<ForwardCurveSnapshot>()?;
    module.add_class::<CreditCurveSnapshot>()?;
    module.add_class::<MarketSnapshot>()?;
    Ok(())
}
