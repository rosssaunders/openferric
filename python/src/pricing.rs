use openferric_core::core::{Greeks as CoreGreeks, PricingEngine, PricingError};
use openferric_core::engines::analytic::{
    DigitalAnalyticEngine, ExoticAnalyticEngine, GarmanKohlhagenEngine, kirk_spread_price,
    margrabe_exchange_price,
};
use openferric_core::greeks::black_scholes_merton_greeks;
use openferric_core::instruments::{
    AbandonmentOption as CoreAbandonmentOption, AssetOrNothingOption, BasketType as CoreBasketType,
    CashOrNothingOption, DeferInvestmentOption as CoreDeferInvestmentOption, ExoticOption,
    ExpandOption as CoreExpandOption, FxOption, LookbackFixedOption, LookbackFloatingOption,
    SpreadOption, TarfType as CoreTarfType,
};
use openferric_core::pricing::american::{crr_binomial_american, longstaff_schwartz_american_put};
use openferric_core::pricing::asian::{
    AsianStrike as CoreAsianStrike, arithmetic_asian_price_mc,
    geometric_asian_discrete_fixed_closed_form, geometric_asian_fixed_closed_form,
    geometric_asian_price_mc,
};
use openferric_core::pricing::autocallable::{
    AutocallableSensitivities as CoreAutocallableSensitivities, autocallable_sensitivities,
    phoenix_autocallable_sensitivities, price_autocallable, price_phoenix_autocallable,
};
use openferric_core::pricing::barrier::{
    barrier_price_closed_form, barrier_price_closed_form_with_carry_and_rebate, barrier_price_mc,
};
use openferric_core::pricing::basket::{
    BasketCopula as CoreBasketCopula, BasketMomentMatchingMethod as CoreBasketMomentMatchingMethod,
    BasketSensitivities as CoreBasketSensitivities, basket_sensitivities, price_basket_mc,
    price_basket_mc_with_copula, price_basket_mc_with_factor_model, price_basket_moment_matching,
    price_outperformance_basket_mc, price_quanto_basket_mc, stressed_correlation_matrix,
};
use openferric_core::pricing::bermudan::longstaff_schwartz_bermudan;
use openferric_core::pricing::discrete_div::{
    bootstrap_dividend_curve, effective_dividend_yield_discrete, escrowed_dividend_adjusted_spot,
    escrowed_dividend_adjusted_spot_mixed, european_price_discrete_div,
    european_price_discrete_div_mixed, forward_price_discrete_div,
};
use openferric_core::pricing::european::{black_76_price, black_scholes_price};
use openferric_core::pricing::payoff::strategy_intrinsic_pnl;
use openferric_core::pricing::range_accrual::{
    RangeAccrualResult as CoreRangeAccrualResult, dual_range_accrual_mc_price,
    range_accrual_mc_price, range_accrual_rate_delta,
};
use openferric_core::pricing::real_option::{
    DecisionTreeNode as CoreDecisionTreeNode, RealOptionDecision as CoreRealOptionDecision,
    RealOptionValuation as CoreRealOptionValuation, european_abandonment_put,
    price_option_to_abandon, price_option_to_defer, price_option_to_expand,
};
use openferric_core::pricing::tarf::{
    TarfPricingResult as CoreTarfPricingResult, tarf_delta, tarf_mc_price, tarf_vega,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::PricingResult;
use crate::fft::heston_fft_prices_cached;
use crate::helpers::{
    DigitalKind, SpreadMethod, build_market, catch_unwind_py, intrinsic_from_option_type,
    option_price_from_call, parse_barrier_direction, parse_barrier_style, parse_digital_kind,
    parse_option_type, parse_spread_method,
};
use crate::instruments::{
    Autocallable, BasketOption, DualRangeAccrual, OutperformanceBasketOption, PhoenixAutocallable,
    QuantoBasketOption, RangeAccrual, RealOptionBinomialSpec, Tarf,
};
use crate::market::{DividendCurveBootstrap, DividendSchedule, PutCallParityQuote};
use crate::math_bindings::{CorrelationStressScenario, FactorCorrelationModel};

fn pricing_error_to_pyerr(err: PricingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn string_error_to_pyerr(err: String) -> PyErr {
    PyValueError::new_err(err)
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct PricingGreeks {
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub rho: f64,
}

impl PricingGreeks {
    #[allow(dead_code)]
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
impl PricingGreeks {
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

    #[allow(clippy::wrong_self_convention)]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("delta", self.delta)?;
        dict.set_item("gamma", self.gamma)?;
        dict.set_item("vega", self.vega)?;
        dict.set_item("theta", self.theta)?;
        dict.set_item("rho", self.rho)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingGreeks(delta={}, gamma={}, vega={}, theta={}, rho={})",
            self.delta, self.gamma, self.vega, self.theta, self.rho
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AsianStrike {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub value: Option<f64>,
}

impl AsianStrike {
    fn to_core(&self) -> PyResult<CoreAsianStrike> {
        match self.kind.to_ascii_lowercase().as_str() {
            "fixed" => self
                .value
                .map(CoreAsianStrike::Fixed)
                .ok_or_else(|| PyValueError::new_err("fixed AsianStrike requires a value")),
            "floating" => Ok(CoreAsianStrike::Floating),
            _ => Err(PyValueError::new_err(format!(
                "unsupported AsianStrike kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl AsianStrike {
    #[staticmethod]
    fn fixed(value: f64) -> Self {
        Self {
            kind: "fixed".to_string(),
            value: Some(value),
        }
    }

    #[staticmethod]
    fn floating() -> Self {
        Self {
            kind: "floating".to_string(),
            value: None,
        }
    }

    fn __repr__(&self) -> String {
        match self.value {
            Some(value) => format!("AsianStrike(kind={:?}, value={})", self.kind, value),
            None => format!("AsianStrike(kind={:?})", self.kind),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AutocallableSensitivities {
    #[pyo3(get)]
    pub delta: Vec<f64>,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub cega: f64,
}

impl AutocallableSensitivities {
    fn from_core(value: CoreAutocallableSensitivities) -> Self {
        Self {
            delta: value.delta,
            vega: value.vega,
            cega: value.cega,
        }
    }
}

#[pymethods]
impl AutocallableSensitivities {
    #[new]
    fn new(delta: Vec<f64>, vega: f64, cega: f64) -> Self {
        Self { delta, vega, cega }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("delta", self.delta.clone())?;
        dict.set_item("vega", self.vega)?;
        dict.set_item("cega", self.cega)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "AutocallableSensitivities(delta={:?}, vega={}, cega={})",
            self.delta, self.vega, self.cega
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BasketType {
    #[pyo3(get)]
    pub kind: String,
}

impl BasketType {
    #[allow(dead_code)]
    fn to_core(&self) -> PyResult<CoreBasketType> {
        match self.kind.to_ascii_lowercase().as_str() {
            "average" => Ok(CoreBasketType::Average),
            "best_of" | "bestof" => Ok(CoreBasketType::BestOf),
            "worst_of" | "worstof" => Ok(CoreBasketType::WorstOf),
            _ => Err(PyValueError::new_err(format!(
                "unsupported BasketType kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl BasketType {
    #[staticmethod]
    fn average() -> Self {
        Self {
            kind: "average".to_string(),
        }
    }

    #[staticmethod]
    fn best_of() -> Self {
        Self {
            kind: "best_of".to_string(),
        }
    }

    #[staticmethod]
    fn worst_of() -> Self {
        Self {
            kind: "worst_of".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("BasketType(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BasketCopula {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub degrees_of_freedom: Option<u32>,
}

impl BasketCopula {
    fn to_core(&self) -> PyResult<CoreBasketCopula> {
        match self.kind.to_ascii_lowercase().as_str() {
            "gaussian" => Ok(CoreBasketCopula::Gaussian),
            "student_t" | "studentt" => Ok(CoreBasketCopula::StudentT {
                degrees_of_freedom: self.degrees_of_freedom.ok_or_else(|| {
                    PyValueError::new_err("student_t BasketCopula requires degrees_of_freedom")
                })?,
            }),
            _ => Err(PyValueError::new_err(format!(
                "unsupported BasketCopula kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl BasketCopula {
    #[staticmethod]
    fn gaussian() -> Self {
        Self {
            kind: "gaussian".to_string(),
            degrees_of_freedom: None,
        }
    }

    #[staticmethod]
    fn student_t(degrees_of_freedom: u32) -> Self {
        Self {
            kind: "student_t".to_string(),
            degrees_of_freedom: Some(degrees_of_freedom),
        }
    }

    fn __repr__(&self) -> String {
        match self.degrees_of_freedom {
            Some(df) => format!(
                "BasketCopula(kind={:?}, degrees_of_freedom={})",
                self.kind, df
            ),
            None => format!("BasketCopula(kind={:?})", self.kind),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BasketMomentMatchingMethod {
    #[pyo3(get)]
    pub kind: String,
}

impl BasketMomentMatchingMethod {
    fn to_core(&self) -> PyResult<CoreBasketMomentMatchingMethod> {
        match self.kind.to_ascii_lowercase().as_str() {
            "levy" => Ok(CoreBasketMomentMatchingMethod::Levy),
            "gentle" => Ok(CoreBasketMomentMatchingMethod::Gentle),
            _ => Err(PyValueError::new_err(format!(
                "unsupported BasketMomentMatchingMethod kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl BasketMomentMatchingMethod {
    #[staticmethod]
    fn levy() -> Self {
        Self {
            kind: "levy".to_string(),
        }
    }

    #[staticmethod]
    fn gentle() -> Self {
        Self {
            kind: "gentle".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("BasketMomentMatchingMethod(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BasketSensitivities {
    #[pyo3(get)]
    pub delta: Vec<f64>,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub cega: f64,
}

impl BasketSensitivities {
    fn from_core(value: CoreBasketSensitivities) -> Self {
        Self {
            delta: value.delta,
            vega: value.vega,
            cega: value.cega,
        }
    }
}

#[pymethods]
impl BasketSensitivities {
    #[new]
    fn new(delta: Vec<f64>, vega: f64, cega: f64) -> Self {
        Self { delta, vega, cega }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("delta", self.delta.clone())?;
        dict.set_item("vega", self.vega)?;
        dict.set_item("cega", self.cega)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "BasketSensitivities(delta={:?}, vega={}, cega={})",
            self.delta, self.vega, self.cega
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RangeAccrualResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub expected_accrual_fraction: f64,
}

impl RangeAccrualResult {
    fn from_core(result: CoreRangeAccrualResult) -> Self {
        Self {
            price: result.price,
            std_error: result.std_error,
            expected_accrual_fraction: result.expected_accrual_fraction,
        }
    }
}

#[pymethods]
impl RangeAccrualResult {
    #[new]
    fn new(price: f64, std_error: f64, expected_accrual_fraction: f64) -> Self {
        Self {
            price,
            std_error,
            expected_accrual_fraction,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("price", self.price)?;
        dict.set_item("std_error", self.std_error)?;
        dict.set_item("expected_accrual_fraction", self.expected_accrual_fraction)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeAccrualResult(price={}, std_error={}, expected_accrual_fraction={})",
            self.price, self.std_error, self.expected_accrual_fraction
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TarfType {
    #[pyo3(get)]
    pub kind: String,
}

impl TarfType {
    #[allow(dead_code)]
    fn to_core(&self) -> PyResult<CoreTarfType> {
        match self.kind.to_ascii_lowercase().as_str() {
            "standard" => Ok(CoreTarfType::Standard),
            "decumulator" => Ok(CoreTarfType::Decumulator),
            _ => Err(PyValueError::new_err(format!(
                "unsupported TarfType kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl TarfType {
    #[staticmethod]
    fn standard() -> Self {
        Self {
            kind: "standard".to_string(),
        }
    }

    #[staticmethod]
    fn decumulator() -> Self {
        Self {
            kind: "decumulator".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("TarfType(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TarfPricingResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub avg_fixings: f64,
    #[pyo3(get)]
    pub prob_target_hit: f64,
    #[pyo3(get)]
    pub prob_ko_hit: f64,
}

impl TarfPricingResult {
    fn from_core(result: CoreTarfPricingResult) -> Self {
        Self {
            price: result.price,
            std_error: result.std_error,
            avg_fixings: result.avg_fixings,
            prob_target_hit: result.prob_target_hit,
            prob_ko_hit: result.prob_ko_hit,
        }
    }
}

#[pymethods]
impl TarfPricingResult {
    #[new]
    fn new(
        price: f64,
        std_error: f64,
        avg_fixings: f64,
        prob_target_hit: f64,
        prob_ko_hit: f64,
    ) -> Self {
        Self {
            price,
            std_error,
            avg_fixings,
            prob_target_hit,
            prob_ko_hit,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("price", self.price)?;
        dict.set_item("std_error", self.std_error)?;
        dict.set_item("avg_fixings", self.avg_fixings)?;
        dict.set_item("prob_target_hit", self.prob_target_hit)?;
        dict.set_item("prob_ko_hit", self.prob_ko_hit)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "TarfPricingResult(price={}, std_error={}, avg_fixings={}, prob_target_hit={}, prob_ko_hit={})",
            self.price, self.std_error, self.avg_fixings, self.prob_target_hit, self.prob_ko_hit
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DeferInvestmentOption {
    model: RealOptionBinomialSpec,
    #[pyo3(get, set)]
    pub investment_cost: f64,
}

impl DeferInvestmentOption {
    fn to_core(&self) -> CoreDeferInvestmentOption {
        CoreDeferInvestmentOption {
            model: self.model.to_core(),
            investment_cost: self.investment_cost,
        }
    }
}

#[pymethods]
impl DeferInvestmentOption {
    #[new]
    fn new(model: &RealOptionBinomialSpec, investment_cost: f64) -> Self {
        Self {
            model: model.clone(),
            investment_cost,
        }
    }

    fn model(&self) -> RealOptionBinomialSpec {
        self.model.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "DeferInvestmentOption(investment_cost={})",
            self.investment_cost
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ExpandOption {
    model: RealOptionBinomialSpec,
    #[pyo3(get, set)]
    pub expansion_multiplier: f64,
    #[pyo3(get, set)]
    pub expansion_cost: f64,
}

impl ExpandOption {
    fn to_core(&self) -> CoreExpandOption {
        CoreExpandOption {
            model: self.model.to_core(),
            expansion_multiplier: self.expansion_multiplier,
            expansion_cost: self.expansion_cost,
        }
    }
}

#[pymethods]
impl ExpandOption {
    #[new]
    fn new(model: &RealOptionBinomialSpec, expansion_multiplier: f64, expansion_cost: f64) -> Self {
        Self {
            model: model.clone(),
            expansion_multiplier,
            expansion_cost,
        }
    }

    fn model(&self) -> RealOptionBinomialSpec {
        self.model.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpandOption(expansion_multiplier={}, expansion_cost={})",
            self.expansion_multiplier, self.expansion_cost
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AbandonmentOption {
    model: RealOptionBinomialSpec,
    #[pyo3(get, set)]
    pub salvage_value: f64,
}

impl AbandonmentOption {
    fn to_core(&self) -> CoreAbandonmentOption {
        CoreAbandonmentOption {
            model: self.model.to_core(),
            salvage_value: self.salvage_value,
        }
    }
}

#[pymethods]
impl AbandonmentOption {
    #[new]
    fn new(model: &RealOptionBinomialSpec, salvage_value: f64) -> Self {
        Self {
            model: model.clone(),
            salvage_value,
        }
    }

    fn model(&self) -> RealOptionBinomialSpec {
        self.model.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("AbandonmentOption(salvage_value={})", self.salvage_value)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RealOptionDecision {
    #[pyo3(get)]
    pub kind: String,
}

impl RealOptionDecision {
    fn from_core(value: CoreRealOptionDecision) -> Self {
        let kind = match value {
            CoreRealOptionDecision::Invest => "invest",
            CoreRealOptionDecision::Defer => "defer",
            CoreRealOptionDecision::Abandon => "abandon",
        };
        Self {
            kind: kind.to_string(),
        }
    }
}

#[pymethods]
impl RealOptionDecision {
    #[staticmethod]
    fn invest() -> Self {
        Self {
            kind: "invest".to_string(),
        }
    }

    #[staticmethod]
    fn defer() -> Self {
        Self {
            kind: "defer".to_string(),
        }
    }

    #[staticmethod]
    fn abandon() -> Self {
        Self {
            kind: "abandon".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("RealOptionDecision(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DecisionTreeNode {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub project_value: f64,
    #[pyo3(get)]
    pub invest_value: f64,
    #[pyo3(get)]
    pub defer_value: f64,
    #[pyo3(get)]
    pub abandon_value: f64,
    #[pyo3(get)]
    pub option_value: f64,
    decision: RealOptionDecision,
}

impl DecisionTreeNode {
    fn from_core(value: CoreDecisionTreeNode) -> Self {
        Self {
            time: value.time,
            project_value: value.project_value,
            invest_value: value.invest_value,
            defer_value: value.defer_value,
            abandon_value: value.abandon_value,
            option_value: value.option_value,
            decision: RealOptionDecision::from_core(value.decision),
        }
    }
}

#[pymethods]
impl DecisionTreeNode {
    fn decision(&self) -> RealOptionDecision {
        self.decision.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DecisionTreeNode(time={}, project_value={}, option_value={}, decision={:?})",
            self.time, self.project_value, self.option_value, self.decision.kind
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RealOptionValuation {
    #[pyo3(get)]
    pub price: f64,
    nodes: Vec<Vec<DecisionTreeNode>>,
}

impl RealOptionValuation {
    fn from_core(value: CoreRealOptionValuation) -> Self {
        Self {
            price: value.price,
            nodes: value
                .nodes
                .into_iter()
                .map(|level| level.into_iter().map(DecisionTreeNode::from_core).collect())
                .collect(),
        }
    }
}

#[pymethods]
impl RealOptionValuation {
    fn nodes(&self) -> Vec<Vec<DecisionTreeNode>> {
        self.nodes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RealOptionValuation(price={}, levels={})",
            self.price,
            self.nodes.len()
        )
    }
}

#[pyfunction]
pub fn py_bs_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    black_scholes_price(option_type, spot, strike, rate, vol, expiry)
}

#[pyfunction]
pub fn py_black76_price(
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    black_76_price(option_type, forward, strike, rate, vol, expiry)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_bs_greeks(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    greek: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let greeks =
        black_scholes_merton_greeks(option_type, spot, strike, rate, div_yield, vol, expiry);

    match greek.to_ascii_lowercase().as_str() {
        "delta" => greeks.delta,
        "gamma" => greeks.gamma,
        "vega" => greeks.vega,
        "theta" => greeks.theta,
        "rho" => greeks.rho,
        "vanna" => greeks.vanna,
        "volga" | "vomma" => greeks.volga,
        _ => f64::NAN,
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_barrier_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    barrier: f64,
    option_type: &str,
    barrier_type: &str,
    barrier_dir: &str,
    rebate: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    let Some(style) = parse_barrier_style(barrier_type) else {
        return f64::NAN;
    };
    let Some(direction) = parse_barrier_direction(barrier_dir) else {
        return f64::NAN;
    };

    barrier_price_closed_form_with_carry_and_rebate(
        option_type,
        style,
        direction,
        spot,
        strike,
        barrier,
        rate,
        div_yield,
        vol,
        expiry,
        rebate,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_barrier_price_closed_form(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    barrier: f64,
    option_type: &str,
    barrier_type: &str,
    barrier_dir: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    let Some(style) = parse_barrier_style(barrier_type) else {
        return f64::NAN;
    };
    let Some(direction) = parse_barrier_direction(barrier_dir) else {
        return f64::NAN;
    };

    barrier_price_closed_form(
        option_type,
        style,
        direction,
        spot,
        strike,
        barrier,
        rate,
        vol,
        expiry,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_barrier_price_mc(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    barrier: f64,
    option_type: &str,
    barrier_type: &str,
    barrier_dir: &str,
    steps: usize,
    num_paths: usize,
    seed: u64,
) -> PyResult<(f64, f64)> {
    let Some(option_type) = parse_option_type(option_type) else {
        return Ok((f64::NAN, f64::NAN));
    };
    let Some(style) = parse_barrier_style(barrier_type) else {
        return Ok((f64::NAN, f64::NAN));
    };
    let Some(direction) = parse_barrier_direction(barrier_dir) else {
        return Ok((f64::NAN, f64::NAN));
    };

    catch_unwind_py(|| {
        barrier_price_mc(
            option_type,
            style,
            direction,
            spot,
            strike,
            barrier,
            rate,
            vol,
            expiry,
            steps.max(1),
            num_paths,
            seed,
        )
    })
}

#[pyfunction]
pub fn py_american_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    option_type: &str,
    steps: usize,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    crr_binomial_american(option_type, spot, strike, rate, vol, expiry, steps.max(1))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_longstaff_schwartz_american_put(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    steps: usize,
    num_paths: usize,
    seed: u64,
) -> PyResult<f64> {
    catch_unwind_py(|| {
        longstaff_schwartz_american_put(spot, strike, rate, vol, expiry, steps, num_paths, seed)
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_heston_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    if expiry <= 0.0 {
        return intrinsic_from_option_type(option_type, spot, strike);
    }

    let call_price = heston_fft_prices_cached(
        spot,
        &[strike],
        expiry,
        rate,
        div_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
    )
    .and_then(|pairs| pairs.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN);

    if !call_price.is_finite() {
        return f64::NAN;
    }

    option_price_from_call(
        option_type,
        call_price,
        spot,
        strike,
        rate,
        div_yield,
        expiry,
    )
}

#[pyfunction]
pub fn py_fx_price(
    spot_fx: f64,
    strike_fx: f64,
    maturity: f64,
    vol: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let instrument = FxOption::new(
        option_type,
        domestic_rate,
        foreign_rate,
        spot_fx,
        strike_fx,
        vol,
        maturity,
    );

    let Some(market) = build_market(spot_fx, domestic_rate, foreign_rate, vol) else {
        return f64::NAN;
    };

    GarmanKohlhagenEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_digital_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    digital_type: &str,
    cash: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    let Some(digital_type) = parse_digital_kind(digital_type) else {
        return f64::NAN;
    };

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    let engine = DigitalAnalyticEngine::new();

    match digital_type {
        DigitalKind::CashOrNothing => {
            let instrument = CashOrNothingOption::new(option_type, strike, cash, expiry);
            engine
                .price(&instrument, &market)
                .map(|x| x.price)
                .unwrap_or(f64::NAN)
        }
        DigitalKind::AssetOrNothing => {
            let instrument = AssetOrNothingOption::new(option_type, strike, expiry);
            engine
                .price(&instrument, &market)
                .map(|x| x.price)
                .unwrap_or(f64::NAN)
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_spread_price(
    s1: f64,
    s2: f64,
    k: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    q1: f64,
    q2: f64,
    r: f64,
    t: f64,
    method: &str,
) -> f64 {
    let Some(method) = parse_spread_method(method) else {
        return f64::NAN;
    };

    let option = SpreadOption {
        s1,
        s2,
        k,
        vol1,
        vol2,
        rho,
        q1,
        q2,
        r,
        t,
    };

    match method {
        SpreadMethod::Kirk => kirk_spread_price(&option),
        SpreadMethod::Margrabe => margrabe_exchange_price(&option),
    }
    .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_lookback_floating(
    spot: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    observed_extreme: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let observed_extreme = if observed_extreme > 0.0 {
        Some(observed_extreme)
    } else {
        None
    };

    let instrument = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type,
        expiry,
        observed_extreme,
    });

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    ExoticAnalyticEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_lookback_fixed(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    observed_extreme: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let observed_extreme = if observed_extreme > 0.0 {
        Some(observed_extreme)
    } else {
        None
    };

    let instrument = ExoticOption::LookbackFixed(LookbackFixedOption {
        option_type,
        strike,
        expiry,
        observed_extreme,
    });

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    ExoticAnalyticEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_geometric_asian_fixed_closed_form(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    geometric_asian_fixed_closed_form(option_type, spot, strike, rate, vol, expiry)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_geometric_asian_discrete_fixed_closed_form(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    observation_times: Vec<f64>,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    geometric_asian_discrete_fixed_closed_form(
        option_type,
        spot,
        strike,
        rate,
        div_yield,
        vol,
        &observation_times,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_arithmetic_asian_price_mc(
    strike: &AsianStrike,
    spot: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    steps: usize,
    num_paths: usize,
    seed: u64,
    option_type: &str,
) -> PyResult<(f64, f64)> {
    let option_type = parse_option_type(option_type)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported option type '{option_type}'")))?;
    Ok(arithmetic_asian_price_mc(
        option_type,
        strike.to_core()?,
        spot,
        rate,
        vol,
        expiry,
        steps.max(1),
        num_paths,
        seed,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_geometric_asian_price_mc(
    strike: &AsianStrike,
    spot: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    steps: usize,
    num_paths: usize,
    seed: u64,
    option_type: &str,
) -> PyResult<(f64, f64)> {
    let option_type = parse_option_type(option_type)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported option type '{option_type}'")))?;
    Ok(geometric_asian_price_mc(
        option_type,
        strike.to_core()?,
        spot,
        rate,
        vol,
        expiry,
        steps.max(1),
        num_paths,
        seed,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_autocallable(
    autocall: &Autocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    div_yield: f64,
    num_paths: usize,
    num_steps: usize,
) -> PricingResult {
    PricingResult::from_core(price_autocallable(
        &autocall.to_core(),
        &spots,
        &vols,
        &corr_matrix,
        rate,
        div_yield,
        num_paths,
        num_steps,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_autocallable_sensitivities(
    autocall: &Autocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    div_yield: f64,
    num_paths: usize,
    num_steps: usize,
) -> PyResult<AutocallableSensitivities> {
    autocallable_sensitivities(
        &autocall.to_core(),
        &spots,
        &vols,
        &corr_matrix,
        rate,
        div_yield,
        num_paths,
        num_steps,
    )
    .map(AutocallableSensitivities::from_core)
    .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_phoenix_autocallable(
    phoenix: &PhoenixAutocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    div_yield: f64,
    num_paths: usize,
    num_steps: usize,
) -> PricingResult {
    PricingResult::from_core(price_phoenix_autocallable(
        &phoenix.to_core(),
        &spots,
        &vols,
        &corr_matrix,
        rate,
        div_yield,
        num_paths,
        num_steps,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_phoenix_autocallable_sensitivities(
    phoenix: &PhoenixAutocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    div_yield: f64,
    num_paths: usize,
    num_steps: usize,
) -> PyResult<AutocallableSensitivities> {
    phoenix_autocallable_sensitivities(
        &phoenix.to_core(),
        &spots,
        &vols,
        &corr_matrix,
        rate,
        div_yield,
        num_paths,
        num_steps,
    )
    .map(AutocallableSensitivities::from_core)
    .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_basket_mc(
    basket: &BasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividends: Vec<f64>,
    num_paths: usize,
) -> PyResult<PricingResult> {
    Ok(PricingResult::from_core(price_basket_mc(
        &basket.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        rate,
        &dividends,
        num_paths,
    )))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_basket_mc_with_copula(
    basket: &BasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividends: Vec<f64>,
    num_paths: usize,
    copula: &BasketCopula,
) -> PyResult<PricingResult> {
    Ok(PricingResult::from_core(price_basket_mc_with_copula(
        &basket.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        rate,
        &dividends,
        num_paths,
        copula.to_core()?,
    )))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_basket_mc_with_factor_model(
    basket: &BasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    factor_model: &FactorCorrelationModel,
    rate: f64,
    dividends: Vec<f64>,
    num_paths: usize,
    copula: &BasketCopula,
) -> PyResult<PricingResult> {
    Ok(PricingResult::from_core(price_basket_mc_with_factor_model(
        &basket.to_core()?,
        &spots,
        &vols,
        &factor_model.to_core()?,
        rate,
        &dividends,
        num_paths,
        copula.to_core()?,
    )))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_basket_moment_matching(
    basket: &BasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividends: Vec<f64>,
    method: &BasketMomentMatchingMethod,
) -> PyResult<f64> {
    price_basket_moment_matching(
        &basket.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        rate,
        &dividends,
        method.to_core()?,
    )
    .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_outperformance_basket_mc(
    option: &OutperformanceBasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividends: Vec<f64>,
    num_paths: usize,
) -> PyResult<PricingResult> {
    Ok(PricingResult::from_core(price_outperformance_basket_mc(
        &option.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        rate,
        &dividends,
        num_paths,
    )))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_price_quanto_basket_mc(
    option: &QuantoBasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    dividends: Vec<f64>,
    num_paths: usize,
) -> PyResult<PricingResult> {
    Ok(PricingResult::from_core(price_quanto_basket_mc(
        &option.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        &dividends,
        num_paths,
    )))
}

#[pyfunction]
pub fn py_stressed_correlation_matrix(
    corr_matrix: Vec<Vec<f64>>,
    scenarios: Vec<CorrelationStressScenario>,
) -> PyResult<Vec<Vec<f64>>> {
    let scenarios = scenarios
        .iter()
        .map(CorrelationStressScenario::to_core)
        .collect::<Vec<_>>();
    stressed_correlation_matrix(&corr_matrix, &scenarios).map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_basket_sensitivities(
    basket: &BasketOption,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividends: Vec<f64>,
    num_paths: usize,
) -> PyResult<BasketSensitivities> {
    basket_sensitivities(
        &basket.to_core()?,
        &spots,
        &vols,
        &corr_matrix,
        rate,
        &dividends,
        num_paths,
    )
    .map(BasketSensitivities::from_core)
    .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_bermudan_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    steps: usize,
    exercise_steps: Vec<usize>,
    num_paths: usize,
    seed: u64,
    option_type: &str,
) -> PyResult<f64> {
    let option_type = parse_option_type(option_type)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported option type '{option_type}'")))?;
    catch_unwind_py(|| {
        longstaff_schwartz_bermudan(
            option_type,
            spot,
            strike,
            rate,
            vol,
            expiry,
            steps,
            &exercise_steps,
            num_paths,
            seed,
        )
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_range_accrual_mc_price(
    instrument: &RangeAccrual,
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> PyResult<RangeAccrualResult> {
    range_accrual_mc_price(
        &instrument.to_core(),
        r0,
        kappa,
        theta,
        sigma,
        discount_rate,
        num_paths,
        seed,
    )
    .map(RangeAccrualResult::from_core)
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_dual_range_accrual_mc_price(
    instrument: &DualRangeAccrual,
    r1_0: f64,
    r2_0: f64,
    kappa1: f64,
    theta1: f64,
    sigma1: f64,
    kappa2: f64,
    theta2: f64,
    sigma2: f64,
    rho: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> PyResult<RangeAccrualResult> {
    dual_range_accrual_mc_price(
        &instrument.to_core(),
        r1_0,
        r2_0,
        kappa1,
        theta1,
        sigma1,
        kappa2,
        theta2,
        sigma2,
        rho,
        discount_rate,
        num_paths,
        seed,
    )
    .map(RangeAccrualResult::from_core)
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_range_accrual_rate_delta(
    instrument: &RangeAccrual,
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> PyResult<f64> {
    range_accrual_rate_delta(
        &instrument.to_core(),
        r0,
        kappa,
        theta,
        sigma,
        discount_rate,
        num_paths,
        seed,
        bump,
    )
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
pub fn py_price_option_to_defer(option: &DeferInvestmentOption) -> PyResult<RealOptionValuation> {
    price_option_to_defer(&option.to_core())
        .map(RealOptionValuation::from_core)
        .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_price_option_to_expand(option: &ExpandOption) -> PyResult<RealOptionValuation> {
    price_option_to_expand(&option.to_core())
        .map(RealOptionValuation::from_core)
        .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_price_option_to_abandon(option: &AbandonmentOption) -> PyResult<RealOptionValuation> {
    price_option_to_abandon(&option.to_core())
        .map(RealOptionValuation::from_core)
        .map_err(pricing_error_to_pyerr)
}

#[pyfunction]
pub fn py_european_abandonment_put(option: &AbandonmentOption) -> PyResult<f64> {
    european_abandonment_put(&option.to_core()).map_err(pricing_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_tarf_mc_price(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
) -> PyResult<TarfPricingResult> {
    tarf_mc_price(
        &tarf.to_core()?,
        spot,
        rate,
        dividend_yield,
        vol,
        num_paths,
        seed,
    )
    .map(TarfPricingResult::from_core)
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_tarf_delta(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> PyResult<f64> {
    tarf_delta(
        &tarf.to_core()?,
        spot,
        rate,
        dividend_yield,
        vol,
        num_paths,
        seed,
        bump,
    )
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_tarf_vega(
    tarf: &Tarf,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> PyResult<f64> {
    tarf_vega(
        &tarf.to_core()?,
        spot,
        rate,
        dividend_yield,
        vol,
        num_paths,
        seed,
        bump,
    )
    .map_err(string_error_to_pyerr)
}

#[pyfunction]
pub fn py_escrowed_dividend_adjusted_spot(
    spot: f64,
    rate: f64,
    expiry: f64,
    dividends: Vec<(f64, f64)>,
) -> f64 {
    escrowed_dividend_adjusted_spot(spot, rate, expiry, &dividends)
}

#[pyfunction]
pub fn py_escrowed_dividend_adjusted_spot_mixed(
    spot: f64,
    rate: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    escrowed_dividend_adjusted_spot_mixed(spot, rate, expiry, &schedule.to_core())
}

#[pyfunction]
pub fn py_forward_price_discrete_div(
    spot: f64,
    rate: f64,
    continuous_dividend_yield: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    forward_price_discrete_div(
        spot,
        rate,
        continuous_dividend_yield,
        expiry,
        &schedule.to_core(),
    )
}

#[pyfunction]
pub fn py_effective_dividend_yield_discrete(
    spot: f64,
    rate: f64,
    continuous_dividend_yield: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    effective_dividend_yield_discrete(
        spot,
        rate,
        continuous_dividend_yield,
        expiry,
        &schedule.to_core(),
    )
}

#[pyfunction]
pub fn py_european_price_discrete_div(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    dividends: Vec<(f64, f64)>,
) -> f64 {
    european_price_discrete_div(spot, strike, rate, vol, expiry, &dividends)
}

#[pyfunction]
pub fn py_european_price_discrete_div_mixed(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    schedule: &DividendSchedule,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    european_price_discrete_div_mixed(
        option_type,
        spot,
        strike,
        rate,
        vol,
        expiry,
        &schedule.to_core(),
    )
}

#[pyfunction]
pub fn py_bootstrap_dividend_curve(
    spot: f64,
    rate: f64,
    parity_quotes: Vec<PutCallParityQuote>,
) -> PyResult<DividendCurveBootstrap> {
    let quotes = parity_quotes
        .iter()
        .copied()
        .map(PutCallParityQuote::to_core)
        .collect::<Vec<_>>();
    bootstrap_dividend_curve(spot, rate, &quotes)
        .map(DividendCurveBootstrap::from_core)
        .map_err(string_error_to_pyerr)
}

#[pyfunction]
pub fn py_strategy_intrinsic_pnl(
    spot_axis: Vec<f64>,
    strikes: Vec<f64>,
    quantities: Vec<f64>,
    is_calls: Vec<u8>,
    total_cost: f64,
) -> PyResult<Vec<f64>> {
    if strikes.len() != quantities.len() || strikes.len() != is_calls.len() {
        return Err(PyValueError::new_err(
            "strikes, quantities, and is_calls must have matching lengths",
        ));
    }
    Ok(strategy_intrinsic_pnl(
        &spot_axis,
        &strikes,
        &quantities,
        &is_calls,
        total_cost,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(py_bs_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_black76_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_bs_greeks, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_barrier_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_barrier_price_closed_form,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_barrier_price_mc, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_american_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_longstaff_schwartz_american_put,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_heston_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_fx_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_digital_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_spread_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_lookback_floating, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_lookback_fixed, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_geometric_asian_fixed_closed_form,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_geometric_asian_discrete_fixed_closed_form,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_arithmetic_asian_price_mc,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_geometric_asian_price_mc, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_autocallable, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_autocallable_sensitivities,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_price_phoenix_autocallable,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_phoenix_autocallable_sensitivities,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_basket_mc, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_price_basket_mc_with_copula,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_price_basket_mc_with_factor_model,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_price_basket_moment_matching,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_price_outperformance_basket_mc,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_quanto_basket_mc, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_stressed_correlation_matrix,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_basket_sensitivities, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_bermudan_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_range_accrual_mc_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_dual_range_accrual_mc_price,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_range_accrual_rate_delta, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_option_to_defer, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_option_to_expand, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_price_option_to_abandon, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_european_abandonment_put, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_tarf_mc_price, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_tarf_delta, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_tarf_vega, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_escrowed_dividend_adjusted_spot,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_escrowed_dividend_adjusted_spot_mixed,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_forward_price_discrete_div,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_effective_dividend_yield_discrete,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_european_price_discrete_div,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_european_price_discrete_div_mixed,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_bootstrap_dividend_curve, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_strategy_intrinsic_pnl, module)?)?;
    module.add_class::<PricingGreeks>()?;
    module.add_class::<AsianStrike>()?;
    module.add_class::<AutocallableSensitivities>()?;
    module.add_class::<BasketType>()?;
    module.add_class::<BasketCopula>()?;
    module.add_class::<BasketMomentMatchingMethod>()?;
    module.add_class::<BasketSensitivities>()?;
    module.add_class::<RangeAccrualResult>()?;
    module.add_class::<TarfType>()?;
    module.add_class::<TarfPricingResult>()?;
    module.add_class::<DeferInvestmentOption>()?;
    module.add_class::<ExpandOption>()?;
    module.add_class::<AbandonmentOption>()?;
    module.add_class::<RealOptionDecision>()?;
    module.add_class::<DecisionTreeNode>()?;
    module.add_class::<RealOptionValuation>()?;
    Ok(())
}
