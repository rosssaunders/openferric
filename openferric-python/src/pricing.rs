use openferric_core::core::{
    Greeks as CoreGreeks, PricingEngine, PricingError, PricingResult as CorePricingResult,
};
use openferric_core::engines::analytic::{
    DigitalAnalyticEngine, ExoticAnalyticEngine, GarmanKohlhagenEngine, kirk_spread_price,
    margrabe_exchange_price,
};
use openferric_core::greeks::black_scholes_merton_greeks;
use openferric_core::instruments::{
    AbandonmentOption as CoreAbandonmentOption, AssetOrNothingOption,
    Autocallable as CoreAutocallable, BasketOption as CoreBasketOption,
    BasketType as CoreBasketType, CashOrNothingOption,
    DeferInvestmentOption as CoreDeferInvestmentOption, DiscreteCashFlow as CoreDiscreteCashFlow,
    DualRangeAccrual as CoreDualRangeAccrual, ExoticOption, ExpandOption as CoreExpandOption,
    FxOption, LookbackFixedOption, LookbackFloatingOption,
    OutperformanceBasketOption as CoreOutperformanceBasketOption,
    PhoenixAutocallable as CorePhoenixAutocallable, QuantoBasketOption as CoreQuantoBasketOption,
    RangeAccrual as CoreRangeAccrual, RealOptionBinomialSpec as CoreRealOptionBinomialSpec,
    SpreadOption, Tarf as CoreTarf, TarfType as CoreTarfType,
};
use openferric_core::market::{
    BootstrappedDividendPoint as CoreBootstrappedDividendPoint,
    DividendCurveBootstrap as CoreDividendCurveBootstrap, DividendEvent as CoreDividendEvent,
    DividendKind as CoreDividendKind, DividendSchedule as CoreDividendSchedule,
    PutCallParityQuote as CorePutCallParityQuote,
};
use openferric_core::math::{
    CorrelationStressScenario as CoreCorrelationStressScenario,
    FactorCorrelationModel as CoreFactorCorrelationModel,
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

use crate::fft::heston_fft_prices_cached;
use crate::helpers::{
    DigitalKind, SpreadMethod, build_market, catch_unwind_py, intrinsic_from_option_type,
    option_price_from_call, parse_barrier_direction, parse_barrier_style, parse_digital_kind,
    parse_option_type, parse_spread_method,
};

fn pricing_error_to_pyerr(err: PricingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn string_error_to_pyerr(err: String) -> PyErr {
    PyValueError::new_err(err)
}

fn diagnostics_to_vec(result: &CorePricingResult) -> Vec<(String, f64)> {
    result
        .diagnostics
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect()
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
pub struct PricingResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub stderr: Option<f64>,
    #[pyo3(get)]
    pub diagnostics: Vec<(String, f64)>,
    greeks: Option<PricingGreeks>,
}

impl PricingResult {
    fn from_core(result: CorePricingResult) -> Self {
        Self {
            price: result.price,
            stderr: result.stderr,
            diagnostics: diagnostics_to_vec(&result),
            greeks: result.greeks.map(PricingGreeks::from_core),
        }
    }
}

#[pymethods]
impl PricingResult {
    #[new]
    fn new(price: f64, stderr: Option<f64>, greeks: Option<PricingGreeks>) -> Self {
        Self {
            price,
            stderr,
            diagnostics: Vec::new(),
            greeks,
        }
    }

    fn greeks(&self) -> Option<PricingGreeks> {
        self.greeks
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("price", self.price)?;
        dict.set_item("stderr", self.stderr)?;
        dict.set_item("diagnostics", self.diagnostics.clone())?;
        dict.set_item("greeks", self.greeks)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "PricingResult(price={}, stderr={:?}, diagnostics={})",
            self.price,
            self.stderr,
            self.diagnostics.len()
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
pub struct Autocallable {
    #[pyo3(get, set)]
    pub underlyings: Vec<usize>,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub autocall_dates: Vec<f64>,
    #[pyo3(get, set)]
    pub autocall_barrier: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub ki_barrier: f64,
    #[pyo3(get, set)]
    pub ki_strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
}

impl Autocallable {
    fn to_core(&self) -> CoreAutocallable {
        CoreAutocallable {
            underlyings: self.underlyings.clone(),
            notional: self.notional,
            autocall_dates: self.autocall_dates.clone(),
            autocall_barrier: self.autocall_barrier,
            coupon_rate: self.coupon_rate,
            ki_barrier: self.ki_barrier,
            ki_strike: self.ki_strike,
            maturity: self.maturity,
        }
    }
}

#[pymethods]
impl Autocallable {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        underlyings: Vec<usize>,
        notional: f64,
        autocall_dates: Vec<f64>,
        autocall_barrier: f64,
        coupon_rate: f64,
        ki_barrier: f64,
        ki_strike: f64,
        maturity: f64,
    ) -> Self {
        Self {
            underlyings,
            notional,
            autocall_dates,
            autocall_barrier,
            coupon_rate,
            ki_barrier,
            ki_strike,
            maturity,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "Autocallable(underlyings={:?}, notional={}, maturity={})",
            self.underlyings, self.notional, self.maturity
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PhoenixAutocallable {
    #[pyo3(get, set)]
    pub underlyings: Vec<usize>,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub autocall_dates: Vec<f64>,
    #[pyo3(get, set)]
    pub autocall_barrier: f64,
    #[pyo3(get, set)]
    pub coupon_barrier: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub memory: bool,
    #[pyo3(get, set)]
    pub ki_barrier: f64,
    #[pyo3(get, set)]
    pub ki_strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
}

impl PhoenixAutocallable {
    fn to_core(&self) -> CorePhoenixAutocallable {
        CorePhoenixAutocallable {
            underlyings: self.underlyings.clone(),
            notional: self.notional,
            autocall_dates: self.autocall_dates.clone(),
            autocall_barrier: self.autocall_barrier,
            coupon_barrier: self.coupon_barrier,
            coupon_rate: self.coupon_rate,
            memory: self.memory,
            ki_barrier: self.ki_barrier,
            ki_strike: self.ki_strike,
            maturity: self.maturity,
        }
    }
}

#[pymethods]
impl PhoenixAutocallable {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        underlyings: Vec<usize>,
        notional: f64,
        autocall_dates: Vec<f64>,
        autocall_barrier: f64,
        coupon_barrier: f64,
        coupon_rate: f64,
        memory: bool,
        ki_barrier: f64,
        ki_strike: f64,
        maturity: f64,
    ) -> Self {
        Self {
            underlyings,
            notional,
            autocall_dates,
            autocall_barrier,
            coupon_barrier,
            coupon_rate,
            memory,
            ki_barrier,
            ki_strike,
            maturity,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "PhoenixAutocallable(underlyings={:?}, notional={}, maturity={}, memory={})",
            self.underlyings, self.notional, self.maturity, self.memory
        )
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
pub struct BasketOption {
    #[pyo3(get, set)]
    pub weights: Vec<f64>,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub is_call: bool,
    basket_type: BasketType,
}

impl BasketOption {
    fn to_core(&self) -> PyResult<CoreBasketOption> {
        Ok(CoreBasketOption {
            weights: self.weights.clone(),
            strike: self.strike,
            maturity: self.maturity,
            is_call: self.is_call,
            basket_type: self.basket_type.to_core()?,
        })
    }
}

#[pymethods]
impl BasketOption {
    #[new]
    fn new(
        weights: Vec<f64>,
        strike: f64,
        maturity: f64,
        is_call: bool,
        basket_type: &BasketType,
    ) -> Self {
        Self {
            weights,
            strike,
            maturity,
            is_call,
            basket_type: basket_type.clone(),
        }
    }

    fn basket_type(&self) -> BasketType {
        self.basket_type.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "BasketOption(strike={}, maturity={}, is_call={}, basket_type={:?})",
            self.strike, self.maturity, self.is_call, self.basket_type.kind
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct OutperformanceBasketOption {
    #[pyo3(get, set)]
    pub leader_index: usize,
    #[pyo3(get, set)]
    pub lagger_weights: Vec<f64>,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub option_type: String,
}

impl OutperformanceBasketOption {
    fn to_core(&self) -> PyResult<CoreOutperformanceBasketOption> {
        let option_type = parse_option_type(&self.option_type).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported option type '{}'", self.option_type))
        })?;
        Ok(CoreOutperformanceBasketOption {
            leader_index: self.leader_index,
            lagger_weights: self.lagger_weights.clone(),
            strike: self.strike,
            maturity: self.maturity,
            option_type,
        })
    }
}

#[pymethods]
impl OutperformanceBasketOption {
    #[new]
    fn new(
        leader_index: usize,
        lagger_weights: Vec<f64>,
        strike: f64,
        maturity: f64,
        option_type: String,
    ) -> Self {
        Self {
            leader_index,
            lagger_weights,
            strike,
            maturity,
            option_type,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "OutperformanceBasketOption(leader_index={}, strike={}, maturity={}, option_type={:?})",
            self.leader_index, self.strike, self.maturity, self.option_type
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct QuantoBasketOption {
    basket: BasketOption,
    #[pyo3(get, set)]
    pub fx_rate: f64,
    #[pyo3(get, set)]
    pub fx_vol: f64,
    #[pyo3(get, set)]
    pub asset_fx_corr: Vec<f64>,
    #[pyo3(get, set)]
    pub domestic_rate: f64,
    #[pyo3(get, set)]
    pub foreign_rate: f64,
}

impl QuantoBasketOption {
    fn to_core(&self) -> PyResult<CoreQuantoBasketOption> {
        Ok(CoreQuantoBasketOption {
            basket: self.basket.to_core()?,
            fx_rate: self.fx_rate,
            fx_vol: self.fx_vol,
            asset_fx_corr: self.asset_fx_corr.clone(),
            domestic_rate: self.domestic_rate,
            foreign_rate: self.foreign_rate,
        })
    }
}

#[pymethods]
impl QuantoBasketOption {
    #[new]
    fn new(
        basket: &BasketOption,
        fx_rate: f64,
        fx_vol: f64,
        asset_fx_corr: Vec<f64>,
        domestic_rate: f64,
        foreign_rate: f64,
    ) -> Self {
        Self {
            basket: basket.clone(),
            fx_rate,
            fx_vol,
            asset_fx_corr,
            domestic_rate,
            foreign_rate,
        }
    }

    fn basket(&self) -> BasketOption {
        self.basket.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantoBasketOption(fx_rate={}, fx_vol={}, domestic_rate={}, foreign_rate={})",
            self.fx_rate, self.fx_vol, self.domestic_rate, self.foreign_rate
        )
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
pub struct FactorCorrelationModel {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub one_factor_loadings: Option<Vec<f64>>,
    #[pyo3(get)]
    pub multi_factor_loadings: Option<Vec<Vec<f64>>>,
}

impl FactorCorrelationModel {
    fn to_core(&self) -> PyResult<CoreFactorCorrelationModel> {
        match self.kind.to_ascii_lowercase().as_str() {
            "one_factor" | "onefactor" => Ok(CoreFactorCorrelationModel::OneFactor {
                loadings: self.one_factor_loadings.clone().ok_or_else(|| {
                    PyValueError::new_err("one_factor model requires one_factor_loadings")
                })?,
            }),
            "multi_factor" | "multifactor" => Ok(CoreFactorCorrelationModel::MultiFactor {
                loadings: self.multi_factor_loadings.clone().ok_or_else(|| {
                    PyValueError::new_err("multi_factor model requires multi_factor_loadings")
                })?,
            }),
            _ => Err(PyValueError::new_err(format!(
                "unsupported FactorCorrelationModel kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl FactorCorrelationModel {
    #[staticmethod]
    fn one_factor(loadings: Vec<f64>) -> Self {
        Self {
            kind: "one_factor".to_string(),
            one_factor_loadings: Some(loadings),
            multi_factor_loadings: None,
        }
    }

    #[staticmethod]
    fn multi_factor(loadings: Vec<Vec<f64>>) -> Self {
        Self {
            kind: "multi_factor".to_string(),
            one_factor_loadings: None,
            multi_factor_loadings: Some(loadings),
        }
    }

    fn correlation_matrix(&self) -> PyResult<Vec<Vec<f64>>> {
        self.to_core()?
            .correlation_matrix()
            .map_err(string_error_to_pyerr)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(string_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("FactorCorrelationModel(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CorrelationStressScenario {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub factor: Option<f64>,
    #[pyo3(get)]
    pub shift: Option<f64>,
    #[pyo3(get)]
    pub floor: Option<f64>,
    #[pyo3(get)]
    pub cap: Option<f64>,
    #[pyo3(get)]
    pub i: Option<usize>,
    #[pyo3(get)]
    pub j: Option<usize>,
    #[pyo3(get)]
    pub value: Option<f64>,
}

impl CorrelationStressScenario {
    fn to_core(&self) -> PyResult<CoreCorrelationStressScenario> {
        match self.kind.to_ascii_lowercase().as_str() {
            "scale_off_diagonal" | "scaleoffdiagonal" => {
                Ok(CoreCorrelationStressScenario::ScaleOffDiagonal {
                    factor: self.factor.ok_or_else(|| {
                        PyValueError::new_err("scale_off_diagonal scenario requires factor")
                    })?,
                })
            }
            "additive_shift" | "additiveshift" => {
                Ok(CoreCorrelationStressScenario::AdditiveShift {
                    shift: self.shift.ok_or_else(|| {
                        PyValueError::new_err("additive_shift scenario requires shift")
                    })?,
                })
            }
            "floor_off_diagonal" | "flooroffdiagonal" => {
                Ok(CoreCorrelationStressScenario::FloorOffDiagonal {
                    floor: self.floor.ok_or_else(|| {
                        PyValueError::new_err("floor_off_diagonal scenario requires floor")
                    })?,
                })
            }
            "cap_off_diagonal" | "capoffdiagonal" => {
                Ok(CoreCorrelationStressScenario::CapOffDiagonal {
                    cap: self.cap.ok_or_else(|| {
                        PyValueError::new_err("cap_off_diagonal scenario requires cap")
                    })?,
                })
            }
            "override_pair" | "overridepair" => Ok(CoreCorrelationStressScenario::OverridePair {
                i: self
                    .i
                    .ok_or_else(|| PyValueError::new_err("override_pair scenario requires i"))?,
                j: self
                    .j
                    .ok_or_else(|| PyValueError::new_err("override_pair scenario requires j"))?,
                value: self.value.ok_or_else(|| {
                    PyValueError::new_err("override_pair scenario requires value")
                })?,
            }),
            _ => Err(PyValueError::new_err(format!(
                "unsupported CorrelationStressScenario kind '{}'",
                self.kind
            ))),
        }
    }
}

#[pymethods]
impl CorrelationStressScenario {
    #[staticmethod]
    fn scale_off_diagonal(factor: f64) -> Self {
        Self {
            kind: "scale_off_diagonal".to_string(),
            factor: Some(factor),
            shift: None,
            floor: None,
            cap: None,
            i: None,
            j: None,
            value: None,
        }
    }

    #[staticmethod]
    fn additive_shift(shift: f64) -> Self {
        Self {
            kind: "additive_shift".to_string(),
            factor: None,
            shift: Some(shift),
            floor: None,
            cap: None,
            i: None,
            j: None,
            value: None,
        }
    }

    #[staticmethod]
    fn floor_off_diagonal(floor: f64) -> Self {
        Self {
            kind: "floor_off_diagonal".to_string(),
            factor: None,
            shift: None,
            floor: Some(floor),
            cap: None,
            i: None,
            j: None,
            value: None,
        }
    }

    #[staticmethod]
    fn cap_off_diagonal(cap: f64) -> Self {
        Self {
            kind: "cap_off_diagonal".to_string(),
            factor: None,
            shift: None,
            floor: None,
            cap: Some(cap),
            i: None,
            j: None,
            value: None,
        }
    }

    #[staticmethod]
    fn override_pair(i: usize, j: usize, value: f64) -> Self {
        Self {
            kind: "override_pair".to_string(),
            factor: None,
            shift: None,
            floor: None,
            cap: None,
            i: Some(i),
            j: Some(j),
            value: Some(value),
        }
    }

    fn __repr__(&self) -> String {
        format!("CorrelationStressScenario(kind={:?})", self.kind)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RangeAccrual {
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub lower_bound: f64,
    #[pyo3(get, set)]
    pub upper_bound: f64,
    #[pyo3(get, set)]
    pub fixing_times: Vec<f64>,
    #[pyo3(get, set)]
    pub payment_time: f64,
}

impl RangeAccrual {
    fn to_core(&self) -> CoreRangeAccrual {
        CoreRangeAccrual {
            notional: self.notional,
            coupon_rate: self.coupon_rate,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            fixing_times: self.fixing_times.clone(),
            payment_time: self.payment_time,
        }
    }
}

#[pymethods]
impl RangeAccrual {
    #[new]
    fn new(
        notional: f64,
        coupon_rate: f64,
        lower_bound: f64,
        upper_bound: f64,
        fixing_times: Vec<f64>,
        payment_time: f64,
    ) -> Self {
        Self {
            notional,
            coupon_rate,
            lower_bound,
            upper_bound,
            fixing_times,
            payment_time,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeAccrual(notional={}, coupon_rate={}, payment_time={})",
            self.notional, self.coupon_rate, self.payment_time
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DualRangeAccrual {
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub lower_bound: f64,
    #[pyo3(get, set)]
    pub upper_bound: f64,
    #[pyo3(get, set)]
    pub fixing_times: Vec<f64>,
    #[pyo3(get, set)]
    pub payment_time: f64,
}

impl DualRangeAccrual {
    fn to_core(&self) -> CoreDualRangeAccrual {
        CoreDualRangeAccrual {
            notional: self.notional,
            coupon_rate: self.coupon_rate,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            fixing_times: self.fixing_times.clone(),
            payment_time: self.payment_time,
        }
    }
}

#[pymethods]
impl DualRangeAccrual {
    #[new]
    fn new(
        notional: f64,
        coupon_rate: f64,
        lower_bound: f64,
        upper_bound: f64,
        fixing_times: Vec<f64>,
        payment_time: f64,
    ) -> Self {
        Self {
            notional,
            coupon_rate,
            lower_bound,
            upper_bound,
            fixing_times,
            payment_time,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "DualRangeAccrual(notional={}, coupon_rate={}, payment_time={})",
            self.notional, self.coupon_rate, self.payment_time
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
pub struct Tarf {
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub notional_per_fixing: f64,
    #[pyo3(get, set)]
    pub ko_barrier: f64,
    #[pyo3(get, set)]
    pub target_profit: f64,
    #[pyo3(get, set)]
    pub downside_leverage: f64,
    #[pyo3(get, set)]
    pub fixing_times: Vec<f64>,
    tarf_type: TarfType,
}

impl Tarf {
    fn to_core(&self) -> PyResult<CoreTarf> {
        Ok(CoreTarf {
            strike: self.strike,
            notional_per_fixing: self.notional_per_fixing,
            ko_barrier: self.ko_barrier,
            target_profit: self.target_profit,
            downside_leverage: self.downside_leverage,
            fixing_times: self.fixing_times.clone(),
            tarf_type: self.tarf_type.to_core()?,
        })
    }
}

#[pymethods]
impl Tarf {
    #[new]
    fn new(
        strike: f64,
        notional_per_fixing: f64,
        ko_barrier: f64,
        target_profit: f64,
        downside_leverage: f64,
        fixing_times: Vec<f64>,
        tarf_type: &TarfType,
    ) -> Self {
        Self {
            strike,
            notional_per_fixing,
            ko_barrier,
            target_profit,
            downside_leverage,
            fixing_times,
            tarf_type: tarf_type.clone(),
        }
    }

    #[staticmethod]
    fn standard(
        strike: f64,
        notional_per_fixing: f64,
        ko_barrier: f64,
        target_profit: f64,
        downside_leverage: f64,
        fixing_times: Vec<f64>,
    ) -> Self {
        Self {
            strike,
            notional_per_fixing,
            ko_barrier,
            target_profit,
            downside_leverage,
            fixing_times,
            tarf_type: TarfType::standard(),
        }
    }

    fn tarf_type(&self) -> TarfType {
        self.tarf_type.clone()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(string_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tarf(strike={}, target_profit={}, ko_barrier={}, tarf_type={:?})",
            self.strike, self.target_profit, self.ko_barrier, self.tarf_type.kind
        )
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
#[derive(Clone, Copy)]
pub struct DiscreteCashFlow {
    #[pyo3(get, set)]
    pub time: f64,
    #[pyo3(get, set)]
    pub amount: f64,
}

impl DiscreteCashFlow {
    fn to_core(self) -> CoreDiscreteCashFlow {
        CoreDiscreteCashFlow {
            time: self.time,
            amount: self.amount,
        }
    }
}

#[pymethods]
impl DiscreteCashFlow {
    #[new]
    fn new(time: f64, amount: f64) -> Self {
        Self { time, amount }
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscreteCashFlow(time={}, amount={})",
            self.time, self.amount
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct RealOptionBinomialSpec {
    #[pyo3(get, set)]
    pub project_value: f64,
    #[pyo3(get, set)]
    pub volatility: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub steps: usize,
    #[pyo3(get, set)]
    pub cash_flows: Vec<DiscreteCashFlow>,
}

impl RealOptionBinomialSpec {
    fn to_core(&self) -> CoreRealOptionBinomialSpec {
        CoreRealOptionBinomialSpec {
            project_value: self.project_value,
            volatility: self.volatility,
            risk_free_rate: self.risk_free_rate,
            maturity: self.maturity,
            steps: self.steps,
            cash_flows: self
                .cash_flows
                .iter()
                .copied()
                .map(DiscreteCashFlow::to_core)
                .collect(),
        }
    }
}

#[pymethods]
impl RealOptionBinomialSpec {
    #[new]
    fn new(
        project_value: f64,
        volatility: f64,
        risk_free_rate: f64,
        maturity: f64,
        steps: usize,
        cash_flows: Vec<DiscreteCashFlow>,
    ) -> Self {
        Self {
            project_value,
            volatility,
            risk_free_rate,
            maturity,
            steps,
            cash_flows,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(pricing_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "RealOptionBinomialSpec(project_value={}, volatility={}, maturity={}, steps={})",
            self.project_value, self.volatility, self.maturity, self.steps
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

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DividendKind {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub value: f64,
}

impl DividendKind {
    fn to_core(&self) -> PyResult<CoreDividendKind> {
        match self.kind.to_ascii_lowercase().as_str() {
            "cash" => Ok(CoreDividendKind::Cash(self.value)),
            "proportional" => Ok(CoreDividendKind::Proportional(self.value)),
            _ => Err(PyValueError::new_err(format!(
                "unsupported DividendKind kind '{}'",
                self.kind
            ))),
        }
    }

    fn from_core(value: CoreDividendKind) -> Self {
        match value {
            CoreDividendKind::Cash(amount) => Self {
                kind: "cash".to_string(),
                value: amount,
            },
            CoreDividendKind::Proportional(ratio) => Self {
                kind: "proportional".to_string(),
                value: ratio,
            },
        }
    }
}

#[pymethods]
impl DividendKind {
    #[staticmethod]
    fn cash(amount: f64) -> Self {
        Self {
            kind: "cash".to_string(),
            value: amount,
        }
    }

    #[staticmethod]
    fn proportional(ratio: f64) -> Self {
        Self {
            kind: "proportional".to_string(),
            value: ratio,
        }
    }

    fn __repr__(&self) -> String {
        format!("DividendKind(kind={:?}, value={})", self.kind, self.value)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DividendEvent {
    #[pyo3(get, set)]
    pub time: f64,
    kind: DividendKind,
}

impl DividendEvent {
    fn to_core(&self) -> PyResult<CoreDividendEvent> {
        Ok(CoreDividendEvent {
            time: self.time,
            kind: self.kind.to_core()?,
        })
    }

    fn from_core(value: CoreDividendEvent) -> Self {
        Self {
            time: value.time,
            kind: DividendKind::from_core(value.kind),
        }
    }
}

#[pymethods]
impl DividendEvent {
    #[new]
    fn new(time: f64, kind: &DividendKind) -> Self {
        Self {
            time,
            kind: kind.clone(),
        }
    }

    fn kind(&self) -> DividendKind {
        self.kind.clone()
    }

    #[staticmethod]
    fn cash(time: f64, amount: f64) -> Self {
        Self {
            time,
            kind: DividendKind::cash(amount),
        }
    }

    #[staticmethod]
    fn proportional(time: f64, ratio: f64) -> Self {
        Self {
            time,
            kind: DividendKind::proportional(ratio),
        }
    }

    fn apply_jump(&self, pre_div_spot: f64) -> PyResult<f64> {
        Ok(self.to_core()?.apply_jump(pre_div_spot))
    }

    fn __repr__(&self) -> String {
        format!(
            "DividendEvent(time={}, kind={:?})",
            self.time, self.kind.kind
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DividendSchedule {
    inner: CoreDividendSchedule,
}

impl DividendSchedule {
    fn to_core(&self) -> CoreDividendSchedule {
        self.inner.clone()
    }
}

#[pymethods]
impl DividendSchedule {
    #[new]
    fn new(events: Vec<DividendEvent>) -> PyResult<Self> {
        let events = events
            .iter()
            .map(DividendEvent::to_core)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: CoreDividendSchedule::new(events).map_err(string_error_to_pyerr)?,
        })
    }

    #[staticmethod]
    fn empty() -> Self {
        Self {
            inner: CoreDividendSchedule::empty(),
        }
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
        self.inner.validate().map_err(string_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("DividendSchedule(events={})", self.inner.events().len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
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
    fn to_core(self) -> CorePutCallParityQuote {
        CorePutCallParityQuote {
            maturity: self.maturity,
            strike: self.strike,
            call_price: self.call_price,
            put_price: self.put_price,
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
        self.to_core().validate().map_err(string_error_to_pyerr)
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
#[derive(Clone, Copy)]
pub struct BootstrappedDividendPoint {
    #[pyo3(get)]
    pub maturity: f64,
    #[pyo3(get)]
    pub forward: f64,
    #[pyo3(get)]
    pub prepaid_forward: f64,
    #[pyo3(get)]
    pub implied_dividend_yield: f64,
    #[pyo3(get)]
    pub cumulative_pv_dividends: f64,
}

impl BootstrappedDividendPoint {
    fn from_core(value: CoreBootstrappedDividendPoint) -> Self {
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
    fn __repr__(&self) -> String {
        format!(
            "BootstrappedDividendPoint(maturity={}, implied_dividend_yield={})",
            self.maturity, self.implied_dividend_yield
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DividendCurveBootstrap {
    #[pyo3(get)]
    pub spot: f64,
    #[pyo3(get)]
    pub rate: f64,
    points: Vec<BootstrappedDividendPoint>,
    inner: CoreDividendCurveBootstrap,
}

impl DividendCurveBootstrap {
    fn from_core(value: CoreDividendCurveBootstrap) -> Self {
        let points = value
            .points
            .iter()
            .copied()
            .map(BootstrappedDividendPoint::from_core)
            .collect();
        Self {
            spot: value.spot,
            rate: value.rate,
            points,
            inner: value,
        }
    }
}

#[pymethods]
impl DividendCurveBootstrap {
    fn points(&self) -> Vec<BootstrappedDividendPoint> {
        self.points.clone()
    }

    fn prepaid_forward_spot(&self, maturity: f64) -> f64 {
        self.inner.prepaid_forward_spot(maturity)
    }

    fn forward_price(&self, maturity: f64) -> f64 {
        self.inner.forward(maturity)
    }

    fn implied_dividend_yield(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 || self.spot <= 0.0 {
            return 0.0;
        }
        let prepaid = self.inner.prepaid_forward_spot(maturity);
        -((prepaid / self.spot).max(1.0e-16)).ln() / maturity
    }

    fn __repr__(&self) -> String {
        format!(
            "DividendCurveBootstrap(spot={}, rate={}, points={})",
            self.spot,
            self.rate,
            self.points.len()
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
        .collect::<PyResult<Vec<_>>>()?;
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
