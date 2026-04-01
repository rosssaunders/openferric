use chrono::{DateTime, Utc};
use openferric_core::core::{
    AsianSpec as CoreAsianSpec, Averaging as CoreAveraging, BarrierDirection as CoreBarrierDirection,
    BarrierSpec as CoreBarrierSpec, BarrierStyle as CoreBarrierStyle,
    ExerciseStyle as CoreExerciseStyle, OptionType as CoreOptionType,
    StrikeType as CoreStrikeType,
};
use openferric_core::instruments::{
    AbandonmentOption as CoreAbandonmentOption, AssetOrNothingOption as CoreAssetOrNothingOption,
    AsianOption as CoreAsianOption, Autocallable as CoreAutocallable,
    BarrierOption as CoreBarrierOption, BarrierOptionBuilder as CoreBarrierOptionBuilder,
    BasketOption as CoreBasketOption, BasketType as CoreBasketType,
    BermudanOption as CoreBermudanOption, BestOfTwoCallOption as CoreBestOfTwoCallOption,
    CallableRangeAccrualNote as CoreCallableRangeAccrualNote,
    CallableRateNote as CoreCallableRateNote,
    CashOrNothingOption as CoreCashOrNothingOption,
    CatastropheBond as CoreCatastropheBond, ChooserOption as CoreChooserOption,
    CliquetOption as CoreCliquetOption, CmsLinkedNote as CoreCmsLinkedNote,
    CommodityForward as CoreCommodityForward, CommodityFutures as CoreCommodityFutures,
    CommodityOption as CoreCommodityOption,
    CommoditySpreadOption as CoreCommoditySpreadOption,
    CompoundOption as CoreCompoundOption, ConstantCpr as CoreConstantCpr,
    ConvertibleBond as CoreConvertibleBond, CouponPeriod as CoreCouponPeriod,
    CouponScheduleBuilder as CoreCouponScheduleBuilder, CouponType as CoreCouponType,
    DegreeDayType as CoreDegreeDayType, DeferInvestmentOption as CoreDeferInvestmentOption,
    DiscreteCashFlow as CoreDiscreteCashFlow,
    DoubleBarrierOption as CoreDoubleBarrierOption,
    DoubleBarrierType as CoreDoubleBarrierType,
    DualRangeAccrual as CoreDualRangeAccrual, EmployeeStockOption as CoreEmployeeStockOption,
    ExerciseSchedule as CoreExerciseSchedule, ExoticOption as CoreExoticOption,
    ExpandOption as CoreExpandOption, ForwardStartOption as CoreForwardStartOption,
    FuturesOption as CoreFuturesOption, FxOption as CoreFxOption, GapOption as CoreGapOption,
    InverseFloaterNote as CoreInverseFloaterNote,
    LookbackFixedOption as CoreLookbackFixedOption,
    LookbackFloatingOption as CoreLookbackFloatingOption, MbsCashflow as CoreMbsCashflow,
    MbsPassThrough as CoreMbsPassThrough,
    OutperformanceBasketOption as CoreOutperformanceBasketOption,
    PhoenixAutocallable as CorePhoenixAutocallable, PowerOption as CorePowerOption,
    PrepaymentModel as CorePrepaymentModel, PsaModel as CorePsaModel,
    QuantoBasketOption as CoreQuantoBasketOption, QuantoOption as CoreQuantoOption,
    RangeAccrual as CoreRangeAccrual, RealOptionBinomialSpec as CoreRealOptionBinomialSpec,
    RealOptionInstrument as CoreRealOptionInstrument, SnowballNote as CoreSnowballNote,
    SpreadOption as CoreSpreadOption, StructuredCoupon as CoreStructuredCoupon,
    SwingOption as CoreSwingOption, Tarf as CoreTarf, TarfType as CoreTarfType,
    TargetRedemptionNote as CoreTargetRedemptionNote,
    Trade as CoreTrade, TradeInstrument as CoreTradeInstrument,
    TradeMetadata as CoreTradeMetadata, VarianceOptionQuote as CoreVarianceOptionQuote,
    VarianceSwap as CoreVarianceSwap, VanillaOption as CoreVanillaOption,
    VolatilitySwap as CoreVolatilitySwap, WeatherOption as CoreWeatherOption,
    WeatherSwap as CoreWeatherSwap, WorstOfTwoCallOption as CoreWorstOfTwoCallOption,
    TwoAssetCorrelationOption as CoreTwoAssetCorrelationOption, Portfolio as CorePortfolio,
};
use openferric_core::models::{
    HullWhite as CoreHullWhite, TwoFactorCommodityProcess as CoreTwoFactorCommodityProcess,
    TwoFactorSpreadModel as CoreTwoFactorSpreadModel,
};
use openferric_core::rates::{Frequency as CoreFrequency, YieldCurve as CoreYieldCurve};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::funding::FundingRateSwap;
use crate::helpers::{format_datetime, parse_datetime};

fn invalid_input(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn map_err_string(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_option_type_str(value: &str) -> PyResult<CoreOptionType> {
    match value.to_ascii_lowercase().as_str() {
        "call" => Ok(CoreOptionType::Call),
        "put" => Ok(CoreOptionType::Put),
        _ => Err(invalid_input("option_type must be 'call' or 'put'")),
    }
}

fn format_option_type(value: CoreOptionType) -> &'static str {
    match value {
        CoreOptionType::Call => "call",
        CoreOptionType::Put => "put",
    }
}

fn parse_averaging(value: &str) -> PyResult<CoreAveraging> {
    match value.to_ascii_lowercase().as_str() {
        "arithmetic" => Ok(CoreAveraging::Arithmetic),
        "geometric" => Ok(CoreAveraging::Geometric),
        _ => Err(invalid_input("averaging must be 'arithmetic' or 'geometric'")),
    }
}

fn format_averaging(value: CoreAveraging) -> &'static str {
    match value {
        CoreAveraging::Arithmetic => "arithmetic",
        CoreAveraging::Geometric => "geometric",
    }
}

fn parse_strike_type(value: &str) -> PyResult<CoreStrikeType> {
    match value.to_ascii_lowercase().as_str() {
        "fixed" => Ok(CoreStrikeType::Fixed),
        "floating" => Ok(CoreStrikeType::Floating),
        _ => Err(invalid_input("strike_type must be 'fixed' or 'floating'")),
    }
}

fn format_strike_type(value: CoreStrikeType) -> &'static str {
    match value {
        CoreStrikeType::Fixed => "fixed",
        CoreStrikeType::Floating => "floating",
    }
}

fn parse_barrier_direction_str(value: &str) -> PyResult<CoreBarrierDirection> {
    match value.to_ascii_lowercase().as_str() {
        "up" => Ok(CoreBarrierDirection::Up),
        "down" => Ok(CoreBarrierDirection::Down),
        _ => Err(invalid_input("barrier direction must be 'up' or 'down'")),
    }
}

fn format_barrier_direction(value: CoreBarrierDirection) -> &'static str {
    match value {
        CoreBarrierDirection::Up => "up",
        CoreBarrierDirection::Down => "down",
    }
}

fn parse_barrier_style_str(value: &str) -> PyResult<CoreBarrierStyle> {
    match value.to_ascii_lowercase().as_str() {
        "in" => Ok(CoreBarrierStyle::In),
        "out" => Ok(CoreBarrierStyle::Out),
        _ => Err(invalid_input("barrier style must be 'in' or 'out'")),
    }
}

fn format_barrier_style(value: CoreBarrierStyle) -> &'static str {
    match value {
        CoreBarrierStyle::In => "in",
        CoreBarrierStyle::Out => "out",
    }
}

fn parse_frequency(value: &str) -> PyResult<CoreFrequency> {
    match value.to_ascii_lowercase().as_str() {
        "annual" => Ok(CoreFrequency::Annual),
        "semiannual" | "semi_annual" | "semi-annual" => Ok(CoreFrequency::SemiAnnual),
        "quarterly" => Ok(CoreFrequency::Quarterly),
        "monthly" => Ok(CoreFrequency::Monthly),
        _ => Err(invalid_input(
            "frequency must be one of: annual, semiannual, quarterly, monthly",
        )),
    }
}

fn format_frequency(value: CoreFrequency) -> &'static str {
    match value {
        CoreFrequency::Annual => "annual",
        CoreFrequency::SemiAnnual => "semiannual",
        CoreFrequency::Quarterly => "quarterly",
        CoreFrequency::Monthly => "monthly",
    }
}

fn parse_basket_type(value: &str) -> PyResult<CoreBasketType> {
    match value.to_ascii_lowercase().as_str() {
        "average" => Ok(CoreBasketType::Average),
        "bestof" | "best_of" | "best-of" => Ok(CoreBasketType::BestOf),
        "worstof" | "worst_of" | "worst-of" => Ok(CoreBasketType::WorstOf),
        _ => Err(invalid_input("basket_type must be 'average', 'best_of', or 'worst_of'")),
    }
}

fn format_basket_type(value: CoreBasketType) -> &'static str {
    match value {
        CoreBasketType::Average => "average",
        CoreBasketType::BestOf => "best_of",
        CoreBasketType::WorstOf => "worst_of",
    }
}

fn parse_double_barrier_type(value: &str) -> PyResult<CoreDoubleBarrierType> {
    match value.to_ascii_lowercase().as_str() {
        "knockout" | "knock_out" | "knock-out" => Ok(CoreDoubleBarrierType::KnockOut),
        "knockin" | "knock_in" | "knock-in" => Ok(CoreDoubleBarrierType::KnockIn),
        _ => Err(invalid_input("double barrier type must be 'knock_out' or 'knock_in'")),
    }
}

fn format_double_barrier_type(value: CoreDoubleBarrierType) -> &'static str {
    match value {
        CoreDoubleBarrierType::KnockOut => "knock_out",
        CoreDoubleBarrierType::KnockIn => "knock_in",
    }
}

fn parse_tarf_type(value: &str) -> PyResult<CoreTarfType> {
    match value.to_ascii_lowercase().as_str() {
        "standard" => Ok(CoreTarfType::Standard),
        "decumulator" => Ok(CoreTarfType::Decumulator),
        _ => Err(invalid_input("tarf_type must be 'standard' or 'decumulator'")),
    }
}

fn format_tarf_type(value: CoreTarfType) -> &'static str {
    match value {
        CoreTarfType::Standard => "standard",
        CoreTarfType::Decumulator => "decumulator",
    }
}

fn parse_degree_day_type(value: &str) -> PyResult<CoreDegreeDayType> {
    match value.to_ascii_lowercase().as_str() {
        "hdd" => Ok(CoreDegreeDayType::HDD),
        "cdd" => Ok(CoreDegreeDayType::CDD),
        _ => Err(invalid_input("degree day type must be 'hdd' or 'cdd'")),
    }
}

fn format_degree_day_type(value: CoreDegreeDayType) -> &'static str {
    match value {
        CoreDegreeDayType::HDD => "hdd",
        CoreDegreeDayType::CDD => "cdd",
    }
}

macro_rules! simple_enum_wrapper {
    ($name:ident, $core:ty, $parse:ident, $format:ident) => {
        #[pyclass(module = "openferric", frozen)]
        #[derive(Clone, Copy)]
        pub struct $name {
            inner: $core,
        }

        impl $name {
            fn to_core(self) -> $core {
                self.inner
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(value: &str) -> PyResult<Self> {
                Ok(Self {
                    inner: $parse(value)?,
                })
            }

            #[getter]
            fn value(&self) -> &'static str {
                $format(self.inner)
            }

            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.value())
            }
        }
    };
}

simple_enum_wrapper!(BasketType, CoreBasketType, parse_basket_type, format_basket_type);
simple_enum_wrapper!(
    DoubleBarrierType,
    CoreDoubleBarrierType,
    parse_double_barrier_type,
    format_double_barrier_type
);
simple_enum_wrapper!(TarfType, CoreTarfType, parse_tarf_type, format_tarf_type);
simple_enum_wrapper!(
    DegreeDayType,
    CoreDegreeDayType,
    parse_degree_day_type,
    format_degree_day_type
);
simple_enum_wrapper!(Frequency, CoreFrequency, parse_frequency, format_frequency);

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ExerciseStyle {
    inner: CoreExerciseStyle,
}

impl ExerciseStyle {
    fn to_core(&self) -> CoreExerciseStyle {
        self.inner.clone()
    }

    fn from_core(inner: CoreExerciseStyle) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl ExerciseStyle {
    #[new]
    #[pyo3(signature = (kind, dates=None))]
    fn new(kind: &str, dates: Option<Vec<f64>>) -> PyResult<Self> {
        let inner = match kind.to_ascii_lowercase().as_str() {
            "european" => CoreExerciseStyle::European,
            "american" => CoreExerciseStyle::American,
            "bermudan" => CoreExerciseStyle::Bermudan {
                dates: dates.unwrap_or_default(),
            },
            _ => {
                return Err(invalid_input(
                    "exercise style must be 'european', 'american', or 'bermudan'",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            CoreExerciseStyle::European => "european",
            CoreExerciseStyle::American => "american",
            CoreExerciseStyle::Bermudan { .. } => "bermudan",
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
            CoreExerciseStyle::European => "ExerciseStyle('european')".to_string(),
            CoreExerciseStyle::American => "ExerciseStyle('american')".to_string(),
            CoreExerciseStyle::Bermudan { dates } => {
                format!("ExerciseStyle('bermudan', dates={dates:?})")
            }
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BarrierSpec {
    #[pyo3(get, set)]
    pub direction: String,
    #[pyo3(get, set)]
    pub style: String,
    #[pyo3(get, set)]
    pub level: f64,
    #[pyo3(get, set)]
    pub rebate: f64,
}

impl BarrierSpec {
    fn to_core(&self) -> PyResult<CoreBarrierSpec> {
        Ok(CoreBarrierSpec {
            direction: parse_barrier_direction_str(&self.direction)?,
            style: parse_barrier_style_str(&self.style)?,
            level: self.level,
            rebate: self.rebate,
        })
    }

    fn from_core(inner: CoreBarrierSpec) -> Self {
        Self {
            direction: format_barrier_direction(inner.direction).to_string(),
            style: format_barrier_style(inner.style).to_string(),
            level: inner.level,
            rebate: inner.rebate,
        }
    }
}

#[pymethods]
impl BarrierSpec {
    #[new]
    fn new(direction: String, style: String, level: f64, rebate: f64) -> PyResult<Self> {
        let out = Self {
            direction,
            style,
            level,
            rebate,
        };
        let _ = out.to_core()?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "BarrierSpec(direction={:?}, style={:?}, level={}, rebate={})",
            self.direction, self.style, self.level, self.rebate
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AsianSpec {
    #[pyo3(get, set)]
    pub averaging: String,
    #[pyo3(get, set)]
    pub strike_type: String,
    #[pyo3(get, set)]
    pub observation_times: Vec<f64>,
}

impl AsianSpec {
    fn to_core(&self) -> PyResult<CoreAsianSpec> {
        Ok(CoreAsianSpec {
            averaging: parse_averaging(&self.averaging)?,
            strike_type: parse_strike_type(&self.strike_type)?,
            observation_times: self.observation_times.clone(),
        })
    }

    fn from_core(inner: CoreAsianSpec) -> Self {
        Self {
            averaging: format_averaging(inner.averaging).to_string(),
            strike_type: format_strike_type(inner.strike_type).to_string(),
            observation_times: inner.observation_times,
        }
    }
}

#[pymethods]
impl AsianSpec {
    #[new]
    fn new(averaging: String, strike_type: String, observation_times: Vec<f64>) -> PyResult<Self> {
        let out = Self {
            averaging,
            strike_type,
            observation_times,
        };
        let _ = out.to_core()?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "AsianSpec(averaging={:?}, strike_type={:?}, observation_times={:?})",
            self.averaging, self.strike_type, self.observation_times
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct YieldCurve {
    inner: CoreYieldCurve,
}

#[pymethods]
impl YieldCurve {
    #[new]
    fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self {
            inner: CoreYieldCurve::new(tenors),
        }
    }

    #[getter]
    fn tenors(&self) -> Vec<(f64, f64)> {
        self.inner.tenors.clone()
    }

    fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }

    fn zero_rate(&self, t: f64) -> f64 {
        self.inner.zero_rate(t)
    }

    fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        self.inner.forward_rate(t1, t2)
    }

    fn __repr__(&self) -> String {
        format!("YieldCurve(tenors={})", self.inner.tenors.len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HullWhite {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub theta: Vec<(f64, f64)>,
}

impl HullWhite {
    fn to_core(&self) -> CoreHullWhite {
        CoreHullWhite {
            a: self.a,
            sigma: self.sigma,
            theta: self.theta.clone(),
        }
    }

    fn from_core(inner: CoreHullWhite) -> Self {
        Self {
            a: inner.a,
            sigma: inner.sigma,
            theta: inner.theta,
        }
    }
}

#[pymethods]
impl HullWhite {
    #[new]
    fn new(a: f64, sigma: f64) -> Self {
        Self {
            a,
            sigma,
            theta: Vec::new(),
        }
    }

    fn calibrate_theta(&mut self, curve: &YieldCurve, times: Vec<f64>) {
        let mut model = self.to_core();
        model.calibrate_theta(&curve.inner, &times);
        self.theta = model.theta;
    }

    fn theta_at(&self, t: f64) -> f64 {
        self.to_core().theta_at(t)
    }

    fn bond_price(&self, t: f64, maturity: f64, short_rate: f64, curve: &YieldCurve) -> f64 {
        self.to_core().bond_price(t, maturity, short_rate, &curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "HullWhite(a={}, sigma={}, theta_points={})",
            self.a,
            self.sigma,
            self.theta.len()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct TwoFactorCommodityProcess {
    #[pyo3(get, set)]
    pub kappa_fast: f64,
    #[pyo3(get, set)]
    pub sigma_fast: f64,
    #[pyo3(get, set)]
    pub sigma_slow: f64,
}

impl TwoFactorCommodityProcess {
    fn to_core(self) -> CoreTwoFactorCommodityProcess {
        CoreTwoFactorCommodityProcess {
            kappa_fast: self.kappa_fast,
            sigma_fast: self.sigma_fast,
            sigma_slow: self.sigma_slow,
        }
    }
}

#[pymethods]
impl TwoFactorCommodityProcess {
    #[new]
    fn new(kappa_fast: f64, sigma_fast: f64, sigma_slow: f64) -> Self {
        Self {
            kappa_fast,
            sigma_fast,
            sigma_slow,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "TwoFactorCommodityProcess(kappa_fast={}, sigma_fast={}, sigma_slow={})",
            self.kappa_fast, self.sigma_fast, self.sigma_slow
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct TwoFactorSpreadModel {
    #[pyo3(get, set)]
    pub leg_1: TwoFactorCommodityProcess,
    #[pyo3(get, set)]
    pub leg_2: TwoFactorCommodityProcess,
    #[pyo3(get, set)]
    pub rho_fast: f64,
    #[pyo3(get, set)]
    pub rho_slow: f64,
}

impl TwoFactorSpreadModel {
    fn to_core(self) -> CoreTwoFactorSpreadModel {
        CoreTwoFactorSpreadModel {
            leg_1: self.leg_1.to_core(),
            leg_2: self.leg_2.to_core(),
            rho_fast: self.rho_fast,
            rho_slow: self.rho_slow,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VanillaOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub exercise: ExerciseStyle,
}

impl VanillaOption {
    fn to_core(&self) -> PyResult<CoreVanillaOption> {
        Ok(CoreVanillaOption {
            option_type: parse_option_type_str(&self.option_type)?,
            strike: self.strike,
            expiry: self.expiry,
            exercise: self.exercise.to_core(),
        })
    }

    fn from_core(inner: CoreVanillaOption) -> Self {
        Self {
            option_type: format_option_type(inner.option_type).to_string(),
            strike: inner.strike,
            expiry: inner.expiry,
            exercise: ExerciseStyle::from_core(inner.exercise),
        }
    }
}

#[pymethods]
impl VanillaOption {
    #[new]
    fn new(option_type: String, strike: f64, expiry: f64, exercise: ExerciseStyle) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            expiry,
            exercise,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn european_call(strike: f64, expiry: f64) -> Self {
        Self::from_core(CoreVanillaOption::european_call(strike, expiry))
    }

    #[staticmethod]
    fn european_put(strike: f64, expiry: f64) -> Self {
        Self::from_core(CoreVanillaOption::european_put(strike, expiry))
    }

    #[staticmethod]
    fn american_call(strike: f64, expiry: f64) -> Self {
        Self::from_core(CoreVanillaOption::american_call(strike, expiry))
    }

    #[staticmethod]
    fn american_put(strike: f64, expiry: f64) -> Self {
        Self::from_core(CoreVanillaOption::american_put(strike, expiry))
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "VanillaOption(option_type={:?}, strike={}, expiry={}, exercise={})",
            self.option_type,
            self.strike,
            self.expiry,
            self.exercise.__repr__()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AsianOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub asian: AsianSpec,
}

impl AsianOption {
    fn to_core(&self) -> PyResult<CoreAsianOption> {
        Ok(CoreAsianOption {
            option_type: parse_option_type_str(&self.option_type)?,
            strike: self.strike,
            expiry: self.expiry,
            asian: self.asian.to_core()?,
        })
    }
}

#[pymethods]
impl AsianOption {
    #[new]
    fn new(option_type: String, strike: f64, expiry: f64, asian: AsianSpec) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            expiry,
            asian,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "AsianOption(option_type={:?}, strike={}, expiry={}, asian={})",
            self.option_type,
            self.strike,
            self.expiry,
            self.asian.__repr__()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BarrierOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub barrier: BarrierSpec,
}

impl BarrierOption {
    fn to_core(&self) -> PyResult<CoreBarrierOption> {
        Ok(CoreBarrierOption {
            option_type: parse_option_type_str(&self.option_type)?,
            strike: self.strike,
            expiry: self.expiry,
            barrier: self.barrier.to_core()?,
        })
    }

    fn from_core(inner: CoreBarrierOption) -> Self {
        Self {
            option_type: format_option_type(inner.option_type).to_string(),
            strike: inner.strike,
            expiry: inner.expiry,
            barrier: BarrierSpec::from_core(inner.barrier),
        }
    }
}

#[pymethods]
impl BarrierOption {
    #[new]
    fn new(option_type: String, strike: f64, expiry: f64, barrier: BarrierSpec) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            expiry,
            barrier,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn builder() -> BarrierOptionBuilder {
        BarrierOptionBuilder::new()
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "BarrierOption(option_type={:?}, strike={}, expiry={}, barrier={})",
            self.option_type,
            self.strike,
            self.expiry,
            self.barrier.__repr__()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BarrierOptionBuilder {
    inner: CoreBarrierOptionBuilder,
}

#[pymethods]
impl BarrierOptionBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreBarrierOption::builder(),
        }
    }

    fn call(&self) -> Self {
        Self {
            inner: self.inner.clone().call(),
        }
    }

    fn put(&self) -> Self {
        Self {
            inner: self.inner.clone().put(),
        }
    }

    fn strike(&self, strike: f64) -> Self {
        Self {
            inner: self.inner.clone().strike(strike),
        }
    }

    fn expiry(&self, expiry: f64) -> Self {
        Self {
            inner: self.inner.clone().expiry(expiry),
        }
    }

    fn up_and_out(&self, level: f64) -> Self {
        Self {
            inner: self.inner.clone().up_and_out(level),
        }
    }

    fn up_and_in(&self, level: f64) -> Self {
        Self {
            inner: self.inner.clone().up_and_in(level),
        }
    }

    fn down_and_out(&self, level: f64) -> Self {
        Self {
            inner: self.inner.clone().down_and_out(level),
        }
    }

    fn down_and_in(&self, level: f64) -> Self {
        Self {
            inner: self.inner.clone().down_and_in(level),
        }
    }

    fn rebate(&self, rebate: f64) -> Self {
        Self {
            inner: self.inner.clone().rebate(rebate),
        }
    }

    fn build(&self) -> PyResult<BarrierOption> {
        Ok(BarrierOption::from_core(
            self.inner.clone().build().map_err(map_err_string)?,
        ))
    }

    fn __repr__(&self) -> String {
        "BarrierOptionBuilder()".to_string()
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
    ) -> PyResult<Self> {
        let out = Self {
            underlyings,
            notional,
            autocall_dates,
            autocall_barrier,
            coupon_rate,
            ki_barrier,
            ki_strike,
            maturity,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
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
    ) -> PyResult<Self> {
        let out = Self {
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
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
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
    #[pyo3(get, set)]
    pub basket_type: String,
}

impl BasketOption {
    fn to_core(&self) -> PyResult<CoreBasketOption> {
        Ok(CoreBasketOption {
            weights: self.weights.clone(),
            strike: self.strike,
            maturity: self.maturity,
            is_call: self.is_call,
            basket_type: parse_basket_type(&self.basket_type)?,
        })
    }

    fn from_core(inner: CoreBasketOption) -> Self {
        Self {
            weights: inner.weights,
            strike: inner.strike,
            maturity: inner.maturity,
            is_call: inner.is_call,
            basket_type: format_basket_type(inner.basket_type).to_string(),
        }
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
        basket_type: String,
    ) -> PyResult<Self> {
        let out = Self {
            weights,
            strike,
            maturity,
            is_call,
            basket_type,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
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
        Ok(CoreOutperformanceBasketOption {
            leader_index: self.leader_index,
            lagger_weights: self.lagger_weights.clone(),
            strike: self.strike,
            maturity: self.maturity,
            option_type: parse_option_type_str(&self.option_type)?,
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
    ) -> PyResult<Self> {
        let out = Self {
            leader_index,
            lagger_weights,
            strike,
            maturity,
            option_type,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct QuantoBasketOption {
    #[pyo3(get, set)]
    pub basket: BasketOption,
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

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BermudanOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub exercise_dates: Vec<f64>,
    #[pyo3(get, set)]
    pub strike_schedule: Vec<f64>,
}

impl BermudanOption {
    fn to_core(&self) -> PyResult<CoreBermudanOption> {
        Ok(CoreBermudanOption::new(
            parse_option_type_str(&self.option_type)?,
            self.expiry,
            self.exercise_dates.clone(),
            self.strike_schedule.clone(),
        ))
    }
}

#[pymethods]
impl BermudanOption {
    #[new]
    fn new(
        option_type: String,
        expiry: f64,
        exercise_dates: Vec<f64>,
        strike_schedule: Vec<f64>,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            expiry,
            exercise_dates,
            strike_schedule,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn with_constant_strike(
        option_type: String,
        strike: f64,
        expiry: f64,
        exercise_dates: Vec<f64>,
    ) -> PyResult<Self> {
        let inner = CoreBermudanOption::with_constant_strike(
            parse_option_type_str(&option_type)?,
            strike,
            expiry,
            exercise_dates,
        );
        Ok(Self {
            option_type,
            expiry: inner.expiry,
            exercise_dates: inner.exercise_dates,
            strike_schedule: inner.strike_schedule,
        })
    }

    fn num_exercise_dates(&self) -> usize {
        self.exercise_dates.len()
    }

    fn strike_at_exercise_index(&self, index: usize) -> Option<f64> {
        self.strike_schedule.get(index).copied()
    }

    fn strike_at_time(&self, time: f64) -> PyResult<f64> {
        self.to_core()?.strike_at_time(time).map_err(map_err_string)
    }

    fn effective_schedule(&self) -> PyResult<Vec<(f64, f64)>> {
        self.to_core()?.effective_schedule().map_err(map_err_string)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FuturesOption {
    #[pyo3(get, set)]
    pub forward: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub r: f64,
    #[pyo3(get, set)]
    pub t: f64,
    #[pyo3(get, set)]
    pub option_type: String,
}

impl FuturesOption {
    fn to_core(self) -> PyResult<CoreFuturesOption> {
        Ok(CoreFuturesOption::new(
            self.forward,
            self.strike,
            self.vol,
            self.r,
            self.t,
            parse_option_type_str(&self.option_type)?,
        ))
    }

    fn from_core(inner: CoreFuturesOption) -> Self {
        Self {
            forward: inner.forward,
            strike: inner.strike,
            vol: inner.vol,
            r: inner.r,
            t: inner.t,
            option_type: format_option_type(inner.option_type).to_string(),
        }
    }
}

#[pymethods]
impl FuturesOption {
    #[new]
    fn new(forward: f64, strike: f64, vol: f64, r: f64, t: f64, option_type: String) -> PyResult<Self> {
        let out = Self {
            forward,
            strike,
            vol,
            r,
            t,
            option_type,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn call(forward: f64, strike: f64, vol: f64, r: f64, t: f64) -> Self {
        Self::from_core(CoreFuturesOption::call(forward, strike, vol, r, t))
    }

    #[staticmethod]
    fn put(forward: f64, strike: f64, vol: f64, r: f64, t: f64) -> Self {
        Self::from_core(CoreFuturesOption::put(forward, strike, vol, r, t))
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ForwardStartOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub strike_ratio: f64,
    #[pyo3(get, set)]
    pub rate: f64,
    #[pyo3(get, set)]
    pub dividend_yield: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub t_start: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
}

impl ForwardStartOption {
    fn to_core(self) -> PyResult<CoreForwardStartOption> {
        Ok(CoreForwardStartOption {
            option_type: parse_option_type_str(&self.option_type)?,
            spot: self.spot,
            strike_ratio: self.strike_ratio,
            rate: self.rate,
            dividend_yield: self.dividend_yield,
            vol: self.vol,
            t_start: self.t_start,
            expiry: self.expiry,
        })
    }

    fn from_core(inner: CoreForwardStartOption) -> Self {
        Self {
            option_type: format_option_type(inner.option_type).to_string(),
            spot: inner.spot,
            strike_ratio: inner.strike_ratio,
            rate: inner.rate,
            dividend_yield: inner.dividend_yield,
            vol: inner.vol,
            t_start: inner.t_start,
            expiry: inner.expiry,
        }
    }
}

#[pymethods]
impl ForwardStartOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        spot: f64,
        strike_ratio: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        t_start: f64,
        expiry: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            spot,
            strike_ratio,
            rate,
            dividend_yield,
            vol,
            t_start,
            expiry,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn atm_call(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        t_start: f64,
        expiry: f64,
    ) -> Self {
        Self::from_core(CoreForwardStartOption::atm_call(
            spot,
            rate,
            dividend_yield,
            vol,
            t_start,
            expiry,
        ))
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }

    fn price_rubinstein(&self) -> PyResult<f64> {
        self.clone().to_core()?.price_rubinstein().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CliquetOption {
    inner: CoreCliquetOption,
}

#[pymethods]
impl CliquetOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        spot: f64,
        strike_ratio: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        t_start: f64,
        expiry: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCliquetOption {
                option_type: parse_option_type_str(&option_type)?,
                spot,
                strike_ratio,
                rate,
                dividend_yield,
                vol,
                t_start,
                expiry,
            },
        })
    }

    #[getter]
    fn option_type(&self) -> String {
        format_option_type(self.inner.option_type).to_string()
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot
    }

    #[getter]
    fn strike_ratio(&self) -> f64 {
        self.inner.strike_ratio
    }

    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }

    #[getter]
    fn dividend_yield(&self) -> f64 {
        self.inner.dividend_yield
    }

    #[getter]
    fn vol(&self) -> f64 {
        self.inner.vol
    }

    #[getter]
    fn t_start(&self) -> f64 {
        self.inner.t_start
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    #[staticmethod]
    fn atm_call(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        t_start: f64,
        expiry: f64,
    ) -> Self {
        Self {
            inner: CoreCliquetOption::atm_call(spot, rate, dividend_yield, vol, t_start, expiry),
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(map_err_string)
    }

    fn price_rubinstein(&self) -> PyResult<f64> {
        self.inner.price_rubinstein().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CommodityForward {
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub storage_cost: f64,
    #[pyo3(get, set)]
    pub convenience_yield: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub is_long: bool,
}

impl CommodityForward {
    fn to_core(self) -> CoreCommodityForward {
        CoreCommodityForward {
            spot: self.spot,
            strike: self.strike,
            notional: self.notional,
            risk_free_rate: self.risk_free_rate,
            storage_cost: self.storage_cost,
            convenience_yield: self.convenience_yield,
            maturity: self.maturity,
            is_long: self.is_long,
        }
    }
}

#[pymethods]
impl CommodityForward {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spot: f64,
        strike: f64,
        notional: f64,
        risk_free_rate: f64,
        storage_cost: f64,
        convenience_yield: f64,
        maturity: f64,
        is_long: bool,
    ) -> PyResult<Self> {
        let out = Self {
            spot,
            strike,
            notional,
            risk_free_rate,
            storage_cost,
            convenience_yield,
            maturity,
            is_long,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }

    fn theoretical_forward_price(&self) -> f64 {
        self.clone().to_core().theoretical_forward_price()
    }

    fn present_value(&self) -> PyResult<f64> {
        self.clone().to_core().present_value().map_err(map_err_string)
    }

    fn mark_to_market(&self, market_forward: f64) -> PyResult<f64> {
        self.clone()
            .to_core()
            .mark_to_market(market_forward)
            .map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CommodityFutures {
    #[pyo3(get, set)]
    pub contract_price: f64,
    #[pyo3(get, set)]
    pub contract_size: f64,
    #[pyo3(get, set)]
    pub is_long: bool,
}

impl CommodityFutures {
    fn to_core(self) -> CoreCommodityFutures {
        CoreCommodityFutures {
            contract_price: self.contract_price,
            contract_size: self.contract_size,
            is_long: self.is_long,
        }
    }
}

#[pymethods]
impl CommodityFutures {
    #[new]
    fn new(contract_price: f64, contract_size: f64, is_long: bool) -> PyResult<Self> {
        let out = Self {
            contract_price,
            contract_size,
            is_long,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }

    fn value(&self, mark_price: f64) -> PyResult<f64> {
        self.clone().to_core().value(mark_price).map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CommodityOption {
    #[pyo3(get, set)]
    pub forward: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub option_type: String,
}

impl CommodityOption {
    fn to_core(self) -> PyResult<CoreCommodityOption> {
        Ok(CoreCommodityOption {
            forward: self.forward,
            strike: self.strike,
            vol: self.vol,
            risk_free_rate: self.risk_free_rate,
            maturity: self.maturity,
            notional: self.notional,
            option_type: parse_option_type_str(&self.option_type)?,
        })
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CommoditySpreadOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub forward_1: f64,
    #[pyo3(get, set)]
    pub forward_2: f64,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub quantity_1: f64,
    #[pyo3(get, set)]
    pub quantity_2: f64,
    #[pyo3(get, set)]
    pub vol_1: f64,
    #[pyo3(get, set)]
    pub vol_2: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub notional: f64,
}

impl CommoditySpreadOption {
    fn to_core(self) -> PyResult<CoreCommoditySpreadOption> {
        Ok(CoreCommoditySpreadOption {
            option_type: parse_option_type_str(&self.option_type)?,
            forward_1: self.forward_1,
            forward_2: self.forward_2,
            strike: self.strike,
            quantity_1: self.quantity_1,
            quantity_2: self.quantity_2,
            vol_1: self.vol_1,
            vol_2: self.vol_2,
            rho: self.rho,
            risk_free_rate: self.risk_free_rate,
            maturity: self.maturity,
            notional: self.notional,
        })
    }
}

#[pymethods]
impl CommoditySpreadOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        forward_1: f64,
        forward_2: f64,
        strike: f64,
        quantity_1: f64,
        quantity_2: f64,
        vol_1: f64,
        vol_2: f64,
        rho: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            forward_1,
            forward_2,
            strike,
            quantity_1,
            quantity_2,
            vol_1,
            vol_2,
            rho,
            risk_free_rate,
            maturity,
            notional,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn crack_spread(
        option_type: String,
        refined_forward: f64,
        crude_forward: f64,
        strike: f64,
        refined_ratio: f64,
        crude_ratio: f64,
        vol_refined: f64,
        vol_crude: f64,
        rho: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
    ) -> PyResult<Self> {
        let inner = CoreCommoditySpreadOption::crack_spread(
            parse_option_type_str(&option_type)?,
            refined_forward,
            crude_forward,
            strike,
            refined_ratio,
            crude_ratio,
            vol_refined,
            vol_crude,
            rho,
            risk_free_rate,
            maturity,
            notional,
        );
        Self::new(
            option_type,
            inner.forward_1,
            inner.forward_2,
            inner.strike,
            inner.quantity_1,
            inner.quantity_2,
            inner.vol_1,
            inner.vol_2,
            inner.rho,
            inner.risk_free_rate,
            inner.maturity,
            inner.notional,
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn spark_spread(
        option_type: String,
        power_forward: f64,
        gas_forward: f64,
        strike: f64,
        heat_rate: f64,
        vol_power: f64,
        vol_gas: f64,
        rho: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
    ) -> PyResult<Self> {
        let inner = CoreCommoditySpreadOption::spark_spread(
            parse_option_type_str(&option_type)?,
            power_forward,
            gas_forward,
            strike,
            heat_rate,
            vol_power,
            vol_gas,
            rho,
            risk_free_rate,
            maturity,
            notional,
        );
        Self::new(
            option_type,
            inner.forward_1,
            inner.forward_2,
            inner.strike,
            inner.quantity_1,
            inner.quantity_2,
            inner.vol_1,
            inner.vol_2,
            inner.rho,
            inner.risk_free_rate,
            inner.maturity,
            inner.notional,
        )
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }

    fn price_kirk(&self) -> PyResult<f64> {
        self.clone().to_core()?.price_kirk().map_err(map_err_string)
    }

    fn price_two_factor_mc(
        &self,
        model: TwoFactorSpreadModel,
        num_paths: usize,
        seed: u64,
    ) -> PyResult<(f64, f64)> {
        self.clone()
            .to_core()?
            .price_two_factor_mc(&model.to_core(), num_paths, seed)
            .map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ConvertibleBond {
    #[pyo3(get, set)]
    pub face_value: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub conversion_ratio: f64,
    #[pyo3(get, set)]
    pub call_price: Option<f64>,
    #[pyo3(get, set)]
    pub put_price: Option<f64>,
}

impl ConvertibleBond {
    fn to_core(&self) -> CoreConvertibleBond {
        CoreConvertibleBond::new(
            self.face_value,
            self.coupon_rate,
            self.maturity,
            self.conversion_ratio,
            self.call_price,
            self.put_price,
        )
    }
}

#[pymethods]
impl ConvertibleBond {
    #[new]
    fn new(
        face_value: f64,
        coupon_rate: f64,
        maturity: f64,
        conversion_ratio: f64,
        call_price: Option<f64>,
        put_price: Option<f64>,
    ) -> PyResult<Self> {
        let out = Self {
            face_value,
            coupon_rate,
            maturity,
            conversion_ratio,
            call_price,
            put_price,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CashOrNothingOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub cash: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
}

impl CashOrNothingOption {
    fn to_core(&self) -> PyResult<CoreCashOrNothingOption> {
        Ok(CoreCashOrNothingOption::new(
            parse_option_type_str(&self.option_type)?,
            self.strike,
            self.cash,
            self.expiry,
        ))
    }
}

#[pymethods]
impl CashOrNothingOption {
    #[new]
    fn new(option_type: String, strike: f64, cash: f64, expiry: f64) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            cash,
            expiry,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AssetOrNothingOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
}

impl AssetOrNothingOption {
    fn to_core(&self) -> PyResult<CoreAssetOrNothingOption> {
        Ok(CoreAssetOrNothingOption::new(
            parse_option_type_str(&self.option_type)?,
            self.strike,
            self.expiry,
        ))
    }
}

#[pymethods]
impl AssetOrNothingOption {
    #[new]
    fn new(option_type: String, strike: f64, expiry: f64) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            expiry,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct GapOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub payoff_strike: f64,
    #[pyo3(get, set)]
    pub trigger_strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
}

impl GapOption {
    fn to_core(&self) -> PyResult<CoreGapOption> {
        Ok(CoreGapOption::new(
            parse_option_type_str(&self.option_type)?,
            self.payoff_strike,
            self.trigger_strike,
            self.expiry,
        ))
    }
}

#[pymethods]
impl GapOption {
    #[new]
    fn new(
        option_type: String,
        payoff_strike: f64,
        trigger_strike: f64,
        expiry: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            payoff_strike,
            trigger_strike,
            expiry,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DoubleBarrierOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub lower_barrier: f64,
    #[pyo3(get, set)]
    pub upper_barrier: f64,
    #[pyo3(get, set)]
    pub barrier_type: String,
    #[pyo3(get, set)]
    pub rebate: f64,
}

impl DoubleBarrierOption {
    fn to_core(&self) -> PyResult<CoreDoubleBarrierOption> {
        Ok(CoreDoubleBarrierOption::new(
            parse_option_type_str(&self.option_type)?,
            self.strike,
            self.expiry,
            self.lower_barrier,
            self.upper_barrier,
            parse_double_barrier_type(&self.barrier_type)?,
            self.rebate,
        ))
    }
}

#[pymethods]
impl DoubleBarrierOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        strike: f64,
        expiry: f64,
        lower_barrier: f64,
        upper_barrier: f64,
        barrier_type: String,
        rebate: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            expiry,
            lower_barrier,
            upper_barrier,
            barrier_type,
            rebate,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct EmployeeStockOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub vesting_period: f64,
    #[pyo3(get, set)]
    pub expected_life: f64,
    #[pyo3(get, set)]
    pub early_exercise_multiple: Option<f64>,
    #[pyo3(get, set)]
    pub forfeiture_rate: f64,
    #[pyo3(get, set)]
    pub shares_outstanding: f64,
    #[pyo3(get, set)]
    pub options_granted: f64,
}

impl EmployeeStockOption {
    fn to_core(&self) -> PyResult<CoreEmployeeStockOption> {
        Ok(CoreEmployeeStockOption::new(
            parse_option_type_str(&self.option_type)?,
            self.strike,
            self.maturity,
            self.vesting_period,
            self.expected_life,
            self.early_exercise_multiple,
            self.forfeiture_rate,
            self.shares_outstanding,
            self.options_granted,
        ))
    }
}

#[pymethods]
impl EmployeeStockOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        strike: f64,
        maturity: f64,
        vesting_period: f64,
        expected_life: f64,
        early_exercise_multiple: Option<f64>,
        forfeiture_rate: f64,
        shares_outstanding: f64,
        options_granted: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            maturity,
            vesting_period,
            expected_life,
            early_exercise_multiple,
            forfeiture_rate,
            shares_outstanding,
            options_granted,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }

    fn effective_maturity(&self) -> PyResult<f64> {
        Ok(self.to_core()?.effective_maturity())
    }

    fn dilution_factor(&self) -> PyResult<f64> {
        Ok(self.to_core()?.dilution_factor())
    }

    fn attrition_factor(&self) -> PyResult<f64> {
        Ok(self.to_core()?.attrition_factor())
    }

    fn price_binomial(
        &self,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        steps: usize,
    ) -> PyResult<f64> {
        self.to_core()?
            .price_binomial(spot, rate, dividend_yield, vol, steps)
            .map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FxOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub domestic_rate: f64,
    #[pyo3(get, set)]
    pub foreign_rate: f64,
    #[pyo3(get, set)]
    pub spot_fx: f64,
    #[pyo3(get, set)]
    pub strike_fx: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
}

impl FxOption {
    fn to_core(&self) -> PyResult<CoreFxOption> {
        Ok(CoreFxOption::new(
            parse_option_type_str(&self.option_type)?,
            self.domestic_rate,
            self.foreign_rate,
            self.spot_fx,
            self.strike_fx,
            self.vol,
            self.maturity,
        ))
    }
}

#[pymethods]
impl FxOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        domestic_rate: f64,
        foreign_rate: f64,
        spot_fx: f64,
        strike_fx: f64,
        vol: f64,
        maturity: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            domestic_rate,
            foreign_rate,
            spot_fx,
            strike_fx,
            vol,
            maturity,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PowerOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
}

impl PowerOption {
    fn to_core(&self) -> PyResult<CorePowerOption> {
        Ok(CorePowerOption::new(
            parse_option_type_str(&self.option_type)?,
            self.strike,
            self.alpha,
            self.expiry,
        ))
    }
}

#[pymethods]
impl PowerOption {
    #[new]
    fn new(option_type: String, strike: f64, alpha: f64, expiry: f64) -> PyResult<Self> {
        let out = Self {
            option_type,
            strike,
            alpha,
            expiry,
        };
        out.validate()?;
        Ok(out)
    }

    #[staticmethod]
    fn call(strike: f64, alpha: f64, expiry: f64) -> Self {
        Self {
            option_type: "call".to_string(),
            strike,
            alpha,
            expiry,
        }
    }

    #[staticmethod]
    fn put(strike: f64, alpha: f64, expiry: f64) -> Self {
        Self {
            option_type: "put".to_string(),
            strike,
            alpha,
            expiry,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SpreadOption {
    #[pyo3(get, set)]
    pub s1: f64,
    #[pyo3(get, set)]
    pub s2: f64,
    #[pyo3(get, set)]
    pub k: f64,
    #[pyo3(get, set)]
    pub vol1: f64,
    #[pyo3(get, set)]
    pub vol2: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub q1: f64,
    #[pyo3(get, set)]
    pub q2: f64,
    #[pyo3(get, set)]
    pub r: f64,
    #[pyo3(get, set)]
    pub t: f64,
}

impl SpreadOption {
    fn to_core(self) -> CoreSpreadOption {
        CoreSpreadOption {
            s1: self.s1,
            s2: self.s2,
            k: self.k,
            vol1: self.vol1,
            vol2: self.vol2,
            rho: self.rho,
            q1: self.q1,
            q2: self.q2,
            r: self.r,
            t: self.t,
        }
    }
}

#[pymethods]
impl SpreadOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
        let out = Self {
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
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }

    fn effective_volatility(&self) -> PyResult<f64> {
        self.clone()
            .to_core()
            .effective_volatility()
            .map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SwingOption {
    #[pyo3(get, set)]
    pub min_exercises: usize,
    #[pyo3(get, set)]
    pub max_exercises: usize,
    #[pyo3(get, set)]
    pub exercise_dates: Vec<f64>,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub payoff_per_exercise: f64,
}

impl SwingOption {
    fn to_core(&self) -> CoreSwingOption {
        CoreSwingOption::new(
            self.min_exercises,
            self.max_exercises,
            self.exercise_dates.clone(),
            self.strike,
            self.payoff_per_exercise,
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PsaModel {
    #[pyo3(get, set)]
    pub psa_speed: f64,
}

impl PsaModel {
    fn to_core(&self) -> CorePsaModel {
        CorePsaModel {
            psa_speed: self.psa_speed,
        }
    }
}

#[pymethods]
impl PsaModel {
    #[new]
    fn new(psa_speed: f64) -> Self {
        Self { psa_speed }
    }

    fn cpr(&self, month: u32) -> f64 {
        self.to_core().cpr(month)
    }

    fn smm(&self, month: u32) -> f64 {
        self.to_core().smm(month)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ConstantCpr {
    #[pyo3(get, set)]
    pub annual_cpr: f64,
}

impl ConstantCpr {
    fn to_core(&self) -> CoreConstantCpr {
        CoreConstantCpr {
            annual_cpr: self.annual_cpr,
        }
    }
}

#[pymethods]
impl ConstantCpr {
    #[new]
    fn new(annual_cpr: f64) -> Self {
        Self { annual_cpr }
    }

    fn smm(&self) -> f64 {
        self.to_core().smm()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PrepaymentModel {
    inner: CorePrepaymentModel,
}

impl PrepaymentModel {
    fn to_core(&self) -> CorePrepaymentModel {
        self.inner.clone()
    }
}

#[pymethods]
impl PrepaymentModel {
    #[new]
    fn new(kind: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match kind.to_ascii_lowercase().as_str() {
            "psa" => CorePrepaymentModel::Psa(value.extract::<PsaModel>()?.to_core()),
            "constantcpr" | "constant_cpr" | "constant-cpr" => {
                CorePrepaymentModel::ConstantCpr(value.extract::<ConstantCpr>()?.to_core())
            }
            _ => {
                return Err(invalid_input(
                    "prepayment model kind must be 'psa' or 'constant_cpr'",
                ));
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CorePrepaymentModel::Psa(_) => "psa",
            CorePrepaymentModel::ConstantCpr(_) => "constant_cpr",
        }
    }

    fn smm(&self, month: u32) -> f64 {
        self.inner.smm(month)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MbsCashflow {
    #[pyo3(get, set)]
    pub month: u32,
    #[pyo3(get, set)]
    pub interest: f64,
    #[pyo3(get, set)]
    pub scheduled_principal: f64,
    #[pyo3(get, set)]
    pub prepayment: f64,
    #[pyo3(get, set)]
    pub total_principal: f64,
    #[pyo3(get, set)]
    pub remaining_balance: f64,
    #[pyo3(get, set)]
    pub total_cashflow: f64,
}

impl MbsCashflow {
    fn from_core(inner: CoreMbsCashflow) -> Self {
        Self {
            month: inner.month,
            interest: inner.interest,
            scheduled_principal: inner.scheduled_principal,
            prepayment: inner.prepayment,
            total_principal: inner.total_principal,
            remaining_balance: inner.remaining_balance,
            total_cashflow: inner.total_cashflow,
        }
    }
}

#[pymethods]
impl MbsCashflow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        month: u32,
        interest: f64,
        scheduled_principal: f64,
        prepayment: f64,
        total_principal: f64,
        remaining_balance: f64,
        total_cashflow: f64,
    ) -> Self {
        Self {
            month,
            interest,
            scheduled_principal,
            prepayment,
            total_principal,
            remaining_balance,
            total_cashflow,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MbsPassThrough {
    #[pyo3(get, set)]
    pub original_balance: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub servicing_fee: f64,
    #[pyo3(get, set)]
    pub original_term: u32,
    #[pyo3(get, set)]
    pub age: u32,
    #[pyo3(get, set)]
    pub prepayment: PrepaymentModel,
}

impl MbsPassThrough {
    fn to_core(&self) -> CoreMbsPassThrough {
        CoreMbsPassThrough {
            original_balance: self.original_balance,
            coupon_rate: self.coupon_rate,
            servicing_fee: self.servicing_fee,
            original_term: self.original_term,
            age: self.age,
            prepayment: self.prepayment.to_core(),
        }
    }
}

#[pymethods]
impl MbsPassThrough {
    #[new]
    fn new(
        original_balance: f64,
        coupon_rate: f64,
        servicing_fee: f64,
        original_term: u32,
        age: u32,
        prepayment: PrepaymentModel,
    ) -> Self {
        Self {
            original_balance,
            coupon_rate,
            servicing_fee,
            original_term,
            age,
            prepayment,
        }
    }

    fn cashflows(&self) -> Vec<MbsCashflow> {
        self.to_core()
            .cashflows()
            .into_iter()
            .map(MbsCashflow::from_core)
            .collect()
    }

    fn price(&self, yield_rate: f64) -> f64 {
        self.to_core().price(yield_rate)
    }

    fn wal(&self) -> f64 {
        self.to_core().wal()
    }

    fn oas(&self, market_price: f64, base_yields: Vec<f64>) -> f64 {
        self.to_core().oas(market_price, &base_yields)
    }

    fn effective_duration(&self, yield_rate: f64) -> f64 {
        self.to_core().effective_duration(yield_rate)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct IoStrip {
    #[pyo3(get, set)]
    pub mbs: MbsPassThrough,
}

#[pymethods]
impl IoStrip {
    #[new]
    fn new(mbs: MbsPassThrough) -> Self {
        Self { mbs }
    }

    fn cashflows(&self) -> Vec<(u32, f64)> {
        self.mbs
            .to_core()
            .cashflows()
            .iter()
            .map(|c| (c.month, c.interest))
            .collect()
    }

    fn price(&self, yield_rate: f64) -> f64 {
        let monthly_yield = yield_rate / 12.0;
        self.cashflows()
            .iter()
            .map(|(m, cf)| cf / (1.0 + monthly_yield).powi(*m as i32))
            .sum()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct PoStrip {
    #[pyo3(get, set)]
    pub mbs: MbsPassThrough,
}

#[pymethods]
impl PoStrip {
    #[new]
    fn new(mbs: MbsPassThrough) -> Self {
        Self { mbs }
    }

    fn cashflows(&self) -> Vec<(u32, f64)> {
        self.mbs
            .to_core()
            .cashflows()
            .iter()
            .map(|c| (c.month, c.total_principal))
            .collect()
    }

    fn price(&self, yield_rate: f64) -> f64 {
        let monthly_yield = yield_rate / 12.0;
        self.cashflows()
            .iter()
            .map(|(m, cf)| cf / (1.0 + monthly_yield).powi(*m as i32))
            .sum()
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
    ) -> PyResult<Self> {
        let out = Self {
            notional,
            coupon_rate,
            lower_bound,
            upper_bound,
            fixing_times,
            payment_time,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
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

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct BestOfTwoCallOption {
    #[pyo3(get, set)]
    pub s1: f64,
    #[pyo3(get, set)]
    pub s2: f64,
    #[pyo3(get, set)]
    pub k: f64,
    #[pyo3(get, set)]
    pub vol1: f64,
    #[pyo3(get, set)]
    pub vol2: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub q1: f64,
    #[pyo3(get, set)]
    pub q2: f64,
    #[pyo3(get, set)]
    pub r: f64,
    #[pyo3(get, set)]
    pub t: f64,
}

impl BestOfTwoCallOption {
    fn to_core(self) -> CoreBestOfTwoCallOption {
        CoreBestOfTwoCallOption {
            s1: self.s1,
            s2: self.s2,
            k: self.k,
            vol1: self.vol1,
            vol2: self.vol2,
            rho: self.rho,
            q1: self.q1,
            q2: self.q2,
            r: self.r,
            t: self.t,
        }
    }
}

#[pymethods]
impl BestOfTwoCallOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
        let out = Self {
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
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct WorstOfTwoCallOption {
    #[pyo3(get, set)]
    pub s1: f64,
    #[pyo3(get, set)]
    pub s2: f64,
    #[pyo3(get, set)]
    pub k: f64,
    #[pyo3(get, set)]
    pub vol1: f64,
    #[pyo3(get, set)]
    pub vol2: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub q1: f64,
    #[pyo3(get, set)]
    pub q2: f64,
    #[pyo3(get, set)]
    pub r: f64,
    #[pyo3(get, set)]
    pub t: f64,
}

impl WorstOfTwoCallOption {
    fn to_core(self) -> CoreWorstOfTwoCallOption {
        CoreWorstOfTwoCallOption {
            s1: self.s1,
            s2: self.s2,
            k: self.k,
            vol1: self.vol1,
            vol2: self.vol2,
            rho: self.rho,
            q1: self.q1,
            q2: self.q2,
            r: self.r,
            t: self.t,
        }
    }
}

#[pymethods]
impl WorstOfTwoCallOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
        let out = Self {
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
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TwoAssetCorrelationOption {
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub s1: f64,
    #[pyo3(get, set)]
    pub s2: f64,
    #[pyo3(get, set)]
    pub k1: f64,
    #[pyo3(get, set)]
    pub k2: f64,
    #[pyo3(get, set)]
    pub vol1: f64,
    #[pyo3(get, set)]
    pub vol2: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub q1: f64,
    #[pyo3(get, set)]
    pub q2: f64,
    #[pyo3(get, set)]
    pub r: f64,
    #[pyo3(get, set)]
    pub t: f64,
}

impl TwoAssetCorrelationOption {
    fn to_core(self) -> PyResult<CoreTwoAssetCorrelationOption> {
        Ok(CoreTwoAssetCorrelationOption {
            option_type: parse_option_type_str(&self.option_type)?,
            s1: self.s1,
            s2: self.s2,
            k1: self.k1,
            k2: self.k2,
            vol1: self.vol1,
            vol2: self.vol2,
            rho: self.rho,
            q1: self.q1,
            q2: self.q2,
            r: self.r,
            t: self.t,
        })
    }
}

#[pymethods]
impl TwoAssetCorrelationOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        option_type: String,
        s1: f64,
        s2: f64,
        k1: f64,
        k2: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        r: f64,
        t: f64,
    ) -> PyResult<Self> {
        let out = Self {
            option_type,
            s1,
            s2,
            k1,
            k2,
            vol1,
            vol2,
            rho,
            q1,
            q2,
            r,
            t,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct VarianceOptionQuote {
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub call_price: f64,
    #[pyo3(get, set)]
    pub put_price: f64,
}

impl VarianceOptionQuote {
    fn to_core(self) -> CoreVarianceOptionQuote {
        CoreVarianceOptionQuote::new(self.strike, self.call_price, self.put_price)
    }
}

#[pymethods]
impl VarianceOptionQuote {
    #[new]
    fn new(strike: f64, call_price: f64, put_price: f64) -> PyResult<Self> {
        let out = Self {
            strike,
            call_price,
            put_price,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }
}

#[pyfunction]
fn hdd_day(average_temperature: f64, base_temperature: f64) -> f64 {
    openferric_core::instruments::hdd_day(average_temperature, base_temperature)
}

#[pyfunction]
fn cdd_day(average_temperature: f64, base_temperature: f64) -> f64 {
    openferric_core::instruments::cdd_day(average_temperature, base_temperature)
}

#[pyfunction]
fn cumulative_hdd(temperatures: Vec<f64>, base_temperature: f64) -> f64 {
    openferric_core::instruments::cumulative_hdd(&temperatures, base_temperature)
}

#[pyfunction]
fn cumulative_cdd(temperatures: Vec<f64>, base_temperature: f64) -> f64 {
    openferric_core::instruments::cumulative_cdd(&temperatures, base_temperature)
}

#[pyfunction]
fn cumulative_degree_days(
    temperatures: Vec<f64>,
    base_temperature: f64,
    index_type: String,
) -> PyResult<f64> {
    Ok(openferric_core::instruments::cumulative_degree_days(
        &temperatures,
        base_temperature,
        parse_degree_day_type(&index_type)?,
    ))
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct WeatherSwap {
    #[pyo3(get, set)]
    pub index_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub tick_size: f64,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub is_payer: bool,
    #[pyo3(get, set)]
    pub discount_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
}

impl WeatherSwap {
    fn to_core(self) -> PyResult<CoreWeatherSwap> {
        Ok(CoreWeatherSwap {
            index_type: parse_degree_day_type(&self.index_type)?,
            strike: self.strike,
            tick_size: self.tick_size,
            notional: self.notional,
            is_payer: self.is_payer,
            discount_rate: self.discount_rate,
            maturity: self.maturity,
        })
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct WeatherOption {
    #[pyo3(get, set)]
    pub index_type: String,
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub tick_size: f64,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub discount_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
}

impl WeatherOption {
    fn to_core(self) -> PyResult<CoreWeatherOption> {
        Ok(CoreWeatherOption {
            index_type: parse_degree_day_type(&self.index_type)?,
            option_type: parse_option_type_str(&self.option_type)?,
            strike: self.strike,
            tick_size: self.tick_size,
            notional: self.notional,
            discount_rate: self.discount_rate,
            maturity: self.maturity,
        })
    }
}

#[pymethods]
impl WeatherOption {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        index_type: String,
        option_type: String,
        strike: f64,
        tick_size: f64,
        notional: f64,
        discount_rate: f64,
        maturity: f64,
    ) -> PyResult<Self> {
        let out = Self {
            index_type,
            option_type,
            strike,
            tick_size,
            notional,
            discount_rate,
            maturity,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }

    fn payoff(&self, realized_index: f64) -> PyResult<f64> {
        self.clone().to_core()?.payoff(realized_index).map_err(map_err_string)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CatastropheBond {
    #[pyo3(get, set)]
    pub principal: f64,
    #[pyo3(get, set)]
    pub coupon_rate: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub coupon_frequency: usize,
    #[pyo3(get, set)]
    pub loss_intensity: f64,
    #[pyo3(get, set)]
    pub expected_loss_per_event: f64,
}

impl CatastropheBond {
    fn to_core(self) -> CoreCatastropheBond {
        CoreCatastropheBond {
            principal: self.principal,
            coupon_rate: self.coupon_rate,
            risk_free_rate: self.risk_free_rate,
            maturity: self.maturity,
            coupon_frequency: self.coupon_frequency,
            loss_intensity: self.loss_intensity,
            expected_loss_per_event: self.expected_loss_per_event,
        }
    }
}

#[pymethods]
impl CatastropheBond {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        principal: f64,
        coupon_rate: f64,
        risk_free_rate: f64,
        maturity: f64,
        coupon_frequency: usize,
        loss_intensity: f64,
        expected_loss_per_event: f64,
    ) -> PyResult<Self> {
        let out = Self {
            principal,
            coupon_rate,
            risk_free_rate,
            maturity,
            coupon_frequency,
            loss_intensity,
            expected_loss_per_event,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core().validate().map_err(map_err_string)
    }

    fn price(&self) -> PyResult<f64> {
        self.clone().to_core().price().map_err(map_err_string)
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
    #[pyo3(get, set)]
    pub tarf_type: String,
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
            tarf_type: parse_tarf_type(&self.tarf_type)?,
        })
    }
}

#[pymethods]
impl Tarf {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        strike: f64,
        notional_per_fixing: f64,
        ko_barrier: f64,
        target_profit: f64,
        downside_leverage: f64,
        fixing_times: Vec<f64>,
        tarf_type: String,
    ) -> PyResult<Self> {
        let out = Self {
            strike,
            notional_per_fixing,
            ko_barrier,
            target_profit,
            downside_leverage,
            fixing_times,
            tarf_type,
        };
        out.validate()?;
        Ok(out)
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
            tarf_type: "standard".to_string(),
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
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

#[pymethods]
impl DiscreteCashFlow {
    #[new]
    fn new(time: f64, amount: f64) -> Self {
        Self { time, amount }
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
                .map(|cf| CoreDiscreteCashFlow {
                    time: cf.time,
                    amount: cf.amount,
                })
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
    ) -> PyResult<Self> {
        let out = Self {
            project_value,
            volatility,
            risk_free_rate,
            maturity,
            steps,
            cash_flows,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }
}

macro_rules! simple_payload_enum {
    ($name:ident) => {
        #[pyclass(module = "openferric")]
        pub struct $name {
            kind: String,
            payload: Py<PyAny>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(kind: String, payload: Py<PyAny>) -> Self {
                Self { kind, payload }
            }

            fn clone_ref(&self, py: Python<'_>) -> Self {
                Self {
                    kind: self.kind.clone(),
                    payload: self.payload.clone_ref(py),
                }
            }

            #[getter]
            fn kind(&self) -> String {
                self.kind.clone()
            }

            #[getter]
            fn payload(&self, py: Python<'_>) -> Py<PyAny> {
                self.payload.clone_ref(py)
            }
        }
    };
}

simple_payload_enum!(RealOptionInstrument);
simple_payload_enum!(ExoticOption);
simple_payload_enum!(StructuredCoupon);
simple_payload_enum!(CouponType);
simple_payload_enum!(TradeInstrument);

#[pyclass(module = "openferric")]
pub struct CouponPeriod {
    #[pyo3(get, set)]
    pub start_time: f64,
    #[pyo3(get, set)]
    pub end_time: f64,
    #[pyo3(get, set)]
    pub payment_time: f64,
    coupon: Py<PyAny>,
}

#[pymethods]
impl CouponPeriod {
    #[new]
    fn new(start_time: f64, end_time: f64, payment_time: f64, coupon: Py<PyAny>) -> Self {
        Self {
            start_time,
            end_time,
            payment_time,
            coupon,
        }
    }

    #[getter]
    fn coupon(&self, py: Python<'_>) -> Py<PyAny> {
        self.coupon.clone_ref(py)
    }

    fn accrual(&self) -> f64 {
        (self.end_time - self.start_time).max(0.0)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ExerciseSchedule {
    #[pyo3(get, set)]
    pub bermudan_dates: Vec<f64>,
    #[pyo3(get, set)]
    pub notice_period: f64,
}

#[pymethods]
impl ExerciseSchedule {
    #[new]
    fn new(bermudan_dates: Vec<f64>, notice_period: f64) -> Self {
        Self {
            bermudan_dates,
            notice_period,
        }
    }

    fn decision_times(&self) -> Vec<f64> {
        let mut dates = self
            .bermudan_dates
            .iter()
            .map(|t| (t - self.notice_period).max(0.0))
            .collect::<Vec<_>>();
        dates.sort_by(|a, b| a.total_cmp(b));
        dates
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TradeMetadata {
    #[pyo3(get, set)]
    pub trade_id: String,
    #[pyo3(get, set)]
    pub version: u64,
    #[pyo3(get, set)]
    pub timestamp_unix_ms: i64,
}

#[pymethods]
impl TradeMetadata {
    #[new]
    fn new(trade_id: String, version: u64, timestamp_unix_ms: i64) -> Self {
        Self {
            trade_id,
            version,
            timestamp_unix_ms,
        }
    }
}

#[pyclass(module = "openferric")]
pub struct Trade {
    #[pyo3(get, set)]
    pub metadata: TradeMetadata,
    instrument: Py<PyAny>,
}

#[pymethods]
impl Trade {
    #[new]
    fn new(metadata: TradeMetadata, instrument: Py<PyAny>) -> Self {
        Self {
            metadata,
            instrument,
        }
    }

    #[getter]
    fn instrument(&self, py: Python<'_>) -> Py<PyAny> {
        self.instrument.clone_ref(py)
    }
}

#[pyclass(module = "openferric")]
pub struct Portfolio {
    #[pyo3(get, set)]
    pub portfolio_id: String,
    #[pyo3(get, set)]
    pub market_snapshot_id: Option<String>,
    trades: Vec<Py<PyAny>>,
}

#[pymethods]
impl Portfolio {
    #[new]
    fn new(
        portfolio_id: String,
        market_snapshot_id: Option<String>,
        trades: Vec<Py<PyAny>>,
    ) -> Self {
        Self {
            portfolio_id,
            market_snapshot_id,
            trades,
        }
    }

    #[getter]
    fn trades(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.trades.iter().map(|trade| trade.clone_ref(py)).collect()
    }

    #[setter]
    fn set_trades(&mut self, trades: Vec<Py<PyAny>>) {
        self.trades = trades;
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hdd_day, module)?)?;
    module.add_function(wrap_pyfunction!(cdd_day, module)?)?;
    module.add_function(wrap_pyfunction!(cumulative_hdd, module)?)?;
    module.add_function(wrap_pyfunction!(cumulative_cdd, module)?)?;
    module.add_function(wrap_pyfunction!(cumulative_degree_days, module)?)?;
    module.add_class::<BasketType>()?;
    module.add_class::<DoubleBarrierType>()?;
    module.add_class::<TarfType>()?;
    module.add_class::<DegreeDayType>()?;
    module.add_class::<Frequency>()?;
    module.add_class::<ExerciseStyle>()?;
    module.add_class::<BarrierSpec>()?;
    module.add_class::<AsianSpec>()?;
    module.add_class::<YieldCurve>()?;
    module.add_class::<HullWhite>()?;
    module.add_class::<TwoFactorCommodityProcess>()?;
    module.add_class::<TwoFactorSpreadModel>()?;
    module.add_class::<VanillaOption>()?;
    module.add_class::<AsianOption>()?;
    module.add_class::<BarrierOption>()?;
    module.add_class::<BarrierOptionBuilder>()?;
    module.add_class::<Autocallable>()?;
    module.add_class::<PhoenixAutocallable>()?;
    module.add_class::<BasketOption>()?;
    module.add_class::<OutperformanceBasketOption>()?;
    module.add_class::<QuantoBasketOption>()?;
    module.add_class::<BermudanOption>()?;
    module.add_class::<FuturesOption>()?;
    module.add_class::<ForwardStartOption>()?;
    module.add_class::<CliquetOption>()?;
    module.add_class::<CommodityForward>()?;
    module.add_class::<CommodityFutures>()?;
    module.add_class::<CommodityOption>()?;
    module.add_class::<CommoditySpreadOption>()?;
    module.add_class::<ConvertibleBond>()?;
    module.add_class::<CashOrNothingOption>()?;
    module.add_class::<AssetOrNothingOption>()?;
    module.add_class::<GapOption>()?;
    module.add_class::<DoubleBarrierOption>()?;
    module.add_class::<EmployeeStockOption>()?;
    module.add_class::<FxOption>()?;
    module.add_class::<PowerOption>()?;
    module.add_class::<SpreadOption>()?;
    module.add_class::<SwingOption>()?;
    module.add_class::<PsaModel>()?;
    module.add_class::<ConstantCpr>()?;
    module.add_class::<PrepaymentModel>()?;
    module.add_class::<MbsCashflow>()?;
    module.add_class::<MbsPassThrough>()?;
    module.add_class::<IoStrip>()?;
    module.add_class::<PoStrip>()?;
    module.add_class::<RangeAccrual>()?;
    module.add_class::<DualRangeAccrual>()?;
    module.add_class::<BestOfTwoCallOption>()?;
    module.add_class::<WorstOfTwoCallOption>()?;
    module.add_class::<TwoAssetCorrelationOption>()?;
    module.add_class::<VarianceOptionQuote>()?;
    module.add_class::<WeatherSwap>()?;
    module.add_class::<WeatherOption>()?;
    module.add_class::<CatastropheBond>()?;
    module.add_class::<Tarf>()?;
    module.add_class::<DiscreteCashFlow>()?;
    module.add_class::<RealOptionBinomialSpec>()?;
    module.add_class::<RealOptionInstrument>()?;
    module.add_class::<ExoticOption>()?;
    module.add_class::<StructuredCoupon>()?;
    module.add_class::<CouponType>()?;
    module.add_class::<CouponPeriod>()?;
    module.add_class::<ExerciseSchedule>()?;
    module.add_class::<TradeMetadata>()?;
    module.add_class::<TradeInstrument>()?;
    module.add_class::<Trade>()?;
    module.add_class::<Portfolio>()?;
    module.add_class::<FundingRateSwap>()?;
    Ok(())
}

#[pymethods]
impl WeatherSwap {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        index_type: String,
        strike: f64,
        tick_size: f64,
        notional: f64,
        is_payer: bool,
        discount_rate: f64,
        maturity: f64,
    ) -> PyResult<Self> {
        let out = Self {
            index_type,
            strike,
            tick_size,
            notional,
            is_payer,
            discount_rate,
            maturity,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }

    fn payoff(&self, realized_index: f64) -> PyResult<f64> {
        self.clone().to_core()?.payoff(realized_index).map_err(map_err_string)
    }

    fn price_from_expected_index(&self, expected_index: f64) -> PyResult<f64> {
        self.clone()
            .to_core()?
            .price_from_expected_index(expected_index)
            .map_err(map_err_string)
    }

    fn price_from_historical_indices(&self, historical_indices: Vec<f64>) -> PyResult<f64> {
        self.clone()
            .to_core()?
            .price_from_historical_indices(&historical_indices)
            .map_err(map_err_string)
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
    ) -> PyResult<Self> {
        let out = Self {
            notional,
            coupon_rate,
            lower_bound,
            upper_bound,
            fixing_times,
            payment_time,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }
}

#[pymethods]
impl SwingOption {
    #[new]
    fn new(
        min_exercises: usize,
        max_exercises: usize,
        exercise_dates: Vec<f64>,
        strike: f64,
        payoff_per_exercise: f64,
    ) -> PyResult<Self> {
        let out = Self {
            min_exercises,
            max_exercises,
            exercise_dates,
            strike,
            payoff_per_exercise,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }
}

#[pymethods]
impl CommodityOption {
    #[new]
    fn new(
        forward: f64,
        strike: f64,
        vol: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
        option_type: String,
    ) -> PyResult<Self> {
        let out = Self {
            forward,
            strike,
            vol,
            risk_free_rate,
            maturity,
            notional,
            option_type,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.clone().to_core()?.validate().map_err(map_err_string)
    }

    fn as_futures_option(&self) -> PyResult<FuturesOption> {
        Ok(FuturesOption::from_core(self.clone().to_core()?.as_futures_option()))
    }

    fn price_black76(&self) -> PyResult<f64> {
        self.clone().to_core()?.price_black76().map_err(map_err_string)
    }
}

#[pymethods]
impl QuantoBasketOption {
    #[new]
    fn new(
        basket: BasketOption,
        fx_rate: f64,
        fx_vol: f64,
        asset_fx_corr: Vec<f64>,
        domestic_rate: f64,
        foreign_rate: f64,
    ) -> PyResult<Self> {
        let out = Self {
            basket,
            fx_rate,
            fx_vol,
            asset_fx_corr,
            domestic_rate,
            foreign_rate,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(map_err_string)
    }
}

#[pymethods]
impl TwoFactorSpreadModel {
    #[new]
    fn new(
        leg_1: TwoFactorCommodityProcess,
        leg_2: TwoFactorCommodityProcess,
        rho_fast: f64,
        rho_slow: f64,
    ) -> Self {
        Self {
            leg_1,
            leg_2,
            rho_fast,
            rho_slow,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(map_err_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "TwoFactorSpreadModel(leg_1={}, leg_2={}, rho_fast={}, rho_slow={})",
            self.leg_1.__repr__(),
            self.leg_2.__repr__(),
            self.rho_fast,
            self.rho_slow
        )
    }
}
