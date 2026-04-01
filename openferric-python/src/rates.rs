use chrono::NaiveDate;
use openferric_core::math::interpolation::ExtrapolationMode;
use openferric_core::rates::adjustments as core_adjustments;
use openferric_core::rates::bond::FixedRateBond as CoreFixedRateBond;
use openferric_core::rates::calendar::{
    BusinessDayConvention as CoreBusinessDayConvention, Calendar as CoreCalendar,
    CustomCalendar as CoreCustomCalendar, FinancialCenter as CoreFinancialCenter,
    Frequency as CoreFrequency, RollConvention as CoreRollConvention,
    ScheduleConfig as CoreScheduleConfig, StubConvention as CoreStubConvention,
    WeekendConvention as CoreWeekendConvention, add_business_days as core_add_business_days,
    add_months as core_add_months, adjust_business_day as core_adjust_business_day,
    business_day_count as core_business_day_count, generate_schedule as core_generate_schedule,
    generate_schedule_with_config as core_generate_schedule_with_config,
    is_cds_standard_date as core_is_cds_standard_date, is_imm_date as core_is_imm_date,
    next_cds_date as core_next_cds_date, next_imm_date as core_next_imm_date,
    previous_cds_date as core_previous_cds_date, previous_imm_date as core_previous_imm_date,
    subtract_business_days as core_subtract_business_days, third_wednesday as core_third_wednesday,
    year_fraction_business_252 as core_year_fraction_business_252,
};
use openferric_core::rates::capfloor::CapFloor as CoreCapFloor;
use openferric_core::rates::cms::{
    CmsConvexityParams as CoreCmsConvexityParams, CmsSpreadOption as CoreCmsSpreadOption,
    CmsSpreadOptionType as CoreCmsSpreadOptionType, CmsSpreadResult as CoreCmsSpreadResult,
    cms_convexity_adjustment as core_cms_convexity_adjustment,
    cms_spread_option_mc as core_cms_spread_option_mc,
    sabr_cms_convexity_adjustment as core_sabr_cms_convexity_adjustment,
};
use openferric_core::rates::day_count::{
    DayCountConvention as CoreDayCountConvention, year_fraction as core_year_fraction,
};
use openferric_core::rates::fra::ForwardRateAgreement as CoreForwardRateAgreement;
use openferric_core::rates::futures::{
    Future as CoreFuture, InterestRateFutureQuote as CoreInterestRateFutureQuote,
};
use openferric_core::rates::inflation::{
    InflationCurve as CoreInflationCurve, InflationCurveBuilder as CoreInflationCurveBuilder,
    InflationIndexedBond as CoreInflationIndexedBond,
    YearOnYearInflationSwap as CoreYearOnYearInflationSwap,
    ZeroCouponInflationSwap as CoreZeroCouponInflationSwap,
};
use openferric_core::rates::multi_curve::{
    MultiCurveEnvironment as CoreMultiCurveEnvironment,
    dual_curve_bootstrap as core_dual_curve_bootstrap,
    price_irs_multi_curve as core_price_irs_multi_curve,
};
use openferric_core::rates::ois::{
    BasisSwap as CoreBasisSwap, OvernightIndexSwap as CoreOvernightIndexSwap,
};
use openferric_core::rates::swap::{
    InterestRateSwap as CoreInterestRateSwap, SwapBuilder as CoreSwapBuilder,
};
use openferric_core::rates::swaption::Swaption as CoreSwaption;
use openferric_core::rates::xccy_swap::XccySwap as CoreXccySwap;
use openferric_core::rates::yield_curve::{
    YieldCurve as CoreYieldCurve, YieldCurveBuilder as CoreYieldCurveBuilder,
    YieldCurveInterpolationMethod as CoreYieldCurveInterpolationMethod,
    YieldCurveInterpolationSettings as CoreYieldCurveInterpolationSettings,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_date(value: &str) -> PyResult<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|_| {
        PyValueError::new_err(format!(
            "invalid date '{value}'; use YYYY-MM-DD like 2026-03-27"
        ))
    })
}

fn format_date(value: NaiveDate) -> String {
    value.format("%Y-%m-%d").to_string()
}

fn map_debug_err<T, E: std::fmt::Debug>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

fn map_string_err<T>(result: Result<T, String>) -> PyResult<T> {
    result.map_err(PyValueError::new_err)
}

fn parse_extrapolation_mode(value: &str) -> PyResult<ExtrapolationMode> {
    match value.to_ascii_lowercase().as_str() {
        "flat" => Ok(ExtrapolationMode::Flat),
        "linear" => Ok(ExtrapolationMode::Linear),
        "error" => Ok(ExtrapolationMode::Error),
        _ => Err(PyValueError::new_err(format!(
            "invalid extrapolation mode '{value}'; use 'flat', 'linear', or 'error'"
        ))),
    }
}

fn extrapolation_mode_name(value: ExtrapolationMode) -> &'static str {
    match value {
        ExtrapolationMode::Flat => "flat",
        ExtrapolationMode::Linear => "linear",
        ExtrapolationMode::Error => "error",
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct DayCountConvention {
    inner: CoreDayCountConvention,
}

impl DayCountConvention {
    fn to_core(self) -> CoreDayCountConvention {
        self.inner
    }

    fn from_core(inner: CoreDayCountConvention) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl DayCountConvention {
    #[staticmethod]
    fn act360() -> Self {
        Self::from_core(CoreDayCountConvention::Act360)
    }

    #[staticmethod]
    fn act365_fixed() -> Self {
        Self::from_core(CoreDayCountConvention::Act365Fixed)
    }

    #[staticmethod]
    fn act_act_isda() -> Self {
        Self::from_core(CoreDayCountConvention::ActActISDA)
    }

    #[staticmethod]
    fn thirty360() -> Self {
        Self::from_core(CoreDayCountConvention::Thirty360)
    }

    #[staticmethod]
    fn thirty_e360() -> Self {
        Self::from_core(CoreDayCountConvention::ThirtyE360)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreDayCountConvention::Act360 => "Act360",
            CoreDayCountConvention::Act365Fixed => "Act365Fixed",
            CoreDayCountConvention::ActActISDA => "ActActISDA",
            CoreDayCountConvention::Thirty360 => "Thirty360",
            CoreDayCountConvention::ThirtyE360 => "ThirtyE360",
        }
    }

    fn __repr__(&self) -> String {
        format!("DayCountConvention.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Frequency {
    inner: CoreFrequency,
}

impl Frequency {
    fn to_core(self) -> CoreFrequency {
        self.inner
    }

    fn from_core(inner: CoreFrequency) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Frequency {
    #[staticmethod]
    fn annual() -> Self {
        Self::from_core(CoreFrequency::Annual)
    }

    #[staticmethod]
    fn semi_annual() -> Self {
        Self::from_core(CoreFrequency::SemiAnnual)
    }

    #[staticmethod]
    fn quarterly() -> Self {
        Self::from_core(CoreFrequency::Quarterly)
    }

    #[staticmethod]
    fn monthly() -> Self {
        Self::from_core(CoreFrequency::Monthly)
    }

    fn months(&self) -> i32 {
        self.inner.months()
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreFrequency::Annual => "Annual",
            CoreFrequency::SemiAnnual => "SemiAnnual",
            CoreFrequency::Quarterly => "Quarterly",
            CoreFrequency::Monthly => "Monthly",
        }
    }

    fn __repr__(&self) -> String {
        format!("Frequency.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct WeekendConvention {
    inner: CoreWeekendConvention,
}

impl WeekendConvention {
    fn to_core(self) -> CoreWeekendConvention {
        self.inner
    }

    fn from_core(inner: CoreWeekendConvention) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl WeekendConvention {
    #[staticmethod]
    fn saturday_sunday() -> Self {
        Self::from_core(CoreWeekendConvention::SaturdaySunday)
    }

    #[staticmethod]
    fn friday_saturday() -> Self {
        Self::from_core(CoreWeekendConvention::FridaySaturday)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreWeekendConvention::SaturdaySunday => "SaturdaySunday",
            CoreWeekendConvention::FridaySaturday => "FridaySaturday",
        }
    }

    fn __repr__(&self) -> String {
        format!("WeekendConvention.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct BusinessDayConvention {
    inner: CoreBusinessDayConvention,
}

impl BusinessDayConvention {
    fn to_core(self) -> CoreBusinessDayConvention {
        self.inner
    }

    fn from_core(inner: CoreBusinessDayConvention) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl BusinessDayConvention {
    #[staticmethod]
    fn following() -> Self {
        Self::from_core(CoreBusinessDayConvention::Following)
    }

    #[staticmethod]
    fn modified_following() -> Self {
        Self::from_core(CoreBusinessDayConvention::ModifiedFollowing)
    }

    #[staticmethod]
    fn preceding() -> Self {
        Self::from_core(CoreBusinessDayConvention::Preceding)
    }

    #[staticmethod]
    fn modified_preceding() -> Self {
        Self::from_core(CoreBusinessDayConvention::ModifiedPreceding)
    }

    #[staticmethod]
    fn unadjusted() -> Self {
        Self::from_core(CoreBusinessDayConvention::Unadjusted)
    }

    #[staticmethod]
    fn nearest() -> Self {
        Self::from_core(CoreBusinessDayConvention::Nearest)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreBusinessDayConvention::Following => "Following",
            CoreBusinessDayConvention::ModifiedFollowing => "ModifiedFollowing",
            CoreBusinessDayConvention::Preceding => "Preceding",
            CoreBusinessDayConvention::ModifiedPreceding => "ModifiedPreceding",
            CoreBusinessDayConvention::Unadjusted => "Unadjusted",
            CoreBusinessDayConvention::Nearest => "Nearest",
        }
    }

    fn __repr__(&self) -> String {
        format!("BusinessDayConvention.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct StubConvention {
    inner: CoreStubConvention,
}

impl StubConvention {
    fn to_core(self) -> CoreStubConvention {
        self.inner
    }

    fn from_core(inner: CoreStubConvention) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl StubConvention {
    #[staticmethod]
    fn short_back() -> Self {
        Self::from_core(CoreStubConvention::ShortBack)
    }

    #[staticmethod]
    fn long_back() -> Self {
        Self::from_core(CoreStubConvention::LongBack)
    }

    #[staticmethod]
    fn short_front() -> Self {
        Self::from_core(CoreStubConvention::ShortFront)
    }

    #[staticmethod]
    fn long_front() -> Self {
        Self::from_core(CoreStubConvention::LongFront)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreStubConvention::ShortBack => "ShortBack",
            CoreStubConvention::LongBack => "LongBack",
            CoreStubConvention::ShortFront => "ShortFront",
            CoreStubConvention::LongFront => "LongFront",
        }
    }

    fn __repr__(&self) -> String {
        format!("StubConvention.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct RollConvention {
    inner: CoreRollConvention,
}

impl RollConvention {
    fn to_core(self) -> CoreRollConvention {
        self.inner
    }

    fn from_core(inner: CoreRollConvention) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl RollConvention {
    #[staticmethod]
    fn none() -> Self {
        Self::from_core(CoreRollConvention::None)
    }

    #[staticmethod]
    fn end_of_month() -> Self {
        Self::from_core(CoreRollConvention::EndOfMonth)
    }

    #[staticmethod]
    fn imm() -> Self {
        Self::from_core(CoreRollConvention::Imm)
    }

    #[staticmethod]
    fn fifteenth() -> Self {
        Self::from_core(CoreRollConvention::Fifteenth)
    }

    #[staticmethod]
    fn day_of_month(day: u32) -> Self {
        Self::from_core(CoreRollConvention::DayOfMonth(day))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreRollConvention::None => "None",
            CoreRollConvention::EndOfMonth => "EndOfMonth",
            CoreRollConvention::Imm => "Imm",
            CoreRollConvention::DayOfMonth(_) => "DayOfMonth",
            CoreRollConvention::Fifteenth => "Fifteenth",
        }
    }

    #[getter]
    fn day(&self) -> Option<u32> {
        match self.inner {
            CoreRollConvention::DayOfMonth(day) => Some(day),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CoreRollConvention::DayOfMonth(day) => format!("RollConvention.DayOfMonth({day})"),
            _ => format!("RollConvention.{}", self.kind()),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FinancialCenter {
    inner: CoreFinancialCenter,
}

impl FinancialCenter {
    fn from_core(inner: CoreFinancialCenter) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl FinancialCenter {
    #[staticmethod]
    fn nyc() -> Self {
        Self::from_core(CoreFinancialCenter::Nyc)
    }

    #[staticmethod]
    fn london() -> Self {
        Self::from_core(CoreFinancialCenter::London)
    }

    #[staticmethod]
    fn target() -> Self {
        Self::from_core(CoreFinancialCenter::Target)
    }

    #[staticmethod]
    fn tokyo() -> Self {
        Self::from_core(CoreFinancialCenter::Tokyo)
    }

    #[staticmethod]
    fn sydney() -> Self {
        Self::from_core(CoreFinancialCenter::Sydney)
    }

    #[staticmethod]
    fn hong_kong() -> Self {
        Self::from_core(CoreFinancialCenter::HongKong)
    }

    #[staticmethod]
    fn singapore() -> Self {
        Self::from_core(CoreFinancialCenter::Singapore)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreFinancialCenter::Nyc => "Nyc",
            CoreFinancialCenter::London => "London",
            CoreFinancialCenter::Target => "Target",
            CoreFinancialCenter::Tokyo => "Tokyo",
            CoreFinancialCenter::Sydney => "Sydney",
            CoreFinancialCenter::HongKong => "HongKong",
            CoreFinancialCenter::Singapore => "Singapore",
        }
    }

    fn __repr__(&self) -> String {
        format!("FinancialCenter.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct YieldCurveInterpolationMethod {
    inner: CoreYieldCurveInterpolationMethod,
}

impl YieldCurveInterpolationMethod {
    fn to_core(self) -> CoreYieldCurveInterpolationMethod {
        self.inner
    }

    fn from_core(inner: CoreYieldCurveInterpolationMethod) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl YieldCurveInterpolationMethod {
    #[staticmethod]
    fn log_linear_discount() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::LogLinearDiscount)
    }

    #[staticmethod]
    fn linear_zero_rate() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::LinearZeroRate)
    }

    #[staticmethod]
    fn monotone_convex() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::MonotoneConvex)
    }

    #[staticmethod]
    fn tension_spline(tension: f64) -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::TensionSpline { tension })
    }

    #[staticmethod]
    fn hermite_monotone() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::HermiteMonotone)
    }

    #[staticmethod]
    fn log_cubic_monotone() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::LogCubicMonotone)
    }

    #[staticmethod]
    fn nelson_siegel() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::NelsonSiegel)
    }

    #[staticmethod]
    fn nelson_siegel_svensson() -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::NelsonSiegelSvensson)
    }

    #[staticmethod]
    fn smith_wilson(ufr: f64, alpha: f64) -> Self {
        Self::from_core(CoreYieldCurveInterpolationMethod::SmithWilson { ufr, alpha })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreYieldCurveInterpolationMethod::LogLinearDiscount => "LogLinearDiscount",
            CoreYieldCurveInterpolationMethod::LinearZeroRate => "LinearZeroRate",
            CoreYieldCurveInterpolationMethod::MonotoneConvex => "MonotoneConvex",
            CoreYieldCurveInterpolationMethod::TensionSpline { .. } => "TensionSpline",
            CoreYieldCurveInterpolationMethod::HermiteMonotone => "HermiteMonotone",
            CoreYieldCurveInterpolationMethod::LogCubicMonotone => "LogCubicMonotone",
            CoreYieldCurveInterpolationMethod::NelsonSiegel => "NelsonSiegel",
            CoreYieldCurveInterpolationMethod::NelsonSiegelSvensson => "NelsonSiegelSvensson",
            CoreYieldCurveInterpolationMethod::SmithWilson { .. } => "SmithWilson",
        }
    }

    #[getter]
    fn tension(&self) -> Option<f64> {
        match self.inner {
            CoreYieldCurveInterpolationMethod::TensionSpline { tension } => Some(tension),
            _ => None,
        }
    }

    #[getter]
    fn ufr(&self) -> Option<f64> {
        match self.inner {
            CoreYieldCurveInterpolationMethod::SmithWilson { ufr, .. } => Some(ufr),
            _ => None,
        }
    }

    #[getter]
    fn alpha(&self) -> Option<f64> {
        match self.inner {
            CoreYieldCurveInterpolationMethod::SmithWilson { alpha, .. } => Some(alpha),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CoreYieldCurveInterpolationMethod::TensionSpline { tension } => {
                format!("YieldCurveInterpolationMethod.TensionSpline(tension={tension})")
            }
            CoreYieldCurveInterpolationMethod::SmithWilson { ufr, alpha } => {
                format!("YieldCurveInterpolationMethod.SmithWilson(ufr={ufr}, alpha={alpha})")
            }
            _ => format!("YieldCurveInterpolationMethod.{}", self.kind()),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct YieldCurveInterpolationSettings {
    inner: CoreYieldCurveInterpolationSettings,
}

impl YieldCurveInterpolationSettings {
    fn to_core(self) -> CoreYieldCurveInterpolationSettings {
        self.inner
    }

    fn from_core(inner: CoreYieldCurveInterpolationSettings) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl YieldCurveInterpolationSettings {
    #[new]
    #[pyo3(signature = (method=None, extrapolation="linear"))]
    fn new(method: Option<&YieldCurveInterpolationMethod>, extrapolation: &str) -> PyResult<Self> {
        Ok(Self::from_core(CoreYieldCurveInterpolationSettings {
            method: method
                .map(|value| value.to_core())
                .unwrap_or(CoreYieldCurveInterpolationMethod::LogLinearDiscount),
            extrapolation: parse_extrapolation_mode(extrapolation)?,
        }))
    }

    #[staticmethod]
    fn default() -> Self {
        Self::from_core(CoreYieldCurveInterpolationSettings::default())
    }

    #[getter]
    fn method(&self) -> YieldCurveInterpolationMethod {
        YieldCurveInterpolationMethod::from_core(self.inner.method)
    }

    #[getter]
    fn extrapolation(&self) -> &'static str {
        extrapolation_mode_name(self.inner.extrapolation)
    }

    fn __repr__(&self) -> String {
        format!(
            "YieldCurveInterpolationSettings(method={}, extrapolation='{}')",
            self.method().kind(),
            self.extrapolation()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct YieldCurve {
    inner: CoreYieldCurve,
}

impl YieldCurve {
    fn from_core(inner: CoreYieldCurve) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl YieldCurve {
    #[new]
    #[pyo3(signature = (tenors, settings=None))]
    fn new(
        tenors: Vec<(f64, f64)>,
        settings: Option<&YieldCurveInterpolationSettings>,
    ) -> PyResult<Self> {
        let inner = if let Some(settings) = settings {
            map_debug_err(CoreYieldCurve::new_with_settings(
                tenors,
                settings.to_core(),
            ))?
        } else {
            CoreYieldCurve::new(tenors)
        };
        Ok(Self::from_core(inner))
    }

    #[staticmethod]
    fn new_with_settings(
        tenors: Vec<(f64, f64)>,
        settings: &YieldCurveInterpolationSettings,
    ) -> PyResult<Self> {
        Ok(Self::from_core(map_debug_err(
            CoreYieldCurve::new_with_settings(tenors, settings.to_core()),
        )?))
    }

    #[getter]
    fn tenors(&self) -> Vec<(f64, f64)> {
        self.inner.tenors.clone()
    }

    fn interpolation_settings(&self) -> YieldCurveInterpolationSettings {
        YieldCurveInterpolationSettings::from_core(self.inner.interpolation_settings())
    }

    fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }

    fn try_discount_factor(&self, t: f64) -> PyResult<f64> {
        map_debug_err(self.inner.try_discount_factor(t))
    }

    fn zero_rate(&self, t: f64) -> f64 {
        self.inner.zero_rate(t)
    }

    fn try_zero_rate(&self, t: f64) -> PyResult<f64> {
        map_debug_err(self.inner.try_zero_rate(t))
    }

    fn forward_rate(&self, t1: f64, t2: f64) -> PyResult<f64> {
        if t2 <= t1 {
            return Err(PyValueError::new_err("t2 must be greater than t1"));
        }
        Ok(self.inner.forward_rate(t1, t2))
    }

    fn try_forward_rate(&self, t1: f64, t2: f64) -> PyResult<f64> {
        if t2 <= t1 {
            return Err(PyValueError::new_err("t2 must be greater than t1"));
        }
        map_debug_err(self.inner.try_forward_rate(t1, t2))
    }

    fn instantaneous_forward_rate(&self, t: f64) -> f64 {
        self.inner.instantaneous_forward_rate(t)
    }

    fn try_instantaneous_forward_rate(&self, t: f64) -> PyResult<f64> {
        map_debug_err(self.inner.try_instantaneous_forward_rate(t))
    }

    fn discount_factor_jacobian(&self, t: f64) -> PyResult<Vec<f64>> {
        map_debug_err(self.inner.discount_factor_jacobian(t))
    }

    fn zero_rate_jacobian(&self, t: f64) -> PyResult<Vec<f64>> {
        map_debug_err(self.inner.zero_rate_jacobian(t))
    }

    fn __repr__(&self) -> String {
        format!("YieldCurve(tenors={})", self.inner.tenors.len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct YieldCurveBuilder;

#[pymethods]
impl YieldCurveBuilder {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn from_deposits(deposits: Vec<(f64, f64)>) -> YieldCurve {
        YieldCurve::from_core(CoreYieldCurveBuilder::from_deposits(&deposits))
    }

    #[staticmethod]
    fn from_deposits_with_settings(
        deposits: Vec<(f64, f64)>,
        settings: &YieldCurveInterpolationSettings,
    ) -> PyResult<YieldCurve> {
        Ok(YieldCurve::from_core(map_debug_err(
            CoreYieldCurveBuilder::from_deposits_with_settings(&deposits, settings.to_core()),
        )?))
    }

    #[staticmethod]
    fn from_swap_rates(swap_rates: Vec<(f64, f64)>, frequency: usize) -> YieldCurve {
        YieldCurve::from_core(CoreYieldCurveBuilder::from_swap_rates(
            &swap_rates,
            frequency,
        ))
    }

    #[staticmethod]
    fn from_swap_rates_with_settings(
        swap_rates: Vec<(f64, f64)>,
        frequency: usize,
        settings: &YieldCurveInterpolationSettings,
    ) -> PyResult<YieldCurve> {
        Ok(YieldCurve::from_core(map_debug_err(
            CoreYieldCurveBuilder::from_swap_rates_with_settings(
                &swap_rates,
                frequency,
                settings.to_core(),
            ),
        )?))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CustomCalendar {
    inner: CoreCustomCalendar,
}

impl CustomCalendar {
    fn to_core(&self) -> CoreCustomCalendar {
        self.inner.clone()
    }

    fn from_core(inner: CoreCustomCalendar) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CustomCalendar {
    #[new]
    fn new(weekend_convention: &WeekendConvention) -> Self {
        Self::from_core(CoreCustomCalendar::new(weekend_convention.to_core()))
    }

    #[staticmethod]
    fn with_holidays(
        weekend_convention: &WeekendConvention,
        holidays: Vec<String>,
    ) -> PyResult<Self> {
        let holidays = holidays
            .into_iter()
            .map(|value| parse_date(&value))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self::from_core(CoreCustomCalendar::with_holidays(
            weekend_convention.to_core(),
            holidays,
        )))
    }

    fn add_holiday(&mut self, date: &str) -> PyResult<()> {
        self.inner.add_holiday(parse_date(date)?);
        Ok(())
    }

    fn add_business_day_override(&mut self, date: &str) -> PyResult<()> {
        self.inner.add_business_day_override(parse_date(date)?);
        Ok(())
    }

    fn is_business_day(&self, date: &str) -> PyResult<bool> {
        let calendar = CoreCalendar::custom(self.to_core());
        Ok(calendar.is_business_day(parse_date(date)?))
    }

    fn __repr__(&self) -> String {
        "CustomCalendar(...)".to_string()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct Calendar {
    inner: CoreCalendar,
}

impl Calendar {
    fn to_core(&self) -> CoreCalendar {
        self.inner.clone()
    }

    fn from_core(inner: CoreCalendar) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Calendar {
    #[staticmethod]
    #[pyo3(signature = (weekend_convention=None))]
    fn weekends_only(weekend_convention: Option<&WeekendConvention>) -> Self {
        let inner = if let Some(weekend_convention) = weekend_convention {
            CoreCalendar::WeekendsOnly(weekend_convention.to_core())
        } else {
            CoreCalendar::weekends_only()
        };
        Self::from_core(inner)
    }

    #[staticmethod]
    fn nyc() -> Self {
        Self::from_core(CoreCalendar::nyc())
    }

    #[staticmethod]
    fn london() -> Self {
        Self::from_core(CoreCalendar::london())
    }

    #[staticmethod]
    fn target() -> Self {
        Self::from_core(CoreCalendar::target())
    }

    #[staticmethod]
    fn tokyo() -> Self {
        Self::from_core(CoreCalendar::tokyo())
    }

    #[staticmethod]
    fn sydney() -> Self {
        Self::from_core(CoreCalendar::sydney())
    }

    #[staticmethod]
    fn hong_kong() -> Self {
        Self::from_core(CoreCalendar::hong_kong())
    }

    #[staticmethod]
    fn singapore() -> Self {
        Self::from_core(CoreCalendar::singapore())
    }

    #[staticmethod]
    fn custom(custom: &CustomCalendar) -> Self {
        Self::from_core(CoreCalendar::custom(custom.to_core()))
    }

    #[staticmethod]
    fn joint(py: Python<'_>, calendars: Vec<Py<Calendar>>) -> Self {
        let calendars = calendars
            .into_iter()
            .map(|calendar| calendar.borrow(py).to_core())
            .collect();
        Self::from_core(CoreCalendar::joint(calendars))
    }

    fn is_business_day(&self, date: &str) -> PyResult<bool> {
        Ok(self.inner.is_business_day(parse_date(date)?))
    }

    fn is_holiday(&self, date: &str) -> PyResult<bool> {
        Ok(self.inner.is_holiday(parse_date(date)?))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreCalendar::WeekendsOnly(_) => "WeekendsOnly",
            CoreCalendar::FinancialCenter(_) => "FinancialCenter",
            CoreCalendar::Custom(_) => "Custom",
            CoreCalendar::Joint(_) => "Joint",
        }
    }

    fn __repr__(&self) -> String {
        format!("Calendar.{}()", self.kind())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ScheduleConfig {
    inner: CoreScheduleConfig,
}

impl ScheduleConfig {
    fn to_core(&self) -> CoreScheduleConfig {
        self.inner.clone()
    }

    fn from_core(inner: CoreScheduleConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl ScheduleConfig {
    #[new]
    #[pyo3(signature = (calendar=None, business_day_convention=None, stub_convention=None, roll_convention=None))]
    fn new(
        calendar: Option<&Calendar>,
        business_day_convention: Option<&BusinessDayConvention>,
        stub_convention: Option<&StubConvention>,
        roll_convention: Option<&RollConvention>,
    ) -> Self {
        let defaults = CoreScheduleConfig::default();
        Self::from_core(CoreScheduleConfig {
            calendar: calendar
                .map(|value| value.to_core())
                .unwrap_or(defaults.calendar),
            business_day_convention: business_day_convention
                .map(|value| value.to_core())
                .unwrap_or(defaults.business_day_convention),
            stub_convention: stub_convention
                .map(|value| value.to_core())
                .unwrap_or(defaults.stub_convention),
            roll_convention: roll_convention
                .map(|value| value.to_core())
                .unwrap_or(defaults.roll_convention),
        })
    }

    #[staticmethod]
    fn default() -> Self {
        Self::from_core(CoreScheduleConfig::default())
    }

    #[getter]
    fn calendar(&self) -> Calendar {
        Calendar::from_core(self.inner.calendar.clone())
    }

    #[getter]
    fn business_day_convention(&self) -> BusinessDayConvention {
        BusinessDayConvention::from_core(self.inner.business_day_convention)
    }

    #[getter]
    fn stub_convention(&self) -> StubConvention {
        StubConvention::from_core(self.inner.stub_convention)
    }

    #[getter]
    fn roll_convention(&self) -> RollConvention {
        RollConvention::from_core(self.inner.roll_convention)
    }

    fn __repr__(&self) -> String {
        "ScheduleConfig(...)".to_string()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FixedRateBond {
    inner: CoreFixedRateBond,
}

impl FixedRateBond {
    fn to_core(self) -> CoreFixedRateBond {
        self.inner
    }
}

#[pymethods]
impl FixedRateBond {
    #[new]
    fn new(
        face_value: f64,
        coupon_rate: f64,
        frequency: u32,
        maturity: f64,
        day_count: &DayCountConvention,
    ) -> Self {
        Self {
            inner: CoreFixedRateBond {
                face_value,
                coupon_rate,
                frequency,
                maturity,
                day_count: day_count.to_core(),
            },
        }
    }

    #[getter]
    fn face_value(&self) -> f64 {
        self.inner.face_value
    }

    #[getter]
    fn coupon_rate(&self) -> f64 {
        self.inner.coupon_rate
    }

    #[getter]
    fn frequency(&self) -> u32 {
        self.inner.frequency
    }

    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }

    #[getter]
    fn day_count(&self) -> DayCountConvention {
        DayCountConvention::from_core(self.inner.day_count)
    }

    fn dirty_price(&self, curve: &YieldCurve) -> f64 {
        self.to_core().dirty_price(&curve.inner)
    }

    fn clean_price(&self, curve: &YieldCurve, settlement: f64) -> f64 {
        self.to_core().clean_price(&curve.inner, settlement)
    }

    fn accrued_interest(&self, settlement: f64) -> f64 {
        self.to_core().accrued_interest(settlement)
    }

    fn duration(&self, curve: &YieldCurve) -> f64 {
        self.to_core().duration(&curve.inner)
    }

    fn convexity(&self, curve: &YieldCurve) -> f64 {
        self.to_core().convexity(&curve.inner)
    }

    fn ytm(&self, market_price: f64) -> f64 {
        self.to_core().ytm(market_price)
    }

    fn __repr__(&self) -> String {
        format!(
            "FixedRateBond(face_value={}, coupon_rate={}, frequency={}, maturity={})",
            self.inner.face_value,
            self.inner.coupon_rate,
            self.inner.frequency,
            self.inner.maturity
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CapFloor {
    inner: CoreCapFloor,
}

impl CapFloor {
    fn from_core(inner: CoreCapFloor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CapFloor {
    #[new]
    fn new(
        notional: f64,
        strike: f64,
        start_date: &str,
        end_date: &str,
        frequency: &Frequency,
        day_count: &DayCountConvention,
        is_cap: bool,
    ) -> PyResult<Self> {
        Ok(Self::from_core(CoreCapFloor {
            notional,
            strike,
            start_date: parse_date(start_date)?,
            end_date: parse_date(end_date)?,
            frequency: frequency.to_core(),
            day_count: day_count.to_core(),
            is_cap,
        }))
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike
    }

    #[getter]
    fn start_date(&self) -> String {
        format_date(self.inner.start_date)
    }

    #[getter]
    fn end_date(&self) -> String {
        format_date(self.inner.end_date)
    }

    #[getter]
    fn frequency(&self) -> Frequency {
        Frequency::from_core(self.inner.frequency)
    }

    #[getter]
    fn day_count(&self) -> DayCountConvention {
        DayCountConvention::from_core(self.inner.day_count)
    }

    #[getter]
    fn is_cap(&self) -> bool {
        self.inner.is_cap
    }

    #[staticmethod]
    fn black_caplet(
        notional: f64,
        discount_factor: f64,
        accrual: f64,
        forward_rate: f64,
        strike: f64,
        vol: f64,
        expiry: f64,
    ) -> f64 {
        CoreCapFloor::black_caplet(
            notional,
            discount_factor,
            accrual,
            forward_rate,
            strike,
            vol,
            expiry,
        )
    }

    #[staticmethod]
    fn black_floorlet(
        notional: f64,
        discount_factor: f64,
        accrual: f64,
        forward_rate: f64,
        strike: f64,
        vol: f64,
        expiry: f64,
    ) -> f64 {
        CoreCapFloor::black_floorlet(
            notional,
            discount_factor,
            accrual,
            forward_rate,
            strike,
            vol,
            expiry,
        )
    }

    fn optionlet_price(
        &self,
        curve: &YieldCurve,
        vol: f64,
        period_start: &str,
        period_end: &str,
    ) -> PyResult<f64> {
        Ok(self.inner.optionlet_price(
            &curve.inner,
            vol,
            parse_date(period_start)?,
            parse_date(period_end)?,
        ))
    }

    fn price(&self, curve: &YieldCurve, vol: f64) -> f64 {
        self.inner.price(&curve.inner, vol)
    }

    fn swap_npv(&self, curve: &YieldCurve) -> f64 {
        self.inner.swap_npv(&curve.inner)
    }

    fn implied_vol(&self, market_price: f64, curve: &YieldCurve) -> f64 {
        self.inner.implied_vol(market_price, &curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "CapFloor(notional={}, strike={}, start_date='{}', end_date='{}', is_cap={})",
            self.inner.notional,
            self.inner.strike,
            format_date(self.inner.start_date),
            format_date(self.inner.end_date),
            self.inner.is_cap
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CmsConvexityParams {
    inner: CoreCmsConvexityParams,
}

impl CmsConvexityParams {
    fn to_core(self) -> CoreCmsConvexityParams {
        self.inner
    }

    fn from_core(inner: CoreCmsConvexityParams) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CmsConvexityParams {
    #[new]
    fn new(swap_rate: f64, annuity: f64, tenor: f64, expiry: f64, vol: f64) -> Self {
        Self::from_core(CoreCmsConvexityParams {
            swap_rate,
            annuity,
            tenor,
            expiry,
            vol,
        })
    }

    #[getter]
    fn swap_rate(&self) -> f64 {
        self.inner.swap_rate
    }

    #[getter]
    fn annuity(&self) -> f64 {
        self.inner.annuity
    }

    #[getter]
    fn tenor(&self) -> f64 {
        self.inner.tenor
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    #[getter]
    fn vol(&self) -> f64 {
        self.inner.vol
    }

    fn __repr__(&self) -> String {
        format!(
            "CmsConvexityParams(swap_rate={}, annuity={}, tenor={}, expiry={}, vol={})",
            self.inner.swap_rate,
            self.inner.annuity,
            self.inner.tenor,
            self.inner.expiry,
            self.inner.vol
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CmsSpreadOptionType {
    inner: CoreCmsSpreadOptionType,
}

impl CmsSpreadOptionType {
    fn to_core(self) -> CoreCmsSpreadOptionType {
        self.inner
    }

    fn from_core(inner: CoreCmsSpreadOptionType) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CmsSpreadOptionType {
    #[staticmethod]
    fn call() -> Self {
        Self::from_core(CoreCmsSpreadOptionType::Call)
    }

    #[staticmethod]
    fn put() -> Self {
        Self::from_core(CoreCmsSpreadOptionType::Put)
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreCmsSpreadOptionType::Call => "Call",
            CoreCmsSpreadOptionType::Put => "Put",
        }
    }

    fn __repr__(&self) -> String {
        format!("CmsSpreadOptionType.{}", self.name())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CmsSpreadOption {
    inner: CoreCmsSpreadOption,
}

impl CmsSpreadOption {
    fn to_core(self) -> CoreCmsSpreadOption {
        self.inner
    }
}

#[pymethods]
impl CmsSpreadOption {
    #[new]
    fn new(strike: f64, option_type: &CmsSpreadOptionType, notional: f64, expiry: f64) -> Self {
        Self {
            inner: CoreCmsSpreadOption {
                strike,
                option_type: option_type.to_core(),
                notional,
                expiry,
            },
        }
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike
    }

    #[getter]
    fn option_type(&self) -> CmsSpreadOptionType {
        CmsSpreadOptionType::from_core(self.inner.option_type)
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    fn __repr__(&self) -> String {
        format!(
            "CmsSpreadOption(strike={}, option_type={}, notional={}, expiry={})",
            self.inner.strike,
            self.option_type().name(),
            self.inner.notional,
            self.inner.expiry
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CmsSpreadResult {
    inner: CoreCmsSpreadResult,
}

impl CmsSpreadResult {
    fn from_core(inner: CoreCmsSpreadResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CmsSpreadResult {
    #[new]
    fn new(price: f64, std_error: f64, expected_cms1: f64, expected_cms2: f64) -> Self {
        Self::from_core(CoreCmsSpreadResult {
            price,
            std_error,
            expected_cms1,
            expected_cms2,
        })
    }

    #[getter]
    fn price(&self) -> f64 {
        self.inner.price
    }

    #[getter]
    fn std_error(&self) -> f64 {
        self.inner.std_error
    }

    #[getter]
    fn expected_cms1(&self) -> f64 {
        self.inner.expected_cms1
    }

    #[getter]
    fn expected_cms2(&self) -> f64 {
        self.inner.expected_cms2
    }

    fn __repr__(&self) -> String {
        format!(
            "CmsSpreadResult(price={}, std_error={}, expected_cms1={}, expected_cms2={})",
            self.inner.price,
            self.inner.std_error,
            self.inner.expected_cms1,
            self.inner.expected_cms2
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct ForwardRateAgreement {
    inner: CoreForwardRateAgreement,
}

impl ForwardRateAgreement {
    fn to_core(self) -> CoreForwardRateAgreement {
        self.inner
    }
}

#[pymethods]
impl ForwardRateAgreement {
    #[new]
    fn new(
        notional: f64,
        fixed_rate: f64,
        start_date: &str,
        end_date: &str,
        day_count: &DayCountConvention,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreForwardRateAgreement {
                notional,
                fixed_rate,
                start_date: parse_date(start_date)?,
                end_date: parse_date(end_date)?,
                day_count: day_count.to_core(),
            },
        })
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn start_date(&self) -> String {
        format_date(self.inner.start_date)
    }

    #[getter]
    fn end_date(&self) -> String {
        format_date(self.inner.end_date)
    }

    #[getter]
    fn day_count(&self) -> DayCountConvention {
        DayCountConvention::from_core(self.inner.day_count)
    }

    fn forward_rate(&self, curve: &YieldCurve) -> f64 {
        self.to_core().forward_rate(&curve.inner)
    }

    fn npv(&self, curve: &YieldCurve) -> f64 {
        self.to_core().npv(&curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ForwardRateAgreement(notional={}, fixed_rate={}, start_date='{}', end_date='{}')",
            self.inner.notional,
            self.inner.fixed_rate,
            format_date(self.inner.start_date),
            format_date(self.inner.end_date)
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct InterestRateSwap {
    inner: CoreInterestRateSwap,
}

impl InterestRateSwap {
    fn from_core(inner: CoreInterestRateSwap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl InterestRateSwap {
    #[new]
    fn new(
        notional: f64,
        fixed_rate: f64,
        float_spread: f64,
        start_date: &str,
        end_date: &str,
        fixed_freq: &Frequency,
        float_freq: &Frequency,
        calendar: &Calendar,
        business_day_convention: &BusinessDayConvention,
        stub_convention: &StubConvention,
        roll_convention: &RollConvention,
        fixed_day_count: &DayCountConvention,
        float_day_count: &DayCountConvention,
    ) -> PyResult<Self> {
        Ok(Self::from_core(CoreInterestRateSwap {
            notional,
            fixed_rate,
            float_spread,
            start_date: parse_date(start_date)?,
            end_date: parse_date(end_date)?,
            fixed_freq: fixed_freq.to_core(),
            float_freq: float_freq.to_core(),
            calendar: calendar.to_core(),
            business_day_convention: business_day_convention.to_core(),
            stub_convention: stub_convention.to_core(),
            roll_convention: roll_convention.to_core(),
            fixed_day_count: fixed_day_count.to_core(),
            float_day_count: float_day_count.to_core(),
        }))
    }

    #[staticmethod]
    fn builder() -> SwapBuilder {
        SwapBuilder {
            inner: CoreInterestRateSwap::builder(),
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn float_spread(&self) -> f64 {
        self.inner.float_spread
    }

    #[getter]
    fn start_date(&self) -> String {
        format_date(self.inner.start_date)
    }

    #[getter]
    fn end_date(&self) -> String {
        format_date(self.inner.end_date)
    }

    #[getter]
    fn fixed_freq(&self) -> Frequency {
        Frequency::from_core(self.inner.fixed_freq)
    }

    #[getter]
    fn float_freq(&self) -> Frequency {
        Frequency::from_core(self.inner.float_freq)
    }

    fn fixed_leg_pv(&self, curve: &YieldCurve) -> f64 {
        self.inner.fixed_leg_pv(&curve.inner)
    }

    fn float_leg_pv(&self, curve: &YieldCurve) -> f64 {
        self.inner.float_leg_pv(&curve.inner)
    }

    fn npv(&self, curve: &YieldCurve) -> f64 {
        self.inner.npv(&curve.inner)
    }

    fn par_rate(&self, curve: &YieldCurve) -> f64 {
        self.inner.par_rate(&curve.inner)
    }

    fn dv01(&self, curve: &YieldCurve) -> f64 {
        self.inner.dv01(&curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "InterestRateSwap(notional={}, fixed_rate={}, float_spread={}, start_date='{}', end_date='{}')",
            self.inner.notional,
            self.inner.fixed_rate,
            self.inner.float_spread,
            format_date(self.inner.start_date),
            format_date(self.inner.end_date)
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SwapBuilder {
    inner: CoreSwapBuilder,
}

#[pymethods]
impl SwapBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreSwapBuilder::default(),
        }
    }

    fn notional(&self, notional: f64) -> Self {
        Self {
            inner: self.inner.clone().notional(notional),
        }
    }

    fn fixed_rate(&self, fixed_rate: f64) -> Self {
        Self {
            inner: self.inner.clone().fixed_rate(fixed_rate),
        }
    }

    fn float_spread(&self, float_spread: f64) -> Self {
        Self {
            inner: self.inner.clone().float_spread(float_spread),
        }
    }

    fn start_date(&self, start_date: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().start_date(parse_date(start_date)?),
        })
    }

    fn end_date(&self, end_date: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().end_date(parse_date(end_date)?),
        })
    }

    fn fixed_freq(&self, fixed_freq: &Frequency) -> Self {
        Self {
            inner: self.inner.clone().fixed_freq(fixed_freq.to_core()),
        }
    }

    fn float_freq(&self, float_freq: &Frequency) -> Self {
        Self {
            inner: self.inner.clone().float_freq(float_freq.to_core()),
        }
    }

    fn calendar(&self, calendar: &Calendar) -> Self {
        Self {
            inner: self.inner.clone().calendar(calendar.to_core()),
        }
    }

    fn business_day_convention(&self, business_day_convention: &BusinessDayConvention) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .business_day_convention(business_day_convention.to_core()),
        }
    }

    fn stub_convention(&self, stub_convention: &StubConvention) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .stub_convention(stub_convention.to_core()),
        }
    }

    fn roll_convention(&self, roll_convention: &RollConvention) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .roll_convention(roll_convention.to_core()),
        }
    }

    fn fixed_day_count(&self, fixed_day_count: &DayCountConvention) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .fixed_day_count(fixed_day_count.to_core()),
        }
    }

    fn float_day_count(&self, float_day_count: &DayCountConvention) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .float_day_count(float_day_count.to_core()),
        }
    }

    fn build(&self) -> InterestRateSwap {
        InterestRateSwap::from_core(self.inner.clone().build())
    }

    fn __repr__(&self) -> String {
        "SwapBuilder(...)".to_string()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Future {
    inner: CoreFuture,
}

impl Future {
    fn to_core(self) -> CoreFuture {
        self.inner
    }
}

#[pymethods]
impl Future {
    #[new]
    fn new(
        underlying_spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        storage_cost: f64,
        convenience_yield: f64,
        expiry: f64,
    ) -> Self {
        Self {
            inner: CoreFuture {
                underlying_spot,
                risk_free_rate,
                dividend_yield,
                storage_cost,
                convenience_yield,
                expiry,
            },
        }
    }

    #[getter]
    fn underlying_spot(&self) -> f64 {
        self.inner.underlying_spot
    }

    #[getter]
    fn risk_free_rate(&self) -> f64 {
        self.inner.risk_free_rate
    }

    #[getter]
    fn dividend_yield(&self) -> f64 {
        self.inner.dividend_yield
    }

    #[getter]
    fn storage_cost(&self) -> f64 {
        self.inner.storage_cost
    }

    #[getter]
    fn convenience_yield(&self) -> f64 {
        self.inner.convenience_yield
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    fn theoretical_price(&self) -> f64 {
        self.to_core().theoretical_price()
    }

    fn basis(&self) -> f64 {
        self.to_core().basis()
    }

    fn implied_repo_rate(&self, market_price: f64) -> f64 {
        self.to_core().implied_repo_rate(market_price)
    }

    fn __repr__(&self) -> String {
        format!(
            "Future(underlying_spot={}, risk_free_rate={}, expiry={})",
            self.inner.underlying_spot, self.inner.risk_free_rate, self.inner.expiry
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct InterestRateFutureQuote;

#[pymethods]
impl InterestRateFutureQuote {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn price_from_rate(rate: f64) -> f64 {
        CoreInterestRateFutureQuote::price_from_rate(rate)
    }

    #[staticmethod]
    fn rate_from_price(price: f64) -> f64 {
        CoreInterestRateFutureQuote::rate_from_price(price)
    }

    #[staticmethod]
    fn convexity_adjustment(vol: f64, t1: f64, t2: f64) -> f64 {
        CoreInterestRateFutureQuote::convexity_adjustment(vol, t1, t2)
    }

    #[staticmethod]
    fn forward_rate_from_futures_rate(futures_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
        CoreInterestRateFutureQuote::forward_rate_from_futures_rate(futures_rate, vol, t1, t2)
    }

    #[staticmethod]
    fn futures_rate_from_forward_rate(forward_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
        CoreInterestRateFutureQuote::futures_rate_from_forward_rate(forward_rate, vol, t1, t2)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct InflationCurve {
    inner: CoreInflationCurve,
}

impl InflationCurve {
    fn from_core(inner: CoreInflationCurve) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl InflationCurve {
    #[new]
    fn new(nodes: Vec<(f64, f64)>) -> Self {
        Self::from_core(CoreInflationCurve::new(nodes))
    }

    #[getter]
    fn nodes(&self) -> Vec<(f64, f64)> {
        self.inner.nodes.clone()
    }

    fn cpi_ratio(&self, t: f64) -> f64 {
        self.inner.cpi_ratio(t)
    }

    fn zero_inflation_rate(&self, t: f64) -> f64 {
        self.inner.zero_inflation_rate(t)
    }

    fn forward_cpi_ratio(&self, t1: f64, t2: f64) -> f64 {
        self.inner.forward_cpi_ratio(t1, t2)
    }

    fn forward_inflation_rate(&self, t1: f64, t2: f64) -> f64 {
        self.inner.forward_inflation_rate(t1, t2)
    }

    fn projected_cpi(&self, cpi0: f64, t: f64) -> f64 {
        self.inner.projected_cpi(cpi0, t)
    }

    fn __repr__(&self) -> String {
        format!("InflationCurve(nodes={})", self.inner.nodes.len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct InflationCurveBuilder;

#[pymethods]
impl InflationCurveBuilder {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    fn from_zc_swap_rates(zc_swap_rates: Vec<(f64, f64)>) -> InflationCurve {
        InflationCurve::from_core(CoreInflationCurveBuilder::from_zc_swap_rates(
            &zc_swap_rates,
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct ZeroCouponInflationSwap {
    inner: CoreZeroCouponInflationSwap,
}

impl ZeroCouponInflationSwap {
    fn to_core(self) -> CoreZeroCouponInflationSwap {
        self.inner
    }
}

#[pymethods]
impl ZeroCouponInflationSwap {
    #[new]
    fn new(
        notional: f64,
        cpi_base: f64,
        fixed_rate: f64,
        tenor: f64,
        receive_inflation: bool,
    ) -> Self {
        Self {
            inner: CoreZeroCouponInflationSwap {
                notional,
                cpi_base,
                fixed_rate,
                tenor,
                receive_inflation,
            },
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn cpi_base(&self) -> f64 {
        self.inner.cpi_base
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn tenor(&self) -> f64 {
        self.inner.tenor
    }

    #[getter]
    fn receive_inflation(&self) -> bool {
        self.inner.receive_inflation
    }

    fn npv(&self, discount_curve: &YieldCurve, terminal_cpi: f64) -> f64 {
        self.to_core().npv(&discount_curve.inner, terminal_cpi)
    }

    fn npv_from_curve(&self, discount_curve: &YieldCurve, inflation_curve: &InflationCurve) -> f64 {
        self.to_core()
            .npv_from_curve(&discount_curve.inner, &inflation_curve.inner)
    }

    fn mtm(
        &self,
        valuation_time: f64,
        realized_cpi: f64,
        discount_curve: &YieldCurve,
        inflation_curve: &InflationCurve,
    ) -> f64 {
        self.to_core().mtm(
            valuation_time,
            realized_cpi,
            &discount_curve.inner,
            &inflation_curve.inner,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "ZeroCouponInflationSwap(notional={}, fixed_rate={}, tenor={}, receive_inflation={})",
            self.inner.notional,
            self.inner.fixed_rate,
            self.inner.tenor,
            self.inner.receive_inflation
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct YearOnYearInflationSwap {
    inner: CoreYearOnYearInflationSwap,
}

impl YearOnYearInflationSwap {
    fn to_core(self) -> CoreYearOnYearInflationSwap {
        self.inner
    }
}

#[pymethods]
impl YearOnYearInflationSwap {
    #[new]
    fn new(notional: f64, fixed_rate: f64, maturity_years: u32, receive_inflation: bool) -> Self {
        Self {
            inner: CoreYearOnYearInflationSwap {
                notional,
                fixed_rate,
                maturity_years,
                receive_inflation,
            },
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn maturity_years(&self) -> u32 {
        self.inner.maturity_years
    }

    #[getter]
    fn receive_inflation(&self) -> bool {
        self.inner.receive_inflation
    }

    fn npv_from_fixings(&self, discount_curve: &YieldCurve, cpi_fixings: Vec<f64>) -> f64 {
        self.to_core()
            .npv_from_fixings(&discount_curve.inner, &cpi_fixings)
    }

    fn npv_from_curve(&self, discount_curve: &YieldCurve, inflation_curve: &InflationCurve) -> f64 {
        self.to_core()
            .npv_from_curve(&discount_curve.inner, &inflation_curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "YearOnYearInflationSwap(notional={}, fixed_rate={}, maturity_years={}, receive_inflation={})",
            self.inner.notional,
            self.inner.fixed_rate,
            self.inner.maturity_years,
            self.inner.receive_inflation
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct InflationIndexedBond {
    inner: CoreInflationIndexedBond,
}

impl InflationIndexedBond {
    fn to_core(self) -> CoreInflationIndexedBond {
        self.inner
    }
}

#[pymethods]
impl InflationIndexedBond {
    #[new]
    fn new(
        face_value: f64,
        coupon_rate: f64,
        maturity_years: u32,
        coupon_frequency: u32,
        cpi_base: f64,
    ) -> Self {
        Self {
            inner: CoreInflationIndexedBond {
                face_value,
                coupon_rate,
                maturity_years,
                coupon_frequency,
                cpi_base,
            },
        }
    }

    #[getter]
    fn face_value(&self) -> f64 {
        self.inner.face_value
    }

    #[getter]
    fn coupon_rate(&self) -> f64 {
        self.inner.coupon_rate
    }

    #[getter]
    fn maturity_years(&self) -> u32 {
        self.inner.maturity_years
    }

    #[getter]
    fn coupon_frequency(&self) -> u32 {
        self.inner.coupon_frequency
    }

    #[getter]
    fn cpi_base(&self) -> f64 {
        self.inner.cpi_base
    }

    fn indexed_principal(&self, cpi_level: f64) -> f64 {
        self.to_core().indexed_principal(cpi_level)
    }

    fn price(&self, nominal_curve: &YieldCurve, inflation_curve: &InflationCurve) -> f64 {
        self.to_core()
            .price(&nominal_curve.inner, &inflation_curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "InflationIndexedBond(face_value={}, coupon_rate={}, maturity_years={})",
            self.inner.face_value, self.inner.coupon_rate, self.inner.maturity_years
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct OvernightIndexSwap {
    inner: CoreOvernightIndexSwap,
}

impl OvernightIndexSwap {
    fn to_core(self) -> CoreOvernightIndexSwap {
        self.inner
    }
}

#[pymethods]
impl OvernightIndexSwap {
    #[new]
    fn new(notional: f64, fixed_rate: f64, float_spread: f64, tenor: f64) -> Self {
        Self {
            inner: CoreOvernightIndexSwap {
                notional,
                fixed_rate,
                float_spread,
                tenor,
            },
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn float_spread(&self) -> f64 {
        self.inner.float_spread
    }

    #[getter]
    fn tenor(&self) -> f64 {
        self.inner.tenor
    }

    fn fixed_leg_pv(&self, ois_discount_curve: &YieldCurve) -> f64 {
        self.to_core().fixed_leg_pv(&ois_discount_curve.inner)
    }

    fn floating_leg_pv(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
    ) -> f64 {
        self.to_core()
            .floating_leg_pv(&ois_discount_curve.inner, &overnight_projection_curve.inner)
    }

    fn npv(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
        pay_fixed: bool,
    ) -> f64 {
        self.to_core().npv(
            &ois_discount_curve.inner,
            &overnight_projection_curve.inner,
            pay_fixed,
        )
    }

    fn par_fixed_rate(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
    ) -> f64 {
        self.to_core()
            .par_fixed_rate(&ois_discount_curve.inner, &overnight_projection_curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "OvernightIndexSwap(notional={}, fixed_rate={}, float_spread={}, tenor={})",
            self.inner.notional, self.inner.fixed_rate, self.inner.float_spread, self.inner.tenor
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct BasisSwap {
    inner: CoreBasisSwap,
}

impl BasisSwap {
    fn to_core(self) -> CoreBasisSwap {
        self.inner
    }
}

#[pymethods]
impl BasisSwap {
    #[new]
    fn new(
        notional: f64,
        spread_on_short_leg: f64,
        tenor: f64,
        short_leg_payments_per_year: u32,
        long_leg_payments_per_year: u32,
    ) -> Self {
        Self {
            inner: CoreBasisSwap {
                notional,
                spread_on_short_leg,
                tenor,
                short_leg_payments_per_year,
                long_leg_payments_per_year,
            },
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn spread_on_short_leg(&self) -> f64 {
        self.inner.spread_on_short_leg
    }

    #[getter]
    fn tenor(&self) -> f64 {
        self.inner.tenor
    }

    #[getter]
    fn short_leg_payments_per_year(&self) -> u32 {
        self.inner.short_leg_payments_per_year
    }

    #[getter]
    fn long_leg_payments_per_year(&self) -> u32 {
        self.inner.long_leg_payments_per_year
    }

    fn npv(
        &self,
        ois_discount_curve: &YieldCurve,
        short_ibor_projection_curve: &YieldCurve,
        long_ibor_projection_curve: &YieldCurve,
        pay_short_plus_spread: bool,
    ) -> f64 {
        self.to_core().npv(
            &ois_discount_curve.inner,
            &short_ibor_projection_curve.inner,
            &long_ibor_projection_curve.inner,
            pay_short_plus_spread,
        )
    }

    fn par_spread_on_short_leg(
        &self,
        ois_discount_curve: &YieldCurve,
        short_ibor_projection_curve: &YieldCurve,
        long_ibor_projection_curve: &YieldCurve,
    ) -> f64 {
        self.to_core().par_spread_on_short_leg(
            &ois_discount_curve.inner,
            &short_ibor_projection_curve.inner,
            &long_ibor_projection_curve.inner,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "BasisSwap(notional={}, spread_on_short_leg={}, tenor={})",
            self.inner.notional, self.inner.spread_on_short_leg, self.inner.tenor
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MultiCurveEnvironment {
    inner: CoreMultiCurveEnvironment,
}

impl MultiCurveEnvironment {
    fn from_core(inner: CoreMultiCurveEnvironment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl MultiCurveEnvironment {
    #[new]
    fn new(discount_curve: &YieldCurve) -> Self {
        Self::from_core(CoreMultiCurveEnvironment::new(discount_curve.inner.clone()))
    }

    fn add_forward_curve(&mut self, tenor_name: &str, curve: &YieldCurve) {
        self.inner
            .add_forward_curve(tenor_name, curve.inner.clone());
    }

    fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }

    fn forward_rate(&self, tenor_name: &str, t1: f64, t2: f64) -> Option<f64> {
        self.inner.forward_rate(tenor_name, t1, t2)
    }

    fn tenor_basis(&self, tenor1: &str, tenor2: &str, t1: f64, t2: f64) -> Option<f64> {
        self.inner.tenor_basis(tenor1, tenor2, t1, t2)
    }

    #[getter]
    fn discount_curve(&self) -> YieldCurve {
        YieldCurve::from_core(self.inner.discount_curve.clone())
    }

    #[getter]
    fn forward_curves(&self) -> Vec<(String, YieldCurve)> {
        self.inner
            .forward_curves
            .iter()
            .cloned()
            .map(|(name, curve)| (name, YieldCurve::from_core(curve)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiCurveEnvironment(forward_curves={})",
            self.inner.forward_curves.len()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct XccySwap {
    inner: CoreXccySwap,
}

impl XccySwap {
    fn to_core(self) -> CoreXccySwap {
        self.inner
    }
}

#[pymethods]
impl XccySwap {
    #[new]
    fn new(
        notional1: f64,
        notional2: f64,
        fixed_rate: f64,
        float_spread: f64,
        tenor: f64,
        fx_spot: f64,
    ) -> Self {
        Self {
            inner: CoreXccySwap {
                notional1,
                notional2,
                fixed_rate,
                float_spread,
                tenor,
                fx_spot,
            },
        }
    }

    #[getter]
    fn notional1(&self) -> f64 {
        self.inner.notional1
    }

    #[getter]
    fn notional2(&self) -> f64 {
        self.inner.notional2
    }

    #[getter]
    fn fixed_rate(&self) -> f64 {
        self.inner.fixed_rate
    }

    #[getter]
    fn float_spread(&self) -> f64 {
        self.inner.float_spread
    }

    #[getter]
    fn tenor(&self) -> f64 {
        self.inner.tenor
    }

    #[getter]
    fn fx_spot(&self) -> f64 {
        self.inner.fx_spot
    }

    fn fixed_leg_pv_ccy1(&self, ccy1_discount_curve: &YieldCurve) -> f64 {
        self.to_core().fixed_leg_pv_ccy1(&ccy1_discount_curve.inner)
    }

    fn float_leg_pv_ccy2(
        &self,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
    ) -> f64 {
        self.to_core()
            .float_leg_pv_ccy2(&ccy2_discount_curve.inner, &ccy2_projection_curve.inner)
    }

    fn npv(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.to_core().npv(
            &ccy1_discount_curve.inner,
            &ccy2_discount_curve.inner,
            pay_fixed_ccy1,
        )
    }

    fn npv_dual_curve(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.to_core().npv_dual_curve(
            &ccy1_discount_curve.inner,
            &ccy2_discount_curve.inner,
            &ccy2_projection_curve.inner,
            pay_fixed_ccy1,
        )
    }

    fn par_fixed_rate(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
    ) -> f64 {
        self.to_core().par_fixed_rate(
            &ccy1_discount_curve.inner,
            &ccy2_discount_curve.inner,
            &ccy2_projection_curve.inner,
        )
    }

    fn mtm_basis_npv(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        current_fx_spot: f64,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.to_core().mtm_basis_npv(
            &ccy1_discount_curve.inner,
            &ccy2_discount_curve.inner,
            &ccy2_projection_curve.inner,
            current_fx_spot,
            pay_fixed_ccy1,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "XccySwap(notional1={}, notional2={}, fixed_rate={}, tenor={}, fx_spot={})",
            self.inner.notional1,
            self.inner.notional2,
            self.inner.fixed_rate,
            self.inner.tenor,
            self.inner.fx_spot
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Swaption {
    inner: CoreSwaption,
}

impl Swaption {
    fn to_core(self) -> CoreSwaption {
        self.inner
    }
}

#[pymethods]
impl Swaption {
    #[new]
    fn new(
        notional: f64,
        strike: f64,
        option_expiry: f64,
        swap_tenor: f64,
        is_payer: bool,
    ) -> Self {
        Self {
            inner: CoreSwaption {
                notional,
                strike,
                option_expiry,
                swap_tenor,
                is_payer,
            },
        }
    }

    #[getter]
    fn notional(&self) -> f64 {
        self.inner.notional
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike
    }

    #[getter]
    fn option_expiry(&self) -> f64 {
        self.inner.option_expiry
    }

    #[getter]
    fn swap_tenor(&self) -> f64 {
        self.inner.swap_tenor
    }

    #[getter]
    fn is_payer(&self) -> bool {
        self.inner.is_payer
    }

    fn annuity_factor(&self, curve: &YieldCurve) -> f64 {
        self.to_core().annuity_factor(&curve.inner)
    }

    fn forward_swap_rate(&self, curve: &YieldCurve) -> f64 {
        self.to_core().forward_swap_rate(&curve.inner)
    }

    fn price(&self, curve: &YieldCurve, vol: f64) -> f64 {
        self.to_core().price(&curve.inner, vol)
    }

    fn implied_vol(&self, market_price: f64, curve: &YieldCurve) -> f64 {
        self.to_core().implied_vol(market_price, &curve.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Swaption(notional={}, strike={}, option_expiry={}, swap_tenor={}, is_payer={})",
            self.inner.notional,
            self.inner.strike,
            self.inner.option_expiry,
            self.inner.swap_tenor,
            self.inner.is_payer
        )
    }
}

#[pyfunction]
pub fn py_swaption_price(
    notional: f64,
    strike: f64,
    swap_tenor: f64,
    option_expiry: f64,
    vol: f64,
    discount_rate: f64,
    option_type: &str,
) -> f64 {
    let is_payer = match option_type.to_ascii_lowercase().as_str() {
        "payer" | "call" => true,
        "receiver" | "put" => false,
        _ => return f64::NAN,
    };
    let swaption = CoreSwaption {
        notional,
        strike,
        option_expiry,
        swap_tenor,
        is_payer,
    };
    let tenors: Vec<(f64, f64)> = (1..=((option_expiry + swap_tenor).ceil() as usize * 4))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-discount_rate * t).exp())
        })
        .collect();
    let curve = CoreYieldCurve::new(tenors);
    swaption.price(&curve, vol)
}

#[pyfunction]
pub fn year_fraction(start: &str, end: &str, convention: &DayCountConvention) -> PyResult<f64> {
    Ok(core_year_fraction(
        parse_date(start)?,
        parse_date(end)?,
        convention.to_core(),
    ))
}

#[pyfunction]
pub fn adjust_business_day(
    date: &str,
    convention: &BusinessDayConvention,
    calendar: &Calendar,
) -> PyResult<String> {
    Ok(format_date(core_adjust_business_day(
        parse_date(date)?,
        convention.to_core(),
        &calendar.to_core(),
    )))
}

#[pyfunction]
pub fn add_business_days(date: &str, days: i32, calendar: &Calendar) -> PyResult<String> {
    Ok(format_date(core_add_business_days(
        parse_date(date)?,
        days,
        &calendar.to_core(),
    )))
}

#[pyfunction]
pub fn subtract_business_days(date: &str, days: i32, calendar: &Calendar) -> PyResult<String> {
    Ok(format_date(core_subtract_business_days(
        parse_date(date)?,
        days,
        &calendar.to_core(),
    )))
}

#[pyfunction]
pub fn business_day_count(start: &str, end: &str, calendar: &Calendar) -> PyResult<i32> {
    Ok(core_business_day_count(
        parse_date(start)?,
        parse_date(end)?,
        &calendar.to_core(),
    ))
}

#[pyfunction]
pub fn year_fraction_business_252(start: &str, end: &str, calendar: &Calendar) -> PyResult<f64> {
    Ok(core_year_fraction_business_252(
        parse_date(start)?,
        parse_date(end)?,
        &calendar.to_core(),
    ))
}

#[pyfunction]
pub fn generate_schedule(start: &str, end: &str, freq: &Frequency) -> PyResult<Vec<String>> {
    Ok(
        core_generate_schedule(parse_date(start)?, parse_date(end)?, freq.to_core())
            .into_iter()
            .map(format_date)
            .collect(),
    )
}

#[pyfunction]
pub fn generate_schedule_with_config(
    start: &str,
    end: &str,
    freq: &Frequency,
    config: &ScheduleConfig,
) -> PyResult<Vec<String>> {
    Ok(core_generate_schedule_with_config(
        parse_date(start)?,
        parse_date(end)?,
        freq.to_core(),
        &config.to_core(),
    )
    .into_iter()
    .map(format_date)
    .collect())
}

#[pyfunction]
pub fn is_imm_date(date: &str) -> PyResult<bool> {
    Ok(core_is_imm_date(parse_date(date)?))
}

#[pyfunction]
pub fn next_imm_date(date: &str) -> PyResult<String> {
    Ok(format_date(core_next_imm_date(parse_date(date)?)))
}

#[pyfunction]
pub fn previous_imm_date(date: &str) -> PyResult<String> {
    Ok(format_date(core_previous_imm_date(parse_date(date)?)))
}

#[pyfunction]
pub fn is_cds_standard_date(date: &str) -> PyResult<bool> {
    Ok(core_is_cds_standard_date(parse_date(date)?))
}

#[pyfunction]
pub fn next_cds_date(date: &str) -> PyResult<String> {
    Ok(format_date(core_next_cds_date(parse_date(date)?)))
}

#[pyfunction]
pub fn previous_cds_date(date: &str) -> PyResult<String> {
    Ok(format_date(core_previous_cds_date(parse_date(date)?)))
}

#[pyfunction]
pub fn third_wednesday(year: i32, month: u32) -> String {
    format_date(core_third_wednesday(year, month))
}

#[pyfunction]
pub fn add_months(date: &str, months: i32) -> PyResult<String> {
    Ok(format_date(core_add_months(parse_date(date)?, months)))
}

#[pyfunction]
pub fn futures_forward_convexity_adjustment(vol: f64, t1: f64, t2: f64) -> f64 {
    core_adjustments::futures_forward_convexity_adjustment(vol, t1, t2)
}

#[pyfunction]
pub fn futures_rate_from_forward(forward_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    core_adjustments::futures_rate_from_forward(forward_rate, vol, t1, t2)
}

#[pyfunction]
pub fn forward_rate_from_futures(futures_rate: f64, vol: f64, t1: f64, t2: f64) -> f64 {
    core_adjustments::forward_rate_from_futures(futures_rate, vol, t1, t2)
}

#[pyfunction]
pub fn cms_convexity_adjustment_simple(
    swap_rate: f64,
    swap_rate_vol: f64,
    annuity_convexity: f64,
) -> f64 {
    core_adjustments::cms_convexity_adjustment(swap_rate, swap_rate_vol, annuity_convexity)
}

#[pyfunction]
pub fn cms_rate_in_arrears(swap_rate: f64, swap_rate_vol: f64, annuity_convexity: f64) -> f64 {
    core_adjustments::cms_rate_in_arrears(swap_rate, swap_rate_vol, annuity_convexity)
}

#[pyfunction]
pub fn timing_adjustment_amount(rate_vol: f64, natural_date: f64, payment_date: f64) -> f64 {
    core_adjustments::timing_adjustment_amount(rate_vol, natural_date, payment_date)
}

#[pyfunction]
pub fn timing_adjusted_rate(rate: f64, rate_vol: f64, natural_date: f64, payment_date: f64) -> f64 {
    core_adjustments::timing_adjusted_rate(rate, rate_vol, natural_date, payment_date)
}

#[pyfunction]
pub fn quanto_drift_adjustment(rho: f64, sigma_r: f64, sigma_fx: f64) -> f64 {
    core_adjustments::quanto_drift_adjustment(rho, sigma_r, sigma_fx)
}

#[pyfunction]
pub fn quanto_adjusted_drift(baseline_drift: f64, rho: f64, sigma_r: f64, sigma_fx: f64) -> f64 {
    core_adjustments::quanto_adjusted_drift(baseline_drift, rho, sigma_r, sigma_fx)
}

#[pyfunction]
pub fn cms_convexity_adjustment(params: &CmsConvexityParams) -> f64 {
    core_cms_convexity_adjustment(&params.to_core())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cms_spread_option_mc(
    option: &CmsSpreadOption,
    cms1_fwd: f64,
    cms2_fwd: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    ca1: f64,
    ca2: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> PyResult<CmsSpreadResult> {
    Ok(CmsSpreadResult::from_core(map_string_err(
        core_cms_spread_option_mc(
            &option.to_core(),
            cms1_fwd,
            cms2_fwd,
            vol1,
            vol2,
            rho,
            ca1,
            ca2,
            discount_rate,
            num_paths,
            seed,
        ),
    )?))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn sabr_cms_convexity_adjustment(
    swap_rate: f64,
    annuity: f64,
    tenor: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    core_sabr_cms_convexity_adjustment(swap_rate, annuity, tenor, expiry, alpha, beta, rho, nu)
}

#[pyfunction]
pub fn dual_curve_bootstrap(
    swap_rates: Vec<(f64, f64)>,
    ois_curve: &YieldCurve,
    frequency: usize,
) -> YieldCurve {
    YieldCurve::from_core(core_dual_curve_bootstrap(
        &swap_rates,
        &ois_curve.inner,
        frequency,
    ))
}

#[pyfunction]
pub fn price_irs_multi_curve(
    env: &MultiCurveEnvironment,
    forward_tenor: &str,
    notional: f64,
    fixed_rate: f64,
    tenor: f64,
    frequency: usize,
) -> Option<f64> {
    core_price_irs_multi_curve(
        &env.inner,
        forward_tenor,
        notional,
        fixed_rate,
        tenor,
        frequency,
    )
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_swaption_price, module)?)?;
    module.add_function(wrap_pyfunction!(year_fraction, module)?)?;
    module.add_function(wrap_pyfunction!(adjust_business_day, module)?)?;
    module.add_function(wrap_pyfunction!(add_business_days, module)?)?;
    module.add_function(wrap_pyfunction!(subtract_business_days, module)?)?;
    module.add_function(wrap_pyfunction!(business_day_count, module)?)?;
    module.add_function(wrap_pyfunction!(year_fraction_business_252, module)?)?;
    module.add_function(wrap_pyfunction!(generate_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(generate_schedule_with_config, module)?)?;
    module.add_function(wrap_pyfunction!(is_imm_date, module)?)?;
    module.add_function(wrap_pyfunction!(next_imm_date, module)?)?;
    module.add_function(wrap_pyfunction!(previous_imm_date, module)?)?;
    module.add_function(wrap_pyfunction!(is_cds_standard_date, module)?)?;
    module.add_function(wrap_pyfunction!(next_cds_date, module)?)?;
    module.add_function(wrap_pyfunction!(previous_cds_date, module)?)?;
    module.add_function(wrap_pyfunction!(third_wednesday, module)?)?;
    module.add_function(wrap_pyfunction!(add_months, module)?)?;
    module.add_function(wrap_pyfunction!(
        futures_forward_convexity_adjustment,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(futures_rate_from_forward, module)?)?;
    module.add_function(wrap_pyfunction!(forward_rate_from_futures, module)?)?;
    module.add_function(wrap_pyfunction!(cms_convexity_adjustment_simple, module)?)?;
    module.add_function(wrap_pyfunction!(cms_rate_in_arrears, module)?)?;
    module.add_function(wrap_pyfunction!(timing_adjustment_amount, module)?)?;
    module.add_function(wrap_pyfunction!(timing_adjusted_rate, module)?)?;
    module.add_function(wrap_pyfunction!(quanto_drift_adjustment, module)?)?;
    module.add_function(wrap_pyfunction!(quanto_adjusted_drift, module)?)?;
    module.add_function(wrap_pyfunction!(cms_convexity_adjustment, module)?)?;
    module.add_function(wrap_pyfunction!(cms_spread_option_mc, module)?)?;
    module.add_function(wrap_pyfunction!(sabr_cms_convexity_adjustment, module)?)?;
    module.add_function(wrap_pyfunction!(dual_curve_bootstrap, module)?)?;
    module.add_function(wrap_pyfunction!(price_irs_multi_curve, module)?)?;

    module.add_class::<DayCountConvention>()?;
    module.add_class::<Frequency>()?;
    module.add_class::<WeekendConvention>()?;
    module.add_class::<BusinessDayConvention>()?;
    module.add_class::<StubConvention>()?;
    module.add_class::<RollConvention>()?;
    module.add_class::<FinancialCenter>()?;
    module.add_class::<YieldCurveInterpolationMethod>()?;
    module.add_class::<YieldCurveInterpolationSettings>()?;
    module.add_class::<YieldCurve>()?;
    module.add_class::<YieldCurveBuilder>()?;
    module.add_class::<CustomCalendar>()?;
    module.add_class::<Calendar>()?;
    module.add_class::<ScheduleConfig>()?;
    module.add_class::<FixedRateBond>()?;
    module.add_class::<CapFloor>()?;
    module.add_class::<CmsConvexityParams>()?;
    module.add_class::<CmsSpreadOptionType>()?;
    module.add_class::<CmsSpreadOption>()?;
    module.add_class::<CmsSpreadResult>()?;
    module.add_class::<ForwardRateAgreement>()?;
    module.add_class::<InterestRateSwap>()?;
    module.add_class::<SwapBuilder>()?;
    module.add_class::<Future>()?;
    module.add_class::<InterestRateFutureQuote>()?;
    module.add_class::<InflationCurve>()?;
    module.add_class::<InflationCurveBuilder>()?;
    module.add_class::<ZeroCouponInflationSwap>()?;
    module.add_class::<YearOnYearInflationSwap>()?;
    module.add_class::<InflationIndexedBond>()?;
    module.add_class::<OvernightIndexSwap>()?;
    module.add_class::<BasisSwap>()?;
    module.add_class::<MultiCurveEnvironment>()?;
    module.add_class::<XccySwap>()?;
    module.add_class::<Swaption>()?;
    Ok(())
}
