use super::{CouponPeriod, ExerciseSchedule, VarianceOptionQuote};
use crate::core::OptionType;
use crate::helpers::{catch_unwind_py, from_python, to_python};
use crate::models::HullWhite;
pub(crate) use crate::pricing::{AbandonmentOption, DeferInvestmentOption, ExpandOption};
use crate::rates::{Frequency, YieldCurve};
use openferric_core::instruments as native;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn tagged<T: serde::de::DeserializeOwned>(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<T> {
    let data = if payload.hasattr("to_dict")? {
        payload.call_method0("to_dict")?
    } else {
        payload.clone()
    };
    let tagged = PyDict::new(payload.py());
    tagged.set_item(kind, data)?;
    from_python(tagged.as_any())
}

macro_rules! native_enum {
    ($name:ident, $core:ty, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name { pub(crate) inner: $core }

        impl $name {
            pub(crate) fn to_core(&self) -> $core { self.inner.clone() }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self { inner: tagged(kind, payload)? })
            }

            #[staticmethod]
            fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self { inner: from_python(data)? })
            }

            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { to_python(py, &self.inner) }

            #[getter]
            fn kind(&self) -> PyResult<String> {
                let data = serde_json::to_value(&self.inner).map_err(error)?;
                Ok(data.as_object().and_then(|object| object.keys().next()).cloned().unwrap_or_default())
            }

            #[getter]
            fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let data = serde_json::to_value(&self.inner).map_err(error)?;
                to_python(py, data.as_object().and_then(|object| object.values().next()).unwrap_or(&data))
            }

            $($methods)*
        }
    };
}

native_enum!(StructuredCoupon, native::StructuredCoupon, {
    #[staticmethod]
    fn range_accrual(
        in_range_coupon_rate: f64,
        out_of_range_coupon_rate: f64,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Self {
        Self {
            inner: native::StructuredCoupon::RangeAccrual {
                in_range_coupon_rate,
                out_of_range_coupon_rate,
                lower_bound,
                upper_bound,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (fixed_rate, leverage, floor=None, cap=None))]
    fn inverse_floater(
        fixed_rate: f64,
        leverage: f64,
        floor: Option<f64>,
        cap: Option<f64>,
    ) -> Self {
        Self {
            inner: native::StructuredCoupon::InverseFloater {
                fixed_rate,
                leverage,
                floor,
                cap,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (multiplier, spread, cms_tenor, swap_payment_frequency, floor=None, cap=None))]
    fn cms_linked(
        multiplier: f64,
        spread: f64,
        cms_tenor: f64,
        swap_payment_frequency: Frequency,
        floor: Option<f64>,
        cap: Option<f64>,
    ) -> Self {
        Self {
            inner: native::StructuredCoupon::CmsLinked {
                multiplier,
                spread,
                cms_tenor,
                swap_payment_frequency: swap_payment_frequency.to_core(),
                floor,
                cap,
            },
        }
    }
});

native_enum!(CouponType, native::CouponType, {
    #[staticmethod]
    fn fixed(rate: f64) -> Self {
        Self {
            inner: native::CouponType::Fixed { rate },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (spread=0.0, floor=None, cap=None))]
    fn floating(spread: f64, floor: Option<f64>, cap: Option<f64>) -> Self {
        Self {
            inner: native::CouponType::Floating { spread, floor, cap },
        }
    }

    #[staticmethod]
    fn structured(coupon: &StructuredCoupon) -> Self {
        Self {
            inner: native::CouponType::Structured(coupon.to_core()),
        }
    }
});

native_enum!(ExoticOption, native::ExoticOption, {
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(error)
    }

    #[staticmethod]
    fn lookback_floating_call(expiry: f64) -> Self {
        Self {
            inner: native::ExoticOption::lookback_floating_call(expiry),
        }
    }
    #[staticmethod]
    fn lookback_floating_put(expiry: f64) -> Self {
        Self {
            inner: native::ExoticOption::lookback_floating_put(expiry),
        }
    }
    #[staticmethod]
    fn lookback_fixed_call(strike: f64, expiry: f64) -> Self {
        Self {
            inner: native::ExoticOption::lookback_fixed_call(strike, expiry),
        }
    }
    #[staticmethod]
    fn lookback_fixed_put(strike: f64, expiry: f64) -> Self {
        Self {
            inner: native::ExoticOption::lookback_fixed_put(strike, expiry),
        }
    }
});

native_enum!(RealOptionInstrument, native::RealOptionInstrument, {
    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(error)
    }
});

macro_rules! exotic_spec {
    ($name:ident, $variant:ident, {$($field:ident: $ty:ty => $convert:expr),* $(,)?}) => {
        #[pyclass(module = "openferric", from_py_object, get_all, set_all)]
        #[derive(Clone)]
        pub struct $name { $(pub $field: $ty),* }

        impl $name {
            pub(crate) fn to_core(&self) -> native::$name { native::$name { $($field: ($convert)(self.$field.clone())),* } }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new($($field: $ty),*) -> PyResult<Self> {
                let result = Self { $($field),* };
                result.validate()?;
                Ok(result)
            }
            fn validate(&self) -> PyResult<()> { native::ExoticOption::$variant(self.to_core()).validate().map_err(error) }
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { to_python(py, &self.to_core()) }
            fn as_exotic(&self) -> ExoticOption { ExoticOption { inner: native::ExoticOption::$variant(self.to_core()) } }
        }
    }
}

exotic_spec!(LookbackFloatingOption, LookbackFloating, {option_type: OptionType => OptionType::to_core, expiry: f64 => std::convert::identity, observed_extreme: Option<f64> => std::convert::identity});
exotic_spec!(LookbackFixedOption, LookbackFixed, {option_type: OptionType => OptionType::to_core, strike: f64 => std::convert::identity, expiry: f64 => std::convert::identity, observed_extreme: Option<f64> => std::convert::identity});
exotic_spec!(ChooserOption, Chooser, {strike: f64 => std::convert::identity, expiry: f64 => std::convert::identity, choose_time: f64 => std::convert::identity});
exotic_spec!(QuantoOption, Quanto, {option_type: OptionType => OptionType::to_core, strike: f64 => std::convert::identity, expiry: f64 => std::convert::identity, fx_rate: f64 => std::convert::identity, foreign_rate: f64 => std::convert::identity, fx_vol: f64 => std::convert::identity, asset_fx_corr: f64 => std::convert::identity});
exotic_spec!(CompoundOption, Compound, {option_type: OptionType => OptionType::to_core, underlying_option_type: OptionType => OptionType::to_core, compound_strike: f64 => std::convert::identity, underlying_strike: f64 => std::convert::identity, compound_expiry: f64 => std::convert::identity, underlying_expiry: f64 => std::convert::identity});

fn schedule_to_core(
    py: Python<'_>,
    schedule: &[Py<CouponPeriod>],
) -> PyResult<Vec<native::CouponPeriod>> {
    schedule
        .iter()
        .map(|period| period.borrow(py).to_core(py))
        .collect()
}

fn schedule_from_core(
    py: Python<'_>,
    schedule: Vec<native::CouponPeriod>,
) -> PyResult<Vec<CouponPeriod>> {
    schedule
        .into_iter()
        .map(|period| CouponPeriod::from_core(py, period))
        .collect()
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CouponScheduleBuilder {
    inner: native::CouponScheduleBuilder,
}

#[pymethods]
impl CouponScheduleBuilder {
    #[new]
    fn new(start_time: f64, end_time: f64, frequency: Frequency) -> PyResult<Self> {
        Ok(Self {
            inner: native::CouponScheduleBuilder::new(start_time, end_time, frequency.to_core())
                .map_err(error)?,
        })
    }

    fn payment_lag(&self, payment_lag: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().payment_lag(payment_lag).map_err(error)?,
        })
    }

    fn build_fixed(&self, py: Python<'_>, rate: f64) -> PyResult<Vec<CouponPeriod>> {
        schedule_from_core(py, self.inner.build_fixed(rate).map_err(error)?)
    }

    #[pyo3(signature = (spread=0.0, floor=None, cap=None))]
    fn build_floating(
        &self,
        py: Python<'_>,
        spread: f64,
        floor: Option<f64>,
        cap: Option<f64>,
    ) -> PyResult<Vec<CouponPeriod>> {
        schedule_from_core(
            py,
            self.inner
                .build_floating(spread, floor, cap)
                .map_err(error)?,
        )
    }

    fn build_structured(
        &self,
        py: Python<'_>,
        structured: &StructuredCoupon,
    ) -> PyResult<Vec<CouponPeriod>> {
        schedule_from_core(
            py,
            self.inner
                .build_structured(structured.to_core())
                .map_err(error)?,
        )
    }
}

macro_rules! note {
    ($name:ident, {$($field:ident: $ty:ty => $convert:expr),* $(,)?}, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone)]
        pub struct $name { pub(crate) inner: native::$name }

        #[pymethods]
        impl $name {
            #[new]
            fn new(py: Python<'_>, $($field: $ty,)* coupon_schedule: Vec<Py<CouponPeriod>>) -> PyResult<Self> {
                let inner = native::$name { $($field: ($convert)($field),)* coupon_schedule: schedule_to_core(py, &coupon_schedule)? };
                inner.validate().map_err(error)?;
                Ok(Self { inner })
            }
            #[staticmethod]
            fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> {
                let inner: native::$name = from_python(data)?;
                inner.validate().map_err(error)?;
                Ok(Self { inner })
            }
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { to_python(py, &self.inner) }
            fn validate(&self) -> PyResult<()> { self.inner.validate().map_err(error) }
            #[getter]
            fn coupon_schedule(&self, py: Python<'_>) -> PyResult<Vec<CouponPeriod>> { schedule_from_core(py, self.inner.coupon_schedule.clone()) }
            fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
                let data = self.to_dict(py)?;
                data.bind(py).get_item(name).map(Bound::unbind).map_err(|_| pyo3::exceptions::PyAttributeError::new_err(name.to_owned()))
            }
            $($methods)*
        }
    }
}

note!(CallableRateNote, {
    notional: f64 => std::convert::identity, redemption: f64 => std::convert::identity,
    call_price: f64 => std::convert::identity, maturity: f64 => std::convert::identity,
    exercise_schedule: ExerciseSchedule => |value: ExerciseSchedule| value.to_core()
}, {
    fn price_hull_white_tree(&self, py: Python<'_>, hw_model: &HullWhite, curve: &YieldCurve, steps: usize) -> PyResult<f64> {
        let model = hw_model.to_core(); let curve = curve.to_core();
        py.detach(|| catch_unwind_py(|| self.inner.price_hull_white_tree(&model, &curve, steps))?.map_err(error))
    }
    fn hold_to_maturity_value(&self, py: Python<'_>, curve: &YieldCurve, projected_floating_rates: Vec<f64>, projected_cms_rates: Vec<f64>) -> PyResult<f64> {
        let curve = curve.to_core();
        py.detach(|| self.inner.hold_to_maturity_value(&curve, &projected_floating_rates, &projected_cms_rates).map_err(error))
    }
});

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CallableRangeAccrualNote {
    inner: native::CallableRangeAccrualNote,
}

#[pymethods]
impl CallableRangeAccrualNote {
    #[new]
    fn new(
        notional: f64,
        maturity: f64,
        frequency: Frequency,
        in_range_coupon_rate: f64,
        out_of_range_coupon_rate: f64,
        lower_bound: f64,
        upper_bound: f64,
        call_price: f64,
        exercise_schedule: ExerciseSchedule,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: native::CallableRangeAccrualNote::new(
                notional,
                maturity,
                frequency.to_core(),
                in_range_coupon_rate,
                out_of_range_coupon_rate,
                lower_bound,
                upper_bound,
                call_price,
                exercise_schedule.to_core(),
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn note(&self) -> CallableRateNote {
        CallableRateNote {
            inner: self.inner.note.clone(),
        }
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }
    fn price_hull_white_tree(
        &self,
        py: Python<'_>,
        hw_model: &HullWhite,
        curve: &YieldCurve,
        steps: usize,
    ) -> PyResult<f64> {
        let model = hw_model.to_core();
        let curve = curve.to_core();
        py.detach(|| {
            catch_unwind_py(|| self.inner.price_hull_white_tree(&model, &curve, steps))?
                .map_err(error)
        })
    }
}

#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone)]
pub struct TarnPricingResult {
    price: f64,
    accrued_coupon: f64,
    knocked_out: bool,
    knockout_time: Option<f64>,
}

#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone)]
pub struct SnowballPricingResult {
    price: f64,
    coupon_path: Vec<f64>,
}

note!(TargetRedemptionNote, {
    notional: f64 => std::convert::identity, redemption: f64 => std::convert::identity,
    target_coupon: f64 => std::convert::identity, spread: f64 => std::convert::identity,
    floor: Option<f64> => std::convert::identity, cap: Option<f64> => std::convert::identity
}, {
    fn price(&self, py: Python<'_>, projected_floating_rates: Vec<f64>, curve: &YieldCurve) -> PyResult<TarnPricingResult> {
        let curve = curve.to_core();
        let result = py.detach(|| self.inner.price(&projected_floating_rates, &curve)).map_err(error)?;
        Ok(TarnPricingResult { price: result.price, accrued_coupon: result.accrued_coupon, knocked_out: result.knocked_out, knockout_time: result.knockout_time })
    }
});

note!(SnowballNote, {
    notional: f64 => std::convert::identity, redemption: f64 => std::convert::identity,
    initial_coupon: f64 => std::convert::identity, spread: f64 => std::convert::identity,
    floor: Option<f64> => std::convert::identity, cap: Option<f64> => std::convert::identity
}, {
    fn price(&self, py: Python<'_>, projected_floating_rates: Vec<f64>, curve: &YieldCurve) -> PyResult<SnowballPricingResult> {
        let curve = curve.to_core();
        let result = py.detach(|| self.inner.price(&projected_floating_rates, &curve)).map_err(error)?;
        Ok(SnowballPricingResult { price: result.price, coupon_path: result.coupon_path })
    }
});

note!(InverseFloaterNote, {
    notional: f64 => std::convert::identity, redemption: f64 => std::convert::identity,
    fixed_rate: f64 => std::convert::identity, leverage: f64 => std::convert::identity,
    floor: Option<f64> => std::convert::identity, cap: Option<f64> => std::convert::identity
}, {
    fn price(&self, py: Python<'_>, projected_floating_rates: Vec<f64>, curve: &YieldCurve) -> PyResult<f64> {
        let curve = curve.to_core();
        py.detach(|| self.inner.price(&projected_floating_rates, &curve)).map_err(error)
    }
});

note!(CmsLinkedNote, {
    notional: f64 => std::convert::identity, redemption: f64 => std::convert::identity,
    multiplier: f64 => std::convert::identity, spread: f64 => std::convert::identity,
    cms_tenor: f64 => std::convert::identity, swap_payment_frequency: Frequency => Frequency::to_core,
    floor: Option<f64> => std::convert::identity, cap: Option<f64> => std::convert::identity
}, {
    fn price_with_projected_cms_rates(&self, py: Python<'_>, projected_cms_rates: Vec<f64>, curve: &YieldCurve) -> PyResult<f64> {
        let curve = curve.to_core();
        py.detach(|| self.inner.price_with_projected_cms_rates(&projected_cms_rates, &curve)).map_err(error)
    }
    fn price_from_curve(&self, py: Python<'_>, curve: &YieldCurve) -> PyResult<f64> {
        let curve = curve.to_core(); py.detach(|| self.inner.price_from_curve(&curve)).map_err(error)
    }
    fn projected_cms_rates_from_curve(&self, py: Python<'_>, curve: &YieldCurve) -> PyResult<Vec<f64>> {
        let curve = curve.to_core(); py.detach(|| self.inner.projected_cms_rates_from_curve(&curve)).map_err(error)
    }
});

macro_rules! variance_contract {
    ($name:ident, {$($extra:ident: $ty:ty),*}) => {
        #[pyclass(module = "openferric", from_py_object, get_all, set_all)]
        #[derive(Clone)]
        pub struct $name {
            pub notional_vega: f64, pub strike_vol: f64, pub expiry: f64,
            pub observed_realized_var: Option<f64>, pub option_quotes: Vec<VarianceOptionQuote>,
            $(pub $extra: $ty),*
        }
        impl $name {
            pub(crate) fn to_core(&self) -> native::$name {
                native::$name { notional_vega: self.notional_vega, strike_vol: self.strike_vol, expiry: self.expiry,
                    observed_realized_var: self.observed_realized_var, option_quotes: self.option_quotes.iter().copied().map(VarianceOptionQuote::to_core).collect(), $($extra: self.$extra),* }
            }
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (notional_vega, strike_vol, expiry, option_quotes, $($extra,)* observed_realized_var=None))]
            fn new(notional_vega: f64, strike_vol: f64, expiry: f64, option_quotes: Vec<VarianceOptionQuote>, $($extra: $ty,)* observed_realized_var: Option<f64>) -> PyResult<Self> {
                let result = Self { notional_vega, strike_vol, expiry, option_quotes, observed_realized_var, $($extra),* }; result.validate()?; Ok(result)
            }
            fn validate(&self) -> PyResult<()> { self.to_core().validate().map_err(error) }
            fn with_observed_realized_var(&self, observed_realized_var: f64) -> PyResult<Self> {
                let mut result = self.clone(); result.observed_realized_var = Some(observed_realized_var); result.validate()?; Ok(result)
            }
            fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { to_python(py, &self.to_core()) }
        }
    }
}
variance_contract!(VarianceSwap, {});
variance_contract!(VolatilitySwap, {var_of_var: f64});

pub(crate) fn trade_instrument(value: &Bound<'_, PyAny>) -> PyResult<native::TradeInstrument> {
    if let Ok(value) = value.extract::<PyRef<TradeInstrument>>() {
        return Ok(value.inner.clone());
    }
    macro_rules! fallible { ($($name:ident),+) => { $(if let Ok(value) = value.extract::<PyRef<super::$name>>() { return Ok(native::TradeInstrument::$name(value.to_core()?)); })+ } }
    macro_rules! infallible { ($($name:ident),+) => { $(if let Ok(value) = value.extract::<PyRef<super::$name>>() { return Ok(native::TradeInstrument::$name(value.to_core())); })+ } }
    fallible!(
        VanillaOption,
        AsianOption,
        BarrierOption,
        BermudanOption,
        FuturesOption,
        ForwardStartOption,
        CommodityOption,
        CommoditySpreadOption,
        CashOrNothingOption,
        AssetOrNothingOption,
        GapOption,
        DoubleBarrierOption,
        EmployeeStockOption,
        FxOption,
        PowerOption,
        TwoAssetCorrelationOption,
        WeatherSwap,
        WeatherOption,
        BasketOption,
        OutperformanceBasketOption,
        QuantoBasketOption,
        Tarf
    );
    infallible!(
        Autocallable,
        PhoenixAutocallable,
        CommodityForward,
        CommodityFutures,
        ConvertibleBond,
        BestOfTwoCallOption,
        WorstOfTwoCallOption,
        SpreadOption,
        SwingOption,
        RangeAccrual,
        DualRangeAccrual,
        MbsPassThrough,
        CatastropheBond,
        DeferInvestmentOption,
        ExpandOption,
        AbandonmentOption,
        ExoticOption,
        LookbackFloatingOption,
        LookbackFixedOption,
        ChooserOption,
        QuantoOption,
        CompoundOption,
        VarianceSwap,
        VolatilitySwap,
        RealOptionInstrument
    );
    if let Ok(value) = value.extract::<PyRef<super::CliquetOption>>() {
        return Ok(native::TradeInstrument::CliquetOption(value.inner));
    }
    if let Ok(value) = value.extract::<PyRef<crate::funding::FundingRateSwap>>() {
        return Ok(native::TradeInstrument::FundingRateSwap(value.to_core()?));
    }
    if let Ok(value) = value.extract::<PyRef<crate::dsl::DslProduct>>() {
        return Ok(native::TradeInstrument::DslProduct(value.inner.clone()));
    }
    macro_rules! notes { ($($name:ident),+) => { $(if let Ok(value) = value.extract::<PyRef<$name>>() { return Ok(native::TradeInstrument::$name(value.inner.clone())); })+ } }
    notes!(
        CallableRateNote,
        CallableRangeAccrualNote,
        TargetRedemptionNote,
        SnowballNote,
        InverseFloaterNote,
        CmsLinkedNote
    );
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a native instrument or TradeInstrument",
    ))
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct TradeInstrument {
    pub(crate) inner: native::TradeInstrument,
}
#[pymethods]
impl TradeInstrument {
    #[new]
    fn new(kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !payload.is_instance_of::<PyDict>() {
            let result = Self {
                inner: trade_instrument(payload)?,
            };
            if result.kind()? != kind {
                return Err(error("instrument kind does not match payload type"));
            }
            return Ok(result);
        }
        let data = PyDict::new(payload.py());
        data.set_item("type", kind)?;
        data.set_item("data", payload)?;
        Ok(Self {
            inner: from_python(data.as_any())?,
        })
    }
    #[staticmethod]
    fn from_instrument(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: trade_instrument(value)?,
        })
    }
    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: from_python(data)?,
        })
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }
    #[getter]
    fn kind(&self) -> PyResult<String> {
        Ok(serde_json::to_value(&self.inner).map_err(error)?["type"]
            .as_str()
            .unwrap_or_default()
            .to_owned())
    }
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(
            py,
            &serde_json::to_value(&self.inner).map_err(error)?["data"],
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct InstrumentPortfolio {
    inner: native::Portfolio,
}
#[pymethods]
impl InstrumentPortfolio {
    #[new]
    #[pyo3(signature = (portfolio_id, trades, market_snapshot_id=None))]
    fn new(
        py: Python<'_>,
        portfolio_id: String,
        trades: Vec<Py<super::Trade>>,
        market_snapshot_id: Option<String>,
    ) -> PyResult<Self> {
        let trades = trades
            .iter()
            .map(|trade| trade.borrow(py).to_core(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: native::Portfolio {
                portfolio_id,
                trades,
                market_snapshot_id,
            },
        })
    }
    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: from_python(data)?,
        })
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }
    #[getter]
    fn portfolio_id(&self) -> &str {
        &self.inner.portfolio_id
    }
    #[getter]
    fn market_snapshot_id(&self) -> Option<String> {
        self.inner.market_snapshot_id.clone()
    }
    #[getter]
    fn trades(&self, py: Python<'_>) -> PyResult<Vec<super::Trade>> {
        self.inner
            .trades
            .iter()
            .cloned()
            .map(|trade| super::Trade::from_core(py, trade))
            .collect()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InstrumentPortfolio>()?;
    module.add_class::<LookbackFloatingOption>()?;
    module.add_class::<LookbackFixedOption>()?;
    module.add_class::<ChooserOption>()?;
    module.add_class::<QuantoOption>()?;
    module.add_class::<CompoundOption>()?;
    module.add_class::<CouponScheduleBuilder>()?;
    module.add_class::<CallableRateNote>()?;
    module.add_class::<CallableRangeAccrualNote>()?;
    module.add_class::<TargetRedemptionNote>()?;
    module.add_class::<TarnPricingResult>()?;
    module.add_class::<SnowballNote>()?;
    module.add_class::<SnowballPricingResult>()?;
    module.add_class::<InverseFloaterNote>()?;
    module.add_class::<CmsLinkedNote>()?;
    module.add_class::<VarianceSwap>()?;
    module.add_class::<VolatilitySwap>()?;
    module.add_class::<DeferInvestmentOption>()?;
    module.add_class::<ExpandOption>()?;
    module.add_class::<AbandonmentOption>()?;
    Ok(())
}
