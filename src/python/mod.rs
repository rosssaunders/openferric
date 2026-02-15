use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{
    BarrierDirection, BarrierSpec, BarrierStyle, ExerciseStyle, OptionType, PricingEngine,
    PricingError, PricingResult,
};
use crate::engines::analytic::{BarrierAnalyticEngine, BlackScholesEngine};
use crate::engines::numerical::AmericanBinomialEngine;
use crate::instruments::{BarrierOption, VanillaOption};
use crate::market::{Market, VolSource};

#[pyclass(module = "openferric")]
#[derive(Debug, Clone, Default)]
pub struct PyMarket {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    flat_vol: Option<f64>,
    reference_date: Option<String>,
}

#[pymethods]
impl PyMarket {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn spot<'py>(mut slf: PyRefMut<'py, Self>, spot: f64) -> PyRefMut<'py, Self> {
        slf.spot = Some(spot);
        slf
    }

    fn rate<'py>(mut slf: PyRefMut<'py, Self>, rate: f64) -> PyRefMut<'py, Self> {
        slf.rate = Some(rate);
        slf
    }

    fn dividend_yield<'py>(
        mut slf: PyRefMut<'py, Self>,
        dividend_yield: f64,
    ) -> PyRefMut<'py, Self> {
        slf.dividend_yield = Some(dividend_yield);
        slf
    }

    fn flat_vol<'py>(mut slf: PyRefMut<'py, Self>, vol: f64) -> PyRefMut<'py, Self> {
        slf.flat_vol = Some(vol);
        slf
    }

    fn reference_date<'py>(
        mut slf: PyRefMut<'py, Self>,
        reference_date: String,
    ) -> PyRefMut<'py, Self> {
        slf.reference_date = Some(reference_date);
        slf
    }

    fn build(&self, py: Python<'_>) -> PyResult<PyObject> {
        let market = self.to_market().map_err(pricing_error_to_py)?;
        market_to_dict(py, &market)
    }
}

impl PyMarket {
    fn to_market(&self) -> Result<Market, PricingError> {
        let mut builder = Market::builder();

        if let Some(spot) = self.spot {
            builder = builder.spot(spot);
        }
        if let Some(rate) = self.rate {
            builder = builder.rate(rate);
        }
        if let Some(dividend_yield) = self.dividend_yield {
            builder = builder.dividend_yield(dividend_yield);
        }
        if let Some(flat_vol) = self.flat_vol {
            builder = builder.flat_vol(flat_vol);
        }
        if let Some(reference_date) = &self.reference_date {
            builder = builder.reference_date(reference_date.clone());
        }

        builder.build()
    }
}

fn pricing_error_to_py(err: PricingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_option_type(value: &str) -> PyResult<OptionType> {
    match value.to_ascii_lowercase().as_str() {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        _ => Err(PyValueError::new_err("option_type must be 'call' or 'put'")),
    }
}

fn parse_barrier_style(value: &str) -> PyResult<BarrierStyle> {
    match value.to_ascii_lowercase().as_str() {
        "in" => Ok(BarrierStyle::In),
        "out" => Ok(BarrierStyle::Out),
        _ => Err(PyValueError::new_err("barrier_type must be 'in' or 'out'")),
    }
}

fn parse_barrier_direction(value: &str) -> PyResult<BarrierDirection> {
    match value.to_ascii_lowercase().as_str() {
        "up" => Ok(BarrierDirection::Up),
        "down" => Ok(BarrierDirection::Down),
        _ => Err(PyValueError::new_err("barrier_dir must be 'up' or 'down'")),
    }
}

fn market_to_dict(py: Python<'_>, market: &Market) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("spot", market.spot)?;
    dict.set_item("rate", market.rate)?;
    dict.set_item("dividend_yield", market.dividend_yield)?;
    dict.set_item("reference_date", market.reference_date.clone())?;

    match &market.vol {
        VolSource::Flat(vol) => dict.set_item("flat_vol", *vol)?,
        VolSource::Surface(_) => {
            return Err(PyTypeError::new_err(
                "only flat volatility is supported by Python bindings",
            ));
        }
    }

    Ok(dict.into())
}

fn pricing_result_to_dict(py: Python<'_>, result: PricingResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("price", result.price)?;
    dict.set_item("stderr", result.stderr)?;

    if let Some(greeks) = result.greeks {
        let greeks_dict = PyDict::new(py);
        greeks_dict.set_item("delta", greeks.delta)?;
        greeks_dict.set_item("gamma", greeks.gamma)?;
        greeks_dict.set_item("vega", greeks.vega)?;
        greeks_dict.set_item("theta", greeks.theta)?;
        greeks_dict.set_item("rho", greeks.rho)?;
        dict.set_item("greeks", greeks_dict)?;
    } else {
        dict.set_item("greeks", py.None())?;
    }

    let diagnostics_dict = PyDict::new(py);
    for (key, value) in result.diagnostics {
        diagnostics_dict.set_item(key, value)?;
    }
    dict.set_item("diagnostics", diagnostics_dict)?;

    Ok(dict.into())
}

#[pyfunction]
pub fn price_european(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
) -> PyResult<PyObject> {
    let option_type = parse_option_type(option_type)?;

    let instrument = VanillaOption {
        option_type,
        strike,
        expiry,
        exercise: ExerciseStyle::European,
    };
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div_yield)
        .flat_vol(vol)
        .build()
        .map_err(pricing_error_to_py)?;

    let engine = BlackScholesEngine::new();
    let result = engine
        .price(&instrument, &market)
        .map_err(pricing_error_to_py)?;
    pricing_result_to_dict(py, result)
}

#[pyfunction(signature = (spot, strike, expiry, vol, rate, div_yield, option_type, steps = 500))]
pub fn price_american(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    steps: usize,
) -> PyResult<PyObject> {
    let option_type = parse_option_type(option_type)?;

    let instrument = VanillaOption {
        option_type,
        strike,
        expiry,
        exercise: ExerciseStyle::American,
    };
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div_yield)
        .flat_vol(vol)
        .build()
        .map_err(pricing_error_to_py)?;

    let engine = AmericanBinomialEngine::new(steps);
    let result = engine
        .price(&instrument, &market)
        .map_err(pricing_error_to_py)?;
    pricing_result_to_dict(py, result)
}

#[pyfunction]
pub fn price_barrier(
    py: Python<'_>,
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
) -> PyResult<PyObject> {
    let option_type = parse_option_type(option_type)?;
    let style = parse_barrier_style(barrier_type)?;
    let direction = parse_barrier_direction(barrier_dir)?;

    let instrument = BarrierOption {
        option_type,
        strike,
        expiry,
        barrier: BarrierSpec {
            direction,
            style,
            level: barrier,
            rebate: 0.0,
        },
    };
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div_yield)
        .flat_vol(vol)
        .build()
        .map_err(pricing_error_to_py)?;

    let engine = BarrierAnalyticEngine::new();
    let result = engine
        .price(&instrument, &market)
        .map_err(pricing_error_to_py)?;
    pricing_result_to_dict(py, result)
}

#[pymodule]
pub fn openferric(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyMarket>()?;
    module.add_function(wrap_pyfunction!(price_european, module)?)?;
    module.add_function(wrap_pyfunction!(price_american, module)?)?;
    module.add_function(wrap_pyfunction!(price_barrier, module)?)?;
    Ok(())
}
