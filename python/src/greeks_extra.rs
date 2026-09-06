use crate::core::OptionType;
use crate::helpers::catch_unwind_py;
use numpy::PyArrayMethods;
use openferric_core::greeks::bsm as native;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

macro_rules! greek_result {
    ($name:ident, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object, get_all)]
        #[derive(Clone, Copy)]
        pub struct $name { delta: f64, gamma: f64, vega: f64, theta: f64, rho: f64, vanna: f64, volga: f64 }
        impl From<native::$name> for $name {
            fn from(value: native::$name) -> Self { Self { delta: value.delta, gamma: value.gamma, vega: value.vega, theta: value.theta, rho: value.rho, vanna: value.vanna, volga: value.volga } }
        }
        #[pymethods]
        impl $name { $($methods)* }
    }
}
greek_result!(EuropeanBsmGreeks, {
    fn vomma(&self) -> f64 {
        self.volga
    }
});
greek_result!(FiniteDifferenceGreeks, {});

#[pyclass(module = "openferric", from_py_object, get_all, set_all)]
#[derive(Clone, Copy)]
pub struct FxGreeks {
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho_domestic: f64,
    rho_foreign: f64,
}
#[pymethods]
impl FxGreeks {
    #[new]
    fn new(
        delta: f64,
        gamma: f64,
        vega: f64,
        theta: f64,
        rho_domestic: f64,
        rho_foreign: f64,
    ) -> Self {
        Self {
            delta,
            gamma,
            vega,
            theta,
            rho_domestic,
            rho_foreign,
        }
    }
}

#[pyfunction]
fn black_scholes_merton_greeks(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> EuropeanBsmGreeks {
    native::black_scholes_merton_greeks(
        option_type.to_core(),
        spot,
        strike,
        rate,
        dividend_yield,
        vol,
        expiry,
    )
    .into()
}

fn checked_callback(
    function: &Bound<'_, PyAny>,
    arguments: (f64, f64, f64, f64, f64),
    failure: &RefCell<Option<PyErr>>,
) -> f64 {
    if failure.borrow().is_some() {
        return f64::NAN;
    }
    let result = function
        .call1(arguments)
        .and_then(|result| result.extract::<f64>())
        .and_then(|result| {
            if result.is_finite() {
                Ok(result)
            } else {
                Err(PyValueError::new_err("pricer must return a finite value"))
            }
        });
    match result {
        Ok(value) => value,
        Err(error) => {
            *failure.borrow_mut() = Some(error);
            f64::NAN
        }
    }
}

macro_rules! finite_difference {
    ($name:ident, $output:ty) => {
        #[pyfunction]
        fn $name(
            pricer: &Bound<'_, PyAny>,
            spot: f64,
            strike: f64,
            rate: f64,
            vol: f64,
            expiry: f64,
            bump_spot: f64,
            bump_rate: f64,
            bump_vol: f64,
            bump_time: f64,
        ) -> PyResult<$output> {
            let failure = RefCell::new(None);
            let callback = |spot, strike, rate, vol, expiry| {
                checked_callback(pricer, (spot, strike, rate, vol, expiry), &failure)
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                native::$name(
                    &callback, spot, strike, rate, vol, expiry, bump_spot, bump_rate, bump_vol,
                    bump_time,
                )
            }))
            .map_err(crate::helpers::panic_to_pyerr);
            if let Some(error) = failure.into_inner() {
                return Err(error);
            }
            result.map(Into::into)
        }
    };
}
finite_difference!(bump_and_reprice, (f64, f64, f64, f64, f64, f64));
finite_difference!(finite_difference_greeks, FiniteDifferenceGreeks);

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BatchSimdBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    WasmSimd128,
}
#[pyfunction]
fn detected_batch_simd_backend() -> BatchSimdBackend {
    use openferric_core::engines::analytic::BatchSimdBackend as Backend;
    match openferric_core::engines::analytic::detected_batch_simd_backend() {
        Backend::Scalar => BatchSimdBackend::Scalar,
        Backend::Avx2 => BatchSimdBackend::Avx2,
        Backend::Avx512 => BatchSimdBackend::Avx512,
        Backend::Neon => BatchSimdBackend::Neon,
        Backend::WasmSimd128 => BatchSimdBackend::WasmSimd128,
    }
}
#[pyfunction]
fn bs_greeks_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::bs_greeks_batch(
                &spots,
                &strikes,
                rate,
                dividend_yield,
                vol,
                expiry,
                is_call,
            )
        })
    })
}
#[pyfunction]
fn normal_cdf_batch_approx(py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
    py.detach(|| {
        catch_unwind_py(|| openferric_core::engines::analytic::normal_cdf_batch_approx(&values))
    })
}

#[pyfunction]
fn bs_price_batch_into(
    spots: Vec<f64>,
    strikes: Vec<f64>,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    output: &Bound<'_, numpy::PyArray1<f64>>,
) -> PyResult<()> {
    let mut output = output
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let output = output
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("output must be a contiguous writable float64 array"))?;
    catch_unwind_py(std::panic::AssertUnwindSafe(|| {
        openferric_core::engines::analytic::bs_price_batch_into(
            &spots,
            &strikes,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
            output,
        )
    }))
}
#[pyfunction]
fn normal_cdf_batch_approx_into(
    values: Vec<f64>,
    output: &Bound<'_, numpy::PyArray1<f64>>,
) -> PyResult<()> {
    let mut output = output
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let output = output
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("output must be a contiguous writable float64 array"))?;
    catch_unwind_py(std::panic::AssertUnwindSafe(|| {
        openferric_core::engines::analytic::normal_cdf_batch_approx_into(&values, output)
    }))
}
#[pyfunction]
fn bs_greeks_batch_into(
    spots: Vec<f64>,
    strikes: Vec<f64>,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    delta: &Bound<'_, numpy::PyArray1<f64>>,
    gamma: &Bound<'_, numpy::PyArray1<f64>>,
    vega: &Bound<'_, numpy::PyArray1<f64>>,
    theta: &Bound<'_, numpy::PyArray1<f64>>,
) -> PyResult<()> {
    let mut delta = delta
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut gamma = gamma
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut vega = vega
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut theta = theta
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let delta = delta
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("delta must be contiguous"))?;
    let gamma = gamma
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("gamma must be contiguous"))?;
    let vega = vega
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("vega must be contiguous"))?;
    let theta = theta
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("theta must be contiguous"))?;
    catch_unwind_py(std::panic::AssertUnwindSafe(|| {
        openferric_core::engines::analytic::bs_greeks_batch_into(
            &spots,
            &strikes,
            rate,
            dividend_yield,
            vol,
            expiry,
            is_call,
            delta,
            gamma,
            vega,
            theta,
        )
    }))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EuropeanBsmGreeks>()?;
    module.add_class::<FiniteDifferenceGreeks>()?;
    module.add_class::<BatchSimdBackend>()?;
    module.add_class::<FxGreeks>()?;
    macro_rules! functions { ($($name:ident),+) => { $(module.add_function(wrap_pyfunction!($name, module)?)?;)+ } }
    functions!(
        black_scholes_merton_greeks,
        bump_and_reprice,
        finite_difference_greeks,
        detected_batch_simd_backend,
        bs_greeks_batch,
        normal_cdf_batch_approx,
        bs_price_batch_into,
        normal_cdf_batch_approx_into,
        bs_greeks_batch_into
    );
    Ok(())
}
