use crate::helpers::catch_unwind_py;
use native::CharacteristicFunction as _;
use num_complex::Complex64;
use openferric_core::engines::fft as native;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyclass(module = "openferric", from_py_object, get_all, set_all)]
#[derive(Clone, Copy)]
pub struct CarrMadanParams {
    pub n: usize,
    pub eta: f64,
    pub alpha: f64,
}
impl CarrMadanParams {
    pub(crate) fn to_core(self) -> native::CarrMadanParams {
        native::CarrMadanParams {
            n: self.n,
            eta: self.eta,
            alpha: self.alpha,
        }
    }
    pub(crate) fn from_core(value: native::CarrMadanParams) -> Self {
        Self {
            n: value.n,
            eta: value.eta,
            alpha: value.alpha,
        }
    }
}
#[pymethods]
impl CarrMadanParams {
    #[new]
    #[pyo3(signature = (n=native::DEFAULT_FFT_N, eta=native::DEFAULT_ETA, alpha=native::DEFAULT_ALPHA))]
    fn new(n: usize, eta: f64, alpha: f64) -> Self {
        Self { n, eta, alpha }
    }
    #[staticmethod]
    fn high_resolution() -> Self {
        Self::from_core(native::CarrMadanParams::high_resolution())
    }
    fn lambda_spacing(&self) -> f64 {
        self.to_core().lambda()
    }
}

macro_rules! char_function {
    ($name:ident, [$($parameter:ident),+], {$($field:ident),+}, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object)]
        #[derive(Clone, Copy)]
        pub struct $name { pub(crate) inner: native::$name }
        #[pymethods]
        impl $name {
            #[new]
            fn new($($parameter: f64),+) -> Self { Self { inner: native::$name::new($($parameter),+) } }
            $(#[getter]
            fn $field(&self) -> f64 { self.inner.$field })+
            fn cf(&self, argument: Complex64) -> Complex64 { self.inner.cf(argument) }
            fn __call__(&self, argument: Complex64) -> Complex64 { self.inner.cf(argument) }
            fn moment_exists(&self, order: f64) -> bool { self.inner.moment_exists(order) }
            fn dcf_dlog_spot(&self, argument: Complex64) -> Option<Complex64> { self.inner.dcf_dlog_spot(argument) }
            fn d2cf_dlog_spot2(&self, argument: Complex64) -> Option<Complex64> { self.inner.d2cf_dlog_spot2(argument) }
            fn dcf_dvol(&self, argument: Complex64) -> Option<Complex64> { self.inner.dcf_dvol(argument) }
            $($methods)*
        }
    }
}
char_function!(BlackScholesCharFn, [spot, rate, dividend_yield, vol, maturity], {ln_spot, rate, dividend_yield, vol, maturity}, {});
char_function!(HestonCharFn, [spot, rate, dividend_yield, maturity, v0, kappa, theta, sigma_v, rho], {ln_spot, rate, dividend_yield, maturity, v0, kappa, theta, sigma_v, rho}, {});

macro_rules! levy_char_function {
    ($name:ident, [$($parameter:ident),+]) => {
        char_function!($name, [spot, drift, maturity, $($parameter),+], {ln_spot, drift, maturity, $($parameter),+}, {
            #[staticmethod]
            fn risk_neutral(spot: f64, rate: f64, dividend_yield: f64, maturity: f64, $($parameter: f64),+) -> PyResult<Self> {
                Ok(Self { inner: native::$name::risk_neutral(spot, rate, dividend_yield, maturity, $($parameter),+).map_err(error)? })
            }
        });
    }
}
levy_char_function!(VarianceGammaCharFn, [sigma, theta, nu]);
levy_char_function!(CgmyCharFn, [c, g, m, y]);
levy_char_function!(NigCharFn, [alpha, beta, delta]);

struct PythonCf {
    callable: Py<PyAny>,
    failure: Arc<Mutex<Option<PyErr>>>,
}
impl PythonCf {
    fn record<T>(&self, result: PyResult<T>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                let mut failure = self
                    .failure
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if failure.is_none() {
                    *failure = Some(error);
                }
                None
            }
        }
    }
    fn derivative(&self, name: &str, argument: Complex64) -> Option<Complex64> {
        self.record(Python::attach(|py| {
            let callable = self.callable.bind(py);
            if !callable.hasattr(name)? {
                return Ok(None);
            }
            let result: Option<Complex64> = callable.call_method1(name, (argument,))?.extract()?;
            if result.is_some_and(|value| !value.is_finite()) {
                return Err(error("characteristic-function derivative must be finite"));
            }
            Ok(result)
        }))
        .flatten()
    }
}
struct DefaultMoment<'a>(&'a PythonCf);
impl native::CharacteristicFunction for DefaultMoment<'_> {
    fn cf(&self, argument: Complex64) -> Complex64 {
        self.0.cf(argument)
    }
}
impl native::CharacteristicFunction for PythonCf {
    fn cf(&self, argument: Complex64) -> Complex64 {
        if self
            .failure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
        {
            return Complex64::new(f64::NAN, f64::NAN);
        }
        self.record(Python::attach(|py| {
            let callable = self.callable.bind(py);
            let value = if callable.hasattr("cf")? {
                callable.call_method1("cf", (argument,))?
            } else {
                callable.call1((argument,))?
            };
            let value: Complex64 = value.extract()?;
            if !value.is_finite() {
                return Err(error(
                    "characteristic function must return a finite complex number",
                ));
            }
            Ok(value)
        }))
        .unwrap_or(Complex64::new(f64::NAN, f64::NAN))
    }
    fn moment_exists(&self, order: f64) -> bool {
        self.record(Python::attach(|py| {
            let callable = self.callable.bind(py);
            if callable.hasattr("moment_exists")? {
                callable.call_method1("moment_exists", (order,))?.extract()
            } else {
                Ok(DefaultMoment(self).moment_exists(order))
            }
        }))
        .unwrap_or(false)
    }
    fn dcf_dlog_spot(&self, argument: Complex64) -> Option<Complex64> {
        self.derivative("dcf_dlog_spot", argument)
    }
    fn d2cf_dlog_spot2(&self, argument: Complex64) -> Option<Complex64> {
        self.derivative("d2cf_dlog_spot2", argument)
    }
    fn dcf_dvol(&self, argument: Complex64) -> Option<Complex64> {
        self.derivative("dcf_dvol", argument)
    }
}

struct SelectedCf {
    inner: Box<dyn native::CharacteristicFunction + Send + Sync>,
    failure: Arc<Mutex<Option<PyErr>>>,
}
impl SelectedCf {
    fn extract(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let failure = Arc::new(Mutex::new(None));
        macro_rules! select { ($($name:ty),+) => { $(if let Ok(value) = value.extract::<PyRef<$name>>() { return Ok(Self { inner: Box::new(value.inner), failure }); })+ } }
        select!(
            BlackScholesCharFn,
            HestonCharFn,
            VarianceGammaCharFn,
            CgmyCharFn,
            NigCharFn
        );
        if !value.is_callable() && !value.hasattr("cf")? {
            return Err(PyTypeError::new_err(
                "expected a characteristic function or callable",
            ));
        }
        Ok(Self {
            inner: Box::new(PythonCf {
                callable: value.clone().unbind(),
                failure: Arc::clone(&failure),
            }),
            failure,
        })
    }
    fn finish<T>(&self, result: Result<T, String>) -> PyResult<T> {
        if let Some(error) = self
            .failure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            return Err(error);
        }
        result.map_err(error)
    }
}
impl native::CharacteristicFunction for SelectedCf {
    fn cf(&self, argument: Complex64) -> Complex64 {
        self.inner.cf(argument)
    }
    fn moment_exists(&self, order: f64) -> bool {
        self.inner.moment_exists(order)
    }
    fn dcf_dlog_spot(&self, argument: Complex64) -> Option<Complex64> {
        self.inner.dcf_dlog_spot(argument)
    }
    fn d2cf_dlog_spot2(&self, argument: Complex64) -> Option<Complex64> {
        self.inner.d2cf_dlog_spot2(argument)
    }
    fn dcf_dvol(&self, argument: Complex64) -> Option<Complex64> {
        self.inner.dcf_dvol(argument)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CarrMadanContext {
    inner: native::CarrMadanContext,
}
#[pymethods]
impl CarrMadanContext {
    #[new]
    #[pyo3(signature = (cf, rate, maturity, spot, params=None))]
    fn new(
        py: Python<'_>,
        cf: &Bound<'_, PyAny>,
        rate: f64,
        maturity: f64,
        spot: f64,
        params: Option<CarrMadanParams>,
    ) -> PyResult<Self> {
        let cf = SelectedCf::extract(cf)?;
        let params = params.map(CarrMadanParams::to_core).unwrap_or_default();
        let result = py.detach(|| native::CarrMadanContext::new(&cf, rate, maturity, spot, params));
        Ok(Self {
            inner: cf.finish(result)?,
        })
    }
    fn rate(&self) -> f64 {
        self.inner.rate()
    }
    fn maturity(&self) -> f64 {
        self.inner.maturity()
    }
    fn spot(&self) -> f64 {
        self.inner.spot()
    }
    fn params(&self) -> CarrMadanParams {
        CarrMadanParams::from_core(self.inner.params())
    }
    fn weighted_samples(&self) -> Vec<Complex64> {
        self.inner.weighted_samples().to_vec()
    }
    fn price_strikes(&self, py: Python<'_>, strikes: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
        py.detach(|| self.inner.price_strikes(&strikes))
            .map_err(error)
    }
    fn price_grid(&self, py: Python<'_>) -> PyResult<Vec<(f64, f64)>> {
        py.detach(|| self.inner.price_grid()).map_err(error)
    }
    fn price_grid_complex(&self, py: Python<'_>) -> PyResult<Vec<(f64, f64)>> {
        py.detach(|| self.inner.price_grid_complex()).map_err(error)
    }
}

#[pyclass(module = "openferric", from_py_object, get_all)]
#[derive(Clone, Copy)]
pub struct CarrMadanGreeksPoint {
    strike: f64,
    call: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
}
impl From<native::CarrMadanGreeksPoint> for CarrMadanGreeksPoint {
    fn from(value: native::CarrMadanGreeksPoint) -> Self {
        Self {
            strike: value.strike,
            call: value.call,
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
        }
    }
}

macro_rules! transform_function {
    ($name:ident, [$($parameter:ident: $ty:ty => $argument:expr),*], $output:ty, $map:expr) => {
        #[pyfunction]
        #[pyo3(signature = (cf, rate, maturity, $($parameter,)* params=None))]
        fn $name(py: Python<'_>, cf: &Bound<'_, PyAny>, rate: f64, maturity: f64, $($parameter: $ty,)* params: Option<CarrMadanParams>) -> PyResult<$output> {
            let cf = SelectedCf::extract(cf)?; let params = params.map(CarrMadanParams::to_core).unwrap_or_default();
            let result = py.detach(|| native::$name(&cf, rate, maturity, $($argument,)* params));
            cf.finish(result).map($map)
        }
    }
}
transform_function!(carr_madan_fft, [spot: f64 => spot], Vec<(f64, f64)>, std::convert::identity);
transform_function!(carr_madan_fft_complex, [spot: f64 => spot], Vec<(f64, f64)>, std::convert::identity);
transform_function!(carr_madan_fft_greeks, [spot: f64 => spot], Vec<CarrMadanGreeksPoint>, |values: Vec<native::CarrMadanGreeksPoint>| values.into_iter().map(Into::into).collect());
transform_function!(carr_madan_fft_strikes, [spot: f64 => spot, strikes: Vec<f64> => &strikes], Vec<(f64, f64)>, std::convert::identity);
transform_function!(carr_madan_price_at_strikes, [spot: f64 => spot, strikes: Vec<f64> => &strikes], Vec<(f64, f64)>, std::convert::identity);
transform_function!(carr_madan_frft_grid, [log_strike_start: f64 => log_strike_start, log_strike_spacing: f64 => log_strike_spacing], Vec<(f64, f64)>, std::convert::identity);

#[pyfunction]
fn frft(py: Python<'_>, input: Vec<Complex64>, beta: f64) -> PyResult<Vec<Complex64>> {
    py.detach(|| catch_unwind_py(|| native::frft(&input, beta)))
}
#[pyfunction]
fn interpolate_strike_prices(strike_slice: Vec<(f64, f64)>, strikes: Vec<f64>) -> Vec<(f64, f64)> {
    native::interpolate_strike_prices(&strike_slice, &strikes)
}
#[pyfunction]
fn carr_madan_price_at_strikes_with_samples(
    py: Python<'_>,
    weighted_samples: Vec<Complex64>,
    strikes: Vec<f64>,
    params: CarrMadanParams,
) -> PyResult<Vec<(f64, f64)>> {
    py.detach(|| {
        native::frft::carr_madan_price_at_strikes_with_samples(
            &weighted_samples,
            &strikes,
            params.to_core(),
        )
    })
    .map_err(error)
}

macro_rules! levy_model {
    ($name:ident, {$($field:ident),+}, {$($methods:tt)*}) => {
        #[pyclass(module = "openferric", from_py_object, get_all, set_all)]
        #[derive(Clone, Copy)]
        pub struct $name { $(pub $field: f64),+ }
        impl $name { fn to_core(self) -> openferric_core::models::$name { openferric_core::models::$name { $($field: self.$field),+ } } }
        #[pymethods]
        impl $name {
            #[new]
            fn new($($field: f64),+) -> PyResult<Self> { let result = Self { $($field),+ }; result.validate()?; Ok(result) }
            fn validate(&self) -> PyResult<()> { self.to_core().validate().map_err(error) }
            fn martingale_correction(&self) -> PyResult<f64> { self.to_core().martingale_correction().map_err(error) }
            fn characteristic_fn(&self, argument: Complex64, spot: f64, rate: f64, dividend_yield: f64, maturity: f64) -> PyResult<Complex64> { self.to_core().characteristic_fn(argument, spot, rate, dividend_yield, maturity).map_err(error) }
            #[pyo3(signature = (spot, strikes, rate, dividend_yield, maturity, params=None))]
            fn european_calls_fft(&self, py: Python<'_>, spot: f64, strikes: Vec<f64>, rate: f64, dividend_yield: f64, maturity: f64, params: Option<CarrMadanParams>) -> PyResult<Vec<(f64, f64)>> {
                py.detach(|| self.to_core().european_calls_fft(spot, &strikes, rate, dividend_yield, maturity, params.map(CarrMadanParams::to_core).unwrap_or_default())).map_err(error)
            }
            fn simulate_terminal_spots(&self, py: Python<'_>, initial_spot: f64, rate: f64, dividend_yield: f64, horizon: f64, num_steps: usize, num_paths: usize, seed: u64) -> PyResult<Vec<f64>> {
                py.detach(|| catch_unwind_py(|| self.to_core().simulate_terminal_spots(initial_spot, rate, dividend_yield, horizon, num_steps, num_paths, seed))?.map_err(error))
            }
            $($methods)*
        }
    }
}
levy_model!(VarianceGamma, {sigma, theta, nu}, {
    fn variance_rate(&self) -> f64 { self.to_core().variance_rate() }
    fn simulate_path(&self, py: Python<'_>, initial_spot: f64, rate: f64, dividend_yield: f64, horizon: f64, num_steps: usize, seed: u64) -> PyResult<Vec<f64>> { py.detach(|| self.to_core().simulate_path(initial_spot, rate, dividend_yield, horizon, num_steps, seed)).map_err(error) }
});
levy_model!(Nig, {alpha, beta, delta}, {});
levy_model!(Cgmy, {c, g, m, y}, {
    #[staticmethod]
    fn calibrate(py: Python<'_>, spot: f64, rate: f64, dividend_yield: f64, maturity: f64, strikes: Vec<f64>, market_prices: Vec<f64>, initial_guess: Cgmy, max_iter: usize) -> PyResult<Self> {
        let result = py.detach(|| openferric_core::models::Cgmy::calibrate(spot, rate, dividend_yield, maturity, &strikes, &market_prices, initial_guess.to_core(), max_iter)).map_err(error)?;
        Ok(Self { c: result.c, g: result.g, m: result.m, y: result.y })
    }
});

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! classes { ($($name:ty),+) => { $(module.add_class::<$name>()?;)+ } }
    classes!(
        CarrMadanParams,
        CarrMadanContext,
        CarrMadanGreeksPoint,
        BlackScholesCharFn,
        HestonCharFn,
        VarianceGammaCharFn,
        CgmyCharFn,
        NigCharFn,
        VarianceGamma,
        Cgmy,
        Nig
    );
    macro_rules! functions { ($($name:ident),+) => { $(module.add_function(wrap_pyfunction!($name, module)?)?;)+ } }
    functions!(
        carr_madan_fft,
        carr_madan_fft_complex,
        carr_madan_fft_greeks,
        carr_madan_fft_strikes,
        carr_madan_price_at_strikes,
        carr_madan_frft_grid,
        frft,
        interpolate_strike_prices,
        carr_madan_price_at_strikes_with_samples
    );
    module.add("DEFAULT_FFT_N", native::DEFAULT_FFT_N)?;
    module.add("HIGH_RES_FFT_N", native::HIGH_RES_FFT_N)?;
    module.add("DEFAULT_ETA", native::DEFAULT_ETA)?;
    module.add("DEFAULT_ALPHA", native::DEFAULT_ALPHA)?;
    Ok(())
}
