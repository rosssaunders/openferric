use crate::helpers::catch_unwind_py;
use openferric_core::models::hjm as native;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HjmFactorShape {
    Parallel,
    Slope,
    Curvature,
}
impl HjmFactorShape {
    fn to_core(self) -> native::HjmFactorShape {
        match self {
            Self::Parallel => native::HjmFactorShape::Parallel,
            Self::Slope => native::HjmFactorShape::Slope,
            Self::Curvature => native::HjmFactorShape::Curvature,
        }
    }
}

#[pyclass(module = "openferric", from_py_object, get_all, set_all)]
#[derive(Clone, Copy)]
pub struct HjmFactor {
    shape: HjmFactorShape,
    volatility: f64,
    mean_reversion: f64,
}
impl HjmFactor {
    fn to_core(self) -> native::HjmFactor {
        native::HjmFactor {
            shape: self.shape.to_core(),
            volatility: self.volatility,
            mean_reversion: self.mean_reversion,
        }
    }
    fn from_core(value: native::HjmFactor) -> Self {
        Self {
            shape: match value.shape {
                native::HjmFactorShape::Parallel => HjmFactorShape::Parallel,
                native::HjmFactorShape::Slope => HjmFactorShape::Slope,
                native::HjmFactorShape::Curvature => HjmFactorShape::Curvature,
            },
            volatility: value.volatility,
            mean_reversion: value.mean_reversion,
        }
    }
}
#[pymethods]
impl HjmFactor {
    #[new]
    fn new(shape: HjmFactorShape, volatility: f64, mean_reversion: f64) -> Self {
        Self {
            shape,
            volatility,
            mean_reversion,
        }
    }
    fn sigma(&self, tau: f64) -> f64 {
        self.to_core().sigma(tau)
    }
    fn integrated_sigma(&self, tau: f64) -> f64 {
        self.to_core().integrated_sigma(tau)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HjmModel {
    inner: native::HjmModel,
}
#[pymethods]
impl HjmModel {
    #[new]
    fn new(factors: Vec<HjmFactor>, correlation: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: native::HjmModel::new(
                factors.into_iter().map(HjmFactor::to_core).collect(),
                correlation,
            )
            .map_err(error)?,
        })
    }
    #[staticmethod]
    fn single_factor_exponential(sigma0: f64, kappa: f64) -> PyResult<Self> {
        let inner = native::HjmModel::single_factor_exponential(sigma0, kappa);
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }
    #[staticmethod]
    #[pyo3(signature = (volatilities, mean_reversions, correlation=None))]
    fn multi_factor_parallel_slope_curvature(
        volatilities: Vec<f64>,
        mean_reversions: Vec<f64>,
        correlation: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: native::HjmModel::multi_factor_parallel_slope_curvature(
                &volatilities,
                &mean_reversions,
                correlation,
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn factors(&self) -> Vec<HjmFactor> {
        self.inner
            .factors
            .iter()
            .copied()
            .map(HjmFactor::from_core)
            .collect()
    }
    #[getter]
    fn correlation(&self) -> Vec<Vec<f64>> {
        self.inner.correlation.clone()
    }
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(error)
    }
    fn factor_volatility(&self, factor_index: usize, time: f64, maturity: f64) -> PyResult<f64> {
        if factor_index >= self.inner.factors.len() {
            return Err(PyIndexError::new_err("factor index out of range"));
        }
        Ok(self.inner.factor_volatility(factor_index, time, maturity))
    }
    fn integrated_factor_volatility(
        &self,
        factor_index: usize,
        time: f64,
        maturity: f64,
    ) -> PyResult<f64> {
        if factor_index >= self.inner.factors.len() {
            return Err(PyIndexError::new_err("factor index out of range"));
        }
        Ok(self
            .inner
            .integrated_factor_volatility(factor_index, time, maturity))
    }
    fn drift(&self, time: f64, maturity: f64) -> f64 {
        self.inner.drift(time, maturity)
    }
    fn simulate_forward_curve_euler(
        &self,
        py: Python<'_>,
        initial_forwards: Vec<f64>,
        maturities: Vec<f64>,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| {
            catch_unwind_py(|| {
                self.inner.simulate_forward_curve_euler(
                    &initial_forwards,
                    &maturities,
                    horizon,
                    num_steps,
                    seed,
                )
            })?
            .map_err(error)
        })
    }
    #[staticmethod]
    fn zero_coupon_bond_price(
        time: f64,
        maturity: f64,
        maturities: Vec<f64>,
        forwards: Vec<f64>,
    ) -> PyResult<f64> {
        native::HjmModel::zero_coupon_bond_price(time, maturity, &maturities, &forwards)
            .map_err(error)
    }
    fn price_swaption_mc(
        &self,
        py: Python<'_>,
        initial_forwards: Vec<f64>,
        maturities: Vec<f64>,
        strike: f64,
        option_expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<f64> {
        py.detach(|| {
            catch_unwind_py(|| {
                self.inner.price_swaption_mc(
                    &initial_forwards,
                    &maturities,
                    strike,
                    option_expiry,
                    swap_start,
                    swap_end,
                    is_payer,
                    notional,
                    num_paths,
                    num_steps,
                    seed,
                )
            })?
            .map_err(error)
        })
    }
    fn price_swaption_mc_with_stderr(
        &self,
        py: Python<'_>,
        initial_forwards: Vec<f64>,
        maturities: Vec<f64>,
        strike: f64,
        option_expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<(f64, f64)> {
        py.detach(|| {
            catch_unwind_py(|| {
                self.inner.price_swaption_mc_with_stderr(
                    &initial_forwards,
                    &maturities,
                    strike,
                    option_expiry,
                    swap_start,
                    swap_end,
                    is_payer,
                    notional,
                    num_paths,
                    num_steps,
                    seed,
                )
            })?
            .map_err(error)
        })
    }
}

#[pyfunction]
fn calibrate_leverage_surface(
    py: Python<'_>,
    market: &crate::market::Market,
    params: &crate::models::SlvParams,
    maturity: f64,
    num_particles: usize,
    num_steps: usize,
) -> PyResult<crate::models::LeverageSurface> {
    let market = market.to_core()?;
    let params = params.to_core();
    let inner = py
        .detach(|| {
            openferric_core::models::slv::calibrate_leverage_surface(
                &market,
                params,
                maturity,
                num_particles,
                num_steps,
            )
        })
        .map_err(error)?;
    Ok(crate::models::LeverageSurface { inner })
}

macro_rules! slv_pricer {
    ($name:ident, $convert:expr, [$($kind:ty),+]) => {
        #[pyfunction]
        fn $name(py: Python<'_>, instrument: &Bound<'_, PyAny>, market: &crate::market::Market, params: &crate::models::SlvParams, num_particles: usize, num_steps: usize) -> PyResult<crate::core::PricingResult> {
            let market = market.to_core()?; let params = params.to_core();
            $(if let Ok(value) = instrument.extract::<PyRef<$kind>>() {
                let contract = value.to_core()?;
                let result = py.detach(|| catch_unwind_py(|| openferric_core::models::slv::$name(&contract, &market, params, num_particles, num_steps)))?;
                return ($convert)(result);
            })+
            Err(pyo3::exceptions::PyTypeError::new_err("unsupported SLV instrument"))
        }
    }
}

slv_pricer!(
    slv_mc_price,
    |value: openferric_core::core::PricingResult| Ok::<_, PyErr>(value.into()),
    [
        crate::instruments::VanillaOption,
        crate::instruments::AsianOption,
        crate::instruments::BarrierOption
    ]
);
slv_pricer!(
    slv_mc_price_checked,
    |value: Result<openferric_core::core::PricingResult, openferric_core::core::PricingError>| {
        value.map(Into::into).map_err(error)
    },
    [
        crate::instruments::VanillaOption,
        crate::instruments::AsianOption,
        crate::instruments::BarrierOption
    ]
);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calibrate_leverage_surface, module)?)?;
    module.add_function(wrap_pyfunction!(slv_mc_price, module)?)?;
    module.add_function(wrap_pyfunction!(slv_mc_price_checked, module)?)?;
    module.add_class::<HjmFactorShape>()?;
    module.add_class::<HjmFactor>()?;
    module.add_class::<HjmModel>()?;
    Ok(())
}
