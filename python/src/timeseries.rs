use openferric_core::math::timeseries as native;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::helpers::catch_unwind_py;
use crate::risk::{ChristoffersenBacktestResult, KupiecBacktestResult, VarBacktestResult};

macro_rules! result_type {
    ($name:ident { $($field:ident: $field_type:ty),* $(,)? }) => {
        #[pyclass(module = "openferric", get_all, from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            $($field: $field_type),*
        }

        impl From<native::$name> for $name {
            fn from(value: native::$name) -> Self {
                Self { $($field: value.$field.into()),* }
            }
        }
    };
}

result_type!(NormalFit {
    mean: f64,
    std_dev: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64
});
result_type!(StudentTFit {
    location: f64,
    scale: f64,
    degrees_of_freedom: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64
});
result_type!(SkewTFit {
    location: f64,
    scale: f64,
    degrees_of_freedom: f64,
    skew_lambda: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64
});
result_type!(LedoitWolfCorrelation { covariance: Vec<Vec<f64>>, correlation: Vec<Vec<f64>>, shrinkage: f64 });

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ReturnDistributionFits {
    inner: native::ReturnDistributionFits,
}

#[pymethods]
impl ReturnDistributionFits {
    #[getter]
    fn normal(&self) -> NormalFit {
        self.inner.normal.into()
    }

    #[getter]
    fn student_t(&self) -> StudentTFit {
        self.inner.student_t.into()
    }

    #[getter]
    fn skew_t(&self) -> SkewTFit {
        self.inner.skew_t.into()
    }

    fn best_model_by_aic(&self) -> &'static str {
        self.inner.best_model_by_aic()
    }
}

macro_rules! value_functions {
    ($($name:ident ($($argument:ident: $argument_type:ty),*) -> $output:ty => $body:expr);* $(;)?) => {
        $(
            #[pyfunction]
            fn $name(py: Python<'_>, $($argument: $argument_type),*) -> PyResult<$output> {
                py.detach(|| catch_unwind_py(|| $body))
            }
        )*

        fn register_values(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_function(wrap_pyfunction!($name, module)?)?;)*
            Ok(())
        }
    };
}

value_functions! {
    simple_returns(prices: Vec<f64>) -> Vec<f64> => native::simple_returns(&prices);
    log_returns(prices: Vec<f64>) -> Vec<f64> => native::log_returns(&prices);
    rolling_mean(series: Vec<f64>, window: usize) -> Vec<f64> => native::rolling_mean(&series, window);
    rolling_std_dev(series: Vec<f64>, window: usize) -> Vec<f64> => native::rolling_std_dev(&series, window);
    rolling_skewness(series: Vec<f64>, window: usize) -> Vec<f64> => native::rolling_skewness(&series, window);
    rolling_excess_kurtosis(series: Vec<f64>, window: usize) -> Vec<f64> => native::rolling_excess_kurtosis(&series, window);
    ewma_volatility(returns: Vec<f64>, decay: f64) -> Vec<f64> => native::ewma_volatility(&returns, decay);
    realized_vol_close_to_close(closes: Vec<f64>, periods_per_year: f64) -> f64 => native::realized_vol_close_to_close(&closes, periods_per_year);
    fit_normal_distribution(returns: Vec<f64>) -> NormalFit => native::fit_normal_distribution(&returns).into();
    fit_student_t_distribution(returns: Vec<f64>) -> StudentTFit => native::fit_student_t_distribution(&returns).into();
    fit_skew_t_distribution(returns: Vec<f64>) -> SkewTFit => native::fit_skew_t_distribution(&returns).into();
    fit_return_distributions(returns: Vec<f64>) -> ReturnDistributionFits => ReturnDistributionFits { inner: native::fit_return_distributions(&returns) };
    autocorrelation(series: Vec<f64>, max_lag: usize) -> Vec<f64> => native::autocorrelation(&series, max_lag);
    partial_autocorrelation(series: Vec<f64>, max_lag: usize) -> Vec<f64> => native::partial_autocorrelation(&series, max_lag);
    var_breach_indicators(losses: Vec<f64>, var_forecasts: Vec<f64>) -> Vec<bool> => native::var_breach_indicators(&losses, &var_forecasts);
    kupiec_test(losses: Vec<f64>, var_forecasts: Vec<f64>, confidence: f64) -> KupiecBacktestResult => KupiecBacktestResult::from_core(native::kupiec_test(&losses, &var_forecasts, confidence));
    christoffersen_test(losses: Vec<f64>, var_forecasts: Vec<f64>, confidence: f64) -> ChristoffersenBacktestResult => ChristoffersenBacktestResult::from_core(native::christoffersen_test(&losses, &var_forecasts, confidence));
    backtest_var(losses: Vec<f64>, var_forecasts: Vec<f64>, confidence: f64) -> VarBacktestResult => VarBacktestResult::from_core(native::backtest_var(&losses, &var_forecasts, confidence));
    correlation_condition_number(correlation: Vec<Vec<f64>>) -> Option<f64> => native::correlation_condition_number(&correlation);
}

macro_rules! checked_functions {
    ($($name:ident ($($argument:ident: $argument_type:ty),*) -> $output:ty => $body:expr);* $(;)?) => {
        $(
            #[pyfunction]
            fn $name(py: Python<'_>, $($argument: $argument_type),*) -> PyResult<$output> {
                py.detach(|| catch_unwind_py(|| $body))?
                    .map(Into::into)
                    .map_err(PyValueError::new_err)
            }
        )*

        fn register_checked(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_function(wrap_pyfunction!($name, module)?)?;)*
            Ok(())
        }
    };
}

checked_functions! {
    realized_vol_parkinson(highs: Vec<f64>, lows: Vec<f64>, periods_per_year: f64) -> f64 => native::realized_vol_parkinson(&highs, &lows, periods_per_year);
    realized_vol_garman_klass(opens: Vec<f64>, highs: Vec<f64>, lows: Vec<f64>, closes: Vec<f64>, periods_per_year: f64) -> f64 => native::realized_vol_garman_klass(&opens, &highs, &lows, &closes, periods_per_year);
    realized_vol_yang_zhang(opens: Vec<f64>, highs: Vec<f64>, lows: Vec<f64>, closes: Vec<f64>, periods_per_year: f64) -> f64 => native::realized_vol_yang_zhang(&opens, &highs, &lows, &closes, periods_per_year);
    sample_correlation_matrix(returns: Vec<Vec<f64>>) -> Vec<Vec<f64>> => native::sample_correlation_matrix(&returns);
    ledoit_wolf_correlation_matrix(returns: Vec<Vec<f64>>) -> LedoitWolfCorrelation => native::ledoit_wolf_correlation_matrix(&returns);
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NormalFit>()?;
    module.add_class::<StudentTFit>()?;
    module.add_class::<SkewTFit>()?;
    module.add_class::<ReturnDistributionFits>()?;
    module.add_class::<LedoitWolfCorrelation>()?;
    module.add_class::<KupiecBacktestResult>()?;
    module.add_class::<ChristoffersenBacktestResult>()?;
    module.add_class::<VarBacktestResult>()?;
    register_values(module)?;
    register_checked(module)?;
    Ok(())
}
