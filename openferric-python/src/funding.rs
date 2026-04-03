use chrono::{DateTime, Utc};
use openferric_core::instruments::FundingRateSwap as CoreFundingRateSwap;
use openferric_core::pricing::funding_rate_swap::{
    FundingRateSwapRisks as CoreFundingRateSwapRisks,
    funding_rate_swap_discount_dv01 as core_discount_dv01, funding_rate_swap_dv01 as core_dv01,
    funding_rate_swap_mtm as core_mtm, funding_rate_swap_risks as core_risks,
    funding_rate_swap_theta as core_theta, funding_rate_swap_vega as core_vega,
};
use openferric_core::rates::{
    FundingRateCurve as CoreFundingRateCurve, FundingRateSnapshot as CoreFundingRateSnapshot,
    FundingRateStats as CoreFundingRateStats, MultiVenueFundingCurve as CoreMultiVenueFundingCurve,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{format_datetime, parse_datetime};
use openferric_core::rates::YieldCurve as CoreYieldCurve;

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FundingRateSnapshot {
    #[pyo3(get, set)]
    pub venue: String,
    #[pyo3(get, set)]
    pub asset: String,
    #[pyo3(get, set)]
    pub rate: f64,
    #[pyo3(get, set)]
    pub timestamp: String,
}

impl FundingRateSnapshot {
    fn to_core(&self) -> PyResult<CoreFundingRateSnapshot> {
        Ok(CoreFundingRateSnapshot {
            venue: self.venue.clone(),
            asset: self.asset.clone(),
            rate: self.rate,
            timestamp: parse_datetime(&self.timestamp)?,
        })
    }

    fn from_core(snapshot: CoreFundingRateSnapshot) -> Self {
        Self {
            venue: snapshot.venue,
            asset: snapshot.asset,
            rate: snapshot.rate,
            timestamp: format_datetime(snapshot.timestamp),
        }
    }
}

#[pymethods]
impl FundingRateSnapshot {
    #[new]
    fn new(venue: String, asset: String, rate: f64, timestamp: String) -> PyResult<Self> {
        let _ = parse_datetime(&timestamp)?;
        Ok(Self {
            venue,
            asset,
            rate,
            timestamp,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FundingRateSnapshot(venue={:?}, asset={:?}, rate={}, timestamp={:?})",
            self.venue, self.asset, self.rate, self.timestamp
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FundingRateStats {
    #[pyo3(get, set)]
    pub window_size: usize,
    #[pyo3(get, set)]
    pub mean: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub skew: f64,
    #[pyo3(get, set)]
    pub kurtosis: f64,
}

impl FundingRateStats {
    fn from_core(stats: CoreFundingRateStats) -> Self {
        Self {
            window_size: stats.window_size,
            mean: stats.mean,
            vol: stats.vol,
            skew: stats.skew,
            kurtosis: stats.kurtosis,
        }
    }
}

#[pymethods]
impl FundingRateStats {
    #[new]
    fn new(window_size: usize, mean: f64, vol: f64, skew: f64, kurtosis: f64) -> Self {
        Self {
            window_size,
            mean,
            vol,
            skew,
            kurtosis,
        }
    }

    #[staticmethod]
    fn from_rates(rates: Vec<f64>) -> Self {
        Self::from_core(CoreFundingRateStats::from_rates(&rates))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("window_size", self.window_size)?;
        dict.set_item("mean", self.mean)?;
        dict.set_item("vol", self.vol)?;
        dict.set_item("skew", self.skew)?;
        dict.set_item("kurtosis", self.kurtosis)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "FundingRateStats(window_size={}, mean={}, vol={}, skew={}, kurtosis={})",
            self.window_size, self.mean, self.vol, self.skew, self.kurtosis
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FundingRateCurve {
    inner: CoreFundingRateCurve,
}

#[pymethods]
impl FundingRateCurve {
    #[new]
    fn new(py: Python<'_>, snapshots: Vec<Py<FundingRateSnapshot>>) -> PyResult<Self> {
        let snapshots = snapshots
            .into_iter()
            .map(|snapshot| snapshot.borrow(py).to_core())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: CoreFundingRateCurve::new(snapshots),
        })
    }

    fn forward_rate(&self, t: f64) -> f64 {
        self.inner.forward_rate(t)
    }

    fn cumulative_index(&self, t: f64) -> f64 {
        self.inner.cumulative_index(t)
    }

    fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }

    #[staticmethod]
    fn per_period_rate_to_apr(rate: f64) -> f64 {
        CoreFundingRateCurve::per_period_rate_to_apr(rate)
    }

    #[staticmethod]
    fn apr_to_per_period_rate(apr: f64) -> f64 {
        CoreFundingRateCurve::apr_to_per_period_rate(apr)
    }

    fn expected_rate(&self, as_of: &str, start: &str, end: &str) -> PyResult<f64> {
        let as_of = parse_datetime(as_of)?;
        let start = parse_datetime(start)?;
        let end = parse_datetime(end)?;
        Ok(self.inner.expected_rate(as_of, start, end))
    }

    fn rolling_stats(&self, py: Python<'_>, window_size: usize) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .rolling_stats(window_size)
            .into_iter()
            .map(|(timestamp, stats)| {
                let dict = PyDict::new(py);
                dict.set_item("timestamp", format_datetime(timestamp))?;
                dict.set_item("window_size", stats.window_size)?;
                dict.set_item("mean", stats.mean)?;
                dict.set_item("vol", stats.vol)?;
                dict.set_item("skew", stats.skew)?;
                dict.set_item("kurtosis", stats.kurtosis)?;
                Ok(dict.unbind())
            })
            .collect()
    }

    #[staticmethod]
    fn flat(apr: f64) -> Self {
        Self {
            inner: CoreFundingRateCurve::flat(apr),
        }
    }

    fn parallel_shifted(&self, bump_apr: f64) -> Self {
        Self {
            inner: self.inner.parallel_shifted(bump_apr),
        }
    }

    fn snapshots(&self) -> Vec<FundingRateSnapshot> {
        self.inner
            .snapshots()
            .iter()
            .cloned()
            .map(FundingRateSnapshot::from_core)
            .collect()
    }

    fn nodes(&self) -> Vec<(f64, f64)> {
        self.inner.nodes().to_vec()
    }

    fn anchor_timestamp(&self) -> Option<String> {
        self.inner.anchor_timestamp().map(format_datetime)
    }

    fn __repr__(&self) -> String {
        format!(
            "FundingRateCurve(snapshots={}, nodes={})",
            self.inner.snapshots().len(),
            self.inner.nodes().len()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MultiVenueFundingCurve {
    inner: CoreMultiVenueFundingCurve,
}

#[pymethods]
impl MultiVenueFundingCurve {
    #[new]
    fn new(py: Python<'_>, curves: Vec<(Py<FundingRateCurve>, f64)>) -> PyResult<Self> {
        let curves = curves
            .into_iter()
            .map(|(curve, weight)| (curve.borrow(py).inner.clone(), weight))
            .collect();
        Ok(Self {
            inner: CoreMultiVenueFundingCurve::new(curves),
        })
    }

    fn forward_rate(&self, t: f64) -> f64 {
        self.inner.forward_rate(t)
    }

    fn cumulative_index(&self, t: f64) -> f64 {
        self.inner.cumulative_index(t)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiVenueFundingCurve(curves={})",
            self.inner.curves().len()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FundingRateSwap {
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub fixed_rate: f64,
    #[pyo3(get, set)]
    pub entry_time: String,
    #[pyo3(get, set)]
    pub maturity: String,
    #[pyo3(get, set)]
    pub settlement_interval_hours: u32,
    #[pyo3(get, set)]
    pub venue: String,
    #[pyo3(get, set)]
    pub asset: String,
}

impl FundingRateSwap {
    fn to_core(&self) -> PyResult<CoreFundingRateSwap> {
        let mut swap = CoreFundingRateSwap::new(
            self.notional,
            self.fixed_rate,
            parse_datetime(&self.entry_time)?,
            parse_datetime(&self.maturity)?,
            self.venue.clone(),
            self.asset.clone(),
        );
        swap.settlement_interval_hours = self.settlement_interval_hours;
        Ok(swap)
    }
}

#[pymethods]
impl FundingRateSwap {
    #[new]
    fn new(
        notional: f64,
        fixed_rate: f64,
        entry_time: String,
        maturity: String,
        venue: String,
        asset: String,
    ) -> PyResult<Self> {
        let _ = parse_datetime(&entry_time)?;
        let _ = parse_datetime(&maturity)?;
        Ok(Self {
            notional,
            fixed_rate,
            entry_time,
            maturity,
            settlement_interval_hours: 8,
            venue,
            asset,
        })
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?
            .validate()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn settlement_schedule(&self) -> PyResult<Vec<String>> {
        Ok(self
            .to_core()?
            .settlement_schedule()
            .into_iter()
            .map(format_datetime)
            .collect())
    }

    fn realized_pnl(&self, fixings: Vec<(String, f64)>) -> PyResult<f64> {
        let fixings = fixings
            .into_iter()
            .map(|(timestamp, rate)| Ok((parse_datetime(&timestamp)?, rate)))
            .collect::<PyResult<Vec<(DateTime<Utc>, f64)>>>()?;
        Ok(self.to_core()?.realized_pnl(&fixings))
    }

    #[staticmethod]
    fn interval_pnl(fixed_rate: f64, floating_rate: f64, notional: f64) -> f64 {
        CoreFundingRateSwap::interval_pnl(fixed_rate, floating_rate, notional)
    }

    fn __repr__(&self) -> String {
        format!(
            "FundingRateSwap(notional={}, fixed_rate={}, entry_time={:?}, maturity={:?}, venue={:?}, asset={:?})",
            self.notional, self.fixed_rate, self.entry_time, self.maturity, self.venue, self.asset
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct FundingRateSwapRisks {
    #[pyo3(get, set)]
    pub mtm: f64,
    #[pyo3(get, set)]
    pub dv01: f64,
    #[pyo3(get, set)]
    pub vega: f64,
    #[pyo3(get, set)]
    pub theta: f64,
}

impl FundingRateSwapRisks {
    fn from_core(risks: CoreFundingRateSwapRisks) -> Self {
        Self {
            mtm: risks.mtm,
            dv01: risks.dv01,
            vega: risks.vega,
            theta: risks.theta,
        }
    }
}

#[pymethods]
impl FundingRateSwapRisks {
    #[new]
    fn new(mtm: f64, dv01: f64, vega: f64, theta: f64) -> Self {
        Self {
            mtm,
            dv01,
            vega,
            theta,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("mtm", self.mtm)?;
        dict.set_item("dv01", self.dv01)?;
        dict.set_item("vega", self.vega)?;
        dict.set_item("theta", self.theta)?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "FundingRateSwapRisks(mtm={}, dv01={}, vega={}, theta={})",
            self.mtm, self.dv01, self.vega, self.theta
        )
    }
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_mtm(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(core_mtm(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_dv01(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(core_dv01(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_discount_dv01(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(core_discount_dv01(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_theta(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(core_theta(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_vega(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(core_vega(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (swap, curve, as_of, discount_curve=None))]
pub fn funding_rate_swap_risks(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: &str,
    discount_curve: Option<Vec<(f64, f64)>>,
) -> PyResult<FundingRateSwapRisks> {
    let dc_ref = discount_curve.map(CoreYieldCurve::new);
    Ok(FundingRateSwapRisks::from_core(core_risks(
        &swap.to_core()?,
        &curve.inner,
        dc_ref.as_ref(),
        parse_datetime(as_of)?,
    )))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(funding_rate_swap_mtm, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(funding_rate_swap_dv01, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        funding_rate_swap_discount_dv01,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(funding_rate_swap_theta, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(funding_rate_swap_vega, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(funding_rate_swap_risks, module)?)?;
    module.add_class::<FundingRateSnapshot>()?;
    module.add_class::<FundingRateStats>()?;
    module.add_class::<FundingRateCurve>()?;
    module.add_class::<MultiVenueFundingCurve>()?;
    module.add_class::<FundingRateSwap>()?;
    module.add_class::<FundingRateSwapRisks>()?;
    Ok(())
}
