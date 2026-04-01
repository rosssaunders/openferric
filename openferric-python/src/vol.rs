use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use openferric_core::core::OptionType as CoreOptionType;
use openferric_core::vol::andreasen_huge::AndreasenHugeInterpolation as CoreAndreasenHugeInterpolation;
use openferric_core::vol::arbitrage::ArbitrageViolation as CoreArbitrageViolation;
use openferric_core::vol::builder::{
    BuiltVolSurface as CoreBuiltVolSurface, MarketOptionQuote as CoreMarketOptionQuote,
    VolSurfaceBuilder as CoreVolSurfaceBuilder,
};
use openferric_core::vol::fengler::FenglerSurface as CoreFenglerSurface;
use openferric_core::vol::forward::{
    AtmSkewPoint as CoreAtmSkewPoint, AtmSkewTermStructure as CoreAtmSkewTermStructure,
    ForwardVarianceCurve as CoreForwardVarianceCurve,
    ForwardVariancePoint as CoreForwardVariancePoint,
    HestonVolOfVolPoint as CoreHestonVolOfVolPoint,
    HestonVolOfVolTermStructure as CoreHestonVolOfVolTermStructure,
    SabrVolOfVolPoint as CoreSabrVolOfVolPoint,
    SabrVolOfVolTermStructure as CoreSabrVolOfVolTermStructure, VixSettings as CoreVixSettings,
    VixStyleIndex as CoreVixStyleIndex, vix_style_index_from_surface,
};
use openferric_core::vol::implied::{
    implied_vol, implied_vol_newton, lets_be_rational_initial_guess,
};
use openferric_core::vol::jaeckel::{
    implied_vol_jaeckel, implied_vol_jaeckel_normalized, normalized_black,
};
use openferric_core::vol::local_vol::{DupireLocalVol as CoreDupireLocalVol, dupire_local_vol};
use openferric_core::vol::mixture::{
    LognormalMixture as CoreLognormalMixture, calibrate_lognormal_mixture,
};
use openferric_core::vol::sabr::{SabrParams as CoreSabrParams, fit_sabr};
use openferric_core::vol::slice::{
    batch_slice_iv, eval_iv_pct, find_25d_strikes_batch, forward_vol_grid, iv_grid,
    iv_grid_clamped, log_moneyness_batch, log_returns_batch, parse_slice, realized_vol,
    slice_fit_diagnostics, solve_delta_k, term_structure_batch,
};
use openferric_core::vol::smile::{
    SmileDynamics as CoreSmileDynamics, SmileSlice as CoreSmileSlice,
    StickyDeltaSmile as CoreStickyDeltaSmile, StickyStrikeSmile as CoreStickyStrikeSmile,
    VannaVolgaQuote as CoreVannaVolgaQuote, sabr_alpha_from_atm_vol, sabr_smile_from_atm,
    shift_smile_for_spot_move, strike_from_delta_analytic, vanna_volga_price,
};
use openferric_core::vol::surface::{
    SviParams as CoreSviParams, VolSurface as CoreVolSurface, calibrate_svi,
    calibrate_svi_weighted, svi_jacobian_row,
};

use crate::helpers::{catch_unwind_py, parse_option_type};

fn value_err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn option_type_from_str(value: &str) -> PyResult<CoreOptionType> {
    parse_option_type(value)
        .ok_or_else(|| PyValueError::new_err(format!("invalid option_type '{value}'")))
}

fn smile_dynamics_from_str(value: &str) -> PyResult<CoreSmileDynamics> {
    match value.to_ascii_lowercase().as_str() {
        "sticky_strike" | "sticky-strike" | "stickystrike" => Ok(CoreSmileDynamics::StickyStrike),
        "sticky_delta" | "sticky-delta" | "stickydelta" => Ok(CoreSmileDynamics::StickyDelta),
        _ => Err(PyValueError::new_err(format!(
            "invalid smile dynamics '{value}'"
        ))),
    }
}

enum SurfaceRef<'py> {
    Vol(PyRef<'py, VolSurface>),
    Built(PyRef<'py, BuiltVolSurface>),
}

fn extract_surface<'py>(surface: &'py Bound<'py, PyAny>) -> PyResult<SurfaceRef<'py>> {
    if let Ok(vol) = surface.extract::<PyRef<'py, VolSurface>>() {
        return Ok(SurfaceRef::Vol(vol));
    }
    if let Ok(built) = surface.extract::<PyRef<'py, BuiltVolSurface>>() {
        return Ok(SurfaceRef::Built(built));
    }
    Err(PyValueError::new_err(
        "surface must be a VolSurface or BuiltVolSurface",
    ))
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SabrParams {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub beta: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub nu: f64,
}

impl SabrParams {
    fn to_core(&self) -> CoreSabrParams {
        CoreSabrParams {
            alpha: self.alpha,
            beta: self.beta,
            rho: self.rho,
            nu: self.nu,
        }
    }

    fn from_core(inner: CoreSabrParams) -> Self {
        Self {
            alpha: inner.alpha,
            beta: inner.beta,
            rho: inner.rho,
            nu: inner.nu,
        }
    }
}

#[pymethods]
impl SabrParams {
    #[new]
    fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
        Self {
            alpha,
            beta,
            rho,
            nu,
        }
    }

    fn implied_vol(&self, forward: f64, strike: f64, expiry: f64) -> f64 {
        self.to_core().implied_vol(forward, strike, expiry)
    }

    fn __repr__(&self) -> String {
        format!(
            "SabrParams(alpha={}, beta={}, rho={}, nu={})",
            self.alpha, self.beta, self.rho, self.nu
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SviParams {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub m: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl SviParams {
    fn to_core(&self) -> CoreSviParams {
        CoreSviParams {
            a: self.a,
            b: self.b,
            rho: self.rho,
            m: self.m,
            sigma: self.sigma,
        }
    }

    fn from_core(inner: CoreSviParams) -> Self {
        Self {
            a: inner.a,
            b: inner.b,
            rho: inner.rho,
            m: inner.m,
            sigma: inner.sigma,
        }
    }
}

#[pymethods]
impl SviParams {
    #[new]
    fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
        Self {
            a,
            b,
            rho,
            m,
            sigma,
        }
    }

    fn total_variance(&self, k: f64) -> f64 {
        self.to_core().total_variance(k)
    }

    fn dw_dk(&self, k: f64) -> f64 {
        self.to_core().dw_dk(k)
    }

    fn __repr__(&self) -> String {
        format!(
            "SviParams(a={}, b={}, rho={}, m={}, sigma={})",
            self.a, self.b, self.rho, self.m, self.sigma
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct MarketOptionQuote {
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub option_type: String,
}

impl MarketOptionQuote {
    fn to_core(&self) -> PyResult<CoreMarketOptionQuote> {
        Ok(CoreMarketOptionQuote::new(
            self.strike,
            self.expiry,
            self.price,
            option_type_from_str(&self.option_type)?,
        ))
    }
}

#[pymethods]
impl MarketOptionQuote {
    #[new]
    fn new(strike: f64, expiry: f64, price: f64, option_type: &str) -> PyResult<Self> {
        let _ = option_type_from_str(option_type)?;
        Ok(Self {
            strike,
            expiry,
            price,
            option_type: option_type.to_ascii_lowercase(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MarketOptionQuote(strike={}, expiry={}, price={}, option_type={:?})",
            self.strike, self.expiry, self.price, self.option_type
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct ForwardVariancePoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub total_variance: f64,
    #[pyo3(get, set)]
    pub forward_variance: f64,
}

impl ForwardVariancePoint {
    fn from_core(inner: CoreForwardVariancePoint) -> Self {
        Self {
            expiry: inner.expiry,
            total_variance: inner.total_variance,
            forward_variance: inner.forward_variance,
        }
    }
}

#[pymethods]
impl ForwardVariancePoint {
    #[new]
    fn new(expiry: f64, total_variance: f64, forward_variance: f64) -> Self {
        Self {
            expiry,
            total_variance,
            forward_variance,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ForwardVarianceCurve {
    inner: CoreForwardVarianceCurve,
}

#[pymethods]
impl ForwardVarianceCurve {
    #[new]
    fn new(total_variance_points: Vec<(f64, f64)>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreForwardVarianceCurve::new(total_variance_points).map_err(value_err)?,
        })
    }

    #[staticmethod]
    fn from_surface(surface: &Bound<'_, PyAny>, expiries: Vec<f64>) -> PyResult<Self> {
        let inner = match extract_surface(surface)? {
            SurfaceRef::Vol(vol) => CoreForwardVarianceCurve::from_surface(&vol.inner, &expiries),
            SurfaceRef::Built(built) => {
                CoreForwardVarianceCurve::from_surface(&built.inner, &expiries)
            }
        }
        .map_err(value_err)?;

        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_surface_expiries(surface: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match extract_surface(surface)? {
            SurfaceRef::Vol(vol) => CoreForwardVarianceCurve::from_surface_expiries(&vol.inner),
            SurfaceRef::Built(built) => {
                CoreForwardVarianceCurve::from_surface_expiries(&built.inner)
            }
        }
        .map_err(value_err)?;

        Ok(Self { inner })
    }

    fn points(&self) -> Vec<ForwardVariancePoint> {
        self.inner
            .points()
            .iter()
            .copied()
            .map(ForwardVariancePoint::from_core)
            .collect()
    }

    fn expiries(&self) -> Vec<f64> {
        self.inner.expiries()
    }

    fn total_variance(&self, expiry: f64) -> f64 {
        self.inner.total_variance(expiry)
    }

    fn forward_variance(&self, t1: f64, t2: f64) -> PyResult<f64> {
        self.inner.forward_variance(t1, t2).map_err(value_err)
    }

    fn forward_vol(&self, t1: f64, t2: f64) -> PyResult<f64> {
        self.inner.forward_vol(t1, t2).map_err(value_err)
    }

    fn fair_forward_variance_swap(&self, start: f64, end: f64) -> PyResult<f64> {
        self.inner
            .fair_forward_variance_swap(start, end)
            .map_err(value_err)
    }

    fn fair_forward_vol_swap(&self, start: f64, end: f64) -> PyResult<f64> {
        self.inner
            .fair_forward_vol_swap(start, end)
            .map_err(value_err)
    }

    fn price_forward_variance_swap(
        &self,
        start: f64,
        end: f64,
        strike_variance: f64,
        variance_notional: f64,
        risk_free_rate: f64,
    ) -> PyResult<f64> {
        self.inner
            .price_forward_variance_swap(
                start,
                end,
                strike_variance,
                variance_notional,
                risk_free_rate,
            )
            .map_err(value_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct AtmSkewPoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub skew: f64,
}

impl AtmSkewPoint {
    fn from_core(inner: CoreAtmSkewPoint) -> Self {
        Self {
            expiry: inner.expiry,
            skew: inner.skew,
        }
    }
}

#[pymethods]
impl AtmSkewPoint {
    #[new]
    fn new(expiry: f64, skew: f64) -> Self {
        Self { expiry, skew }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AtmSkewTermStructure {
    inner: CoreAtmSkewTermStructure,
}

#[pymethods]
impl AtmSkewTermStructure {
    #[staticmethod]
    fn from_surface(surface: &Bound<'_, PyAny>, expiries: Vec<f64>) -> PyResult<Self> {
        let inner = match extract_surface(surface)? {
            SurfaceRef::Vol(vol) => CoreAtmSkewTermStructure::from_surface(&vol.inner, &expiries),
            SurfaceRef::Built(built) => {
                CoreAtmSkewTermStructure::from_surface(&built.inner, &expiries)
            }
        }
        .map_err(value_err)?;

        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_surface_expiries(surface: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match extract_surface(surface)? {
            SurfaceRef::Vol(vol) => CoreAtmSkewTermStructure::from_surface_expiries(&vol.inner),
            SurfaceRef::Built(built) => {
                CoreAtmSkewTermStructure::from_surface_expiries(&built.inner)
            }
        }
        .map_err(value_err)?;

        Ok(Self { inner })
    }

    fn points(&self) -> Vec<AtmSkewPoint> {
        self.inner
            .points()
            .iter()
            .copied()
            .map(AtmSkewPoint::from_core)
            .collect()
    }

    fn skew(&self, expiry: f64) -> f64 {
        self.inner.skew(expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct HestonVolOfVolPoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub sigma_v: f64,
}

impl HestonVolOfVolPoint {
    fn to_core(&self) -> CoreHestonVolOfVolPoint {
        CoreHestonVolOfVolPoint {
            expiry: self.expiry,
            sigma_v: self.sigma_v,
        }
    }

    fn from_core(inner: CoreHestonVolOfVolPoint) -> Self {
        Self {
            expiry: inner.expiry,
            sigma_v: inner.sigma_v,
        }
    }
}

#[pymethods]
impl HestonVolOfVolPoint {
    #[new]
    fn new(expiry: f64, sigma_v: f64) -> Self {
        Self { expiry, sigma_v }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct HestonVolOfVolTermStructure {
    inner: CoreHestonVolOfVolTermStructure,
}

#[pymethods]
impl HestonVolOfVolTermStructure {
    #[new]
    fn new(py: Python<'_>, points: Vec<Py<HestonVolOfVolPoint>>) -> PyResult<Self> {
        let points = points
            .into_iter()
            .map(|point| point.borrow(py).to_core())
            .collect::<Vec<_>>();

        Ok(Self {
            inner: CoreHestonVolOfVolTermStructure::new(points).map_err(value_err)?,
        })
    }

    fn points(&self) -> Vec<HestonVolOfVolPoint> {
        self.inner
            .points()
            .iter()
            .copied()
            .map(HestonVolOfVolPoint::from_core)
            .collect()
    }

    fn sigma_v(&self, expiry: f64) -> f64 {
        self.inner.sigma_v(expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SabrVolOfVolPoint {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub nu: f64,
}

impl SabrVolOfVolPoint {
    fn to_core(&self) -> CoreSabrVolOfVolPoint {
        CoreSabrVolOfVolPoint {
            expiry: self.expiry,
            alpha: self.alpha,
            nu: self.nu,
        }
    }

    fn from_core(inner: CoreSabrVolOfVolPoint) -> Self {
        Self {
            expiry: inner.expiry,
            alpha: inner.alpha,
            nu: inner.nu,
        }
    }
}

#[pymethods]
impl SabrVolOfVolPoint {
    #[new]
    fn new(expiry: f64, alpha: f64, nu: f64) -> Self {
        Self { expiry, alpha, nu }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SabrVolOfVolTermStructure {
    inner: CoreSabrVolOfVolTermStructure,
}

#[pymethods]
impl SabrVolOfVolTermStructure {
    #[new]
    fn new(py: Python<'_>, points: Vec<Py<SabrVolOfVolPoint>>) -> PyResult<Self> {
        let points = points
            .into_iter()
            .map(|point| point.borrow(py).to_core())
            .collect::<Vec<_>>();

        Ok(Self {
            inner: CoreSabrVolOfVolTermStructure::new(points).map_err(value_err)?,
        })
    }

    #[staticmethod]
    fn from_sabr_params(py: Python<'_>, points: Vec<(f64, Py<SabrParams>)>) -> PyResult<Self> {
        let points = points
            .into_iter()
            .map(|(expiry, params)| (expiry, params.borrow(py).to_core()))
            .collect::<Vec<_>>();

        Ok(Self {
            inner: CoreSabrVolOfVolTermStructure::from_sabr_params(&points).map_err(value_err)?,
        })
    }

    fn points(&self) -> Vec<SabrVolOfVolPoint> {
        self.inner
            .points()
            .iter()
            .copied()
            .map(SabrVolOfVolPoint::from_core)
            .collect()
    }

    fn alpha(&self, expiry: f64) -> f64 {
        self.inner.alpha(expiry)
    }

    fn nu(&self, expiry: f64) -> f64 {
        self.inner.nu(expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct VixSettings {
    #[pyo3(get, set)]
    pub target_days: f64,
    #[pyo3(get, set)]
    pub strike_count: usize,
    #[pyo3(get, set)]
    pub log_moneyness_span: f64,
}

impl VixSettings {
    fn to_core(&self) -> CoreVixSettings {
        CoreVixSettings {
            target_days: self.target_days,
            strike_count: self.strike_count,
            log_moneyness_span: self.log_moneyness_span,
        }
    }

    fn from_core(inner: CoreVixSettings) -> Self {
        Self {
            target_days: inner.target_days,
            strike_count: inner.strike_count,
            log_moneyness_span: inner.log_moneyness_span,
        }
    }
}

#[pymethods]
impl VixSettings {
    #[new]
    fn new(target_days: f64, strike_count: usize, log_moneyness_span: f64) -> Self {
        Self {
            target_days,
            strike_count,
            log_moneyness_span,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        Self::from_core(CoreVixSettings::default())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct VixStyleIndex {
    #[pyo3(get, set)]
    pub target_days: f64,
    #[pyo3(get, set)]
    pub near_expiry: f64,
    #[pyo3(get, set)]
    pub next_expiry: f64,
    #[pyo3(get, set)]
    pub near_variance: f64,
    #[pyo3(get, set)]
    pub next_variance: f64,
    #[pyo3(get, set)]
    pub target_variance: f64,
    #[pyo3(get, set)]
    pub index: f64,
}

impl VixStyleIndex {
    fn from_core(inner: CoreVixStyleIndex) -> Self {
        Self {
            target_days: inner.target_days,
            near_expiry: inner.near_expiry,
            next_expiry: inner.next_expiry,
            near_variance: inner.near_variance,
            next_variance: inner.next_variance,
            target_variance: inner.target_variance,
            index: inner.index,
        }
    }
}

#[pymethods]
impl VixStyleIndex {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        target_days: f64,
        near_expiry: f64,
        next_expiry: f64,
        near_variance: f64,
        next_variance: f64,
        target_variance: f64,
        index: f64,
    ) -> Self {
        Self {
            target_days,
            near_expiry,
            next_expiry,
            near_variance,
            next_variance,
            target_variance,
            index,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct ArbitrageViolation {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub strike: f64,
    #[pyo3(get)]
    pub expiry: Option<f64>,
    #[pyo3(get)]
    pub t1: Option<f64>,
    #[pyo3(get)]
    pub t2: Option<f64>,
    #[pyo3(get)]
    pub density: Option<f64>,
    #[pyo3(get)]
    pub dw_dt: Option<f64>,
}

impl ArbitrageViolation {
    fn from_core(inner: &CoreArbitrageViolation) -> Self {
        match inner {
            CoreArbitrageViolation::Butterfly {
                strike,
                expiry,
                density,
            } => Self {
                kind: "butterfly".to_string(),
                strike: *strike,
                expiry: Some(*expiry),
                t1: None,
                t2: None,
                density: Some(*density),
                dw_dt: None,
            },
            CoreArbitrageViolation::Calendar {
                strike,
                t1,
                t2,
                dw_dt,
            } => Self {
                kind: "calendar".to_string(),
                strike: *strike,
                expiry: None,
                t1: Some(*t1),
                t2: Some(*t2),
                density: None,
                dw_dt: Some(*dw_dt),
            },
        }
    }
}

#[pymethods]
impl ArbitrageViolation {
    #[new]
    fn new(
        kind: String,
        strike: f64,
        expiry: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        density: Option<f64>,
        dw_dt: Option<f64>,
    ) -> PyResult<Self> {
        match kind.to_ascii_lowercase().as_str() {
            "butterfly" => Ok(Self {
                kind: "butterfly".to_string(),
                strike,
                expiry,
                t1: None,
                t2: None,
                density,
                dw_dt: None,
            }),
            "calendar" => Ok(Self {
                kind: "calendar".to_string(),
                strike,
                expiry: None,
                t1,
                t2,
                density: None,
                dw_dt,
            }),
            _ => Err(PyValueError::new_err(format!(
                "invalid violation kind '{kind}'"
            ))),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LognormalMixture {
    inner: CoreLognormalMixture,
}

#[pymethods]
impl LognormalMixture {
    #[new]
    fn new(weights: Vec<f64>, vols: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLognormalMixture::new(weights, vols).map_err(value_err)?,
        })
    }

    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    fn vols(&self) -> Vec<f64> {
        self.inner.vols.clone()
    }

    fn price(
        &self,
        option_type: &str,
        spot: f64,
        strike: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        Ok(self.inner.price(
            option_type_from_str(option_type)?,
            spot,
            strike,
            rate,
            expiry,
        ))
    }

    fn d2_call_dk2(&self, spot: f64, strike: f64, rate: f64, expiry: f64, dk: f64) -> f64 {
        self.inner.d2_call_dk2(spot, strike, rate, expiry, dk)
    }

    fn implied_density(&self, spot: f64, strike: f64, rate: f64, expiry: f64, dk: f64) -> f64 {
        self.inner.implied_density(spot, strike, rate, expiry, dk)
    }

    fn implied_density_curve(
        &self,
        spot: f64,
        strikes: Vec<f64>,
        rate: f64,
        expiry: f64,
        dk: f64,
    ) -> Vec<(f64, f64)> {
        self.inner
            .implied_density_curve(spot, &strikes, rate, expiry, dk)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SmileDynamics {
    #[pyo3(get)]
    pub kind: String,
}

impl SmileDynamics {
    fn to_core(&self) -> PyResult<CoreSmileDynamics> {
        smile_dynamics_from_str(&self.kind)
    }
}

#[pymethods]
impl SmileDynamics {
    #[new]
    fn new(kind: &str) -> PyResult<Self> {
        let _ = smile_dynamics_from_str(kind)?;
        Ok(Self {
            kind: kind.to_ascii_lowercase(),
        })
    }

    #[staticmethod]
    fn sticky_strike() -> Self {
        Self {
            kind: "sticky_strike".to_string(),
        }
    }

    #[staticmethod]
    fn sticky_delta() -> Self {
        Self {
            kind: "sticky_delta".to_string(),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct SmileSlice {
    inner: CoreSmileSlice,
}

#[pymethods]
impl SmileSlice {
    #[new]
    fn new(strikes: Vec<f64>, vols: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSmileSlice::new(strikes, vols).map_err(value_err)?,
        })
    }

    fn strikes(&self) -> Vec<f64> {
        self.inner.strikes.clone()
    }

    fn vols(&self) -> Vec<f64> {
        self.inner.vols.clone()
    }

    fn vol_at_strike(&self, strike: f64) -> f64 {
        self.inner.vol_at_strike(strike)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct StickyStrikeSmile {
    inner: CoreStickyStrikeSmile,
}

#[pymethods]
impl StickyStrikeSmile {
    #[new]
    fn new(expiry: f64, strikes: Vec<f64>, vols: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreStickyStrikeSmile::new(expiry, strikes, vols).map_err(value_err)?,
        })
    }

    #[staticmethod]
    fn from_built_surface(
        surface: &BuiltVolSurface,
        expiry: f64,
        strikes: Vec<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreStickyStrikeSmile::from_built_surface(&surface.inner, expiry, strikes)
                .map_err(value_err)?,
        })
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry
    }

    fn slice(&self) -> SmileSlice {
        SmileSlice {
            inner: self.inner.slice.clone(),
        }
    }

    fn strikes(&self) -> Vec<f64> {
        self.inner.slice.strikes.clone()
    }

    fn vols(&self) -> Vec<f64> {
        self.inner.slice.vols.clone()
    }

    fn vol(&self, strike: f64) -> f64 {
        self.inner.vol(strike)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct StickyDeltaSmile {
    inner: CoreStickyDeltaSmile,
    deltas: Vec<f64>,
    vols: Vec<f64>,
}

#[pymethods]
impl StickyDeltaSmile {
    #[new]
    fn new(deltas: Vec<f64>, vols: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreStickyDeltaSmile::new(deltas.clone(), vols.clone()).map_err(value_err)?,
            deltas,
            vols,
        })
    }

    fn deltas(&self) -> Vec<f64> {
        self.deltas.clone()
    }

    fn vols(&self) -> Vec<f64> {
        self.vols.clone()
    }

    fn vol_at_delta(&self, delta: f64) -> f64 {
        self.inner.vol_at_delta(delta)
    }

    #[allow(clippy::too_many_arguments)]
    fn strike_from_delta(
        &self,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        expiry: f64,
        target_delta: f64,
        initial_strike: f64,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<f64> {
        self.inner
            .strike_from_delta(
                spot,
                rate,
                dividend_yield,
                expiry,
                target_delta,
                initial_strike,
                tol,
                max_iter,
            )
            .map_err(value_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct VannaVolgaQuote {
    #[pyo3(get, set)]
    pub atm_vol: f64,
    #[pyo3(get, set)]
    pub rr_25d: f64,
    #[pyo3(get, set)]
    pub bf_25d: f64,
}

impl VannaVolgaQuote {
    fn to_core(&self) -> CoreVannaVolgaQuote {
        CoreVannaVolgaQuote::new(self.atm_vol, self.rr_25d, self.bf_25d)
    }
}

#[pymethods]
impl VannaVolgaQuote {
    #[new]
    fn new(atm_vol: f64, rr_25d: f64, bf_25d: f64) -> Self {
        Self {
            atm_vol,
            rr_25d,
            bf_25d,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BuiltVolSurface {
    inner: CoreBuiltVolSurface,
}

#[pymethods]
impl BuiltVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.implied_vol(strike, expiry)
    }

    fn local_vol(&self, spot: f64, expiry: f64) -> f64 {
        self.inner.local_vol(spot, expiry)
    }

    fn spot(&self) -> f64 {
        self.inner.spot()
    }

    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    fn expiries(&self) -> Vec<f64> {
        self.inner.expiries().to_vec()
    }

    fn forward_price(&self, expiry: f64) -> f64 {
        self.inner.forward_price(expiry)
    }

    fn forward_variance_curve(&self, expiries: Vec<f64>) -> PyResult<ForwardVarianceCurve> {
        Ok(ForwardVarianceCurve {
            inner: self
                .inner
                .forward_variance_curve(&expiries)
                .map_err(value_err)?,
        })
    }

    fn atm_skew_term_structure(&self, expiries: Vec<f64>) -> PyResult<AtmSkewTermStructure> {
        Ok(AtmSkewTermStructure {
            inner: self
                .inner
                .atm_skew_term_structure(&expiries)
                .map_err(value_err)?,
        })
    }

    fn vix_style_index(&self, settings: &VixSettings) -> PyResult<VixStyleIndex> {
        Ok(VixStyleIndex::from_core(
            self.inner
                .vix_style_index(settings.to_core())
                .map_err(value_err)?,
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VolSurfaceBuilder {
    inner: CoreVolSurfaceBuilder,
}

#[pymethods]
impl VolSurfaceBuilder {
    #[new]
    fn new(spot: f64, rate: f64) -> Self {
        Self {
            inner: CoreVolSurfaceBuilder::new(spot, rate),
        }
    }

    #[staticmethod]
    fn from_quotes(
        py: Python<'_>,
        spot: f64,
        rate: f64,
        quotes: Vec<Py<MarketOptionQuote>>,
    ) -> PyResult<Self> {
        let quotes = quotes
            .into_iter()
            .map(|quote| quote.borrow(py).to_core())
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            inner: CoreVolSurfaceBuilder::from_quotes(spot, rate, quotes),
        })
    }

    fn with_solver_params(&self, tol: f64, max_iter: usize) -> Self {
        Self {
            inner: self.inner.clone().with_solver_params(tol, max_iter),
        }
    }

    fn add_quote(&self, quote: &MarketOptionQuote) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().add_quote(quote.to_core()?),
        })
    }

    fn add_quotes(&self, py: Python<'_>, quotes: Vec<Py<MarketOptionQuote>>) -> PyResult<Self> {
        let quotes = quotes
            .into_iter()
            .map(|quote| quote.borrow(py).to_core())
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            inner: self.inner.clone().add_quotes(quotes),
        })
    }

    fn build(&self) -> PyResult<BuiltVolSurface> {
        Ok(BuiltVolSurface {
            inner: self.inner.build().map_err(value_err)?,
        })
    }

    fn build_with_forward_variance_curve(
        &self,
        expiries: Vec<f64>,
    ) -> PyResult<(BuiltVolSurface, ForwardVarianceCurve)> {
        let (surface, curve) = self
            .inner
            .build_with_forward_variance_curve(&expiries)
            .map_err(value_err)?;

        Ok((
            BuiltVolSurface { inner: surface },
            ForwardVarianceCurve { inner: curve },
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct FenglerSurface {
    inner: CoreFenglerSurface,
}

#[pymethods]
impl FenglerSurface {
    #[new]
    fn new(quotes: Vec<(f64, f64, f64)>, forward_curve: Vec<(f64, f64)>) -> PyResult<Self> {
        let inner = catch_unwind_py(|| CoreFenglerSurface::new(&quotes, &forward_curve))?;
        Ok(Self { inner })
    }

    fn total_variance(&self, log_moneyness: f64, expiry: f64) -> f64 {
        self.inner.total_variance(log_moneyness, expiry)
    }

    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.implied_vol(strike, expiry)
    }

    fn check_arbitrage(&self) -> Vec<ArbitrageViolation> {
        self.inner
            .check_arbitrage()
            .iter()
            .map(ArbitrageViolation::from_core)
            .collect()
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct AndreasenHugeInterpolation {
    inner: CoreAndreasenHugeInterpolation,
}

#[pymethods]
impl AndreasenHugeInterpolation {
    #[new]
    fn new(quotes: Vec<(f64, f64, f64)>, spot: f64, rate: f64, dividend: f64) -> PyResult<Self> {
        let inner =
            catch_unwind_py(|| CoreAndreasenHugeInterpolation::new(&quotes, spot, rate, dividend))?;
        Ok(Self { inner })
    }

    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.implied_vol(strike, expiry)
    }

    fn local_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.local_vol(strike, expiry)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VolSurface {
    inner: CoreVolSurface,
}

#[pymethods]
impl VolSurface {
    #[new]
    fn new(py: Python<'_>, slices: Vec<(f64, Py<SviParams>)>, forward: f64) -> PyResult<Self> {
        let slices = slices
            .into_iter()
            .map(|(expiry, params)| (expiry, params.borrow(py).to_core()))
            .collect::<Vec<_>>();

        Ok(Self {
            inner: CoreVolSurface::new(slices, forward).map_err(value_err)?,
        })
    }

    fn total_variance(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.total_variance(strike, expiry)
    }

    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        self.inner.vol(strike, expiry)
    }

    fn expiries(&self) -> Vec<f64> {
        self.inner.expiries().to_vec()
    }

    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    fn forward_price(&self, expiry: f64) -> f64 {
        self.inner.forward_price(expiry)
    }

    fn forward_variance_curve(&self, expiries: Vec<f64>) -> PyResult<ForwardVarianceCurve> {
        Ok(ForwardVarianceCurve {
            inner: self
                .inner
                .forward_variance_curve(&expiries)
                .map_err(value_err)?,
        })
    }

    fn atm_skew_term_structure(&self, expiries: Vec<f64>) -> PyResult<AtmSkewTermStructure> {
        Ok(AtmSkewTermStructure {
            inner: self
                .inner
                .atm_skew_term_structure(&expiries)
                .map_err(value_err)?,
        })
    }

    fn vix_style_index(
        &self,
        risk_free_rate: f64,
        settings: &VixSettings,
    ) -> PyResult<VixStyleIndex> {
        Ok(VixStyleIndex::from_core(
            self.inner
                .vix_style_index(risk_free_rate, settings.to_core())
                .map_err(value_err)?,
        ))
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct DupireLocalVol {
    inner: CoreDupireLocalVol<CoreVolSurface>,
}

#[pymethods]
impl DupireLocalVol {
    #[new]
    fn new(surface: &VolSurface, forward: f64) -> Self {
        Self {
            inner: CoreDupireLocalVol::new(surface.inner.clone(), forward),
        }
    }

    fn with_bumps(&self, strike_bump_rel: f64, time_bump: f64) -> Self {
        Self {
            inner: self.inner.clone().with_bumps(strike_bump_rel, time_bump),
        }
    }

    fn local_vol(&self, spot: f64, expiry: f64) -> f64 {
        self.inner.local_vol(spot, expiry)
    }
}

#[pyfunction]
pub fn py_implied_vol(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    market_price: f64,
    option_type: &str,
) -> f64 {
    let Ok(option_type) = option_type_from_str(option_type) else {
        return f64::NAN;
    };

    implied_vol(
        option_type,
        spot,
        strike,
        rate,
        expiry,
        market_price,
        1e-10,
        100,
    )
    .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_implied_vol_newton(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    market_price: f64,
    option_type: &str,
    tol: f64,
    max_iter: usize,
) -> PyResult<f64> {
    implied_vol_newton(
        option_type_from_str(option_type)?,
        spot,
        strike,
        rate,
        expiry,
        market_price,
        tol,
        max_iter,
    )
    .map_err(value_err)
}

#[pyfunction]
pub fn py_lets_be_rational_initial_guess(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    market_price: f64,
    option_type: &str,
) -> PyResult<f64> {
    Ok(lets_be_rational_initial_guess(
        option_type_from_str(option_type)?,
        spot,
        strike,
        rate,
        expiry,
        market_price,
    ))
}

#[pyfunction]
pub fn py_sabr_vol(
    forward: f64,
    strike: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    CoreSabrParams {
        alpha,
        beta,
        rho,
        nu,
    }
    .implied_vol(forward, strike, expiry)
}

#[pyfunction]
pub fn py_fit_sabr(
    forward: f64,
    strikes: Vec<f64>,
    market_vols: Vec<f64>,
    expiry: f64,
    beta: f64,
) -> SabrParams {
    SabrParams::from_core(fit_sabr(forward, &strikes, &market_vols, expiry, beta))
}

#[pyfunction]
pub fn py_svi_vol(strike: f64, forward: f64, a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> f64 {
    let k = (strike / forward).ln();
    let total_var = a + b * (rho * (k - m) + ((k - m).powi(2) + sigma * sigma).sqrt());
    if total_var > 0.0 {
        total_var.sqrt()
    } else {
        f64::NAN
    }
}

#[pyfunction]
pub fn py_svi_jacobian_row(params: &SviParams, k: f64) -> [f64; 5] {
    svi_jacobian_row(&params.to_core(), k)
}

#[pyfunction]
pub fn py_calibrate_svi(
    points: Vec<(f64, f64)>,
    init: &SviParams,
    max_iter: usize,
    learning_rate: f64,
) -> SviParams {
    SviParams::from_core(calibrate_svi(
        &points,
        init.to_core(),
        max_iter,
        learning_rate,
    ))
}

#[pyfunction]
pub fn py_calibrate_svi_weighted(
    points: Vec<(f64, f64)>,
    weights: Vec<f64>,
    init: &SviParams,
    max_iter: usize,
) -> SviParams {
    SviParams::from_core(calibrate_svi_weighted(
        &points,
        &weights,
        init.to_core(),
        max_iter,
    ))
}

#[pyfunction]
pub fn py_eval_iv_pct(model_type: u8, params: Vec<f64>, k: f64, t: f64, forward: f64) -> f64 {
    eval_iv_pct(model_type, &params, k, t, forward)
}

#[pyfunction]
pub fn py_parse_slice(
    headers: Vec<f64>,
    params: Vec<f64>,
    idx: usize,
    n_slices: usize,
) -> (u8, Vec<f64>, f64, f64) {
    let (model_type, param_slice, t, forward) = parse_slice(&headers, &params, idx, n_slices);
    (model_type, param_slice.to_vec(), t, forward)
}

#[pyfunction]
pub fn py_solve_delta_k(
    model_type: u8,
    params: Vec<f64>,
    t: f64,
    forward: f64,
    sqrt_t: f64,
    target_d1: f64,
) -> f64 {
    solve_delta_k(model_type, &params, t, forward, sqrt_t, target_d1)
}

#[pyfunction]
pub fn py_iv_grid(slice_headers: Vec<f64>, slice_params: Vec<f64>, k_grid: Vec<f64>) -> Vec<f64> {
    iv_grid(&slice_headers, &slice_params, &k_grid)
}

#[pyfunction]
pub fn py_iv_grid_clamped(
    slice_headers: Vec<f64>,
    slice_params: Vec<f64>,
    k_grid: Vec<f64>,
    k_bounds: Vec<f64>,
) -> Vec<f64> {
    iv_grid_clamped(&slice_headers, &slice_params, &k_grid, &k_bounds)
}

#[pyfunction]
pub fn py_batch_slice_iv(
    slice_headers: Vec<f64>,
    slice_params: Vec<f64>,
    k_values: Vec<f64>,
    slice_indices: Vec<u32>,
) -> Vec<f64> {
    batch_slice_iv(&slice_headers, &slice_params, &k_values, &slice_indices)
}

#[pyfunction]
pub fn py_slice_fit_diagnostics(
    model_type: u8,
    params: Vec<f64>,
    t: f64,
    forward: f64,
    market_ks: Vec<f64>,
    market_ivs_pct: Vec<f64>,
    strikes: Vec<f64>,
) -> Vec<f64> {
    slice_fit_diagnostics(
        model_type,
        &params,
        t,
        forward,
        &market_ks,
        &market_ivs_pct,
        &strikes,
    )
}

#[pyfunction]
pub fn py_find_25d_strikes_batch(slice_headers: Vec<f64>, slice_params: Vec<f64>) -> Vec<f64> {
    find_25d_strikes_batch(&slice_headers, &slice_params)
}

#[pyfunction]
pub fn py_term_structure_batch(slice_headers: Vec<f64>, slice_params: Vec<f64>) -> Vec<f64> {
    term_structure_batch(&slice_headers, &slice_params)
}

#[pyfunction]
pub fn py_forward_vol_grid(
    slice_headers: Vec<f64>,
    slice_params: Vec<f64>,
    k_points: Vec<f64>,
) -> Vec<f64> {
    forward_vol_grid(&slice_headers, &slice_params, &k_points)
}

#[pyfunction]
pub fn py_realized_vol(log_returns: Vec<f64>, obs_per_year: f64) -> f64 {
    realized_vol(&log_returns, obs_per_year)
}

#[pyfunction]
pub fn py_log_moneyness_batch(strikes: Vec<f64>, forwards: Vec<f64>) -> Vec<f64> {
    log_moneyness_batch(&strikes, &forwards)
}

#[pyfunction]
pub fn py_log_returns_batch(prices: Vec<f64>) -> Vec<f64> {
    log_returns_batch(&prices)
}

#[pyfunction]
pub fn py_normalized_black(x: f64, s: f64, is_call: bool) -> f64 {
    normalized_black(x, s, is_call)
}

#[pyfunction]
pub fn py_implied_vol_jaeckel_normalized(beta: f64, x: f64, is_call: bool) -> PyResult<f64> {
    implied_vol_jaeckel_normalized(beta, x, is_call).map_err(value_err)
}

#[pyfunction]
pub fn py_implied_vol_jaeckel(
    price: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    is_call: bool,
) -> PyResult<f64> {
    implied_vol_jaeckel(price, forward, strike, expiry, is_call).map_err(value_err)
}

#[pyfunction]
pub fn py_calibrate_lognormal_mixture(
    option_type: &str,
    spot: f64,
    rate: f64,
    expiry: f64,
    strikes: Vec<f64>,
    market_prices: Vec<f64>,
    components: usize,
) -> PyResult<LognormalMixture> {
    Ok(LognormalMixture {
        inner: calibrate_lognormal_mixture(
            option_type_from_str(option_type)?,
            spot,
            rate,
            expiry,
            &strikes,
            &market_prices,
            components,
        )
        .map_err(value_err)?,
    })
}

#[pyfunction]
pub fn py_sabr_alpha_from_atm_vol(
    forward: f64,
    expiry: f64,
    atm_vol: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    sabr_alpha_from_atm_vol(forward, expiry, atm_vol, beta, rho, nu)
}

#[pyfunction]
pub fn py_sabr_smile_from_atm(
    forward: f64,
    expiry: f64,
    atm_vol: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    strikes: Vec<f64>,
) -> (SabrParams, Vec<(f64, f64)>) {
    let (params, smile) = sabr_smile_from_atm(forward, expiry, atm_vol, beta, rho, nu, &strikes);
    (SabrParams::from_core(params), smile)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_vanna_volga_price(
    option_type: &str,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    quote: &VannaVolgaQuote,
) -> PyResult<f64> {
    Ok(vanna_volga_price(
        option_type_from_str(option_type)?,
        spot,
        strike,
        rate,
        dividend_yield,
        expiry,
        quote.to_core(),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_shift_smile_for_spot_move(
    slice: &SmileSlice,
    spot_old: f64,
    spot_new: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    dynamics: &SmileDynamics,
) -> PyResult<SmileSlice> {
    Ok(SmileSlice {
        inner: shift_smile_for_spot_move(
            &slice.inner,
            spot_old,
            spot_new,
            rate,
            dividend_yield,
            expiry,
            dynamics.to_core()?,
        )
        .map_err(value_err)?,
    })
}

#[pyfunction]
pub fn py_strike_from_delta_analytic(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    vol: f64,
    delta: f64,
) -> Option<f64> {
    strike_from_delta_analytic(spot, rate, dividend_yield, expiry, vol, delta)
}

#[pyfunction]
pub fn py_dupire_local_vol(surface: &VolSurface, forward: f64, spot: f64, expiry: f64) -> f64 {
    dupire_local_vol(surface.inner.clone(), forward, spot, expiry)
}

#[pyfunction]
pub fn py_vix_style_index_from_surface(
    surface: &Bound<'_, PyAny>,
    risk_free_rate: f64,
    settings: &VixSettings,
) -> PyResult<VixStyleIndex> {
    let index = match extract_surface(surface)? {
        SurfaceRef::Vol(vol) => {
            vix_style_index_from_surface(&vol.inner, risk_free_rate, settings.to_core())
        }
        SurfaceRef::Built(built) => {
            vix_style_index_from_surface(&built.inner, risk_free_rate, settings.to_core())
        }
    }
    .map_err(value_err)?;

    Ok(VixStyleIndex::from_core(index))
}
