use openferric_core::core::PricingResult as CorePricingResult;
use openferric_core::instruments::VanillaOption;
use openferric_core::models::commodity::{
    CommodityForwardCurve as CoreCommodityForwardCurve,
    CommoditySeasonalityModel as CoreCommoditySeasonalityModel,
    CommodityStorageContract as CoreCommodityStorageContract, CurveStructure, ForwardInterpolation,
    FuturesQuote as CoreFuturesQuote, SchwartzOneFactor as CoreSchwartzOneFactor,
    SchwartzSmithTwoFactor as CoreSchwartzSmithTwoFactor, SeasonalityMode,
    StorageLsmConfig as CoreStorageLsmConfig, StorageValuation as CoreStorageValuation,
    TwoFactorCommodityProcess as CoreTwoFactorCommodityProcess,
    TwoFactorSpreadModel as CoreTwoFactorSpreadModel,
    VolumeConstrainedSwing as CoreVolumeConstrainedSwing, convenience_yield_from_term_structure,
    implied_convenience_yield, intrinsic_storage_value, value_storage_intrinsic_extrinsic,
};
use openferric_core::models::hw_calibration::{
    calibrate_hull_white_params, hw_atm_swaption_vol_approx,
};
use openferric_core::models::lmm::{
    LmmModel as CoreLmmModel, LmmParams as CoreLmmParams, black_swaption_price,
    initial_swap_rate_annuity,
};
use openferric_core::models::rough_bergomi::{
    fbm_covariance, fbm_path_cholesky, fbm_path_hybrid, rbergomi_european_mc,
    rbergomi_implied_vol_surface,
};
use openferric_core::models::short_rate::{CIR as CoreCIR, HullWhite as CoreHullWhite};
use openferric_core::models::slv::{
    LeverageSlice as CoreLeverageSlice, LeverageSurface as CoreLeverageSurface,
    SlvParams as CoreSlvParams, calibrate_leverage_surface, nadaraya_watson_conditional_mean,
    slv_mc_price, slv_mc_price_checked,
};
use openferric_core::models::stochastic::{Gbm as CoreGbm, Heston as CoreHeston, Sabr as CoreSabr};
use openferric_core::pricing::OptionType;
use openferric_core::rates::YieldCurve;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::helpers::{build_market, parse_option_type};

fn string_err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_forward_interpolation(value: &str) -> PyResult<ForwardInterpolation> {
    match value.to_ascii_lowercase().as_str() {
        "piecewise_flat" | "piecewiseflat" | "flat" => Ok(ForwardInterpolation::PiecewiseFlat),
        "linear" => Ok(ForwardInterpolation::Linear),
        "cubic_spline" | "cubicspline" | "cubic" | "spline" => {
            Ok(ForwardInterpolation::CubicSpline)
        }
        _ => Err(PyValueError::new_err(format!(
            "invalid interpolation '{value}'; expected piecewise_flat, linear, or cubic_spline"
        ))),
    }
}

fn forward_interpolation_name(value: ForwardInterpolation) -> &'static str {
    match value {
        ForwardInterpolation::PiecewiseFlat => "piecewise_flat",
        ForwardInterpolation::Linear => "linear",
        ForwardInterpolation::CubicSpline => "cubic_spline",
    }
}

fn curve_structure_name(value: CurveStructure) -> &'static str {
    match value {
        CurveStructure::Contango => "contango",
        CurveStructure::Backwardation => "backwardation",
        CurveStructure::Flat => "flat",
        CurveStructure::Mixed => "mixed",
    }
}

fn parse_seasonality_mode(value: &str) -> PyResult<SeasonalityMode> {
    match value.to_ascii_lowercase().as_str() {
        "additive" => Ok(SeasonalityMode::Additive),
        "multiplicative" => Ok(SeasonalityMode::Multiplicative),
        _ => Err(PyValueError::new_err(format!(
            "invalid seasonality mode '{value}'; expected additive or multiplicative"
        ))),
    }
}

fn seasonality_mode_name(value: SeasonalityMode) -> &'static str {
    match value {
        SeasonalityMode::Additive => "additive",
        SeasonalityMode::Multiplicative => "multiplicative",
    }
}

fn build_yield_curve(nodes: Vec<(f64, f64)>) -> YieldCurve {
    YieldCurve::new(nodes)
}

fn pricing_result_to_dict(py: Python<'_>, result: CorePricingResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("price", result.price)?;
    dict.set_item("stderr", result.stderr)?;

    let diagnostics = PyDict::new(py);
    for (key, value) in result.diagnostics.iter() {
        diagnostics.set_item(key, *value)?;
    }
    dict.set_item("diagnostics", diagnostics)?;
    Ok(dict.unbind())
}

fn vec_to_monthly_factors(values: Vec<f64>) -> PyResult<[f64; 12]> {
    values
        .try_into()
        .map_err(|_| PyValueError::new_err("expected exactly 12 monthly factors"))
}

fn core_option_type(value: &str) -> PyResult<OptionType> {
    parse_option_type(value)
        .ok_or_else(|| PyValueError::new_err(format!("invalid option type '{value}'")))
}

fn build_vanilla_option(option_type: &str, strike: f64, expiry: f64) -> PyResult<VanillaOption> {
    let option_type = core_option_type(option_type)?;
    let instrument = match option_type {
        OptionType::Call => VanillaOption::european_call(strike, expiry),
        OptionType::Put => VanillaOption::european_put(strike, expiry),
    };
    instrument.validate().map_err(string_err)?;
    Ok(instrument)
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Gbm {
    #[pyo3(get, set)]
    pub mu: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl Gbm {
    fn to_core(self) -> CoreGbm {
        CoreGbm {
            mu: self.mu,
            sigma: self.sigma,
        }
    }
}

#[pymethods]
impl Gbm {
    #[new]
    fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    fn drift(&self, s: f64) -> f64 {
        self.to_core().drift(s)
    }

    fn diffusion(&self, s: f64) -> f64 {
        self.to_core().diffusion(s)
    }

    fn step_exact(&self, s: f64, dt: f64, z: f64) -> f64 {
        self.to_core().step_exact(s, dt, z)
    }

    fn step_euler(&self, s: f64, dt: f64, z: f64) -> f64 {
        self.to_core().step_euler(s, dt, z)
    }

    fn __repr__(&self) -> String {
        format!("Gbm(mu={}, sigma={})", self.mu, self.sigma)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Heston {
    #[pyo3(get, set)]
    pub mu: f64,
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub xi: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub v0: f64,
}

impl Heston {
    fn to_core(self) -> CoreHeston {
        CoreHeston {
            mu: self.mu,
            kappa: self.kappa,
            theta: self.theta,
            xi: self.xi,
            rho: self.rho,
            v0: self.v0,
        }
    }
}

#[pymethods]
impl Heston {
    #[new]
    fn new(mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64, v0: f64) -> Self {
        Self {
            mu,
            kappa,
            theta,
            xi,
            rho,
            v0,
        }
    }

    fn validate(&self) -> bool {
        self.to_core().validate()
    }

    fn step_euler(&self, s: f64, v: f64, dt: f64, z1: f64, z2: f64) -> (f64, f64) {
        self.to_core().step_euler(s, v, dt, z1, z2)
    }

    fn __repr__(&self) -> String {
        format!(
            "Heston(mu={}, kappa={}, theta={}, xi={}, rho={}, v0={})",
            self.mu, self.kappa, self.theta, self.xi, self.rho, self.v0
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct Sabr {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub beta: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub nu: f64,
}

impl Sabr {
    fn to_core(self) -> CoreSabr {
        CoreSabr {
            alpha: self.alpha,
            beta: self.beta,
            rho: self.rho,
            nu: self.nu,
        }
    }
}

#[pymethods]
impl Sabr {
    #[new]
    fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
        Self {
            alpha,
            beta,
            rho,
            nu,
        }
    }

    fn validate(&self) -> bool {
        self.to_core().validate()
    }

    fn step_euler(&self, f: f64, alpha_t: f64, dt: f64, z1: f64, z2: f64) -> (f64, f64) {
        self.to_core().step_euler(f, alpha_t, dt, z1, z2)
    }

    fn __repr__(&self) -> String {
        format!(
            "Sabr(alpha={}, beta={}, rho={}, nu={})",
            self.alpha, self.beta, self.rho, self.nu
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct CIR {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl CIR {
    fn to_core(self) -> CoreCIR {
        CoreCIR {
            a: self.a,
            b: self.b,
            sigma: self.sigma,
        }
    }
}

#[pymethods]
impl CIR {
    #[new]
    fn new(a: f64, b: f64, sigma: f64) -> Self {
        Self { a, b, sigma }
    }

    fn bond_price(&self, t: f64, maturity: f64, short_rate: f64) -> f64 {
        self.to_core().bond_price(t, maturity, short_rate)
    }

    fn __repr__(&self) -> String {
        format!("CIR(a={}, b={}, sigma={})", self.a, self.b, self.sigma)
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
    pub(crate) fn to_core(&self) -> CoreHullWhite {
        CoreHullWhite {
            a: self.a,
            sigma: self.sigma,
            theta: self.theta.clone(),
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

    fn calibrate_theta(&mut self, initial_curve: Vec<(f64, f64)>, times: Vec<f64>) {
        let mut model = self.to_core();
        model.calibrate_theta(&build_yield_curve(initial_curve), &times);
        self.theta = model.theta;
    }

    fn theta_at(&self, t: f64) -> f64 {
        self.to_core().theta_at(t)
    }

    fn bond_price(
        &self,
        t: f64,
        maturity: f64,
        short_rate: f64,
        initial_curve: Vec<(f64, f64)>,
    ) -> f64 {
        self.to_core()
            .bond_price(t, maturity, short_rate, &build_yield_curve(initial_curve))
    }

    #[staticmethod]
    fn instantaneous_forward(initial_curve: Vec<(f64, f64)>, t: f64) -> f64 {
        CoreHullWhite::instantaneous_forward(&build_yield_curve(initial_curve), t)
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
pub struct AtmSwaptionVolQuote {
    #[pyo3(get, set)]
    pub expiry: f64,
    #[pyo3(get, set)]
    pub tenor: f64,
    #[pyo3(get, set)]
    pub market_vol: f64,
}

impl AtmSwaptionVolQuote {
    fn to_tuple(self) -> (f64, f64, f64) {
        (self.expiry, self.tenor, self.market_vol)
    }
}

#[pymethods]
impl AtmSwaptionVolQuote {
    #[new]
    fn new(expiry: f64, tenor: f64, market_vol: f64) -> Self {
        Self {
            expiry,
            tenor,
            market_vol,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LmmParams {
    #[pyo3(get, set)]
    pub volatilities: Vec<f64>,
    #[pyo3(get, set)]
    pub correlation: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub tenors: Vec<f64>,
}

impl LmmParams {
    fn to_core(&self) -> CoreLmmParams {
        CoreLmmParams {
            volatilities: self.volatilities.clone(),
            correlation: self.correlation.clone(),
            tenors: self.tenors.clone(),
        }
    }

    fn from_core(params: CoreLmmParams) -> Self {
        Self {
            volatilities: params.volatilities,
            correlation: params.correlation,
            tenors: params.tenors,
        }
    }
}

#[pymethods]
impl LmmParams {
    #[new]
    fn new(volatilities: Vec<f64>, correlation: Vec<Vec<f64>>, tenors: Vec<f64>) -> Self {
        Self {
            volatilities,
            correlation,
            tenors,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "LmmParams(dim={}, tenors={})",
            self.volatilities.len(),
            self.tenors.len()
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LmmModel {
    inner: CoreLmmModel,
}

#[pymethods]
impl LmmModel {
    #[new]
    fn new(params: &LmmParams) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLmmModel::new(params.to_core()).map_err(string_err)?,
        })
    }

    #[getter]
    fn params(&self) -> LmmParams {
        LmmParams::from_core(self.inner.params.clone())
    }

    fn simulate_terminal_forwards(
        &self,
        initial_forwards: Vec<f64>,
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.inner
            .simulate_terminal_forwards(&initial_forwards, horizon, num_steps, num_paths, seed)
            .map_err(string_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn price_european_swaption_mc(
        &self,
        initial_forwards: Vec<f64>,
        strike: f64,
        expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<f64> {
        self.inner
            .price_european_swaption_mc(
                &initial_forwards,
                strike,
                expiry,
                swap_start,
                swap_end,
                is_payer,
                notional,
                num_paths,
                num_steps,
                seed,
            )
            .map_err(string_err)
    }

    fn __repr__(&self) -> String {
        format!("LmmModel(dim={})", self.inner.params.volatilities.len())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SlvParams {
    #[pyo3(get, set)]
    pub v0: f64,
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub xi: f64,
    #[pyo3(get, set)]
    pub rho: f64,
}

impl SlvParams {
    fn to_core(self) -> CoreSlvParams {
        CoreSlvParams {
            v0: self.v0,
            kappa: self.kappa,
            theta: self.theta,
            xi: self.xi,
            rho: self.rho,
        }
    }
}

#[pymethods]
impl SlvParams {
    #[new]
    fn new(v0: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> Self {
        Self {
            v0,
            kappa,
            theta,
            xi,
            rho,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "SlvParams(v0={}, kappa={}, theta={}, xi={}, rho={})",
            self.v0, self.kappa, self.theta, self.xi, self.rho
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LeverageSlice {
    #[pyo3(get, set)]
    pub time: f64,
    #[pyo3(get, set)]
    pub spots: Vec<f64>,
    #[pyo3(get, set)]
    pub leverage: Vec<f64>,
}

impl LeverageSlice {
    fn to_core(&self) -> CoreLeverageSlice {
        CoreLeverageSlice {
            time: self.time,
            spots: self.spots.clone(),
            leverage: self.leverage.clone(),
        }
    }

    fn from_core(slice: CoreLeverageSlice) -> Self {
        Self {
            time: slice.time,
            spots: slice.spots,
            leverage: slice.leverage,
        }
    }
}

#[pymethods]
impl LeverageSlice {
    #[new]
    fn new(time: f64, spots: Vec<f64>, leverage: Vec<f64>) -> Self {
        Self {
            time,
            spots,
            leverage,
        }
    }

    fn value(&self, spot: f64) -> f64 {
        self.to_core().value(spot)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct LeverageSurface {
    inner: CoreLeverageSurface,
}

#[pymethods]
impl LeverageSurface {
    #[new]
    fn new(maturity: f64, n_steps: usize, slices: Vec<LeverageSlice>) -> Self {
        Self {
            inner: CoreLeverageSurface {
                maturity,
                n_steps,
                slices: slices.into_iter().map(|slice| slice.to_core()).collect(),
            },
        }
    }

    #[getter]
    fn maturity(&self) -> f64 {
        self.inner.maturity
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    #[getter]
    fn slices(&self) -> Vec<LeverageSlice> {
        self.inner
            .slices
            .iter()
            .cloned()
            .map(LeverageSlice::from_core)
            .collect()
    }

    fn value(&self, spot: f64, time: f64) -> f64 {
        self.inner.value(spot, time)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct FuturesQuote {
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub price: f64,
}

impl FuturesQuote {
    fn to_core(self) -> CoreFuturesQuote {
        CoreFuturesQuote {
            maturity: self.maturity,
            price: self.price,
        }
    }
}

#[pymethods]
impl FuturesQuote {
    #[new]
    fn new(maturity: f64, price: f64) -> Self {
        Self { maturity, price }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SchwartzOneFactor {
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub mu: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

impl SchwartzOneFactor {
    fn to_core(self) -> CoreSchwartzOneFactor {
        CoreSchwartzOneFactor {
            kappa: self.kappa,
            mu: self.mu,
            sigma: self.sigma,
        }
    }
}

#[pymethods]
impl SchwartzOneFactor {
    #[new]
    fn new(kappa: f64, mu: f64, sigma: f64) -> Self {
        Self { kappa, mu, sigma }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }

    fn long_run_log_mean(&self) -> f64 {
        self.to_core().long_run_log_mean()
    }

    fn step_log_exact(&self, log_spot: f64, dt: f64, z: f64) -> PyResult<f64> {
        self.to_core()
            .step_log_exact(log_spot, dt, z)
            .map_err(string_err)
    }

    fn step_exact(&self, spot: f64, dt: f64, z: f64) -> PyResult<f64> {
        self.to_core().step_exact(spot, dt, z).map_err(string_err)
    }

    fn simulate_path(
        &self,
        initial_spot: f64,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<Vec<f64>> {
        self.to_core()
            .simulate_path(initial_spot, horizon, num_steps, seed)
            .map_err(string_err)
    }

    fn simulate_terminal_spots(
        &self,
        initial_spot: f64,
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> PyResult<Vec<f64>> {
        self.to_core()
            .simulate_terminal_spots(initial_spot, horizon, num_steps, num_paths, seed)
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct SchwartzSmithTwoFactor {
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub sigma_chi: f64,
    #[pyo3(get, set)]
    pub mu_xi: f64,
    #[pyo3(get, set)]
    pub sigma_xi: f64,
    #[pyo3(get, set)]
    pub rho: f64,
}

impl SchwartzSmithTwoFactor {
    fn to_core(self) -> CoreSchwartzSmithTwoFactor {
        CoreSchwartzSmithTwoFactor {
            kappa: self.kappa,
            sigma_chi: self.sigma_chi,
            mu_xi: self.mu_xi,
            sigma_xi: self.sigma_xi,
            rho: self.rho,
        }
    }
}

#[pymethods]
impl SchwartzSmithTwoFactor {
    #[new]
    fn new(kappa: f64, sigma_chi: f64, mu_xi: f64, sigma_xi: f64, rho: f64) -> Self {
        Self {
            kappa,
            sigma_chi,
            mu_xi,
            sigma_xi,
            rho,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }

    fn step_exact(&self, chi: f64, xi: f64, dt: f64, z1: f64, z2: f64) -> PyResult<(f64, f64)> {
        self.to_core()
            .step_exact(chi, xi, dt, z1, z2)
            .map_err(string_err)
    }

    #[staticmethod]
    fn spot_from_factors(chi: f64, xi: f64) -> f64 {
        CoreSchwartzSmithTwoFactor::spot_from_factors(chi, xi)
    }

    fn simulate_path(
        &self,
        initial_chi: f64,
        initial_xi: f64,
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        self.to_core()
            .simulate_path(initial_chi, initial_xi, horizon, num_steps, seed)
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CommodityForwardCurve {
    inner: CoreCommodityForwardCurve,
}

#[pymethods]
impl CommodityForwardCurve {
    #[new]
    #[pyo3(signature=(quotes, interpolation="linear"))]
    fn new(quotes: Vec<FuturesQuote>, interpolation: &str) -> PyResult<Self> {
        let quotes: Vec<_> = quotes.into_iter().map(FuturesQuote::to_core).collect();
        Ok(Self {
            inner: CoreCommodityForwardCurve::from_futures_quotes_with_interpolation(
                &quotes,
                parse_forward_interpolation(interpolation)?,
            )
            .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn from_futures_quotes(quotes: Vec<FuturesQuote>) -> PyResult<Self> {
        let quotes: Vec<_> = quotes.into_iter().map(FuturesQuote::to_core).collect();
        Ok(Self {
            inner: CoreCommodityForwardCurve::from_futures_quotes(&quotes).map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn bootstrap_from_futures(quotes: Vec<FuturesQuote>) -> PyResult<Self> {
        let quotes: Vec<_> = quotes.into_iter().map(FuturesQuote::to_core).collect();
        Ok(Self {
            inner: CoreCommodityForwardCurve::bootstrap_from_futures(&quotes)
                .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn from_futures_quotes_with_interpolation(
        quotes: Vec<FuturesQuote>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let quotes: Vec<_> = quotes.into_iter().map(FuturesQuote::to_core).collect();
        Ok(Self {
            inner: CoreCommodityForwardCurve::from_futures_quotes_with_interpolation(
                &quotes,
                parse_forward_interpolation(interpolation)?,
            )
            .map_err(string_err)?,
        })
    }

    fn with_interpolation(&self, interpolation: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .with_interpolation(parse_forward_interpolation(interpolation)?),
        })
    }

    fn maturities(&self) -> Vec<f64> {
        self.inner.maturities().to_vec()
    }

    fn forwards(&self) -> Vec<f64> {
        self.inner.forwards().to_vec()
    }

    #[getter]
    fn interpolation(&self) -> &'static str {
        forward_interpolation_name(self.inner.interpolation())
    }

    fn forward(&self, maturity: f64) -> f64 {
        self.inner.forward(maturity)
    }

    fn forward_with_method(&self, maturity: f64, method: &str) -> PyResult<f64> {
        Ok(self
            .inner
            .forward_with_method(maturity, parse_forward_interpolation(method)?))
    }

    #[getter]
    fn structure(&self) -> &'static str {
        curve_structure_name(self.inner.structure())
    }

    fn is_contango(&self) -> bool {
        self.inner.is_contango()
    }

    fn is_backwardation(&self) -> bool {
        self.inner.is_backwardation()
    }

    fn convenience_yield_curve(
        &self,
        spot: f64,
        risk_free_rate: f64,
        storage_cost: f64,
    ) -> PyResult<Vec<(f64, f64)>> {
        self.inner
            .convenience_yield_curve(spot, risk_free_rate, storage_cost)
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CommoditySeasonalityModel {
    inner: CoreCommoditySeasonalityModel,
}

#[pymethods]
impl CommoditySeasonalityModel {
    #[new]
    fn new(mode: &str, monthly_factors: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCommoditySeasonalityModel::from_monthly_factors(
                parse_seasonality_mode(mode)?,
                vec_to_monthly_factors(monthly_factors)?,
            )
            .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn additive(monthly_additives: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCommoditySeasonalityModel::additive(vec_to_monthly_factors(
                monthly_additives,
            )?)
            .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn multiplicative(monthly_multipliers: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCommoditySeasonalityModel::multiplicative(vec_to_monthly_factors(
                monthly_multipliers,
            )?)
            .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn natural_gas_winter_summer(
        mode: &str,
        winter_factor: f64,
        summer_factor: f64,
        shoulder_factor: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCommoditySeasonalityModel::natural_gas_winter_summer(
                parse_seasonality_mode(mode)?,
                winter_factor,
                summer_factor,
                shoulder_factor,
            )
            .map_err(string_err)?,
        })
    }

    #[staticmethod]
    fn default_natural_gas() -> Self {
        Self {
            inner: CoreCommoditySeasonalityModel::default_natural_gas(),
        }
    }

    #[getter]
    fn mode(&self) -> &'static str {
        seasonality_mode_name(self.inner.mode())
    }

    fn monthly_factors(&self) -> Vec<f64> {
        self.inner.monthly_factors().to_vec()
    }

    fn factor_for_month(&self, month: u32) -> PyResult<f64> {
        self.inner.factor_for_month(month).map_err(string_err)
    }

    fn apply(&self, base_level: f64, month: u32) -> PyResult<f64> {
        self.inner.apply(base_level, month).map_err(string_err)
    }

    fn deseasonalise(&self, observations: Vec<(u32, f64)>) -> PyResult<Vec<f64>> {
        self.inner.deseasonalise(&observations).map_err(string_err)
    }

    fn deseasonalised_log_returns(&self, observations: Vec<(u32, f64)>) -> PyResult<Vec<f64>> {
        self.inner
            .deseasonalised_log_returns(&observations)
            .map_err(string_err)
    }

    fn estimate_deseasonalised_volatility(
        &self,
        observations: Vec<(u32, f64)>,
        observations_per_year: f64,
    ) -> PyResult<f64> {
        self.inner
            .estimate_deseasonalised_volatility(&observations, observations_per_year)
            .map_err(string_err)
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
    pub(crate) fn to_core(self) -> CoreTwoFactorCommodityProcess {
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
        self.to_core().validate().map_err(string_err)
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
    pub(crate) fn to_core(self) -> CoreTwoFactorSpreadModel {
        CoreTwoFactorSpreadModel {
            leg_1: self.leg_1.to_core(),
            leg_2: self.leg_2.to_core(),
            rho_fast: self.rho_fast,
            rho_slow: self.rho_slow,
        }
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
        self.to_core().validate().map_err(string_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn price_spread_option_mc(
        &self,
        option_type: &str,
        forward_1: f64,
        forward_2: f64,
        strike: f64,
        quantity_1: f64,
        quantity_2: f64,
        risk_free_rate: f64,
        maturity: f64,
        num_paths: usize,
        seed: u64,
    ) -> PyResult<(f64, f64)> {
        self.to_core()
            .price_spread_option_mc(
                core_option_type(option_type)?,
                forward_1,
                forward_2,
                strike,
                quantity_1,
                quantity_2,
                risk_free_rate,
                maturity,
                num_paths,
                seed,
            )
            .map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CommodityStorageContract {
    #[pyo3(get, set)]
    pub decision_times: Vec<f64>,
    #[pyo3(get, set)]
    pub min_inventory: f64,
    #[pyo3(get, set)]
    pub max_inventory: f64,
    #[pyo3(get, set)]
    pub initial_inventory: f64,
    #[pyo3(get, set)]
    pub max_injection: f64,
    #[pyo3(get, set)]
    pub max_withdrawal: f64,
    #[pyo3(get, set)]
    pub variable_cost: f64,
    #[pyo3(get, set)]
    pub terminal_inventory_target: Option<f64>,
}

impl CommodityStorageContract {
    fn to_core(&self) -> CoreCommodityStorageContract {
        CoreCommodityStorageContract {
            decision_times: self.decision_times.clone(),
            min_inventory: self.min_inventory,
            max_inventory: self.max_inventory,
            initial_inventory: self.initial_inventory,
            max_injection: self.max_injection,
            max_withdrawal: self.max_withdrawal,
            variable_cost: self.variable_cost,
            terminal_inventory_target: self.terminal_inventory_target,
        }
    }
}

#[pymethods]
impl CommodityStorageContract {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        decision_times: Vec<f64>,
        min_inventory: f64,
        max_inventory: f64,
        initial_inventory: f64,
        max_injection: f64,
        max_withdrawal: f64,
        variable_cost: f64,
        terminal_inventory_target: Option<f64>,
    ) -> Self {
        Self {
            decision_times,
            min_inventory,
            max_inventory,
            initial_inventory,
            max_injection,
            max_withdrawal,
            variable_cost,
            terminal_inventory_target,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct StorageLsmConfig {
    #[pyo3(get, set)]
    pub num_paths: usize,
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub seed: u64,
}

impl StorageLsmConfig {
    fn to_core(self) -> CoreStorageLsmConfig {
        CoreStorageLsmConfig {
            num_paths: self.num_paths,
            kappa: self.kappa,
            sigma: self.sigma,
            seed: self.seed,
        }
    }
}

#[pymethods]
impl StorageLsmConfig {
    #[new]
    fn new(num_paths: usize, kappa: f64, sigma: f64, seed: u64) -> Self {
        Self {
            num_paths,
            kappa,
            sigma,
            seed,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core().validate().map_err(string_err)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct StorageValuation {
    #[pyo3(get, set)]
    pub intrinsic: f64,
    #[pyo3(get, set)]
    pub extrinsic: f64,
    #[pyo3(get, set)]
    pub total: f64,
    #[pyo3(get, set)]
    pub stderr: f64,
}

impl StorageValuation {
    fn from_core(value: CoreStorageValuation) -> Self {
        Self {
            intrinsic: value.intrinsic,
            extrinsic: value.extrinsic,
            total: value.total,
            stderr: value.stderr,
        }
    }
}

#[pymethods]
impl StorageValuation {
    #[new]
    fn new(intrinsic: f64, extrinsic: f64, total: f64, stderr: f64) -> Self {
        Self {
            intrinsic,
            extrinsic,
            total,
            stderr,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct VolumeConstrainedSwing {
    #[pyo3(get, set)]
    pub exercise_times: Vec<f64>,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub option_type: String,
    #[pyo3(get, set)]
    pub min_period_volume: f64,
    #[pyo3(get, set)]
    pub max_period_volume: f64,
    #[pyo3(get, set)]
    pub min_total_volume: f64,
    #[pyo3(get, set)]
    pub max_total_volume: f64,
}

impl VolumeConstrainedSwing {
    fn to_core(&self) -> PyResult<CoreVolumeConstrainedSwing> {
        Ok(CoreVolumeConstrainedSwing {
            exercise_times: self.exercise_times.clone(),
            strike: self.strike,
            option_type: core_option_type(&self.option_type)?,
            min_period_volume: self.min_period_volume,
            max_period_volume: self.max_period_volume,
            min_total_volume: self.min_total_volume,
            max_total_volume: self.max_total_volume,
        })
    }
}

#[pymethods]
impl VolumeConstrainedSwing {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        exercise_times: Vec<f64>,
        strike: f64,
        option_type: String,
        min_period_volume: f64,
        max_period_volume: f64,
        min_total_volume: f64,
        max_total_volume: f64,
    ) -> Self {
        Self {
            exercise_times,
            strike,
            option_type,
            min_period_volume,
            max_period_volume,
            min_total_volume,
            max_total_volume,
        }
    }

    fn validate(&self) -> PyResult<()> {
        self.to_core()?.validate().map_err(string_err)
    }

    fn intrinsic_value(
        &self,
        curve: &CommodityForwardCurve,
        risk_free_rate: f64,
        total_volume_grid: usize,
    ) -> PyResult<f64> {
        self.to_core()?
            .intrinsic_value(&curve.inner, risk_free_rate, total_volume_grid)
            .map_err(string_err)
    }
}

#[pyfunction]
pub fn hw_atm_swaption_vol_approx_py(a: f64, sigma: f64, expiry: f64, tenor: f64) -> f64 {
    hw_atm_swaption_vol_approx(a, sigma, expiry, tenor)
}

#[pyfunction]
pub fn calibrate_hull_white_params_py(quotes: Vec<AtmSwaptionVolQuote>) -> Option<(f64, f64)> {
    let quotes: Vec<_> = quotes
        .into_iter()
        .map(AtmSwaptionVolQuote::to_tuple)
        .collect();
    calibrate_hull_white_params(&quotes)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn black_swaption_price_py(
    notional: f64,
    forward_swap_rate: f64,
    strike: f64,
    annuity: f64,
    vol: f64,
    expiry: f64,
    is_payer: bool,
) -> f64 {
    black_swaption_price(
        notional,
        forward_swap_rate,
        strike,
        annuity,
        vol,
        expiry,
        is_payer,
    )
}

#[pyfunction]
pub fn initial_swap_rate_annuity_py(
    initial_forwards: Vec<f64>,
    tenors: Vec<f64>,
    swap_start: f64,
    swap_end: f64,
) -> Option<(f64, f64)> {
    initial_swap_rate_annuity(&initial_forwards, &tenors, swap_start, swap_end)
}

#[pyfunction]
pub fn fbm_covariance_py(s: f64, t: f64, hurst: f64) -> f64 {
    fbm_covariance(s, t, hurst)
}

#[pyfunction]
pub fn fbm_path_cholesky_py(
    hurst: f64,
    maturity: f64,
    n_steps: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    fbm_path_cholesky(hurst, maturity, n_steps, seed).map_err(string_err)
}

#[pyfunction]
pub fn fbm_path_hybrid_py(
    hurst: f64,
    maturity: f64,
    n_steps: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    fbm_path_hybrid(hurst, maturity, n_steps, seed).map_err(string_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rbergomi_european_mc_py(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<Py<PyDict>> {
    pricing_result_to_dict(
        py,
        rbergomi_european_mc(
            spot, strike, r, q, maturity, hurst, eta, rho, xi0, n_paths, n_steps,
        ),
    )
}

#[pyfunction]
pub fn rbergomi_implied_vol_surface_py(
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    expiries: Vec<f64>,
    strikes: Vec<f64>,
) -> Vec<Vec<f64>> {
    rbergomi_implied_vol_surface(hurst, eta, rho, xi0, &expiries, &strikes)
}

#[pyfunction]
pub fn nadaraya_watson_conditional_mean_py(
    x: f64,
    sample_x: Vec<f64>,
    sample_y: Vec<f64>,
    bandwidth: f64,
    fallback_mean: f64,
) -> f64 {
    nadaraya_watson_conditional_mean(x, &sample_x, &sample_y, bandwidth, fallback_mean)
}

#[pyfunction]
pub fn calibrate_leverage_surface_py(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    params: &SlvParams,
    maturity: f64,
    n_particles: usize,
    n_steps: usize,
) -> PyResult<LeverageSurface> {
    let market = build_market(spot, rate, dividend_yield, vol)
        .ok_or_else(|| PyValueError::new_err("failed to build market"))?;
    Ok(LeverageSurface {
        inner: calibrate_leverage_surface(
            &market,
            params.to_core(),
            maturity,
            n_particles,
            n_steps,
        )
        .map_err(string_err)?,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn slv_mc_price_py(
    py: Python<'_>,
    option_type: &str,
    strike: f64,
    expiry: f64,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    params: &SlvParams,
    n_particles: usize,
    n_steps: usize,
) -> PyResult<Py<PyDict>> {
    let market = build_market(spot, rate, dividend_yield, vol)
        .ok_or_else(|| PyValueError::new_err("failed to build market"))?;
    let instrument = build_vanilla_option(option_type, strike, expiry)?;
    pricing_result_to_dict(
        py,
        slv_mc_price(&instrument, &market, params.to_core(), n_particles, n_steps),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn slv_mc_price_checked_py(
    py: Python<'_>,
    option_type: &str,
    strike: f64,
    expiry: f64,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    params: &SlvParams,
    n_particles: usize,
    n_steps: usize,
) -> PyResult<Py<PyDict>> {
    let market = build_market(spot, rate, dividend_yield, vol)
        .ok_or_else(|| PyValueError::new_err("failed to build market"))?;
    let instrument = build_vanilla_option(option_type, strike, expiry)?;
    let result = slv_mc_price_checked(&instrument, &market, params.to_core(), n_particles, n_steps)
        .map_err(string_err)?;
    pricing_result_to_dict(py, result)
}

#[pyfunction]
pub fn implied_convenience_yield_py(
    spot: f64,
    futures_price: f64,
    risk_free_rate: f64,
    storage_cost: f64,
    maturity: f64,
) -> Option<f64> {
    implied_convenience_yield(spot, futures_price, risk_free_rate, storage_cost, maturity)
}

#[pyfunction]
pub fn convenience_yield_from_term_structure_py(
    spot: f64,
    quotes: Vec<FuturesQuote>,
    risk_free_rate: f64,
    storage_cost: f64,
) -> PyResult<Vec<(f64, f64)>> {
    let quotes: Vec<_> = quotes.into_iter().map(FuturesQuote::to_core).collect();
    convenience_yield_from_term_structure(spot, &quotes, risk_free_rate, storage_cost)
        .map_err(string_err)
}

#[pyfunction]
pub fn value_storage_intrinsic_extrinsic_py(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    risk_free_rate: f64,
    inventory_grid: usize,
    lsm_config: &StorageLsmConfig,
) -> PyResult<StorageValuation> {
    value_storage_intrinsic_extrinsic(
        &contract.to_core(),
        &curve.inner,
        risk_free_rate,
        inventory_grid,
        lsm_config.to_core(),
    )
    .map(StorageValuation::from_core)
    .map_err(string_err)
}

#[pyfunction]
pub fn intrinsic_storage_value_py(
    contract: &CommodityStorageContract,
    curve: &CommodityForwardCurve,
    risk_free_rate: f64,
    inventory_grid: usize,
) -> PyResult<f64> {
    intrinsic_storage_value(
        &contract.to_core(),
        &curve.inner,
        risk_free_rate,
        inventory_grid,
    )
    .map_err(string_err)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hw_atm_swaption_vol_approx_py, module)?)?;
    module.add_function(wrap_pyfunction!(calibrate_hull_white_params_py, module)?)?;
    module.add_function(wrap_pyfunction!(black_swaption_price_py, module)?)?;
    module.add_function(wrap_pyfunction!(initial_swap_rate_annuity_py, module)?)?;
    module.add_function(wrap_pyfunction!(fbm_covariance_py, module)?)?;
    module.add_function(wrap_pyfunction!(fbm_path_cholesky_py, module)?)?;
    module.add_function(wrap_pyfunction!(fbm_path_hybrid_py, module)?)?;
    module.add_function(wrap_pyfunction!(rbergomi_european_mc_py, module)?)?;
    module.add_function(wrap_pyfunction!(rbergomi_implied_vol_surface_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        nadaraya_watson_conditional_mean_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(calibrate_leverage_surface_py, module)?)?;
    module.add_function(wrap_pyfunction!(slv_mc_price_py, module)?)?;
    module.add_function(wrap_pyfunction!(slv_mc_price_checked_py, module)?)?;
    module.add_function(wrap_pyfunction!(implied_convenience_yield_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        convenience_yield_from_term_structure_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        value_storage_intrinsic_extrinsic_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(intrinsic_storage_value_py, module)?)?;

    module.add_class::<Gbm>()?;
    module.add_class::<Heston>()?;
    module.add_class::<Sabr>()?;
    module.add_class::<CIR>()?;
    module.add_class::<HullWhite>()?;
    module.add_class::<AtmSwaptionVolQuote>()?;
    module.add_class::<LmmParams>()?;
    module.add_class::<LmmModel>()?;
    module.add_class::<SlvParams>()?;
    module.add_class::<LeverageSlice>()?;
    module.add_class::<LeverageSurface>()?;
    module.add_class::<FuturesQuote>()?;
    module.add_class::<SchwartzOneFactor>()?;
    module.add_class::<SchwartzSmithTwoFactor>()?;
    module.add_class::<CommodityForwardCurve>()?;
    module.add_class::<CommoditySeasonalityModel>()?;
    module.add_class::<TwoFactorCommodityProcess>()?;
    module.add_class::<TwoFactorSpreadModel>()?;
    module.add_class::<CommodityStorageContract>()?;
    module.add_class::<StorageLsmConfig>()?;
    module.add_class::<StorageValuation>()?;
    module.add_class::<VolumeConstrainedSwing>()?;
    Ok(())
}
