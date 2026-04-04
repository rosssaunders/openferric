use chrono::NaiveDate;
use openferric_core::credit::cds_option::CdsOption as CoreCdsOption;
use openferric_core::credit::{
    BasketDefaultSimulation as CoreBasketDefaultSimulation, CdoTranche as CoreCdoTranche,
    Cds as CoreCds, CdsDateRule as CoreCdsDateRule, CdsIndex as CoreCdsIndex,
    CdsPriceResult as CoreCdsPriceResult, DatedCds as CoreDatedCds,
    GaussianCopula as CoreGaussianCopula, IsdaConventions as CoreIsdaConventions,
    NthToDefaultBasket as CoreNthToDefaultBasket, ProtectionSide as CoreProtectionSide,
    SurvivalCurve as CoreSurvivalCurve, SyntheticCdo as CoreSyntheticCdo, price_isda_flat,
    price_midpoint_flat,
};
use openferric_core::rates::YieldCurve;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::helpers::tenor_grid;

fn parse_naive_date(value: &str) -> PyResult<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|err| {
        PyValueError::new_err(format!(
            "invalid date '{value}'; expected YYYY-MM-DD ({err})"
        ))
    })
}

fn format_naive_date(value: NaiveDate) -> String {
    value.format("%Y-%m-%d").to_string()
}

fn yield_curve_from_nodes(nodes: &[(f64, f64)]) -> YieldCurve {
    YieldCurve::new(nodes.to_vec())
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct Cds {
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub spread: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
    #[pyo3(get, set)]
    pub payment_freq: usize,
}

impl Cds {
    fn to_core(self) -> CoreCds {
        CoreCds {
            notional: self.notional,
            spread: self.spread,
            maturity: self.maturity,
            recovery_rate: self.recovery_rate,
            payment_freq: self.payment_freq,
        }
    }

    #[allow(dead_code)]
    fn from_core(value: CoreCds) -> Self {
        Self {
            notional: value.notional,
            spread: value.spread,
            maturity: value.maturity,
            recovery_rate: value.recovery_rate,
            payment_freq: value.payment_freq,
        }
    }
}

#[pymethods]
impl Cds {
    #[new]
    fn new(
        notional: f64,
        spread: f64,
        maturity: f64,
        recovery_rate: f64,
        payment_freq: usize,
    ) -> Self {
        Self {
            notional,
            spread,
            maturity,
            recovery_rate,
            payment_freq,
        }
    }

    fn premium_leg_pv(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curve: &SurvivalCurve,
    ) -> f64 {
        self.to_core().premium_leg_pv(
            &yield_curve_from_nodes(&discount_curve_nodes),
            &survival_curve.to_core(),
        )
    }

    fn protection_leg_pv(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curve: &SurvivalCurve,
    ) -> f64 {
        self.to_core().protection_leg_pv(
            &yield_curve_from_nodes(&discount_curve_nodes),
            &survival_curve.to_core(),
        )
    }

    fn npv(&self, discount_curve_nodes: Vec<(f64, f64)>, survival_curve: &SurvivalCurve) -> f64 {
        self.to_core().npv(
            &yield_curve_from_nodes(&discount_curve_nodes),
            &survival_curve.to_core(),
        )
    }

    fn fair_spread(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curve: &SurvivalCurve,
    ) -> f64 {
        self.to_core().fair_spread(
            &yield_curve_from_nodes(&discount_curve_nodes),
            &survival_curve.to_core(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Cds(notional={}, spread={}, maturity={}, recovery_rate={}, payment_freq={})",
            self.notional, self.spread, self.maturity, self.recovery_rate, self.payment_freq
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct SurvivalCurve {
    #[pyo3(get, set)]
    pub tenors: Vec<(f64, f64)>,
}

impl SurvivalCurve {
    pub(crate) fn to_core(&self) -> CoreSurvivalCurve {
        CoreSurvivalCurve::new(self.tenors.clone())
    }

    pub(crate) fn from_core(value: CoreSurvivalCurve) -> Self {
        Self {
            tenors: value.tenors,
        }
    }
}

#[pymethods]
impl SurvivalCurve {
    #[new]
    fn new(tenors: Vec<(f64, f64)>) -> Self {
        Self::from_core(CoreSurvivalCurve::new(tenors))
    }

    #[staticmethod]
    fn from_piecewise_hazard(tenors: Vec<f64>, hazards: Vec<f64>) -> Self {
        Self::from_core(CoreSurvivalCurve::from_piecewise_hazard(&tenors, &hazards))
    }

    #[staticmethod]
    fn bootstrap_from_cds_spreads(
        cds_spreads: Vec<(f64, f64)>,
        recovery_rate: f64,
        payment_freq: usize,
        discount_curve_nodes: Vec<(f64, f64)>,
    ) -> Self {
        Self::from_core(CoreSurvivalCurve::bootstrap_from_cds_spreads(
            &cds_spreads,
            recovery_rate,
            payment_freq,
            &yield_curve_from_nodes(&discount_curve_nodes),
        ))
    }

    fn survival_prob(&self, t: f64) -> f64 {
        self.to_core().survival_prob(t)
    }

    fn hazard_rate(&self, t: f64) -> f64 {
        self.to_core().hazard_rate(t)
    }

    fn default_prob(&self, t1: f64, t2: f64) -> f64 {
        self.to_core().default_prob(t1, t2)
    }

    fn inverse_survival_prob(&self, p: f64) -> f64 {
        self.to_core().inverse_survival_prob(p)
    }

    fn __repr__(&self) -> String {
        format!("SurvivalCurve(tenors={:?})", self.tenors)
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct CdoTranche {
    #[pyo3(get, set)]
    pub attachment: f64,
    #[pyo3(get, set)]
    pub detachment: f64,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub spread: f64,
}

impl CdoTranche {
    fn to_core(self) -> CoreCdoTranche {
        CoreCdoTranche {
            attachment: self.attachment,
            detachment: self.detachment,
            notional: self.notional,
            spread: self.spread,
        }
    }
}

#[pymethods]
impl CdoTranche {
    #[new]
    fn new(attachment: f64, detachment: f64, notional: f64, spread: f64) -> Self {
        Self {
            attachment,
            detachment,
            notional,
            spread,
        }
    }

    fn width(&self) -> f64 {
        self.to_core().width()
    }

    fn expected_loss_fraction(
        &self,
        default_probability: f64,
        recovery_rate: f64,
        correlation: f64,
    ) -> f64 {
        self.to_core()
            .expected_loss_fraction(default_probability, recovery_rate, correlation)
    }

    fn __repr__(&self) -> String {
        format!(
            "CdoTranche(attachment={}, detachment={}, notional={}, spread={})",
            self.attachment, self.detachment, self.notional, self.spread
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct SyntheticCdo {
    #[pyo3(get, set)]
    pub num_names: usize,
    #[pyo3(get, set)]
    pub pool_spread: f64,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
    #[pyo3(get, set)]
    pub correlation: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub payment_freq: usize,
}

impl SyntheticCdo {
    fn to_core(self) -> CoreSyntheticCdo {
        CoreSyntheticCdo {
            num_names: self.num_names,
            pool_spread: self.pool_spread,
            recovery_rate: self.recovery_rate,
            correlation: self.correlation,
            risk_free_rate: self.risk_free_rate,
            maturity: self.maturity,
            payment_freq: self.payment_freq,
        }
    }
}

#[pymethods]
impl SyntheticCdo {
    #[new]
    fn new(
        num_names: usize,
        pool_spread: f64,
        recovery_rate: f64,
        correlation: f64,
        risk_free_rate: f64,
        maturity: f64,
        payment_freq: usize,
    ) -> Self {
        Self {
            num_names,
            pool_spread,
            recovery_rate,
            correlation,
            risk_free_rate,
            maturity,
            payment_freq,
        }
    }

    fn hazard_rate(&self) -> f64 {
        self.to_core().hazard_rate()
    }

    fn default_probability(&self, t: f64) -> f64 {
        self.to_core().default_probability(t)
    }

    fn portfolio_expected_loss(&self, t: f64) -> f64 {
        self.to_core().portfolio_expected_loss(t)
    }

    fn expected_tranche_loss(&self, tranche: &CdoTranche, t: f64) -> f64 {
        self.to_core().expected_tranche_loss(&tranche.to_core(), t)
    }

    fn protection_leg_pv(&self, tranche: &CdoTranche) -> f64 {
        self.to_core().protection_leg_pv(&tranche.to_core())
    }

    fn premium_leg_pv(&self, tranche: &CdoTranche, spread: f64) -> f64 {
        self.to_core().premium_leg_pv(&tranche.to_core(), spread)
    }

    fn fair_spread(&self, tranche: &CdoTranche) -> f64 {
        self.to_core().fair_spread(&tranche.to_core())
    }

    fn npv(&self, tranche: &CdoTranche) -> f64 {
        self.to_core().npv(&tranche.to_core())
    }

    fn __repr__(&self) -> String {
        format!(
            "SyntheticCdo(num_names={}, pool_spread={}, recovery_rate={}, correlation={}, risk_free_rate={}, maturity={}, payment_freq={})",
            self.num_names,
            self.pool_spread,
            self.recovery_rate,
            self.correlation,
            self.risk_free_rate,
            self.maturity,
            self.payment_freq
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct CdsOption {
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub strike_spread: f64,
    #[pyo3(get, set)]
    pub option_expiry: f64,
    #[pyo3(get, set)]
    pub cds_maturity: f64,
    #[pyo3(get, set)]
    pub is_payer: bool,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
}

impl CdsOption {
    fn to_core(self) -> CoreCdsOption {
        CoreCdsOption {
            notional: self.notional,
            strike_spread: self.strike_spread,
            option_expiry: self.option_expiry,
            cds_maturity: self.cds_maturity,
            is_payer: self.is_payer,
            recovery_rate: self.recovery_rate,
        }
    }
}

#[pymethods]
impl CdsOption {
    #[new]
    fn new(
        notional: f64,
        strike_spread: f64,
        option_expiry: f64,
        cds_maturity: f64,
        is_payer: bool,
        recovery_rate: f64,
    ) -> Self {
        Self {
            notional,
            strike_spread,
            option_expiry,
            cds_maturity,
            is_payer,
            recovery_rate,
        }
    }

    fn black_price(&self, forward_spread: f64, vol: f64, risky_annuity: f64) -> f64 {
        self.to_core()
            .black_price(forward_spread, vol, risky_annuity)
    }

    fn __repr__(&self) -> String {
        format!(
            "CdsOption(notional={}, strike_spread={}, option_expiry={}, cds_maturity={}, is_payer={}, recovery_rate={})",
            self.notional,
            self.strike_spread,
            self.option_expiry,
            self.cds_maturity,
            self.is_payer,
            self.recovery_rate
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct CdsIndex {
    #[pyo3(get, set)]
    pub constituents: Vec<Cds>,
    #[pyo3(get, set)]
    pub weights: Vec<f64>,
}

impl CdsIndex {
    fn to_core(&self) -> CoreCdsIndex {
        CoreCdsIndex {
            constituents: self
                .constituents
                .iter()
                .copied()
                .map(Cds::to_core)
                .collect(),
            weights: self.weights.clone(),
        }
    }
}

#[pymethods]
impl CdsIndex {
    #[new]
    fn new(constituents: Vec<Cds>, weights: Vec<f64>) -> Self {
        Self {
            constituents,
            weights,
        }
    }

    fn npv(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curves: Vec<SurvivalCurve>,
    ) -> f64 {
        let curves = survival_curves
            .iter()
            .map(SurvivalCurve::to_core)
            .collect::<Vec<_>>();
        self.to_core()
            .npv(&yield_curve_from_nodes(&discount_curve_nodes), &curves)
    }

    fn fair_spread(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curves: Vec<SurvivalCurve>,
    ) -> f64 {
        let curves = survival_curves
            .iter()
            .map(SurvivalCurve::to_core)
            .collect::<Vec<_>>();
        self.to_core()
            .fair_spread(&yield_curve_from_nodes(&discount_curve_nodes), &curves)
    }

    fn __repr__(&self) -> String {
        format!(
            "CdsIndex(constituents={}, weights={:?})",
            self.constituents.len(),
            self.weights
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct NthToDefaultBasket {
    #[pyo3(get, set)]
    pub n: usize,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub maturity: f64,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
    #[pyo3(get, set)]
    pub payment_freq: usize,
}

impl NthToDefaultBasket {
    fn to_core(self) -> CoreNthToDefaultBasket {
        CoreNthToDefaultBasket {
            n: self.n,
            notional: self.notional,
            maturity: self.maturity,
            recovery_rate: self.recovery_rate,
            payment_freq: self.payment_freq,
        }
    }
}

#[pymethods]
impl NthToDefaultBasket {
    #[new]
    fn new(
        n: usize,
        notional: f64,
        maturity: f64,
        recovery_rate: f64,
        payment_freq: usize,
    ) -> Self {
        Self {
            n,
            notional,
            maturity,
            recovery_rate,
            payment_freq,
        }
    }

    fn fair_spread(
        &self,
        discount_curve_nodes: Vec<(f64, f64)>,
        survival_curves: Vec<SurvivalCurve>,
    ) -> f64 {
        let curves = survival_curves
            .iter()
            .map(SurvivalCurve::to_core)
            .collect::<Vec<_>>();
        self.to_core()
            .fair_spread(&yield_curve_from_nodes(&discount_curve_nodes), &curves)
    }

    fn __repr__(&self) -> String {
        format!(
            "NthToDefaultBasket(n={}, notional={}, maturity={}, recovery_rate={}, payment_freq={})",
            self.n, self.notional, self.maturity, self.recovery_rate, self.payment_freq
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct BasketDefaultSimulation {
    #[pyo3(get, set)]
    pub default_times: Vec<f64>,
    #[pyo3(get, set)]
    pub market_factor: f64,
    #[pyo3(get, set)]
    pub latent_variables: Vec<f64>,
}

impl BasketDefaultSimulation {
    fn from_core(value: CoreBasketDefaultSimulation) -> Self {
        Self {
            default_times: value.default_times,
            market_factor: value.market_factor,
            latent_variables: value.latent_variables,
        }
    }
}

#[pymethods]
impl BasketDefaultSimulation {
    #[new]
    fn new(default_times: Vec<f64>, market_factor: f64, latent_variables: Vec<f64>) -> Self {
        Self {
            default_times,
            market_factor,
            latent_variables,
        }
    }

    fn defaults_by(&self, horizon: f64) -> usize {
        self.default_times
            .iter()
            .filter(|&&tau| tau <= horizon)
            .count()
    }

    fn __repr__(&self) -> String {
        format!(
            "BasketDefaultSimulation(default_times={:?}, market_factor={}, latent_variables={:?})",
            self.default_times, self.market_factor, self.latent_variables
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct GaussianCopula {
    #[pyo3(get, set)]
    pub rho: f64,
}

impl GaussianCopula {
    fn to_core(self) -> CoreGaussianCopula {
        CoreGaussianCopula::new(self.rho)
    }
}

#[pymethods]
impl GaussianCopula {
    #[new]
    fn new(rho: f64) -> Self {
        let inner = CoreGaussianCopula::new(rho);
        Self { rho: inner.rho }
    }

    fn simulate_homogeneous(
        &self,
        num_names: usize,
        survival_curve: &SurvivalCurve,
        seed: Option<u64>,
    ) -> BasketDefaultSimulation {
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
        BasketDefaultSimulation::from_core(self.to_core().simulate_homogeneous(
            num_names,
            &survival_curve.to_core(),
            &mut rng,
        ))
    }

    fn simulate(
        &self,
        survival_curves: Vec<SurvivalCurve>,
        seed: Option<u64>,
    ) -> BasketDefaultSimulation {
        let curves = survival_curves
            .iter()
            .map(SurvivalCurve::to_core)
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
        BasketDefaultSimulation::from_core(self.to_core().simulate(&curves, &mut rng))
    }

    fn __repr__(&self) -> String {
        format!("GaussianCopula(rho={})", self.rho)
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProtectionSide {
    Buyer,
    Seller,
}

impl ProtectionSide {
    fn to_core(self) -> CoreProtectionSide {
        match self {
            Self::Buyer => CoreProtectionSide::Buyer,
            Self::Seller => CoreProtectionSide::Seller,
        }
    }

    fn from_core(value: CoreProtectionSide) -> Self {
        match value {
            CoreProtectionSide::Buyer => Self::Buyer,
            CoreProtectionSide::Seller => Self::Seller,
        }
    }
}

#[pymethods]
impl ProtectionSide {
    fn __repr__(&self) -> String {
        match self {
            Self::Buyer => "ProtectionSide.Buyer".to_string(),
            Self::Seller => "ProtectionSide.Seller".to_string(),
        }
    }
}

#[pyclass(eq, eq_int, module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CdsDateRule {
    TwentiethImm,
    QuarterlyImm,
}

impl CdsDateRule {
    fn to_core(self) -> CoreCdsDateRule {
        match self {
            Self::TwentiethImm => CoreCdsDateRule::TwentiethImm,
            Self::QuarterlyImm => CoreCdsDateRule::QuarterlyImm,
        }
    }

    fn from_core(value: CoreCdsDateRule) -> Self {
        match value {
            CoreCdsDateRule::TwentiethImm => Self::TwentiethImm,
            CoreCdsDateRule::QuarterlyImm => Self::QuarterlyImm,
        }
    }
}

#[pymethods]
impl CdsDateRule {
    fn __repr__(&self) -> String {
        match self {
            Self::TwentiethImm => "CdsDateRule.TwentiethImm".to_string(),
            Self::QuarterlyImm => "CdsDateRule.QuarterlyImm".to_string(),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct DatedCds {
    #[pyo3(get, set)]
    pub side: ProtectionSide,
    #[pyo3(get, set)]
    pub notional: f64,
    #[pyo3(get, set)]
    pub running_spread: f64,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
    #[pyo3(get, set)]
    pub issue_date: String,
    #[pyo3(get, set)]
    pub maturity_date: String,
    #[pyo3(get, set)]
    pub coupon_interval_months: i32,
    #[pyo3(get, set)]
    pub date_rule: CdsDateRule,
}

impl DatedCds {
    fn to_core(&self) -> PyResult<CoreDatedCds> {
        Ok(CoreDatedCds {
            side: self.side.to_core(),
            notional: self.notional,
            running_spread: self.running_spread,
            recovery_rate: self.recovery_rate,
            issue_date: parse_naive_date(&self.issue_date)?,
            maturity_date: parse_naive_date(&self.maturity_date)?,
            coupon_interval_months: self.coupon_interval_months,
            date_rule: self.date_rule.to_core(),
        })
    }

    fn from_core(value: CoreDatedCds) -> Self {
        Self {
            side: ProtectionSide::from_core(value.side),
            notional: value.notional,
            running_spread: value.running_spread,
            recovery_rate: value.recovery_rate,
            issue_date: format_naive_date(value.issue_date),
            maturity_date: format_naive_date(value.maturity_date),
            coupon_interval_months: value.coupon_interval_months,
            date_rule: CdsDateRule::from_core(value.date_rule),
        }
    }
}

#[pymethods]
impl DatedCds {
    #[new]
    fn new(
        side: ProtectionSide,
        notional: f64,
        running_spread: f64,
        recovery_rate: f64,
        issue_date: String,
        maturity_date: String,
        coupon_interval_months: i32,
        date_rule: CdsDateRule,
    ) -> PyResult<Self> {
        let _ = parse_naive_date(&issue_date)?;
        let _ = parse_naive_date(&maturity_date)?;
        Ok(Self {
            side,
            notional,
            running_spread,
            recovery_rate,
            issue_date,
            maturity_date,
            coupon_interval_months,
            date_rule,
        })
    }

    #[staticmethod]
    fn standard_imm(
        side: ProtectionSide,
        trade_date: String,
        tenor_years: i32,
        notional: f64,
        running_spread: f64,
        recovery_rate: f64,
    ) -> PyResult<Self> {
        let trade_date = parse_naive_date(&trade_date)?;
        Ok(Self::from_core(CoreDatedCds::standard_imm(
            side.to_core(),
            trade_date,
            tenor_years,
            notional,
            running_spread,
            recovery_rate,
        )))
    }

    fn price_midpoint_flat(
        &self,
        valuation_date: String,
        hazard_rate: f64,
        discount_rate: f64,
        conventions: Option<&IsdaConventions>,
    ) -> PyResult<CdsPriceResult> {
        let valuation_date = parse_naive_date(&valuation_date)?;
        let conventions = conventions.map(|c| c.to_core()).unwrap_or_default();
        Ok(CdsPriceResult::from_core(price_midpoint_flat(
            &self.to_core()?,
            valuation_date,
            hazard_rate,
            discount_rate,
            conventions,
        )))
    }

    fn price_isda_flat(
        &self,
        valuation_date: String,
        hazard_rate: f64,
        discount_rate: f64,
        conventions: Option<&IsdaConventions>,
    ) -> PyResult<CdsPriceResult> {
        let valuation_date = parse_naive_date(&valuation_date)?;
        let conventions = conventions.map(|c| c.to_core()).unwrap_or_default();
        Ok(CdsPriceResult::from_core(price_isda_flat(
            &self.to_core()?,
            valuation_date,
            hazard_rate,
            discount_rate,
            conventions,
        )))
    }

    fn __repr__(&self) -> String {
        format!(
            "DatedCds(side={:?}, notional={}, running_spread={}, recovery_rate={}, issue_date={:?}, maturity_date={:?}, coupon_interval_months={}, date_rule={:?})",
            self.side,
            self.notional,
            self.running_spread,
            self.recovery_rate,
            self.issue_date,
            self.maturity_date,
            self.coupon_interval_months,
            self.date_rule
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IsdaConventions {
    #[pyo3(get, set)]
    pub step_in_days: usize,
    #[pyo3(get, set)]
    pub cash_settle_days: usize,
}

impl IsdaConventions {
    fn to_core(self) -> CoreIsdaConventions {
        CoreIsdaConventions {
            step_in_days: self.step_in_days,
            cash_settle_days: self.cash_settle_days,
        }
    }
}

#[pymethods]
impl IsdaConventions {
    #[new]
    fn new(step_in_days: Option<usize>, cash_settle_days: Option<usize>) -> Self {
        let defaults = CoreIsdaConventions::default();
        Self {
            step_in_days: step_in_days.unwrap_or(defaults.step_in_days),
            cash_settle_days: cash_settle_days.unwrap_or(defaults.cash_settle_days),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IsdaConventions(step_in_days={}, cash_settle_days={})",
            self.step_in_days, self.cash_settle_days
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct CdsPriceResult {
    #[pyo3(get, set)]
    pub clean_npv: f64,
    #[pyo3(get, set)]
    pub dirty_npv: f64,
    #[pyo3(get, set)]
    pub premium_leg_pv: f64,
    #[pyo3(get, set)]
    pub protection_leg_pv: f64,
    #[pyo3(get, set)]
    pub accrued_premium_pv: f64,
    #[pyo3(get, set)]
    pub fair_spread: f64,
    #[pyo3(get, set)]
    pub step_in_date: String,
    #[pyo3(get, set)]
    pub cash_settle_date: String,
}

impl CdsPriceResult {
    fn from_core(value: CoreCdsPriceResult) -> Self {
        Self {
            clean_npv: value.clean_npv,
            dirty_npv: value.dirty_npv,
            premium_leg_pv: value.premium_leg_pv,
            protection_leg_pv: value.protection_leg_pv,
            accrued_premium_pv: value.accrued_premium_pv,
            fair_spread: value.fair_spread,
            step_in_date: format_naive_date(value.step_in_date),
            cash_settle_date: format_naive_date(value.cash_settle_date),
        }
    }
}

#[pymethods]
impl CdsPriceResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        clean_npv: f64,
        dirty_npv: f64,
        premium_leg_pv: f64,
        protection_leg_pv: f64,
        accrued_premium_pv: f64,
        fair_spread: f64,
        step_in_date: String,
        cash_settle_date: String,
    ) -> PyResult<Self> {
        let _ = parse_naive_date(&step_in_date)?;
        let _ = parse_naive_date(&cash_settle_date)?;
        Ok(Self {
            clean_npv,
            dirty_npv,
            premium_leg_pv,
            protection_leg_pv,
            accrued_premium_pv,
            fair_spread,
            step_in_date,
            cash_settle_date,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CdsPriceResult(clean_npv={}, dirty_npv={}, premium_leg_pv={}, protection_leg_pv={}, accrued_premium_pv={}, fair_spread={}, step_in_date={:?}, cash_settle_date={:?})",
            self.clean_npv,
            self.dirty_npv,
            self.premium_leg_pv,
            self.protection_leg_pv,
            self.accrued_premium_pv,
            self.fair_spread,
            self.step_in_date,
            self.cash_settle_date
        )
    }
}

#[pyfunction]
pub fn py_cds_npv(
    notional: f64,
    spread: f64,
    maturity: f64,
    recovery_rate: f64,
    payment_freq: usize,
    discount_rate: f64,
    hazard_rate: f64,
) -> f64 {
    if payment_freq == 0 {
        return f64::NAN;
    }

    let cds = CoreCds {
        notional,
        spread,
        maturity,
        recovery_rate,
        payment_freq,
    };

    let tenors = tenor_grid(maturity, payment_freq);
    let discount_curve = YieldCurve::new(
        tenors
            .iter()
            .map(|t| (*t, (-discount_rate * *t).exp()))
            .collect(),
    );

    let hazards = vec![hazard_rate.max(0.0); tenors.len()];
    let survival_curve = CoreSurvivalCurve::from_piecewise_hazard(&tenors, &hazards);

    cds.npv(&discount_curve, &survival_curve)
}

#[pyfunction]
pub fn py_survival_prob(hazard_rate: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }

    let tt = t.max(1e-8);
    let tenors = vec![tt];
    let hazards = vec![hazard_rate.max(0.0)];
    let curve = CoreSurvivalCurve::from_piecewise_hazard(&tenors, &hazards);
    curve.survival_prob(tt)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_cds_npv, module)?)?;
    module.add_function(wrap_pyfunction!(py_survival_prob, module)?)?;
    module.add_class::<Cds>()?;
    module.add_class::<CdoTranche>()?;
    module.add_class::<SyntheticCdo>()?;
    module.add_class::<CdsIndex>()?;
    module.add_class::<NthToDefaultBasket>()?;
    module.add_class::<CdsOption>()?;
    module.add_class::<GaussianCopula>()?;
    module.add_class::<BasketDefaultSimulation>()?;
    module.add_class::<SurvivalCurve>()?;
    module.add_class::<ProtectionSide>()?;
    module.add_class::<CdsDateRule>()?;
    module.add_class::<DatedCds>()?;
    module.add_class::<IsdaConventions>()?;
    module.add_class::<CdsPriceResult>()?;
    Ok(())
}
