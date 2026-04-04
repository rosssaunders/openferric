use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use openferric_core::core::{
    AsianSpec, Averaging, ExerciseStyle, OptionType, PricingEngine as _, StrikeType,
};
use openferric_core::engines::analytic::black_scholes::{
    bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega,
};
use openferric_core::engines::{
    analytic as analytic_core, monte_carlo as mc_core, pde as pde_core,
};
use openferric_core::instruments::{
    AsianOption, AssetOrNothingOption, BarrierOption, BermudanOption, BestOfTwoCallOption,
    CashOrNothingOption, DoubleBarrierOption, DoubleBarrierType, ExoticOption, FuturesOption,
    FxOption, LookbackFixedOption, LookbackFloatingOption, PowerOption, SpreadOption,
    TwoAssetCorrelationOption, VarianceOptionQuote, VarianceSwap, VolatilitySwap,
    WorstOfTwoCallOption,
};
use openferric_core::market::Market;
use openferric_core::math::AccuracyTier;
use openferric_core::models::Heston;

use crate::core::PricingResult;
use crate::helpers::{
    build_market, parse_barrier_direction, parse_barrier_style, parse_option_type,
};

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn option_type_from_str(value: &str) -> PyResult<OptionType> {
    parse_option_type(value).ok_or_else(|| py_value_error("option_type must be 'call' or 'put'"))
}

fn exercise_style_from_inputs(
    style: Option<&str>,
    bermudan_dates: Option<Vec<f64>>,
) -> PyResult<ExerciseStyle> {
    match style.unwrap_or("european").to_ascii_lowercase().as_str() {
        "european" => Ok(ExerciseStyle::European),
        "american" => Ok(ExerciseStyle::American),
        "bermudan" => Ok(ExerciseStyle::Bermudan {
            dates: bermudan_dates.unwrap_or_default(),
        }),
        other => Err(py_value_error(format!(
            "unsupported exercise style '{other}'"
        ))),
    }
}

fn build_market_checked(spot: f64, rate: f64, div_yield: f64, vol: f64) -> PyResult<Market> {
    build_market(spot, rate, div_yield, vol).ok_or_else(|| py_value_error("invalid market inputs"))
}

fn map_pricing_err<T>(result: Result<T, openferric_core::core::PricingError>) -> PyResult<T> {
    result.map_err(|err| py_value_error(format!("{err:?}")))
}

fn quotes_from_tuples(quotes: Vec<(f64, f64, f64)>) -> Vec<VarianceOptionQuote> {
    quotes
        .into_iter()
        .map(|(strike, call_price, put_price)| {
            VarianceOptionQuote::new(strike, call_price, put_price)
        })
        .collect()
}

fn accuracy_tier_from_str(value: Option<&str>) -> PyResult<Option<AccuracyTier>> {
    match value.map(|item| item.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) if value == "high" => Ok(Some(AccuracyTier::High)),
        Some(value) if value == "fast" => Ok(Some(AccuracyTier::Fast)),
        Some(value) => Err(py_value_error(format!(
            "unsupported accuracy tier '{value}'"
        ))),
    }
}

fn variance_reduction_from_str(value: Option<&str>) -> PyResult<mc_core::VarianceReduction> {
    match value.unwrap_or("none").to_ascii_lowercase().as_str() {
        "none" => Ok(mc_core::VarianceReduction::None),
        "antithetic" => Ok(mc_core::VarianceReduction::Antithetic),
        "control_variate" | "control-variate" => Ok(mc_core::VarianceReduction::ControlVariate),
        other => Err(py_value_error(format!(
            "unsupported variance reduction '{other}'"
        ))),
    }
}

fn rng_kind_from_str(value: Option<&str>) -> PyResult<openferric_core::math::FastRngKind> {
    match value
        .unwrap_or("xoshiro256plusplus")
        .to_ascii_lowercase()
        .as_str()
    {
        "xoshiro256plusplus" | "xoshiro256_plus_plus" | "xoshiro" => {
            Ok(openferric_core::math::FastRngKind::Xoshiro256PlusPlus)
        }
        "pcg64" | "pcg" => Ok(openferric_core::math::FastRngKind::Pcg64),
        "threadrng" | "thread_rng" | "thread" => Ok(openferric_core::math::FastRngKind::ThreadRng),
        "stdrng" | "std_rng" | "std" => Ok(openferric_core::math::FastRngKind::StdRng),
        other => Err(py_value_error(format!("unsupported rng kind '{other}'"))),
    }
}

fn double_barrier_type_from_str(value: &str) -> PyResult<DoubleBarrierType> {
    match value.to_ascii_lowercase().as_str() {
        "knock_out" | "knock-out" | "knockout" => Ok(DoubleBarrierType::KnockOut),
        "knock_in" | "knock-in" | "knockin" => Ok(DoubleBarrierType::KnockIn),
        other => Err(py_value_error(format!(
            "unsupported double barrier type '{other}'"
        ))),
    }
}

fn binary_barrier_type_from_str(value: &str) -> PyResult<analytic_core::BinaryBarrierType> {
    match value.to_ascii_lowercase().as_str() {
        "down_in" | "down-in" | "downin" => Ok(analytic_core::BinaryBarrierType::DownIn),
        "up_in" | "up-in" | "upin" => Ok(analytic_core::BinaryBarrierType::UpIn),
        "down_out" | "down-out" | "downout" => Ok(analytic_core::BinaryBarrierType::DownOut),
        "up_out" | "up-out" | "upout" => Ok(analytic_core::BinaryBarrierType::UpOut),
        other => Err(py_value_error(format!(
            "unsupported binary barrier type '{other}'"
        ))),
    }
}

fn adi_scheme_from_str(value: Option<&str>) -> PyResult<pde_core::AdiScheme> {
    match value.unwrap_or("craig_sneyd").to_ascii_lowercase().as_str() {
        "douglas_rachford" | "douglas-rachford" => Ok(pde_core::AdiScheme::DouglasRachford),
        "craig_sneyd" | "craig-sneyd" => Ok(pde_core::AdiScheme::CraigSneyd),
        other => Err(py_value_error(format!("unsupported ADI scheme '{other}'"))),
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct GreeksResult {
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub rho: f64,
}

impl From<openferric_core::core::Greeks> for GreeksResult {
    fn from(value: openferric_core::core::Greeks) -> Self {
        Self {
            delta: value.delta,
            gamma: value.gamma,
            vega: value.vega,
            theta: value.theta,
            rho: value.rho,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy)]
pub struct PdeExerciseBoundaryPoint {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub strike: f64,
    #[pyo3(get)]
    pub boundary_spot: Option<f64>,
}

impl From<pde_core::PdeExerciseBoundaryPoint> for PdeExerciseBoundaryPoint {
    fn from(value: pde_core::PdeExerciseBoundaryPoint) -> Self {
        Self {
            time: value.time,
            strike: value.strike,
            boundary_spot: value.boundary_spot,
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone)]
pub struct BermudanPdeOutput {
    #[pyo3(get)]
    pub result: PricingResult,
    #[pyo3(get)]
    pub exercise_boundary: Vec<PdeExerciseBoundaryPoint>,
}

impl From<pde_core::BermudanPdeOutput> for BermudanPdeOutput {
    fn from(value: pde_core::BermudanPdeOutput) -> Self {
        Self {
            result: value.result.into(),
            exercise_boundary: value
                .exercise_boundary
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct AnalyticEngine;

#[pymethods]
impl AnalyticEngine {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    pub fn bachelier_price(
        option_type: &str,
        forward: f64,
        strike: f64,
        rate: f64,
        sigma_n: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::bachelier_price(
            option_type_from_str(option_type)?,
            forward,
            strike,
            rate,
            sigma_n,
            expiry,
        ))
    }

    #[staticmethod]
    pub fn bachelier_greeks(
        option_type: &str,
        forward: f64,
        strike: f64,
        rate: f64,
        sigma_n: f64,
        expiry: f64,
    ) -> PyResult<GreeksResult> {
        Ok(map_pricing_err(analytic_core::bachelier_greeks(
            option_type_from_str(option_type)?,
            forward,
            strike,
            rate,
            sigma_n,
            expiry,
        ))?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn black_scholes_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        Ok(bs_price(
            option_type_from_str(option_type)?,
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            expiry,
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn black_scholes_greeks(
        option_type: &str,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<GreeksResult> {
        let option_type = option_type_from_str(option_type)?;
        Ok(GreeksResult {
            delta: bs_delta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
            gamma: bs_gamma(spot, strike, rate, dividend_yield, vol, expiry),
            vega: bs_vega(spot, strike, rate, dividend_yield, vol, expiry),
            theta: bs_theta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
            rho: bs_rho(option_type, spot, strike, rate, dividend_yield, vol, expiry),
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn black_scholes_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = match option_type_from_str(option_type)? {
            OptionType::Call => {
                openferric_core::instruments::VanillaOption::european_call(strike, expiry)
            }
            OptionType::Put => {
                openferric_core::instruments::VanillaOption::european_put(strike, expiry)
            }
        };
        Ok(
            map_pricing_err(analytic_core::BlackScholesEngine::new().price(&instrument, &market))?
                .into(),
        )
    }

    #[staticmethod]
    pub fn black76_price(
        option_type: &str,
        forward: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::black76_price(
            option_type_from_str(option_type)?,
            forward,
            strike,
            rate,
            vol,
            expiry,
        ))
    }

    #[staticmethod]
    pub fn black76_greeks(
        option_type: &str,
        forward: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<GreeksResult> {
        Ok(map_pricing_err(analytic_core::black76_greeks(
            option_type_from_str(option_type)?,
            forward,
            strike,
            rate,
            vol,
            expiry,
        ))?
        .into())
    }

    #[staticmethod]
    pub fn black76_engine_price(
        option_type: &str,
        forward: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<PricingResult> {
        let instrument = FuturesOption::new(
            forward,
            strike,
            vol,
            rate,
            expiry,
            option_type_from_str(option_type)?,
        );
        let market = build_market_checked(1.0, rate, 0.0, vol)?;
        Ok(
            map_pricing_err(analytic_core::Black76Engine::new().price(&instrument, &market))?
                .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn fx_engine_price(
        option_type: &str,
        domestic_rate: f64,
        foreign_rate: f64,
        spot_fx: f64,
        strike_fx: f64,
        vol: f64,
        maturity: f64,
    ) -> PyResult<PricingResult> {
        let instrument = FxOption::new(
            option_type_from_str(option_type)?,
            domestic_rate,
            foreign_rate,
            spot_fx,
            strike_fx,
            vol,
            maturity,
        );
        let market = build_market_checked(spot_fx, domestic_rate, foreign_rate, vol)?;
        Ok(map_pricing_err(
            analytic_core::GarmanKohlhagenEngine::new().price(&instrument, &market),
        )?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn barrier_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        barrier_style: &str,
        barrier_direction: &str,
        barrier_level: f64,
        rebate: f64,
    ) -> PyResult<PricingResult> {
        let style = parse_barrier_style(barrier_style)
            .ok_or_else(|| py_value_error("barrier_style must be 'in' or 'out'"))?;
        let direction = parse_barrier_direction(barrier_direction)
            .ok_or_else(|| py_value_error("barrier_direction must be 'up' or 'down'"))?;
        let instrument = BarrierOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            barrier: openferric_core::core::BarrierSpec {
                direction,
                style,
                level: barrier_level,
                rebate,
            },
        };
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(map_pricing_err(
            analytic_core::BarrierAnalyticEngine::new().price(&instrument, &market),
        )?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn digital_cash_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        cash: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
    ) -> PyResult<PricingResult> {
        let instrument =
            CashOrNothingOption::new(option_type_from_str(option_type)?, strike, cash, expiry);
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(map_pricing_err(
            analytic_core::DigitalAnalyticEngine::new().price(&instrument, &market),
        )?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn digital_asset_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
    ) -> PyResult<PricingResult> {
        let instrument =
            AssetOrNothingOption::new(option_type_from_str(option_type)?, strike, expiry);
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(map_pricing_err(
            analytic_core::DigitalAnalyticEngine::new().price(&instrument, &market),
        )?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn cash_or_nothing_barrier_price(
        option_type: &str,
        barrier_type: &str,
        spot: f64,
        strike: f64,
        barrier: f64,
        cash: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::cash_or_nothing_barrier_price(
            option_type_from_str(option_type)?,
            binary_barrier_type_from_str(barrier_type)?,
            spot,
            strike,
            barrier,
            cash,
            rate,
            dividend_yield,
            vol,
            expiry,
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn geometric_asian_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        observation_times: Vec<f64>,
    ) -> PyResult<PricingResult> {
        let instrument = AsianOption::new(
            option_type_from_str(option_type)?,
            strike,
            expiry,
            AsianSpec {
                averaging: Averaging::Geometric,
                strike_type: StrikeType::Fixed,
                observation_times,
            },
        );
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(
            map_pricing_err(
                analytic_core::GeometricAsianEngine::new().price(&instrument, &market),
            )?
            .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn lookback_floating_engine_price(
        option_type: &str,
        spot: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        observed_extreme: Option<f64>,
    ) -> PyResult<PricingResult> {
        let instrument = ExoticOption::LookbackFloating(LookbackFloatingOption {
            option_type: option_type_from_str(option_type)?,
            expiry,
            observed_extreme,
        });
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(
            map_pricing_err(
                analytic_core::ExoticAnalyticEngine::new().price(&instrument, &market),
            )?
            .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn lookback_fixed_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        observed_extreme: Option<f64>,
    ) -> PyResult<PricingResult> {
        let instrument = ExoticOption::LookbackFixed(LookbackFixedOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            observed_extreme,
        });
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(
            map_pricing_err(
                analytic_core::ExoticAnalyticEngine::new().price(&instrument, &market),
            )?
            .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn double_barrier_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        lower_barrier: f64,
        upper_barrier: f64,
        barrier_type: &str,
        rebate: f64,
        series_terms: Option<usize>,
    ) -> PyResult<PricingResult> {
        let instrument = DoubleBarrierOption::new(
            option_type_from_str(option_type)?,
            strike,
            expiry,
            lower_barrier,
            upper_barrier,
            double_barrier_type_from_str(barrier_type)?,
            rebate,
        );
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let engine = analytic_core::DoubleBarrierAnalyticEngine::new()
            .with_series_terms(series_terms.unwrap_or(5));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn power_option_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        alpha: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::power_option_price(
            option_type_from_str(option_type)?,
            spot,
            strike,
            rate,
            dividend_yield,
            vol,
            alpha,
            expiry,
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn power_engine_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        alpha: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
    ) -> PyResult<PricingResult> {
        let instrument =
            PowerOption::new(option_type_from_str(option_type)?, strike, alpha, expiry);
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        Ok(
            map_pricing_err(analytic_core::PowerOptionEngine::new().price(&instrument, &market))?
                .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn margrabe_exchange_price(
        s1: f64,
        s2: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        let option = SpreadOption {
            s1,
            s2,
            k: 0.0,
            vol1,
            vol2,
            rho,
            q1,
            q2,
            r: rate,
            t: expiry,
        };
        map_pricing_err(analytic_core::margrabe_exchange_price(&option))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn kirk_spread_price(
        s1: f64,
        s2: f64,
        k: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        let option = SpreadOption {
            s1,
            s2,
            k,
            vol1,
            vol2,
            rho,
            q1,
            q2,
            r: rate,
            t: expiry,
        };
        map_pricing_err(analytic_core::kirk_spread_price(&option))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn spread_engine_price(
        method: &str,
        s1: f64,
        s2: f64,
        k: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<PricingResult> {
        let method = match method.to_ascii_lowercase().as_str() {
            "margrabe" => analytic_core::SpreadAnalyticMethod::Margrabe,
            "kirk" => analytic_core::SpreadAnalyticMethod::Kirk,
            other => {
                return Err(py_value_error(format!(
                    "unsupported spread method '{other}'"
                )));
            }
        };
        let option = SpreadOption {
            s1,
            s2,
            k,
            vol1,
            vol2,
            rho,
            q1,
            q2,
            r: rate,
            t: expiry,
        };
        let market = build_market_checked(1.0, rate, 0.0, vol1.max(vol2).max(1e-8))?;
        Ok(map_pricing_err(
            analytic_core::SpreadAnalyticEngine::new(method).price(&option, &market),
        )?
        .into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn best_of_two_call_price(
        s1: f64,
        s2: f64,
        k: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::best_of_two_call_price(
            &BestOfTwoCallOption {
                s1,
                s2,
                k,
                vol1,
                vol2,
                rho,
                q1,
                q2,
                r: rate,
                t: expiry,
            },
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn worst_of_two_call_price(
        s1: f64,
        s2: f64,
        k: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::worst_of_two_call_price(
            &WorstOfTwoCallOption {
                s1,
                s2,
                k,
                vol1,
                vol2,
                rho,
                q1,
                q2,
                r: rate,
                t: expiry,
            },
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn two_asset_correlation_price(
        option_type: &str,
        s1: f64,
        s2: f64,
        k1: f64,
        k2: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::two_asset_correlation_price(
            &TwoAssetCorrelationOption {
                option_type: option_type_from_str(option_type)?,
                s1,
                s2,
                k1,
                k2,
                vol1,
                vol2,
                rho,
                q1,
                q2,
                r: rate,
                t: expiry,
            },
        ))
    }

    #[staticmethod]
    pub fn fair_variance_strike_from_quotes(
        expiry: f64,
        rate: f64,
        spot: f64,
        dividend_yield: f64,
        quotes: Vec<(f64, f64, f64)>,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::fair_variance_strike_from_quotes(
            expiry,
            rate,
            spot,
            dividend_yield,
            &quotes_from_tuples(quotes),
        ))
    }

    #[staticmethod]
    pub fn fair_volatility_strike_from_variance(
        fair_variance: f64,
        var_of_var: f64,
    ) -> PyResult<f64> {
        map_pricing_err(analytic_core::fair_volatility_strike_from_variance(
            fair_variance,
            var_of_var,
        ))
    }

    #[staticmethod]
    pub fn variance_swap_mtm(
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        quotes: Vec<(f64, f64, f64)>,
        fair_variance: f64,
        rate: f64,
        observed_realized_var: Option<f64>,
    ) -> PyResult<f64> {
        let mut instrument = VarianceSwap::new(
            notional_vega,
            strike_vol,
            expiry,
            quotes_from_tuples(quotes),
        );
        if let Some(observed_realized_var) = observed_realized_var {
            instrument = instrument.with_observed_realized_var(observed_realized_var);
        }
        map_pricing_err(analytic_core::variance_swap_mtm(
            &instrument,
            fair_variance,
            rate,
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn volatility_swap_mtm(
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        quotes: Vec<(f64, f64, f64)>,
        var_of_var: f64,
        fair_variance: f64,
        fair_volatility: f64,
        rate: f64,
        observed_realized_var: Option<f64>,
    ) -> PyResult<f64> {
        let mut instrument = VolatilitySwap::new(
            notional_vega,
            strike_vol,
            expiry,
            quotes_from_tuples(quotes),
            var_of_var,
        );
        if let Some(observed_realized_var) = observed_realized_var {
            instrument = instrument.with_observed_realized_var(observed_realized_var);
        }
        map_pricing_err(analytic_core::volatility_swap_mtm(
            &instrument,
            fair_variance,
            fair_volatility,
            rate,
        ))
    }

    #[staticmethod]
    pub fn variance_swap_engine_price(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        quotes: Vec<(f64, f64, f64)>,
        observed_realized_var: Option<f64>,
    ) -> PyResult<PricingResult> {
        let mut instrument = VarianceSwap::new(
            notional_vega,
            strike_vol,
            expiry,
            quotes_from_tuples(quotes),
        );
        if let Some(observed_realized_var) = observed_realized_var {
            instrument = instrument.with_observed_realized_var(observed_realized_var);
        }
        let market = build_market_checked(spot, rate, dividend_yield, 0.2)?;
        Ok(
            map_pricing_err(analytic_core::VarianceSwapEngine::new().price(&instrument, &market))?
                .into(),
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn volatility_swap_engine_price(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        quotes: Vec<(f64, f64, f64)>,
        var_of_var: f64,
        observed_realized_var: Option<f64>,
    ) -> PyResult<PricingResult> {
        let mut instrument = VolatilitySwap::new(
            notional_vega,
            strike_vol,
            expiry,
            quotes_from_tuples(quotes),
            var_of_var,
        );
        if let Some(observed_realized_var) = observed_realized_var {
            instrument = instrument.with_observed_realized_var(observed_realized_var);
        }
        let market = build_market_checked(spot, rate, dividend_yield, 0.2)?;
        Ok(
            map_pricing_err(analytic_core::VarianceSwapEngine::new().price(&instrument, &market))?
                .into(),
        )
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PdeEngine;

#[pymethods]
impl PdeEngine {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn crank_nicolson_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        time_steps: usize,
        space_steps: usize,
        s_max_multiplier: Option<f64>,
        exercise_style: Option<&str>,
        bermudan_dates: Option<Vec<f64>>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = openferric_core::instruments::VanillaOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            exercise: exercise_style_from_inputs(exercise_style, bermudan_dates)?,
        };
        let engine = pde_core::CrankNicolsonEngine::new(time_steps, space_steps)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn crank_nicolson_bermudan_with_boundary(
        option_type: &str,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        expiry: f64,
        exercise_dates: Vec<f64>,
        strike_schedule: Vec<f64>,
        time_steps: usize,
        space_steps: usize,
        s_max_multiplier: Option<f64>,
    ) -> PyResult<BermudanPdeOutput> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = BermudanOption::new(
            option_type_from_str(option_type)?,
            expiry,
            exercise_dates,
            strike_schedule,
        );
        let engine = pde_core::CrankNicolsonEngine::new(time_steps, space_steps)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0));
        Ok(map_pricing_err(engine.price_bermudan_with_boundary(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn explicit_fd_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        time_steps: usize,
        space_steps: usize,
        s_max_multiplier: Option<f64>,
        grid_stretch: Option<f64>,
        cfl_safety_factor: Option<f64>,
        enforce_cfl: Option<bool>,
        exercise_style: Option<&str>,
        bermudan_dates: Option<Vec<f64>>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = openferric_core::instruments::VanillaOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            exercise: exercise_style_from_inputs(exercise_style, bermudan_dates)?,
        };
        let engine = pde_core::ExplicitFdEngine::new(time_steps, space_steps)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0))
            .with_grid_stretch(grid_stretch.unwrap_or(0.15))
            .with_cfl_safety_factor(cfl_safety_factor.unwrap_or(0.95))
            .with_enforce_cfl(enforce_cfl.unwrap_or(true));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_fd_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        time_steps: usize,
        space_steps: usize,
        s_max_multiplier: Option<f64>,
        grid_stretch: Option<f64>,
        exercise_style: Option<&str>,
        bermudan_dates: Option<Vec<f64>>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = openferric_core::instruments::VanillaOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            exercise: exercise_style_from_inputs(exercise_style, bermudan_dates)?,
        };
        let engine = pde_core::ImplicitFdEngine::new(time_steps, space_steps)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0))
            .with_grid_stretch(grid_stretch.unwrap_or(0.15));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn hopscotch_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        time_steps: usize,
        space_steps: usize,
        s_max_multiplier: Option<f64>,
        grid_stretch: Option<f64>,
        exercise_style: Option<&str>,
        bermudan_dates: Option<Vec<f64>>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = openferric_core::instruments::VanillaOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            exercise: exercise_style_from_inputs(exercise_style, bermudan_dates)?,
        };
        let engine = pde_core::HopscotchEngine::new(time_steps, space_steps)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0))
            .with_grid_stretch(grid_stretch.unwrap_or(0.15));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn adi_heston_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        time_steps: usize,
        spot_steps: usize,
        variance_steps: usize,
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        v0: f64,
        scheme: Option<&str>,
        s_max_multiplier: Option<f64>,
        v_max_multiplier: Option<f64>,
        theta_adi: Option<f64>,
        enforce_feller: Option<bool>,
    ) -> PyResult<PricingResult> {
        let instrument = match option_type_from_str(option_type)? {
            OptionType::Call => {
                openferric_core::instruments::VanillaOption::european_call(strike, expiry)
            }
            OptionType::Put => {
                openferric_core::instruments::VanillaOption::european_put(strike, expiry)
            }
        };
        let market = build_market_checked(spot, rate, dividend_yield, 0.2)?;
        let model = Heston {
            mu,
            kappa,
            theta,
            xi,
            rho,
            v0,
        };
        let engine = pde_core::AdiHestonEngine::new(model, time_steps, spot_steps, variance_steps)
            .with_scheme(adi_scheme_from_str(scheme)?)
            .with_s_max_multiplier(s_max_multiplier.unwrap_or(4.0))
            .with_v_max_multiplier(v_max_multiplier.unwrap_or(5.0))
            .with_theta_adi(theta_adi.unwrap_or(0.5))
            .with_enforce_feller(enforce_feller.unwrap_or(true));
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }
}

#[pyclass(module = "openferric", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct McEngine;

#[pymethods]
impl McEngine {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn vanilla_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
        variance_reduction: Option<&str>,
        rng_kind: Option<&str>,
        reproducible: Option<bool>,
        accuracy_tier: Option<&str>,
        exercise_style: Option<&str>,
        bermudan_dates: Option<Vec<f64>>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = openferric_core::instruments::VanillaOption {
            option_type: option_type_from_str(option_type)?,
            strike,
            expiry,
            exercise: exercise_style_from_inputs(exercise_style, bermudan_dates)?,
        };
        let mut engine = mc_core::MonteCarloPricingEngine::new(num_paths, num_steps, seed)
            .with_variance_reduction(variance_reduction_from_str(variance_reduction)?)
            .with_rng_kind(rng_kind_from_str(rng_kind)?);
        if reproducible == Some(false) {
            engine = engine.with_randomized_streams();
        }
        if let Some(tier) = accuracy_tier_from_str(accuracy_tier)? {
            engine = engine.with_accuracy_tier(tier);
        }
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn arithmetic_asian_price(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        observation_times: Vec<f64>,
        paths: usize,
        steps: usize,
        seed: u64,
        control_variate: Option<bool>,
        rng_kind: Option<&str>,
        reproducible: Option<bool>,
    ) -> PyResult<PricingResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = AsianOption::new(
            option_type_from_str(option_type)?,
            strike,
            expiry,
            AsianSpec {
                averaging: Averaging::Arithmetic,
                strike_type: StrikeType::Fixed,
                observation_times,
            },
        );
        let mut engine = mc_core::ArithmeticAsianMC::new(paths, steps, seed)
            .with_control_variate(control_variate.unwrap_or(true))
            .with_rng_kind(rng_kind_from_str(rng_kind)?);
        if reproducible == Some(false) {
            engine = engine.with_randomized_streams();
        }
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn greeks_pathwise(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        num_paths: usize,
        seed: u64,
        antithetic: Option<bool>,
        spot_bump_rel: Option<f64>,
        rng_kind: Option<&str>,
        reproducible: Option<bool>,
    ) -> PyResult<GreeksResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = match option_type_from_str(option_type)? {
            OptionType::Call => {
                openferric_core::instruments::VanillaOption::european_call(strike, expiry)
            }
            OptionType::Put => {
                openferric_core::instruments::VanillaOption::european_put(strike, expiry)
            }
        };
        let mut engine = mc_core::MonteCarloGreeksEngine::new(num_paths, seed)
            .with_antithetic(antithetic.unwrap_or(true))
            .with_spot_bump_rel(spot_bump_rel.unwrap_or(1.0e-2))
            .with_rng_kind(rng_kind_from_str(rng_kind)?);
        if reproducible == Some(false) {
            engine = engine.with_randomized_streams();
        }
        Ok(map_pricing_err(engine.estimate_pathwise(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn greeks_likelihood_ratio(
        option_type: &str,
        spot: f64,
        strike: f64,
        expiry: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        num_paths: usize,
        seed: u64,
        antithetic: Option<bool>,
        spot_bump_rel: Option<f64>,
        rng_kind: Option<&str>,
        reproducible: Option<bool>,
    ) -> PyResult<GreeksResult> {
        let market = build_market_checked(spot, rate, dividend_yield, vol)?;
        let instrument = match option_type_from_str(option_type)? {
            OptionType::Call => {
                openferric_core::instruments::VanillaOption::european_call(strike, expiry)
            }
            OptionType::Put => {
                openferric_core::instruments::VanillaOption::european_put(strike, expiry)
            }
        };
        let mut engine = mc_core::MonteCarloGreeksEngine::new(num_paths, seed)
            .with_antithetic(antithetic.unwrap_or(true))
            .with_spot_bump_rel(spot_bump_rel.unwrap_or(1.0e-2))
            .with_rng_kind(rng_kind_from_str(rng_kind)?);
        if reproducible == Some(false) {
            engine = engine.with_randomized_streams();
        }
        Ok(map_pricing_err(engine.estimate_likelihood_ratio(&instrument, &market))?.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn spread_price(
        s1: f64,
        s2: f64,
        k: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
        q1: f64,
        q2: f64,
        rate: f64,
        expiry: f64,
        num_paths: usize,
        seed: u64,
        antithetic: Option<bool>,
        rng_kind: Option<&str>,
        reproducible: Option<bool>,
    ) -> PyResult<PricingResult> {
        let instrument = SpreadOption {
            s1,
            s2,
            k,
            vol1,
            vol2,
            rho,
            q1,
            q2,
            r: rate,
            t: expiry,
        };
        let market = build_market_checked(1.0, rate, 0.0, vol1.max(vol2).max(1e-8))?;
        let mut engine = mc_core::SpreadMonteCarloEngine::new(num_paths, seed)
            .with_antithetic(antithetic.unwrap_or(true))
            .with_rng_kind(rng_kind_from_str(rng_kind)?);
        if reproducible == Some(false) {
            engine = engine.with_randomized_streams();
        }
        Ok(map_pricing_err(engine.price(&instrument, &market))?.into())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GreeksResult>()?;
    module.add_class::<PdeExerciseBoundaryPoint>()?;
    module.add_class::<BermudanPdeOutput>()?;
    module.add_class::<AnalyticEngine>()?;
    module.add_class::<PdeEngine>()?;
    module.add_class::<McEngine>()?;
    Ok(())
}
