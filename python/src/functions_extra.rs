use crate::helpers::catch_unwind_py;
use pyo3::prelude::*;
fn error(value: impl ToString) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(value.to_string())
}

#[pyfunction]
fn cash_settle_date(valuation_date: &str) -> PyResult<String> {
    let date = chrono::NaiveDate::parse_from_str(valuation_date, "%Y-%m-%d").map_err(error)?;
    catch_unwind_py(|| {
        openferric_core::credit::cash_settle_date(date)
            .format("%Y-%m-%d")
            .to_string()
    })
}

#[pyfunction]
fn market_fx_delta(
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    convention: crate::market::FxDeltaConvention,
    premium_currency: crate::market::PremiumCurrency,
) -> PyResult<f64> {
    openferric_core::market::fx::fx_delta(
        option_type.to_core(),
        spot,
        strike,
        domestic_rate,
        foreign_rate,
        vol,
        expiry,
        convention.to_core(),
        premium_currency.to_core(),
    )
    .map_err(error)
}

#[pyfunction]
fn step_in_date(valuation_date: &str) -> PyResult<String> {
    let date = chrono::NaiveDate::parse_from_str(valuation_date, "%Y-%m-%d").map_err(error)?;
    catch_unwind_py(|| {
        openferric_core::credit::step_in_date(date)
            .format("%Y-%m-%d")
            .to_string()
    })
}

#[pyfunction]
fn atm_strike(
    spot: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    atm_convention: crate::market::FxAtmConvention,
    delta_convention: crate::market::FxDeltaConvention,
) -> PyResult<f64> {
    openferric_core::market::fx::atm_strike(
        spot,
        domestic_rate,
        foreign_rate,
        vol,
        expiry,
        atm_convention.to_core(),
        delta_convention.to_core(),
    )
    .map_err(error)
}

#[pyfunction]
fn strike_from_delta(
    spot: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    target_delta: f64,
    convention: crate::market::FxDeltaConvention,
    premium_currency: crate::market::PremiumCurrency,
) -> PyResult<f64> {
    openferric_core::market::fx::strike_from_delta(
        spot,
        domestic_rate,
        foreign_rate,
        vol,
        expiry,
        target_delta,
        convention.to_core(),
        premium_currency.to_core(),
    )
    .map_err(error)
}

#[pyfunction]
fn bootstrap_survival_curve_from_cds_spreads(
    py: Python<'_>,
    cds_spreads: Vec<(f64, f64)>,
    recovery_rate: f64,
    payment_freq: usize,
    discount_curve: &crate::rates::YieldCurve,
) -> PyResult<crate::credit::SurvivalCurve> {
    let discount_curve = discount_curve.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::bootstrap::bootstrap_survival_curve_from_cds_spreads(
                &cds_spreads,
                recovery_rate,
                payment_freq,
                &discount_curve,
            )
        })
        .map(crate::credit::SurvivalCurve::from_core)
    })
}

#[pyfunction]
fn vasicek_portfolio_loss_cdf(
    py: Python<'_>,
    loss_fraction: f64,
    default_probability: f64,
    recovery_rate: f64,
    correlation: f64,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::cdo::vasicek_portfolio_loss_cdf(
                loss_fraction,
                default_probability,
                recovery_rate,
                correlation,
            )
        })
    })
}

#[pyfunction]
fn first_to_default_spread_copula(
    py: Python<'_>,
    notional: f64,
    maturity: f64,
    recovery_rate: f64,
    payment_freq: usize,
    discount_curve: &crate::rates::YieldCurve,
    survival_curves: Vec<crate::credit::SurvivalCurve>,
    copula: &crate::credit::GaussianCopula,
    num_paths: usize,
    seed: u64,
) -> PyResult<f64> {
    let discount_curve = discount_curve.to_core();
    let survival_curves = survival_curves
        .iter()
        .map(|value| value.to_core())
        .collect::<Vec<_>>();
    let copula = copula.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::cds_index::first_to_default_spread_copula(
                notional,
                maturity,
                recovery_rate,
                payment_freq,
                &discount_curve,
                &survival_curves,
                &copula,
                num_paths,
                seed,
            )
        })
    })
}

#[pyfunction]
fn fair_spread_from_hazard(
    py: Python<'_>,
    payment_freq: u32,
    cds_tenor: f64,
    hazard_rate: f64,
    risk_free_rate: f64,
    recovery: f64,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::cds_option::fair_spread_from_hazard(
                payment_freq,
                cds_tenor,
                hazard_rate,
                risk_free_rate,
                recovery,
            )
        })
    })
}

#[pyfunction]
fn risky_annuity(
    py: Python<'_>,
    payment_freq: u32,
    cds_tenor: f64,
    hazard_rate: f64,
    risk_free_rate: f64,
    recovery: f64,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::cds_option::risky_annuity(
                payment_freq,
                cds_tenor,
                hazard_rate,
                risk_free_rate,
                recovery,
            )
        })
    })
}

#[pyfunction]
fn cash_settle_date_with_calendar(
    py: Python<'_>,
    valuation_date: &str,
    calendar: &crate::rates::Calendar,
) -> PyResult<String> {
    let valuation_date =
        chrono::NaiveDate::parse_from_str(valuation_date, "%Y-%m-%d").map_err(error)?;
    let calendar = calendar.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::isda::cash_settle_date_with_calendar(valuation_date, &calendar)
        })
        .map(|date| date.format("%Y-%m-%d").to_string())
    })
}

#[pyfunction]
fn generate_imm_schedule(
    py: Python<'_>,
    issue_date: &str,
    maturity_date: &str,
    interval_months: i32,
    rule: crate::credit::CdsDateRule,
) -> PyResult<Vec<String>> {
    let issue_date = chrono::NaiveDate::parse_from_str(issue_date, "%Y-%m-%d").map_err(error)?;
    let maturity_date =
        chrono::NaiveDate::parse_from_str(maturity_date, "%Y-%m-%d").map_err(error)?;
    let rule = rule.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::isda::generate_imm_schedule(
                issue_date,
                maturity_date,
                interval_months,
                rule,
            )
        })
        .map(|dates| {
            dates
                .into_iter()
                .map(|date| date.format("%Y-%m-%d").to_string())
                .collect()
        })
    })
}

#[pyfunction]
fn hazard_from_par_spread(py: Python<'_>, par_spread: f64, recovery_rate: f64) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::isda::hazard_from_par_spread(par_spread, recovery_rate)
        })
    })
}

#[pyfunction]
fn next_imm_twentieth(py: Python<'_>, date: &str) -> PyResult<String> {
    let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(error)?;
    py.detach(|| {
        catch_unwind_py(|| openferric_core::credit::isda::next_imm_twentieth(date))
            .map(|date| date.format("%Y-%m-%d").to_string())
    })
}

#[pyfunction]
fn previous_imm_twentieth(py: Python<'_>, date: &str) -> PyResult<String> {
    let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(error)?;
    py.detach(|| {
        catch_unwind_py(|| openferric_core::credit::isda::previous_imm_twentieth(date))
            .map(|date| date.format("%Y-%m-%d").to_string())
    })
}

#[pyfunction]
fn step_in_date_with_calendar(
    py: Python<'_>,
    valuation_date: &str,
    _calendar: &crate::rates::Calendar,
) -> PyResult<String> {
    let valuation_date =
        chrono::NaiveDate::parse_from_str(valuation_date, "%Y-%m-%d").map_err(error)?;
    let _calendar = _calendar.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::credit::isda::step_in_date_with_calendar(valuation_date, &_calendar)
        })
        .map(|date| date.format("%Y-%m-%d").to_string())
    })
}

#[pyfunction]
fn black_scholes(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::black_scholes(
                option_type,
                spot,
                strike,
                rate,
                vol,
                expiry,
            )
        })?
        .map_err(error)
    })
}

#[pyfunction]
fn bs_delta(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::bs_delta(
                option_type,
                spot,
                strike,
                rate,
                dividend_yield,
                vol,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn bs_gamma(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::bs_gamma(
                spot,
                strike,
                rate,
                dividend_yield,
                vol,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn bs_rho(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::bs_rho(
                option_type,
                spot,
                strike,
                rate,
                dividend_yield,
                vol,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn bs_theta(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::bs_theta(
                option_type,
                spot,
                strike,
                rate,
                dividend_yield,
                vol,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn bs_vega(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::black_scholes::bs_vega(
                spot,
                strike,
                rate,
                dividend_yield,
                vol,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn norm_cdf(py: Python<'_>, value: f64) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| openferric_core::engines::analytic::black_scholes::norm_cdf(value))
    })
}

#[pyfunction]
fn norm_pdf(py: Python<'_>, value: f64) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| openferric_core::engines::analytic::black_scholes::norm_pdf(value))
    })
}

#[pyfunction]
fn bs_price_asm(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::analytic::bs_inline::bs_price_asm(
                spot,
                strike,
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
fn has_fma_bs_kernel(py: Python<'_>) -> PyResult<bool> {
    py.detach(|| catch_unwind_py(openferric_core::engines::analytic::bs_inline::has_fma_bs_kernel))
}

#[pyfunction]
fn normal_cdf_approx(py: Python<'_>, value: f64) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| openferric_core::engines::analytic::bs_simd::normal_cdf_approx(value))
    })
}

#[pyfunction]
fn heston_price_fft(
    py: Python<'_>,
    spot: f64,
    strike_grid: Vec<f64>,
    rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> PyResult<Vec<(f64, f64)>> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::fft::carr_madan::heston_price_fft(
                spot,
                &strike_grid,
                rate,
                dividend_yield,
                v0,
                kappa,
                theta,
                sigma_v,
                rho,
                maturity,
            )
        })
    })
}

#[pyfunction]
fn try_heston_price_fft(
    py: Python<'_>,
    spot: f64,
    strike_grid: Vec<f64>,
    rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> PyResult<Vec<(f64, f64)>> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::engines::fft::carr_madan::try_heston_price_fft(
                spot,
                &strike_grid,
                rate,
                dividend_yield,
                v0,
                kappa,
                theta,
                sigma_v,
                rho,
                maturity,
            )
        })?
        .map_err(error)
    })
}

#[pyfunction]
fn bootstrap_dividend_curve_from_put_call_parity(
    py: Python<'_>,
    spot: f64,
    rate: f64,
    quotes: Vec<crate::market::PutCallParityQuote>,
) -> PyResult<crate::market::DividendCurveBootstrap> {
    let quotes = quotes
        .iter()
        .map(|value| value.to_core())
        .collect::<Vec<_>>();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::market::dividends::bootstrap_dividend_curve_from_put_call_parity(
                spot, rate, &quotes,
            )
        })?
        .map_err(error)
        .map(crate::market::DividendCurveBootstrap::from_core)
    })
}

#[pyfunction]
fn canonical_pair(py: Python<'_>, ccy_a: &str, ccy_b: &str) -> PyResult<(String, String)> {
    py.detach(|| {
        catch_unwind_py(|| openferric_core::market::fx::canonical_pair(ccy_a, ccy_b))?
            .map_err(error)
    })
}

#[pyfunction]
fn convert_premium(
    py: Python<'_>,
    amount: f64,
    from_currency: crate::market::PremiumCurrency,
    to_currency: crate::market::PremiumCurrency,
    spot: f64,
) -> PyResult<f64> {
    let from_currency = from_currency.to_core();
    let to_currency = to_currency.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::market::fx::convert_premium(amount, from_currency, to_currency, spot)
        })?
        .map_err(error)
    })
}

#[pyfunction]
fn fx_option_premium(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    vol: f64,
    expiry: f64,
    premium_currency: crate::market::PremiumCurrency,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    let premium_currency = premium_currency.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::market::fx::fx_option_premium(
                option_type,
                spot,
                strike,
                domestic_rate,
                foreign_rate,
                vol,
                expiry,
                premium_currency,
            )
        })?
        .map_err(error)
    })
}

#[pyfunction]
fn crr_binomial_american(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    expiry: f64,
    steps: usize,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::american::crr_binomial_american(
                option_type,
                spot,
                strike,
                rate,
                sigma,
                expiry,
                steps,
            )
        })
    })
}

#[pyfunction]
fn price_autocallable_with_greeks(
    py: Python<'_>,
    autocall: &crate::instruments::Autocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividend_yield: f64,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<crate::core::PricingResult> {
    let autocall = autocall.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::autocallable::price_autocallable_with_greeks(
                &autocall,
                &spots,
                &vols,
                &corr_matrix,
                rate,
                dividend_yield,
                n_paths,
                n_steps,
            )
        })
        .map(Into::into)
    })
}

#[pyfunction]
fn price_phoenix_autocallable_with_greeks(
    py: Python<'_>,
    phoenix: &crate::instruments::PhoenixAutocallable,
    spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    rate: f64,
    dividend_yield: f64,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<crate::core::PricingResult> {
    let phoenix = phoenix.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::autocallable::price_phoenix_autocallable_with_greeks(
                &phoenix,
                &spots,
                &vols,
                &corr_matrix,
                rate,
                dividend_yield,
                n_paths,
                n_steps,
            )
        })
        .map(Into::into)
    })
}

#[pyfunction]
fn barrier_price_closed_form_with_carry_and_rebate(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    style: crate::core::BarrierStyle,
    direction: crate::core::BarrierDirection,
    spot: f64,
    strike: f64,
    barrier: f64,
    rate: f64,
    dividend_yield: f64,
    sigma: f64,
    expiry: f64,
    rebate: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    let style = style.to_core();
    let direction = direction.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate(
                option_type,
                style,
                direction,
                spot,
                strike,
                barrier,
                rate,
                dividend_yield,
                sigma,
                expiry,
                rebate,
            )
        })
    })
}

#[pyfunction]
fn longstaff_schwartz_bermudan(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    expiry: f64,
    steps: usize,
    exercise_steps: Vec<usize>,
    num_paths: usize,
    seed: u64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::bermudan::longstaff_schwartz_bermudan(
                option_type,
                spot,
                strike,
                rate,
                sigma,
                expiry,
                steps,
                &exercise_steps,
                num_paths,
                seed,
            )
        })
    })
}

#[pyfunction]
fn black_76_price(
    py: Python<'_>,
    option_type: crate::core::OptionType,
    forward: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    expiry: f64,
) -> PyResult<f64> {
    let option_type = option_type.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::pricing::european::black_76_price(
                option_type,
                forward,
                strike,
                rate,
                sigma,
                expiry,
            )
        })
    })
}

#[pyfunction]
fn historical_expected_shortfall(py: Python<'_>, pnl: Vec<f64>, confidence: f64) -> PyResult<f64> {
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::risk::var::historical_expected_shortfall(&pnl, confidence)
        })
    })
}

#[pyfunction]
fn vanna_volga_pivot_strikes(
    py: Python<'_>,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    quote: crate::vol::VannaVolgaQuote,
) -> PyResult<(f64, f64, f64)> {
    let quote = quote.to_core();
    py.detach(|| {
        catch_unwind_py(|| {
            openferric_core::vol::smile::vanna_volga_pivot_strikes(
                spot,
                rate,
                dividend_yield,
                expiry,
                quote,
            )
        })
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(market_fx_delta, module)?)?;
    module.add_function(wrap_pyfunction!(cash_settle_date, module)?)?;
    module.add_function(wrap_pyfunction!(step_in_date, module)?)?;
    module.add_function(wrap_pyfunction!(atm_strike, module)?)?;
    module.add_function(wrap_pyfunction!(strike_from_delta, module)?)?;
    module.add_function(wrap_pyfunction!(
        bootstrap_survival_curve_from_cds_spreads,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(vasicek_portfolio_loss_cdf, module)?)?;
    module.add_function(wrap_pyfunction!(first_to_default_spread_copula, module)?)?;
    module.add_function(wrap_pyfunction!(fair_spread_from_hazard, module)?)?;
    module.add_function(wrap_pyfunction!(risky_annuity, module)?)?;
    module.add_function(wrap_pyfunction!(cash_settle_date_with_calendar, module)?)?;
    module.add_function(wrap_pyfunction!(generate_imm_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(hazard_from_par_spread, module)?)?;
    module.add_function(wrap_pyfunction!(next_imm_twentieth, module)?)?;
    module.add_function(wrap_pyfunction!(previous_imm_twentieth, module)?)?;
    module.add_function(wrap_pyfunction!(step_in_date_with_calendar, module)?)?;
    module.add_function(wrap_pyfunction!(black_scholes, module)?)?;
    module.add_function(wrap_pyfunction!(bs_delta, module)?)?;
    module.add_function(wrap_pyfunction!(bs_gamma, module)?)?;
    module.add_function(wrap_pyfunction!(bs_rho, module)?)?;
    module.add_function(wrap_pyfunction!(bs_theta, module)?)?;
    module.add_function(wrap_pyfunction!(bs_vega, module)?)?;
    module.add_function(wrap_pyfunction!(norm_cdf, module)?)?;
    module.add_function(wrap_pyfunction!(norm_pdf, module)?)?;
    module.add_function(wrap_pyfunction!(bs_price_asm, module)?)?;
    module.add_function(wrap_pyfunction!(has_fma_bs_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(normal_cdf_approx, module)?)?;
    module.add_function(wrap_pyfunction!(heston_price_fft, module)?)?;
    module.add_function(wrap_pyfunction!(try_heston_price_fft, module)?)?;
    module.add_function(wrap_pyfunction!(
        bootstrap_dividend_curve_from_put_call_parity,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(canonical_pair, module)?)?;
    module.add_function(wrap_pyfunction!(convert_premium, module)?)?;
    module.add_function(wrap_pyfunction!(fx_option_premium, module)?)?;
    module.add_function(wrap_pyfunction!(crr_binomial_american, module)?)?;
    module.add_function(wrap_pyfunction!(price_autocallable_with_greeks, module)?)?;
    module.add_function(wrap_pyfunction!(
        price_phoenix_autocallable_with_greeks,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        barrier_price_closed_form_with_carry_and_rebate,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(longstaff_schwartz_bermudan, module)?)?;
    module.add_function(wrap_pyfunction!(black_76_price, module)?)?;
    module.add_function(wrap_pyfunction!(historical_expected_shortfall, module)?)?;
    module.add_function(wrap_pyfunction!(vanna_volga_pivot_strikes, module)?)?;
    Ok(())
}
