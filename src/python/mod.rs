#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

use crate::core::{BarrierDirection, BarrierStyle, ExerciseStyle, OptionType, PricingEngine};
use crate::credit::{Cds, SurvivalCurve};
use crate::engines::analytic::{
    DigitalAnalyticEngine, ExoticAnalyticEngine, GarmanKohlhagenEngine, HestonEngine,
    kirk_spread_price, margrabe_exchange_price,
};
use crate::greeks::black_scholes_merton_greeks;
use crate::instruments::{
    AssetOrNothingOption, CashOrNothingOption, ExoticOption, FxOption, LookbackFixedOption,
    LookbackFloatingOption, SpreadOption, VanillaOption,
};
use crate::market::Market;
use crate::pricing::american::crr_binomial_american;
use crate::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use crate::pricing::european::black_scholes_price;
use crate::rates::YieldCurve;
use crate::vol::implied::implied_vol;
use crate::vol::sabr::SabrParams;

fn parse_option_type(value: &str) -> Option<OptionType> {
    match value.to_ascii_lowercase().as_str() {
        "call" => Some(OptionType::Call),
        "put" => Some(OptionType::Put),
        _ => None,
    }
}

fn parse_barrier_style(value: &str) -> Option<BarrierStyle> {
    match value.to_ascii_lowercase().as_str() {
        "in" => Some(BarrierStyle::In),
        "out" => Some(BarrierStyle::Out),
        _ => None,
    }
}

fn parse_barrier_direction(value: &str) -> Option<BarrierDirection> {
    match value.to_ascii_lowercase().as_str() {
        "up" => Some(BarrierDirection::Up),
        "down" => Some(BarrierDirection::Down),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum DigitalKind {
    CashOrNothing,
    AssetOrNothing,
}

fn parse_digital_kind(value: &str) -> Option<DigitalKind> {
    match value.to_ascii_lowercase().as_str() {
        "cash" | "cash-or-nothing" | "cash_or_nothing" => Some(DigitalKind::CashOrNothing),
        "asset" | "asset-or-nothing" | "asset_or_nothing" => Some(DigitalKind::AssetOrNothing),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum SpreadMethod {
    Kirk,
    Margrabe,
}

fn parse_spread_method(value: &str) -> Option<SpreadMethod> {
    match value.to_ascii_lowercase().as_str() {
        "kirk" => Some(SpreadMethod::Kirk),
        "margrabe" => Some(SpreadMethod::Margrabe),
        _ => None,
    }
}

fn build_market(spot: f64, rate: f64, div_yield: f64, vol: f64) -> Option<Market> {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div_yield)
        .flat_vol(vol.max(1e-8))
        .build()
        .ok()
}

fn tenor_grid(maturity: f64, payment_freq: usize) -> Vec<f64> {
    if maturity <= 0.0 || payment_freq == 0 {
        return vec![];
    }

    let dt = 1.0 / payment_freq as f64;
    let mut times = Vec::new();
    let mut t = 0.0;

    while t + dt < maturity - 1e-12 {
        t += dt;
        times.push(t);
    }
    times.push(maturity);
    times
}

#[pyfunction]
pub fn py_bs_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    black_scholes_price(option_type, spot, strike, rate, vol, expiry)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_bs_greeks(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    greek: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let greeks =
        black_scholes_merton_greeks(option_type, spot, strike, rate, div_yield, vol, expiry);

    match greek.to_ascii_lowercase().as_str() {
        "delta" => greeks.delta,
        "gamma" => greeks.gamma,
        "vega" => greeks.vega,
        "theta" => greeks.theta,
        "rho" => greeks.rho,
        "vanna" => greeks.vanna,
        "volga" | "vomma" => greeks.volga,
        _ => f64::NAN,
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_barrier_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    barrier: f64,
    option_type: &str,
    barrier_type: &str,
    barrier_dir: &str,
    rebate: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    let Some(style) = parse_barrier_style(barrier_type) else {
        return f64::NAN;
    };
    let Some(direction) = parse_barrier_direction(barrier_dir) else {
        return f64::NAN;
    };

    barrier_price_closed_form_with_carry_and_rebate(
        option_type,
        style,
        direction,
        spot,
        strike,
        barrier,
        rate,
        div_yield,
        vol,
        expiry,
        rebate,
    )
}

#[pyfunction]
pub fn py_american_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    option_type: &str,
    steps: usize,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    crr_binomial_american(option_type, spot, strike, rate, vol, expiry, steps.max(1))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_heston_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let instrument = VanillaOption {
        option_type,
        strike,
        expiry,
        exercise: ExerciseStyle::European,
    };

    let Some(market) = build_market(spot, rate, div_yield, v0.abs().sqrt()) else {
        return f64::NAN;
    };

    HestonEngine::new(v0, kappa, theta, sigma_v, rho)
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_fx_price(
    spot_fx: f64,
    strike_fx: f64,
    maturity: f64,
    vol: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    option_type: &str,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let instrument = FxOption::new(
        option_type,
        domestic_rate,
        foreign_rate,
        spot_fx,
        strike_fx,
        vol,
        maturity,
    );

    let Some(market) = build_market(spot_fx, domestic_rate, foreign_rate, vol) else {
        return f64::NAN;
    };

    GarmanKohlhagenEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_digital_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    digital_type: &str,
    cash: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };
    let Some(digital_type) = parse_digital_kind(digital_type) else {
        return f64::NAN;
    };

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    let engine = DigitalAnalyticEngine::new();

    match digital_type {
        DigitalKind::CashOrNothing => {
            let instrument = CashOrNothingOption::new(option_type, strike, cash, expiry);
            engine
                .price(&instrument, &market)
                .map(|x| x.price)
                .unwrap_or(f64::NAN)
        }
        DigitalKind::AssetOrNothing => {
            let instrument = AssetOrNothingOption::new(option_type, strike, expiry);
            engine
                .price(&instrument, &market)
                .map(|x| x.price)
                .unwrap_or(f64::NAN)
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_spread_price(
    s1: f64,
    s2: f64,
    k: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    q1: f64,
    q2: f64,
    r: f64,
    t: f64,
    method: &str,
) -> f64 {
    let Some(method) = parse_spread_method(method) else {
        return f64::NAN;
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
        r,
        t,
    };

    match method {
        SpreadMethod::Kirk => kirk_spread_price(&option),
        SpreadMethod::Margrabe => margrabe_exchange_price(&option),
    }
    .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_lookback_floating(
    spot: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    observed_extreme: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let observed_extreme = if observed_extreme > 0.0 {
        Some(observed_extreme)
    } else {
        None
    };

    let instrument = ExoticOption::LookbackFloating(LookbackFloatingOption {
        option_type,
        expiry,
        observed_extreme,
    });

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    ExoticAnalyticEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_lookback_fixed(
    spot: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    rate: f64,
    div_yield: f64,
    option_type: &str,
    observed_extreme: f64,
) -> f64 {
    let Some(option_type) = parse_option_type(option_type) else {
        return f64::NAN;
    };

    let observed_extreme = if observed_extreme > 0.0 {
        Some(observed_extreme)
    } else {
        None
    };

    let instrument = ExoticOption::LookbackFixed(LookbackFixedOption {
        option_type,
        strike,
        expiry,
        observed_extreme,
    });

    let Some(market) = build_market(spot, rate, div_yield, vol) else {
        return f64::NAN;
    };

    ExoticAnalyticEngine::new()
        .price(&instrument, &market)
        .map(|x| x.price)
        .unwrap_or(f64::NAN)
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
    let Some(option_type) = parse_option_type(option_type) else {
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
pub fn py_sabr_vol(
    forward: f64,
    strike: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    SabrParams {
        alpha,
        beta,
        rho,
        nu,
    }
    .implied_vol(forward, strike, expiry)
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

    let cds = Cds {
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
    let survival_curve = SurvivalCurve::from_piecewise_hazard(&tenors, &hazards);

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
    let curve = SurvivalCurve::from_piecewise_hazard(&tenors, &hazards);
    curve.survival_prob(tt)
}

// ── FFT / Lévy process pricing ──────────────────────────────────────────

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_heston_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
) -> f64 {
    use crate::engines::fft::{CarrMadanParams, HestonCharFn, carr_madan_price_at_strikes};
    let cf = HestonCharFn::new(spot, rate, div_yield, expiry, v0, kappa, theta, sigma_v, rho);
    carr_madan_price_at_strikes(&cf, rate, expiry, spot, &[strike], CarrMadanParams::default())
        .ok()
        .and_then(|v| v.first().map(|(_, p)| *p))
        .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_vg_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    sigma: f64,
    theta_vg: f64,
    nu: f64,
) -> f64 {
    use crate::engines::fft::CarrMadanParams;
    use crate::models::VarianceGamma;
    let vg = VarianceGamma { sigma, theta: theta_vg, nu };
    vg.european_calls_fft(spot, &[strike], rate, div_yield, expiry, CarrMadanParams::default())
        .ok()
        .and_then(|v| v.first().map(|(_, p)| *p))
        .unwrap_or(f64::NAN)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_cgmy_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    c: f64,
    g: f64,
    m: f64,
    y: f64,
) -> f64 {
    use crate::engines::fft::CarrMadanParams;
    use crate::models::Cgmy;
    let cgmy = Cgmy { c, g, m, y };
    cgmy.european_calls_fft(spot, &[strike], rate, div_yield, expiry, CarrMadanParams::default())
        .ok()
        .and_then(|v| v.first().map(|(_, p)| *p))
        .unwrap_or(f64::NAN)
}

#[pyfunction]
pub fn py_nig_fft_price(
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    div_yield: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
) -> f64 {
    use crate::engines::fft::CarrMadanParams;
    use crate::models::Nig;
    let nig = Nig { alpha, beta, delta };
    nig.european_calls_fft(spot, &[strike], rate, div_yield, expiry, CarrMadanParams::default())
        .ok()
        .and_then(|v| v.first().map(|(_, p)| *p))
        .unwrap_or(f64::NAN)
}

// ── Rates ───────────────────────────────────────────────────────────────

#[pyfunction]
pub fn py_swaption_price(
    notional: f64,
    strike: f64,
    swap_tenor: f64,
    option_expiry: f64,
    vol: f64,
    discount_rate: f64,
    option_type: &str,
) -> f64 {
    use crate::rates::swaption::Swaption;
    let is_payer = match option_type.to_ascii_lowercase().as_str() {
        "payer" | "call" => true,
        "receiver" | "put" => false,
        _ => return f64::NAN,
    };
    let swaption = Swaption {
        notional,
        strike,
        option_expiry,
        swap_tenor,
        is_payer,
    };
    let tenors: Vec<(f64, f64)> = (1..=((option_expiry + swap_tenor).ceil() as usize * 4))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-discount_rate * t).exp())
        })
        .collect();
    let curve = YieldCurve::new(tenors);
    swaption.price(&curve, vol)
}

// ── XVA ─────────────────────────────────────────────────────────────────

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_cva(
    times: Vec<f64>,
    ee_profile: Vec<f64>,
    discount_rate: f64,
    hazard_rate: f64,
    lgd: f64,
) -> f64 {
    use crate::risk::XvaCalculator;
    let discount_curve = YieldCurve::new(
        times.iter().map(|t| (*t, (-discount_rate * *t).exp())).collect(),
    );
    let hazards = vec![hazard_rate; times.len()];
    let survival = SurvivalCurve::from_piecewise_hazard(&times, &hazards);
    let own_survival = SurvivalCurve::from_piecewise_hazard(&times, &vec![0.0; times.len()]);
    let calc = XvaCalculator::new(discount_curve, survival, own_survival, lgd, 0.0);
    calc.cva_from_expected_exposure(&times, &ee_profile)
}

#[pyfunction]
pub fn py_sa_ccr_ead(
    replacement_cost: f64,
    notional: f64,
    maturity: f64,
    asset_class: &str,
) -> f64 {
    use crate::risk::kva::{SaCcrAssetClass, sa_ccr_ead};
    let ac = match asset_class.to_ascii_lowercase().as_str() {
        "ir" | "interest_rate" => SaCcrAssetClass::InterestRate,
        "fx" | "foreign_exchange" => SaCcrAssetClass::ForeignExchange,
        "credit" => SaCcrAssetClass::Credit,
        "equity" => SaCcrAssetClass::Equity,
        "commodity" => SaCcrAssetClass::Commodity,
        _ => return f64::NAN,
    };
    sa_ccr_ead(replacement_cost, notional, maturity, ac)
}

// ── VaR ─────────────────────────────────────────────────────────────────

#[pyfunction]
pub fn py_historical_var(returns: Vec<f64>, confidence: f64) -> f64 {
    use crate::risk::var::historical_var;
    historical_var(&returns, confidence)
}

#[pyfunction]
pub fn py_historical_es(returns: Vec<f64>, confidence: f64) -> f64 {
    use crate::risk::var::historical_expected_shortfall;
    historical_expected_shortfall(&returns, confidence)
}

// ── Vol surface ─────────────────────────────────────────────────────────

#[pyfunction]
pub fn py_svi_vol(
    strike: f64,
    forward: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
) -> f64 {
    let k = (strike / forward).ln();
    let total_var = a + b * (rho * (k - m) + ((k - m).powi(2) + sigma * sigma).sqrt());
    if total_var > 0.0 { total_var.sqrt() } else { f64::NAN }
}

#[pymodule]
pub fn openferric(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_bs_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_bs_greeks, module)?)?;
    module.add_function(wrap_pyfunction!(py_barrier_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_american_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_heston_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_fx_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_digital_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_spread_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_lookback_floating, module)?)?;
    module.add_function(wrap_pyfunction!(py_lookback_fixed, module)?)?;
    module.add_function(wrap_pyfunction!(py_implied_vol, module)?)?;
    module.add_function(wrap_pyfunction!(py_sabr_vol, module)?)?;
    module.add_function(wrap_pyfunction!(py_cds_npv, module)?)?;
    module.add_function(wrap_pyfunction!(py_survival_prob, module)?)?;
    // FFT / Lévy
    module.add_function(wrap_pyfunction!(py_heston_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_vg_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_cgmy_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(py_nig_fft_price, module)?)?;
    // Rates
    module.add_function(wrap_pyfunction!(py_swaption_price, module)?)?;
    // XVA / Risk
    module.add_function(wrap_pyfunction!(py_cva, module)?)?;
    module.add_function(wrap_pyfunction!(py_sa_ccr_ead, module)?)?;
    module.add_function(wrap_pyfunction!(py_historical_var, module)?)?;
    module.add_function(wrap_pyfunction!(py_historical_es, module)?)?;
    // Vol
    module.add_function(wrap_pyfunction!(py_svi_vol, module)?)?;
    Ok(())
}
