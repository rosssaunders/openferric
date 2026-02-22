use pyo3::prelude::*;

use openferric_core::core::PricingEngine;
use openferric_core::engines::analytic::{
    DigitalAnalyticEngine, ExoticAnalyticEngine, GarmanKohlhagenEngine, kirk_spread_price,
    margrabe_exchange_price,
};
use openferric_core::greeks::black_scholes_merton_greeks;
use openferric_core::instruments::{
    AssetOrNothingOption, CashOrNothingOption, ExoticOption, FxOption, LookbackFixedOption,
    LookbackFloatingOption, SpreadOption,
};
use openferric_core::pricing::american::crr_binomial_american;
use openferric_core::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use openferric_core::pricing::european::black_scholes_price;

use crate::fft::heston_fft_prices_cached;
use crate::helpers::{
    DigitalKind, SpreadMethod, build_market, intrinsic_from_option_type, option_price_from_call,
    parse_barrier_direction, parse_barrier_style, parse_digital_kind, parse_option_type,
    parse_spread_method,
};

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
    if expiry <= 0.0 {
        return intrinsic_from_option_type(option_type, spot, strike);
    }

    let call_price = heston_fft_prices_cached(
        spot,
        &[strike],
        expiry,
        rate,
        div_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
    )
    .and_then(|pairs| pairs.first().map(|(_, p)| *p))
    .unwrap_or(f64::NAN);

    if !call_price.is_finite() {
        return f64::NAN;
    }

    option_price_from_call(
        option_type,
        call_price,
        spot,
        strike,
        rate,
        div_yield,
        expiry,
    )
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
