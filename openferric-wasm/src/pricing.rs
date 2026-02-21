use wasm_bindgen::prelude::*;

use openferric::core::types::{BarrierDirection, BarrierStyle, OptionType};
use openferric::greeks::black_scholes_merton_greeks;
use openferric::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use openferric::pricing::european::black_scholes_price;
use openferric::vol::implied::implied_vol;

/// Black-Scholes European option price.
#[wasm_bindgen]
pub fn bs_price(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    maturity: f64,
    is_call: bool,
) -> f64 {
    let ot = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    // Adjust for continuous dividend yield: S_adj = S * e^{-q*T}
    let s_adj = spot * (-div_yield * maturity).exp();
    black_scholes_price(ot, s_adj, strike, rate, vol, maturity)
}

/// Black-Scholes implied volatility.
#[wasm_bindgen]
pub fn bs_implied_vol(
    price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    maturity: f64,
    is_call: bool,
) -> f64 {
    let ot = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let s_adj = spot * (-div_yield * maturity).exp();
    implied_vol(ot, s_adj, strike, rate, maturity, price, 1e-12, 64).unwrap_or(f64::NAN)
}

/// BSM Greeks: returns [delta, gamma, vega, theta, rho, vanna, volga].
#[wasm_bindgen]
pub fn bsm_greeks_wasm(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> Vec<f64> {
    let option_type = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let g = black_scholes_merton_greeks(option_type, spot, strike, rate, div_yield, vol, expiry);
    vec![g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga]
}

/// Batch Black-Scholes pricing: one WASM call for N options.
///
/// All input slices must have the same length.  `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a `Vec<f64>` of prices in the same order.
#[wasm_bindgen]
pub fn bs_price_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rates: &[f64],
    div_yields: &[f64],
    vols: &[f64],
    maturities: &[f64],
    is_calls: &[u8],
) -> Vec<f64> {
    let n = spots.len();
    debug_assert!(
        n == strikes.len()
            && n == rates.len()
            && n == div_yields.len()
            && n == vols.len()
            && n == maturities.len()
            && n == is_calls.len()
    );

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ot = if is_calls[i] != 0 {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let s_adj = spots[i] * (-div_yields[i] * maturities[i]).exp();
        out.push(black_scholes_price(ot, s_adj, strikes[i], rates[i], vols[i], maturities[i]));
    }
    out
}

/// Batch BSM Greeks: one WASM call for N options.
///
/// All input slices must have the same length.  `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a flat `Vec<f64>` of 7 values per option:
/// `[delta, gamma, vega, theta, rho, vanna, volga]` repeated N times.
#[wasm_bindgen]
pub fn bsm_greeks_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rates: &[f64],
    div_yields: &[f64],
    vols: &[f64],
    expiries: &[f64],
    is_calls: &[u8],
) -> Vec<f64> {
    let n = spots.len();
    debug_assert!(
        n == strikes.len()
            && n == rates.len()
            && n == div_yields.len()
            && n == vols.len()
            && n == expiries.len()
            && n == is_calls.len()
    );

    let mut out = Vec::with_capacity(n * 7);
    for i in 0..n {
        let ot = if is_calls[i] != 0 {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let g = black_scholes_merton_greeks(
            ot, spots[i], strikes[i], rates[i], div_yields[i], vols[i], expiries[i],
        );
        out.push(g.delta);
        out.push(g.gamma);
        out.push(g.vega);
        out.push(g.theta);
        out.push(g.rho);
        out.push(g.vanna);
        out.push(g.volga);
    }
    out
}

/// Barrier option price (closed-form).
///
/// `barrier_type` is one of: "up-in", "up-out", "down-in", "down-out".
#[wasm_bindgen]
pub fn barrier_price(
    spot: f64,
    strike: f64,
    barrier: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    maturity: f64,
    barrier_type: &str,
    is_call: bool,
) -> f64 {
    let ot = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let (direction, style) = match barrier_type.to_lowercase().as_str() {
        "up-in" => (BarrierDirection::Up, BarrierStyle::In),
        "up-out" => (BarrierDirection::Up, BarrierStyle::Out),
        "down-in" => (BarrierDirection::Down, BarrierStyle::In),
        "down-out" => (BarrierDirection::Down, BarrierStyle::Out),
        _ => return f64::NAN,
    };
    barrier_price_closed_form_with_carry_and_rebate(
        ot, style, direction, spot, strike, barrier, rate, div_yield, vol, maturity, 0.0,
    )
}

/// Simple fixed-rate bond dirty price using flat yield discounting.
#[wasm_bindgen]
pub fn bond_price(
    face_value: f64,
    coupon_rate: f64,
    maturity_years: f64,
    yield_rate: f64,
    frequency: u32,
) -> f64 {
    if frequency == 0 || maturity_years <= 0.0 {
        return f64::NAN;
    }
    let freq = frequency as f64;
    let coupon = face_value * coupon_rate / freq;
    let n_periods = (maturity_years * freq).round() as u32;
    let r_per = yield_rate / freq;

    let mut pv = 0.0;
    for i in 1..=n_periods {
        let df = (1.0 + r_per).powi(-(i as i32));
        pv += coupon * df;
    }
    pv += face_value * (1.0 + r_per).powi(-(n_periods as i32));
    pv
}
