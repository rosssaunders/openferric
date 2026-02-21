use wasm_bindgen::prelude::*;

use crate::helpers::{heston_intrinsic, option_price_from_call};

/// Heston European option price via FFT. Put prices are computed by parity.
#[wasm_bindgen]
pub fn heston_price(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
    is_call: bool,
) -> f64 {
    if maturity <= 0.0 {
        return heston_intrinsic(spot, strike, is_call);
    }

    let call_price = heston_fft_prices(
        spot,
        &[strike],
        rate,
        div_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        maturity,
    )
    .into_iter()
    .next()
    .unwrap_or(f64::NAN);

    if !call_price.is_finite() {
        return f64::NAN;
    }

    option_price_from_call(call_price, spot, strike, rate, div_yield, maturity, is_call)
}

/// Heston FFT prices for a strike array.
///
/// Returns a vector of call prices matching input strike order.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn heston_fft_prices(
    spot: f64,
    strikes: &[f64],
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> Vec<f64> {
    use openferric::engines::fft::{CarrMadanContext, CarrMadanParams, HestonCharFn};

    if strikes.is_empty() {
        return Vec::new();
    }

    let cf = HestonCharFn::new(
        spot, rate, div_yield, maturity, v0, kappa, theta, sigma_v, rho,
    );
    let ctx = match CarrMadanContext::new(&cf, rate, maturity, spot, CarrMadanParams::default()) {
        Ok(ctx) => ctx,
        Err(_) => return vec![f64::NAN; strikes.len()],
    };

    ctx.price_strikes(strikes)
        .map(|pairs| pairs.into_iter().map(|(_, p)| p).collect())
        .unwrap_or_else(|_| vec![f64::NAN; strikes.len()])
}

/// Heston FFT price for a single strike (convenience wrapper).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn heston_fft_price(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> f64 {
    heston_fft_prices(
        spot,
        &[strike],
        rate,
        div_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        maturity,
    )
    .into_iter()
    .next()
    .unwrap_or(f64::NAN)
}
