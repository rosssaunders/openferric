use wasm_bindgen::prelude::*;

use crate::greeks::black_scholes_merton_greeks;
use crate::pricing::OptionType;
use crate::vol::surface::{SviParams, calibrate_svi};

/// SVI parameters exposed to JavaScript via wasm-bindgen.
#[wasm_bindgen]
pub struct WasmSviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

#[wasm_bindgen]
impl WasmSviParams {
    /// Evaluate total variance at log-moneyness `k`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let inner = SviParams {
            a: self.a,
            b: self.b,
            rho: self.rho,
            m: self.m,
            sigma: self.sigma,
        };
        inner.total_variance(k)
    }

    /// First derivative dw/dk at log-moneyness `k`.
    pub fn dw_dk(&self, k: f64) -> f64 {
        let inner = SviParams {
            a: self.a,
            b: self.b,
            rho: self.rho,
            m: self.m,
            sigma: self.sigma,
        };
        inner.dw_dk(k)
    }
}

/// Calibrate SVI parameters from a flat array of `[k0, w0, k1, w1, ...]` pairs.
///
/// Returns a `WasmSviParams` struct with calibrated parameters.
#[wasm_bindgen]
pub fn calibrate_svi_wasm(
    points_flat: &[f64],
    init_a: f64,
    init_b: f64,
    init_rho: f64,
    init_m: f64,
    init_sigma: f64,
    max_iter: u32,
    learning_rate: f64,
) -> WasmSviParams {
    let n = points_flat.len() / 2;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        points.push((points_flat[i * 2], points_flat[i * 2 + 1]));
    }

    let init = SviParams {
        a: init_a,
        b: init_b,
        rho: init_rho,
        m: init_m,
        sigma: init_sigma,
    };

    let result = calibrate_svi(&points, init, max_iter as usize, learning_rate);

    WasmSviParams {
        a: result.a,
        b: result.b,
        rho: result.rho,
        m: result.m,
        sigma: result.sigma,
    }
}

/// Compute BSM Greeks for a single option.
///
/// Returns a `Float64Array` of `[delta, gamma, vega, theta, rho, vanna, volga]`.
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
