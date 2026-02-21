use wasm_bindgen::prelude::*;

use openferric::vol::sabr::{SabrParams, fit_sabr};
use openferric::vol::surface::{SviParams, calibrate_svi};

// ---------------------------------------------------------------------------
// SVI calibration
// ---------------------------------------------------------------------------

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

/// Calibrate SVI parameters from flat `[k0, w0, k1, w1, ...]` pairs.
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

// ---------------------------------------------------------------------------
// SABR
// ---------------------------------------------------------------------------

/// SABR parameters exposed to JavaScript via wasm-bindgen.
#[wasm_bindgen]
pub struct WasmSabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
}

#[wasm_bindgen]
impl WasmSabrParams {
    /// Compute SABR implied vol for a given forward, strike, and time to expiry.
    pub fn implied_vol(&self, forward: f64, strike: f64, t: f64) -> f64 {
        let inner = SabrParams {
            alpha: self.alpha,
            beta: self.beta,
            rho: self.rho,
            nu: self.nu,
        };
        inner.implied_vol(forward, strike, t)
    }
}

/// SABR implied volatility from parameters.
#[wasm_bindgen]
pub fn sabr_vol(
    forward: f64,
    strike: f64,
    t: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    let params = SabrParams {
        alpha,
        beta,
        rho,
        nu,
    };
    params.implied_vol(forward, strike, t)
}

/// Calibrate SABR parameters from flat arrays.
#[wasm_bindgen]
pub fn fit_sabr_wasm(
    forward: f64,
    strikes_flat: &[f64],
    vols_flat: &[f64],
    t: f64,
    beta: f64,
) -> WasmSabrParams {
    let result = fit_sabr(forward, strikes_flat, vols_flat, t, beta);
    WasmSabrParams {
        alpha: result.alpha,
        beta: result.beta,
        rho: result.rho,
        nu: result.nu,
    }
}

// ---------------------------------------------------------------------------
// IV slice operations — thin wrappers delegating to vol::slice
// ---------------------------------------------------------------------------

/// Compute a full IV grid: n_slices rows x n_k columns (row-major).
#[wasm_bindgen]
pub fn iv_grid(slice_headers: &[f64], slice_params: &[f64], k_grid: &[f64]) -> Vec<f64> {
    openferric::vol::slice::iv_grid(slice_headers, slice_params, k_grid)
}

/// Batch IV evaluation for irregular (per-option) lookups.
#[wasm_bindgen]
pub fn batch_slice_iv(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_values: &[f64],
    slice_indices: &[u32],
) -> Vec<f64> {
    openferric::vol::slice::batch_slice_iv(slice_headers, slice_params, k_values, slice_indices)
}

/// Compute fit diagnostics for a single slice.
#[wasm_bindgen]
pub fn slice_fit_diagnostics(
    model_type: u8,
    params: &[f64],
    t: f64,
    forward: f64,
    market_ks: &[f64],
    market_ivs_pct: &[f64],
    strikes: &[f64],
) -> Vec<f64> {
    openferric::vol::slice::slice_fit_diagnostics(
        model_type,
        params,
        t,
        forward,
        market_ks,
        market_ivs_pct,
        strikes,
    )
}

/// Find 25-delta strikes for all slices via Newton's method.
#[wasm_bindgen]
pub fn find_25d_strikes_batch(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    openferric::vol::slice::find_25d_strikes_batch(slice_headers, slice_params)
}

/// Combined term structure computation: 25-delta strikes + ATM IV + RR25/BF25.
#[wasm_bindgen]
pub fn term_structure_batch_wasm(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    openferric::vol::slice::term_structure_batch(slice_headers, slice_params)
}

/// Forward vol grid: spot vol, forward vol, and forward skew for adjacent slice pairs.
#[wasm_bindgen]
pub fn forward_vol_grid_wasm(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_points: &[f64],
) -> Vec<f64> {
    openferric::vol::slice::forward_vol_grid(slice_headers, slice_params, k_points)
}

// ---------------------------------------------------------------------------
// Slice calibration — delegates to calibrate module
// ---------------------------------------------------------------------------

/// Calibrate a vol slice from raw market quotes. Returns packed flat array.
///
/// See `calibrate::calibrate_slice` for return format.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn calibrate_slice_wasm(
    strikes: &[f64],
    mark_ivs: &[f64],
    bid_ivs: &[f64],
    ask_ivs: &[f64],
    open_interests: &[f64],
    forward: f64,
    t: f64,
    model_type: u8,
) -> Vec<f64> {
    crate::calibrate::calibrate_slice(
        strikes,
        mark_ivs,
        bid_ivs,
        ask_ivs,
        open_interests,
        forward,
        t,
        model_type,
    )
}

/// Batch log-moneyness: `ln(strikes[i] / forwards[i])`.
/// If `forwards` has length 1, it broadcasts to all strikes.
#[wasm_bindgen]
pub fn log_moneyness_batch_wasm(strikes: &[f64], forwards: &[f64]) -> Vec<f64> {
    openferric::vol::slice::log_moneyness_batch(strikes, forwards)
}

/// Batch log returns: `ln(prices[i] / prices[i-1])` for i=1..n.
#[wasm_bindgen]
pub fn log_returns_batch_wasm(prices: &[f64]) -> Vec<f64> {
    openferric::vol::slice::log_returns_batch(prices)
}

/// Annualized realized volatility from log returns, returned as percentage.
#[wasm_bindgen]
pub fn realized_vol(log_returns: &[f64], obs_per_year: f64) -> f64 {
    openferric::vol::slice::realized_vol(log_returns, obs_per_year)
}
