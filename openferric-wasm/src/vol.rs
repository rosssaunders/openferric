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

#[cfg(test)]
mod tests {
    use super::*;

    // -- sabr_vol --

    #[test]
    fn sabr_vol_atm_approx_alpha() {
        let vol = sabr_vol(100.0, 100.0, 1.0, 0.20, 1.0, -0.3, 0.4);
        assert!((vol - 0.20).abs() < 0.02);
    }

    #[test]
    fn sabr_vol_smile_skew() {
        let vol_low = sabr_vol(100.0, 80.0, 1.0, 0.20, 0.5, -0.4, 0.6);
        let vol_atm = sabr_vol(100.0, 100.0, 1.0, 0.20, 0.5, -0.4, 0.6);
        // Negative rho → downside skew
        assert!(vol_low > vol_atm);
    }

    #[test]
    fn sabr_vol_positive() {
        let vol = sabr_vol(100.0, 100.0, 1.0, 0.25, 0.5, 0.0, 0.3);
        assert!(vol > 0.0);
    }

    // -- fit_sabr_wasm --

    #[test]
    fn fit_sabr_round_trip() {
        let alpha = 0.25;
        let beta = 0.5;
        let rho = -0.2;
        let nu = 0.4;
        let forward = 100.0;
        let t = 1.0;
        let strikes: Vec<f64> = (80..=120).step_by(5).map(|s| s as f64).collect();
        let vols: Vec<f64> = strikes
            .iter()
            .map(|&k| sabr_vol(forward, k, t, alpha, beta, rho, nu))
            .collect();
        let result = fit_sabr_wasm(forward, &strikes, &vols, t, beta);
        assert!((result.alpha - alpha).abs() < 0.05);
        // beta is fixed
        assert!((result.beta - beta).abs() < 1e-10);
    }

    // -- WasmSviParams --

    #[test]
    fn svi_total_variance_atm() {
        let params = WasmSviParams { a: 0.04, b: 0.1, rho: 0.0, m: 0.0, sigma: 0.1 };
        let tv = params.total_variance(0.0); // k=0 (ATM)
        let expected = 0.04 + 0.1 * 0.1; // a + b*sigma when rho=0, k=m=0
        assert!((tv - expected).abs() < 1e-10);
    }

    #[test]
    fn svi_total_variance_positive_wings() {
        let params = WasmSviParams { a: 0.04, b: 0.1, rho: -0.2, m: 0.0, sigma: 0.1 };
        let tv_low = params.total_variance(-0.3);
        let tv_high = params.total_variance(0.3);
        assert!(tv_low > 0.0);
        assert!(tv_high > 0.0);
    }

    #[test]
    fn svi_dw_dk_finite() {
        let params = WasmSviParams { a: 0.04, b: 0.1, rho: -0.2, m: 0.0, sigma: 0.1 };
        let dw = params.dw_dk(0.0);
        assert!(dw.is_finite());
    }

    // -- calibrate_svi_wasm --

    #[test]
    fn calibrate_svi_wasm_recovers_shape() {
        let a_true = 0.04;
        let b_true = 0.10;
        let rho_true = -0.2;
        let m_true = 0.0;
        let sigma_true = 0.1;
        let svi = SviParams { a: a_true, b: b_true, rho: rho_true, m: m_true, sigma: sigma_true };
        let mut points_flat = Vec::new();
        for k in (-5..=5).map(|i| i as f64 * 0.05) {
            let tv = svi.total_variance(k);
            points_flat.push(k);
            points_flat.push(tv);
        }
        let result = calibrate_svi_wasm(&points_flat, 0.04, 0.1, -0.1, 0.0, 0.15, 2000, 0.002);
        // Check it reproduces total variance at ATM
        let tv_atm = result.total_variance(0.0);
        let expected_atm = svi.total_variance(0.0);
        assert!((tv_atm - expected_atm).abs() < 0.01);
    }

    // -- log_moneyness_batch_wasm --

    #[test]
    fn log_moneyness_atm_is_zero() {
        let km = log_moneyness_batch_wasm(&[100.0], &[100.0]);
        assert!((km[0]).abs() < 1e-10);
    }

    #[test]
    fn log_moneyness_otm() {
        let km = log_moneyness_batch_wasm(&[110.0], &[100.0]);
        let expected = (110.0_f64 / 100.0).ln();
        assert!((km[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn log_moneyness_broadcast_forward() {
        let km = log_moneyness_batch_wasm(&[90.0, 100.0, 110.0], &[100.0]);
        assert_eq!(km.len(), 3);
        assert!(km[0] < 0.0);
        assert!((km[1]).abs() < 1e-10);
        assert!(km[2] > 0.0);
    }

    // -- log_returns_batch_wasm --

    #[test]
    fn log_returns_basic() {
        let prices = [100.0, 110.0, 105.0];
        let lr = log_returns_batch_wasm(&prices);
        assert_eq!(lr.len(), 2);
        assert!((lr[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((lr[1] - (105.0_f64 / 110.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn log_returns_single_price_empty() {
        let lr = log_returns_batch_wasm(&[100.0]);
        assert!(lr.is_empty());
    }

    // -- realized_vol --

    #[test]
    fn realized_vol_positive() {
        let log_ret = [0.01, -0.02, 0.015, -0.005, 0.008];
        let rv = realized_vol(&log_ret, 252.0);
        assert!(rv > 0.0);
    }

    #[test]
    fn realized_vol_zero_returns() {
        let log_ret = [0.0, 0.0, 0.0];
        let rv = realized_vol(&log_ret, 252.0);
        assert!(rv.abs() < 1e-10 || rv == 0.0);
    }
}
