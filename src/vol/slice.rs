//! Module `vol::slice`.
//!
//! Implements slice workflows with concrete routines such as `eval_iv_pct`, `parse_slice`, `solve_delta_k`, `iv_grid`.
//!
//! References: Gatheral (2006), Derman and Kani (1994), static-arbitrage constraints around total variance Eq. (2.2).
//!
//! Primary API surface: free functions `eval_iv_pct`, `parse_slice`, `solve_delta_k`, `iv_grid`.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.

use crate::vol::sabr::SabrParams;
use crate::vol::surface::SviParams;

// ---------------------------------------------------------------------------
// Model type constants
// ---------------------------------------------------------------------------

/// SVI model type code.
pub const MODEL_SVI: u8 = 0;
/// SABR model type code.
pub const MODEL_SABR: u8 = 1;
/// Vanna-volga model type code.
pub const MODEL_VV: u8 = 2;

// ---------------------------------------------------------------------------
// Core dispatcher
// ---------------------------------------------------------------------------

/// Evaluate IV% for a single (model, params, k, T, forward) tuple.
///
/// Dispatches to SVI, SABR, or vanna-volga depending on `model_type`.
#[inline]
pub fn eval_iv_pct(model_type: u8, params: &[f64], k: f64, t: f64, forward: f64) -> f64 {
    match model_type {
        MODEL_SVI => {
            // params: [a, b, rho, m, sigma]
            if params.len() < 5 || t <= 0.0 {
                return f64::NAN;
            }
            let svi = SviParams {
                a: params[0],
                b: params[1],
                rho: params[2],
                m: params[3],
                sigma: params[4],
            };
            let w = svi.total_variance(k);
            (w.max(1e-12) / t).sqrt() * 100.0
        }
        MODEL_SABR => {
            // params: [alpha, beta, rho, nu]
            if params.len() < 4 || t <= 0.0 || forward <= 0.0 {
                return f64::NAN;
            }
            let sabr = SabrParams {
                alpha: params[0],
                beta: params[1],
                rho: params[2],
                nu: params[3],
            };
            let strike = forward * k.exp();
            sabr.implied_vol(forward, strike, t) * 100.0
        }
        MODEL_VV => {
            // params: [atmVol, rr25, bf25]
            if params.len() < 3 || t <= 0.0 {
                return f64::NAN;
            }
            let atm_vol = params[0];
            let rr25 = params[1];
            let bf25 = params[2];
            let scale = (atm_vol * t.sqrt()).max(1e-8);
            let vanna_w = k / scale;
            let volga_w = (k * k) / (scale * scale);
            let vol = atm_vol + vanna_w * rr25 * 0.5 + volga_w * bf25;
            vol.max(1e-8) * 100.0
        }
        _ => f64::NAN,
    }
}

// ---------------------------------------------------------------------------
// Header parser
// ---------------------------------------------------------------------------

/// Parse a single slice from the header/params arrays.
///
/// Header layout per slice: `[model_type, T, forward, param_offset]` (4 f64 each).
/// Returns `(model_type, param_slice, T, forward)`.
#[inline]
pub fn parse_slice<'a>(
    headers: &[f64],
    params: &'a [f64],
    idx: usize,
    n_slices: usize,
) -> (u8, &'a [f64], f64, f64) {
    let base = idx * 4;
    let model_type = headers[base] as u8;
    let t = headers[base + 1];
    let forward = headers[base + 2];
    let param_offset = headers[base + 3] as usize;
    let param_len = match model_type {
        MODEL_SVI => 5,
        MODEL_SABR => 4,
        MODEL_VV => 3,
        _ => 0,
    };
    // Determine end of this slice's params
    let param_end = if idx + 1 < n_slices {
        (headers[(idx + 1) * 4 + 3] as usize).min(params.len())
    } else {
        (param_offset + param_len).min(params.len())
    };
    let p = &params[param_offset..param_end];
    (model_type, p, t, forward)
}

// ---------------------------------------------------------------------------
// Newton solver for delta strikes
// ---------------------------------------------------------------------------

/// Newton solver: find k where d1 = target_d1.
#[inline]
pub fn solve_delta_k(
    model_type: u8,
    params: &[f64],
    t: f64,
    forward: f64,
    sqrt_t: f64,
    target_d1: f64,
) -> f64 {
    let mut k = 0.0_f64;
    let dk = 0.001_f64;

    for _ in 0..20 {
        let iv_pct = eval_iv_pct(model_type, params, k, t, forward);
        if !iv_pct.is_finite() || iv_pct <= 0.0 {
            return f64::NAN;
        }
        let sigma = iv_pct / 100.0;
        let d1 = (-k + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let err = d1 - target_d1;

        if err.abs() < 1e-6 {
            break;
        }

        let iv_pct_p = eval_iv_pct(model_type, params, k + dk, t, forward);
        if !iv_pct_p.is_finite() || iv_pct_p <= 0.0 {
            return f64::NAN;
        }
        let sigma_p = iv_pct_p / 100.0;
        let d1_p = (-(k + dk) + 0.5 * sigma_p * sigma_p * t) / (sigma_p * sqrt_t);
        let deriv = (d1_p - d1) / dk;

        if deriv.abs() < 1e-12 {
            break;
        }
        k -= err / deriv;
    }
    k
}

// ---------------------------------------------------------------------------
// Batch operations
// ---------------------------------------------------------------------------

/// Compute a full IV grid: n_slices rows x n_k columns (row-major).
///
/// - `slice_headers`: 4 f64 per slice `[model_type, T, forward, param_offset]`
/// - `slice_params`: concatenated model params
/// - `k_grid`: shared k values evaluated for every slice
pub fn iv_grid(slice_headers: &[f64], slice_params: &[f64], k_grid: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let n_k = k_grid.len();
    let mut out = Vec::with_capacity(n_slices * n_k);

    for i in 0..n_slices {
        let (model_type, params, t, forward) =
            parse_slice(slice_headers, slice_params, i, n_slices);
        for &k in k_grid {
            out.push(eval_iv_pct(model_type, params, k, t, forward));
        }
    }
    out
}

/// Like `iv_grid` but with per-slice k-bounds for flat extrapolation.
///
/// - `k_bounds`: 2 f64 per slice `[k_min, k_max, k_min, k_max, ...]`
///   If a grid k falls outside `[k_min, k_max]` for a slice, the k is clamped
///   to the boundary (flat wing extrapolation).
///   Pass an empty slice to disable clamping.
pub fn iv_grid_clamped(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_grid: &[f64],
    k_bounds: &[f64],
) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let n_k = k_grid.len();
    let has_bounds = k_bounds.len() >= n_slices * 2;
    let mut out = Vec::with_capacity(n_slices * n_k);

    for i in 0..n_slices {
        let (model_type, params, t, forward) =
            parse_slice(slice_headers, slice_params, i, n_slices);
        let (k_min, k_max) = if has_bounds {
            (k_bounds[i * 2], k_bounds[i * 2 + 1])
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        for &k in k_grid {
            let k_clamped = k.clamp(k_min, k_max);
            out.push(eval_iv_pct(model_type, params, k_clamped, t, forward));
        }
    }
    out
}

/// Batch IV evaluation for irregular (per-option) lookups.
///
/// - `k_values[i]` is evaluated against slice `slice_indices[i]`
/// - Returns IV% array same length as `k_values`
pub fn batch_slice_iv(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_values: &[f64],
    slice_indices: &[u32],
) -> Vec<f64> {
    let n = k_values.len();
    let n_slices = slice_headers.len() / 4;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let idx = slice_indices[i] as usize;
        if idx >= n_slices {
            out.push(f64::NAN);
            continue;
        }
        let (model_type, params, t, forward) =
            parse_slice(slice_headers, slice_params, idx, n_slices);
        out.push(eval_iv_pct(model_type, params, k_values[i], t, forward));
    }
    out
}

/// Compute fit diagnostics for a single slice.
///
/// Returns `[rmse, skew, kurtProxy, fitted_iv_0, fitted_iv_1, ...]`
pub fn slice_fit_diagnostics(
    model_type: u8,
    params: &[f64],
    t: f64,
    forward: f64,
    market_ks: &[f64],
    market_ivs_pct: &[f64],
    strikes: &[f64],
) -> Vec<f64> {
    let n = market_ks.len().min(market_ivs_pct.len()).min(strikes.len());

    // Compute fitted IVs and RMSE
    let mut err_sq = 0.0;
    let mut fitted_ivs = Vec::with_capacity(n);
    for i in 0..n {
        let fitted = eval_iv_pct(model_type, params, market_ks[i], t, forward);
        let err = fitted - market_ivs_pct[i];
        err_sq += err * err;
        fitted_ivs.push(fitted);
    }
    let rmse = if n > 0 {
        (err_sq / n as f64).sqrt()
    } else {
        0.0
    };

    // Skew / kurtosis proxy
    let eps = 0.001;
    let iv_at_0 = eval_iv_pct(model_type, params, 0.0, t, forward);
    let iv_plus = eval_iv_pct(model_type, params, eps, t, forward);
    let iv_minus = eval_iv_pct(model_type, params, -eps, t, forward);
    let skew = if iv_plus.is_finite() && iv_minus.is_finite() {
        (iv_plus - iv_minus) / (2.0 * eps)
    } else {
        0.0
    };
    let kurt_proxy = if iv_plus.is_finite() && iv_minus.is_finite() && iv_at_0.is_finite() {
        (iv_plus - 2.0 * iv_at_0 + iv_minus) / (eps * eps)
    } else {
        0.0
    };

    // Pack result: [rmse, skew, kurtProxy, fitted_iv_0, fitted_iv_1, ...]
    let mut out = Vec::with_capacity(3 + n);
    out.push(rmse);
    out.push(skew);
    out.push(kurt_proxy);
    out.extend_from_slice(&fitted_ivs);
    out
}

/// Find 25-delta strikes for all slices via Newton's method.
///
/// Returns flat `[kCall, kPut, ivCall_pct, ivPut_pct]` per slice (4 values each).
pub fn find_25d_strikes_batch(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let mut out = Vec::with_capacity(n_slices * 4);

    let d1_call: f64 = -0.6745;
    let d1_put: f64 = 0.6745;

    for i in 0..n_slices {
        let (model_type, params, t, forward) =
            parse_slice(slice_headers, slice_params, i, n_slices);

        if t <= 0.0 || forward <= 0.0 {
            out.extend_from_slice(&[f64::NAN; 4]);
            continue;
        }

        let sqrt_t = t.sqrt();
        let k_call = solve_delta_k(model_type, params, t, forward, sqrt_t, d1_call);
        let k_put = solve_delta_k(model_type, params, t, forward, sqrt_t, d1_put);

        let iv_call = eval_iv_pct(model_type, params, k_call, t, forward);
        let iv_put = eval_iv_pct(model_type, params, k_put, t, forward);

        out.push(k_call);
        out.push(k_put);
        out.push(iv_call);
        out.push(iv_put);
    }
    out
}

/// Combined term structure computation: 25-delta strikes + ATM IV + RR25/BF25.
///
/// Returns 7 values per slice (flat): `[kCall, kPut, ivCall%, ivPut%, atmIv%, rr25, bf25]`
pub fn term_structure_batch(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let mut out = Vec::with_capacity(n_slices * 7);

    let d1_call: f64 = -0.6745;
    let d1_put: f64 = 0.6745;

    for i in 0..n_slices {
        let (model_type, params, t, forward) =
            parse_slice(slice_headers, slice_params, i, n_slices);

        if t <= 0.0 || forward <= 0.0 {
            out.extend_from_slice(&[f64::NAN; 7]);
            continue;
        }

        let sqrt_t = t.sqrt();
        let k_call = solve_delta_k(model_type, params, t, forward, sqrt_t, d1_call);
        let k_put = solve_delta_k(model_type, params, t, forward, sqrt_t, d1_put);

        let iv_call = eval_iv_pct(model_type, params, k_call, t, forward);
        let iv_put = eval_iv_pct(model_type, params, k_put, t, forward);
        let atm_iv = eval_iv_pct(model_type, params, 0.0, t, forward);

        let rr25 = iv_call - iv_put;
        let bf25 = 0.5 * (iv_call + iv_put) - atm_iv;

        out.push(k_call);
        out.push(k_put);
        out.push(iv_call);
        out.push(iv_put);
        out.push(atm_iv);
        out.push(rr25);
        out.push(bf25);
    }
    out
}

/// Forward vol grid: spot vol, forward vol, and forward skew for adjacent slice pairs.
///
/// Returns `(2 + n_k)` values per pair (flat):
/// `[spotVol%, fwdVol%, fwdSkew_k0%, fwdSkew_k1%, ..., fwdSkew_kN%]`
///
/// Returns NaN for invalid pairs (dT <= 0 or negative variance).
pub fn forward_vol_grid(slice_headers: &[f64], slice_params: &[f64], k_points: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    if n_slices < 2 {
        return Vec::new();
    }
    let n_k = k_points.len();
    let stride = 2 + n_k;
    let mut out = Vec::with_capacity((n_slices - 1) * stride);

    for i in 0..n_slices - 1 {
        let (mt1, p1, t1, fwd1) = parse_slice(slice_headers, slice_params, i, n_slices);
        let (mt2, p2, t2, fwd2) = parse_slice(slice_headers, slice_params, i + 1, n_slices);

        let dt = t2 - t1;
        if dt <= 0.0 {
            for _ in 0..stride {
                out.push(f64::NAN);
            }
            continue;
        }

        // ATM IVs for spot vol and forward vol
        let atm_iv1 = eval_iv_pct(mt1, p1, 0.0, t1, fwd1);
        let atm_iv2 = eval_iv_pct(mt2, p2, 0.0, t2, fwd2);

        // Spot vol = later slice ATM IV
        out.push(atm_iv2);

        // Forward vol from ATM
        let v1 = atm_iv1 / 100.0;
        let v2 = atm_iv2 / 100.0;
        let fwd_var = (v2 * v2 * t2 - v1 * v1 * t1) / dt;
        out.push(if fwd_var > 0.0 {
            fwd_var.sqrt() * 100.0
        } else {
            0.0
        });

        // Forward skew at each k point
        for &k in k_points {
            let iv1 = eval_iv_pct(mt1, p1, k, t1, fwd1);
            let iv2 = eval_iv_pct(mt2, p2, k, t2, fwd2);

            if iv1.is_finite() && iv2.is_finite() && iv1 > 0.0 && iv2 > 0.0 {
                let s1 = iv1 / 100.0;
                let s2 = iv2 / 100.0;
                let fv = (s2 * s2 * t2 - s1 * s1 * t1) / dt;
                out.push(if fv > 0.0 {
                    fv.sqrt() * 100.0
                } else {
                    f64::NAN
                });
            } else {
                out.push(f64::NAN);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Realized volatility
// ---------------------------------------------------------------------------

/// Annualized realized volatility from log returns, returned as percentage.
///
/// Computes the sample standard deviation of `log_returns` and scales by
/// `sqrt(obs_per_year)`. Returns `NaN` if fewer than 2 returns.
pub fn realized_vol(log_returns: &[f64], obs_per_year: f64) -> f64 {
    let n = log_returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = log_returns.iter().sum::<f64>() / n as f64;
    let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (variance * obs_per_year).sqrt() * 100.0
}

// ---------------------------------------------------------------------------
// Batch log-moneyness
// ---------------------------------------------------------------------------

/// Compute `ln(strikes[i] / forwards[i])` for each pair.
///
/// If `forwards.len() == 1`, the single forward is broadcast to all strikes.
pub fn log_moneyness_batch(strikes: &[f64], forwards: &[f64]) -> Vec<f64> {
    let n = strikes.len();
    let mut out = Vec::with_capacity(n);

    if forwards.len() == 1 {
        let f = forwards[0];
        for &s in strikes {
            out.push((s / f).ln());
        }
    } else {
        let m = n.min(forwards.len());
        for i in 0..m {
            out.push((strikes[i] / forwards[i]).ln());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Batch log returns
// ---------------------------------------------------------------------------

/// Compute `ln(prices[i] / prices[i-1])` for i=1..n.
pub fn log_returns_batch(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            out.push((prices[i] / prices[i - 1]).ln());
        } else {
            out.push(0.0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a single SVI slice header + params.
    fn svi_fixture() -> (Vec<f64>, Vec<f64>) {
        // SVI params: a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.5
        let headers = vec![MODEL_SVI as f64, 0.25, 100.0, 0.0];
        let params = vec![0.04, 0.4, -0.4, 0.0, 0.5];
        (headers, params)
    }

    /// Helper: build a single VV slice header + params.
    fn vv_fixture() -> (Vec<f64>, Vec<f64>) {
        // VV params: atmVol=0.20, rr25=-0.02, bf25=0.005
        let headers = vec![MODEL_VV as f64, 0.25, 100.0, 0.0];
        let params = vec![0.20, -0.02, 0.005];
        (headers, params)
    }

    #[test]
    fn test_eval_iv_pct_svi() {
        let params = [0.04, 0.4, -0.4, 0.0, 0.5];
        let iv = eval_iv_pct(MODEL_SVI, &params, 0.0, 0.25, 100.0);
        assert!(iv.is_finite());
        assert!(iv > 0.0);
        // At k=0, total_var = a + b*sigma = 0.04 + 0.4*0.5 = 0.24
        // IV = sqrt(0.24/0.25)*100 = sqrt(0.96)*100 ≈ 97.98
        let expected = (0.24_f64 / 0.25).sqrt() * 100.0;
        assert!((iv - expected).abs() < 0.01);
    }

    #[test]
    fn test_eval_iv_pct_sabr() {
        let params = [0.2, 0.5, -0.3, 0.4];
        let iv = eval_iv_pct(MODEL_SABR, &params, 0.0, 0.25, 100.0);
        assert!(iv.is_finite());
        assert!(iv > 0.0);
    }

    #[test]
    fn test_eval_iv_pct_vv() {
        let params = [0.20, -0.02, 0.005];
        let iv = eval_iv_pct(MODEL_VV, &params, 0.0, 0.25, 100.0);
        assert!(iv.is_finite());
        // At k=0 the vanna/volga terms vanish → iv = atm_vol * 100
        assert!((iv - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_iv_pct_invalid() {
        assert!(eval_iv_pct(99, &[], 0.0, 1.0, 100.0).is_nan());
        assert!(eval_iv_pct(MODEL_SVI, &[0.04, 0.4], 0.0, 1.0, 100.0).is_nan()); // too few params
        assert!(eval_iv_pct(MODEL_SVI, &[0.04, 0.4, -0.4, 0.0, 0.5], 0.0, 0.0, 100.0).is_nan()); // t=0
    }

    #[test]
    fn test_iv_grid_single_slice() {
        let (headers, params) = svi_fixture();
        let k_grid = vec![-0.1, 0.0, 0.1];
        let result = iv_grid(&headers, &params, &k_grid);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_finite() && *v > 0.0));
        // Smile: IV at k=-0.1 and k=0.1 should differ from ATM
        assert!((result[0] - result[1]).abs() > 0.01 || (result[2] - result[1]).abs() > 0.01);
    }

    #[test]
    fn test_iv_grid_multi_slice() {
        // Two SVI slices with different expiries
        let headers = vec![
            MODEL_SVI as f64,
            0.25,
            100.0,
            0.0, // slice 0
            MODEL_SVI as f64,
            1.0,
            100.0,
            5.0, // slice 1
        ];
        let params = vec![
            0.04, 0.4, -0.4, 0.0, 0.5, // slice 0
            0.08, 0.3, -0.3, 0.0, 0.4, // slice 1
        ];
        let k_grid = vec![-0.05, 0.0, 0.05];
        let result = iv_grid(&headers, &params, &k_grid);
        assert_eq!(result.len(), 6); // 2 slices * 3 k-points
        assert!(result.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn test_term_structure_batch() {
        let (headers, params) = svi_fixture();
        let result = term_structure_batch(&headers, &params);
        assert_eq!(result.len(), 7);
        // kCall, kPut, ivCall, ivPut, atmIv, rr25, bf25
        let atm_iv = result[4];
        assert!(atm_iv.is_finite() && atm_iv > 0.0);
        // rr25 = ivCall - ivPut
        let rr25 = result[5];
        assert!(rr25.is_finite());
    }

    #[test]
    fn test_forward_vol_grid() {
        // Use same SVI shape but higher `a` for slice 1 so total variance increases with T
        let headers = vec![
            MODEL_SVI as f64,
            0.25,
            100.0,
            0.0,
            MODEL_SVI as f64,
            1.0,
            100.0,
            5.0,
        ];
        let params = vec![
            0.01, 0.2, -0.4, 0.0, 0.3, // slice 0: low variance at T=0.25
            0.06, 0.2, -0.4, 0.0, 0.3, // slice 1: higher variance at T=1.0
        ];
        let k_points = vec![-0.05, 0.0, 0.05];
        let result = forward_vol_grid(&headers, &params, &k_points);
        // 1 pair * (2 + 3) = 5 values
        assert_eq!(result.len(), 5);
        // spot vol and fwd vol should be positive
        assert!(result[0] > 0.0 && result[0].is_finite());
        assert!(result[1] > 0.0 && result[1].is_finite());
    }

    #[test]
    fn test_forward_vol_grid_single_slice() {
        let (headers, params) = svi_fixture();
        let result = forward_vol_grid(&headers, &params, &[0.0]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_solve_delta_k() {
        let params = [0.04, 0.4, -0.4, 0.0, 0.5];
        let t = 0.25_f64;
        let sqrt_t = t.sqrt();
        let k = solve_delta_k(MODEL_SVI, &params, t, 100.0, sqrt_t, -0.6745);
        assert!(k.is_finite());
        // 25-delta call strike should be positive (OTM call → k > 0)
        // (Not strictly guaranteed for all models but typical)
    }

    #[test]
    fn test_batch_slice_iv() {
        let (headers, params) = svi_fixture();
        let k_values = vec![0.0, 0.05, -0.05];
        let slice_indices = vec![0, 0, 0];
        let result = batch_slice_iv(&headers, &params, &k_values, &slice_indices);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn test_batch_slice_iv_oob_index() {
        let (headers, params) = svi_fixture();
        let k_values = vec![0.0];
        let slice_indices = vec![5]; // out of bounds
        let result = batch_slice_iv(&headers, &params, &k_values, &slice_indices);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan());
    }

    #[test]
    fn test_slice_fit_diagnostics() {
        let params = [0.04, 0.4, -0.4, 0.0, 0.5];
        let t = 0.25;
        let forward = 100.0;
        let ks = vec![-0.1, 0.0, 0.1];
        // Use the model's own IVs as "market" → RMSE should be ~0
        let market_ivs: Vec<f64> = ks
            .iter()
            .map(|&k| eval_iv_pct(MODEL_SVI, &params, k, t, forward))
            .collect();
        let strikes: Vec<f64> = ks.iter().map(|&k| forward * k.exp()).collect();

        let result =
            slice_fit_diagnostics(MODEL_SVI, &params, t, forward, &ks, &market_ivs, &strikes);
        assert!(result.len() >= 3 + ks.len());
        let rmse = result[0];
        assert!(rmse < 1e-10); // perfect fit
    }

    #[test]
    fn test_find_25d_strikes_batch() {
        let (headers, params) = vv_fixture();
        let result = find_25d_strikes_batch(&headers, &params);
        assert_eq!(result.len(), 4);
        // kCall, kPut, ivCall, ivPut — all should be finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_log_moneyness_batch_broadcast() {
        let strikes = vec![90.0, 100.0, 110.0];
        let forwards = vec![100.0];
        let result = log_moneyness_batch(&strikes, &forwards);
        assert_eq!(result.len(), 3);
        assert!((result[0] - (90.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((result[1] - 0.0).abs() < 1e-12);
        assert!((result[2] - (110.0_f64 / 100.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_log_moneyness_batch_paired() {
        let strikes = vec![90.0, 100.0, 110.0];
        let forwards = vec![95.0, 100.0, 105.0];
        let result = log_moneyness_batch(&strikes, &forwards);
        assert_eq!(result.len(), 3);
        assert!((result[0] - (90.0_f64 / 95.0).ln()).abs() < 1e-12);
        assert!((result[1] - 0.0).abs() < 1e-12);
        assert!((result[2] - (110.0_f64 / 105.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_log_returns_batch() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let result = log_returns_batch(&prices);
        assert_eq!(result.len(), 3);
        assert!((result[0] - (101.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((result[1] - (99.0_f64 / 101.0).ln()).abs() < 1e-12);
        assert!((result[2] - (102.0_f64 / 99.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_log_returns_batch_short() {
        assert!(log_returns_batch(&[100.0]).is_empty());
        assert!(log_returns_batch(&[]).is_empty());
    }

    #[test]
    fn test_realized_vol() {
        let returns = vec![0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.004];
        let rv = realized_vol(&returns, 252.0);
        assert!(rv.is_finite() && rv > 0.0);
        // With these small returns, annualized vol should be in a reasonable range
        assert!(rv < 200.0);
    }

    #[test]
    fn test_realized_vol_short() {
        assert!(realized_vol(&[0.01], 252.0).is_nan());
        assert!(realized_vol(&[], 252.0).is_nan());
    }
}
