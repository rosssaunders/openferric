use wasm_bindgen::prelude::*;

use crate::core::types::{BarrierDirection, BarrierStyle, OptionType};
use crate::greeks::black_scholes_merton_greeks;
use crate::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use crate::pricing::european::black_scholes_price;
use crate::risk::var::historical_var;
use crate::vol::implied::implied_vol;
use crate::vol::sabr::{SabrParams, fit_sabr};
use crate::vol::surface::{SviParams, calibrate_svi};

// ---------------------------------------------------------------------------
// Black-Scholes pricing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Heston pricing (fast FFT path)
// ---------------------------------------------------------------------------

#[inline]
fn heston_intrinsic(spot: f64, strike: f64, is_call: bool) -> f64 {
    if is_call {
        (spot - strike).max(0.0)
    } else {
        (strike - spot).max(0.0)
    }
}

#[inline]
fn option_price_from_call(
    call_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    maturity: f64,
    is_call: bool,
) -> f64 {
    if is_call {
        call_price
    } else {
        call_price - spot * (-div_yield * maturity).exp() + strike * (-rate * maturity).exp()
    }
}

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
    use crate::engines::fft::{CarrMadanContext, CarrMadanParams, HestonCharFn};

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

// ---------------------------------------------------------------------------
// Barrier options
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Bond pricing (simple fixed-rate bond with flat yield)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CDS fair spread (simplified flat hazard rate model)
// ---------------------------------------------------------------------------

/// Approximate CDS fair spread from flat hazard rate and flat discount rate.
#[wasm_bindgen]
pub fn cds_fair_spread(
    _notional: f64,
    maturity: f64,
    recovery_rate: f64,
    hazard_rate: f64,
    discount_rate: f64,
) -> f64 {
    // Build simple flat curves and use the CDS struct
    use crate::credit::Cds;
    use crate::credit::survival_curve::SurvivalCurve;
    use crate::rates::YieldCurve;

    let tenors: Vec<(f64, f64)> = (1..=((maturity * 4.0).ceil() as u32))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-discount_rate * t).exp())
        })
        .collect();
    let discount_curve = YieldCurve::new(tenors);

    let surv_nodes: Vec<(f64, f64)> = (1..=((maturity * 4.0).ceil() as u32))
        .map(|i| {
            let t = i as f64 * 0.25;
            (t, (-hazard_rate * t).exp())
        })
        .collect();
    let survival_curve = SurvivalCurve::new(surv_nodes);

    let cds = Cds {
        notional: 1.0,
        spread: 0.01, // dummy
        maturity,
        recovery_rate,
        payment_freq: 4,
    };
    cds.fair_spread(&discount_curve, &survival_curve)
}

// ---------------------------------------------------------------------------
// Historical VaR
// ---------------------------------------------------------------------------

/// Historical Value-at-Risk from a flat array of P&L returns.
#[wasm_bindgen]
pub fn var_historical(returns_flat: &[f64], confidence: f64) -> f64 {
    if returns_flat.is_empty() {
        return f64::NAN;
    }
    historical_var(returns_flat, confidence)
}

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
// IV evaluation helpers (Rust replacement for JS sliceIvPct dispatcher)
// ---------------------------------------------------------------------------

/// Model type constants for slice headers.
const MODEL_SVI: u8 = 0;
const MODEL_SABR: u8 = 1;
const MODEL_VV: u8 = 2;

/// Evaluate IV% for a single (model, params, k, T, forward) tuple.
/// This is the Rust replacement for the JS `sliceIvPct` dispatcher.
#[inline]
fn eval_iv_pct(model_type: u8, params: &[f64], k: f64, t: f64, forward: f64) -> f64 {
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

/// Parse a single slice from the header/params arrays.
/// Header layout per slice: [model_type, T, forward, param_offset] (4 f64 each).
/// Returns (model_type, param_slice, T, forward).
#[inline]
fn parse_slice<'a>(
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

/// Compute a full IV grid: n_slices rows x n_k columns (row-major).
///
/// - `slice_headers`: 4 f64 per slice `[model_type, T, forward, param_offset]`
/// - `slice_params`: concatenated model params
/// - `k_grid`: shared k values evaluated for every slice
#[wasm_bindgen]
pub fn iv_grid(slice_headers: &[f64], slice_params: &[f64], k_grid: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let n_k = k_grid.len();
    let mut out = Vec::with_capacity(n_slices * n_k);

    for i in 0..n_slices {
        let (model_type, params, t, forward) = parse_slice(slice_headers, slice_params, i, n_slices);
        for &k in k_grid {
            out.push(eval_iv_pct(model_type, params, k, t, forward));
        }
    }
    out
}

/// Batch IV evaluation for irregular (per-option) lookups.
///
/// - `k_values[i]` is evaluated against slice `slice_indices[i]`
/// - Returns IV% array same length as `k_values`
#[wasm_bindgen]
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
        let (model_type, params, t, forward) = parse_slice(slice_headers, slice_params, idx, n_slices);
        out.push(eval_iv_pct(model_type, params, k_values[i], t, forward));
    }
    out
}

/// Compute fit diagnostics for a single slice.
///
/// Returns `[rmse, skew, kurtProxy, fitted_iv_0, fitted_iv_1, ...]`
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
    let rmse = if n > 0 { (err_sq / n as f64).sqrt() } else { 0.0 };

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
#[wasm_bindgen]
pub fn find_25d_strikes_batch(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    let n_slices = slice_headers.len() / 4;
    let mut out = Vec::with_capacity(n_slices * 4);

    let d1_call: f64 = -0.6745;
    let d1_put: f64 = 0.6745;

    for i in 0..n_slices {
        let (model_type, params, t, forward) = parse_slice(slice_headers, slice_params, i, n_slices);

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

/// Newton solver: find k where d1 = target_d1.
#[inline]
fn solve_delta_k(
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

/// Annualized realized volatility from log returns, returned as percentage.
#[wasm_bindgen]
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
// GPU Monte Carlo (WebGPU)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
/// GPU Monte Carlo result exposed to JavaScript via wasm-bindgen.
#[wasm_bindgen]
pub struct WasmGpuMcResult {
    pub price: f64,
    pub stderr: f64,
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
/// GPU Monte Carlo European option pricing via WebGPU compute shaders.
///
/// Uses `u32` for `num_paths` and `num_steps` to avoid BigInt in JS.
#[wasm_bindgen]
pub async fn gpu_mc_price_european(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    num_paths: u32,
    num_steps: u32,
    seed: u32,
    is_call: bool,
) -> Result<WasmGpuMcResult, JsError> {
    let result = crate::engines::gpu::mc_european_gpu_async(
        spot, strike, rate, vol, expiry, num_paths, num_steps, seed, is_call,
    )
    .await
    .map_err(|e| JsError::new(&e))?;

    Ok(WasmGpuMcResult {
        price: result.price,
        stderr: result.stderr,
    })
}
