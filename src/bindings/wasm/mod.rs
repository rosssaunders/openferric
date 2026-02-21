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
// IV slice operations — thin wrappers delegating to vol::slice
// ---------------------------------------------------------------------------

/// Compute a full IV grid: n_slices rows x n_k columns (row-major).
#[wasm_bindgen]
pub fn iv_grid(slice_headers: &[f64], slice_params: &[f64], k_grid: &[f64]) -> Vec<f64> {
    crate::vol::slice::iv_grid(slice_headers, slice_params, k_grid)
}

/// Batch IV evaluation for irregular (per-option) lookups.
#[wasm_bindgen]
pub fn batch_slice_iv(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_values: &[f64],
    slice_indices: &[u32],
) -> Vec<f64> {
    crate::vol::slice::batch_slice_iv(slice_headers, slice_params, k_values, slice_indices)
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
    crate::vol::slice::slice_fit_diagnostics(model_type, params, t, forward, market_ks, market_ivs_pct, strikes)
}

/// Find 25-delta strikes for all slices via Newton's method.
#[wasm_bindgen]
pub fn find_25d_strikes_batch(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    crate::vol::slice::find_25d_strikes_batch(slice_headers, slice_params)
}

/// Combined term structure computation: 25-delta strikes + ATM IV + RR25/BF25.
#[wasm_bindgen]
pub fn term_structure_batch_wasm(slice_headers: &[f64], slice_params: &[f64]) -> Vec<f64> {
    crate::vol::slice::term_structure_batch(slice_headers, slice_params)
}

/// Forward vol grid: spot vol, forward vol, and forward skew for adjacent slice pairs.
#[wasm_bindgen]
pub fn forward_vol_grid_wasm(
    slice_headers: &[f64],
    slice_params: &[f64],
    k_points: &[f64],
) -> Vec<f64> {
    crate::vol::slice::forward_vol_grid(slice_headers, slice_params, k_points)
}

// ---------------------------------------------------------------------------
// Strategy payoff — thin wrapper delegating to pricing::payoff
// ---------------------------------------------------------------------------

/// Strategy intrinsic PnL at expiry for a set of option legs across a spot axis.
#[wasm_bindgen]
pub fn strategy_intrinsic_pnl_wasm(
    spot_axis: &[f64],
    strikes: &[f64],
    quantities: &[f64],
    is_calls: &[u8],
    total_cost: f64,
) -> Vec<f64> {
    crate::pricing::payoff::strategy_intrinsic_pnl(spot_axis, strikes, quantities, is_calls, total_cost)
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
