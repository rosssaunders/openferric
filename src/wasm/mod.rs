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

// ---------------------------------------------------------------------------
// Heston pricing (semi-analytic via engine)
// ---------------------------------------------------------------------------

/// Heston semi-analytic European option price.
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
    use crate::core::PricingEngine;
    use crate::engines::analytic::heston::HestonEngine;
    use crate::instruments::vanilla::VanillaOption;
    use crate::market::Market;

    let ot = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let option = if is_call {
        VanillaOption::european_call(strike, maturity)
    } else {
        VanillaOption::european_put(strike, maturity)
    };
    let market = Market {
        spot,
        rate,
        dividend_yield: div_yield,
        vol: crate::market::VolSource::Flat(v0.sqrt()),
        reference_date: None,
    };
    let engine = HestonEngine::new(v0, kappa, theta, sigma_v, rho);
    engine
        .price(&option, &market)
        .map(|r| r.price)
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
pub fn sabr_vol(forward: f64, strike: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
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
// GPU Monte Carlo (WebGPU)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
/// GPU Monte Carlo result exposed to JavaScript via wasm-bindgen.
#[wasm_bindgen]
pub struct WasmGpuMcResult {
    pub price: f64,
    pub stderr: f64,
}

#[cfg(feature = "gpu")]
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
        spot, strike, rate, vol, expiry,
        num_paths, num_steps, seed, is_call,
    )
    .await
    .map_err(|e| JsError::new(&e))?;

    Ok(WasmGpuMcResult {
        price: result.price,
        stderr: result.stderr,
    })
}
