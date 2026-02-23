use wasm_bindgen::prelude::*;

use openferric::core::types::{BarrierDirection, BarrierStyle, OptionType};
use openferric::greeks::black_scholes_merton_greeks;
use openferric::math::{normal_cdf, normal_pdf};
use openferric::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use openferric::pricing::european::{black_76_price, black_scholes_price};
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

#[inline]
fn black76_greeks_7(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
) -> [f64; 7] {
    if forward <= 0.0 || strike <= 0.0 || vol <= 0.0 || expiry <= 0.0 {
        return [0.0; 7];
    }

    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df = (-rate * expiry).exp();
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let pdf_d1 = normal_pdf(d1);

    let call = df * (forward * nd1 - strike * nd2);
    let put = call - df * (forward - strike);
    let price = match option_type {
        OptionType::Call => call,
        OptionType::Put => put,
    };

    let delta = match option_type {
        OptionType::Call => df * nd1,
        OptionType::Put => df * (nd1 - 1.0),
    };
    let gamma = df * pdf_d1 / (forward * vol * sqrt_t);
    let vega = df * forward * pdf_d1 * sqrt_t;
    let theta = rate.mul_add(price, -(df * forward * pdf_d1 * vol / (2.0 * sqrt_t)));
    // Rho here is dV/dr with forward held fixed.
    let rho = -expiry * price;
    let vanna = -df * pdf_d1 * d2 / vol;
    let volga = vega * d1 * d2 / vol;

    [delta, gamma, vega, theta, rho, vanna, volga]
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
        out.push(black_scholes_price(
            ot,
            s_adj,
            strikes[i],
            rates[i],
            vols[i],
            maturities[i],
        ));
    }
    out
}

/// Batch Black-76 pricing: one WASM call for N options.
///
/// All input slices must have the same length. `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a `Vec<f64>` of prices in the same order.
#[wasm_bindgen]
pub fn black76_price_batch_wasm(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    vols: &[f64],
    maturities: &[f64],
    is_calls: &[u8],
) -> Vec<f64> {
    let n = forwards.len();
    debug_assert!(
        n == strikes.len()
            && n == rates.len()
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
        out.push(black_76_price(
            ot,
            forwards[i],
            strikes[i],
            rates[i],
            vols[i],
            maturities[i],
        ));
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
            ot,
            spots[i],
            strikes[i],
            rates[i],
            div_yields[i],
            vols[i],
            expiries[i],
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

/// Batch Black-76 Greeks: one WASM call for N options.
///
/// All input slices must have the same length. `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a flat `Vec<f64>` of 7 values per option:
/// `[delta, gamma, vega, theta, rho, vanna, volga]` repeated N times.
///
/// `delta` is forward delta (`dV/dF`), consistent with Deribit convention.
#[wasm_bindgen]
pub fn black76_greeks_batch_wasm(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    vols: &[f64],
    expiries: &[f64],
    is_calls: &[u8],
) -> Vec<f64> {
    let n = forwards.len();
    debug_assert!(
        n == strikes.len()
            && n == rates.len()
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
        let g = black76_greeks_7(ot, forwards[i], strikes[i], rates[i], vols[i], expiries[i]);
        out.extend_from_slice(&g);
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

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // -- bs_price --

    #[test]
    fn bs_price_atm_call() {
        let price = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((price - 10.4506).abs() < 0.01);
    }

    #[test]
    fn bs_price_atm_put() {
        let price = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false);
        assert!((price - 5.5735).abs() < 0.01);
    }

    #[test]
    fn bs_price_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.0;
        let t = 1.0;
        let call = bs_price(s, k, r, q, 0.20, t, true);
        let put = bs_price(s, k, r, q, 0.20, t, false);
        let s_adj = s * (-q * t).exp();
        let parity = call - put - (s_adj - k * (-r * t).exp());
        assert!(parity.abs() < 1e-8);
    }

    #[test]
    fn bs_price_with_dividend() {
        let no_div = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let with_div = bs_price(100.0, 100.0, 0.05, 0.03, 0.20, 1.0, true);
        assert!(with_div < no_div);
    }

    // -- bs_implied_vol --

    #[test]
    fn bs_implied_vol_round_trip() {
        let vol = 0.25;
        let price = bs_price(100.0, 100.0, 0.05, 0.0, vol, 1.0, true);
        let recovered = bs_implied_vol(price, 100.0, 100.0, 0.05, 0.0, 1.0, true);
        assert!((recovered - vol).abs() < 1e-4);
    }

    #[test]
    fn bs_implied_vol_put_round_trip() {
        let vol = 0.30;
        let price = bs_price(100.0, 110.0, 0.03, 0.0, vol, 0.5, false);
        let recovered = bs_implied_vol(price, 100.0, 110.0, 0.03, 0.0, 0.5, false);
        assert!((recovered - vol).abs() < 1e-3);
    }

    // -- bsm_greeks_wasm --

    #[test]
    fn bsm_greeks_call_has_7_values() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert_eq!(g.len(), 7);
    }

    #[test]
    fn bsm_greeks_call_delta_range() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let delta = g[0];
        assert!(delta > 0.0 && delta < 1.0);
    }

    #[test]
    fn bsm_greeks_put_delta_negative() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false);
        assert!(g[0] < 0.0);
    }

    #[test]
    fn bsm_greeks_gamma_positive() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(g[1] > 0.0);
    }

    #[test]
    fn bsm_greeks_vega_positive() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(g[2] > 0.0);
    }

    // -- bs_price_batch_wasm --

    #[test]
    fn bs_price_batch_matches_scalar() {
        let spots = [100.0, 100.0];
        let strikes = [100.0, 110.0];
        let rates = [0.05, 0.05];
        let divs = [0.0, 0.0];
        let vols = [0.20, 0.20];
        let mats = [1.0, 1.0];
        let calls = [1u8, 0u8];
        let batch = bs_price_batch_wasm(&spots, &strikes, &rates, &divs, &vols, &mats, &calls);
        assert_eq!(batch.len(), 2);
        let p1 = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let p2 = bs_price(100.0, 110.0, 0.05, 0.0, 0.20, 1.0, false);
        assert!((batch[0] - p1).abs() < TOL);
        assert!((batch[1] - p2).abs() < TOL);
    }

    // -- bsm_greeks_batch_wasm --

    #[test]
    fn bsm_greeks_batch_matches_scalar() {
        let spots = [100.0];
        let strikes = [100.0];
        let rates = [0.05];
        let divs = [0.0];
        let vols = [0.20];
        let expiries = [1.0];
        let calls = [1u8];
        let batch =
            bsm_greeks_batch_wasm(&spots, &strikes, &rates, &divs, &vols, &expiries, &calls);
        let single = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert_eq!(batch.len(), 7);
        for i in 0..7 {
            assert!((batch[i] - single[i]).abs() < TOL);
        }
    }

    // -- black76_price_batch_wasm --

    #[test]
    fn black76_price_batch_matches_bsm_q_eq_r() {
        let forwards = [100.0, 105.0];
        let strikes = [100.0, 110.0];
        let rates = [0.05, 0.03];
        let vols = [0.20, 0.35];
        let mats = [1.0, 0.75];
        let calls = [1u8, 0u8];

        let black = black76_price_batch_wasm(&forwards, &strikes, &rates, &vols, &mats, &calls);
        let bsm = bs_price_batch_wasm(&forwards, &strikes, &rates, &rates, &vols, &mats, &calls);

        assert_eq!(black.len(), bsm.len());
        for i in 0..black.len() {
            assert!((black[i] - bsm[i]).abs() < TOL);
        }
    }

    // -- black76_greeks_batch_wasm --

    #[test]
    fn black76_greeks_batch_matches_bsm_q_eq_r() {
        let forwards = [100.0];
        let strikes = [100.0];
        let rates = [0.05];
        let vols = [0.20];
        let expiries = [1.0];
        let calls = [1u8];

        let black =
            black76_greeks_batch_wasm(&forwards, &strikes, &rates, &vols, &expiries, &calls);
        let bsm = bsm_greeks_batch_wasm(
            &forwards, &strikes, &rates, &rates, &vols, &expiries, &calls,
        );

        assert_eq!(black.len(), 7);
        assert_eq!(bsm.len(), 7);
        // Delta/gamma/vega/vanna/volga align with BSM when q = r and S = F.
        // Theta/rho are convention-sensitive because Black-76 keeps F fixed.
        for i in [0usize, 1usize, 2usize, 5usize, 6usize] {
            assert!((black[i] - bsm[i]).abs() < TOL);
        }
        assert!(black[3].is_finite());
        assert!(black[4].is_finite());
    }

    // -- barrier_price --

    #[test]
    fn barrier_up_out_call_less_than_vanilla() {
        let vanilla = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let bp = barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-out", true);
        assert!(bp > 0.0 && bp < vanilla);
    }

    #[test]
    fn barrier_in_plus_out_equals_vanilla() {
        let in_p = barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-in", true);
        let out_p = barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-out", true);
        let vanilla = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((in_p + out_p - vanilla).abs() < 1e-3);
    }

    #[test]
    fn barrier_invalid_type_returns_nan() {
        assert!(
            barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "sideways", true).is_nan()
        );
    }

    #[test]
    fn barrier_down_in_put() {
        let price = barrier_price(100.0, 100.0, 80.0, 0.05, 0.0, 0.20, 1.0, "down-in", false);
        assert!(price > 0.0);
    }

    // -- bond_price --

    #[test]
    fn bond_price_par() {
        // When coupon rate == yield rate, bond price ≈ face value
        let price = bond_price(1000.0, 0.05, 10.0, 0.05, 2);
        assert!((price - 1000.0).abs() < 0.01);
    }

    #[test]
    fn bond_price_premium() {
        // Coupon > yield → premium
        let price = bond_price(1000.0, 0.08, 10.0, 0.05, 2);
        assert!(price > 1000.0);
    }

    #[test]
    fn bond_price_discount() {
        // Coupon < yield → discount
        let price = bond_price(1000.0, 0.03, 10.0, 0.05, 2);
        assert!(price < 1000.0);
    }

    #[test]
    fn bond_price_zero_freq_nan() {
        assert!(bond_price(1000.0, 0.05, 10.0, 0.05, 0).is_nan());
    }

    #[test]
    fn bond_price_negative_maturity_nan() {
        assert!(bond_price(1000.0, 0.05, -1.0, 0.05, 2).is_nan());
    }
}
