use wasm_bindgen::prelude::*;

use crate::error::{
    check_batch_lengths, js_error, require_finite, require_non_negative, require_positive,
};
use openferric::core::types::{BarrierDirection, BarrierStyle, OptionType};
use openferric::engines::analytic::{
    black76_greeks as validated_black76_greeks, black76_price as validated_black76_price,
};
use openferric::greeks::black_scholes_merton_greeks;
use openferric::math::normal_pdf;
use openferric::pricing::barrier::barrier_price_closed_form_with_carry_and_rebate;
use openferric::pricing::european::black_scholes_price;
use openferric::vol::implied::implied_vol;

#[inline]
fn input_label(scalar_name: &str, batch_name: &str, index: Option<usize>) -> String {
    match index {
        Some(index) => format!("{batch_name}[{index}]"),
        None => scalar_name.to_string(),
    }
}

#[inline]
fn validate_finite_input(
    scalar_name: &str,
    batch_name: &str,
    index: Option<usize>,
    value: f64,
) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!(
            "`{}` must be finite",
            input_label(scalar_name, batch_name, index)
        ))
    }
}

#[inline]
fn validate_positive_input(
    scalar_name: &str,
    batch_name: &str,
    index: Option<usize>,
    value: f64,
) -> Result<(), String> {
    validate_finite_input(scalar_name, batch_name, index, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "`{}` must be positive",
            input_label(scalar_name, batch_name, index)
        ))
    }
}

#[inline]
fn validate_non_negative_input(
    scalar_name: &str,
    batch_name: &str,
    index: Option<usize>,
    value: f64,
) -> Result<(), String> {
    validate_finite_input(scalar_name, batch_name, index, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "`{}` must be non-negative",
            input_label(scalar_name, batch_name, index)
        ))
    }
}

#[inline]
fn checked_finite_output(name: &str, index: Option<usize>, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        let output = match index {
            Some(index) => format!("{name}[{index}]"),
            None => name.to_string(),
        };
        Err(format!("`{output}` is non-finite for the supplied inputs"))
    }
}

#[inline]
fn checked_option_flag(index: usize, value: u8) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!(
            "`is_calls[{index}]` must be 0 for a put or 1 for a call"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn checked_bs_price(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    maturity: f64,
    is_call: bool,
    index: Option<usize>,
) -> Result<f64, String> {
    validate_positive_input("spot", "spots", index, spot)?;
    validate_positive_input("strike", "strikes", index, strike)?;
    validate_non_negative_input("vol", "vols", index, vol)?;
    validate_non_negative_input("maturity", "maturities", index, maturity)?;
    validate_finite_input("rate", "rates", index, rate)?;
    validate_finite_input("div_yield", "div_yields", index, div_yield)?;

    let option_type = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let adjusted_spot = spot * (-div_yield * maturity).exp();
    if !adjusted_spot.is_finite() {
        let location = index.map_or_else(
            || "adjusted spot".to_string(),
            |index| format!("adjusted spot at batch index {index}"),
        );
        return Err(format!(
            "`{location}` is non-finite for the supplied inputs"
        ));
    }
    checked_finite_output(
        "price",
        index,
        black_scholes_price(option_type, adjusted_spot, strike, rate, vol, maturity),
    )
}

#[allow(clippy::too_many_arguments)]
fn checked_bsm_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    index: Option<usize>,
) -> Result<[f64; 7], String> {
    validate_positive_input("spot", "spots", index, spot)?;
    validate_positive_input("strike", "strikes", index, strike)?;
    validate_non_negative_input("vol", "vols", index, vol)?;
    validate_non_negative_input("expiry", "expiries", index, expiry)?;
    validate_finite_input("rate", "rates", index, rate)?;
    validate_finite_input("div_yield", "div_yields", index, div_yield)?;

    let option_type = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let greeks =
        black_scholes_merton_greeks(option_type, spot, strike, rate, div_yield, vol, expiry);
    let values = [
        greeks.delta,
        greeks.gamma,
        greeks.vega,
        greeks.theta,
        greeks.rho,
        greeks.vanna,
        greeks.volga,
    ];
    if let Some((greek_index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        const NAMES: [&str; 7] = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"];
        let location = index.map_or_else(
            || format!("greeks.{}", NAMES[greek_index]),
            |index| format!("greeks[{index}].{}", NAMES[greek_index]),
        );
        return Err(format!(
            "`{location}` is non-finite for the supplied inputs"
        ));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn checked_black76_price(
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    is_call: bool,
    index: Option<usize>,
) -> Result<f64, String> {
    validate_positive_input("forward", "forwards", index, forward)?;
    validate_positive_input("strike", "strikes", index, strike)?;
    validate_finite_input("rate", "rates", index, rate)?;
    validate_non_negative_input("vol", "vols", index, vol)?;
    validate_non_negative_input("maturity", "maturities", index, maturity)?;

    let option_type = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let price = validated_black76_price(option_type, forward, strike, rate, vol, maturity)
        .map_err(|error| match index {
            Some(index) => format!("Black-76 input at batch index {index} is invalid: {error}"),
            None => format!("Black-76 input is invalid: {error}"),
        })?;
    checked_finite_output("price", index, price)
}

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
) -> Result<f64, JsValue> {
    checked_bs_price(spot, strike, rate, div_yield, vol, maturity, is_call, None).map_err(js_error)
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
) -> Result<f64, JsValue> {
    require_finite("price", price)?;
    require_positive("spot", spot)?;
    require_positive("strike", strike)?;
    require_positive("maturity", maturity)?;
    require_finite("rate", rate)?;
    require_finite("div_yield", div_yield)?;

    let ot = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let s_adj = spot * (-div_yield * maturity).exp();
    implied_vol(ot, s_adj, strike, rate, maturity, price, 1e-12, 64)
        .map_err(|e| js_error(format!("implied volatility solve failed: {e}")))
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
) -> Result<Vec<f64>, JsValue> {
    checked_bsm_greeks(spot, strike, rate, div_yield, vol, expiry, is_call, None)
        .map(Vec::from)
        .map_err(js_error)
}

#[inline]
fn stable_black76_d1_d2(forward: f64, strike: f64, vol: f64, expiry: f64) -> (f64, f64) {
    let relative_moneyness = (forward - strike) / strike;
    let log_moneyness = if relative_moneyness.abs() <= 0.5 {
        relative_moneyness.ln_1p()
    } else {
        forward.ln() - strike.ln()
    };
    let width = vol * expiry.sqrt();
    let midpoint = if width != 0.0 {
        log_moneyness / width
    } else if log_moneyness == 0.0 {
        0.0
    } else {
        let log_width = vol.ln() + 0.5 * expiry.ln();
        log_moneyness.signum() * (log_moneyness.abs().ln() - log_width).exp()
    };
    let half_width = 0.5 * width;
    (midpoint + half_width, midpoint - half_width)
}

#[inline]
fn black76_vanna_volga(
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    vega: f64,
) -> (f64, f64) {
    if vol <= 0.0 || expiry <= 0.0 {
        return (0.0, 0.0);
    }

    let (d1, d2) = stable_black76_d1_d2(forward, strike, vol, expiry);
    let df = (-rate * expiry).exp();
    let pdf_d1 = normal_pdf(d1);
    let vanna = -df * pdf_d1 * d2 / vol;
    let volga = vega * d1 * d2 / vol;
    (vanna, volga)
}

#[allow(clippy::too_many_arguments)]
fn checked_black76_greeks(
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    index: Option<usize>,
) -> Result<[f64; 7], String> {
    validate_positive_input("forward", "forwards", index, forward)?;
    validate_positive_input("strike", "strikes", index, strike)?;
    validate_finite_input("rate", "rates", index, rate)?;
    validate_non_negative_input("vol", "vols", index, vol)?;
    validate_non_negative_input("expiry", "expiries", index, expiry)?;

    let option_type = if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let greeks = validated_black76_greeks(option_type, forward, strike, rate, vol, expiry)
        .map_err(|error| match index {
            Some(index) => format!("Black-76 input at batch index {index} is invalid: {error}"),
            None => format!("Black-76 input is invalid: {error}"),
        })?;
    let (vanna, volga) = black76_vanna_volga(forward, strike, rate, vol, expiry, greeks.vega);
    let values = [
        greeks.delta,
        greeks.gamma,
        greeks.vega,
        greeks.theta,
        greeks.rho,
        vanna,
        volga,
    ];
    if let Some((greek_index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        const NAMES: [&str; 7] = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"];
        let location = index.map_or_else(
            || format!("greeks.{}", NAMES[greek_index]),
            |index| format!("greeks[{index}].{}", NAMES[greek_index]),
        );
        return Err(format!(
            "`{location}` is non-finite for the supplied inputs"
        ));
    }
    Ok(values)
}

/// Batch Black-Scholes pricing: one WASM call for N options.
///
/// All input slices must have the same length.  `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns prices in the same order, or throws a JavaScript error when the
/// input array lengths differ.
#[wasm_bindgen]
pub fn bs_price_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rates: &[f64],
    div_yields: &[f64],
    vols: &[f64],
    maturities: &[f64],
    is_calls: &[u8],
) -> Result<Vec<f64>, JsValue> {
    let n = spots.len();
    check_batch_lengths(
        "spots",
        n,
        &[
            ("strikes", strikes.len()),
            ("rates", rates.len()),
            ("div_yields", div_yields.len()),
            ("vols", vols.len()),
            ("maturities", maturities.len()),
            ("is_calls", is_calls.len()),
        ],
    )?;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let is_call = checked_option_flag(i, is_calls[i]).map_err(js_error)?;
        out.push(
            checked_bs_price(
                spots[i],
                strikes[i],
                rates[i],
                div_yields[i],
                vols[i],
                maturities[i],
                is_call,
                Some(i),
            )
            .map_err(js_error)?,
        );
    }
    Ok(out)
}

/// Batch Black-Scholes pricing with uniform market parameters.
///
/// This shape maps directly to the core analytic batch kernel. In the opt-in
/// `simd` package built with WebAssembly `simd128`, pairs of options execute
/// the explicit `f64x2` pricing path; the portable package uses the same API
/// with the scalar fallback. `spots` and `strikes` must have equal lengths.
#[wasm_bindgen]
pub fn bs_price_uniform_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rate: f64,
    div_yield: f64,
    vol: f64,
    maturity: f64,
    is_call: bool,
) -> Result<Vec<f64>, JsValue> {
    check_batch_lengths("spots", spots.len(), &[("strikes", strikes.len())])?;
    require_finite("rate", rate)?;
    require_finite("div_yield", div_yield)?;
    require_non_negative("vol", vol)?;
    require_non_negative("maturity", maturity)?;

    for index in 0..spots.len() {
        validate_positive_input("spot", "spots", Some(index), spots[index]).map_err(js_error)?;
        validate_positive_input("strike", "strikes", Some(index), strikes[index])
            .map_err(js_error)?;
    }

    let prices = openferric::engines::analytic::bs_price_batch(
        spots, strikes, rate, div_yield, vol, maturity, is_call,
    );
    for (index, &price) in prices.iter().enumerate() {
        checked_finite_output("price", Some(index), price).map_err(js_error)?;
    }
    Ok(prices)
}

/// Batch Black-Scholes delta, gamma, vega, and theta with uniform parameters.
///
/// The result is interleaved as
/// `[delta, gamma, vega, theta]` for each option. The opt-in SIMD128 package
/// uses the same explicit `f64x2` analytic backend as uniform batch pricing.
#[wasm_bindgen]
pub fn bsm_greeks_uniform_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rate: f64,
    div_yield: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
) -> Result<Vec<f64>, JsValue> {
    check_batch_lengths("spots", spots.len(), &[("strikes", strikes.len())])?;
    require_finite("rate", rate)?;
    require_finite("div_yield", div_yield)?;
    require_non_negative("vol", vol)?;
    require_non_negative("expiry", expiry)?;

    for index in 0..spots.len() {
        validate_positive_input("spot", "spots", Some(index), spots[index]).map_err(js_error)?;
        validate_positive_input("strike", "strikes", Some(index), strikes[index])
            .map_err(js_error)?;
    }

    let (delta, gamma, vega, theta) = openferric::engines::analytic::bs_greeks_batch(
        spots, strikes, rate, div_yield, vol, expiry, is_call,
    );
    let mut out = Vec::with_capacity(spots.len() * 4);
    for index in 0..spots.len() {
        let values = [delta[index], gamma[index], vega[index], theta[index]];
        for (greek_name, value) in ["delta", "gamma", "vega", "theta"].into_iter().zip(values) {
            if !value.is_finite() {
                return Err(js_error(format!(
                    "`greeks[{index}].{greek_name}` is non-finite for the supplied inputs"
                )));
            }
        }
        out.extend_from_slice(&values);
    }
    Ok(out)
}

/// Batch Black-76 pricing: one WASM call for N options.
///
/// All input slices must have the same length. `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns prices in the same order, or throws a JavaScript error when the
/// input array lengths differ.
#[wasm_bindgen]
pub fn black76_price_batch_wasm(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    vols: &[f64],
    maturities: &[f64],
    is_calls: &[u8],
) -> Result<Vec<f64>, JsValue> {
    let n = forwards.len();
    check_batch_lengths(
        "forwards",
        n,
        &[
            ("strikes", strikes.len()),
            ("rates", rates.len()),
            ("vols", vols.len()),
            ("maturities", maturities.len()),
            ("is_calls", is_calls.len()),
        ],
    )?;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let is_call = checked_option_flag(i, is_calls[i]).map_err(js_error)?;
        out.push(
            checked_black76_price(
                forwards[i],
                strikes[i],
                rates[i],
                vols[i],
                maturities[i],
                is_call,
                Some(i),
            )
            .map_err(js_error)?,
        );
    }
    Ok(out)
}

/// Batch BSM Greeks: one WASM call for N options.
///
/// All input slices must have the same length.  `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a flat array of 7 values per option:
/// `[delta, gamma, vega, theta, rho, vanna, volga]` repeated N times.
/// A JavaScript error is thrown when the input array lengths differ.
#[wasm_bindgen]
pub fn bsm_greeks_batch_wasm(
    spots: &[f64],
    strikes: &[f64],
    rates: &[f64],
    div_yields: &[f64],
    vols: &[f64],
    expiries: &[f64],
    is_calls: &[u8],
) -> Result<Vec<f64>, JsValue> {
    let n = spots.len();
    check_batch_lengths(
        "spots",
        n,
        &[
            ("strikes", strikes.len()),
            ("rates", rates.len()),
            ("div_yields", div_yields.len()),
            ("vols", vols.len()),
            ("expiries", expiries.len()),
            ("is_calls", is_calls.len()),
        ],
    )?;

    let mut out = Vec::with_capacity(n * 7);
    for i in 0..n {
        let is_call = checked_option_flag(i, is_calls[i]).map_err(js_error)?;
        let greeks = checked_bsm_greeks(
            spots[i],
            strikes[i],
            rates[i],
            div_yields[i],
            vols[i],
            expiries[i],
            is_call,
            Some(i),
        )
        .map_err(js_error)?;
        out.extend_from_slice(&greeks);
    }
    Ok(out)
}

/// Batch Black-76 Greeks: one WASM call for N options.
///
/// All input slices must have the same length. `is_calls` uses `1` = call,
/// `0` = put (wasm-bindgen cannot pass `&[bool]`).
/// Returns a flat array of 7 values per option:
/// `[delta, gamma, vega, theta, rho, vanna, volga]` repeated N times.
/// A JavaScript error is thrown when the input array lengths differ.
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
) -> Result<Vec<f64>, JsValue> {
    let n = forwards.len();
    check_batch_lengths(
        "forwards",
        n,
        &[
            ("strikes", strikes.len()),
            ("rates", rates.len()),
            ("vols", vols.len()),
            ("expiries", expiries.len()),
            ("is_calls", is_calls.len()),
        ],
    )?;

    let mut out = Vec::with_capacity(n * 7);
    for i in 0..n {
        let is_call = checked_option_flag(i, is_calls[i]).map_err(js_error)?;
        let greeks = checked_black76_greeks(
            forwards[i],
            strikes[i],
            rates[i],
            vols[i],
            expiries[i],
            is_call,
            Some(i),
        )
        .map_err(js_error)?;
        out.extend_from_slice(&greeks);
    }
    Ok(out)
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
) -> Result<f64, JsValue> {
    require_positive("spot", spot)?;
    require_positive("strike", strike)?;
    require_positive("barrier", barrier)?;
    require_non_negative("vol", vol)?;
    require_non_negative("maturity", maturity)?;
    require_finite("rate", rate)?;
    require_finite("div_yield", div_yield)?;

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
        _ => {
            return Err(js_error(
                "`barrier_type` must be one of up-in, up-out, down-in, down-out",
            ));
        }
    };
    Ok(barrier_price_closed_form_with_carry_and_rebate(
        ot, style, direction, spot, strike, barrier, rate, div_yield, vol, maturity, 0.0,
    ))
}

/// Simple fixed-rate bond dirty price using flat yield discounting.
fn checked_bond_price(
    face_value: f64,
    coupon_rate: f64,
    maturity_years: f64,
    yield_rate: f64,
    frequency: u32,
) -> Result<f64, String> {
    validate_positive_input("face_value", "face_values", None, face_value)?;
    validate_non_negative_input("coupon_rate", "coupon_rates", None, coupon_rate)?;
    validate_positive_input("maturity_years", "maturities", None, maturity_years)?;
    validate_finite_input("yield_rate", "yield_rates", None, yield_rate)?;
    if frequency == 0 {
        return Err("`frequency` must be positive".to_string());
    }

    let frequency_f64 = f64::from(frequency);
    let raw_periods = maturity_years * frequency_f64;
    if !raw_periods.is_finite() {
        return Err("`maturity_years * frequency` must be finite".to_string());
    }
    let periods = raw_periods.round();
    if periods < 1.0 {
        return Err(
            "`maturity_years * frequency` must produce at least one coupon period".to_string(),
        );
    }
    // Above 2^53, f64 cannot represent every integer period count exactly.
    if periods > 9_007_199_254_740_992.0 {
        return Err("coupon period count exceeds the exactly representable range".to_string());
    }
    // Allow ordinary multiplication roundoff, but never let a scale-relative
    // tolerance grow large enough to classify a representable fractional
    // period as an integer.
    let period_tolerance = (8.0 * f64::EPSILON * raw_periods.abs().max(1.0)).min(0.25);
    if (raw_periods - periods).abs() > period_tolerance {
        return Err(
            "`maturity_years * frequency` must be a whole number of coupon periods".to_string(),
        );
    }

    let rate_per_period = yield_rate / frequency_f64;
    let discount_base = 1.0 + rate_per_period;
    if !discount_base.is_finite() || discount_base <= 0.0 {
        return Err("`1 + yield_rate / frequency` must be finite and positive".to_string());
    }

    let coupon = face_value * coupon_rate / frequency_f64;
    if !coupon.is_finite() {
        return Err("coupon payment is non-finite for the supplied inputs".to_string());
    }

    let price = if rate_per_period == 0.0 {
        coupon.mul_add(periods, face_value)
    } else {
        // Geometric-series closed form. exp_m1 avoids cancellation when the
        // per-period yield is close to zero.
        let discount_exponent = -periods * rate_per_period.ln_1p();
        let maturity_discount = discount_exponent.exp();
        let annuity_factor = -discount_exponent.exp_m1() / rate_per_period;
        let coupon_pv = if coupon == 0.0 {
            0.0
        } else {
            coupon * annuity_factor
        };
        face_value.mul_add(maturity_discount, coupon_pv)
    };

    if !price.is_finite() || price < 0.0 {
        Err("bond price is non-finite or negative for the supplied inputs".to_string())
    } else {
        Ok(price)
    }
}

#[wasm_bindgen]
pub fn bond_price(
    face_value: f64,
    coupon_rate: f64,
    maturity_years: f64,
    yield_rate: f64,
    frequency: u32,
) -> Result<f64, JsValue> {
    checked_bond_price(
        face_value,
        coupon_rate,
        maturity_years,
        yield_rate,
        frequency,
    )
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // -- bs_price --

    #[test]
    fn bs_price_atm_call() {
        let price = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert!((price - 10.450_583_572_185_565).abs() < 2.0e-12);
    }

    #[test]
    fn bs_price_atm_put() {
        let price = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false).unwrap();
        assert!((price - 5.573_526_022_256_971).abs() < 2.0e-12);
    }

    #[test]
    fn bs_price_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.0;
        let t = 1.0;
        let call = bs_price(s, k, r, q, 0.20, t, true).unwrap();
        let put = bs_price(s, k, r, q, 0.20, t, false).unwrap();
        let s_adj = s * (-q * t).exp();
        let parity = call - put - (s_adj - k * (-r * t).exp());
        assert!(parity.abs() < 2.0e-12);
    }

    #[test]
    fn bs_price_with_dividend() {
        let with_div = bs_price(100.0, 100.0, 0.05, 0.03, 0.20, 1.0, true).unwrap();
        assert!((with_div - 8.652_528_553_942_709).abs() < 2.0e-12);
    }

    // -- bs_implied_vol --

    #[test]
    fn bs_implied_vol_round_trip() {
        let vol = 0.25;
        let price = bs_price(100.0, 100.0, 0.05, 0.0, vol, 1.0, true).unwrap();
        let recovered = bs_implied_vol(price, 100.0, 100.0, 0.05, 0.0, 1.0, true).unwrap();
        assert!((recovered - vol).abs() < 1.0e-12);
    }

    #[test]
    fn bs_implied_vol_put_round_trip() {
        let vol = 0.30;
        let price = bs_price(100.0, 110.0, 0.03, 0.0, vol, 0.5, false).unwrap();
        let recovered = bs_implied_vol(price, 100.0, 110.0, 0.03, 0.0, 0.5, false).unwrap();
        assert!((recovered - vol).abs() < 1.0e-12);
    }

    // -- bsm_greeks_wasm --

    #[test]
    fn bsm_greeks_call_has_7_values() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert_eq!(g.len(), 7);
    }

    #[test]
    fn bsm_greeks_call_delta_range() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert!((g[0] - 0.636_830_651_175_619_1).abs() < 2.0e-14);
    }

    #[test]
    fn bsm_greeks_put_delta_negative() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false).unwrap();
        assert!((g[0] + 0.363_169_348_824_380_9).abs() < 2.0e-14);
    }

    #[test]
    fn bsm_greeks_gamma_positive() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert!((g[1] - 0.018_762_017_345_846_895).abs() < 2.0e-15);
    }

    #[test]
    fn bsm_greeks_vega_positive() {
        let g = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert!((g[2] - 37.524_034_691_693_79).abs() < 2.0e-12);
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
        let batch = bs_price_batch_wasm(&spots, &strikes, &rates, &divs, &vols, &mats, &calls)
            .expect("matching batch lengths");
        assert_eq!(batch.len(), 2);
        let p1 = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        let p2 = bs_price(100.0, 110.0, 0.05, 0.0, 0.20, 1.0, false).unwrap();
        assert!((batch[0] - p1).abs() < TOL);
        assert!((batch[1] - p2).abs() < TOL);
    }

    #[test]
    fn checked_scalar_and_batch_bs_validation_are_consistent_and_indexed() {
        let scalar = checked_bs_price(-1.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, None)
            .expect_err("negative scalar spot must be rejected");
        let batch = checked_bs_price(-1.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, Some(3))
            .expect_err("negative batch spot must be rejected");
        assert!(scalar.contains("`spot`"));
        assert!(batch.contains("`spots[3]`"));

        let scalar = checked_bsm_greeks(100.0, 100.0, 0.05, 0.0, f64::NAN, 1.0, true, None)
            .expect_err("non-finite scalar volatility must be rejected");
        let batch = checked_bsm_greeks(100.0, 100.0, 0.05, 0.0, f64::NAN, 1.0, true, Some(4))
            .expect_err("non-finite batch volatility must be rejected");
        assert!(scalar.contains("`vol`"));
        assert!(batch.contains("`vols[4]`"));
    }

    #[test]
    fn batch_option_flags_accept_only_documented_zero_or_one_values() {
        assert!(!checked_option_flag(0, 0).unwrap());
        assert!(checked_option_flag(1, 1).unwrap());
        let error = checked_option_flag(3, 2).unwrap_err();
        assert!(error.contains("`is_calls[3]`"), "unexpected: {error}");
    }

    #[test]
    fn checked_pricing_rejects_non_finite_results() {
        let bs = checked_bs_price(100.0, 100.0, -f64::MAX, 0.0, 0.2, 1.0, false, None);
        assert!(bs.is_err());

        let black76 = checked_black76_price(100.0, 100.0, -f64::MAX, 0.2, 1.0, true, Some(2))
            .expect_err("overflowing discount factor must not cross the WASM ABI");
        assert!(black76.contains("price[2]"));
    }

    #[test]
    fn batch_length_mismatch_is_rejected() {
        let err = crate::error::batch_length_error("spots", 2, &[("strikes", 1)]).unwrap_err();
        assert!(err.contains("strikes"));
        assert!(
            crate::error::batch_length_error("spots", 2, &[("strikes", 2), ("vols", 2)]).is_ok()
        );
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
            bsm_greeks_batch_wasm(&spots, &strikes, &rates, &divs, &vols, &expiries, &calls)
                .expect("matching batch lengths");
        let single = bsm_greeks_wasm(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
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

        let black = black76_price_batch_wasm(&forwards, &strikes, &rates, &vols, &mats, &calls)
            .expect("matching batch lengths");
        let bsm = bs_price_batch_wasm(&forwards, &strikes, &rates, &rates, &vols, &mats, &calls)
            .expect("matching batch lengths");

        assert_eq!(black.len(), bsm.len());
        for i in 0..black.len() {
            assert!((black[i] - bsm[i]).abs() < TOL);
        }
    }

    #[test]
    fn checked_black76_rejects_invalid_domains_with_batch_index() {
        let price = checked_black76_price(-1.0, 100.0, 0.03, 0.2, 1.0, true, Some(6))
            .expect_err("negative forward must be rejected");
        assert!(price.contains("`forwards[6]`"));

        let greeks = checked_black76_greeks(100.0, 100.0, 0.03, -0.2, 1.0, true, Some(7))
            .expect_err("negative volatility must be rejected");
        assert!(greeks.contains("`vols[7]`"));
    }

    #[test]
    fn checked_black76_zero_volatility_retains_greeks_and_rejects_atm_kinks() {
        let discount = (-0.05_f64 * 1.5).exp();
        let price = 20.0 * discount;
        for (forward, is_call, sign) in [(120.0, true, 1.0), (80.0, false, -1.0)] {
            let greeks =
                checked_black76_greeks(forward, 100.0, 0.05, 0.0, 1.5, is_call, Some(2)).unwrap();
            let expected = [
                sign * discount,
                0.0,
                0.0,
                0.05 * price,
                -1.5 * price,
                0.0,
                0.0,
            ];
            for (actual, expected) in greeks.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1.0e-13);
            }
        }
        let error =
            checked_black76_greeks(100.0, 100.0, 0.05, 0.0, 1.5, true, Some(3)).unwrap_err();
        assert!(error.contains("greeks[3].delta"));
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
            black76_greeks_batch_wasm(&forwards, &strikes, &rates, &vols, &expiries, &calls)
                .expect("matching batch lengths");
        let bsm = bsm_greeks_batch_wasm(
            &forwards, &strikes, &rates, &rates, &vols, &expiries, &calls,
        )
        .expect("matching batch lengths");

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

    #[test]
    fn black76_deep_tail_put_greeks_use_stable_core_values() {
        let actual = checked_black76_greeks(5.0e18, 1.0e18, 0.0, 0.2, 1.0, false, Some(0)).unwrap();
        // Independent SciPy ndtr references for the Black-76 formulas.
        let expected = [
            -1.862_397_753_202_606_3e-16,
            1.539_548_175_331_966_1e-33,
            7.697_740_876_659_831e3,
            -7.697_740_876_659_832e2,
            -2.275_288_460_097_726_8e1,
            -6.117_540_594_728_432e-14,
            2.492_038_143_976_294_4e6,
        ];
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 2.0e-11,
                "Greek {index}: actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}"
            );
        }
    }

    // -- barrier_price --

    #[test]
    fn barrier_up_out_call_less_than_vanilla() {
        let bp = barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-out", true).unwrap();
        // QuantLib 1.43 AnalyticBarrierEngine reference.
        assert!((bp - 1.176_065_399_650_372_7).abs() < 2.0e-11);
    }

    #[test]
    fn barrier_in_plus_out_equals_vanilla() {
        let in_p = barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-in", true).unwrap();
        let out_p =
            barrier_price(100.0, 100.0, 120.0, 0.05, 0.0, 0.20, 1.0, "up-out", true).unwrap();
        let vanilla = bs_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true).unwrap();
        assert!((in_p - 9.274_518_172_535_206).abs() < 2.0e-11);
        assert!((out_p - 1.176_065_399_650_372_7).abs() < 2.0e-11);
        assert!((in_p + out_p - vanilla).abs() < 2.0e-12);
    }

    #[test]
    fn barrier_down_in_put() {
        let price =
            barrier_price(100.0, 100.0, 80.0, 0.05, 0.0, 0.20, 1.0, "down-in", false).unwrap();
        assert!((price - 3.952_510_513_175_199_4).abs() < 2.0e-11);
    }

    // -- bond_price --

    #[test]
    fn bond_price_par() {
        let price = bond_price(1000.0, 0.05, 10.0, 0.05, 2).unwrap();
        assert!((price - 1000.0).abs() < 2.0e-12);
    }

    #[test]
    fn bond_price_premium() {
        let price = bond_price(1000.0, 0.08, 10.0, 0.05, 2).unwrap();
        assert!((price - 1_233.837_434_284_703_7).abs() < 2.0e-12);
    }

    #[test]
    fn bond_price_discount() {
        let price = bond_price(1000.0, 0.03, 10.0, 0.05, 2).unwrap();
        assert!((price - 844.108_377_143_533_2).abs() < 2.0e-12);
    }

    #[test]
    fn bond_price_zero_yield_uses_exact_coupon_sum() {
        let price = checked_bond_price(1_000.0, 0.06, 10.0, 0.0, 2).unwrap();
        assert_eq!(price, 1_600.0);
    }

    #[test]
    fn bond_price_requires_a_valid_coupon_schedule_and_discount_base() {
        assert!(
            checked_bond_price(1_000.0, 0.05, 0.1, 0.05, 2)
                .unwrap_err()
                .contains("at least one")
        );
        assert!(
            checked_bond_price(1_000.0, 0.05, 1.25, 0.05, 2)
                .unwrap_err()
                .contains("whole number")
        );
        assert!(
            checked_bond_price(1_000.0, 0.05, 1.0, -2.0, 2)
                .unwrap_err()
                .contains("positive")
        );

        let large_half_period = 1_000_000_000_000_000.5_f64;
        assert_eq!(large_half_period.fract(), 0.5);
        assert!(
            checked_bond_price(1_000.0, 0.05, large_half_period, 0.05, 1)
                .unwrap_err()
                .contains("whole number")
        );
    }

    #[test]
    fn bond_price_large_period_count_is_finite_without_period_iteration() {
        let price = checked_bond_price(1_000.0, 0.05, 1_000_000.0, 0.05, 365)
            .expect("closed-form annuity should handle a large integral period count");
        assert!(price.is_finite());
        assert!(price > 0.0);
    }

    #[test]
    fn bond_price_rejects_overflowing_result() {
        let result = checked_bond_price(f64::MAX, f64::MAX, 1.0, 0.0, 1);
        assert!(result.is_err());
    }
}
