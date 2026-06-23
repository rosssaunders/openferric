use wasm_bindgen::prelude::*;

use crate::error::{js_error, require_finite, require_non_negative, require_positive};
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
) -> Result<f64, JsValue> {
    validate_heston_inputs(spot, &[strike], v0, kappa, theta, sigma_v, rho, maturity)?;

    if maturity <= 0.0 {
        return Ok(heston_intrinsic(spot, strike, is_call));
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
    )?
    .into_iter()
    .next()
    .ok_or_else(|| js_error("Heston FFT returned no prices"))?;

    if !call_price.is_finite() {
        return Err(js_error("Heston FFT returned a non-finite price"));
    }

    Ok(option_price_from_call(
        call_price, spot, strike, rate, div_yield, maturity, is_call,
    ))
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
) -> Result<Vec<f64>, JsValue> {
    use openferric::engines::fft::{CarrMadanContext, CarrMadanParams, HestonCharFn};

    if strikes.is_empty() {
        return Ok(Vec::new());
    }
    validate_heston_inputs(spot, strikes, v0, kappa, theta, sigma_v, rho, maturity)?;
    require_finite("rate", rate)?;
    require_finite("div_yield", div_yield)?;

    let cf = HestonCharFn::new(
        spot, rate, div_yield, maturity, v0, kappa, theta, sigma_v, rho,
    );
    let ctx = match CarrMadanContext::new(&cf, rate, maturity, spot, CarrMadanParams::default()) {
        Ok(ctx) => ctx,
        Err(e) => return Err(js_error(format!("Heston FFT setup failed: {e}"))),
    };

    ctx.price_strikes(strikes)
        .map(|pairs| pairs.into_iter().map(|(_, p)| p).collect())
        .map_err(|e| js_error(format!("Heston FFT pricing failed: {e}")))
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
) -> Result<f64, JsValue> {
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
    )?
    .into_iter()
    .next()
    .ok_or_else(|| js_error("Heston FFT returned no prices"))
}

fn validate_heston_inputs(
    spot: f64,
    strikes: &[f64],
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> Result<(), JsValue> {
    require_positive("spot", spot)?;
    require_non_negative("maturity", maturity)?;
    require_non_negative("v0", v0)?;
    require_positive("kappa", kappa)?;
    require_non_negative("theta", theta)?;
    require_non_negative("sigma_v", sigma_v)?;
    require_finite("rho", rho)?;
    if !(-1.0..=1.0).contains(&rho) {
        return Err(js_error("`rho` must be between -1 and 1"));
    }
    for &strike in strikes {
        require_positive("strike", strike)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Alan Lewis reference parameters
    const SPOT: f64 = 100.0;
    const V0: f64 = 0.04;
    const RHO: f64 = -0.5;
    const SIGMA_V: f64 = 1.0;
    const KAPPA: f64 = 4.0;
    const THETA: f64 = 0.25;
    const R: f64 = 0.01;
    const Q: f64 = 0.02;
    const T: f64 = 1.0;

    #[test]
    fn heston_fft_price_reference_k100() {
        let price = heston_fft_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T).unwrap();
        assert!((price - 16.070154917028834).abs() < 0.05);
    }

    #[test]
    fn heston_fft_price_reference_k80() {
        let price = heston_fft_price(SPOT, 80.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T).unwrap();
        assert!((price - 26.774758743998854).abs() < 0.05);
    }

    #[test]
    fn heston_fft_prices_batch() {
        let prices = heston_fft_prices(
            SPOT,
            &[80.0, 100.0, 120.0],
            R,
            Q,
            V0,
            KAPPA,
            THETA,
            SIGMA_V,
            RHO,
            T,
        )
        .unwrap();
        assert_eq!(prices.len(), 3);
        // Call prices decrease with strike
        assert!(prices[0] > prices[1]);
        assert!(prices[1] > prices[2]);
    }

    #[test]
    fn heston_fft_prices_empty() {
        let prices = heston_fft_prices(SPOT, &[], R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T).unwrap();
        assert!(prices.is_empty());
    }

    #[test]
    fn heston_fft_price_matches_batch() {
        let scalar =
            heston_fft_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T).unwrap();
        let batch =
            heston_fft_prices(SPOT, &[100.0], R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T).unwrap();
        assert!((scalar - batch[0]).abs() < 1e-10);
    }

    #[test]
    fn heston_price_call_reference() {
        let price =
            heston_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T, true).unwrap();
        assert!((price - 16.070154917028834).abs() < 0.05);
    }

    #[test]
    fn heston_price_put_reference() {
        let price =
            heston_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T, false).unwrap();
        assert!((price - 17.055_270_961_270_11).abs() < 0.05);
    }

    #[test]
    fn heston_price_put_call_parity() {
        let call =
            heston_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T, true).unwrap();
        let put =
            heston_price(SPOT, 100.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, T, false).unwrap();
        let fwd = SPOT * (-Q * T).exp();
        let pv_k = 100.0 * (-R * T).exp();
        let parity = call - put - (fwd - pv_k);
        assert!(parity.abs() < 1e-4);
    }

    #[test]
    fn heston_price_zero_maturity() {
        let call =
            heston_price(SPOT, 90.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, 0.0, true).unwrap();
        assert!((call - 10.0).abs() < 1e-10); // intrinsic
        let put = heston_price(
            SPOT, 110.0, R, Q, V0, KAPPA, THETA, SIGMA_V, RHO, 0.0, false,
        )
        .unwrap();
        assert!((put - 10.0).abs() < 1e-10); // intrinsic
    }
}
