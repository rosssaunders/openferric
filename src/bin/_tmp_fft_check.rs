//! Command-line entry point for Tmp Fft Check workflows.
//!
//! This binary wires OpenFerric models and engines into an executable utility.

use openferric::core::PricingEngine;
use openferric::engines::analytic::HestonEngine;
use openferric::engines::fft::{
    CarrMadanParams, HestonCharFn, carr_madan_fft, interpolate_strike_prices,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;

fn main() {
    let spot = 100.0;
    let rate = 0.02;
    let q = 0.0;
    let maturity = 1.0;

    let v0 = 0.04;
    let kappa = 3.0;
    let theta = 0.04;
    let sigma_v = 0.1;
    let rho = -0.2;

    let heston_cf = HestonCharFn::new(spot, rate, q, maturity, v0, kappa, theta, sigma_v, rho);
    let params = CarrMadanParams::default();
    let slice = carr_madan_fft(&heston_cf, rate, maturity, spot, params).unwrap();

    let engine = HestonEngine::new(v0, kappa, theta, sigma_v, rho);
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .flat_vol(0.2)
        .build()
        .unwrap();

    let strikes = [
        300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0,
    ];
    let fft = interpolate_strike_prices(&slice, &strikes);
    for (k, fft_px) in fft {
        let gl = engine
            .price(&VanillaOption::european_call(k, maturity), &market)
            .unwrap()
            .price;
        let diff = (fft_px - gl).abs();
        println!("k={k} fft={fft_px:.8e} gl={gl:.8e} diff={diff:.8e}");
    }
}
