//! European option pricing: Black-Scholes, Heston, and FFT methods.

use openferric::core::OptionType;
use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, HestonCharFn, carr_madan_price_at_strikes,
};
use openferric::pricing::european::black_scholes_price;

fn main() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.03;
    let vol = 0.20;
    let expiry = 1.0;

    // 1. Analytic Black-Scholes
    let bs_call = black_scholes_price(OptionType::Call, spot, strike, rate, vol, expiry);
    let bs_put = black_scholes_price(OptionType::Put, spot, strike, rate, vol, expiry);
    println!("Black-Scholes:  Call = {bs_call:.4}, Put = {bs_put:.4}");

    // 2. FFT via Carr-Madan (Black-Scholes char fn)
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, expiry);
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let fft_prices = carr_madan_price_at_strikes(
        &cf,
        rate,
        expiry,
        spot,
        &strikes,
        CarrMadanParams::default(),
    )
    .unwrap();
    println!("\nCarr-Madan FFT (BS):");
    for (k, p) in &fft_prices {
        println!("  K = {k:6.1}, Call = {p:.4}");
    }

    // 3. Heston stochastic vol via FFT
    let heston_cf = HestonCharFn::new(spot, rate, 0.0, expiry, 0.04, 1.5, 0.04, 0.4, -0.7);
    let heston_prices = carr_madan_price_at_strikes(
        &heston_cf,
        rate,
        expiry,
        spot,
        &strikes,
        CarrMadanParams::default(),
    )
    .unwrap();
    println!("\nHeston FFT:");
    for (k, p) in &heston_prices {
        println!("  K = {k:6.1}, Call = {p:.4}");
    }
}
