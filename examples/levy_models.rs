//! Lévy process pricing: Variance-Gamma, CGMY, and NIG via FFT.

use openferric::engines::fft::CarrMadanParams;
use openferric::models::{Cgmy, Nig, VarianceGamma};

fn main() {
    let spot = 100.0;
    let rate = 0.03;
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let params = CarrMadanParams::default();

    // Variance-Gamma
    let vg = VarianceGamma { sigma: 0.2, theta: -0.1, nu: 0.2 };
    let vg_prices = vg.european_calls_fft(spot, &strikes, rate, 0.0, 1.0, params).unwrap();
    println!("Variance-Gamma:");
    for (k, p) in &vg_prices { println!("  K={k:6.1}  C={p:.4}"); }

    // CGMY
    let cgmy = Cgmy { c: 1.0, g: 5.0, m: 10.0, y: 0.5 };
    let cgmy_prices = cgmy.european_calls_fft(spot, &strikes, rate, 0.0, 1.0, params).unwrap();
    println!("\nCGMY (C=1, G=5, M=10, Y=0.5):");
    for (k, p) in &cgmy_prices { println!("  K={k:6.1}  C={p:.4}"); }

    // NIG
    let nig = Nig { alpha: 15.0, beta: -5.0, delta: 0.5 };
    let nig_prices = nig.european_calls_fft(spot, &strikes, rate, 0.0, 1.0, params).unwrap();
    println!("\nNIG (α=15, β=-5, δ=0.5):");
    for (k, p) in &nig_prices { println!("  K={k:6.1}  C={p:.4}"); }
}
