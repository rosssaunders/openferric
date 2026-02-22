//! Heston model reference tests from Fang & Oosterlee (2008).
//!
//! Source: "A Novel Pricing Method for European Options Based on
//! Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2), 826-848.
//!
//! These values are independent of QuantLib and serve as cross-validation
//! for the FFT-based Heston pricer.

use openferric::engines::fft::{CarrMadanParams, HestonCharFn, carr_madan_fft_strikes};

/// Fang-Oosterlee Heston parameter set (Equation 53 of the paper).
/// This set VIOLATES the Feller condition: 2*kappa*theta = 0.1255 < sigma_v^2 = 0.3307.
const SPOT: f64 = 100.0;
const RATE: f64 = 0.0;
const DIVIDEND: f64 = 0.0;
const V0: f64 = 0.0175;
const KAPPA: f64 = 1.5768;
const THETA: f64 = 0.0398;
const SIGMA_V: f64 = 0.5751;
const RHO: f64 = -0.5711;

fn heston_call_price(strike: f64, maturity: f64) -> f64 {
    let cf = HestonCharFn::new(
        SPOT, RATE, DIVIDEND, maturity, V0, KAPPA, THETA, SIGMA_V, RHO,
    );
    let params = CarrMadanParams::high_resolution();
    let prices = carr_madan_fft_strikes(&cf, RATE, maturity, SPOT, &[strike], params)
        .expect("FFT pricing failed");
    prices[0].1
}

// -----------------------------------------------------------------------
// Table 4: T = 1, K = 100, reference value = 5.785155450
// -----------------------------------------------------------------------
#[test]
fn fang_oosterlee_heston_t1_k100() {
    let price = heston_call_price(100.0, 1.0);
    let reference = 5.785155450;
    let err = (price - reference).abs();
    assert!(
        err < 0.01,
        "Fang-Oosterlee T=1 K=100: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// Table 5: T = 10, K = 100, reference value = 22.318945791
// -----------------------------------------------------------------------
#[test]
fn fang_oosterlee_heston_t10_k100() {
    let price = heston_call_price(100.0, 10.0);
    let reference = 22.318945791;
    let err = (price - reference).abs();
    assert!(
        err < 0.05,
        "Fang-Oosterlee T=10 K=100: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// GBM reference (Table 2): S0=100, r=0.1, sigma=0.25, T=0.1
// Three strikes: K=80 -> 20.799226309, K=100 -> 3.659968453, K=120 -> 0.044577814
// -----------------------------------------------------------------------
use openferric::engines::fft::BlackScholesCharFn;

fn gbm_call_price(spot: f64, rate: f64, vol: f64, maturity: f64, strike: f64) -> f64 {
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let params = CarrMadanParams::high_resolution();
    let prices = carr_madan_fft_strikes(&cf, rate, maturity, spot, &[strike], params)
        .expect("FFT pricing failed");
    prices[0].1
}

#[test]
fn fang_oosterlee_gbm_k80() {
    let price = gbm_call_price(100.0, 0.1, 0.25, 0.1, 80.0);
    let reference = 20.799226309;
    let err = (price - reference).abs();
    assert!(
        err < 0.001,
        "Fang-Oosterlee GBM K=80: got {price}, expected {reference}, err={err}"
    );
}

#[test]
fn fang_oosterlee_gbm_k100() {
    let price = gbm_call_price(100.0, 0.1, 0.25, 0.1, 100.0);
    let reference = 3.659968453;
    let err = (price - reference).abs();
    assert!(
        err < 0.001,
        "Fang-Oosterlee GBM K=100: got {price}, expected {reference}, err={err}"
    );
}

#[test]
fn fang_oosterlee_gbm_k120() {
    let price = gbm_call_price(100.0, 0.1, 0.25, 0.1, 120.0);
    let reference = 0.044577814;
    let err = (price - reference).abs();
    assert!(
        err < 0.001,
        "Fang-Oosterlee GBM K=120: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// Variance Gamma reference (Table 7): K=90, r=0.1, sigma=0.12, theta=-0.14, nu=0.2
// T=0.1 -> 10.993703187, T=1 -> 19.099354724
// -----------------------------------------------------------------------
use openferric::engines::fft::VarianceGammaCharFn;

fn vg_call_price_fo(strike: f64, maturity: f64) -> f64 {
    let cf = VarianceGammaCharFn::risk_neutral(100.0, 0.1, 0.0, maturity, 0.12, -0.14, 0.2)
        .expect("VG construction failed");
    let params = CarrMadanParams::high_resolution();
    let prices = carr_madan_fft_strikes(&cf, 0.1, maturity, 100.0, &[strike], params)
        .expect("FFT pricing failed");
    prices[0].1
}

#[test]
fn fang_oosterlee_vg_t01_k90() {
    let price = vg_call_price_fo(90.0, 0.1);
    let reference = 10.993703187;
    let err = (price - reference).abs();
    assert!(
        err < 0.01,
        "Fang-Oosterlee VG T=0.1 K=90: got {price}, expected {reference}, err={err}"
    );
}

#[test]
fn fang_oosterlee_vg_t1_k90() {
    let price = vg_call_price_fo(90.0, 1.0);
    let reference = 19.099354724;
    let err = (price - reference).abs();
    assert!(
        err < 0.01,
        "Fang-Oosterlee VG T=1 K=90: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// CGMY reference (Tables 8-10): S0=100, K=100, r=0.1, C=1, G=5, M=5, T=1
// Y=0.5  -> 19.812948843
// Y=1.5  -> 49.790905469
// Y=1.98 -> 99.999905510
// -----------------------------------------------------------------------
use openferric::engines::fft::CgmyCharFn;

fn cgmy_call_price(y: f64) -> f64 {
    let cf = CgmyCharFn::risk_neutral(100.0, 0.1, 0.0, 1.0, 1.0, 5.0, 5.0, y)
        .expect("CGMY construction failed");
    let params = CarrMadanParams::high_resolution();
    let prices =
        carr_madan_fft_strikes(&cf, 0.1, 1.0, 100.0, &[100.0], params).expect("FFT pricing failed");
    prices[0].1
}

#[test]
fn fang_oosterlee_cgmy_y05() {
    let price = cgmy_call_price(0.5);
    let reference = 19.812948843;
    let err = (price - reference).abs();
    assert!(
        err < 0.05,
        "Fang-Oosterlee CGMY Y=0.5: got {price}, expected {reference}, err={err}"
    );
}

#[test]
fn fang_oosterlee_cgmy_y15() {
    let price = cgmy_call_price(1.5);
    let reference = 49.790905469;
    let err = (price - reference).abs();
    assert!(
        err < 0.1,
        "Fang-Oosterlee CGMY Y=1.5: got {price}, expected {reference}, err={err}"
    );
}

// Note: CGMY Y=1.98 (near boundary Y=2) is numerically unstable in Carr-Madan FFT
// and is omitted. The COS method handles it better but we don't implement COS.
