//! FFT Lévy process reference tests from fypy (jkirkby3/fypy).
//!
//! Source: Test_Carr_Madan_European.py in the fypy library.
//! Reference values computed with N=2^20 grid points (very high accuracy),
//! validated to 5 decimal places.
//!
//! Common parameters: S0=100, r=0.05, q=0.01, T=1.0, K=100 (ATM).

use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, CgmyCharFn, NigCharFn, VarianceGammaCharFn,
    carr_madan_fft_strikes, carr_madan_price_at_strikes,
};

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const DIVIDEND: f64 = 0.01;
const MATURITY: f64 = 1.0;
const STRIKE: f64 = 100.0;

fn price_call(cf: &impl openferric::engines::fft::CharacteristicFunction) -> f64 {
    let params = CarrMadanParams::high_resolution();
    let prices = carr_madan_fft_strikes(cf, RATE, MATURITY, SPOT, &[STRIKE], params)
        .expect("FFT pricing failed");
    prices[0].1
}

// -----------------------------------------------------------------------
// Black-Scholes: sigma=0.15 -> 7.94871378854164
// -----------------------------------------------------------------------
#[test]
fn fypy_bs_atm_call() {
    let cf = BlackScholesCharFn::new(SPOT, RATE, DIVIDEND, 0.15, MATURITY);
    let price = price_call(&cf);
    let reference = 7.94871378854164;
    let err = (price - reference).abs();
    assert!(
        err < 2.0e-12,
        "fypy BS: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// Variance Gamma: sigma=0.2, theta=0.1, nu=0.85 -> 10.13935062748614
// -----------------------------------------------------------------------
#[test]
fn fypy_vg_atm_call() {
    let cf = VarianceGammaCharFn::risk_neutral(SPOT, RATE, DIVIDEND, MATURITY, 0.2, 0.1, 0.85)
        .expect("VG construction failed");
    let price = price_call(&cf);
    let reference = 10.13935062748614;
    let err = (price - reference).abs();
    assert!(
        err < 5.0e-9,
        "fypy VG: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// NIG: alpha=15, beta=-5, delta=0.5 -> 9.63000693130414
// -----------------------------------------------------------------------
#[test]
fn fypy_nig_atm_call() {
    let cf = NigCharFn::risk_neutral(SPOT, RATE, DIVIDEND, MATURITY, 15.0, -5.0, 0.5)
        .expect("NIG construction failed");
    let price = price_call(&cf);
    let reference = 9.63000693130414;
    let err = (price - reference).abs();
    assert!(
        err < 5.0e-12,
        "fypy NIG: got {price}, expected {reference}, err={err}"
    );
}

// -----------------------------------------------------------------------
// CGMY: C=0.02, G=5.0, M=15.0, Y=1.2 -> 5.80222163947386
// -----------------------------------------------------------------------
#[test]
fn fypy_cgmy_atm_call() {
    let cf = CgmyCharFn::risk_neutral(SPOT, RATE, DIVIDEND, MATURITY, 0.02, 5.0, 15.0, 1.2)
        .expect("CGMY construction failed");
    let price = price_call(&cf);
    let reference = 5.80222163947386;
    let err = (price - reference).abs();
    assert!(
        err < 5.0e-6,
        "fypy CGMY: got {price}, expected {reference}, err={err}"
    );
}

/// Deterministic default-grid regression for the CGMY production path.
///
/// The fypy assertion above is an external, converged-method reference and
/// deliberately carries its source grid error. These values instead lock the
/// exact finite Carr-Madan grid (`n=4096`, `eta=0.25`, `alpha=1.5`) used by
/// OpenFerric so a numerical implementation change cannot hide inside that
/// external tolerance.
#[test]
fn cgmy_default_finite_grid_binary64_lock() {
    let cf = CgmyCharFn::risk_neutral(SPOT, RATE, DIVIDEND, MATURITY, 0.02, 5.0, 15.0, 1.2)
        .expect("CGMY construction failed");
    let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
    let expected: [f64; 5] = [
        23.005_036_990_350_163,
        13.817_853_639_255_867,
        5.802_224_526_781_332,
        1.292_798_206_377_928_9,
        0.184_960_954_698_136_45,
    ];
    let actual = carr_madan_price_at_strikes(
        &cf,
        RATE,
        MATURITY,
        SPOT,
        &strikes,
        CarrMadanParams::default(),
    )
    .expect("CGMY default-grid pricing failed");

    for (((actual_strike, actual_price), expected_strike), expected_price) in
        actual.iter().zip(strikes).zip(expected)
    {
        assert_eq!(*actual_strike, expected_strike);
        let binary64_budget = 256.0 * f64::EPSILON * expected_price.abs().max(1.0);
        assert!(
            (actual_price - expected_price).abs() <= binary64_budget,
            "CGMY finite-grid lock drifted at K={expected_strike}: actual={actual_price}, expected={expected_price}, budget={binary64_budget}"
        );
    }
}

// -----------------------------------------------------------------------
// QuantLib VG reference set 1: S=6000, r=0.05, q=0, sigma=0.20, nu=0.05, theta=-0.50
// Selected strikes from the 21-point grid
// -----------------------------------------------------------------------
#[test]
fn quantlib_vg_set1_selected_strikes() {
    let spot = 6000.0;
    let rate = 0.05;
    let q = 0.0;
    let sigma = 0.20;
    let theta = -0.50;
    let nu = 0.05;
    let maturity = 1.0;

    let cf = VarianceGammaCharFn::risk_neutral(spot, rate, q, maturity, sigma, theta, nu)
        .expect("VG construction failed");
    let params = CarrMadanParams::high_resolution();

    let strikes = vec![5550.0, 5800.0, 6000.0, 6200.0, 6500.0];
    let expected = [955.1637, 799.6303, 687.2032, 585.6379, 453.4700];

    let prices = carr_madan_fft_strikes(&cf, rate, maturity, spot, &strikes, params)
        .expect("FFT pricing failed");
    let locked_grid = [
        955.164_400_054_570_8,
        799.630_853_973_125_5,
        687.203_178_347_213_5,
        585.639_908_950_341_9,
        453.471_440_785_217_1,
    ];

    for (i, ((_, call), &exp)) in prices.iter().zip(expected.iter()).enumerate() {
        let err = (call - exp).abs();
        assert!((call - locked_grid[i]).abs() < 1.0e-8);
        assert!(
            err < 2.1e-3,
            "QuantLib VG set1 strike {}: got {call}, expected {exp}, err={err}",
            strikes[i]
        );
    }
}

// -----------------------------------------------------------------------
// QuantLib VG reference set 2: S=6000, r=0.05, q=0.02, sigma=0.15, nu=0.01, theta=-0.50
// Selected strikes
// -----------------------------------------------------------------------
#[test]
fn quantlib_vg_set2_selected_strikes() {
    let spot = 6000.0;
    let rate = 0.05;
    let q = 0.02;
    let sigma = 0.15;
    let theta = -0.50;
    let nu = 0.01;
    let maturity = 1.0;

    let cf = VarianceGammaCharFn::risk_neutral(spot, rate, q, maturity, sigma, theta, nu)
        .expect("VG construction failed");
    let params = CarrMadanParams::high_resolution();

    let strikes = vec![5550.0, 5800.0, 6000.0, 6200.0, 6500.0];
    let expected = [732.8705, 570.5068, 457.9064, 361.1102, 244.6552];

    let prices = carr_madan_fft_strikes(&cf, rate, maturity, spot, &strikes, params)
        .expect("FFT pricing failed");
    let locked_grid = [
        732.872_346_557_732_6,
        570.508_278_425_467_6,
        457.906_653_569_325,
        361.113_791_859_400_5,
        244.657_526_431_685_48,
    ];

    for (i, ((_, call), &exp)) in prices.iter().zip(expected.iter()).enumerate() {
        let err = (call - exp).abs();
        assert!((call - locked_grid[i]).abs() < 1.0e-8);
        assert!(
            err < 3.7e-3,
            "QuantLib VG set2 strike {}: got {call}, expected {exp}, err={err}",
            strikes[i]
        );
    }
}
