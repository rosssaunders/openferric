//! Variance-Gamma model wrapper tests derived from QuantLib's variancegamma.cpp.
//!
//! Source: vendor/QuantLib/test-suite/variancegamma.cpp and the upstream
//! QuantLib test-suite. The expected prices are QuantLib's cached values.

use num_complex::Complex;

use openferric::engines::fft::CarrMadanParams;
use openferric::models::VarianceGamma;

#[test]
fn variance_gamma_model_prices_quantlib_grid_set_1() {
    let model = VarianceGamma {
        sigma: 0.20,
        nu: 0.05,
        theta: -0.50,
    };
    let strikes = [5550.0, 5800.0, 6000.0, 6200.0, 6500.0];
    let expected = [955.1637, 799.6303, 687.2032, 585.6379, 453.4700];

    let prices = model
        .european_calls_fft(
            6000.0,
            &strikes,
            0.05,
            0.0,
            1.0,
            CarrMadanParams::high_resolution(),
        )
        .expect("VG FFT pricing succeeds");

    for ((strike, price), expected) in prices.iter().zip(expected) {
        let err = (price - expected).abs();
        assert!(
            err < 5.0e-5,
            "strike={strike} expected={expected} got={price} err={err}"
        );
    }
}

#[test]
fn variance_gamma_model_prices_quantlib_grid_set_2() {
    let model = VarianceGamma {
        sigma: 0.15,
        nu: 0.01,
        theta: -0.50,
    };
    let strikes = [5550.0, 5800.0, 6000.0, 6200.0, 6500.0];
    let expected = [732.8705, 570.5068, 457.9064, 361.1102, 244.6552];

    let prices = model
        .european_calls_fft(
            6000.0,
            &strikes,
            0.05,
            0.02,
            1.0,
            CarrMadanParams::high_resolution(),
        )
        .expect("VG FFT pricing succeeds");

    for ((strike, price), expected) in prices.iter().zip(expected) {
        let err = (price - expected).abs();
        assert!(
            err < 7.0e-4,
            "strike={strike} expected={expected} got={price} err={err}"
        );
    }
}

#[test]
fn variance_gamma_characteristic_function_and_path_api_are_well_behaved() {
    let model = VarianceGamma {
        sigma: 0.2,
        theta: -0.1,
        nu: 0.2,
    };

    let cf_at_zero = model
        .characteristic_fn(Complex::new(0.0, 0.0), 100.0, 0.03, 0.01, 1.0)
        .expect("characteristic function succeeds");
    assert!((cf_at_zero - Complex::new(1.0, 0.0)).norm() < 1.0e-12);

    let path = model
        .simulate_path(100.0, 0.03, 0.01, 1.0, 16, 42)
        .expect("path simulation succeeds");
    assert_eq!(path.len(), 17);
    assert!(path.iter().all(|spot| spot.is_finite() && *spot > 0.0));
}

#[test]
fn variance_gamma_rejects_invalid_or_non_martingale_parameters() {
    assert!(
        VarianceGamma {
            sigma: -0.1,
            theta: 0.0,
            nu: 0.2,
        }
        .validate()
        .is_err()
    );
    assert!(
        VarianceGamma {
            sigma: 0.2,
            theta: 0.0,
            nu: 0.0,
        }
        .validate()
        .is_err()
    );
    assert!(
        VarianceGamma {
            sigma: 2.0,
            theta: 0.0,
            nu: 1.0,
        }
        .martingale_correction()
        .is_err()
    );
}
