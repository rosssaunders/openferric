use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, HestonCharFn, carr_madan_fft, carr_madan_fft_complex,
    carr_madan_fft_greeks, carr_madan_fft_strikes, carr_madan_price_at_strikes, heston_price_fft,
    interpolate_strike_prices,
};
use openferric::models::VarianceGamma;
use openferric::pricing::OptionType;
use openferric::pricing::european::{black_scholes_greeks, black_scholes_price};

#[test]
fn carr_madan_bs_matches_analytic_at_floating_point_precision() {
    let spot = 100.0;
    let rate = 0.03;
    let maturity = 0.25;
    let vol = 0.2;

    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let params = CarrMadanParams::default();
    let slice = carr_madan_fft(&cf, rate, maturity, spot, params).expect("fft slice");

    let center = params.n / 2;
    let mut max_err = 0.0_f64;
    for i in 0..10 {
        let idx = center + i * 8;
        let (k, fft_call) = slice[idx];
        let bs = black_scholes_price(OptionType::Call, spot, k, rate, vol, maturity);
        max_err = max_err.max((fft_call - bs).abs());
    }
    assert!(max_err < 3.0e-13, "maximum BS error={max_err}");
}

#[test]
fn carr_madan_heston_exact_matches_its_fft_grid_at_floating_point_precision() {
    let spot = 100.0;
    let rate = 0.02;
    let q = 0.0;
    let maturity = 0.25;

    let v0 = 0.04;
    let kappa = 4.0;
    let theta = 0.04;
    let sigma_v = 0.02;
    let rho = -0.2;

    let heston_cf = HestonCharFn::new(spot, rate, q, maturity, v0, kappa, theta, sigma_v, rho);
    let fft_prices = carr_madan_fft(&heston_cf, rate, maturity, spot, CarrMadanParams::default())
        .expect("heston fft prices");
    let center = CarrMadanParams::default().n / 2;
    let strikes: Vec<f64> = (0..10).map(|i| fft_prices[center + i * 6].0).collect();

    let exact = heston_price_fft(
        spot, &strikes, rate, q, v0, kappa, theta, sigma_v, rho, maturity,
    );
    let interp = interpolate_strike_prices(&fft_prices, &strikes);

    let mut max_err = 0.0_f64;
    for ((k_exact, exact_call), (k_interp, interp_call)) in exact.iter().zip(interp.iter()) {
        assert!((k_exact - k_interp).abs() < 1e-12);
        max_err = max_err.max((exact_call - interp_call).abs());
    }
    assert!(max_err < 2.0e-13, "maximum Heston grid error={max_err}");
}

#[test]
fn variance_gamma_prices_positive_and_show_heavier_tails_than_bs() {
    let vg = VarianceGamma {
        sigma: 0.12,
        theta: 0.0,
        nu: 0.6,
    };

    let spot = 100.0;
    let rate = 0.02;
    let q = 0.0;
    let maturity = 1.0;
    let strikes = [90.0, 100.0, 140.0];

    let vg_prices = vg
        .european_calls_fft(
            spot,
            &strikes,
            rate,
            q,
            maturity,
            CarrMadanParams::default(),
        )
        .expect("vg fft prices");

    assert!(vg_prices.iter().all(|(_, p)| *p > 0.0 && p.is_finite()));

    let bs_vol = vg.variance_rate().sqrt();
    let vg_otm = vg_prices[2].1;
    let bs_otm = black_scholes_price(OptionType::Call, spot, 140.0, rate, bs_vol, maturity);
    assert!(
        vg_otm > bs_otm,
        "Expected heavier tails in VG: vg_otm={vg_otm}, bs_otm={bs_otm}"
    );
}

#[test]
fn fft_delta_matches_closed_form_black_scholes() {
    let spot = 100.0;
    let rate = 0.03;
    let maturity = 1.0;
    let vol = 0.25;
    let strike = 100.0;

    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let greeks = carr_madan_fft_greeks(&cf, rate, maturity, spot, CarrMadanParams::default())
        .expect("fft greeks");

    let greeks_pairs: Vec<(f64, f64)> = greeks.iter().map(|g| (g.strike, g.delta)).collect();
    let delta_fft = interpolate_strike_prices(&greeks_pairs, &[strike])[0].1;

    let analytic_delta =
        black_scholes_greeks(OptionType::Call, spot, strike, rate, vol, maturity).delta;

    assert!(
        (delta_fft - analytic_delta).abs() < 2.0e-12,
        "delta mismatch fft={delta_fft} analytic={analytic_delta}"
    );
}

#[test]
fn fractional_fft_exact_strikes_match_standard_fft_grid_at_roundoff() {
    let spot = 100.0;
    let rate = 0.02;
    let maturity = 1.0;
    let vol = 0.2;

    let params = CarrMadanParams {
        n: 8192,
        eta: 0.25,
        alpha: 1.5,
    };
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);

    let fft_slice = carr_madan_fft(&cf, rate, maturity, spot, params).expect("fft slice");
    let center = params.n / 2;
    let exact_strikes: Vec<f64> = (0..24).map(|i| fft_slice[center - 60 + i * 5].0).collect();

    let frft_exact = carr_madan_price_at_strikes(&cf, rate, maturity, spot, &exact_strikes, params)
        .expect("frft exact strikes");
    let fft_interp = interpolate_strike_prices(&fft_slice, &exact_strikes);

    let mut max_err = 0.0_f64;
    for ((_, p_frft), (_, p_fft)) in frft_exact.iter().zip(fft_interp.iter()) {
        max_err = max_err.max((p_frft - p_fft).abs());
    }
    assert!(max_err < 5.0e-12, "maximum FRFT/FFT error={max_err}");
}

#[test]
fn fft_strike_helper_matches_bs_with_measured_default_grid_interpolation_error() {
    let spot = 100.0;
    let rate = 0.02;
    let maturity = 1.0;
    let vol = 0.22;

    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let strikes = [85.0, 100.0, 120.0];

    let via_interp = carr_madan_fft_strikes(
        &cf,
        rate,
        maturity,
        spot,
        &strikes,
        CarrMadanParams::default(),
    )
    .expect("interpolated strikes");
    let via_exact = carr_madan_price_at_strikes(
        &cf,
        rate,
        maturity,
        spot,
        &strikes,
        CarrMadanParams::default(),
    )
    .expect("exact strikes");

    let mut max_interp_error = 0.0_f64;
    for (((_, interpolated), (_, exact)), strike) in
        via_interp.iter().zip(via_exact.iter()).zip(strikes)
    {
        let analytic = black_scholes_price(OptionType::Call, spot, strike, rate, vol, maturity);
        assert!((exact - analytic).abs() < 3.0e-12);
        max_interp_error = max_interp_error.max((interpolated - analytic).abs());
    }
    // This is the measured linear-in-log-strike interpolation error for the
    // stated default 4096-point grid, not an acceptable price range.
    assert!(
        max_interp_error < 4.9e-4,
        "maximum default-grid interpolation error={max_interp_error}"
    );
}

#[test]
fn carr_madan_dispatch_matches_complex_fft_at_roundoff() {
    let spot = 100.0;
    let rate = 0.03;
    let maturity = 1.0;
    let vol = 0.2;
    let params = CarrMadanParams {
        n: 4096,
        eta: 0.25,
        alpha: 1.5,
    };
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);

    let dispatch = carr_madan_fft(&cf, rate, maturity, spot, params).expect("dispatch prices");
    let complex_only =
        carr_madan_fft_complex(&cf, rate, maturity, spot, params).expect("complex prices");

    for ((k1, p1), (k2, p2)) in dispatch.iter().zip(complex_only.iter()) {
        assert!((k1 - k2).abs() < 1e-12);
        assert!(
            (p1 - p2).abs() < 1e-10,
            "dispatch/complex mismatch k={k1} dispatch={p1} complex={p2}"
        );
    }
}
