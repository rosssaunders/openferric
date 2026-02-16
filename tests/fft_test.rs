use openferric::core::PricingEngine;
use openferric::engines::analytic::HestonEngine;
use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, HestonCharFn, carr_madan_fft, carr_madan_fft_greeks,
    carr_madan_fft_strikes, carr_madan_price_at_strikes, interpolate_strike_prices,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::models::VarianceGamma;
use openferric::pricing::OptionType;
use openferric::pricing::european::black_scholes_price;

#[test]
fn carr_madan_bs_matches_analytic_within_1e4_for_ten_strikes() {
    let spot = 100.0;
    let rate = 0.03;
    let maturity = 0.25;
    let vol = 0.2;

    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let params = CarrMadanParams::default();
    let slice = carr_madan_fft(&cf, rate, maturity, spot, params).expect("fft slice");

    let center = params.n / 2;
    for i in 0..10 {
        let idx = center + i * 8;
        let (k, fft_call) = slice[idx];
        let bs = black_scholes_price(OptionType::Call, spot, k, rate, vol, maturity);
        assert!(
            (fft_call - bs).abs() < 1e-4,
            "BS mismatch K={k} fft={fft_call} bs={bs}"
        );
    }
}

#[test]
fn carr_madan_heston_matches_gauss_laguerre_within_1e3() {
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

    let heston_engine = HestonEngine::new(v0, kappa, theta, sigma_v, rho);
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .flat_vol(0.2)
        .build()
        .expect("valid market");

    let center = CarrMadanParams::default().n / 2;
    for i in 0..10 {
        let idx = center + i * 6;
        let (k, fft_call) = fft_prices[idx];
        let option = VanillaOption::european_call(k, maturity);
        let gl_call = heston_engine
            .price(&option, &market)
            .expect("heston gl price")
            .price;

        assert!(
            (fft_call - gl_call).abs() < 0.01,
            "Heston mismatch K={k} fft={fft_call} gauss={gl_call}"
        );
    }
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
fn fft_delta_matches_finite_difference_within_1e3() {
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

    let ds = 1e-3 * spot;
    let p_up = black_scholes_price(OptionType::Call, spot + ds, strike, rate, vol, maturity);
    let p_dn = black_scholes_price(OptionType::Call, spot - ds, strike, rate, vol, maturity);
    let delta_fd = (p_up - p_dn) / (2.0 * ds);

    assert!(
        (delta_fft - delta_fd).abs() < 1e-3,
        "delta mismatch fft={delta_fft} fd={delta_fd}"
    );
}

#[test]
fn fractional_fft_exact_strikes_match_standard_fft_interpolation_within_1e4() {
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

    for ((k_frft, p_frft), (_, p_fft)) in frft_exact.iter().zip(fft_interp.iter()) {
        assert!(
            (p_frft - p_fft).abs() < 1e-4,
            "FRFT/FFT mismatch K={k_frft} frft={p_frft} fft_interp={p_fft}"
        );
    }
}

#[test]
fn fft_strike_helper_matches_exact_strike_api_for_bs() {
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

    for ((_, p1), (_, p2)) in via_interp.iter().zip(via_exact.iter()) {
        assert!((p1 - p2).abs() < 1e-2);
    }
}
