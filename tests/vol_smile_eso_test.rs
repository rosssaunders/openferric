use openferric::core::{OptionType, PricingEngine};
use openferric::engines::tree::BinomialTreeEngine;
use openferric::instruments::{EmployeeStockOption, VanillaOption};
use openferric::market::Market;
use openferric::math::normal_pdf;
use openferric::pricing::european::black_scholes_price;
use openferric::vol::builder::{MarketOptionQuote, VolSurfaceBuilder};
use openferric::vol::mixture::{LognormalMixture, calibrate_lognormal_mixture};
use openferric::vol::smile::{
    StickyStrikeSmile, VannaVolgaQuote, vanna_volga_pivot_strikes, vanna_volga_price,
};

#[test]
fn sticky_strike_smile_is_monotone_in_equity_skew_wings() {
    let spot = 100.0;
    let rate = 0.01;
    let expiry = 1.0;

    let strikes = vec![70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0];
    let vols = [0.34, 0.30, 0.26, 0.22, 0.20, 0.19, 0.18];

    let quotes: Vec<MarketOptionQuote> = strikes
        .iter()
        .zip(vols.iter())
        .map(|(&k, &v)| {
            let price = black_scholes_price(OptionType::Call, spot, k, rate, v, expiry);
            MarketOptionQuote::new(k, expiry, price, OptionType::Call)
        })
        .collect();

    let surface = VolSurfaceBuilder::from_quotes(spot, rate, quotes)
        .build()
        .expect("surface build succeeds");

    let smile = StickyStrikeSmile::from_built_surface(&surface, expiry, strikes.clone())
        .expect("sticky-strike build succeeds");

    for (&strike, &quoted_vol) in strikes.iter().zip(&vols) {
        let recovered = smile.vol(strike);
        assert!(
            (recovered - quoted_vol).abs() <= 2.0e-9,
            "strike={strike}: recovered_vol={recovered}, quoted_vol={quoted_vol}"
        );
    }

    let left_wing = [70.0, 80.0, 90.0, 100.0];
    for pair in left_wing.windows(2) {
        let v0 = smile.vol(pair[0]);
        let v1 = smile.vol(pair[1]);
        assert!(v0 >= v1, "left wing should decrease as strike increases");
    }

    let right_wing = [100.0, 110.0, 120.0, 130.0];
    for pair in right_wing.windows(2) {
        let v0 = smile.vol(pair[0]);
        let v1 = smile.vol(pair[1]);
        assert!(v0 >= v1, "right wing should decrease as strike increases");
    }
}

#[test]
fn vanna_volga_matches_atm_mid_with_zero_adjustment_weight() {
    // At the ATM pivot strike the pivot market vol equals the base vol, so the
    // Castagna-Mercurio correction terms vanish and VV == BS(atm vol) exactly.
    let quote = VannaVolgaQuote::new(0.21, 0.03, 0.01);

    let (_, k_atm, _) = vanna_volga_pivot_strikes(100.0, 0.02, 0.0, 1.0, quote);
    let vv = vanna_volga_price(OptionType::Call, 100.0, k_atm, 0.02, 0.0, 1.0, quote);
    let mid = black_scholes_price(OptionType::Call, 100.0, k_atm, 0.02, 0.21, 1.0);

    assert!((vv - mid).abs() < 1e-12);
}

#[test]
fn two_component_mixture_reprices_five_strike_smile() {
    let spot = 100.0;
    let rate = 0.01;
    let expiry = 1.0;
    let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];

    let true_mix = LognormalMixture::new(vec![0.6, 0.4], vec![0.15, 0.33]).unwrap();
    let market_prices: Vec<f64> = strikes
        .iter()
        .map(|&k| true_mix.price(OptionType::Call, spot, k, rate, expiry))
        .collect();

    let fitted = calibrate_lognormal_mixture(
        OptionType::Call,
        spot,
        rate,
        expiry,
        &strikes,
        &market_prices,
        2,
    )
    .expect("mixture calibration succeeds");

    for (k, market_price) in strikes.iter().zip(market_prices.iter()) {
        let fit_price = fitted.price(OptionType::Call, spot, *k, rate, expiry);
        assert!(
            (fit_price - market_price).abs() <= 1.0e-9,
            "strike {k}: target price {market_price}, fit price {fit_price}, weights={:?}, vols={:?}",
            fitted.weights,
            fitted.vols
        );
    }
}

#[test]
fn implied_density_from_mixture_matches_analytic_density() {
    let mix = LognormalMixture::new(vec![0.7, 0.3], vec![0.16, 0.31]).unwrap();
    let spot: f64 = 100.0;
    let rate: f64 = 0.01;
    let expiry: f64 = 1.0;

    for k in (60..=140).step_by(5) {
        let strike = k as f64;
        let exact: f64 = mix
            .weights
            .iter()
            .zip(&mix.vols)
            .map(|(&weight, &sigma)| {
                let d2 = ((spot / strike).ln() + (rate - 0.5 * sigma * sigma) * expiry)
                    / (sigma * expiry.sqrt());
                weight * normal_pdf(d2) / (strike * sigma * expiry.sqrt())
            })
            .sum();
        let coarse = mix.implied_density(spot, strike, rate, expiry, 0.5);
        let density = mix.implied_density(spot, strike, rate, expiry, 0.25);
        let richardson_error = (coarse - density).abs() / 3.0;
        assert!(
            (density - exact).abs() <= 1.05 * richardson_error + 2.0e-12,
            "density at strike {k}: finite_difference={density}, analytic={exact}, coarse={coarse}, Richardson error={richardson_error}"
        );
    }
}

#[test]
fn eso_with_forfeiture_and_forced_exercise_matches_independent_crr_reference() {
    let eso = EmployeeStockOption::new(
        OptionType::Call,
        100.0,
        5.0,
        1.0,
        3.0,
        Some(1.7),
        0.08,
        1_000_000.0,
        0.0,
    );

    let eso_value = eso.price_binomial(100.0, 0.03, 0.0, 0.30, 900).unwrap();
    // Independently generated with a Python CRR backward recurrence using the
    // same 900-step exercise dates, vesting date, forced-exercise boundary,
    // and survival scaling.  A fixed-grid oracle avoids hiding the sizeable
    // boundary-alignment oscillation behind a comparison range.
    assert!((eso_value - 18.113_605_235_923_11).abs() <= 2.0e-10);
}

#[test]
fn eso_without_forfeiture_or_boundary_matches_american_value() {
    let steps = 1200;
    let eso = EmployeeStockOption::new(
        OptionType::Put,
        100.0,
        1.0,
        0.0,
        1.0,
        None,
        0.0,
        1_000_000.0,
        0.0,
    );

    let eso_value = eso.price_binomial(100.0, 0.05, 0.0, 0.25, steps).unwrap();

    let option = VanillaOption::american_put(100.0, 1.0);
    let market = Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.25)
        .build()
        .unwrap();

    let american = BinomialTreeEngine::new(steps)
        .price(&option, &market)
        .unwrap()
        .price;

    assert!((eso_value - american).abs() < 1e-8);
}

#[test]
fn dilution_adjustment_reduces_eso_value_proportionally() {
    let undiluted = EmployeeStockOption::new(
        OptionType::Call,
        100.0,
        3.0,
        0.5,
        2.5,
        None,
        0.0,
        100.0,
        0.0,
    );

    let diluted = EmployeeStockOption::new(
        OptionType::Call,
        100.0,
        3.0,
        0.5,
        2.5,
        None,
        0.0,
        100.0,
        25.0,
    );

    let v_undiluted = undiluted
        .price_binomial(100.0, 0.03, 0.0, 0.28, 900)
        .unwrap();
    let v_diluted = diluted.price_binomial(100.0, 0.03, 0.0, 0.28, 900).unwrap();

    let expected = v_undiluted * (100.0 / 125.0);
    assert!((v_diluted - expected).abs() < 1e-8);
}
