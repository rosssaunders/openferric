use openferric::math::fast_norm::{beasley_springer_moro_inv_cdf, fast_norm_cdf};
use statrs::distribution::{ContinuousCDF, Normal};

#[cfg(feature = "parallel")]
use openferric::engines::monte_carlo::{
    mc_european_parallel, mc_european_sequential, mc_greeks_grid_parallel,
    mc_greeks_grid_sequential,
};
#[cfg(feature = "parallel")]
use openferric::instruments::VanillaOption;
#[cfg(feature = "parallel")]
use openferric::{core::OptionType, market::Market};

#[test]
fn fast_norm_cdf_matches_statrs_within_one_e_minus_seven() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    for i in -800..=800 {
        let x = i as f64 / 100.0;
        let err = (fast_norm_cdf(x) - normal.cdf(x)).abs();
        assert!(err <= 1.0e-7, "x={x} err={err}");
    }
}

#[test]
fn bsm_inverse_matches_statrs_within_one_e_minus_six() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    for i in 1..=999 {
        let p = i as f64 / 1000.0;
        let err = (beasley_springer_moro_inv_cdf(p) - normal.inverse_cdf(p)).abs();
        assert!(err <= 1.0e-6, "p={p} err={err}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_mc_price_matches_sequential_within_two_percent() {
    let option = VanillaOption::european_call(100.0, 1.0);
    let market = Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.2)
        .build()
        .unwrap();

    let seq = mc_european_sequential(&option, &market, 100_000, 252);
    let par = mc_european_parallel(&option, &market, 100_000, 252);

    let rel_err = (par.price - seq.price).abs() / seq.price.abs().max(1.0e-12);
    assert!(
        rel_err <= 0.02,
        "parallel/sequential mismatch: seq={} par={} rel_err={}",
        seq.price,
        par.price,
        rel_err
    );
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_greeks_grid_matches_sequential_exactly() {
    let spots = (0..50).map(|i| 75.0 + i as f64).collect::<Vec<_>>();
    let vols = (0..50).map(|i| 0.08 + 0.004 * i as f64).collect::<Vec<_>>();

    let seq = mc_greeks_grid_sequential(OptionType::Call, 100.0, 0.03, 0.0, 1.0, &spots, &vols);
    let par = mc_greeks_grid_parallel(OptionType::Call, 100.0, 0.03, 0.0, 1.0, &spots, &vols);

    assert_eq!(seq, par);
}
