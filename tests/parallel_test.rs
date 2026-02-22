use openferric::math::fast_norm::{beasley_springer_moro_inv_cdf, fast_norm_cdf};

#[cfg(feature = "parallel")]
use openferric::engines::monte_carlo::{
    mc_european_parallel, mc_european_sequential, mc_greeks_grid_parallel,
    mc_greeks_grid_sequential,
};
#[cfg(feature = "parallel")]
use openferric::instruments::VanillaOption;
#[cfg(feature = "parallel")]
use openferric::{core::OptionType, market::Market};

/// NIST reference values for the standard normal CDF.
const CDF_REFERENCE: &[(f64, f64)] = &[
    (-3.0, 0.0013498980316300946),
    (-2.0, 0.02275013194817921),
    (-1.0, 0.15865525393145702),
    (-0.5, 0.308_537_538_725_986_9),
    (0.0, 0.5),
    (0.5, 0.691_462_461_274_013_1),
    (1.0, 0.841_344_746_068_542_9),
    (2.0, 0.977_249_868_051_820_8),
    (3.0, 0.99865010196837),
];

#[test]
fn fast_norm_cdf_matches_nist_within_one_e_minus_seven() {
    for &(x, expected) in CDF_REFERENCE {
        let err = (fast_norm_cdf(x) - expected).abs();
        assert!(
            err <= 1.0e-7,
            "x={x} expected={expected} got={} err={err}",
            fast_norm_cdf(x)
        );
    }
}

#[test]
fn bsm_inverse_round_trips_cdf_within_one_e_minus_six() {
    for i in 1..=999 {
        let p = i as f64 / 1000.0;
        let x = beasley_springer_moro_inv_cdf(p);
        let p_back = fast_norm_cdf(x);
        let err = (p_back - p).abs();
        assert!(err <= 1.0e-6, "p={p} x={x} p_back={p_back} err={err}");
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
