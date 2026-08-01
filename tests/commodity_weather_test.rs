use openferric::core::OptionType;
use openferric::instruments::{
    CatastropheBond, DegreeDayType, WeatherOption, cumulative_cdd, cumulative_hdd,
};
use openferric::models::{SchwartzOneFactor, implied_convenience_yield};

#[test]
fn schwartz_one_factor_simulation_is_mean_reverting() {
    let model = SchwartzOneFactor {
        kappa: 2.5,
        mu: 100.0_f64.ln(),
        sigma: 0.25,
    };

    let initial_spot = 220.0;
    let terminals = model
        .simulate_terminal_spots(initial_spot, 1.0, 252, 8_000, 42)
        .expect("simulation should succeed");

    let mean_terminal_log = terminals.iter().map(|s| s.ln()).sum::<f64>() / terminals.len() as f64;
    let long_run = model.long_run_log_mean();
    let initial_log = initial_spot.ln();

    assert!(
        (mean_terminal_log - long_run).abs() < (initial_log - long_run).abs(),
        "terminal mean log spot should move toward long-run level"
    );
}

#[test]
fn convenience_yield_sign_matches_contango_and_backwardation() {
    let spot = 100.0;
    let r = 0.03;
    let storage = 0.01;
    let t = 1.0;

    let y_contango = implied_convenience_yield(spot, 110.0, r, storage, t).unwrap();
    let y_backwardation = implied_convenience_yield(spot, 90.0, r, storage, t).unwrap();

    assert!(
        y_contango < 0.0,
        "contango should imply lower/negative convenience yield"
    );
    assert!(
        y_backwardation > r + storage,
        "backwardation should imply high convenience yield"
    );
}

#[test]
fn hdd_and_cdd_accumulate_from_temperature_series() {
    let temperatures = vec![50.0, 60.0, 70.0, 80.0];
    let base = 65.0;

    let hdd = cumulative_hdd(&temperatures, base);
    let cdd = cumulative_cdd(&temperatures, base);

    assert!((hdd - 20.0).abs() < 1e-12);
    assert!((cdd - 20.0).abs() < 1e-12);
}

#[test]
fn weather_option_burn_price_matches_discounted_historical_average() {
    let option = WeatherOption {
        index_type: DegreeDayType::HDD,
        option_type: OptionType::Call,
        strike: 100.0,
        tick_size: 1.0,
        notional: 1.0,
        discount_rate: 0.02,
        maturity: 0.5,
    };

    let historical_indices = vec![80.0, 95.0, 100.0, 120.0, 140.0];
    let price = option
        .price_burn_analysis(&historical_indices)
        .expect("burn analysis should price");

    // Historical payoffs are [0, 0, 0, 20, 40], so the undiscounted
    // arithmetic mean is exactly 12 and PV = 12 * exp(-0.02 * 0.5).
    let expected = 11.880_598_004_990_016;
    assert!((price - expected).abs() <= 2.0e-12);
}

#[test]
fn cat_bond_matches_exact_poisson_expected_loss_cashflows() {
    let low_intensity = CatastropheBond {
        principal: 100.0,
        coupon_rate: 0.08,
        risk_free_rate: 0.03,
        maturity: 3.0,
        coupon_frequency: 4,
        loss_intensity: 0.2,
        expected_loss_per_event: 0.6,
    }
    .price()
    .unwrap();

    let high_intensity = CatastropheBond {
        principal: 100.0,
        coupon_rate: 0.08,
        risk_free_rate: 0.03,
        maturity: 3.0,
        coupon_frequency: 4,
        loss_intensity: 1.0,
        expected_loss_per_event: 0.6,
    }
    .price()
    .unwrap();

    // Independent direct summation of the twelve quarterly expected coupons
    // and surviving principal under exp(-lambda * loss_fraction * t).
    assert!((low_intensity - 82.729_206_664_943_52).abs() <= 2.0e-12);
    assert!((high_intensity - 25.060_568_372_450_263).abs() <= 2.0e-12);
    assert!(high_intensity < low_intensity);
}
