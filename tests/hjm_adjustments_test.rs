use approx::assert_relative_eq;

use openferric::instruments::{AbandonmentOption, DeferInvestmentOption, RealOptionBinomialSpec};
use openferric::models::{HjmModel, HullWhite};
use openferric::pricing::real_option::{
    european_abandonment_put, price_option_to_abandon, price_option_to_defer,
};
use openferric::rates::{
    YieldCurve, cms_rate_in_arrears, futures_forward_convexity_adjustment, timing_adjusted_rate,
    timing_adjustment_amount,
};

fn flat_discount_curve(rate: f64, max_t: f64, dt: f64) -> YieldCurve {
    let points = (1..=((max_t / dt).round() as usize))
        .map(|i| {
            let t = i as f64 * dt;
            (t, (-rate * t).exp())
        })
        .collect::<Vec<_>>();
    YieldCurve::new(points)
}

fn flat_forward_curve(rate: f64, max_t: f64, dt: f64) -> (Vec<f64>, Vec<f64>) {
    let maturities = (0..=((max_t / dt).round() as usize))
        .map(|i| i as f64 * dt)
        .collect::<Vec<_>>();
    let forwards = vec![rate; maturities.len()];
    (maturities, forwards)
}

#[test]
fn single_factor_hjm_recovers_hull_white_bond_prices_within_one_percent() {
    let rate = 0.03;
    let maturity = 7.0;
    let _hjm = HjmModel::single_factor_exponential(0.01, 0.20);
    let (maturities, forwards) = flat_forward_curve(rate, 30.0, 0.25);

    let hjm_bond = HjmModel::zero_coupon_bond_price(0.0, maturity, &maturities, &forwards).unwrap();

    let hw = HullWhite::new(0.20, 0.01);
    let curve = flat_discount_curve(rate, 30.0, 0.25);
    let hw_bond = hw.bond_price(0.0, maturity, rate, &curve);

    let rel_err = (hjm_bond - hw_bond).abs() / hw_bond.abs().max(1.0e-12);
    assert!(rel_err <= 0.01);
}

#[test]
fn hjm_drift_condition_is_satisfied() {
    let model = HjmModel::single_factor_exponential(0.015, 0.25);
    let t = 1.0;
    let maturity = 4.5;

    let lhs = model.drift(t, maturity);
    let sigma = model.factor_volatility(0, t, maturity);
    let integrated = model.integrated_factor_volatility(0, t, maturity);
    let rhs = sigma * integrated;

    assert_relative_eq!(lhs, rhs, epsilon = 1.0e-12);
}

#[test]
fn convexity_adjustment_is_positive_and_increases_with_maturity() {
    let short = futures_forward_convexity_adjustment(0.012, 0.5, 0.75);
    let long = futures_forward_convexity_adjustment(0.012, 2.0, 2.25);

    assert!(short > 0.0);
    assert!(long > short);
}

#[test]
fn cms_rate_is_above_swap_rate_from_convexity_bias() {
    let swap_rate = 0.03;
    let cms = cms_rate_in_arrears(swap_rate, 0.25, 3.0);
    assert!(cms > swap_rate);
}

#[test]
fn timing_adjustment_vanishes_when_payment_equals_natural_date() {
    let rate = 0.0275;
    let adjusted = timing_adjusted_rate(rate, 0.2, 1.5, 1.5);
    assert_relative_eq!(
        timing_adjustment_amount(0.2, 1.5, 1.5),
        0.0,
        epsilon = 1.0e-16
    );
    assert_relative_eq!(adjusted, rate, epsilon = 1.0e-16);
}

#[test]
fn real_option_to_defer_is_at_least_intrinsic_value() {
    let option = DeferInvestmentOption {
        model: RealOptionBinomialSpec {
            project_value: 120.0,
            volatility: 0.30,
            risk_free_rate: 0.05,
            maturity: 2.0,
            steps: 200,
            cash_flows: vec![],
        },
        investment_cost: 100.0,
    };

    let value = price_option_to_defer(&option).unwrap().price;
    assert!(value >= (option.model.project_value - option.investment_cost).max(0.0));
}

#[test]
fn option_to_abandon_is_above_european_put_equivalent() {
    let option = AbandonmentOption {
        model: RealOptionBinomialSpec {
            project_value: 90.0,
            volatility: 0.35,
            risk_free_rate: 0.04,
            maturity: 2.0,
            steps: 200,
            cash_flows: vec![],
        },
        salvage_value: 100.0,
    };

    let american = price_option_to_abandon(&option).unwrap().price;
    let european = european_abandonment_put(&option).unwrap();

    assert!(american >= european - 1.0e-10);
}
