use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{
    CapFloor, DayCountConvention, Frequency, Future, InterestRateFutureQuote, Swaption, YieldCurve,
};

fn flat_curve_continuous(rate: f64, max_tenor_years: u32) -> YieldCurve {
    let tenors = (1..=max_tenor_years)
        .map(|t| {
            let tf = t as f64;
            (tf, (-rate * tf).exp())
        })
        .collect();
    YieldCurve::new(tenors)
}

#[test]
fn futures_cost_of_carry_matches_manual_calc() {
    let fut = Future {
        underlying_spot: 100.0,
        risk_free_rate: 0.05,
        dividend_yield: 0.01,
        storage_cost: 0.02,
        convenience_yield: 0.005,
        expiry: 1.5,
    };

    let expected = 100.0 * ((0.05_f64 - 0.01 + 0.02 - 0.005) * 1.5).exp();
    assert_relative_eq!(fut.theoretical_price(), expected, epsilon = 1.0e-12);
    assert_relative_eq!(fut.implied_repo_rate(expected), 0.05, epsilon = 1.0e-12);

    let quote = InterestRateFutureQuote::price_from_rate(0.0525);
    assert_relative_eq!(quote, 94.75, epsilon = 1.0e-12);
    assert_relative_eq!(InterestRateFutureQuote::rate_from_price(quote), 0.0525, epsilon = 1.0e-12);

    let convexity = InterestRateFutureQuote::convexity_adjustment(0.01, 1.0, 1.25);
    let fwd = InterestRateFutureQuote::forward_rate_from_futures_rate(0.05, 0.01, 1.0, 1.25);
    assert!(convexity > 0.0);
    assert_relative_eq!(fwd, 0.05 - convexity, epsilon = 1.0e-12);
}

#[test]
fn caplet_black_formula_matches_known_value() {
    let price = CapFloor::black_caplet(1_000_000.0, 0.95, 0.5, 0.04, 0.035, 0.2, 1.0);
    assert_relative_eq!(price, 2_909.787_480_907_682_5, epsilon = 1.0e-3);
}

#[test]
fn cap_floor_parity_matches_swap_npv() {
    let curve = flat_curve_continuous(0.04, 10);
    let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2028, 1, 1).unwrap();

    let cap = CapFloor {
        notional: 1_000_000.0,
        strike: 0.045,
        start_date: start,
        end_date: end,
        frequency: Frequency::SemiAnnual,
        day_count: DayCountConvention::Act365Fixed,
        is_cap: true,
    };

    let floor = CapFloor {
        is_cap: false,
        ..cap.clone()
    };

    let vol = 0.25;
    let lhs = cap.price(&curve, vol) - floor.price(&curve, vol);
    let rhs = cap.swap_npv(&curve);
    assert_relative_eq!(lhs, rhs, epsilon = 1.0e-8);
}

#[test]
fn swaption_atm_payer_positive_and_put_call_parity_holds() {
    let curve = flat_curve_continuous(0.03, 30);

    let template = Swaption {
        notional: 1_000_000.0,
        strike: 0.03,
        option_expiry: 2.0,
        swap_tenor: 5.0,
        is_payer: true,
    };

    let atm_strike = template.forward_swap_rate(&curve);
    let payer = Swaption {
        strike: atm_strike,
        is_payer: true,
        ..template
    };
    let receiver = Swaption {
        strike: atm_strike,
        is_payer: false,
        ..template
    };

    let vol = 0.20;
    let payer_price = payer.price(&curve, vol);
    let receiver_price = receiver.price(&curve, vol);

    assert!(payer_price > 0.0);

    let annuity = payer.annuity_factor(&curve);
    let forward = payer.forward_swap_rate(&curve);
    let parity_rhs = payer.notional * annuity * (forward - payer.strike);
    assert_relative_eq!(payer_price - receiver_price, parity_rhs, epsilon = 1.0e-8);
}
