use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{
    CapFloor, DayCountConvention, Frequency, Future, InterestRateFutureQuote, Swaption, YieldCurve,
};
use statrs::distribution::{ContinuousCDF, Normal};

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
    assert_relative_eq!(
        InterestRateFutureQuote::rate_from_price(quote),
        0.0525,
        epsilon = 1.0e-12
    );

    let convexity = InterestRateFutureQuote::convexity_adjustment(0.01, 1.0, 1.25);
    let fwd = InterestRateFutureQuote::forward_rate_from_futures_rate(0.05, 0.01, 1.0, 1.25);
    assert_relative_eq!(convexity, 0.000_062_5, epsilon = 1.0e-15);
    assert_relative_eq!(fwd, 0.05 - convexity, epsilon = 1.0e-12);
}

#[test]
fn caplet_black_formula_matches_known_value() {
    let price = CapFloor::black_caplet(1_000_000.0, 0.95, 0.5, 0.04, 0.035, 0.2, 1.0);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = ((0.04_f64 / 0.035).ln() + 0.5 * 0.2_f64.powi(2)) / 0.2;
    let d2 = d1 - 0.2;
    let expected = 1_000_000.0 * 0.95 * 0.5 * (0.04 * normal.cdf(d1) - 0.035 * normal.cdf(d2));
    // OpenFerric's fast normal CDF approximation differs from statrs by about
    // 4e-7 currency units for this case.
    assert_relative_eq!(price, expected, epsilon = 5.0e-7);
    assert_relative_eq!(price, 2_909.787_480_907_682_5, epsilon = 1.0e-9);
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
    let cap_price = cap.price(&curve, vol);
    let floor_price = floor.price(&curve, vol);

    // Independent manual semiannual strip with statrs' normal CDF. Listing the
    // dates explicitly also checks the schedule used by the aggregate pricer.
    let dates = [
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2025, 7, 1).unwrap(),
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
        NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2027, 7, 1).unwrap(),
        // Jan 1, 2028 is Saturday, so the default Modified-Following schedule
        // ends on Monday Jan 3.
        NaiveDate::from_ymd_opt(2028, 1, 3).unwrap(),
    ];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (expected_cap, expected_floor) = dates.windows(2).fold((0.0, 0.0), |acc, period| {
        let t1 = (period[0] - start).num_days() as f64 / 365.0;
        let t2 = (period[1] - start).num_days() as f64 / 365.0;
        let accrual = (period[1] - period[0]).num_days() as f64 / 365.0;
        let df1 = (-0.04 * t1).exp();
        let df2 = (-0.04 * t2).exp();
        let forward = (df1 / df2 - 1.0) / accrual;
        let scale = cap.notional * df2 * accrual;
        if t1 == 0.0 {
            (
                acc.0 + scale * (forward - cap.strike).max(0.0),
                acc.1 + scale * (cap.strike - forward).max(0.0),
            )
        } else {
            let sigma_sqrt_t = vol * t1.sqrt();
            let d1 = ((forward / cap.strike).ln() + 0.5 * vol * vol * t1) / sigma_sqrt_t;
            let d2 = d1 - sigma_sqrt_t;
            (
                acc.0 + scale * (forward * normal.cdf(d1) - cap.strike * normal.cdf(d2)),
                acc.1 + scale * (cap.strike * normal.cdf(-d2) - forward * normal.cdf(-d1)),
            )
        }
    });
    assert_relative_eq!(cap_price, expected_cap, epsilon = 1.0e-8);
    assert_relative_eq!(floor_price, expected_floor, epsilon = 1.0e-8);

    let lhs = cap_price - floor_price;
    let rhs = cap.swap_npv(&curve);
    assert_relative_eq!(lhs, rhs, epsilon = 1.0e-8);

    assert_relative_eq!(cap.implied_vol(cap_price, &curve), vol, epsilon = 1.0e-8);
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
