use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{
    DayCountConvention, ForwardRateAgreement, Frequency, InterestRateSwap, SwapBuilder, YieldCurve,
    year_fraction,
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
fn swap_npv_near_zero_for_flat_five_percent() {
    let curve = flat_curve_continuous(0.05, 10);
    let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();

    let swap = SwapBuilder::default()
        .notional(100.0)
        .fixed_rate(0.05)
        .float_spread(0.0)
        .start_date(start)
        .end_date(end)
        .fixed_freq(Frequency::Annual)
        .float_freq(Frequency::Annual)
        .fixed_day_count(DayCountConvention::Thirty360)
        .float_day_count(DayCountConvention::Thirty360)
        .build();

    assert_relative_eq!(swap.npv(&curve), 0.0, epsilon = 1.0e-10);
}

#[test]
fn swap_par_rate_near_five_percent_on_flat_curve() {
    let curve = flat_curve_continuous(0.05, 10);
    let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();

    let swap = InterestRateSwap::builder()
        .notional(100.0)
        .start_date(start)
        .end_date(end)
        .fixed_freq(Frequency::Annual)
        .float_freq(Frequency::Annual)
        .fixed_day_count(DayCountConvention::Thirty360)
        .float_day_count(DayCountConvention::Thirty360)
        .build();

    assert_relative_eq!(swap.par_rate(&curve), 0.05, epsilon = 1.0e-10);
}

#[test]
fn swap_dv01_around_four_point_five_bps_per_100_notional() {
    let curve = flat_curve_continuous(0.05, 10);
    let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();

    let swap = InterestRateSwap::builder()
        .notional(100.0)
        .fixed_rate(0.05)
        .start_date(start)
        .end_date(end)
        .fixed_freq(Frequency::Annual)
        .float_freq(Frequency::Annual)
        .fixed_day_count(DayCountConvention::Thirty360)
        .float_day_count(DayCountConvention::Thirty360)
        .build();

    let dv01 = swap.dv01(&curve);
    let dv01_bps_per_100 = dv01 / swap.notional * 10_000.0;
    assert!(dv01_bps_per_100 > 4.0 && dv01_bps_per_100 < 5.0);
}

#[test]
fn fra_forward_rate_matches_curve_forward_rate() {
    let curve = flat_curve_continuous(0.05, 10);
    let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2025, 7, 1).unwrap();
    let day_count = DayCountConvention::Act365Fixed;

    let fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.05,
        start_date: start,
        end_date: end,
        day_count,
    };

    let tau = year_fraction(start, end, day_count);
    assert_relative_eq!(fra.forward_rate(&curve), curve.forward_rate(0.0, tau), epsilon = 1.0e-12);
}
