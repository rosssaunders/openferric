//! Independent exact-arithmetic references for perpetual funding-rate swaps.
//!
//! The cached values in this file were generated offline with Python 3.11.15
//! `decimal` at precision 80.  The generator enumerated UTC settlement times,
//! represented every stated input as an exact decimal, evaluated each cashflow
//! as `(floating_apr - fixed_apr) * notional * hours / 8760`, and applied the
//! stated discount factor at each settlement.  Funding- and discount-DV01
//! references independently repriced the Decimal cashflow sum at the same 1 bp
//! bump as the public API.  No OpenFerric code or output entered the generator.
//! The Linux Rust-vs-Decimal session measured a maximum 6.67e-14 absolute gap
//! for non-cancelling quantities and 1.55e-15 for discount DV01.

use chrono::{DateTime, Duration, NaiveDate, Utc};

use openferric::instruments::FundingRateSwap;
use openferric::pricing::funding_rate_swap::{
    funding_rate_swap_discount_dv01, funding_rate_swap_dv01,
};
use openferric::rates::{FundingRateCurve, FundingRateSnapshot, YieldCurve};

const INTERVAL_HOURS: i64 = 8;
const INTERVAL_YEARS: f64 = 8.0 / 8_760.0;
const RELATIVE_ROUNDOFF_BUDGET: f64 = 2.0e-12;

fn dt(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
    DateTime::from_naive_utc_and_offset(
        NaiveDate::from_ymd_opt(year, month, day)
            .expect("valid date")
            .and_hms_opt(hour, 0, 0)
            .expect("valid hour"),
        Utc,
    )
}

fn assert_decimal_reference(label: &str, actual: f64, reference: f64) {
    let roundoff_budget = RELATIVE_ROUNDOFF_BUDGET * reference.abs();
    assert!(
        (actual - reference).abs() <= roundoff_budget,
        "{label}: actual={actual:.17e}, independent Decimal reference={reference:.17e}, \
         roundoff budget={roundoff_budget:.3e}"
    );
}

fn reference_fixture() -> (FundingRateSwap, FundingRateCurve, YieldCurve) {
    let entry_time = dt(2026, 1, 1, 0);
    let swap = FundingRateSwap {
        notional: 3_750_000.0,
        fixed_rate: 0.0375,
        entry_time,
        maturity: dt(2026, 1, 2, 8),
        settlement_interval_hours: INTERVAL_HOURS as u32,
        venue: "reference".to_string(),
        asset: "BTCUSD".to_string(),
    };

    // Exact Decimal APR inputs {0.0612, -0.0045, 0.0827, 0.0291}, divided by
    // the documented 1095 funding periods per year before conversion to f64.
    let per_period_rates = [
        5.589_041_095_890_411e-5,
        -4.109_589_041_095_89e-6,
        7.552_511_415_525_114e-5,
        2.657_534_246_575_342_4e-5,
    ];
    let snapshots = per_period_rates
        .into_iter()
        .enumerate()
        .map(|(i, rate)| FundingRateSnapshot {
            venue: "reference".to_string(),
            asset: "BTCUSD".to_string(),
            rate,
            timestamp: entry_time + Duration::hours(i as i64 * INTERVAL_HOURS),
        })
        .collect();
    let funding_curve = FundingRateCurve::new(snapshots);

    // Pillars coincide exactly with the four settlement year fractions.  This
    // locks the supplied non-flat DFs without introducing an interpolation
    // convention into the NPV reference.
    let discount_curve = YieldCurve::new(vec![
        (INTERVAL_YEARS, 0.99993),
        (2.0 * INTERVAL_YEARS, 0.99971),
        (3.0 * INTERVAL_YEARS, 0.99932),
        (4.0 * INTERVAL_YEARS, 0.99876),
    ]);

    (swap, funding_curve, discount_curve)
}

#[test]
fn funding_rate_curve_matches_independent_decimal_linear_integral() {
    let (_, funding_curve, _) = reference_fixture();

    // At 1.5 intervals the piecewise-linear forward is the midpoint of the
    // second and third APR nodes, converted to a per-8-hour rate.  At four
    // intervals the cumulative index is three trapezoids plus one final flat
    // interval, matching FundingRateCurve's documented extrapolation.
    const DECIMAL_MIDPOINT_FORWARD: f64 = 3.570_776_255_707_762_5e-5;
    const DECIMAL_FOUR_INTERVAL_INDEX: f64 = 1.392_237_442_922_374_4e-4;
    let midpoint_forward = funding_curve.forward_rate(1.5 * INTERVAL_YEARS);
    let cumulative_index = funding_curve.cumulative_index(4.0 * INTERVAL_YEARS);

    assert_decimal_reference(
        "piecewise-linear funding forward",
        midpoint_forward,
        DECIMAL_MIDPOINT_FORWARD,
    );
    assert_decimal_reference(
        "piecewise-linear cumulative funding index",
        cumulative_index,
        DECIMAL_FOUR_INTERVAL_INDEX,
    );
}

#[test]
fn funding_rate_swap_realized_pnl_matches_independent_decimal_sum() {
    let (swap, _, _) = reference_fixture();
    let fixings = [
        (dt(2026, 1, 1, 4), 0.90), // not a scheduled settlement
        (dt(2026, 1, 1, 8), 0.0550),
        (dt(2026, 1, 1, 16), 0.0210),
        (dt(2026, 1, 2, 0), -0.0070),
        (dt(2026, 1, 2, 8), 0.0640),
    ];

    // Decimal sum over the four scheduled fixings only.
    const DECIMAL_REALIZED_PNL: f64 = -58.219_178_082_191_78;
    assert_decimal_reference(
        "realized funding PnL",
        swap.realized_pnl(&fixings),
        DECIMAL_REALIZED_PNL,
    );
}

#[test]
fn funding_rate_swap_mtm_matches_independent_decimal_discounted_cashflows() {
    let (swap, funding_curve, discount_curve) = reference_fixture();

    // Decimal explicitly sampled APRs {0.0612, -0.0045, 0.0827, 0.0291}
    // at the four interval starts and applied settlement DFs
    // {0.99993, 0.99971, 0.99932, 0.99876} cashflow by cashflow.
    const DECIMAL_DISCOUNTED_MTM: f64 = 63.322_606_164_383_565;
    let mtm = swap.mark_to_market(&funding_curve, Some(&discount_curve), swap.entry_time);
    assert_decimal_reference("discounted funding-swap MTM", mtm, DECIMAL_DISCOUNTED_MTM);
}

#[test]
fn funding_rate_swap_dv01_matches_independent_decimal_central_difference() {
    let (swap, funding_curve, discount_curve) = reference_fixture();

    // The Decimal generator evaluated (PV(f + 1bp) - PV(f - 1bp)) / 2.
    // Because every funding cashflow is affine in its forward APR, this central
    // cash change is exactly the API's one-sided PV(f + 1bp) - PV(f).
    const DECIMAL_FUNDING_DV01: f64 = 1.369_082_191_780_822;
    let dv01 = funding_rate_swap_dv01(
        &swap,
        &funding_curve,
        Some(&discount_curve),
        swap.entry_time,
    );
    assert_decimal_reference("funding-curve DV01", dv01, DECIMAL_FUNDING_DV01);
}

#[test]
fn funding_rate_swap_discount_dv01_matches_independent_decimal_central_decomposition() {
    let (swap, funding_curve, discount_curve) = reference_fixture();

    // The API defines the discrete one-sided change PV(z + 1bp) - PV(z).
    // The independent Decimal calculation decomposed that exact change into
    // its central odd and even-curvature parts at the same bump:
    //
    //   [PV(+h) - PV(-h)] / 2 = -1.30331384875212719422e-5
    //   [PV(+h) + PV(-h) - 2 PV(0)] / 2 = 1.82849896876829718e-12.
    //
    // Their sum is the one-sided API quantity below; directly equating a
    // central derivative with a discrete one-sided PV change would omit the
    // stated curvature term.
    const DECIMAL_DISCOUNT_DV01: f64 = -1.303_313_665_902_230_3e-5;
    const DECIMAL_BASE_PV: f64 = 63.322_606_164_383_565;
    let dv01 = funding_rate_swap_discount_dv01(
        &swap,
        &funding_curve,
        Some(&discount_curve),
        swap.entry_time,
    );
    // The API subtracts two O(63) PVs to obtain an O(1e-5) change.  Budget
    // accumulated cancellation at 128 binary64 epsilons of the two-operand
    // scale, rather than applying a misleading relative tolerance to DV01.
    let cancellation_scale = 2.0 * DECIMAL_BASE_PV.abs();
    let roundoff_budget = 128.0 * f64::EPSILON * cancellation_scale;
    assert!(
        (dv01 - DECIMAL_DISCOUNT_DV01).abs() <= roundoff_budget,
        "discount-curve DV01: actual={dv01:.17e}, independent Decimal \
         reference={DECIMAL_DISCOUNT_DV01:.17e}, cancellation \
         roundoff budget={roundoff_budget:.3e}"
    );
}
