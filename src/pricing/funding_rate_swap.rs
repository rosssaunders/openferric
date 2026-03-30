//! Module `pricing::funding_rate_swap`.
//!
//! Pricing helpers and bump sensitivities for discrete funding-rate swaps.

use chrono::{DateTime, Duration, Utc};

use crate::instruments::FundingRateSwap;
use crate::rates::FundingRateCurve;

/// Standard 1bp rate bump.
pub const FUNDING_RATE_BUMP_BP: f64 = 1.0e-4;
/// Standard 1 vol-point bump.
pub const FUNDING_RATE_VOL_BUMP: f64 = 1.0e-2;

/// Risk outputs for a funding-rate swap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundingRateSwapRisks {
    pub mtm: f64,
    pub dv01: f64,
    pub vega: f64,
    pub theta: f64,
}

/// Mark-to-market of the swap at the given valuation time.
pub fn funding_rate_swap_mtm(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
) -> f64 {
    swap.mark_to_market(curve, as_of)
}

/// PV change from a 1bp parallel shift in the funding curve.
pub fn funding_rate_swap_dv01(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
) -> f64 {
    let bumped_curve = curve.parallel_shifted(FUNDING_RATE_BUMP_BP);
    swap.mark_to_market(&bumped_curve, as_of) - swap.mark_to_market(curve, as_of)
}

/// PV change from a one vol-point shift in funding-rate volatility.
pub fn funding_rate_swap_vega(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
) -> f64 {
    let bumped_curve = curve.volatility_shifted(FUNDING_RATE_VOL_BUMP);
    swap.mark_to_market(&bumped_curve, as_of) - swap.mark_to_market(curve, as_of)
}

/// Time decay from advancing one funding settlement interval.
pub fn funding_rate_swap_theta(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
) -> f64 {
    let next_as_of = swap
        .settlement_schedule()
        .into_iter()
        .find(|settlement| *settlement > as_of)
        .unwrap_or(as_of + Duration::hours(i64::from(swap.settlement_interval_hours)));
    swap.mark_to_market(curve, next_as_of) - swap.mark_to_market(curve, as_of)
}

/// Aggregates MTM and the standard discrete funding sensitivities.
pub fn funding_rate_swap_risks(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of: DateTime<Utc>,
) -> FundingRateSwapRisks {
    FundingRateSwapRisks {
        mtm: funding_rate_swap_mtm(swap, curve, as_of),
        dv01: funding_rate_swap_dv01(swap, curve, as_of),
        vega: funding_rate_swap_vega(swap, curve, as_of),
        theta: funding_rate_swap_theta(swap, curve, as_of),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::{DateTime, NaiveDate, Utc};

    use super::{FUNDING_RATE_BUMP_BP, funding_rate_swap_dv01};
    use crate::instruments::FundingRateSwap;
    use crate::rates::FundingRateCurve;

    fn dt(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
        DateTime::from_naive_utc_and_offset(
            NaiveDate::from_ymd_opt(year, month, day)
                .expect("valid date")
                .and_hms_opt(hour, 0, 0)
                .expect("valid hour"),
            Utc,
        )
    }

    #[test]
    fn dv01_matches_flat_curve_finite_difference() {
        let swap = FundingRateSwap {
            notional: 5_000.0,
            fixed_rate: 0.04,
            entry_time: dt(2026, 1, 1, 0),
            maturity: dt(2026, 1, 2, 0),
            settlement_interval_hours: 8,
            venue: "OKX".to_string(),
            asset: "ETHUSDT".to_string(),
        };
        let curve = FundingRateCurve::flat(0.05);
        let as_of = dt(2026, 1, 1, 0);

        let remaining_intervals = 3.0;
        let expected = remaining_intervals
            * swap.notional
            * swap.interval_year_fraction()
            * FUNDING_RATE_BUMP_BP;

        assert_relative_eq!(
            funding_rate_swap_dv01(&swap, &curve, as_of),
            expected,
            epsilon = 1.0e-12
        );
    }
}
