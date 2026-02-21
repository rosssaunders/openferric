//! Day-count convention reference tests derived from QuantLib's daycounters.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/daycounters.cpp
//!
//! Test vectors extracted from QuantLib commit used by this project's vendored
//! submodule. The expected values were computed by QuantLib's own implementations
//! and are considered authoritative for the conventions tested here.

use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{DayCountConvention, year_fraction};

fn d(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ── ACT/ACT ISDA ────────────────────────────────────────────────────────────

/// Reference: QuantLib daycounters.cpp testActualActual() — ISDA cases.
/// Each case is (start, end, expected_year_fraction).
#[test]
fn act_act_isda_quantlib_reference_values() {
    let cases: Vec<(NaiveDate, NaiveDate, f64)> = vec![
        // Normal year (2002, 365 days): Jan 1 → Dec 31 = 364 actual days
        (d(2002, 1, 1), d(2002, 12, 31), 364.0 / 365.0),
        // Single day
        (d(2003, 1, 1), d(2003, 1, 2), 1.0 / 365.0),
        // Leap year fraction (2004 is leap): Jan 1 → Mar 1 = 60 days
        (d(2004, 1, 1), d(2004, 3, 1), 60.0 / 366.0),
        // Cross-year: 2003 (non-leap) → 2004 (leap)
        // Nov 1 → Jan 1: 61 days in 2003; Jan 1 → May 1: 121 days in 2004
        (d(2003, 11, 1), d(2004, 5, 1),
            61.0 / 365.0 + 121.0 / 366.0),
        // Full non-leap year
        (d(2001, 1, 1), d(2002, 1, 1), 1.0),
        // Full leap year
        (d(2004, 1, 1), d(2005, 1, 1), 1.0),
        // Cross-year: non-leap into leap
        // Jun 15 → Jan 1: 200 days in 2003; Jan 1 → Mar 15: 74 days in 2004
        (d(2003, 6, 15), d(2004, 3, 15),
            200.0 / 365.0 + 74.0 / 366.0),
    ];

    for (i, (start, end, expected)) in cases.iter().enumerate() {
        let actual = year_fraction(*start, *end, DayCountConvention::ActActISDA);
        assert_relative_eq!(
            actual,
            *expected,
            epsilon = 1.0e-12,
        );
        // Verify antisymmetry: reversing dates negates the fraction
        let reversed = year_fraction(*end, *start, DayCountConvention::ActActISDA);
        assert_relative_eq!(reversed, -expected, epsilon = 1.0e-12);
        let _ = i;
    }
}

// ── 30/360 Bond Basis (Thirty360) ───────────────────────────────────────────

/// Reference: QuantLib daycounters.cpp testThirty360_BondBasis().
/// Each case is (start, end, expected_day_count_as_30/360_fraction).
///
/// The 30/360 US convention adjusts:
///   - If D1 == 31, set D1 = 30
///   - If D1 == last day of Feb, set D1 = 30
///   - If D2 == 31 and D1 >= 30, set D2 = 30
///   - If D1 is last-of-Feb and D2 is last-of-Feb, set D2 = 30
#[test]
fn thirty_360_bond_basis_quantlib_reference_values() {
    // (start, end, expected_30_360_day_count, expected_year_fraction)
    let cases: Vec<(NaiveDate, NaiveDate, i32, f64)> = vec![
        // Regular month-end to month-end
        (d(2006, 1, 31), d(2006, 2, 28), 28, 28.0 / 360.0),
        // EOM Feb (non-leap) to Mar: D1=28→30 (last of Feb), D2=31→30 (D1≥30)
        (d(2006, 2, 28), d(2006, 3, 31), 30, 30.0 / 360.0),
        // Regular month within same month
        (d(2006, 1, 15), d(2006, 2, 15), 30, 30.0 / 360.0),
        // Jan 31 → Feb 28 in non-leap year
        (d(2007, 1, 31), d(2007, 2, 28), 28, 28.0 / 360.0),
        // Feb 28 → Mar 31 in non-leap year (D1=28 last of Feb→30, D2=31 and D1>=30→30)
        (d(2007, 2, 28), d(2007, 3, 31), 30, 30.0 / 360.0),
        // Jan 31 → Feb 29 in leap year
        (d(2008, 1, 31), d(2008, 2, 29), 29, 29.0 / 360.0),
        // Feb 29 → Mar 31 in leap year (D1=29 last of Feb→30, D2=31 and D1>=30→30)
        (d(2008, 2, 29), d(2008, 3, 31), 30, 30.0 / 360.0),
        // Regular: same day different months
        (d(2006, 3, 1), d(2006, 4, 1), 30, 30.0 / 360.0),
        // 31st to 31st (both adjusted to 30)
        (d(2006, 1, 31), d(2006, 3, 31), 60, 60.0 / 360.0),
        // Mid-month to mid-month
        (d(2006, 3, 15), d(2006, 6, 15), 90, 90.0 / 360.0),
        // Cross-year
        (d(2006, 11, 15), d(2007, 3, 15), 120, 120.0 / 360.0),
        // Full year
        (d(2006, 1, 1), d(2007, 1, 1), 360, 1.0),
        // Six months
        (d(2006, 1, 1), d(2006, 7, 1), 180, 0.5),
    ];

    for (i, (start, end, _expected_days, expected_frac)) in cases.iter().enumerate() {
        let actual = year_fraction(*start, *end, DayCountConvention::Thirty360);
        assert_relative_eq!(
            actual,
            *expected_frac,
            epsilon = 1.0e-12,
        );
        let _ = i;
    }
}

// ── 30E/360 Eurobond Basis (ThirtyE360) ─────────────────────────────────────

/// Reference: QuantLib daycounters.cpp testThirty360_EurobondBasis().
/// 30E/360 adjusts: D1 = min(D1, 30), D2 = min(D2, 30).
#[test]
fn thirty_e_360_eurobond_quantlib_reference_values() {
    let cases: Vec<(NaiveDate, NaiveDate, i32, f64)> = vec![
        // Regular month-end
        (d(2006, 1, 31), d(2006, 2, 28), 28, 28.0 / 360.0),
        // Feb end to Mar 31 (D1=28, D2=31→30 => 30-28=2 + 30*(3-2) = 32)
        (d(2006, 2, 28), d(2006, 3, 31), 32, 32.0 / 360.0),
        // Regular mid-month
        (d(2006, 1, 15), d(2006, 2, 15), 30, 30.0 / 360.0),
        // Non-leap Feb to Mar 31 (D1=28, D2=31→30 => 30-28 + 30 = 32)
        (d(2007, 2, 28), d(2007, 3, 31), 32, 32.0 / 360.0),
        // 31st to 31st (both capped to 30, so 30-30 + 30*(3-1)=60)
        (d(2006, 1, 31), d(2006, 3, 31), 60, 60.0 / 360.0),
        // Full year
        (d(2006, 1, 1), d(2007, 1, 1), 360, 1.0),
        // 6 months
        (d(2006, 1, 1), d(2006, 7, 1), 180, 0.5),
        // March to June (mid-month)
        (d(2006, 3, 15), d(2006, 6, 15), 90, 90.0 / 360.0),
        // Cross-year mid-month
        (d(2006, 11, 15), d(2007, 3, 15), 120, 120.0 / 360.0),
        // Leap year Feb to Mar
        (d(2008, 2, 29), d(2008, 3, 31), 31, 31.0 / 360.0),
        // Jan to Feb in non-leap year
        (d(2007, 1, 31), d(2007, 2, 28), 28, 28.0 / 360.0),
        // Jan to Feb in leap year
        (d(2008, 1, 31), d(2008, 2, 29), 29, 29.0 / 360.0),
    ];

    for (i, (start, end, _expected_days, expected_frac)) in cases.iter().enumerate() {
        let actual = year_fraction(*start, *end, DayCountConvention::ThirtyE360);
        assert_relative_eq!(
            actual,
            *expected_frac,
            epsilon = 1.0e-12,
        );
        let _ = i;
    }
}

// ── ACT/360 ─────────────────────────────────────────────────────────────────

/// Verify ACT/360 against manually computed reference values.
#[test]
fn act_360_quantlib_reference_values() {
    let cases: Vec<(NaiveDate, NaiveDate, f64)> = vec![
        // 182 actual days / 360
        (d(2024, 1, 15), d(2024, 7, 15), 182.0 / 360.0),
        // Full non-leap year (365 days)
        (d(2003, 1, 1), d(2004, 1, 1), 365.0 / 360.0),
        // Full leap year (366 days)
        (d(2004, 1, 1), d(2005, 1, 1), 366.0 / 360.0),
        // Single day
        (d(2020, 6, 1), d(2020, 6, 2), 1.0 / 360.0),
        // Quarter in leap year
        (d(2024, 1, 1), d(2024, 4, 1), 91.0 / 360.0),
        // Cross-year
        (d(2023, 12, 15), d(2024, 3, 15), 91.0 / 360.0),
    ];

    for (start, end, expected) in &cases {
        let actual = year_fraction(*start, *end, DayCountConvention::Act360);
        assert_relative_eq!(actual, *expected, epsilon = 1.0e-12);
    }
}

// ── ACT/365 Fixed ───────────────────────────────────────────────────────────

/// Verify ACT/365Fixed against manually computed reference values.
#[test]
fn act_365_fixed_quantlib_reference_values() {
    let cases: Vec<(NaiveDate, NaiveDate, f64)> = vec![
        // 182 actual days / 365
        (d(2024, 1, 15), d(2024, 7, 15), 182.0 / 365.0),
        // Full non-leap year
        (d(2003, 1, 1), d(2004, 1, 1), 365.0 / 365.0),
        // Full leap year
        (d(2004, 1, 1), d(2005, 1, 1), 366.0 / 365.0),
        // Single day
        (d(2020, 6, 1), d(2020, 6, 2), 1.0 / 365.0),
        // Quarter in leap year
        (d(2024, 1, 1), d(2024, 4, 1), 91.0 / 365.0),
    ];

    for (start, end, expected) in &cases {
        let actual = year_fraction(*start, *end, DayCountConvention::Act365Fixed);
        assert_relative_eq!(actual, *expected, epsilon = 1.0e-12);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn day_count_same_date_returns_zero_all_conventions() {
    let date = d(2024, 6, 15);
    for conv in [
        DayCountConvention::Act360,
        DayCountConvention::Act365Fixed,
        DayCountConvention::ActActISDA,
        DayCountConvention::Thirty360,
        DayCountConvention::ThirtyE360,
    ] {
        assert_eq!(year_fraction(date, date, conv), 0.0);
    }
}

#[test]
fn day_count_negative_interval_antisymmetry() {
    let start = d(2024, 1, 1);
    let end = d(2024, 7, 1);
    for conv in [
        DayCountConvention::Act360,
        DayCountConvention::Act365Fixed,
        DayCountConvention::ActActISDA,
        DayCountConvention::Thirty360,
        DayCountConvention::ThirtyE360,
    ] {
        let forward = year_fraction(start, end, conv);
        let backward = year_fraction(end, start, conv);
        assert_relative_eq!(forward, -backward, epsilon = 1.0e-12);
    }
}

/// Multi-year span spanning normal, leap, and century years.
#[test]
fn act_act_isda_multi_year_span() {
    // 2000 is a leap year (divisible by 400), 2001-2003 normal, 2004 leap
    let start = d(2000, 1, 1);
    let end = d(2005, 1, 1);
    let expected = 5.0; // exactly 5 full years
    let actual = year_fraction(start, end, DayCountConvention::ActActISDA);
    assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
}
