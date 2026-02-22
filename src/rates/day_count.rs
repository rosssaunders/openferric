//! Module `rates::day_count`.
//!
//! Implements day count workflows with concrete routines such as `year_fraction`.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `DayCountConvention` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use chrono::{Datelike, NaiveDate};

/// Supported day-count conventions for fixed-income instruments.
///
/// Conventions follow standard market definitions used in coupon accrual and
/// curve construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayCountConvention {
    /// Actual day count over a 360-day year.
    Act360,
    /// Actual day count over a 365-day year.
    Act365Fixed,
    /// ISDA actual/actual convention.
    ActActISDA,
    /// 30/360 US (bond basis).
    Thirty360,
    /// 30E/360 European convention.
    ThirtyE360,
}

/// Computes year fraction between two dates under a day-count convention.
///
/// Parameters:
/// - `start`, `end`: accrual period boundaries.
/// - `convention`: day-count algorithm.
///
/// Edge cases:
/// - If `start == end`, returns `0.0`.
/// - If `start > end`, the result is negative and antisymmetric.
///
/// # Examples
/// ```rust
/// use chrono::NaiveDate;
/// use openferric::rates::{DayCountConvention, year_fraction};
///
/// let s = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
/// let e = NaiveDate::from_ymd_opt(2025, 7, 1).unwrap();
/// let yf = year_fraction(s, e, DayCountConvention::Act360);
/// assert!(yf > 0.49 && yf < 0.51);
/// ```
///
/// ```rust
/// use chrono::NaiveDate;
/// use openferric::rates::{DayCountConvention, year_fraction};
///
/// let s = NaiveDate::from_ymd_opt(2025, 3, 1).unwrap();
/// let e = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();
/// assert_eq!(
///     year_fraction(s, e, DayCountConvention::Act365Fixed),
///     -year_fraction(e, s, DayCountConvention::Act365Fixed)
/// );
/// ```
pub fn year_fraction(start: NaiveDate, end: NaiveDate, convention: DayCountConvention) -> f64 {
    if start == end {
        return 0.0;
    }
    if start > end {
        return -year_fraction(end, start, convention);
    }

    match convention {
        DayCountConvention::Act360 => (end - start).num_days() as f64 / 360.0,
        DayCountConvention::Act365Fixed => (end - start).num_days() as f64 / 365.0,
        DayCountConvention::ActActISDA => year_fraction_act_act_isda(start, end),
        DayCountConvention::Thirty360 => year_fraction_thirty_360(start, end),
        DayCountConvention::ThirtyE360 => year_fraction_thirty_e_360(start, end),
    }
}

fn year_fraction_act_act_isda(start: NaiveDate, end: NaiveDate) -> f64 {
    if start.year() == end.year() {
        return (end - start).num_days() as f64 / days_in_year(start.year()) as f64;
    }

    let mut fraction = 0.0;
    let start_of_next_year = NaiveDate::from_ymd_opt(start.year() + 1, 1, 1).unwrap();
    fraction += (start_of_next_year - start).num_days() as f64 / days_in_year(start.year()) as f64;

    for year in (start.year() + 1)..end.year() {
        let _ = year;
        fraction += 1.0;
    }

    let start_of_end_year = NaiveDate::from_ymd_opt(end.year(), 1, 1).unwrap();
    fraction += (end - start_of_end_year).num_days() as f64 / days_in_year(end.year()) as f64;
    fraction
}

fn year_fraction_thirty_360(start: NaiveDate, end: NaiveDate) -> f64 {
    let y1 = start.year();
    let m1 = start.month() as i32;
    let mut d1 = start.day() as i32;
    let y2 = end.year();
    let m2 = end.month() as i32;
    let mut d2 = end.day() as i32;

    if is_last_day_of_feb(start) {
        d1 = 30;
    }
    if d1 == 31 {
        d1 = 30;
    }
    if is_last_day_of_feb(start) && is_last_day_of_feb(end) {
        d2 = 30;
    }
    if d2 == 31 && d1 >= 30 {
        d2 = 30;
    }

    let days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1);
    days as f64 / 360.0
}

fn year_fraction_thirty_e_360(start: NaiveDate, end: NaiveDate) -> f64 {
    let y1 = start.year();
    let m1 = start.month() as i32;
    let mut d1 = start.day() as i32;
    let y2 = end.year();
    let m2 = end.month() as i32;
    let mut d2 = end.day() as i32;

    if d1 == 31 {
        d1 = 30;
    }
    if d2 == 31 {
        d2 = 30;
    }

    let days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1);
    days as f64 / 360.0
}

fn days_in_year(year: i32) -> i32 {
    if is_leap_year(year) { 366 } else { 365 }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn is_last_day_of_feb(date: NaiveDate) -> bool {
    date.month() == 2 && date.day() == days_in_month(date.year(), date.month())
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => unreachable!("invalid month"),
    }
}
