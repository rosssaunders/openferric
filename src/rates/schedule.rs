use chrono::{Datelike, NaiveDate};

/// Payment frequency for coupon schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    Annual,
    SemiAnnual,
    Quarterly,
    Monthly,
}

impl Frequency {
    /// Returns the number of months in one coupon period.
    pub fn months(self) -> u32 {
        match self {
            Self::Annual => 12,
            Self::SemiAnnual => 6,
            Self::Quarterly => 3,
            Self::Monthly => 1,
        }
    }
}

/// Business-day adjustment convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessDayConvention {
    ModifiedFollowing,
}

/// Generates schedule dates from `start` to `end`, including both endpoints.
pub fn generate_schedule(start: NaiveDate, end: NaiveDate, freq: Frequency) -> Vec<NaiveDate> {
    if end <= start {
        return vec![start];
    }

    let mut schedule = vec![adjust_business_day(
        start,
        BusinessDayConvention::ModifiedFollowing,
    )];
    let mut current = start;
    let step_months = freq.months();

    loop {
        let next = add_months(current, step_months);
        if next >= end {
            break;
        }

        schedule.push(adjust_business_day(
            next,
            BusinessDayConvention::ModifiedFollowing,
        ));
        current = next;
    }

    schedule.push(adjust_business_day(
        end,
        BusinessDayConvention::ModifiedFollowing,
    ));
    schedule
}

fn adjust_business_day(date: NaiveDate, _convention: BusinessDayConvention) -> NaiveDate {
    // Stub implementation: leave dates unadjusted for now.
    date
}

fn add_months(date: NaiveDate, months: u32) -> NaiveDate {
    let total_months = date.month0() + months;
    let year = date.year() + (total_months / 12) as i32;
    let month = (total_months % 12) + 1;
    let day = date.day().min(days_in_month(year, month));

    NaiveDate::from_ymd_opt(year, month, day).expect("valid y-m-d in add_months")
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

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
