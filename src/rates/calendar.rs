//! Module `rates::calendar`.
//!
//! Implements calendar workflows with concrete routines such as `adjust_business_day`, `add_business_days`, `subtract_business_days`, `business_day_count`.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `Frequency`, `WeekendConvention`, `BusinessDayConvention`, `StubConvention`, `RollConvention` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::BTreeSet;

/// Payment frequency for coupon schedules.
///
/// The frequency determines the regular schedule interval in months:
///
/// - annual: 12 months
/// - semi-annual: 6 months
/// - quarterly: 3 months
/// - monthly: 1 month
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Frequency {
    Annual,
    SemiAnnual,
    Quarterly,
    Monthly,
}

impl Frequency {
    /// Number of months in one regular coupon period.
    pub fn months(self) -> i32 {
        match self {
            Self::Annual => 12,
            Self::SemiAnnual => 6,
            Self::Quarterly => 3,
            Self::Monthly => 1,
        }
    }
}

/// Weekend definition used by a calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeekendConvention {
    /// Saturday/Sunday weekends (most global markets).
    SaturdaySunday,
    /// Friday/Saturday weekends (common in parts of the Middle East).
    FridaySaturday,
}

impl WeekendConvention {
    fn is_weekend(self, weekday: Weekday) -> bool {
        match self {
            Self::SaturdaySunday => matches!(weekday, Weekday::Sat | Weekday::Sun),
            Self::FridaySaturday => matches!(weekday, Weekday::Fri | Weekday::Sat),
        }
    }
}

/// Business-day adjustment rule.
///
/// Let `d` be an unadjusted date:
///
/// - `Following`: first business day `>= d`
/// - `ModifiedFollowing`: following unless month changes, then preceding
/// - `Preceding`: last business day `<= d`
/// - `ModifiedPreceding`: preceding unless month changes, then following
/// - `Unadjusted`: leave `d` unchanged
/// - `Nearest`: closest business day; ties resolve to following
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessDayConvention {
    Following,
    ModifiedFollowing,
    Preceding,
    ModifiedPreceding,
    Unadjusted,
    Nearest,
}

/// Stub-period placement when generating schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StubConvention {
    /// Generate forward from start date; short final period if needed.
    ShortBack,
    /// Generate forward and merge final short stub into prior period.
    LongBack,
    /// Generate backward from end date; short initial period if needed.
    ShortFront,
    /// Generate backward and merge initial short stub into next period.
    LongFront,
}

/// Rule for day-of-month rolling in regular schedule generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollConvention {
    /// Keep the anchor day-of-month from schedule boundary.
    None,
    /// End-of-month rolling.
    EndOfMonth,
    /// Quarterly IMM rolling (third Wednesday of Mar/Jun/Sep/Dec).
    Imm,
    /// Fixed day of month.
    DayOfMonth(u32),
    /// 15th day of month.
    Fifteenth,
}

/// Built-in market holiday centers.
///
/// Rules follow standard market calendars and match QuantLib behavior for
/// core fixed-income/derivatives use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinancialCenter {
    Nyc,
    London,
    Target,
    Tokyo,
    Sydney,
    HongKong,
    Singapore,
}

/// User-defined holiday calendar.
///
/// This supports:
///
/// - arbitrary holiday dates
/// - weekend convention override
/// - explicit business-day overrides (rare, but useful for ad-hoc make-up days)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomCalendar {
    weekend_convention: WeekendConvention,
    holidays: BTreeSet<NaiveDate>,
    business_day_overrides: BTreeSet<NaiveDate>,
}

impl CustomCalendar {
    /// Creates an empty custom calendar.
    pub fn new(weekend_convention: WeekendConvention) -> Self {
        Self {
            weekend_convention,
            holidays: BTreeSet::new(),
            business_day_overrides: BTreeSet::new(),
        }
    }

    /// Creates a custom calendar preloaded with holidays.
    pub fn with_holidays<I>(weekend_convention: WeekendConvention, holidays: I) -> Self
    where
        I: IntoIterator<Item = NaiveDate>,
    {
        Self {
            weekend_convention,
            holidays: holidays.into_iter().collect(),
            business_day_overrides: BTreeSet::new(),
        }
    }

    /// Adds a holiday date.
    pub fn add_holiday(&mut self, date: NaiveDate) {
        self.holidays.insert(date);
        self.business_day_overrides.remove(&date);
    }

    /// Adds a date override that is treated as business day even if weekend/holiday.
    pub fn add_business_day_override(&mut self, date: NaiveDate) {
        self.business_day_overrides.insert(date);
        self.holidays.remove(&date);
    }

    fn is_business_day(&self, date: NaiveDate) -> bool {
        if self.business_day_overrides.contains(&date) {
            return true;
        }
        if self.holidays.contains(&date) {
            return false;
        }
        !self.weekend_convention.is_weekend(date.weekday())
    }
}

/// Business-day calendar.
///
/// Calendars can be:
///
/// - built-in market centers (`FinancialCenter`)
/// - custom calendars with user holidays
/// - joint calendars, where a day is business only if all child calendars
///   are business days (union of holidays/weekends)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Calendar {
    WeekendsOnly(WeekendConvention),
    FinancialCenter(FinancialCenter),
    Custom(CustomCalendar),
    Joint(Vec<Calendar>),
}

impl Calendar {
    /// Weekend-only calendar using Saturday/Sunday weekends.
    pub fn weekends_only() -> Self {
        Self::WeekendsOnly(WeekendConvention::SaturdaySunday)
    }

    /// New York calendar.
    pub fn nyc() -> Self {
        Self::FinancialCenter(FinancialCenter::Nyc)
    }

    /// London calendar.
    pub fn london() -> Self {
        Self::FinancialCenter(FinancialCenter::London)
    }

    /// TARGET calendar.
    pub fn target() -> Self {
        Self::FinancialCenter(FinancialCenter::Target)
    }

    /// Tokyo calendar.
    pub fn tokyo() -> Self {
        Self::FinancialCenter(FinancialCenter::Tokyo)
    }

    /// Sydney calendar.
    pub fn sydney() -> Self {
        Self::FinancialCenter(FinancialCenter::Sydney)
    }

    /// Hong Kong calendar.
    pub fn hong_kong() -> Self {
        Self::FinancialCenter(FinancialCenter::HongKong)
    }

    /// Singapore calendar.
    pub fn singapore() -> Self {
        Self::FinancialCenter(FinancialCenter::Singapore)
    }

    /// Creates a custom calendar.
    pub fn custom(custom: CustomCalendar) -> Self {
        Self::Custom(custom)
    }

    /// Creates a joint calendar (holiday union).
    pub fn joint(calendars: Vec<Calendar>) -> Self {
        Self::Joint(calendars)
    }

    /// Returns true if `date` is a business day.
    pub fn is_business_day(&self, date: NaiveDate) -> bool {
        match self {
            Self::WeekendsOnly(weekend) => !weekend.is_weekend(date.weekday()),
            Self::FinancialCenter(center) => {
                !WeekendConvention::SaturdaySunday.is_weekend(date.weekday())
                    && !is_center_holiday(*center, date)
            }
            Self::Custom(custom) => custom.is_business_day(date),
            Self::Joint(calendars) => {
                if calendars.is_empty() {
                    !WeekendConvention::SaturdaySunday.is_weekend(date.weekday())
                } else {
                    calendars.iter().all(|cal| cal.is_business_day(date))
                }
            }
        }
    }

    /// Returns true if `date` is not a business day.
    pub fn is_holiday(&self, date: NaiveDate) -> bool {
        !self.is_business_day(date)
    }
}

/// Schedule-generation parameters.
///
/// This keeps schedule generation deterministic and explicit:
///
/// - business-day calendar and adjustment convention
/// - stub positioning
/// - date roll convention
///
/// # Numerical Notes
///
/// - Regular periods are generated first (unadjusted), then business-day
///   conventions are applied.
/// - If business-day adjustment collapses two adjacent boundaries to the same
///   date, duplicates are removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleConfig {
    pub calendar: Calendar,
    pub business_day_convention: BusinessDayConvention,
    pub stub_convention: StubConvention,
    pub roll_convention: RollConvention,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            calendar: Calendar::weekends_only(),
            business_day_convention: BusinessDayConvention::ModifiedFollowing,
            stub_convention: StubConvention::ShortBack,
            roll_convention: RollConvention::None,
        }
    }
}

/// Adjusts a date according to a business-day convention under `calendar`.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use openferric::rates::{
///     adjust_business_day, BusinessDayConvention, Calendar,
/// };
///
/// let calendar = Calendar::weekends_only();
/// let saturday = NaiveDate::from_ymd_opt(2026, 1, 3).unwrap();
/// let adjusted = adjust_business_day(
///     saturday,
///     BusinessDayConvention::Following,
///     &calendar,
/// );
/// assert_eq!(adjusted, NaiveDate::from_ymd_opt(2026, 1, 5).unwrap());
/// ```
pub fn adjust_business_day(
    date: NaiveDate,
    convention: BusinessDayConvention,
    calendar: &Calendar,
) -> NaiveDate {
    match convention {
        BusinessDayConvention::Unadjusted => date,
        BusinessDayConvention::Following => next_business_day(date, calendar),
        BusinessDayConvention::Preceding => previous_business_day(date, calendar),
        BusinessDayConvention::ModifiedFollowing => {
            let following = next_business_day(date, calendar);
            if following.month() != date.month() {
                previous_business_day(date, calendar)
            } else {
                following
            }
        }
        BusinessDayConvention::ModifiedPreceding => {
            let preceding = previous_business_day(date, calendar);
            if preceding.month() != date.month() {
                next_business_day(date, calendar)
            } else {
                preceding
            }
        }
        BusinessDayConvention::Nearest => nearest_business_day(date, calendar),
    }
}

/// Adds business days to a date.
///
/// Negative `days` moves backward.
pub fn add_business_days(date: NaiveDate, days: i32, calendar: &Calendar) -> NaiveDate {
    if days == 0 {
        return date;
    }

    let step = if days > 0 { 1_i64 } else { -1_i64 };
    let mut left = days.abs();
    let mut current = date;

    while left > 0 {
        current += Duration::days(step);
        if calendar.is_business_day(current) {
            left -= 1;
        }
    }

    current
}

/// Subtracts business days from a date.
pub fn subtract_business_days(date: NaiveDate, days: i32, calendar: &Calendar) -> NaiveDate {
    add_business_days(date, -days, calendar)
}

/// Counts business days in `(start, end]`.
///
/// Returns negative count when `start > end`.
pub fn business_day_count(start: NaiveDate, end: NaiveDate, calendar: &Calendar) -> i32 {
    if start == end {
        return 0;
    }
    if start > end {
        return -business_day_count(end, start, calendar);
    }

    let mut d = start;
    let mut count = 0_i32;
    while d < end {
        d += Duration::days(1);
        if calendar.is_business_day(d) {
            count += 1;
        }
    }
    count
}

/// Business/252 year fraction using the supplied calendar.
pub fn year_fraction_business_252(start: NaiveDate, end: NaiveDate, calendar: &Calendar) -> f64 {
    business_day_count(start, end, calendar) as f64 / 252.0
}

/// Generates a schedule with default calendar settings.
///
/// Defaults:
///
/// - weekend-only Saturday/Sunday calendar
/// - Modified Following business-day adjustment
/// - short back stub
/// - no roll convention
pub fn generate_schedule(start: NaiveDate, end: NaiveDate, freq: Frequency) -> Vec<NaiveDate> {
    generate_schedule_with_config(start, end, freq, &ScheduleConfig::default())
}

/// Generates a schedule from `start` to `end` with explicit schedule config.
///
/// Returned dates are increasing and include both endpoints (post-adjustment,
/// unless duplicates collapse).
pub fn generate_schedule_with_config(
    start: NaiveDate,
    end: NaiveDate,
    freq: Frequency,
    config: &ScheduleConfig,
) -> Vec<NaiveDate> {
    if end <= start {
        return vec![adjust_business_day(
            start,
            config.business_day_convention,
            &config.calendar,
        )];
    }

    let unadjusted = generate_unadjusted_schedule(
        start,
        end,
        freq,
        config.stub_convention,
        config.roll_convention,
    );

    let mut adjusted = Vec::with_capacity(unadjusted.len());
    for d in unadjusted {
        let a = adjust_business_day(d, config.business_day_convention, &config.calendar);
        if adjusted.last().copied() != Some(a) {
            adjusted.push(a);
        }
    }

    adjusted
}

/// Returns true if `date` is a standard quarterly IMM date.
pub fn is_imm_date(date: NaiveDate) -> bool {
    is_imm_month(date.month()) && date.weekday() == Weekday::Wed && (15..=21).contains(&date.day())
}

/// Next IMM date on or after `date`.
pub fn next_imm_date(date: NaiveDate) -> NaiveDate {
    let mut year = date.year();
    loop {
        for month in [3_u32, 6, 9, 12] {
            let candidate = third_wednesday(year, month);
            if candidate >= date {
                return candidate;
            }
        }
        year += 1;
    }
}

/// Previous IMM date on or before `date`.
pub fn previous_imm_date(date: NaiveDate) -> NaiveDate {
    let mut year = date.year();
    loop {
        for month in [12_u32, 9, 6, 3] {
            let candidate = third_wednesday(year, month);
            if candidate <= date {
                return candidate;
            }
        }
        year -= 1;
    }
}

/// Returns true if `date` is a standard CDS date (20th of IMM months).
pub fn is_cds_standard_date(date: NaiveDate) -> bool {
    is_imm_month(date.month()) && date.day() == 20
}

/// Next standard CDS date (20th of Mar/Jun/Sep/Dec) on or after `date`.
pub fn next_cds_date(date: NaiveDate) -> NaiveDate {
    let mut year = date.year();
    loop {
        for month in [3_u32, 6, 9, 12] {
            if let Some(candidate) = NaiveDate::from_ymd_opt(year, month, 20)
                && candidate >= date
            {
                return candidate;
            }
        }
        year += 1;
    }
}

/// Previous standard CDS date (20th of Mar/Jun/Sep/Dec) on or before `date`.
pub fn previous_cds_date(date: NaiveDate) -> NaiveDate {
    let mut year = date.year();
    loop {
        for month in [12_u32, 9, 6, 3] {
            if let Some(candidate) = NaiveDate::from_ymd_opt(year, month, 20)
                && candidate <= date
            {
                return candidate;
            }
        }
        year -= 1;
    }
}

/// Third Wednesday of a month.
pub fn third_wednesday(year: i32, month: u32) -> NaiveDate {
    nth_weekday_of_month(year, month, Weekday::Wed, 3)
}

/// Adds calendar months with end-of-month clamping.
pub fn add_months(date: NaiveDate, months: i32) -> NaiveDate {
    let month0 = date.month0() as i32;
    let total = month0 + months;

    let mut year = date.year() + total.div_euclid(12);
    let mut month0_new = total.rem_euclid(12);
    if month0_new < 0 {
        year -= 1;
        month0_new += 12;
    }

    let month = month0_new as u32 + 1;
    let day = date.day().min(days_in_month(year, month));
    NaiveDate::from_ymd_opt(year, month, day).expect("valid add_months result")
}

fn generate_unadjusted_schedule(
    start: NaiveDate,
    end: NaiveDate,
    freq: Frequency,
    stub_convention: StubConvention,
    roll_convention: RollConvention,
) -> Vec<NaiveDate> {
    let step_months = freq.months();
    match stub_convention {
        StubConvention::ShortBack => {
            generate_forward_schedule(start, end, step_months, roll_convention, false)
        }
        StubConvention::LongBack => {
            generate_forward_schedule(start, end, step_months, roll_convention, true)
        }
        StubConvention::ShortFront => {
            generate_backward_schedule(start, end, step_months, roll_convention, false)
        }
        StubConvention::LongFront => {
            generate_backward_schedule(start, end, step_months, roll_convention, true)
        }
    }
}

fn generate_forward_schedule(
    start: NaiveDate,
    end: NaiveDate,
    step_months: i32,
    roll_convention: RollConvention,
    long_back: bool,
) -> Vec<NaiveDate> {
    let anchor_day = anchor_day(start, roll_convention);
    let mut out = vec![start];
    let mut current = start;
    let imm_steps = ((step_months / 3).max(1)) as usize;

    let had_stub = loop {
        let next = if roll_convention == RollConvention::Imm && current == start {
            let mut n = next_imm_date(start + Duration::days(1));
            for _ in 1..imm_steps {
                n = add_months_with_roll(n, 3, RollConvention::Imm, anchor_day, Direction::Forward);
            }
            n
        } else {
            add_months_with_roll(
                current,
                step_months,
                roll_convention,
                anchor_day,
                Direction::Forward,
            )
        };
        if next >= end {
            break next > end;
        }
        out.push(next);
        current = next;
    };

    if long_back && had_stub && out.len() > 1 {
        out.pop();
    }
    out.push(end);
    out
}

fn generate_backward_schedule(
    start: NaiveDate,
    end: NaiveDate,
    step_months: i32,
    roll_convention: RollConvention,
    long_front: bool,
) -> Vec<NaiveDate> {
    let anchor_day = anchor_day(end, roll_convention);
    let mut rev = vec![end];
    let mut current = end;
    let imm_steps = ((step_months / 3).max(1)) as usize;

    let had_stub = loop {
        let prev = if roll_convention == RollConvention::Imm && current == end {
            let mut p = previous_imm_date(end - Duration::days(1));
            for _ in 1..imm_steps {
                p = add_months_with_roll(
                    p,
                    -3,
                    RollConvention::Imm,
                    anchor_day,
                    Direction::Backward,
                );
            }
            p
        } else {
            add_months_with_roll(
                current,
                -step_months,
                roll_convention,
                anchor_day,
                Direction::Backward,
            )
        };
        if prev <= start {
            break prev < start;
        }
        rev.push(prev);
        current = prev;
    };

    if long_front && had_stub && rev.len() > 1 {
        rev.pop();
    }

    rev.push(start);
    rev.reverse();
    rev
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Backward,
}

fn add_months_with_roll(
    date: NaiveDate,
    months: i32,
    roll_convention: RollConvention,
    anchor_day: u32,
    direction: Direction,
) -> NaiveDate {
    let target = add_months(date, months);
    match roll_convention {
        RollConvention::None => roll_day(target.year(), target.month(), anchor_day),
        RollConvention::DayOfMonth(day) => roll_day(target.year(), target.month(), day),
        RollConvention::Fifteenth => roll_day(target.year(), target.month(), 15),
        RollConvention::EndOfMonth => {
            let dim = days_in_month(target.year(), target.month());
            NaiveDate::from_ymd_opt(target.year(), target.month(), dim)
                .expect("valid end-of-month roll")
        }
        RollConvention::Imm => {
            let (y, m) = nearest_imm_month(target.year(), target.month(), direction);
            third_wednesday(y, m)
        }
    }
}

fn nearest_imm_month(year: i32, month: u32, direction: Direction) -> (i32, u32) {
    let mut y = year;
    let mut m = month;
    loop {
        if is_imm_month(m) {
            return (y, m);
        }
        match direction {
            Direction::Forward => {
                if m == 12 {
                    m = 1;
                    y += 1;
                } else {
                    m += 1;
                }
            }
            Direction::Backward => {
                if m == 1 {
                    m = 12;
                    y -= 1;
                } else {
                    m -= 1;
                }
            }
        }
    }
}

fn anchor_day(anchor: NaiveDate, roll: RollConvention) -> u32 {
    match roll {
        RollConvention::None => anchor.day(),
        RollConvention::DayOfMonth(day) => day.clamp(1, 31),
        RollConvention::Fifteenth => 15,
        RollConvention::EndOfMonth => 31,
        RollConvention::Imm => 20,
    }
}

fn roll_day(year: i32, month: u32, day: u32) -> NaiveDate {
    let clamped = day.clamp(1, 31).min(days_in_month(year, month));
    NaiveDate::from_ymd_opt(year, month, clamped).expect("valid rolled date")
}

fn next_business_day(date: NaiveDate, calendar: &Calendar) -> NaiveDate {
    let mut d = date;
    while !calendar.is_business_day(d) {
        d += Duration::days(1);
    }
    d
}

fn previous_business_day(date: NaiveDate, calendar: &Calendar) -> NaiveDate {
    let mut d = date;
    while !calendar.is_business_day(d) {
        d -= Duration::days(1);
    }
    d
}

fn nearest_business_day(date: NaiveDate, calendar: &Calendar) -> NaiveDate {
    if calendar.is_business_day(date) {
        return date;
    }

    for distance in 1..32_i64 {
        let prev = date - Duration::days(distance);
        let next = date + Duration::days(distance);
        let prev_ok = calendar.is_business_day(prev);
        let next_ok = calendar.is_business_day(next);
        if prev_ok && next_ok {
            return next;
        }
        if next_ok {
            return next;
        }
        if prev_ok {
            return prev;
        }
    }

    next_business_day(date, calendar)
}

fn is_center_holiday(center: FinancialCenter, date: NaiveDate) -> bool {
    match center {
        FinancialCenter::Nyc => is_nyc_holiday(date),
        FinancialCenter::London => is_london_holiday(date),
        FinancialCenter::Target => is_target_holiday(date),
        FinancialCenter::Tokyo => is_tokyo_holiday(date),
        FinancialCenter::Sydney => is_sydney_holiday(date),
        FinancialCenter::HongKong => is_hong_kong_holiday(date),
        FinancialCenter::Singapore => is_singapore_holiday(date),
    }
}

fn is_nyc_holiday(date: NaiveDate) -> bool {
    let y = date.year();
    date == easter_sunday(y) - Duration::days(2)
        || is_us_observed_fixed_holiday(date, 1, 1)
        || date == nth_weekday_of_month(y, 1, Weekday::Mon, 3)
        || date == nth_weekday_of_month(y, 2, Weekday::Mon, 3)
        || date == last_weekday_of_month(y, 5, Weekday::Mon)
        || (y >= 2022 && is_us_observed_fixed_holiday(date, 6, 19))
        || is_us_observed_fixed_holiday(date, 7, 4)
        || date == nth_weekday_of_month(y, 9, Weekday::Mon, 1)
        || date == nth_weekday_of_month(y, 10, Weekday::Mon, 2)
        || is_us_observed_fixed_holiday(date, 11, 11)
        || date == nth_weekday_of_month(y, 11, Weekday::Thu, 4)
        || is_us_observed_fixed_holiday(date, 12, 25)
}

fn is_london_holiday(date: NaiveDate) -> bool {
    let y = date.year();
    if is_uk_new_year_holiday(date) {
        return true;
    }

    let easter = easter_sunday(y);
    if date == easter - Duration::days(2) || date == easter + Duration::days(1) {
        return true;
    }

    if date == nth_weekday_of_month(y, 5, Weekday::Mon, 1)
        || date == last_weekday_of_month(y, 5, Weekday::Mon)
        || date == last_weekday_of_month(y, 8, Weekday::Mon)
    {
        return true;
    }

    uk_christmas_and_boxing_holidays(y).contains(&date)
}

fn is_target_holiday(date: NaiveDate) -> bool {
    let y = date.year();
    let easter = easter_sunday(y);
    matches!(
        (date.month(), date.day()),
        (1, 1) | (5, 1) | (12, 25) | (12, 26)
    ) || date == easter - Duration::days(2)
        || date == easter + Duration::days(1)
}

fn is_tokyo_holiday(date: NaiveDate) -> bool {
    japan_holidays(date.year()).contains(&date)
}

fn is_sydney_holiday(date: NaiveDate) -> bool {
    australia_nsw_holidays(date.year()).contains(&date)
}

fn is_hong_kong_holiday(date: NaiveDate) -> bool {
    hong_kong_holidays(date.year()).contains(&date)
}

fn is_singapore_holiday(date: NaiveDate) -> bool {
    singapore_holidays(date.year()).contains(&date)
}

fn is_imm_month(month: u32) -> bool {
    matches!(month, 3 | 6 | 9 | 12)
}

fn is_us_observed_fixed_holiday(date: NaiveDate, month: u32, day: u32) -> bool {
    for y in [date.year() - 1, date.year(), date.year() + 1] {
        let actual = NaiveDate::from_ymd_opt(y, month, day).expect("valid fixed holiday date");
        if date == actual {
            return true;
        }
        if actual.weekday() == Weekday::Sat && date == actual - Duration::days(1) {
            return true;
        }
        if actual.weekday() == Weekday::Sun && date == actual + Duration::days(1) {
            return true;
        }
    }
    false
}

fn observed_monday_if_weekend(year: i32, month: u32, day: u32) -> BTreeSet<NaiveDate> {
    let actual = NaiveDate::from_ymd_opt(year, month, day).expect("valid fixed holiday");
    let mut out = BTreeSet::new();
    out.insert(actual);
    match actual.weekday() {
        Weekday::Sat => {
            out.insert(actual + Duration::days(2));
        }
        Weekday::Sun => {
            out.insert(actual + Duration::days(1));
        }
        _ => {}
    }
    out
}

fn is_uk_new_year_holiday(date: NaiveDate) -> bool {
    let actual = NaiveDate::from_ymd_opt(date.year(), 1, 1).expect("valid UK new year date");
    if date == actual {
        return true;
    }
    match actual.weekday() {
        Weekday::Sat => date == actual + Duration::days(2),
        Weekday::Sun => date == actual + Duration::days(1),
        _ => false,
    }
}

fn uk_christmas_and_boxing_holidays(year: i32) -> BTreeSet<NaiveDate> {
    let christmas = NaiveDate::from_ymd_opt(year, 12, 25).expect("valid christmas date");
    let boxing = NaiveDate::from_ymd_opt(year, 12, 26).expect("valid boxing date");
    let mut out = BTreeSet::new();
    out.insert(christmas);
    out.insert(boxing);

    match christmas.weekday() {
        Weekday::Fri => {
            if boxing.weekday() == Weekday::Sat {
                out.insert(boxing + Duration::days(2));
            }
        }
        Weekday::Sat => {
            out.insert(christmas + Duration::days(2));
            out.insert(christmas + Duration::days(3));
        }
        Weekday::Sun => {
            out.insert(christmas + Duration::days(2));
        }
        _ => {
            if boxing.weekday() == Weekday::Sun {
                out.insert(boxing + Duration::days(1));
            }
            if boxing.weekday() == Weekday::Sat {
                out.insert(boxing + Duration::days(2));
            }
        }
    }

    out
}

fn japan_holidays(year: i32) -> BTreeSet<NaiveDate> {
    let mut holidays = BTreeSet::new();

    holidays.insert(NaiveDate::from_ymd_opt(year, 1, 1).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 1, 2).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 1, 3).expect("valid date"));
    holidays.insert(nth_weekday_of_month(year, 1, Weekday::Mon, 2)); // Coming of age day
    holidays.insert(NaiveDate::from_ymd_opt(year, 2, 11).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 2, 23).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 3, vernal_equinox_day(year)).expect("valid"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 4, 29).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 5, 3).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 5, 4).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 5, 5).expect("valid date"));
    holidays.insert(nth_weekday_of_month(year, 7, Weekday::Mon, 3)); // Marine day
    holidays.insert(NaiveDate::from_ymd_opt(year, 8, 11).expect("valid date"));
    holidays.insert(nth_weekday_of_month(year, 9, Weekday::Mon, 3)); // Respect for aged day
    holidays.insert(NaiveDate::from_ymd_opt(year, 9, autumnal_equinox_day(year)).expect("valid"));
    holidays.insert(nth_weekday_of_month(year, 10, Weekday::Mon, 2)); // Sports day
    holidays.insert(NaiveDate::from_ymd_opt(year, 11, 3).expect("valid date"));
    holidays.insert(NaiveDate::from_ymd_opt(year, 11, 23).expect("valid date"));

    let originals = holidays.clone();
    for h in originals {
        if h.weekday() == Weekday::Sun {
            let mut sub = h + Duration::days(1);
            while holidays.contains(&sub) {
                sub += Duration::days(1);
            }
            holidays.insert(sub);
        }
    }

    holidays
}

fn australia_nsw_holidays(year: i32) -> BTreeSet<NaiveDate> {
    let mut holidays = BTreeSet::new();
    holidays.extend(observed_monday_if_weekend(year, 1, 1)); // New year
    holidays.extend(observed_monday_if_weekend(year, 1, 26)); // Australia day

    let easter = easter_sunday(year);
    holidays.insert(easter - Duration::days(2)); // Good Friday
    holidays.insert(easter + Duration::days(1)); // Easter Monday

    holidays.insert(NaiveDate::from_ymd_opt(year, 4, 25).expect("valid date")); // ANZAC day
    holidays.insert(nth_weekday_of_month(year, 6, Weekday::Mon, 2)); // King's birthday
    holidays.insert(nth_weekday_of_month(year, 10, Weekday::Mon, 1)); // Labour day
    holidays.extend(uk_christmas_and_boxing_holidays(year));

    holidays
}

fn hong_kong_holidays(year: i32) -> BTreeSet<NaiveDate> {
    let mut holidays = BTreeSet::new();
    holidays.extend(observed_monday_if_weekend(year, 1, 1)); // New year

    let easter = easter_sunday(year);
    holidays.insert(easter - Duration::days(2)); // Good Friday
    holidays.insert(easter + Duration::days(1)); // Easter Monday

    holidays.extend(observed_monday_if_weekend(year, 5, 1)); // Labour day
    holidays.extend(observed_monday_if_weekend(year, 7, 1)); // HKSAR day
    holidays.extend(observed_monday_if_weekend(year, 10, 1)); // National day
    holidays.extend(uk_christmas_and_boxing_holidays(year));

    holidays
}

fn singapore_holidays(year: i32) -> BTreeSet<NaiveDate> {
    let mut holidays = BTreeSet::new();
    holidays.extend(observed_monday_if_weekend(year, 1, 1)); // New year
    holidays.insert(easter_sunday(year) - Duration::days(2)); // Good Friday
    holidays.extend(observed_monday_if_weekend(year, 5, 1)); // Labour day
    holidays.extend(observed_monday_if_weekend(year, 8, 9)); // National day
    holidays.extend(observed_monday_if_weekend(year, 12, 25)); // Christmas
    holidays
}

fn easter_sunday(year: i32) -> NaiveDate {
    // Gregorian calendar (Meeus/Jones/Butcher algorithm).
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    NaiveDate::from_ymd_opt(year, month as u32, day as u32).expect("valid easter sunday")
}

fn vernal_equinox_day(year: i32) -> u32 {
    // QuantLib-compatible approximation in Gregorian years.
    (20.8431 + 0.242194 * (year - 1980) as f64 - ((year - 1980) / 4) as f64).floor() as u32
}

fn autumnal_equinox_day(year: i32) -> u32 {
    (23.2488 + 0.242194 * (year - 1980) as f64 - ((year - 1980) / 4) as f64).floor() as u32
}

fn nth_weekday_of_month(year: i32, month: u32, weekday: Weekday, n: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid first-of-month date");
    let first_w = first.weekday().num_days_from_monday() as i32;
    let target_w = weekday.num_days_from_monday() as i32;
    let offset = (7 + target_w - first_w) % 7;
    let day = 1 + offset as u32 + 7 * (n - 1);
    NaiveDate::from_ymd_opt(year, month, day).expect("valid nth weekday date")
}

fn last_weekday_of_month(year: i32, month: u32, weekday: Weekday) -> NaiveDate {
    let dim = days_in_month(year, month);
    let last = NaiveDate::from_ymd_opt(year, month, dim).expect("valid last-of-month date");
    let last_w = last.weekday().num_days_from_monday() as i32;
    let target_w = weekday.num_days_from_monday() as i32;
    let offset = (7 + last_w - target_w) % 7;
    last - Duration::days(offset as i64)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_calendars_match_quantlib_reference_days() {
        // Reference dates from well-known QuantLib calendars.
        assert!(Calendar::nyc().is_holiday(NaiveDate::from_ymd_opt(2024, 7, 4).unwrap()));
        assert!(Calendar::london().is_holiday(NaiveDate::from_ymd_opt(2024, 3, 29).unwrap()));
        assert!(Calendar::target().is_holiday(NaiveDate::from_ymd_opt(2024, 4, 1).unwrap()));
        assert!(Calendar::tokyo().is_holiday(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()));
        assert!(Calendar::sydney().is_holiday(NaiveDate::from_ymd_opt(2024, 12, 25).unwrap()));
        assert!(Calendar::hong_kong().is_holiday(NaiveDate::from_ymd_opt(2024, 10, 1).unwrap()));
        assert!(Calendar::singapore().is_holiday(NaiveDate::from_ymd_opt(2024, 8, 9).unwrap()));
    }

    #[test]
    fn joint_calendar_uses_union_of_holidays() {
        let joint = Calendar::joint(vec![Calendar::nyc(), Calendar::london()]);
        assert!(joint.is_holiday(NaiveDate::from_ymd_opt(2024, 7, 4).unwrap())); // US only
        assert!(joint.is_holiday(NaiveDate::from_ymd_opt(2024, 8, 26).unwrap())); // UK only
        assert!(joint.is_business_day(NaiveDate::from_ymd_opt(2024, 7, 2).unwrap()));
    }

    #[test]
    fn custom_calendar_supports_weekends_and_overrides() {
        let mut custom = CustomCalendar::with_holidays(
            WeekendConvention::FridaySaturday,
            [NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()],
        );
        custom.add_business_day_override(NaiveDate::from_ymd_opt(2026, 1, 2).unwrap()); // Friday
        let calendar = Calendar::custom(custom);

        assert!(calendar.is_holiday(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()));
        assert!(calendar.is_business_day(NaiveDate::from_ymd_opt(2026, 1, 2).unwrap()));
        assert!(calendar.is_holiday(NaiveDate::from_ymd_opt(2026, 1, 3).unwrap())); // Saturday
    }

    #[test]
    fn business_day_adjustments_cover_all_conventions() {
        let calendar = Calendar::weekends_only();
        let saturday = NaiveDate::from_ymd_opt(2026, 1, 31).unwrap();

        assert_eq!(
            adjust_business_day(saturday, BusinessDayConvention::Following, &calendar),
            NaiveDate::from_ymd_opt(2026, 2, 2).unwrap()
        );
        assert_eq!(
            adjust_business_day(
                saturday,
                BusinessDayConvention::ModifiedFollowing,
                &calendar
            ),
            NaiveDate::from_ymd_opt(2026, 1, 30).unwrap()
        );
        assert_eq!(
            adjust_business_day(saturday, BusinessDayConvention::Preceding, &calendar),
            NaiveDate::from_ymd_opt(2026, 1, 30).unwrap()
        );
        assert_eq!(
            adjust_business_day(
                saturday,
                BusinessDayConvention::ModifiedPreceding,
                &calendar
            ),
            NaiveDate::from_ymd_opt(2026, 1, 30).unwrap()
        );
        assert_eq!(
            adjust_business_day(saturday, BusinessDayConvention::Unadjusted, &calendar),
            saturday
        );
        assert_eq!(
            adjust_business_day(saturday, BusinessDayConvention::Nearest, &calendar),
            NaiveDate::from_ymd_opt(2026, 1, 30).unwrap()
        );
    }

    #[test]
    fn business_day_arithmetic_and_year_fraction_work() {
        let calendar = Calendar::weekends_only();
        let d = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap(); // Friday
        let plus_two = add_business_days(d, 2, &calendar);
        let minus_two = subtract_business_days(plus_two, 2, &calendar);

        assert_eq!(plus_two, NaiveDate::from_ymd_opt(2026, 1, 6).unwrap());
        assert_eq!(minus_two, d);

        let count = business_day_count(d, NaiveDate::from_ymd_opt(2026, 1, 9).unwrap(), &calendar);
        assert_eq!(count, 5);
        assert!(
            (year_fraction_business_252(
                d,
                NaiveDate::from_ymd_opt(2026, 1, 9).unwrap(),
                &calendar
            ) - 5.0 / 252.0)
                .abs()
                <= 1.0e-14
        );
    }

    #[test]
    fn schedule_generation_supports_stub_and_roll_conventions() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 7, 20).unwrap();

        let cfg_short = ScheduleConfig {
            calendar: Calendar::weekends_only(),
            business_day_convention: BusinessDayConvention::Unadjusted,
            stub_convention: StubConvention::ShortBack,
            roll_convention: RollConvention::EndOfMonth,
        };
        let short = generate_schedule_with_config(start, end, Frequency::Quarterly, &cfg_short);
        assert_eq!(
            short,
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
                NaiveDate::from_ymd_opt(2024, 4, 30).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 20).unwrap(),
            ]
        );

        let cfg_long_back = ScheduleConfig {
            stub_convention: StubConvention::LongBack,
            ..cfg_short.clone()
        };
        let long_back =
            generate_schedule_with_config(start, end, Frequency::Quarterly, &cfg_long_back);
        assert_eq!(
            long_back,
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 20).unwrap(),
            ]
        );

        let cfg_imm = ScheduleConfig {
            calendar: Calendar::weekends_only(),
            business_day_convention: BusinessDayConvention::Unadjusted,
            stub_convention: StubConvention::ShortBack,
            roll_convention: RollConvention::Imm,
        };
        let imm = generate_schedule_with_config(
            NaiveDate::from_ymd_opt(2025, 1, 15).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 20).unwrap(),
            Frequency::Quarterly,
            &cfg_imm,
        );
        assert_eq!(
            imm,
            vec![
                NaiveDate::from_ymd_opt(2025, 1, 15).unwrap(),
                NaiveDate::from_ymd_opt(2025, 3, 19).unwrap(),
                NaiveDate::from_ymd_opt(2025, 6, 18).unwrap(),
                NaiveDate::from_ymd_opt(2025, 9, 17).unwrap(),
                NaiveDate::from_ymd_opt(2025, 12, 17).unwrap(),
                NaiveDate::from_ymd_opt(2025, 12, 20).unwrap(),
            ]
        );

        let cfg_15 = ScheduleConfig {
            roll_convention: RollConvention::Fifteenth,
            ..cfg_short
        };
        let fifteenth = generate_schedule_with_config(
            NaiveDate::from_ymd_opt(2024, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(2024, 7, 10).unwrap(),
            Frequency::Quarterly,
            &cfg_15,
        );
        assert_eq!(
            fifteenth,
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 10).unwrap(),
                NaiveDate::from_ymd_opt(2024, 4, 15).unwrap(),
                NaiveDate::from_ymd_opt(2024, 7, 10).unwrap(),
            ]
        );
    }

    #[test]
    fn imm_and_cds_date_utilities_match_conventions() {
        let d = NaiveDate::from_ymd_opt(2026, 2, 16).unwrap();
        assert_eq!(
            next_imm_date(d),
            NaiveDate::from_ymd_opt(2026, 3, 18).unwrap()
        );
        assert_eq!(
            previous_imm_date(d),
            NaiveDate::from_ymd_opt(2025, 12, 17).unwrap()
        );
        assert_eq!(
            next_cds_date(d),
            NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
        );
        assert_eq!(
            previous_cds_date(d),
            NaiveDate::from_ymd_opt(2025, 12, 20).unwrap()
        );
    }
}
