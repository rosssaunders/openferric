//! Compatibility exports for schedule/calendar functionality.
//!
//! New code should import from `crate::rates::calendar`.

pub use crate::rates::calendar::{
    BusinessDayConvention, Calendar, CustomCalendar, FinancialCenter, Frequency, RollConvention,
    ScheduleConfig, StubConvention, WeekendConvention, add_business_days, add_months,
    adjust_business_day, business_day_count, generate_schedule, generate_schedule_with_config,
    is_cds_standard_date, is_imm_date, next_cds_date, next_imm_date, previous_cds_date,
    previous_imm_date, subtract_business_days, third_wednesday, year_fraction_business_252,
};
