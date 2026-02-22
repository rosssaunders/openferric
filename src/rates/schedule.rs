//! Module `rates::schedule`.
//!
//! Implements schedule abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.

pub use crate::rates::calendar::{
    BusinessDayConvention, Calendar, CustomCalendar, FinancialCenter, Frequency, RollConvention,
    ScheduleConfig, StubConvention, WeekendConvention, add_business_days, add_months,
    adjust_business_day, business_day_count, generate_schedule, generate_schedule_with_config,
    is_cds_standard_date, is_imm_date, next_cds_date, next_imm_date, previous_cds_date,
    previous_imm_date, subtract_business_days, third_wednesday, year_fraction_business_252,
};
