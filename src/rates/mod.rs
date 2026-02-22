//! Module `rates::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.

pub mod adjustments;
pub mod bond;
pub mod calendar;
pub mod capfloor;
pub mod cms;
pub mod day_count;
pub mod fra;
pub mod futures;
pub mod inflation;
pub mod multi_curve;
pub mod ois;
pub mod schedule;
pub mod swap;
pub mod swaption;
pub mod xccy_swap;
pub mod yield_curve;

pub use adjustments::{
    cms_convexity_adjustment, cms_rate_in_arrears, forward_rate_from_futures,
    futures_forward_convexity_adjustment, futures_rate_from_forward, quanto_adjusted_drift,
    quanto_drift_adjustment, timing_adjusted_rate, timing_adjustment_amount,
};
pub use bond::FixedRateBond;
pub use calendar::{
    BusinessDayConvention, Calendar, CustomCalendar, FinancialCenter, Frequency, RollConvention,
    ScheduleConfig, StubConvention, WeekendConvention, add_business_days, add_months,
    adjust_business_day, business_day_count, generate_schedule, generate_schedule_with_config,
    is_cds_standard_date, is_imm_date, next_cds_date, next_imm_date, previous_cds_date,
    previous_imm_date, subtract_business_days, third_wednesday, year_fraction_business_252,
};
pub use capfloor::CapFloor;
pub use day_count::{DayCountConvention, year_fraction};
pub use fra::ForwardRateAgreement;
pub use futures::{Future, InterestRateFutureQuote};
pub use inflation::{
    InflationCurve, InflationCurveBuilder, InflationIndexedBond, YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
};
pub use ois::{BasisSwap, OvernightIndexSwap};
pub use swap::{InterestRateSwap, SwapBuilder};
pub use swaption::Swaption;
pub use xccy_swap::XccySwap;
pub use yield_curve::{
    YieldCurve, YieldCurveBuilder, YieldCurveInterpolationMethod, YieldCurveInterpolationSettings,
};
