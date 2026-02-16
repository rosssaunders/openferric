//! Fixed-income primitives: day counts, yield curves, and vanilla bond analytics.

pub mod bond;
pub mod day_count;
pub mod fra;
pub mod schedule;
pub mod swap;
pub mod yield_curve;

pub use bond::FixedRateBond;
pub use day_count::{DayCountConvention, year_fraction};
pub use fra::ForwardRateAgreement;
pub use schedule::{BusinessDayConvention, Frequency, generate_schedule};
pub use swap::{InterestRateSwap, SwapBuilder};
pub use yield_curve::{YieldCurve, YieldCurveBuilder};
