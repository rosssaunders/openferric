//! Fixed-income primitives: day counts, yield curves, and vanilla bond analytics.

pub mod bond;
pub mod day_count;
pub mod yield_curve;

pub use bond::FixedRateBond;
pub use day_count::{DayCountConvention, year_fraction};
pub use yield_curve::{YieldCurve, YieldCurveBuilder};
