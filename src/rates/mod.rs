//! Fixed-income primitives: day counts, yield curves, and vanilla bond analytics.

pub mod bond;
pub mod capfloor;
pub mod day_count;
pub mod fra;
pub mod futures;
pub mod inflation;
pub mod ois;
pub mod schedule;
pub mod swap;
pub mod swaption;
pub mod xccy_swap;
pub mod yield_curve;

pub use bond::FixedRateBond;
pub use capfloor::CapFloor;
pub use day_count::{DayCountConvention, year_fraction};
pub use fra::ForwardRateAgreement;
pub use futures::{Future, InterestRateFutureQuote};
pub use inflation::{
    InflationCurve, InflationCurveBuilder, InflationIndexedBond, YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
};
pub use ois::{BasisSwap, OvernightIndexSwap};
pub use schedule::{BusinessDayConvention, Frequency, generate_schedule};
pub use swap::{InterestRateSwap, SwapBuilder};
pub use swaption::Swaption;
pub use xccy_swap::XccySwap;
pub use yield_curve::{YieldCurve, YieldCurveBuilder};
