//! Fixed-income primitives: day counts, yield curves, and vanilla bond analytics.

pub mod adjustments;
pub mod bond;
pub mod capfloor;
pub mod cms;
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

pub use adjustments::{
    cms_convexity_adjustment, cms_rate_in_arrears, forward_rate_from_futures,
    futures_forward_convexity_adjustment, futures_rate_from_forward, quanto_adjusted_drift,
    quanto_drift_adjustment, timing_adjusted_rate, timing_adjustment_amount,
};
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
