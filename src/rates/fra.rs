//! Forward Rate Agreement (FRA) payoff and valuation helper.
//!
//! [`ForwardRateAgreement`] computes implied forward rate from the curve and discounted NPV
//! over a single accrual period.
//! Reference: Hull (2018), FRAs and forward-rate agreements chapter.
//! The implementation is intentionally compact and assumes deterministic curves/day-count conversion.
//! Use this module for simple FRA valuation components or regression tests against larger swap stacks.

use chrono::NaiveDate;

use crate::rates::{DayCountConvention, YieldCurve, year_fraction};

/// Forward rate agreement over a single accrual period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardRateAgreement {
    pub notional: f64,
    pub fixed_rate: f64,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub day_count: DayCountConvention,
}

impl ForwardRateAgreement {
    /// Continuously-compounded forward rate implied by the curve.
    pub fn forward_rate(&self, curve: &YieldCurve) -> f64 {
        let tau = year_fraction(self.start_date, self.end_date, self.day_count);
        if tau <= 0.0 {
            return 0.0;
        }
        curve.forward_rate(0.0, tau)
    }

    /// FRA PV discounted to period end.
    pub fn npv(&self, curve: &YieldCurve) -> f64 {
        let tau = year_fraction(self.start_date, self.end_date, self.day_count);
        if tau <= 0.0 {
            return 0.0;
        }

        let fwd = self.forward_rate(curve);
        let df = curve.discount_factor(tau);
        self.notional * (fwd - self.fixed_rate) * tau * df
    }
}
