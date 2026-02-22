//! Rates analytics for Fra.
//!
//! Module openferric::rates::fra contains pricing and conventions for fixed-income instruments.

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
