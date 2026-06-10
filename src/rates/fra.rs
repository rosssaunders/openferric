//! Module `rates::fra`.
//!
//! Implements fra abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `ForwardRateAgreement` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use chrono::NaiveDate;

use crate::rates::{DayCountConvention, YieldCurve, year_fraction};

/// Forward rate agreement over a single accrual period.
///
/// `valuation_date` anchors the curve's time axis so forward-starting FRAs
/// (e.g. a 3x6) project the forward over `[start, end]` rather than treating
/// the accrual period as starting today.
///
/// # Scope: seasoned FRAs
///
/// Valuation assumes `start_date >= valuation_date`. Seasoned FRAs (periods
/// whose rate has already fixed, i.e. `start_date < valuation_date`) are out
/// of scope: the start time is clamped to 0 while the full accrual `tau` is
/// kept, so a full-period forward is projected from today instead of using
/// the historical fixing. Fully expired FRAs (`end_date <= valuation_date`)
/// return an NPV of 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardRateAgreement {
    pub notional: f64,
    pub fixed_rate: f64,
    pub valuation_date: NaiveDate,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub day_count: DayCountConvention,
}

impl ForwardRateAgreement {
    fn period_times(&self) -> (f64, f64, f64) {
        let tau = year_fraction(self.start_date, self.end_date, self.day_count);
        let t1 = year_fraction(self.valuation_date, self.start_date, self.day_count).max(0.0);
        (t1, t1 + tau, tau)
    }

    /// Simple (money-market) forward rate over `[start, end]` implied by the curve.
    pub fn forward_rate(&self, curve: &YieldCurve) -> f64 {
        let (t1, t2, tau) = self.period_times();
        if tau <= 0.0 {
            return 0.0;
        }
        let df1 = curve.discount_factor(t1);
        let df2 = curve.discount_factor(t2);
        (df1 / df2 - 1.0) / tau
    }

    /// FRA PV: (forward - fixed) accrued over the period, discounted from period end.
    ///
    /// Assumes `start_date >= valuation_date` (see the type-level note on
    /// seasoned FRAs). Returns 0 once the accrual period has fully expired.
    pub fn npv(&self, curve: &YieldCurve) -> f64 {
        if self.end_date <= self.valuation_date {
            return 0.0;
        }
        let (_, t2, tau) = self.period_times();
        if tau <= 0.0 {
            return 0.0;
        }

        let fwd = self.forward_rate(curve);
        let df = curve.discount_factor(t2);
        self.notional * (fwd - self.fixed_rate) * tau * df
    }
}
