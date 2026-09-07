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
/// Use `npv_with_fixing` for seasoned trades and explicit advance/arrears
/// settlement. `npv` is period-end settlement without a historical fixing;
/// it returns NaN when a required fixing is missing, never inventing one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardRateAgreement {
    pub notional: f64,
    pub fixed_rate: f64,
    pub valuation_date: NaiveDate,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub day_count: DayCountConvention,
    /// Day count of the curve time axis, independent of coupon accrual.
    pub curve_day_count: DayCountConvention,
}

impl ForwardRateAgreement {
    fn period_times(&self) -> (f64, f64, f64) {
        let tau = year_fraction(self.start_date, self.end_date, self.day_count);
        let start_time = year_fraction(self.valuation_date, self.start_date, self.curve_day_count);
        let end_time = year_fraction(self.valuation_date, self.end_date, self.curve_day_count);
        (start_time, end_time, tau)
    }

    /// Simple (money-market) forward rate over `[start, end]` implied by the curve.
    pub fn forward_rate(&self, curve: &YieldCurve) -> f64 {
        if self.start_date < self.valuation_date {
            return f64::NAN;
        }
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
        self.npv_with_fixing(curve, None, false, false)
            .unwrap_or(f64::NAN)
    }

    /// Values a known fixing or a future reset. Advance settlement pays
    /// `N*tau*(fixing-K)/(1+tau*fixing)` at start; arrears pays the numerator
    /// at end. `include_settlement_date` distinguishes before/after payment
    /// on valuation date. Fixings after valuation are rejected.
    pub fn npv_with_fixing(
        &self,
        curve: &YieldCurve,
        fixing: Option<f64>,
        settle_in_advance: bool,
        include_settlement_date: bool,
    ) -> Result<f64, String> {
        if self.end_date < self.start_date
            || !self.notional.is_finite()
            || !self.fixed_rate.is_finite()
            || fixing.is_some_and(|value| !value.is_finite())
            || (self.start_date > self.valuation_date && fixing.is_some())
        {
            return Err("invalid FRA dates, amount or fixing".into());
        }
        let (start, end, accrual) = self.period_times();
        if accrual == 0.0 {
            return Ok(0.0);
        }
        let settlement = if settle_in_advance {
            self.start_date
        } else {
            self.end_date
        };
        if settlement < self.valuation_date
            || (settlement == self.valuation_date && !include_settlement_date)
        {
            return Ok(0.0);
        }
        let rate = match fixing {
            Some(rate) => rate,
            None if self.start_date < self.valuation_date => {
                return Err("historical FRA fixing is required".into());
            }
            None => self.forward_rate(curve),
        };
        let denominator = if settle_in_advance {
            1.0 + accrual * rate
        } else {
            1.0
        };
        if !rate.is_finite() || denominator <= 0.0 {
            return Err("invalid FRA settlement rate".into());
        }
        let value = self.notional * accrual * (rate - self.fixed_rate) / denominator
            * curve.discount_factor(if settle_in_advance { start } else { end });
        if !value.is_finite() {
            return Err("non-finite FRA value".into());
        }
        Ok(value)
    }
}
