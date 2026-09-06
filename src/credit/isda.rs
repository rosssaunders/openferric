//! Module `credit::isda`.
//!
//! Implements isda workflows with concrete routines such as `price_midpoint_flat`, `price_isda_flat`, `hazard_from_par_spread`, `step_in_date`.
//!
//! References: Hull (11th ed.) Ch. 24-25, O'Kane (2008) Ch. 3, representative cashflow identities as in Eq. (24.7) and Eq. (25.5).
//!
//! Key types and purpose: `ProtectionSide`, `CdsDateRule`, `DatedCds`, `IsdaConventions`, `CdsPriceResult` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use these routines for CDS/tranche and survival-curve workflows; consider structural credit models when capital-structure dynamics are required explicitly.
use chrono::{Datelike, Duration, NaiveDate};

use crate::rates::{
    BusinessDayConvention, Calendar, DayCountConvention, add_business_days, adjust_business_day,
    next_cds_date, previous_cds_date, year_fraction,
};

/// Protection side of a CDS trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionSide {
    Buyer,
    Seller,
}

impl ProtectionSide {
    fn sign(self) -> f64 {
        match self {
            Self::Buyer => 1.0,
            Self::Seller => -1.0,
        }
    }
}

/// Date-generation style for CDS schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdsDateRule {
    /// Legacy CDS schedule on the 20th of IMM months.
    TwentiethImm,
    /// Standard quarterly IMM schedule.
    QuarterlyImm,
}

/// Dated CDS contract for midpoint/ISDA-style pricing.
#[derive(Debug, Clone, PartialEq)]
pub struct DatedCds {
    pub side: ProtectionSide,
    pub notional: f64,
    /// Running spread in decimal (e.g. 0.01 for 100 bps).
    pub running_spread: f64,
    pub recovery_rate: f64,
    pub issue_date: NaiveDate,
    pub maturity_date: NaiveDate,
    /// Coupon interval in months (typically 6 for legacy, 3 for standard).
    pub coupon_interval_months: i32,
    pub date_rule: CdsDateRule,
}

impl DatedCds {
    fn is_valid(&self) -> bool {
        self.notional > 0.0
            && self.running_spread >= 0.0
            && (0.0..1.0).contains(&self.recovery_rate)
            && self.maturity_date > self.issue_date
            && self.coupon_interval_months > 0
            && 12 % self.coupon_interval_months == 0
    }

    /// Builds a standard quarterly IMM CDS from trade date and tenor in years.
    pub fn standard_imm(
        side: ProtectionSide,
        trade_date: NaiveDate,
        tenor_years: i32,
        notional: f64,
        running_spread: f64,
        recovery_rate: f64,
    ) -> Self {
        Self::standard_imm_with_calendar(
            side,
            trade_date,
            tenor_years,
            notional,
            running_spread,
            recovery_rate,
            &Calendar::weekends_only(),
        )
    }

    /// Builds a standard quarterly IMM CDS alongside an explicit contract
    /// calendar.
    ///
    /// The schedule anchor follows `DateGeneration::CDS`: it starts at the
    /// previous unadjusted IMM twentieth relative to trade date, except that a
    /// roll whose Following-adjusted date is after trade date belongs to the
    /// new coupon period and causes the preceding quarterly roll to be added.
    /// Step-in remains T+1 calendar day and maturity is selected from it.
    pub fn standard_imm_with_calendar(
        side: ProtectionSide,
        trade_date: NaiveDate,
        tenor_years: i32,
        notional: f64,
        running_spread: f64,
        recovery_rate: f64,
        calendar: &Calendar,
    ) -> Self {
        let step_in = step_in_date(trade_date);
        let previous_roll = previous_imm_twentieth(trade_date);
        let start =
            if adjust_business_day(previous_roll, BusinessDayConvention::Following, calendar)
                > trade_date
            {
                add_months(previous_roll, -3)
            } else {
                previous_roll
            };
        let raw_maturity = add_months(step_in, 12 * tenor_years);
        let maturity = next_imm_twentieth(raw_maturity);

        Self {
            side,
            notional,
            running_spread,
            recovery_rate,
            issue_date: start,
            maturity_date: maturity,
            coupon_interval_months: 3,
            date_rule: CdsDateRule::QuarterlyImm,
        }
    }
}

/// ISDA market conventions used for valuation alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsdaConventions {
    /// Requested step-in offset in calendar days from valuation date.
    ///
    /// The standard ISDA engine path floors this at T+1, matching QuantLib's
    /// effective-protection-start rule. Midpoint and legacy paths use the
    /// requested offset exactly.
    pub step_in_days: usize,
    /// Cash-settlement date offset in business days from valuation date.
    /// Offsets beyond the representable date range saturate at
    /// [`NaiveDate::MAX`].
    pub cash_settle_days: usize,
}

impl Default for IsdaConventions {
    fn default() -> Self {
        Self {
            step_in_days: 1,
            cash_settle_days: 3,
        }
    }
}

/// Valuation output for dated CDS pricing.
#[derive(Debug, Clone, PartialEq)]
pub struct CdsPriceResult {
    /// NPV after the accrued-premium/rebate adjustment, from the trade side.
    pub clean_npv: f64,
    /// Protection-minus-premium leg NPV before the accrued-premium adjustment.
    pub dirty_npv: f64,
    pub premium_leg_pv: f64,
    pub protection_leg_pv: f64,
    pub accrued_premium_pv: f64,
    pub fair_spread: f64,
    pub step_in_date: NaiveDate,
    pub cash_settle_date: NaiveDate,
}

/// Midpoint-style CDS valuation with dated schedule and accrual-on-default.
pub fn price_midpoint_flat(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
) -> CdsPriceResult {
    price_midpoint_flat_with_calendar(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        &Calendar::weekends_only(),
    )
}

/// Midpoint-style CDS valuation with an explicit business calendar.
///
/// Step-in is a calendar-day lag; `calendar` controls only the cash-settlement
/// business-day lag. This compatibility method retains unadjusted coupon
/// boundaries and Act/360 curve times.
pub fn price_midpoint_flat_with_calendar(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
    calendar: &Calendar,
) -> CdsPriceResult {
    price_flat_with_model(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        calendar,
        PricingModel::Midpoint,
    )
}

/// Standard CDS valuation under flat continuously compounded hazard and
/// discount rates.
///
/// This applies the standard dated cashflow conventions used by QuantLib's
/// ISDA engine: T+1 calendar-day step-in, Following-adjusted regular accrual
/// boundaries, unadjusted contractual maturity, Following-adjusted payment
/// dates, Act/365F curve times, final-day-inclusive Act/360 for the last coupon
/// of a multi-period schedule, half-day default accrual bias, and the
/// accrued-premium rebate paid at cash settlement.
pub fn price_isda_flat(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
) -> CdsPriceResult {
    price_isda_flat_with_calendar(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        &Calendar::weekends_only(),
    )
}

/// Standard flat-rate CDS valuation with an explicit contract calendar.
///
/// `calendar` controls regular accrual-date adjustment, every coupon payment
/// date, and the T+cash-settlement-business-day lag. Step-in remains a calendar
/// day convention and is deliberately not adjusted through holidays. A zero
/// cash-settlement lag still Following-adjusts a non-business valuation date;
/// a rebate settling exactly on a business valuation date is treated as
/// already occurred.
pub fn price_isda_flat_with_calendar(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
    calendar: &Calendar,
) -> CdsPriceResult {
    price_isda_standard_flat_with_calendar(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        calendar,
    )
}

/// Legacy year-fraction flat-integral CDS calculation.
///
/// This preserves the earlier analytic methodology for callers that require
/// it: all coupon boundaries are unadjusted, curve times use Act/360, and PVs
/// are reported at cash settlement. New standard CDS work should use
/// [`price_isda_flat`] or [`price_isda_flat_with_calendar`].
pub fn price_isda_flat_legacy_analytic(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
) -> CdsPriceResult {
    price_flat_with_model(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        &Calendar::weekends_only(),
        PricingModel::LegacyFlatIntegral,
    )
}

/// Converts a par running spread to a flat hazard intensity under flat-LGD approximation.
pub fn hazard_from_par_spread(par_spread: f64, recovery_rate: f64) -> f64 {
    if !(0.0..1.0).contains(&recovery_rate) {
        return 0.0;
    }
    (par_spread.max(0.0) / (1.0 - recovery_rate)).max(0.0)
}

/// Standard CDS step-in date (T+1 calendar day).
pub fn step_in_date(valuation_date: NaiveDate) -> NaiveDate {
    valuation_date + Duration::days(1)
}

/// Standard CDS step-in date when a contract calendar is also in scope.
///
/// Step-in is T+1 calendar day, so `calendar` intentionally has no effect.
pub fn step_in_date_with_calendar(valuation_date: NaiveDate, _calendar: &Calendar) -> NaiveDate {
    step_in_date(valuation_date)
}

/// Standard CDS cash-settlement date (T+3 business days).
pub fn cash_settle_date(valuation_date: NaiveDate) -> NaiveDate {
    cash_settle_date_with_calendar(valuation_date, &Calendar::weekends_only())
}

/// CDS cash-settlement date (T+3 business days) under an explicit calendar.
pub fn cash_settle_date_with_calendar(valuation_date: NaiveDate, calendar: &Calendar) -> NaiveDate {
    add_business_days(valuation_date, 3, calendar)
}

/// Previous quarterly IMM date (20th of Mar/Jun/Sep/Dec) on or before `date`.
pub fn previous_imm_twentieth(date: NaiveDate) -> NaiveDate {
    previous_cds_date(date)
}

/// Next quarterly IMM date (20th of Mar/Jun/Sep/Dec) on or after `date`.
pub fn next_imm_twentieth(date: NaiveDate) -> NaiveDate {
    next_cds_date(date)
}

/// Generates a coupon-boundary schedule including one boundary before issue date.
///
/// Returned vector is strictly increasing and starts with a boundary that is
/// less than or equal to `issue_date`.
pub fn generate_imm_schedule(
    issue_date: NaiveDate,
    maturity_date: NaiveDate,
    interval_months: i32,
    _rule: CdsDateRule,
) -> Vec<NaiveDate> {
    if maturity_date <= issue_date || interval_months <= 0 {
        return vec![issue_date];
    }

    let first = next_imm_twentieth(issue_date + Duration::days(1));
    let mut prev = add_months(first, -interval_months);

    // Keep previous boundary not too far from issue date to avoid giant front stubs.
    while prev > issue_date {
        prev = add_months(prev, -interval_months);
    }

    let end = next_imm_twentieth(maturity_date);
    let mut dates = vec![prev];
    let mut d = first;
    while d < end {
        dates.push(d);
        d = add_months(d, interval_months);
    }
    dates.push(end);
    dates
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StandardCdsPeriod {
    accrual_start: NaiveDate,
    accrual_end: NaiveDate,
    payment_date: NaiveDate,
    is_final: bool,
}

fn standard_cds_periods(cds: &DatedCds, calendar: &Calendar) -> Vec<StandardCdsPeriod> {
    let unadjusted = generate_imm_schedule(
        cds.issue_date,
        cds.maturity_date,
        cds.coupon_interval_months,
        cds.date_rule,
    );
    let final_index = unadjusted.len().saturating_sub(1);
    let accrual_dates = unadjusted
        .iter()
        .enumerate()
        .map(|(index, date)| {
            if index == final_index {
                *date
            } else {
                adjust_business_day(*date, BusinessDayConvention::Following, calendar)
            }
        })
        .collect::<Vec<_>>();

    unadjusted
        .windows(2)
        .enumerate()
        .map(|(index, window)| StandardCdsPeriod {
            accrual_start: accrual_dates[index],
            accrual_end: accrual_dates[index + 1],
            payment_date: adjust_business_day(
                window[1],
                BusinessDayConvention::Following,
                calendar,
            ),
            is_final: index + 1 == final_index,
        })
        .collect()
}

fn price_isda_standard_flat_with_calendar(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
    calendar: &Calendar,
) -> CdsPriceResult {
    let minimum_step_in = advance_calendar_days(valuation_date, 1);
    let step_in =
        advance_calendar_days(valuation_date, conventions.step_in_days).max(minimum_step_in);
    let cash_settle =
        standard_cash_settle_date(valuation_date, conventions.cash_settle_days, calendar);

    if !cds.is_valid() {
        return zero_cds_result(step_in, cash_settle);
    }

    let hazard = hazard_rate.max(0.0);
    let periods = standard_cds_periods(cds, calendar);
    let final_coupon_includes_maturity = periods.len() > 1;
    let mut scheduled_coupon_annuity = 0.0;
    let mut default_accrual_annuity = 0.0;

    for period in &periods {
        if period.payment_date > step_in {
            let mut accrual_days = (period.accrual_end - period.accrual_start).num_days();
            if period.is_final && final_coupon_includes_maturity {
                // QuantLib's standard CDS leg uses Actual/360 including the
                // contractual maturity day for the final coupon only when the
                // schedule contains more than one coupon period.
                accrual_days += 1;
            }
            let accrual = accrual_days as f64 / 360.0;
            let discount_time = act365_time(valuation_date, period.payment_date);
            let survival_time =
                act365_time(valuation_date, period.payment_date - Duration::days(1));
            scheduled_coupon_annuity +=
                accrual * (-discount_rate * discount_time).exp() * (-hazard * survival_time).exp();
        }

        if period.accrual_end <= step_in {
            continue;
        }

        let default_start = period.accrual_start.max(step_in) - Duration::days(1);
        let default_end = period.payment_date - Duration::days(1);
        if default_end <= default_start {
            continue;
        }

        let t0 = act365_time(valuation_date, default_start);
        let t1 = act365_time(valuation_date, default_end);
        let accrual_origin =
            act365_time(valuation_date, period.accrual_start - Duration::days(1)) - 1.0 / 730.0;
        default_accrual_annuity +=
            quantlib_flat_default_accrual(t0, t1, accrual_origin, discount_rate, hazard) * 365.0
                / 360.0;
    }

    let protection_start = step_in - Duration::days(1);
    let protection_term = flat_default_leg_integral(
        act365_time(valuation_date, protection_start),
        act365_time(valuation_date, cds.maturity_date),
        discount_rate,
        hazard,
    );

    // Standard CDS accrued rebate is determined at trade+1, independently of
    // a later requested protection start. Like QuantLib's default
    // includeSettlementDateFlows=false setting, a rebate settling on the
    // valuation date has already occurred and contributes no PV or fair-spread
    // annuity.
    let rebate_reference = minimum_step_in;
    let accrued_fraction = standard_accrued_fraction(&periods, rebate_reference);
    let accrued_rebate_annuity = if cash_settle > valuation_date {
        let settlement_discount = (-discount_rate * act365_time(valuation_date, cash_settle)).exp();
        accrued_fraction * settlement_discount
    } else {
        0.0
    };
    let risky_annuity = scheduled_coupon_annuity + default_accrual_annuity;

    let premium_leg_pv = cds.notional * cds.running_spread * risky_annuity;
    let protection_leg_pv = cds.notional * (1.0 - cds.recovery_rate) * protection_term;
    let accrued_premium_pv = cds.notional * cds.running_spread * accrued_rebate_annuity;
    let dirty_npv_buyer = protection_leg_pv - premium_leg_pv;
    let clean_npv_buyer = dirty_npv_buyer + accrued_premium_pv;
    let fair_annuity = risky_annuity - accrued_rebate_annuity;
    let fair_spread = if fair_annuity.abs() <= 1.0e-14 {
        0.0
    } else {
        ((1.0 - cds.recovery_rate) * protection_term / fair_annuity).max(0.0)
    };

    let sign = cds.side.sign();
    CdsPriceResult {
        clean_npv: sign * clean_npv_buyer,
        dirty_npv: sign * dirty_npv_buyer,
        premium_leg_pv,
        protection_leg_pv,
        accrued_premium_pv,
        fair_spread,
        step_in_date: step_in,
        cash_settle_date: cash_settle,
    }
}

fn zero_cds_result(step_in_date: NaiveDate, cash_settle_date: NaiveDate) -> CdsPriceResult {
    CdsPriceResult {
        clean_npv: 0.0,
        dirty_npv: 0.0,
        premium_leg_pv: 0.0,
        protection_leg_pv: 0.0,
        accrued_premium_pv: 0.0,
        fair_spread: 0.0,
        step_in_date,
        cash_settle_date,
    }
}

fn standard_accrued_fraction(periods: &[StandardCdsPeriod], step_in: NaiveDate) -> f64 {
    let final_coupon_includes_maturity = periods.len() > 1;
    for period in periods {
        if step_in > period.payment_date {
            continue;
        }
        if step_in == period.payment_date {
            return if period.is_final && final_coupon_includes_maturity {
                ((period.accrual_end - period.accrual_start).num_days() + 1) as f64 / 360.0
            } else {
                0.0
            };
        }
        if step_in <= period.accrual_start {
            return 0.0;
        }
        let accrued_end = step_in.min(period.accrual_end);
        return (accrued_end - period.accrual_start).num_days() as f64 / 360.0;
    }
    0.0
}

fn act365_time(reference: NaiveDate, date: NaiveDate) -> f64 {
    (date - reference).num_days() as f64 / 365.0
}

fn flat_default_leg_integral(t0: f64, t1: f64, rate: f64, hazard: f64) -> f64 {
    if t1 <= t0 || hazard <= 0.0 {
        return 0.0;
    }
    let combined = rate + hazard;
    let dt = t1 - t0;
    if combined.abs() <= 1.0e-12 {
        hazard * dt
    } else {
        hazard / combined * (-combined * t0).exp() * -(-combined * dt).exp_m1()
    }
}

fn quantlib_flat_default_accrual(
    t0: f64,
    t1: f64,
    accrual_origin: f64,
    rate: f64,
    hazard: f64,
) -> f64 {
    if t1 <= t0 || hazard <= 0.0 {
        return 0.0;
    }

    let dt = t1 - t0;
    let fhat = rate * dt;
    let hhat = hazard * dt;
    let combined_hat = fhat + hhat;
    let p0q0 = (-(rate + hazard) * t0).exp();

    // Match QuantLib's `IsdaCdsEngine::Taylor` branch. It avoids cancellation
    // for very short intervals and is also the documented standard-engine
    // numerical fix used by the external fixture.
    if combined_hat.abs() < 1.0e-4 {
        let combined2 = combined_hat * combined_hat;
        hhat * p0q0
            * ((t0 - accrual_origin)
                * (1.0 - 0.5 * combined_hat + combined2 / 6.0 - combined2 * combined_hat / 24.0)
                + dt * (0.5 - combined_hat / 3.0 + combined2 / 8.0
                    - combined2 * combined_hat / 30.0))
    } else {
        let p1q1 = (-(rate + hazard) * t1).exp();
        (hhat / combined_hat)
            * (dt * ((p0q0 - p1q1) / combined_hat - p1q1) + (t0 - accrual_origin) * (p0q0 - p1q1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PricingModel {
    Midpoint,
    LegacyFlatIntegral,
}

fn price_flat_with_model(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
    calendar: &Calendar,
    model: PricingModel,
) -> CdsPriceResult {
    let step_in = advance_calendar_days(valuation_date, conventions.step_in_days);
    let cash_settle = advance_business_days(valuation_date, conventions.cash_settle_days, calendar);

    if !cds.is_valid() {
        return CdsPriceResult {
            clean_npv: 0.0,
            dirty_npv: 0.0,
            premium_leg_pv: 0.0,
            protection_leg_pv: 0.0,
            accrued_premium_pv: 0.0,
            fair_spread: 0.0,
            step_in_date: step_in,
            cash_settle_date: cash_settle,
        };
    }

    let h = hazard_rate.max(0.0);
    let r = discount_rate;

    let schedule = generate_imm_schedule(
        cds.issue_date,
        cds.maturity_date,
        cds.coupon_interval_months,
        cds.date_rule,
    );

    let mut coupon_annuity = 0.0;
    let mut accrual_on_default = 0.0;
    let mut protection_term = 0.0;

    let mut accrued_fraction = 0.0;

    for window in schedule.windows(2) {
        let period_start = window[0];
        let period_end = window[1];

        if period_end <= step_in {
            continue;
        }

        let accrual = year_fraction(period_start, period_end, DayCountConvention::Act360);
        let t_pay = year_fraction(valuation_date, period_end, DayCountConvention::Act360);
        let survival_pay = (-h * t_pay.max(0.0)).exp();
        let discount_pay = (-r * t_pay.max(0.0)).exp();

        coupon_annuity += accrual * discount_pay * survival_pay;

        if period_start < step_in && step_in <= period_end {
            accrued_fraction = year_fraction(period_start, step_in, DayCountConvention::Act360);
        }

        let default_start = period_start.max(step_in);
        if default_start >= period_end {
            continue;
        }

        let t1 = year_fraction(valuation_date, default_start, DayCountConvention::Act360).max(0.0);
        let t2 = year_fraction(valuation_date, period_end, DayCountConvention::Act360).max(0.0);

        // ISDA standard model convention: on default the protection buyer owes
        // premium accrued from the period start, even though survival-weighted
        // integration only starts at the step-in date. For the stub period
        // containing step-in this shifts the accrual origin back by
        // `accrual_offset = period_start -> default_start`; for all later
        // periods the offset is zero.
        let accrual_offset = year_fraction(period_start, default_start, DayCountConvention::Act360);

        match model {
            PricingModel::Midpoint => {
                let default_prob = ((-h * t1).exp() - (-h * t2).exp()).max(0.0);
                let t_mid = 0.5 * (t1 + t2);
                let df_mid = (-r * t_mid).exp();
                let accrual_default =
                    year_fraction(default_start, period_end, DayCountConvention::Act360);

                protection_term += df_mid * default_prob;
                accrual_on_default +=
                    (accrual_offset + 0.5 * accrual_default) * df_mid * default_prob;
            }
            PricingModel::LegacyFlatIntegral => {
                let (accrual_term, protection_piece) =
                    exact_flat_interval_terms(t1, t2, r, h, accrual_offset);
                protection_term += protection_piece;
                accrual_on_default += accrual_term;
            }
        }
    }

    let settlement_df = (-r
        * year_fraction(valuation_date, cash_settle, DayCountConvention::Act360).max(0.0))
    .exp()
    .max(1.0e-12);
    let settlement_scale = 1.0 / settlement_df;

    let premium_leg_pv = cds.notional * cds.running_spread * (coupon_annuity + accrual_on_default);
    let protection_leg_pv = cds.notional * (1.0 - cds.recovery_rate) * protection_term;
    let accrued_premium_pv = cds.notional * cds.running_spread * accrued_fraction;

    let dirty_npv_buyer = (protection_leg_pv - premium_leg_pv) * settlement_scale;
    let clean_npv_buyer = dirty_npv_buyer + accrued_premium_pv * settlement_scale;

    let risky_annuity = coupon_annuity + accrual_on_default;
    let fair_spread = if (risky_annuity - accrued_fraction).abs() <= 1.0e-14 {
        0.0
    } else {
        ((1.0 - cds.recovery_rate) * protection_term / (risky_annuity - accrued_fraction)).max(0.0)
    };

    let sign = cds.side.sign();
    CdsPriceResult {
        clean_npv: sign * clean_npv_buyer,
        dirty_npv: sign * dirty_npv_buyer,
        premium_leg_pv: premium_leg_pv * settlement_scale,
        protection_leg_pv: protection_leg_pv * settlement_scale,
        accrued_premium_pv: accrued_premium_pv * settlement_scale,
        fair_spread,
        step_in_date: step_in,
        cash_settle_date: cash_settle,
    }
}

fn exact_flat_interval_terms(t1: f64, t2: f64, r: f64, h: f64, accrual_offset: f64) -> (f64, f64) {
    if t2 <= t1 || h <= 0.0 {
        return (0.0, 0.0);
    }

    let combined_rate = r + h;
    let interval = t2 - t1;
    let exponent = combined_rate * interval;
    let (zeroth_moment, first_moment) = if exponent.abs() < 1.0e-4 {
        let squared = exponent * exponent;
        (
            1.0 - exponent / 2.0 + squared / 6.0 - squared * exponent / 24.0,
            0.5 - exponent / 3.0 + squared / 8.0 - squared * exponent / 30.0,
        )
    } else {
        let decay = (-exponent).exp();
        (
            -(-exponent).exp_m1() / exponent,
            (-(-exponent).exp_m1() - exponent * decay) / (exponent * exponent),
        )
    };
    let density_at_start = h * (-combined_rate * t1).exp();
    let protection = density_at_start * interval * zeroth_moment;
    let accrual = density_at_start * interval * interval * first_moment
        + accrual_offset.max(0.0) * protection;
    (accrual, protection)
}

fn advance_business_days(date: NaiveDate, days: usize, calendar: &Calendar) -> NaiveDate {
    let remaining_calendar_days = (NaiveDate::MAX - date).num_days();
    let Ok(remaining_calendar_days) = usize::try_from(remaining_calendar_days) else {
        return NaiveDate::MAX;
    };
    if days > remaining_calendar_days {
        return NaiveDate::MAX;
    }

    let mut current = date;
    let mut left = days;
    while left > 0 {
        let Some(next) = current.succ_opt() else {
            return NaiveDate::MAX;
        };
        current = next;
        if calendar.is_business_day(current) {
            left -= 1;
        }
    }
    current
}

fn standard_cash_settle_date(date: NaiveDate, days: usize, calendar: &Calendar) -> NaiveDate {
    if days == 0 {
        // QuantLib Calendar::advance(0 Days, Following) still adjusts a
        // non-business trade date.
        adjust_business_day(date, BusinessDayConvention::Following, calendar)
    } else {
        advance_business_days(date, days, calendar)
    }
}

fn advance_calendar_days(date: NaiveDate, days: usize) -> NaiveDate {
    let days = i64::try_from(days).unwrap_or(i64::MAX);
    Duration::try_days(days)
        .and_then(|offset| date.checked_add_signed(offset))
        .unwrap_or(NaiveDate::MAX)
}

fn add_months(date: NaiveDate, months: i32) -> NaiveDate {
    let month0 = date.month0() as i32;
    let total = month0 + months;

    let mut year = date.year() + total.div_euclid(12);
    let mut month0_new = total.rem_euclid(12);
    if month0_new < 0 {
        year -= 1;
        month0_new += 12;
    }

    let month = (month0_new as u32) + 1;
    let day = date.day().min(days_in_month(year, month));
    NaiveDate::from_ymd_opt(year, month, day).expect("valid add_months result")
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => unreachable!("invalid month"),
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    #[test]
    fn flat_default_integrals_support_zero_and_negative_combined_rates() {
        for rate in [-0.5, -0.02, -0.02 + 1.0e-12, -0.02 - 1.0e-12] {
            let hazard = 0.02;
            let start = 0.25;
            let end = 1.0;
            let offset = 0.1;
            let intervals = 10_000;
            let step = (end - start) / intervals as f64;
            let mut expected_protection = 0.0;
            let mut expected_accrual = 0.0;
            for index in 0..=intervals {
                let time = start + index as f64 * step;
                let weight = if index == 0 || index == intervals {
                    1.0
                } else if index % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                let density = hazard * (-(rate + hazard) * time).exp();
                expected_protection += weight * density * step / 3.0;
                expected_accrual += weight * (time - start + offset) * density * step / 3.0;
            }
            let (accrual, protection) =
                super::exact_flat_interval_terms(start, end, rate, hazard, offset);
            assert!(
                (protection - expected_protection).abs() <= 1.0e-14,
                "rate={rate} protection={protection} expected={expected_protection}"
            );
            assert!(
                (accrual - expected_accrual).abs() <= 1.0e-14,
                "rate={rate} accrual={accrual} expected={expected_accrual}"
            );
            let standard =
                super::quantlib_flat_default_accrual(start, end, start - offset, rate, hazard);
            assert!(
                (standard - expected_accrual).abs() <= 1.0e-14,
                "rate={rate} standard={standard} expected={expected_accrual}"
            );
        }
    }

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn imm_utilities_generate_expected_dates() {
        let d = NaiveDate::from_ymd_opt(2026, 2, 16).unwrap();
        assert_eq!(
            previous_imm_twentieth(d),
            NaiveDate::from_ymd_opt(2025, 12, 20).unwrap()
        );
        assert_eq!(
            next_imm_twentieth(d),
            NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
        );

        let step_in = step_in_date(d);
        let cash_settle = cash_settle_date(d);
        assert!(step_in > d);
        assert!(cash_settle > step_in);
    }

    #[test]
    fn oversized_convention_offsets_saturate_without_wrapping() {
        let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let calendar = Calendar::weekends_only();

        assert_eq!(
            advance_calendar_days(valuation_date, usize::MAX),
            NaiveDate::MAX
        );
        assert_eq!(
            advance_business_days(valuation_date, usize::MAX, &calendar),
            NaiveDate::MAX
        );
    }

    #[test]
    fn standard_schedule_separates_accrual_maturity_and_payment_dates() {
        let trade_date = NaiveDate::from_ymd_opt(2026, 10, 2).unwrap();
        let calendar = Calendar::target();
        let cds = DatedCds::standard_imm_with_calendar(
            ProtectionSide::Buyer,
            trade_date,
            5,
            1.0,
            0.01,
            0.4,
            &calendar,
        );
        let periods = standard_cds_periods(&cds, &calendar);

        // The raw 20-Sep-2026 and 20-Dec-2026 roll dates are Sundays, so the
        // first regular accrual period is Following-adjusted on both ends.
        assert_eq!(
            periods[0].accrual_start,
            NaiveDate::from_ymd_opt(2026, 9, 21).unwrap()
        );
        assert_eq!(
            periods[0].accrual_end,
            NaiveDate::from_ymd_opt(2026, 12, 21).unwrap()
        );
        assert_eq!(periods[0].payment_date, periods[0].accrual_end);

        // Contractual maturity 20-Dec-2031 is a Saturday and remains the final
        // accrual boundary, while its cash payment follows on Monday 22-Dec.
        let final_period = periods.last().unwrap();
        assert!(final_period.is_final);
        assert_eq!(
            final_period.accrual_end,
            NaiveDate::from_ymd_opt(2031, 12, 20).unwrap()
        );
        assert_eq!(
            final_period.payment_date,
            NaiveDate::from_ymd_opt(2031, 12, 22).unwrap()
        );
    }

    #[test]
    fn standard_constructor_anchors_schedule_on_trade_date_roll_period() {
        let target = Calendar::target();

        // A trade immediately before the June roll is still in the coupon
        // period that began on 20 March, even though T+1 lands on 20 June.
        let pre_roll_trade = NaiveDate::from_ymd_opt(2026, 6, 19).unwrap();
        let pre_roll = DatedCds::standard_imm_with_calendar(
            ProtectionSide::Buyer,
            pre_roll_trade,
            5,
            1.0,
            0.01,
            0.4,
            &target,
        );
        assert_eq!(
            pre_roll.issue_date,
            NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
        );

        // 20 September is a Sunday. DateGeneration::CDS compares its
        // Following-adjusted date (Monday 21st) with trade date, so the
        // unadjusted June roll must be prepended.
        let weekend_roll_trade = NaiveDate::from_ymd_opt(2026, 9, 20).unwrap();
        let weekend_roll = DatedCds::standard_imm_with_calendar(
            ProtectionSide::Buyer,
            weekend_roll_trade,
            5,
            1.0,
            0.01,
            0.4,
            &target,
        );
        assert_eq!(
            weekend_roll.issue_date,
            NaiveDate::from_ymd_opt(2026, 6, 20).unwrap()
        );

        let business_roll_trade = NaiveDate::from_ymd_opt(2026, 3, 20).unwrap();
        let business_roll = DatedCds::standard_imm_with_calendar(
            ProtectionSide::Buyer,
            business_roll_trade,
            5,
            1.0,
            0.01,
            0.4,
            &target,
        );
        assert_eq!(business_roll.issue_date, business_roll_trade);
    }

    #[test]
    fn exact_flat_interval_terms_hand_derived_stub_values() {
        // Stub period: integration on [t1, t2] = [0.5, 1.0], accrual measured from
        // t0 = t1 - 0.2 = 0.3, r = 0.04, h = 0.06, k = r + h = 0.1.
        //
        // Hand derivation:
        //   exp1 = e^{-0.05} = 0.95122942450071400
        //   exp2 = e^{-0.10} = 0.90483741803595957
        //   protection = (h/k)(exp1 - exp2) = 0.6 * 0.04639200646475443
        //              = 0.02783520387885266
        //   base accrual (from t1) = h[(exp1 - exp2)/k^2 - (t2 - t1) exp2 / k]
        //              = 0.06 * (4.639200646475443 - 4.524187090179798)
        //              = 0.00690081337773873
        //   offset term = 0.2 * protection = 0.00556704077577053
        //   accrual = 0.01246785415350926
        let (accrual, protection) = exact_flat_interval_terms(0.5, 1.0, 0.04, 0.06, 0.2);
        assert_relative_eq!(protection, 0.02783520387885266, epsilon = 1.0e-12);
        assert_relative_eq!(accrual, 0.01246785415350926, epsilon = 1.0e-12);

        // Zero offset reproduces the plain (t - t1) accrual integral.
        let (accrual0, _) = exact_flat_interval_terms(0.5, 1.0, 0.04, 0.06, 0.0);
        assert_relative_eq!(accrual0, 0.00690081337773873, epsilon = 1.0e-12);
    }

    #[test]
    fn exact_flat_interval_terms_match_numerical_quadrature() {
        // Independent Simpson quadrature of
        //   protection = ∫_t1^t2 h e^{-(r+h)t} dt
        //   accrual    = ∫_t1^t2 (t - t0) h e^{-(r+h)t} dt,  t0 = t1 - offset.
        let cases = [
            (0.5_f64, 1.0_f64, 0.04_f64, 0.06_f64, 0.2_f64),
            (0.008, 0.03, 0.05, 0.0167, 0.233),
            (1.25, 1.5, 0.0, 0.10, 0.0),
            (0.0, 0.25, 0.03, 0.02, 0.13),
        ];

        for &(t1, t2, r, h, offset) in &cases {
            let (accrual, protection) = exact_flat_interval_terms(t1, t2, r, h, offset);

            let n = 10_000usize;
            let dt = (t2 - t1) / n as f64;
            let t0 = t1 - offset;
            let simpson = |f: &dyn Fn(f64) -> f64| {
                let mut s = f(t1) + f(t2);
                for i in 1..n {
                    let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                    s += w * f(t1 + i as f64 * dt);
                }
                s * dt / 3.0
            };

            let prot_num = simpson(&|t: f64| h * (-(r + h) * t).exp());
            let acc_num = simpson(&|t: f64| (t - t0) * h * (-(r + h) * t).exp());

            assert_relative_eq!(
                protection,
                prot_num,
                epsilon = 1.0e-12,
                max_relative = 1.0e-10
            );
            assert_relative_eq!(accrual, acc_num, epsilon = 1.0e-12, max_relative = 1.0e-10);
        }
    }

    #[test]
    fn isda_stub_period_accrues_from_period_start() {
        // Rebuild the premium and protection legs by numerical quadrature with the
        // accrual-on-default measured from each period's start date and compare
        // against the retained legacy flat-integral methodology. Valuation
        // falls mid-period so the stub period has period_start < step_in.
        let eval = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let hazard = 0.02;
        let rate = 0.03;
        let conventions = IsdaConventions::default();
        let cds = DatedCds::standard_imm(ProtectionSide::Buyer, eval, 5, 10_000_000.0, 0.01, 0.4);

        let result = price_isda_flat_legacy_analytic(&cds, eval, hazard, rate, conventions);

        let step_in = step_in_date(eval);
        assert!(
            cds.issue_date < step_in,
            "test setup must produce a front stub"
        );

        let schedule = generate_imm_schedule(
            cds.issue_date,
            cds.maturity_date,
            cds.coupon_interval_months,
            cds.date_rule,
        );

        let mut coupon_annuity = 0.0;
        let mut accrual_on_default = 0.0;
        let mut protection_term = 0.0;
        for window in schedule.windows(2) {
            let period_start = window[0];
            let period_end = window[1];
            if period_end <= step_in {
                continue;
            }

            let accrual = year_fraction(period_start, period_end, DayCountConvention::Act360);
            let t_pay = year_fraction(eval, period_end, DayCountConvention::Act360);
            coupon_annuity += accrual * (-(rate + hazard) * t_pay).exp();

            let default_start = period_start.max(step_in);
            // Accrual origin: the period start, which may precede the valuation date.
            let t0 = year_fraction(eval, period_start, DayCountConvention::Act360);
            let t1 = year_fraction(eval, default_start, DayCountConvention::Act360);
            let t2 = year_fraction(eval, period_end, DayCountConvention::Act360);

            let n = 10_000usize;
            let dt = (t2 - t1) / n as f64;
            let simpson = |f: &dyn Fn(f64) -> f64| {
                let mut s = f(t1) + f(t2);
                for i in 1..n {
                    let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                    s += w * f(t1 + i as f64 * dt);
                }
                s * dt / 3.0
            };
            protection_term += simpson(&|t: f64| hazard * (-(rate + hazard) * t).exp());
            accrual_on_default +=
                simpson(&|t: f64| (t - t0) * hazard * (-(rate + hazard) * t).exp());
        }

        let cash_settle = cash_settle_date(eval);
        let scale =
            (rate * year_fraction(eval, cash_settle, DayCountConvention::Act360).max(0.0)).exp();

        let premium_expected =
            cds.notional * cds.running_spread * (coupon_annuity + accrual_on_default) * scale;
        let protection_expected =
            cds.notional * (1.0 - cds.recovery_rate) * protection_term * scale;

        assert_relative_eq!(
            result.premium_leg_pv,
            premium_expected,
            max_relative = 1.0e-9
        );
        assert_relative_eq!(
            result.protection_leg_pv,
            protection_expected,
            max_relative = 1.0e-9
        );
    }

    #[test]
    fn isda_fair_spread_reprices_exactly_to_par() {
        let eval = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let spread = 0.01;
        let recovery = 0.4;
        let hazard = hazard_from_par_spread(spread, recovery);

        let mut cds = DatedCds::standard_imm(
            ProtectionSide::Buyer,
            eval,
            5,
            10_000_000.0,
            spread,
            recovery,
        );

        let conventions = IsdaConventions::default();
        let initial = price_isda_flat(&cds, eval, hazard, 0.05, conventions);
        cds.running_spread = initial.fair_spread;
        let par = price_isda_flat(&cds, eval, hazard, 0.05, conventions);

        // A quoted fair spread must reprice the same dated cashflows to par;
        // the previous $30,000 tolerance could hide a material convention bug.
        assert_relative_eq!(par.clean_npv, 0.0, epsilon = 1.0e-8);
    }
}
