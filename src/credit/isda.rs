use chrono::{Datelike, Duration, NaiveDate};

use crate::rates::{
    Calendar, DayCountConvention, add_business_days, next_cds_date, previous_cds_date,
    year_fraction,
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
        let step_in = step_in_date(trade_date);
        let start = previous_imm_twentieth(step_in);
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
    /// Step-in date offset in business days from valuation date.
    pub step_in_days: usize,
    /// Cash-settlement date offset in business days from valuation date.
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
    /// NPV from the trade side perspective (buyer positive when valuable to buyer).
    pub clean_npv: f64,
    /// NPV including accrued premium.
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
    price_flat_with_model(
        cds,
        valuation_date,
        hazard_rate,
        discount_rate,
        conventions,
        PricingModel::Midpoint,
    )
}

/// ISDA-style CDS valuation using exact flat-hazard/flat-discount integrals.
pub fn price_isda_flat(
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
        PricingModel::IsdaStandard,
    )
}

/// Converts a par running spread to a flat hazard intensity under flat-LGD approximation.
pub fn hazard_from_par_spread(par_spread: f64, recovery_rate: f64) -> f64 {
    if !(0.0..1.0).contains(&recovery_rate) {
        return 0.0;
    }
    (par_spread.max(0.0) / (1.0 - recovery_rate)).max(0.0)
}

/// Standard CDS step-in date (T+1 business day).
pub fn step_in_date(valuation_date: NaiveDate) -> NaiveDate {
    add_business_days(valuation_date, 1, &Calendar::weekends_only())
}

/// Standard CDS cash-settlement date (T+3 business days).
pub fn cash_settle_date(valuation_date: NaiveDate) -> NaiveDate {
    add_business_days(valuation_date, 3, &Calendar::weekends_only())
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
enum PricingModel {
    Midpoint,
    IsdaStandard,
}

fn price_flat_with_model(
    cds: &DatedCds,
    valuation_date: NaiveDate,
    hazard_rate: f64,
    discount_rate: f64,
    conventions: IsdaConventions,
    model: PricingModel,
) -> CdsPriceResult {
    let step_in = advance_business_days(valuation_date, conventions.step_in_days);
    let cash_settle = advance_business_days(valuation_date, conventions.cash_settle_days);

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

        match model {
            PricingModel::Midpoint => {
                let default_prob = ((-h * t1).exp() - (-h * t2).exp()).max(0.0);
                let t_mid = 0.5 * (t1 + t2);
                let df_mid = (-r * t_mid).exp();
                let accrual_default =
                    year_fraction(default_start, period_end, DayCountConvention::Act360);

                protection_term += df_mid * default_prob;
                accrual_on_default += 0.5 * accrual_default * df_mid * default_prob;
            }
            PricingModel::IsdaStandard => {
                let (accrual_term, protection_piece) = exact_flat_interval_terms(t1, t2, r, h);
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

fn exact_flat_interval_terms(t1: f64, t2: f64, r: f64, h: f64) -> (f64, f64) {
    if t2 <= t1 || h <= 0.0 {
        return (0.0, 0.0);
    }

    let k = (r + h).max(1.0e-14);
    let exp1 = (-k * t1).exp();
    let exp2 = (-k * t2).exp();

    // Protection integral: ∫_t1^t2 DF(t) h S(t) dt.
    let protection = (h / k) * (exp1 - exp2);

    // Accrual-on-default integral using accrual from interval start t1:
    // ∫_t1^t2 (t - t1) DF(t) h S(t) dt.
    let dt = t2 - t1;
    let accrual = h * ((exp1 - exp2) / (k * k) - dt * exp2 / k);

    (accrual.max(0.0), protection.max(0.0))
}

fn advance_business_days(date: NaiveDate, days: usize) -> NaiveDate {
    add_business_days(date, days as i32, &Calendar::weekends_only())
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
    fn isda_par_cds_is_near_zero_when_hazard_is_calibrated_from_spread() {
        let eval = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let spread = 0.01;
        let recovery = 0.4;
        let hazard = hazard_from_par_spread(spread, recovery);

        let cds = DatedCds::standard_imm(
            ProtectionSide::Buyer,
            eval,
            5,
            10_000_000.0,
            spread,
            recovery,
        );

        let result = price_isda_flat(&cds, eval, hazard, 0.05, IsdaConventions::default());
        assert_relative_eq!(result.clean_npv, 0.0, epsilon = 3.0e4);
    }
}
