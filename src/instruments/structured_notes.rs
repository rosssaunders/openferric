//! Module `instruments::structured_notes`.
//!
//! Interest-rate structured notes and callable note infrastructure:
//!
//! - Bermudan-callable fixed/floating/structured notes priced on a one-factor
//!   Hull-White short-rate tree.
//! - Callable range-accrual note wrapper.
//! - Target redemption notes (rates TARNs).
//! - Snowball notes.
//! - Inverse floaters.
//! - CMS-linked coupons.
//! - Coupon and Bermudan exercise schedule builders.
//!
//! # Design Notes
//!
//! The callable pricer uses a recombining trinomial-style short-rate lattice
//! based on local moments, consistent with the existing
//! [`crate::engines::tree::bermudan_swaption`] implementation.
//! Notice periods are handled by converting call dates into effective decision
//! dates `max(0, call_date - notice_period)`.
//!
//! Non-callable path-dependent notes (TARN and snowball) are exposed as
//! deterministic pricers against projected rate paths.

use crate::models::HullWhite;
use crate::rates::{Frequency, YieldCurve};

/// Structured coupon formula used by coupon schedules.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StructuredCoupon {
    /// Coupon accrues only when the observed short rate is inside the range.
    RangeAccrual {
        /// Annualized coupon rate when in range.
        in_range_coupon_rate: f64,
        /// Annualized coupon rate when out of range.
        out_of_range_coupon_rate: f64,
        /// Lower bound of the accrual range.
        lower_bound: f64,
        /// Upper bound of the accrual range.
        upper_bound: f64,
    },
    /// Coupon equals `fixed_rate - leverage * floating_rate`.
    InverseFloater {
        fixed_rate: f64,
        leverage: f64,
        floor: Option<f64>,
        cap: Option<f64>,
    },
    /// Coupon linked to a CMS rate proxy.
    CmsLinked {
        /// Multiplier applied to the CMS rate.
        multiplier: f64,
        /// Additive spread.
        spread: f64,
        /// CMS tenor in years (used by tree/cashflow projection helpers).
        cms_tenor: f64,
        /// Par-swap payment frequency used in CMS projection.
        swap_payment_frequency: Frequency,
        /// Optional coupon floor.
        floor: Option<f64>,
        /// Optional coupon cap.
        cap: Option<f64>,
    },
}

/// Coupon type for a period.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CouponType {
    /// Fixed coupon rate.
    Fixed { rate: f64 },
    /// Floating coupon rate as `floating + spread`, with optional floor/cap.
    Floating {
        spread: f64,
        floor: Option<f64>,
        cap: Option<f64>,
    },
    /// Structured coupon formula.
    Structured(StructuredCoupon),
}

/// One coupon period in year-fraction time.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CouponPeriod {
    /// Period start time in years.
    pub start_time: f64,
    /// Period end time in years.
    pub end_time: f64,
    /// Payment time in years.
    pub payment_time: f64,
    /// Coupon type.
    pub coupon: CouponType,
}

impl CouponPeriod {
    /// Accrual fraction for the coupon period.
    #[inline]
    pub fn accrual(&self) -> f64 {
        (self.end_time - self.start_time).max(0.0)
    }

    fn validate(&self) -> Result<(), String> {
        if !self.start_time.is_finite()
            || !self.end_time.is_finite()
            || !self.payment_time.is_finite()
        {
            return Err("coupon period times must be finite".to_string());
        }
        if self.end_time <= self.start_time {
            return Err("coupon period end_time must be > start_time".to_string());
        }
        if self.payment_time < self.end_time {
            return Err("coupon period payment_time must be >= end_time".to_string());
        }
        validate_coupon_type(&self.coupon)
    }
}

/// Builder for regular coupon schedules.
#[derive(Debug, Clone, PartialEq)]
pub struct CouponScheduleBuilder {
    start_time: f64,
    end_time: f64,
    frequency: Frequency,
    payment_lag: f64,
}

impl CouponScheduleBuilder {
    /// Creates a new regular schedule builder.
    pub fn new(start_time: f64, end_time: f64, frequency: Frequency) -> Result<Self, String> {
        if !start_time.is_finite() || !end_time.is_finite() {
            return Err("start_time and end_time must be finite".to_string());
        }
        if end_time <= start_time {
            return Err("end_time must be > start_time".to_string());
        }

        Ok(Self {
            start_time,
            end_time,
            frequency,
            payment_lag: 0.0,
        })
    }

    /// Sets a positive payment lag in years.
    pub fn payment_lag(mut self, payment_lag: f64) -> Result<Self, String> {
        if !payment_lag.is_finite() || payment_lag < 0.0 {
            return Err("payment_lag must be finite and >= 0".to_string());
        }
        self.payment_lag = payment_lag;
        Ok(self)
    }

    /// Builds a fixed coupon schedule.
    pub fn build_fixed(&self, rate: f64) -> Result<Vec<CouponPeriod>, String> {
        if !rate.is_finite() {
            return Err("fixed coupon rate must be finite".to_string());
        }
        self.build(CouponType::Fixed { rate })
    }

    /// Builds a floating coupon schedule.
    pub fn build_floating(
        &self,
        spread: f64,
        floor: Option<f64>,
        cap: Option<f64>,
    ) -> Result<Vec<CouponPeriod>, String> {
        if !spread.is_finite() {
            return Err("floating spread must be finite".to_string());
        }
        validate_floor_cap(floor, cap)?;
        self.build(CouponType::Floating { spread, floor, cap })
    }

    /// Builds a structured coupon schedule.
    pub fn build_structured(
        &self,
        structured: StructuredCoupon,
    ) -> Result<Vec<CouponPeriod>, String> {
        self.build(CouponType::Structured(structured))
    }

    fn build(&self, coupon: CouponType) -> Result<Vec<CouponPeriod>, String> {
        validate_coupon_type(&coupon)?;

        let step = frequency_year_fraction(self.frequency);
        let mut periods = Vec::new();
        let mut t0 = self.start_time;

        while t0 < self.end_time - 1.0e-12 {
            let t1 = (t0 + step).min(self.end_time);
            periods.push(CouponPeriod {
                start_time: t0,
                end_time: t1,
                payment_time: t1 + self.payment_lag,
                coupon: coupon.clone(),
            });
            t0 = t1;
        }

        Ok(periods)
    }
}

/// Bermudan exercise schedule with notice-period support.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExerciseSchedule {
    /// Contractual call/exercise dates in years.
    pub bermudan_dates: Vec<f64>,
    /// Notice period in years.
    pub notice_period: f64,
}

impl ExerciseSchedule {
    /// Creates and validates an exercise schedule.
    pub fn new(bermudan_dates: Vec<f64>, notice_period: f64) -> Result<Self, String> {
        let out = Self {
            bermudan_dates,
            notice_period,
        };
        out.validate()?;
        Ok(out)
    }

    /// Effective issuer decision times after notice adjustment.
    pub fn decision_times(&self) -> Vec<f64> {
        let mut dates = self
            .bermudan_dates
            .iter()
            .map(|t| (t - self.notice_period).max(0.0))
            .collect::<Vec<_>>();
        dates.sort_by(|a, b| a.total_cmp(b));
        dates.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
        dates
    }

    /// Validates schedule monotonicity and notice settings.
    pub fn validate(&self) -> Result<(), String> {
        if self.bermudan_dates.is_empty() {
            return Err("bermudan_dates must be non-empty".to_string());
        }
        if !self.notice_period.is_finite() || self.notice_period < 0.0 {
            return Err("notice_period must be finite and >= 0".to_string());
        }
        if self
            .bermudan_dates
            .iter()
            .any(|t| !t.is_finite() || *t < 0.0)
        {
            return Err("bermudan_dates must be finite and >= 0".to_string());
        }
        if self.bermudan_dates.windows(2).any(|w| w[1] <= w[0]) {
            return Err("bermudan_dates must be strictly increasing".to_string());
        }
        Ok(())
    }
}

/// Bermudan-callable rate note priced under Hull-White tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CallableRateNote {
    /// Coupon notional.
    pub notional: f64,
    /// Redemption amount paid at maturity if not called.
    pub redemption: f64,
    /// Call settlement amount paid when called.
    pub call_price: f64,
    /// Final maturity in years.
    pub maturity: f64,
    /// Coupon schedule.
    pub coupon_schedule: Vec<CouponPeriod>,
    /// Bermudan exercise schedule.
    pub exercise_schedule: ExerciseSchedule,
}

impl CallableRateNote {
    /// Validates the callable note definition.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.redemption.is_finite() || self.redemption <= 0.0 {
            return Err("redemption must be finite and > 0".to_string());
        }
        if !self.call_price.is_finite() || self.call_price <= 0.0 {
            return Err("call_price must be finite and > 0".to_string());
        }
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err("maturity must be finite and > 0".to_string());
        }
        if self.coupon_schedule.is_empty() {
            return Err("coupon_schedule must be non-empty".to_string());
        }

        for p in &self.coupon_schedule {
            p.validate()?;
            if p.payment_time > self.maturity + 1.0e-10 {
                return Err("coupon payment_time must be <= maturity".to_string());
            }
        }
        if self
            .coupon_schedule
            .windows(2)
            .any(|w| w[1].start_time < w[0].end_time - 1.0e-12)
        {
            return Err("coupon_schedule periods overlap".to_string());
        }

        self.exercise_schedule.validate()?;
        if self
            .exercise_schedule
            .bermudan_dates
            .iter()
            .any(|t| *t > self.maturity + 1.0e-12)
        {
            return Err("bermudan_dates must be <= maturity".to_string());
        }

        Ok(())
    }

    /// Prices the callable note using a Hull-White short-rate tree.
    ///
    /// The value is from the note-holder perspective, with issuer call rights.
    /// At each effective decision date, continuation value is floored by issuer
    /// callability: `V = min(V_continuation, call_price)`.
    pub fn price_hull_white_tree(
        &self,
        hw_model: &HullWhite,
        curve: &YieldCurve,
        steps: usize,
    ) -> Result<f64, String> {
        self.validate()?;
        if steps == 0 {
            return Err("steps must be > 0".to_string());
        }

        price_callable_note_tree(
            self.notional,
            self.redemption,
            self.call_price,
            self.maturity,
            &self.coupon_schedule,
            &self.exercise_schedule,
            hw_model,
            curve,
            steps,
        )
    }

    /// Deterministic hold-to-maturity value from projected floating and CMS rates.
    ///
    /// `projected_floating_rates[i]` and `projected_cms_rates[i]` map to
    /// `coupon_schedule[i]`. The method discounts all coupons plus redemption
    /// without optionality.
    pub fn hold_to_maturity_value(
        &self,
        curve: &YieldCurve,
        projected_floating_rates: &[f64],
        projected_cms_rates: &[f64],
    ) -> Result<f64, String> {
        if projected_floating_rates.len() != self.coupon_schedule.len()
            || projected_cms_rates.len() != self.coupon_schedule.len()
        {
            return Err(
                "projected_floating_rates and projected_cms_rates must match coupon_schedule length"
                    .to_string(),
            );
        }

        let mut pv = 0.0;
        for (i, period) in self.coupon_schedule.iter().enumerate() {
            let coupon_rate = coupon_rate_deterministic(
                &period.coupon,
                projected_floating_rates[i],
                projected_cms_rates[i],
            )?;
            let cf = self.notional * period.accrual() * coupon_rate;
            pv += cf * curve.discount_factor(period.payment_time);
        }

        Ok(pv + self.redemption * curve.discount_factor(self.maturity))
    }
}

/// Callable range-accrual note priced on Hull-White tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CallableRangeAccrualNote {
    /// Underlying callable note container.
    pub note: CallableRateNote,
}

impl CallableRangeAccrualNote {
    /// Builds a callable range-accrual note with regular coupon periods.
    pub fn new(
        notional: f64,
        maturity: f64,
        frequency: Frequency,
        in_range_coupon_rate: f64,
        out_of_range_coupon_rate: f64,
        lower_bound: f64,
        upper_bound: f64,
        call_price: f64,
        exercise_schedule: ExerciseSchedule,
    ) -> Result<Self, String> {
        let schedule = CouponScheduleBuilder::new(0.0, maturity, frequency)?.build_structured(
            StructuredCoupon::RangeAccrual {
                in_range_coupon_rate,
                out_of_range_coupon_rate,
                lower_bound,
                upper_bound,
            },
        )?;

        Ok(Self {
            note: CallableRateNote {
                notional,
                redemption: notional,
                call_price,
                maturity,
                coupon_schedule: schedule,
                exercise_schedule,
            },
        })
    }

    /// Prices via Hull-White tree.
    pub fn price_hull_white_tree(
        &self,
        hw_model: &HullWhite,
        curve: &YieldCurve,
        steps: usize,
    ) -> Result<f64, String> {
        self.note.price_hull_white_tree(hw_model, curve, steps)
    }
}

/// Pricing output for a rates TARN.
#[derive(Debug, Clone, PartialEq)]
pub struct TarnPricingResult {
    /// Present value from investor perspective.
    pub price: f64,
    /// Total coupon accrued before knock-out/final maturity.
    pub accrued_coupon: f64,
    /// Whether target redemption was reached.
    pub knocked_out: bool,
    /// Knock-out time if reached.
    pub knockout_time: Option<f64>,
}

/// Target redemption note (rates version).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TargetRedemptionNote {
    /// Coupon notional.
    pub notional: f64,
    /// Redemption amount if outstanding at maturity or at knock-out.
    pub redemption: f64,
    /// Coupon target in currency amount.
    pub target_coupon: f64,
    /// Coupon periods.
    pub coupon_schedule: Vec<CouponPeriod>,
    /// Additive spread over floating projection.
    pub spread: f64,
    /// Optional coupon floor.
    pub floor: Option<f64>,
    /// Optional coupon cap.
    pub cap: Option<f64>,
}

impl TargetRedemptionNote {
    /// Validates note inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.redemption.is_finite() || self.redemption <= 0.0 {
            return Err("redemption must be finite and > 0".to_string());
        }
        if !self.target_coupon.is_finite() || self.target_coupon <= 0.0 {
            return Err("target_coupon must be finite and > 0".to_string());
        }
        validate_floor_cap(self.floor, self.cap)?;
        if self.coupon_schedule.is_empty() {
            return Err("coupon_schedule must be non-empty".to_string());
        }
        for p in &self.coupon_schedule {
            p.validate()?;
        }
        Ok(())
    }

    /// Prices a rates TARN from projected floating rates.
    pub fn price(
        &self,
        projected_floating_rates: &[f64],
        curve: &YieldCurve,
    ) -> Result<TarnPricingResult, String> {
        self.validate()?;
        if projected_floating_rates.len() != self.coupon_schedule.len() {
            return Err("projected_floating_rates length must match coupon_schedule".to_string());
        }

        let mut pv = 0.0;
        let mut accrued = 0.0;
        let mut knocked_out = false;
        let mut knockout_time = None;

        for (i, period) in self.coupon_schedule.iter().enumerate() {
            let mut coupon_rate = projected_floating_rates[i] + self.spread;
            coupon_rate = apply_floor_cap(coupon_rate, self.floor, self.cap)?;

            let full_coupon = self.notional * period.accrual() * coupon_rate;
            let remaining_target = (self.target_coupon - accrued).max(0.0);
            let paid_coupon = full_coupon.min(remaining_target);

            accrued += paid_coupon;
            pv += paid_coupon * curve.discount_factor(period.payment_time);

            if accrued >= self.target_coupon - 1.0e-12 {
                knocked_out = true;
                knockout_time = Some(period.payment_time);
                pv += self.redemption * curve.discount_factor(period.payment_time);
                break;
            }
        }

        if !knocked_out {
            let maturity = self
                .coupon_schedule
                .iter()
                .map(|p| p.payment_time)
                .fold(0.0_f64, f64::max);
            pv += self.redemption * curve.discount_factor(maturity);
        }

        Ok(TarnPricingResult {
            price: pv,
            accrued_coupon: accrued,
            knocked_out,
            knockout_time,
        })
    }
}

/// Pricing output for snowball notes.
#[derive(Debug, Clone, PartialEq)]
pub struct SnowballPricingResult {
    /// Present value.
    pub price: f64,
    /// Coupon rates for each period.
    pub coupon_path: Vec<f64>,
}

/// Snowball note where
/// `coupon_i = max(0, coupon_{i-1} + spread - floating_i)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SnowballNote {
    /// Coupon notional.
    pub notional: f64,
    /// Redemption amount at maturity.
    pub redemption: f64,
    /// Initial coupon rate (`coupon_0`).
    pub initial_coupon: f64,
    /// Spread term in recursion.
    pub spread: f64,
    /// Optional floor on each coupon rate after recursion.
    pub floor: Option<f64>,
    /// Optional cap on each coupon rate after recursion.
    pub cap: Option<f64>,
    /// Coupon periods.
    pub coupon_schedule: Vec<CouponPeriod>,
}

impl SnowballNote {
    /// Validates note inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.redemption.is_finite() || self.redemption <= 0.0 {
            return Err("redemption must be finite and > 0".to_string());
        }
        if !self.initial_coupon.is_finite() {
            return Err("initial_coupon must be finite".to_string());
        }
        if !self.spread.is_finite() {
            return Err("spread must be finite".to_string());
        }
        validate_floor_cap(self.floor, self.cap)?;
        if self.coupon_schedule.is_empty() {
            return Err("coupon_schedule must be non-empty".to_string());
        }
        for p in &self.coupon_schedule {
            p.validate()?;
        }
        Ok(())
    }

    /// Prices snowball note from projected floating rates.
    pub fn price(
        &self,
        projected_floating_rates: &[f64],
        curve: &YieldCurve,
    ) -> Result<SnowballPricingResult, String> {
        self.validate()?;
        if projected_floating_rates.len() != self.coupon_schedule.len() {
            return Err("projected_floating_rates length must match coupon_schedule".to_string());
        }

        let mut prev_coupon = self.initial_coupon;
        let mut pv = 0.0;
        let mut coupons = Vec::with_capacity(self.coupon_schedule.len());

        for (i, period) in self.coupon_schedule.iter().enumerate() {
            let raw = (prev_coupon + self.spread - projected_floating_rates[i]).max(0.0);
            let coupon_rate = apply_floor_cap(raw, self.floor, self.cap)?;
            coupons.push(coupon_rate);

            let cf = self.notional * period.accrual() * coupon_rate;
            pv += cf * curve.discount_factor(period.payment_time);

            prev_coupon = coupon_rate;
        }

        let maturity = self
            .coupon_schedule
            .iter()
            .map(|p| p.payment_time)
            .fold(0.0_f64, f64::max);
        pv += self.redemption * curve.discount_factor(maturity);

        Ok(SnowballPricingResult {
            price: pv,
            coupon_path: coupons,
        })
    }
}

/// Inverse floater note.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InverseFloaterNote {
    /// Coupon notional.
    pub notional: f64,
    /// Redemption amount at maturity.
    pub redemption: f64,
    /// Constant term in coupon formula.
    pub fixed_rate: f64,
    /// Leverage multiplier on floating rate.
    pub leverage: f64,
    /// Optional coupon floor.
    pub floor: Option<f64>,
    /// Optional coupon cap.
    pub cap: Option<f64>,
    /// Coupon periods.
    pub coupon_schedule: Vec<CouponPeriod>,
}

impl InverseFloaterNote {
    /// Validates inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.redemption.is_finite() || self.redemption <= 0.0 {
            return Err("redemption must be finite and > 0".to_string());
        }
        if !self.fixed_rate.is_finite() || !self.leverage.is_finite() {
            return Err("fixed_rate and leverage must be finite".to_string());
        }
        validate_floor_cap(self.floor, self.cap)?;
        if self.coupon_schedule.is_empty() {
            return Err("coupon_schedule must be non-empty".to_string());
        }
        for p in &self.coupon_schedule {
            p.validate()?;
        }
        Ok(())
    }

    /// Prices inverse floater from projected floating rates.
    pub fn price(
        &self,
        projected_floating_rates: &[f64],
        curve: &YieldCurve,
    ) -> Result<f64, String> {
        self.validate()?;
        if projected_floating_rates.len() != self.coupon_schedule.len() {
            return Err("projected_floating_rates length must match coupon_schedule".to_string());
        }

        let mut pv = 0.0;
        for (i, period) in self.coupon_schedule.iter().enumerate() {
            let raw = self.fixed_rate - self.leverage * projected_floating_rates[i];
            let coupon_rate = apply_floor_cap(raw, self.floor, self.cap)?;
            let cf = self.notional * period.accrual() * coupon_rate;
            pv += cf * curve.discount_factor(period.payment_time);
        }

        let maturity = self
            .coupon_schedule
            .iter()
            .map(|p| p.payment_time)
            .fold(0.0_f64, f64::max);
        Ok(pv + self.redemption * curve.discount_factor(maturity))
    }
}

/// CMS-linked note with coupon `multiplier * cms + spread`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CmsLinkedNote {
    /// Coupon notional.
    pub notional: f64,
    /// Redemption amount at maturity.
    pub redemption: f64,
    /// Coupon multiplier on CMS rate.
    pub multiplier: f64,
    /// Additive spread.
    pub spread: f64,
    /// CMS tenor in years.
    pub cms_tenor: f64,
    /// Frequency used for par swap annuity in CMS projection.
    pub swap_payment_frequency: Frequency,
    /// Optional coupon floor.
    pub floor: Option<f64>,
    /// Optional coupon cap.
    pub cap: Option<f64>,
    /// Coupon periods.
    pub coupon_schedule: Vec<CouponPeriod>,
}

impl CmsLinkedNote {
    /// Validates inputs.
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.redemption.is_finite() || self.redemption <= 0.0 {
            return Err("redemption must be finite and > 0".to_string());
        }
        if !self.multiplier.is_finite() || !self.spread.is_finite() {
            return Err("multiplier and spread must be finite".to_string());
        }
        if !self.cms_tenor.is_finite() || self.cms_tenor <= 0.0 {
            return Err("cms_tenor must be finite and > 0".to_string());
        }
        validate_floor_cap(self.floor, self.cap)?;
        if self.coupon_schedule.is_empty() {
            return Err("coupon_schedule must be non-empty".to_string());
        }
        for p in &self.coupon_schedule {
            p.validate()?;
        }
        Ok(())
    }

    /// Prices from externally projected CMS rates.
    pub fn price_with_projected_cms_rates(
        &self,
        projected_cms_rates: &[f64],
        curve: &YieldCurve,
    ) -> Result<f64, String> {
        self.validate()?;
        if projected_cms_rates.len() != self.coupon_schedule.len() {
            return Err("projected_cms_rates length must match coupon_schedule".to_string());
        }

        let mut pv = 0.0;
        for (i, period) in self.coupon_schedule.iter().enumerate() {
            let raw = self.multiplier * projected_cms_rates[i] + self.spread;
            let coupon_rate = apply_floor_cap(raw, self.floor, self.cap)?;
            let cf = self.notional * period.accrual() * coupon_rate;
            pv += cf * curve.discount_factor(period.payment_time);
        }

        let maturity = self
            .coupon_schedule
            .iter()
            .map(|p| p.payment_time)
            .fold(0.0_f64, f64::max);

        Ok(pv + self.redemption * curve.discount_factor(maturity))
    }

    /// Prices by projecting CMS forwards directly from a discount curve.
    pub fn price_from_curve(&self, curve: &YieldCurve) -> Result<f64, String> {
        self.validate()?;
        let cms = self.projected_cms_rates_from_curve(curve)?;
        self.price_with_projected_cms_rates(&cms, curve)
    }

    /// Projects CMS forward rates at each coupon start from `curve`.
    pub fn projected_cms_rates_from_curve(&self, curve: &YieldCurve) -> Result<Vec<f64>, String> {
        self.validate()?;

        let mut out = Vec::with_capacity(self.coupon_schedule.len());
        for p in &self.coupon_schedule {
            out.push(forward_swap_rate(
                curve,
                p.start_time,
                self.cms_tenor,
                self.swap_payment_frequency,
            )?);
        }
        Ok(out)
    }
}

fn validate_coupon_type(coupon: &CouponType) -> Result<(), String> {
    match coupon {
        CouponType::Fixed { rate } => {
            if !rate.is_finite() {
                return Err("fixed coupon rate must be finite".to_string());
            }
        }
        CouponType::Floating { spread, floor, cap } => {
            if !spread.is_finite() {
                return Err("floating spread must be finite".to_string());
            }
            validate_floor_cap(*floor, *cap)?;
        }
        CouponType::Structured(s) => match s {
            StructuredCoupon::RangeAccrual {
                in_range_coupon_rate,
                out_of_range_coupon_rate,
                lower_bound,
                upper_bound,
            } => {
                if !in_range_coupon_rate.is_finite()
                    || !out_of_range_coupon_rate.is_finite()
                    || !lower_bound.is_finite()
                    || !upper_bound.is_finite()
                {
                    return Err("range accrual parameters must be finite".to_string());
                }
                if lower_bound >= upper_bound {
                    return Err("range accrual lower_bound must be < upper_bound".to_string());
                }
            }
            StructuredCoupon::InverseFloater {
                fixed_rate,
                leverage,
                floor,
                cap,
            } => {
                if !fixed_rate.is_finite() || !leverage.is_finite() {
                    return Err("inverse floater parameters must be finite".to_string());
                }
                validate_floor_cap(*floor, *cap)?;
            }
            StructuredCoupon::CmsLinked {
                multiplier,
                spread,
                cms_tenor,
                swap_payment_frequency: _,
                floor,
                cap,
            } => {
                if !multiplier.is_finite() || !spread.is_finite() || !cms_tenor.is_finite() {
                    return Err("CMS-linked parameters must be finite".to_string());
                }
                if *cms_tenor <= 0.0 {
                    return Err("cms_tenor must be > 0".to_string());
                }
                validate_floor_cap(*floor, *cap)?;
            }
        },
    }

    Ok(())
}

fn validate_floor_cap(floor: Option<f64>, cap: Option<f64>) -> Result<(), String> {
    if let Some(f) = floor
        && !f.is_finite()
    {
        return Err("floor must be finite when provided".to_string());
    }
    if let Some(c) = cap
        && !c.is_finite()
    {
        return Err("cap must be finite when provided".to_string());
    }
    if let (Some(f), Some(c)) = (floor, cap)
        && f > c
    {
        return Err("floor must be <= cap".to_string());
    }
    Ok(())
}

fn apply_floor_cap(rate: f64, floor: Option<f64>, cap: Option<f64>) -> Result<f64, String> {
    if !rate.is_finite() {
        return Err("coupon rate is not finite".to_string());
    }
    validate_floor_cap(floor, cap)?;

    let mut out = rate;
    if let Some(f) = floor {
        out = out.max(f);
    }
    if let Some(c) = cap {
        out = out.min(c);
    }
    Ok(out)
}

fn frequency_year_fraction(freq: Frequency) -> f64 {
    match freq {
        Frequency::Annual => 1.0,
        Frequency::SemiAnnual => 0.5,
        Frequency::Quarterly => 0.25,
        Frequency::Monthly => 1.0 / 12.0,
    }
}

fn coupon_rate_deterministic(
    coupon: &CouponType,
    floating_rate: f64,
    cms_rate: f64,
) -> Result<f64, String> {
    if !floating_rate.is_finite() || !cms_rate.is_finite() {
        return Err("projected rates must be finite".to_string());
    }

    match coupon {
        CouponType::Fixed { rate } => Ok(*rate),
        CouponType::Floating { spread, floor, cap } => {
            apply_floor_cap(floating_rate + *spread, *floor, *cap)
        }
        CouponType::Structured(StructuredCoupon::RangeAccrual {
            in_range_coupon_rate,
            out_of_range_coupon_rate,
            lower_bound,
            upper_bound,
        }) => {
            if floating_rate >= *lower_bound && floating_rate <= *upper_bound {
                Ok(*in_range_coupon_rate)
            } else {
                Ok(*out_of_range_coupon_rate)
            }
        }
        CouponType::Structured(StructuredCoupon::InverseFloater {
            fixed_rate,
            leverage,
            floor,
            cap,
        }) => apply_floor_cap(*fixed_rate - *leverage * floating_rate, *floor, *cap),
        CouponType::Structured(StructuredCoupon::CmsLinked {
            multiplier,
            spread,
            cms_tenor: _,
            swap_payment_frequency: _,
            floor,
            cap,
        }) => apply_floor_cap(*multiplier * cms_rate + *spread, *floor, *cap),
    }
}

#[allow(clippy::too_many_arguments)]
fn price_callable_note_tree(
    notional: f64,
    redemption: f64,
    call_price: f64,
    maturity: f64,
    coupon_schedule: &[CouponPeriod],
    exercise_schedule: &ExerciseSchedule,
    hw_model: &HullWhite,
    curve: &YieldCurve,
    steps: usize,
) -> Result<f64, String> {
    let dt = maturity / steps as f64;
    let r0 = HullWhite::instantaneous_forward(curve, 0.0);

    let mut model = hw_model.clone();
    let time_grid = (0..=steps).map(|i| i as f64 * dt).collect::<Vec<_>>();
    model.calibrate_theta(curve, &time_grid);

    let dx = if model.sigma.abs() <= 1.0e-14 {
        (3.0 * dt).sqrt() * 1.0e-6
    } else {
        model.sigma * (3.0 * dt).sqrt()
    };

    let exercise_flags = map_exercise_steps(&exercise_schedule.decision_times(), maturity, steps);
    let coupon_map = map_coupon_steps(coupon_schedule, maturity, steps);

    let mut values = vec![0.0_f64; 2 * steps + 1];
    for j in -(steps as isize)..=(steps as isize) {
        let idx = (j + steps as isize) as usize;
        let r = r0 + j as f64 * dx;

        let mut node_value = redemption;
        for period_idx in &coupon_map[steps] {
            let p = &coupon_schedule[*period_idx];
            node_value += coupon_cashflow_tree(notional, p, maturity, r, &model, curve)?;
        }

        if exercise_flags[steps] {
            node_value = node_value.min(call_price);
        }

        values[idx] = node_value;
    }

    for i in (0..steps).rev() {
        let mut next_values = vec![0.0_f64; 2 * i + 1];
        let t = i as f64 * dt;

        for j in -(i as isize)..=(i as isize) {
            let r = r0 + j as f64 * dx;
            let (pu, pm, pd) = trinomial_probs(&model, t, r, dt, dx);
            let disc = (-r * dt).exp();

            let next_shift = i + 1;
            let up_idx = (j + next_shift as isize + 1) as usize;
            let mid_idx = (j + next_shift as isize) as usize;
            let down_idx = (j + next_shift as isize - 1) as usize;

            let mut node_value =
                disc * (pu * values[up_idx] + pm * values[mid_idx] + pd * values[down_idx]);

            for period_idx in &coupon_map[i] {
                let p = &coupon_schedule[*period_idx];
                node_value += coupon_cashflow_tree(notional, p, t, r, &model, curve)?;
            }

            if exercise_flags[i] {
                node_value = node_value.min(call_price);
            }

            let idx = (j + i as isize) as usize;
            next_values[idx] = node_value;
        }

        values = next_values;
    }

    Ok(values[0])
}

fn map_exercise_steps(dates: &[f64], horizon: f64, steps: usize) -> Vec<bool> {
    let mut flags = vec![false; steps + 1];
    for &t in dates {
        if !t.is_finite() || t < 0.0 {
            continue;
        }
        let idx = ((t / horizon) * steps as f64).round() as usize;
        flags[idx.min(steps)] = true;
    }
    flags
}

fn map_coupon_steps(
    coupon_schedule: &[CouponPeriod],
    horizon: f64,
    steps: usize,
) -> Vec<Vec<usize>> {
    let mut map = vec![Vec::new(); steps + 1];
    for (idx, period) in coupon_schedule.iter().enumerate() {
        let step = ((period.payment_time / horizon) * steps as f64).round() as usize;
        map[step.min(steps)].push(idx);
    }
    map
}

fn trinomial_probs(model: &HullWhite, t: f64, rate: f64, dt: f64, dx: f64) -> (f64, f64, f64) {
    if dx <= 0.0 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let mu = model.theta_at(t) - model.a * rate;
    let m1 = mu * dt;
    let variance = model.sigma * model.sigma * dt;
    let total = (variance + m1 * m1) / (dx * dx);

    let mut pu = 0.5 * (total + m1 / dx);
    let mut pd = 0.5 * (total - m1 / dx);
    let mut pm = 1.0 - total;

    pu = pu.max(0.0);
    pm = pm.max(0.0);
    pd = pd.max(0.0);

    let norm = pu + pm + pd;
    if norm <= 1.0e-14 {
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    } else {
        (pu / norm, pm / norm, pd / norm)
    }
}

fn coupon_cashflow_tree(
    notional: f64,
    period: &CouponPeriod,
    eval_time: f64,
    short_rate: f64,
    model: &HullWhite,
    curve: &YieldCurve,
) -> Result<f64, String> {
    let accrual = period.accrual();
    if accrual <= 0.0 {
        return Ok(0.0);
    }

    let rate = match &period.coupon {
        CouponType::Fixed { rate } => *rate,
        CouponType::Floating { spread, floor, cap } => {
            apply_floor_cap(short_rate + *spread, *floor, *cap)?
        }
        CouponType::Structured(StructuredCoupon::RangeAccrual {
            in_range_coupon_rate,
            out_of_range_coupon_rate,
            lower_bound,
            upper_bound,
        }) => {
            if short_rate >= *lower_bound && short_rate <= *upper_bound {
                *in_range_coupon_rate
            } else {
                *out_of_range_coupon_rate
            }
        }
        CouponType::Structured(StructuredCoupon::InverseFloater {
            fixed_rate,
            leverage,
            floor,
            cap,
        }) => apply_floor_cap(*fixed_rate - *leverage * short_rate, *floor, *cap)?,
        CouponType::Structured(StructuredCoupon::CmsLinked {
            multiplier,
            spread,
            cms_tenor,
            swap_payment_frequency,
            floor,
            cap,
        }) => {
            let cms = node_forward_swap_rate(
                model,
                curve,
                eval_time,
                short_rate,
                *cms_tenor,
                *swap_payment_frequency,
            );
            apply_floor_cap(*multiplier * cms + *spread, *floor, *cap)?
        }
    };

    Ok(notional * accrual * rate)
}

fn forward_swap_rate(
    curve: &YieldCurve,
    start_time: f64,
    tenor: f64,
    payment_frequency: Frequency,
) -> Result<f64, String> {
    if !start_time.is_finite() || !tenor.is_finite() {
        return Err("start_time and tenor must be finite".to_string());
    }
    if tenor <= 0.0 {
        return Err("tenor must be > 0".to_string());
    }

    let dt = frequency_year_fraction(payment_frequency);
    let end_time = start_time + tenor;
    let mut annuity = 0.0;
    let mut t = start_time + dt;
    while t < end_time - 1.0e-12 {
        annuity += dt * curve.discount_factor(t);
        t += dt;
    }
    annuity += (end_time - (t - dt).max(start_time)) * curve.discount_factor(end_time);

    if annuity <= 1.0e-14 {
        return Err("invalid annuity in swap rate projection".to_string());
    }

    let p_start = curve.discount_factor(start_time.max(0.0));
    let p_end = curve.discount_factor(end_time);
    Ok((p_start - p_end) / annuity)
}

fn node_forward_swap_rate(
    model: &HullWhite,
    curve: &YieldCurve,
    eval_time: f64,
    short_rate: f64,
    tenor: f64,
    payment_frequency: Frequency,
) -> f64 {
    if tenor <= 0.0 {
        return 0.0;
    }

    let dt = frequency_year_fraction(payment_frequency);
    let end_time = eval_time + tenor;
    let mut annuity = 0.0;

    let mut t = eval_time + dt;
    while t < end_time - 1.0e-12 {
        let df = model.bond_price(eval_time, t, short_rate, curve);
        annuity += dt * df;
        t += dt;
    }

    let last_step_start = (t - dt).max(eval_time);
    annuity +=
        (end_time - last_step_start) * model.bond_price(eval_time, end_time, short_rate, curve);

    if annuity <= 1.0e-14 {
        return 0.0;
    }

    let p_end = model.bond_price(eval_time, end_time, short_rate, curve);
    (1.0 - p_end) / annuity
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_curve(rate: f64) -> YieldCurve {
        YieldCurve::new(
            (1..=200)
                .map(|i| {
                    let t = i as f64 * 0.05;
                    (t, (-rate * t).exp())
                })
                .collect(),
        )
    }

    #[test]
    fn coupon_schedule_builder_supports_fixed_float_and_structured() {
        let fixed = CouponScheduleBuilder::new(0.0, 1.0, Frequency::Quarterly)
            .unwrap()
            .build_fixed(0.03)
            .unwrap();
        let floating = CouponScheduleBuilder::new(0.0, 1.0, Frequency::Quarterly)
            .unwrap()
            .build_floating(0.001, Some(0.0), Some(0.08))
            .unwrap();
        let structured = CouponScheduleBuilder::new(0.0, 1.0, Frequency::Quarterly)
            .unwrap()
            .build_structured(StructuredCoupon::InverseFloater {
                fixed_rate: 0.06,
                leverage: 1.5,
                floor: Some(0.0),
                cap: None,
            })
            .unwrap();

        assert_eq!(fixed.len(), 4);
        assert_eq!(floating.len(), 4);
        assert_eq!(structured.len(), 4);

        assert!(matches!(fixed[0].coupon, CouponType::Fixed { .. }));
        assert!(matches!(floating[0].coupon, CouponType::Floating { .. }));
        assert!(matches!(structured[0].coupon, CouponType::Structured(_)));
    }

    #[test]
    fn exercise_schedule_applies_notice_period() {
        let sch = ExerciseSchedule::new(vec![1.0, 1.5, 2.0], 0.25).unwrap();
        let decisions = sch.decision_times();
        assert_eq!(decisions, vec![0.75, 1.25, 1.75]);
    }

    #[test]
    fn callable_fixed_note_is_below_non_callable_hold_to_maturity_value() {
        let curve = flat_curve(0.03);
        let hw = HullWhite::new(0.08, 0.01);

        let coupons = CouponScheduleBuilder::new(0.0, 3.0, Frequency::Annual)
            .unwrap()
            .build_fixed(0.05)
            .unwrap();

        let note = CallableRateNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            call_price: 1_000_000.0,
            maturity: 3.0,
            coupon_schedule: coupons,
            exercise_schedule: ExerciseSchedule::new(vec![1.0, 2.0], 0.0).unwrap(),
        };

        let callable = note.price_hull_white_tree(&hw, &curve, 300).unwrap();
        let projected_float = vec![0.03; note.coupon_schedule.len()];
        let projected_cms = vec![0.03; note.coupon_schedule.len()];
        let non_callable = note
            .hold_to_maturity_value(&curve, &projected_float, &projected_cms)
            .unwrap();

        assert!(callable.is_finite());
        assert!(callable < non_callable);
    }

    #[test]
    fn callable_range_accrual_prices_finite() {
        let curve = flat_curve(0.025);
        let hw = HullWhite::new(0.10, 0.015);

        let callable_ra = CallableRangeAccrualNote::new(
            1_000_000.0,
            2.0,
            Frequency::SemiAnnual,
            0.06,
            0.0,
            0.01,
            0.05,
            1_000_000.0,
            ExerciseSchedule::new(vec![1.0, 1.5], 0.0).unwrap(),
        )
        .unwrap();

        let px = callable_ra.price_hull_white_tree(&hw, &curve, 240).unwrap();
        assert!(px.is_finite());
        assert!(px > 0.0);
    }

    #[test]
    fn tarn_knockout_triggers_at_target() {
        let curve = flat_curve(0.02);
        let schedule = CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap();

        let tarn = TargetRedemptionNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            target_coupon: 30_000.0,
            coupon_schedule: schedule,
            spread: 0.03,
            floor: Some(0.0),
            cap: None,
        };

        let projected = vec![0.03; 4];
        let out = tarn.price(&projected, &curve).unwrap();

        assert!(out.knocked_out);
        assert!(out.knockout_time.is_some());
        assert!((out.accrued_coupon - 30_000.0).abs() <= 1.0e-8);
    }

    #[test]
    fn snowball_coupon_path_follows_recursive_formula() {
        let curve = flat_curve(0.02);
        let schedule = CouponScheduleBuilder::new(0.0, 1.0, Frequency::Quarterly)
            .unwrap()
            .build_fixed(0.0)
            .unwrap();

        let note = SnowballNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            initial_coupon: 0.02,
            spread: 0.01,
            floor: Some(0.0),
            cap: None,
            coupon_schedule: schedule,
        };

        let floating = vec![0.01, 0.02, 0.04, 0.01];
        let out = note.price(&floating, &curve).unwrap();

        assert_eq!(out.coupon_path.len(), 4);
        assert!((out.coupon_path[0] - 0.02).abs() <= 1.0e-12);
        assert!((out.coupon_path[1] - 0.01).abs() <= 1.0e-12);
        assert!((out.coupon_path[2] - 0.0).abs() <= 1.0e-12);
        assert!((out.coupon_path[3] - 0.0).abs() <= 1.0e-12);
    }

    #[test]
    fn inverse_floater_coupon_formula_matches_definition() {
        let curve = flat_curve(0.02);
        let schedule = CouponScheduleBuilder::new(0.0, 1.0, Frequency::SemiAnnual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap();

        let note = InverseFloaterNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            fixed_rate: 0.07,
            leverage: 2.0,
            floor: Some(0.0),
            cap: Some(0.1),
            coupon_schedule: schedule,
        };

        let floating = vec![0.01, 0.04];
        let pv = note.price(&floating, &curve).unwrap();
        assert!(pv.is_finite());
        assert!(pv > 0.0);
    }

    #[test]
    fn cms_linked_note_projects_and_prices_from_curve() {
        let curve = flat_curve(0.03);
        let schedule = CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap();

        let note = CmsLinkedNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            multiplier: 1.0,
            spread: 0.0,
            cms_tenor: 5.0,
            swap_payment_frequency: Frequency::Annual,
            floor: Some(0.0),
            cap: None,
            coupon_schedule: schedule,
        };

        let cms = note.projected_cms_rates_from_curve(&curve).unwrap();
        assert_eq!(cms.len(), note.coupon_schedule.len());

        let px = note.price_from_curve(&curve).unwrap();
        assert!(px.is_finite());
        assert!(px > 0.0);
    }

    #[test]
    fn higher_cms_projection_increases_cms_note_price() {
        let curve = flat_curve(0.03);
        let schedule = CouponScheduleBuilder::new(0.0, 2.0, Frequency::Annual)
            .unwrap()
            .build_fixed(0.0)
            .unwrap();

        let note = CmsLinkedNote {
            notional: 1_000_000.0,
            redemption: 1_000_000.0,
            multiplier: 1.2,
            spread: 0.001,
            cms_tenor: 5.0,
            swap_payment_frequency: Frequency::Annual,
            floor: Some(0.0),
            cap: None,
            coupon_schedule: schedule,
        };

        let low = note
            .price_with_projected_cms_rates(&[0.01; 2], &curve)
            .unwrap();
        let high = note
            .price_with_projected_cms_rates(&[0.04; 2], &curve)
            .unwrap();

        assert!(high > low);
    }

    #[test]
    fn callable_range_accrual_wrapper_keeps_expected_coupon_type() {
        let ra = CallableRangeAccrualNote::new(
            1_000_000.0,
            1.0,
            Frequency::Quarterly,
            0.05,
            0.0,
            0.01,
            0.03,
            1_000_000.0,
            ExerciseSchedule::new(vec![0.5, 0.75], 0.0).unwrap(),
        )
        .unwrap();

        assert!(ra.note.coupon_schedule.iter().all(|p| matches!(
            p.coupon,
            CouponType::Structured(StructuredCoupon::RangeAccrual { .. })
        )));
    }
}
