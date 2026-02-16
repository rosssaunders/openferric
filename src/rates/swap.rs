use chrono::NaiveDate;

use crate::rates::schedule::{Frequency, generate_schedule};
use crate::rates::{DayCountConvention, YieldCurve, year_fraction};

/// Plain-vanilla fixed-for-floating interest-rate swap.
#[derive(Debug, Clone, PartialEq)]
pub struct InterestRateSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub float_spread: f64,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub fixed_freq: Frequency,
    pub float_freq: Frequency,
    pub fixed_day_count: DayCountConvention,
    pub float_day_count: DayCountConvention,
}

impl InterestRateSwap {
    /// Starts a swap builder with market-style defaults.
    pub fn builder() -> SwapBuilder {
        SwapBuilder::default()
    }

    /// PV of the fixed leg as discounted coupon cashflows.
    pub fn fixed_leg_pv(&self, curve: &YieldCurve) -> f64 {
        let schedule = generate_schedule(self.start_date, self.end_date, self.fixed_freq);
        if schedule.len() < 2 {
            return 0.0;
        }

        schedule
            .windows(2)
            .map(|period| {
                let accrual = year_fraction(period[0], period[1], self.fixed_day_count);
                if accrual <= 0.0 {
                    return 0.0;
                }

                let pay_time = year_fraction(self.start_date, period[1], self.fixed_day_count);
                let df = curve.discount_factor(pay_time.max(0.0));
                self.notional * self.fixed_rate * accrual * df
            })
            .sum()
    }

    /// PV of the floating leg as discounted forward coupons.
    pub fn float_leg_pv(&self, curve: &YieldCurve) -> f64 {
        let schedule = generate_schedule(self.start_date, self.end_date, self.float_freq);
        if schedule.len() < 2 {
            return 0.0;
        }

        schedule
            .windows(2)
            .map(|period| {
                let accrual = year_fraction(period[0], period[1], self.float_day_count);
                if accrual <= 0.0 {
                    return 0.0;
                }

                let t1 = year_fraction(self.start_date, period[0], self.float_day_count).max(0.0);
                let t2 = year_fraction(self.start_date, period[1], self.float_day_count).max(0.0);
                if t2 <= t1 {
                    return 0.0;
                }

                let fwd = curve.forward_rate(t1, t2);
                let df = curve.discount_factor(t2);
                self.notional * (fwd + self.float_spread) * accrual * df
            })
            .sum()
    }

    /// NPV for a receiver-float / payer-fixed swap.
    pub fn npv(&self, curve: &YieldCurve) -> f64 {
        self.float_leg_pv(curve) - self.fixed_leg_pv(curve)
    }

    /// Fixed par rate that sets the swap NPV to zero.
    pub fn par_rate(&self, curve: &YieldCurve) -> f64 {
        let annuity = self.fixed_leg_annuity(curve);
        if annuity <= 0.0 {
            return f64::NAN;
        }
        self.float_leg_pv(curve) / annuity
    }

    /// One-basis-point parallel curve-shift sensitivity.
    pub fn dv01(&self, curve: &YieldCurve) -> f64 {
        let bumped_curve = bump_curve_parallel(curve, 1.0e-4);
        self.npv(&bumped_curve) - self.npv(curve)
    }

    fn fixed_leg_annuity(&self, curve: &YieldCurve) -> f64 {
        let schedule = generate_schedule(self.start_date, self.end_date, self.fixed_freq);
        if schedule.len() < 2 {
            return 0.0;
        }

        self.notional
            * schedule
                .windows(2)
                .map(|period| {
                    let accrual = year_fraction(period[0], period[1], self.fixed_day_count);
                    if accrual <= 0.0 {
                        return 0.0;
                    }

                    let pay_time =
                        year_fraction(self.start_date, period[1], self.fixed_day_count).max(0.0);
                    accrual * curve.discount_factor(pay_time)
                })
                .sum::<f64>()
    }
}

fn bump_curve_parallel(curve: &YieldCurve, bump: f64) -> YieldCurve {
    let bumped_nodes = curve
        .tenors
        .iter()
        .map(|(t, df)| {
            let z = if *t > 0.0 { -df.ln() / t } else { 0.0 };
            let bumped_df = (-(z + bump) * t).exp();
            (*t, bumped_df)
        })
        .collect();
    YieldCurve::new(bumped_nodes)
}

/// Builder for [`InterestRateSwap`].
#[derive(Debug, Clone)]
pub struct SwapBuilder {
    notional: f64,
    fixed_rate: f64,
    float_spread: f64,
    start_date: NaiveDate,
    end_date: NaiveDate,
    fixed_freq: Frequency,
    float_freq: Frequency,
    fixed_day_count: DayCountConvention,
    float_day_count: DayCountConvention,
}

impl Default for SwapBuilder {
    fn default() -> Self {
        Self {
            notional: 1_000_000.0,
            fixed_rate: 0.0,
            float_spread: 0.0,
            start_date: NaiveDate::from_ymd_opt(2025, 1, 1).expect("valid default start date"),
            end_date: NaiveDate::from_ymd_opt(2030, 1, 1).expect("valid default end date"),
            fixed_freq: Frequency::Annual,
            float_freq: Frequency::Quarterly,
            fixed_day_count: DayCountConvention::Act365Fixed,
            float_day_count: DayCountConvention::Act360,
        }
    }
}

impl SwapBuilder {
    pub fn notional(mut self, notional: f64) -> Self {
        self.notional = notional;
        self
    }

    pub fn fixed_rate(mut self, fixed_rate: f64) -> Self {
        self.fixed_rate = fixed_rate;
        self
    }

    pub fn float_spread(mut self, float_spread: f64) -> Self {
        self.float_spread = float_spread;
        self
    }

    pub fn start_date(mut self, start_date: NaiveDate) -> Self {
        self.start_date = start_date;
        self
    }

    pub fn end_date(mut self, end_date: NaiveDate) -> Self {
        self.end_date = end_date;
        self
    }

    pub fn fixed_freq(mut self, fixed_freq: Frequency) -> Self {
        self.fixed_freq = fixed_freq;
        self
    }

    pub fn float_freq(mut self, float_freq: Frequency) -> Self {
        self.float_freq = float_freq;
        self
    }

    pub fn fixed_day_count(mut self, fixed_day_count: DayCountConvention) -> Self {
        self.fixed_day_count = fixed_day_count;
        self
    }

    pub fn float_day_count(mut self, float_day_count: DayCountConvention) -> Self {
        self.float_day_count = float_day_count;
        self
    }

    pub fn build(self) -> InterestRateSwap {
        InterestRateSwap {
            notional: self.notional,
            fixed_rate: self.fixed_rate,
            float_spread: self.float_spread,
            start_date: self.start_date,
            end_date: self.end_date,
            fixed_freq: self.fixed_freq,
            float_freq: self.float_freq,
            fixed_day_count: self.fixed_day_count,
            float_day_count: self.float_day_count,
        }
    }
}
