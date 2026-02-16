use chrono::NaiveDate;

use crate::math::normal_cdf;
use crate::rates::schedule::generate_schedule;
use crate::rates::{DayCountConvention, Frequency, YieldCurve, year_fraction};

/// Black-76 cap/floor instrument on forward rates.
#[derive(Debug, Clone, PartialEq)]
pub struct CapFloor {
    pub notional: f64,
    pub strike: f64,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub frequency: Frequency,
    pub day_count: DayCountConvention,
    pub is_cap: bool,
}

impl CapFloor {
    /// Black caplet price for one accrual period.
    pub fn black_caplet(
        notional: f64,
        discount_factor: f64,
        accrual: f64,
        forward_rate: f64,
        strike: f64,
        vol: f64,
        expiry: f64,
    ) -> f64 {
        black_optionlet(
            notional,
            discount_factor,
            accrual,
            forward_rate,
            strike,
            vol,
            expiry,
            true,
        )
    }

    /// Black floorlet price for one accrual period.
    pub fn black_floorlet(
        notional: f64,
        discount_factor: f64,
        accrual: f64,
        forward_rate: f64,
        strike: f64,
        vol: f64,
        expiry: f64,
    ) -> f64 {
        black_optionlet(
            notional,
            discount_factor,
            accrual,
            forward_rate,
            strike,
            vol,
            expiry,
            false,
        )
    }

    /// Price of a single caplet/floorlet on the supplied period.
    pub fn optionlet_price(
        &self,
        curve: &YieldCurve,
        vol: f64,
        period_start: NaiveDate,
        period_end: NaiveDate,
    ) -> f64 {
        let accrual = year_fraction(period_start, period_end, self.day_count);
        if accrual <= 0.0 {
            return 0.0;
        }

        let t1 = year_fraction(self.start_date, period_start, self.day_count).max(0.0);
        let t2 = year_fraction(self.start_date, period_end, self.day_count).max(0.0);
        if t2 <= t1 {
            return 0.0;
        }

        let df1 = curve.discount_factor(t1);
        let df2 = curve.discount_factor(t2);
        if df2 <= 0.0 {
            return 0.0;
        }

        let forward = (df1 / df2 - 1.0) / accrual;

        if self.is_cap {
            Self::black_caplet(self.notional, df2, accrual, forward, self.strike, vol, t1)
        } else {
            Self::black_floorlet(self.notional, df2, accrual, forward, self.strike, vol, t1)
        }
    }

    /// Cap/Floor price as the sum of caplets/floorlets.
    pub fn price(&self, curve: &YieldCurve, vol: f64) -> f64 {
        let schedule = generate_schedule(self.start_date, self.end_date, self.frequency);
        if schedule.len() < 2 {
            return 0.0;
        }

        schedule
            .windows(2)
            .map(|period| self.optionlet_price(curve, vol, period[0], period[1]))
            .sum()
    }

    /// Underlying swap NPV at strike `K`: `sum DF*delta*(F-K) * notional`.
    pub fn swap_npv(&self, curve: &YieldCurve) -> f64 {
        let schedule = generate_schedule(self.start_date, self.end_date, self.frequency);
        if schedule.len() < 2 {
            return 0.0;
        }

        schedule
            .windows(2)
            .map(|period| {
                let accrual = year_fraction(period[0], period[1], self.day_count);
                if accrual <= 0.0 {
                    return 0.0;
                }

                let t1 = year_fraction(self.start_date, period[0], self.day_count).max(0.0);
                let t2 = year_fraction(self.start_date, period[1], self.day_count).max(0.0);
                if t2 <= t1 {
                    return 0.0;
                }

                let df1 = curve.discount_factor(t1);
                let df2 = curve.discount_factor(t2);
                if df2 <= 0.0 {
                    return 0.0;
                }

                let forward = (df1 / df2 - 1.0) / accrual;
                self.notional * df2 * accrual * (forward - self.strike)
            })
            .sum()
    }

    /// Implied Black volatility from market cap/floor price.
    pub fn implied_vol(&self, market_price: f64, curve: &YieldCurve) -> f64 {
        if market_price < 0.0 {
            return f64::NAN;
        }

        let intrinsic = self.price(curve, 0.0);
        if (market_price - intrinsic).abs() <= 1.0e-12 || market_price < intrinsic {
            return 0.0;
        }

        let mut lo = 1.0e-6;
        let mut hi = 5.0;
        let mut flo = self.price(curve, lo) - market_price;
        let fhi = self.price(curve, hi) - market_price;

        if flo * fhi > 0.0 {
            return f64::NAN;
        }

        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let fm = self.price(curve, mid) - market_price;
            if fm.abs() <= 1.0e-10 {
                return mid;
            }

            if flo * fm <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
                flo = fm;
            }
        }

        0.5 * (lo + hi)
    }
}

#[allow(clippy::too_many_arguments)]
fn black_optionlet(
    notional: f64,
    discount_factor: f64,
    accrual: f64,
    forward_rate: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    is_cap: bool,
) -> f64 {
    if notional <= 0.0
        || discount_factor <= 0.0
        || accrual <= 0.0
        || forward_rate <= 0.0
        || strike <= 0.0
    {
        return 0.0;
    }

    let scale = notional * discount_factor * accrual;
    if vol <= 0.0 || expiry <= 0.0 {
        let intrinsic = if is_cap {
            (forward_rate - strike).max(0.0)
        } else {
            (strike - forward_rate).max(0.0)
        };
        return scale * intrinsic;
    }

    let sig_sqrt_t = vol * expiry.sqrt();
    let d1 = ((forward_rate / strike).ln() + 0.5 * vol * vol * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let option_value = if is_cap {
        forward_rate * normal_cdf(d1) - strike * normal_cdf(d2)
    } else {
        strike * normal_cdf(-d2) - forward_rate * normal_cdf(-d1)
    };

    scale * option_value
}
