//! Module `rates::bond`.
//!
//! Implements bond abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `FixedRateBond` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use crate::rates::{DayCountConvention, YieldCurve};

/// Plain fixed-rate bond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedRateBond {
    /// Notional/face amount.
    pub face_value: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Coupon payments per year.
    pub frequency: u32,
    /// Maturity in years.
    pub maturity: f64,
    /// Accrual day-count convention.
    pub day_count: DayCountConvention,
}

impl FixedRateBond {
    /// Present value including accrued coupon.
    pub fn dirty_price(&self, curve: &YieldCurve) -> f64 {
        self.cashflows()
            .iter()
            .map(|(t, cf)| cf * self.discount_at(curve, *t))
            .sum()
    }

    /// Present value excluding accrued coupon at settlement.
    pub fn clean_price(&self, curve: &YieldCurve, settlement: f64) -> f64 {
        self.dirty_price(curve) - self.accrued_interest(settlement)
    }

    /// Coupon accrued from the last payment date up to settlement.
    pub fn accrued_interest(&self, settlement: f64) -> f64 {
        if self.frequency == 0 || settlement <= 0.0 || settlement >= self.maturity {
            return 0.0;
        }

        let period = 1.0 / self.frequency as f64;
        let last_coupon = (settlement / period).floor() * period;
        let next_coupon = (last_coupon + period).min(self.maturity);
        if next_coupon <= last_coupon {
            return 0.0;
        }

        let coupon = self.face_value * self.coupon_rate / self.frequency as f64;
        let accrual = (settlement - last_coupon) / (next_coupon - last_coupon);
        coupon * accrual.clamp(0.0, 1.0)
    }

    /// Macaulay duration in years under the supplied curve.
    pub fn duration(&self, curve: &YieldCurve) -> f64 {
        let price = self.dirty_price(curve);
        if price <= 0.0 {
            return 0.0;
        }

        self.cashflows()
            .iter()
            .map(|(t, cf)| t * cf * self.discount_at(curve, *t))
            .sum::<f64>()
            / price
    }

    /// Convexity in years squared under the supplied curve.
    pub fn convexity(&self, curve: &YieldCurve) -> f64 {
        let price = self.dirty_price(curve);
        if price <= 0.0 {
            return 0.0;
        }

        self.cashflows()
            .iter()
            .map(|(t, cf)| t * t * cf * self.discount_at(curve, *t))
            .sum::<f64>()
            / price
    }

    /// Yield-to-maturity solved by Newton-Raphson.
    pub fn ytm(&self, market_price: f64) -> f64 {
        if self.frequency == 0 || market_price <= 0.0 {
            return f64::NAN;
        }

        let cashflows = self.cashflows();
        if cashflows.is_empty() {
            return f64::NAN;
        }

        let m = self.frequency as f64;
        let mut y = self.coupon_rate.max(1.0e-6);

        for _ in 0..100 {
            let base = (1.0 + y / m).max(1.0e-12);
            let mut f = -market_price;
            let mut fp = 0.0;

            for (t, cf) in &cashflows {
                let n = m * *t;
                let discount = base.powf(-n);
                f += cf * discount;
                fp += cf * discount * (-n) / (m * base);
            }

            if fp.abs() <= 1.0e-14 {
                break;
            }

            let step = f / fp;
            y -= step;
            if step.abs() <= 1.0e-12 {
                break;
            }

            if y <= -0.99 * m {
                y = -0.99 * m;
            }
        }

        y
    }

    fn discount_at(&self, curve: &YieldCurve, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let m = self.frequency.max(1) as f64;
        let z = curve.zero_rate(t);
        (1.0 + z / m).powf(-m * t)
    }

    fn cashflows(&self) -> Vec<(f64, f64)> {
        if self.frequency == 0 || self.maturity <= 0.0 {
            return Vec::new();
        }

        let period = 1.0 / self.frequency as f64;
        let coupon = self.face_value * self.coupon_rate / self.frequency as f64;

        let mut out = Vec::new();
        let mut t = period;
        while t < self.maturity - 1.0e-12 {
            out.push((t, coupon));
            t += period;
        }
        out.push((self.maturity, coupon + self.face_value));

        out
    }
}
