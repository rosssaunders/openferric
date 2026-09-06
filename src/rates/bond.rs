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
use crate::rates::schedule::year_fraction_periods;
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
    /// Maturity in years, with a prorated short final coupon when off-grid.
    pub maturity: f64,
    /// Informational day-count convention metadata.
    ///
    /// The bond model is dateless (all times are year fractions), so accrual
    /// is computed as the elapsed fraction of the current coupon period
    /// (equivalent to Act/Act-ICMA on the bond's abstract unit periods).
    /// Date-based conventions such as Act/360 or 30E/360 cannot be applied
    /// without calendar dates; this field records intent only and does not
    /// affect pricing or accrual.
    pub day_count: DayCountConvention,
}

impl FixedRateBond {
    /// Dirty price at time zero: the `t = 0` present value of all cashflows.
    ///
    /// Equivalent to [`Self::dirty_price_at`] with `settlement = 0`.
    pub fn dirty_price(&self, curve: &YieldCurve) -> f64 {
        self.cashflows()
            .iter()
            .map(|(t, cf)| cf * self.discount_at(curve, *t))
            .sum()
    }

    /// Dirty price at `settlement`: the value at `settlement` of all
    /// cashflows paid strictly after `settlement`.
    ///
    /// Cashflows are forward-valued with the curve's implied forward discount
    /// factor `DF(s, t) = DF(0, t) / DF(0, s)`; coupons paid on or before
    /// `settlement` (including a coupon paid exactly at `settlement`) are
    /// excluded. For `settlement <= 0` this reduces exactly to
    /// [`Self::dirty_price`].
    pub fn dirty_price_at(&self, curve: &YieldCurve, settlement: f64) -> f64 {
        if settlement <= 0.0 {
            return self.dirty_price(curve);
        }

        // Allow small floating-point differences when excluding a coupon
        // that falls exactly on the settlement date.
        let pv_at_zero: f64 = self
            .cashflows()
            .iter()
            .filter(|(t, _)| *t > settlement + 1.0e-12)
            .map(|(t, cf)| cf * self.discount_at(curve, *t))
            .sum();

        if pv_at_zero == 0.0 {
            return 0.0;
        }

        pv_at_zero / curve.discount_factor(settlement)
    }

    /// Clean price at `settlement`: [`Self::dirty_price_at`] minus
    /// [`Self::accrued_interest`] at the same settlement time.
    pub fn clean_price(&self, curve: &YieldCurve, settlement: f64) -> f64 {
        self.dirty_price_at(curve, settlement) - self.accrued_interest(settlement)
    }

    /// Coupon accrued from the last payment date up to settlement.
    ///
    /// Accrual is the elapsed fraction of the current coupon period
    /// (Act/Act-ICMA on the bond's abstract unit periods); the `day_count`
    /// field is informational metadata and does not alter this fraction
    /// because the dateless model has no calendar dates to apply it to.
    pub fn accrued_interest(&self, settlement: f64) -> f64 {
        if self.frequency == 0 || settlement <= 0.0 || settlement >= self.maturity {
            return 0.0;
        }

        let period = 1.0 / self.frequency as f64;
        let last_coupon = (settlement / period).floor() * period;
        let elapsed = settlement - last_coupon;
        if elapsed <= 1.0e-12 || period - elapsed <= 1.0e-12 {
            return 0.0;
        }
        self.face_value * self.coupon_rate * elapsed
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

    /// Yield-to-maturity solved by bracketed bisection in log-compounding space.
    ///
    /// This is the quoted-yield path: the solved yield `y` is compounded
    /// `frequency` times per year, i.e. cashflows discount as
    /// `(1 + y/m)^(-m t)`. It is intentionally distinct from curve-based
    /// pricing, which uses `YieldCurve::discount_factor` directly.
    pub fn ytm(&self, market_price: f64) -> f64 {
        if self.frequency == 0
            || !market_price.is_finite()
            || market_price <= 0.0
            || !self.face_value.is_finite()
            || self.face_value <= 0.0
            || !self.coupon_rate.is_finite()
            || self.coupon_rate < 0.0
        {
            return f64::NAN;
        }

        let cashflows = self.cashflows();
        if cashflows.is_empty() {
            return f64::NAN;
        }

        let frequency = self.frequency as f64;
        let price = |log_base: f64| -> f64 {
            cashflows
                .iter()
                .map(|(time, amount)| amount * (-frequency * time * log_base).exp())
                .sum()
        };
        let mut lower = -1.0;
        let mut upper = 1.0;
        for _ in 0..64 {
            if price(lower) >= market_price && price(upper) <= market_price {
                break;
            }
            if price(lower) < market_price {
                lower *= 2.0;
            }
            if price(upper) > market_price {
                upper *= 2.0;
            }
        }
        if price(lower) < market_price || price(upper) > market_price {
            return f64::NAN;
        }
        for _ in 0..200 {
            let middle = 0.5 * (lower + upper);
            if middle == lower || middle == upper {
                break;
            }
            if price(middle) > market_price {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        frequency * (0.5 * (lower + upper)).exp_m1()
    }

    /// Discounts a cashflow at time `t` using the curve's own discount factor.
    ///
    /// The curve is the pricing primitive: `P(t) = curve.discount_factor(t)`.
    /// Discrete-compounding yield conventions only apply to the yield-based
    /// path ([`Self::ytm`]), never to curve-based pricing.
    fn discount_at(&self, curve: &YieldCurve, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        curve.discount_factor(t)
    }

    fn cashflows(&self) -> Vec<(f64, f64)> {
        if self.frequency == 0 || self.maturity <= 0.0 {
            return Vec::new();
        }

        year_fraction_periods(self.maturity, self.frequency as usize)
            .into_iter()
            .map(|(start, end)| {
                let coupon = self.face_value * self.coupon_rate * (end - start);
                (
                    end,
                    coupon
                        + if end == self.maturity {
                            self.face_value
                        } else {
                            0.0
                        },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat continuously-compounded curve with half-year nodes so log-linear
    /// interpolation reproduces `exp(-r t)` exactly at every time used below.
    fn flat_cc_curve(rate: f64, max_tenor: f64) -> YieldCurve {
        let n = (max_tenor * 2.0).ceil() as usize + 1;
        let points: Vec<(f64, f64)> = (1..=n)
            .map(|i| {
                let t = i as f64 * 0.5;
                (t, (-rate * t).exp())
            })
            .collect();
        YieldCurve::new(points)
    }

    fn sample_bond() -> FixedRateBond {
        FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.06,
            frequency: 2,
            maturity: 10.0,
            day_count: DayCountConvention::Act365Fixed,
        }
    }

    #[test]
    fn dirty_price_at_zero_settlement_matches_dirty_price() {
        let curve = flat_cc_curve(0.04, 12.0);
        let bond = sample_bond();
        let dirty = bond.dirty_price(&curve);
        assert!((bond.dirty_price_at(&curve, 0.0) - dirty).abs() < 1.0e-12);
        assert!((bond.clean_price(&curve, 0.0) - dirty).abs() < 1.0e-12);
    }

    #[test]
    fn dirty_price_at_excludes_coupon_paid_exactly_at_settlement() {
        let curve = flat_cc_curve(0.04, 12.0);
        let bond = sample_bond();

        // At s = 3.5 the coupon at t = 3.5 was just paid and must be
        // excluded. Hand-computed forward value of the 13 remaining coupons
        // of 3.0 at t = 4.0..10.0 plus 100 redemption, each discounted by
        // exp(-0.04 * (t - 3.5)): 111.105142823 (same anchor as
        // tests/audit_bond_clean_price.rs, hand computation documented there).
        let dirty = bond.dirty_price_at(&curve, 3.5);
        assert!((dirty - 111.105_142_823).abs() < 1.0e-9);
        // Accrued at a coupon date is zero, so clean == dirty there.
        assert!((bond.clean_price(&curve, 3.5) - dirty).abs() < 1.0e-12);
    }

    #[test]
    fn dirty_price_at_maturity_or_later_is_zero() {
        let curve = flat_cc_curve(0.04, 12.0);
        let bond = sample_bond();
        assert_eq!(bond.dirty_price_at(&curve, 10.0), 0.0);
        assert_eq!(bond.dirty_price_at(&curve, 11.0), 0.0);
        assert_eq!(bond.clean_price(&curve, 10.0), 0.0);
    }
}
