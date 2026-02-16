use crate::rates::YieldCurve;

/// Overnight index swap: fixed leg vs overnight floating leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OvernightIndexSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub float_spread: f64,
    pub tenor: f64,
}

impl OvernightIndexSwap {
    /// PV of fixed leg discounted on OIS curve.
    pub fn fixed_leg_pv(&self, ois_discount_curve: &YieldCurve) -> f64 {
        if self.notional <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for (start, end) in annual_periods(self.tenor) {
            let accrual = end - start;
            let df = ois_discount_curve.discount_factor(end);
            pv += self.notional * self.fixed_rate * accrual * df;
        }

        pv
    }

    /// PV of overnight floating leg using projection curve and OIS discounting.
    pub fn floating_leg_pv(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
    ) -> f64 {
        if self.notional <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for (start, end) in annual_periods(self.tenor) {
            let accrual = end - start;
            if accrual <= 0.0 {
                continue;
            }

            let compounded_overnight = overnight_projection_curve.forward_rate(start, end);
            let df = ois_discount_curve.discount_factor(end);
            pv += self.notional * (compounded_overnight + self.float_spread) * accrual * df;
        }

        pv
    }

    /// NPV under OIS discounting.
    ///
    /// If `pay_fixed` is true, NPV = receive-floating - pay-fixed.
    pub fn npv(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
        pay_fixed: bool,
    ) -> f64 {
        let fixed = self.fixed_leg_pv(ois_discount_curve);
        let floating = self.floating_leg_pv(ois_discount_curve, overnight_projection_curve);

        if pay_fixed {
            floating - fixed
        } else {
            fixed - floating
        }
    }

    /// Fixed rate that sets payer-fixed OIS NPV to zero.
    pub fn par_fixed_rate(
        &self,
        ois_discount_curve: &YieldCurve,
        overnight_projection_curve: &YieldCurve,
    ) -> f64 {
        if self.notional <= 0.0 || self.tenor <= 0.0 {
            return f64::NAN;
        }

        let annuity: f64 = annual_periods(self.tenor)
            .iter()
            .map(|(start, end)| (end - start) * ois_discount_curve.discount_factor(*end))
            .sum();

        if annuity <= 0.0 {
            return f64::NAN;
        }

        self.floating_leg_pv(ois_discount_curve, overnight_projection_curve)
            / (self.notional * annuity)
    }
}

/// Basis swap: two floating legs with different reset tenors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisSwap {
    pub notional: f64,
    pub spread_on_short_leg: f64,
    pub tenor: f64,
    /// Payments per year for short leg (e.g. 4 for 3M).
    pub short_leg_payments_per_year: u32,
    /// Payments per year for long leg (e.g. 2 for 6M).
    pub long_leg_payments_per_year: u32,
}

impl BasisSwap {
    /// NPV with OIS discounting and separate IBOR projection curves.
    ///
    /// If `pay_short_plus_spread` is true: NPV = receive long leg - pay short leg(+spread).
    pub fn npv(
        &self,
        ois_discount_curve: &YieldCurve,
        short_ibor_projection_curve: &YieldCurve,
        long_ibor_projection_curve: &YieldCurve,
        pay_short_plus_spread: bool,
    ) -> f64 {
        let short_leg = self.floating_leg_pv(
            ois_discount_curve,
            short_ibor_projection_curve,
            self.short_leg_payments_per_year,
            self.spread_on_short_leg,
        );

        let long_leg = self.floating_leg_pv(
            ois_discount_curve,
            long_ibor_projection_curve,
            self.long_leg_payments_per_year,
            0.0,
        );

        if pay_short_plus_spread {
            long_leg - short_leg
        } else {
            short_leg - long_leg
        }
    }

    /// Fair spread on short leg that sets the basis swap NPV to zero
    /// for a receive-long / pay-short+spread position.
    pub fn par_spread_on_short_leg(
        &self,
        ois_discount_curve: &YieldCurve,
        short_ibor_projection_curve: &YieldCurve,
        long_ibor_projection_curve: &YieldCurve,
    ) -> f64 {
        if self.notional <= 0.0 || self.tenor <= 0.0 || self.short_leg_payments_per_year == 0 {
            return f64::NAN;
        }

        let short_no_spread = self.floating_leg_pv(
            ois_discount_curve,
            short_ibor_projection_curve,
            self.short_leg_payments_per_year,
            0.0,
        );
        let long_leg = self.floating_leg_pv(
            ois_discount_curve,
            long_ibor_projection_curve,
            self.long_leg_payments_per_year,
            0.0,
        );

        let spread_pv01 = self.spread_pv01(ois_discount_curve, self.short_leg_payments_per_year);
        if spread_pv01 <= 0.0 {
            return f64::NAN;
        }

        (long_leg - short_no_spread) / spread_pv01
    }

    fn floating_leg_pv(
        &self,
        ois_discount_curve: &YieldCurve,
        projection_curve: &YieldCurve,
        payments_per_year: u32,
        spread: f64,
    ) -> f64 {
        if self.notional <= 0.0 || self.tenor <= 0.0 || payments_per_year == 0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for (start, end) in periods(self.tenor, payments_per_year) {
            let accrual = end - start;
            if accrual <= 0.0 {
                continue;
            }

            let forward = projection_curve.forward_rate(start, end);
            let df = ois_discount_curve.discount_factor(end);
            pv += self.notional * (forward + spread) * accrual * df;
        }

        pv
    }

    fn spread_pv01(&self, ois_discount_curve: &YieldCurve, payments_per_year: u32) -> f64 {
        periods(self.tenor, payments_per_year)
            .iter()
            .map(|(start, end)| {
                let accrual = end - start;
                self.notional * accrual * ois_discount_curve.discount_factor(*end)
            })
            .sum()
    }
}

fn annual_periods(tenor: f64) -> Vec<(f64, f64)> {
    periods(tenor, 1)
}

fn periods(tenor: f64, payments_per_year: u32) -> Vec<(f64, f64)> {
    if tenor <= 0.0 || payments_per_year == 0 {
        return Vec::new();
    }

    let step = 1.0 / payments_per_year as f64;
    let mut out = Vec::new();
    let mut start = 0.0;

    while start < tenor - 1.0e-12 {
        let end = (start + step).min(tenor);
        out.push((start, end));
        start = end;
    }

    out
}
