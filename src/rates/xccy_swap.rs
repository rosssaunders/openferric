//! Module `rates::xccy_swap`.
//!
//! Implements xccy swap abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 4, 6, and 7; Brigo and Mercurio (2006), curve and accrual identities around Eq. (4.2) and Eq. (7.1).
//!
//! Key types and purpose: `XccySwap` define the core data contracts for this module.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use this module for curve, accrual, and vanilla rates analytics; move to HJM/LMM or full XVA stacks for stochastic-rate or counterparty-intensive use cases.
use crate::rates::{Frequency, YieldCurve};

/// Cross-currency swap: fixed leg in currency 1 vs floating leg in currency 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XccySwap {
    /// Notional for fixed leg in currency 1.
    pub notional1: f64,
    /// Notional for floating leg in currency 2.
    pub notional2: f64,
    /// Fixed coupon rate paid/received on currency-1 notional.
    pub fixed_rate: f64,
    /// Spread added to currency-2 floating forward rate.
    pub float_spread: f64,
    /// Swap maturity in years.
    pub tenor: f64,
    /// Spot FX quoted as ccy1 per 1 ccy2.
    pub fx_spot: f64,
}

impl XccySwap {
    /// PV of the currency-1 fixed leg in currency 1.
    ///
    /// Includes coupon payments and terminal notional exchange.
    pub fn fixed_leg_pv_ccy1(&self, ccy1_discount_curve: &YieldCurve) -> f64 {
        self.fixed_leg_pv_ccy1_with_frequency(ccy1_discount_curve, Frequency::Annual)
    }

    /// PV of the currency-1 fixed leg with an explicit coupon frequency.
    ///
    /// This year-fraction API uses equal calendar-month periods represented as
    /// exact fractions of a year. A final stub is included when `tenor` is not
    /// an integer multiple of the requested period.
    pub fn fixed_leg_pv_ccy1_with_frequency(
        &self,
        ccy1_discount_curve: &YieldCurve,
        frequency: Frequency,
    ) -> f64 {
        if self.notional1 <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let annuity = coupon_annuity(ccy1_discount_curve, self.tenor, frequency);
        let principal = self.notional1 * ccy1_discount_curve.discount_factor(self.tenor);
        self.notional1 * self.fixed_rate * annuity + principal
    }

    /// PV of the currency-2 floating leg in currency 2 under a dual-curve setup.
    ///
    /// `ccy2_projection_curve` is used to project floating rates,
    /// `ccy2_discount_curve` is used for discounting.
    pub fn float_leg_pv_ccy2(
        &self,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
    ) -> f64 {
        self.float_leg_pv_ccy2_with_frequency(
            ccy2_discount_curve,
            ccy2_projection_curve,
            Frequency::Annual,
        )
    }

    /// PV of the currency-2 floating leg with an explicit coupon frequency.
    pub fn float_leg_pv_ccy2_with_frequency(
        &self,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        frequency: Frequency,
    ) -> f64 {
        if self.notional2 <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for (start, end) in coupon_periods(self.tenor, frequency) {
            let accrual = end - start;
            if accrual <= 0.0 {
                continue;
            }

            // Simple forward on the projection curve: (DF(s)/DF(e) - 1)/accrual.
            let df1 = ccy2_projection_curve.discount_factor(start);
            let df2 = ccy2_projection_curve.discount_factor(end);
            let fwd = (df1 / df2 - 1.0) / accrual;
            let df = ccy2_discount_curve.discount_factor(end);
            pv += self.notional2 * (fwd + self.float_spread) * accrual * df;
        }

        pv + self.notional2 * ccy2_discount_curve.discount_factor(self.tenor)
    }

    /// NPV in currency 1, converted through spot FX.
    ///
    /// If `pay_fixed_ccy1` is true, NPV = receive-float(ccy2)-pay-fixed(ccy1).
    /// Otherwise, NPV = receive-fixed(ccy1)-pay-float(ccy2).
    pub fn npv(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.npv_dual_curve(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_discount_curve,
            pay_fixed_ccy1,
        )
    }

    /// NPV in currency 1 with explicit discount/projection curves for currency 2.
    pub fn npv_dual_curve(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.npv_dual_curve_with_frequencies(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_projection_curve,
            Frequency::Annual,
            Frequency::Annual,
            pay_fixed_ccy1,
        )
    }

    /// NPV in currency 1 with explicit discount/projection curves and coupon
    /// frequencies for both legs.
    pub fn npv_dual_curve_with_frequencies(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        fixed_frequency: Frequency,
        float_frequency: Frequency,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        let fixed_leg_ccy1 =
            self.fixed_leg_pv_ccy1_with_frequency(ccy1_discount_curve, fixed_frequency);
        let float_leg_ccy2 = self.float_leg_pv_ccy2_with_frequency(
            ccy2_discount_curve,
            ccy2_projection_curve,
            float_frequency,
        );
        let float_leg_ccy1 = float_leg_ccy2 * self.fx_spot;

        if pay_fixed_ccy1 {
            float_leg_ccy1 - fixed_leg_ccy1
        } else {
            fixed_leg_ccy1 - float_leg_ccy1
        }
    }

    /// Fixed rate that makes a pay-fixed / receive-float trade have zero NPV.
    pub fn par_fixed_rate(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
    ) -> f64 {
        self.par_fixed_rate_with_frequencies(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_projection_curve,
            Frequency::Annual,
            Frequency::Annual,
        )
    }

    /// Fixed rate that makes the trade par for explicit fixed/floating coupon
    /// frequencies.
    pub fn par_fixed_rate_with_frequencies(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        fixed_frequency: Frequency,
        float_frequency: Frequency,
    ) -> f64 {
        if self.notional1 <= 0.0 || self.tenor <= 0.0 {
            return f64::NAN;
        }

        let annuity = coupon_annuity(ccy1_discount_curve, self.tenor, fixed_frequency);
        if annuity <= 0.0 {
            return f64::NAN;
        }

        let float_pv_ccy1 = self.float_leg_pv_ccy2_with_frequency(
            ccy2_discount_curve,
            ccy2_projection_curve,
            float_frequency,
        ) * self.fx_spot;
        let fixed_principal = self.notional1 * ccy1_discount_curve.discount_factor(self.tenor);

        (float_pv_ccy1 - fixed_principal) / (self.notional1 * annuity)
    }

    /// Mark-to-market NPV in currency 1 under a new spot FX level.
    pub fn mtm_basis_npv(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        current_fx_spot: f64,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        self.mtm_basis_npv_with_frequencies(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_projection_curve,
            current_fx_spot,
            Frequency::Annual,
            Frequency::Annual,
            pay_fixed_ccy1,
        )
    }

    /// Mark-to-market NPV with explicit fixed/floating coupon frequencies.
    pub fn mtm_basis_npv_with_frequencies(
        &self,
        ccy1_discount_curve: &YieldCurve,
        ccy2_discount_curve: &YieldCurve,
        ccy2_projection_curve: &YieldCurve,
        current_fx_spot: f64,
        fixed_frequency: Frequency,
        float_frequency: Frequency,
        pay_fixed_ccy1: bool,
    ) -> f64 {
        let mut shifted = *self;
        shifted.fx_spot = current_fx_spot;
        shifted.npv_dual_curve_with_frequencies(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_projection_curve,
            fixed_frequency,
            float_frequency,
            pay_fixed_ccy1,
        )
    }
}

fn coupon_periods(tenor: f64, frequency: Frequency) -> Vec<(f64, f64)> {
    if tenor <= 0.0 {
        return Vec::new();
    }

    let step = frequency.months() as f64 / 12.0;
    let mut out = Vec::with_capacity((tenor / step).ceil() as usize);
    let mut start = 0.0;
    while start < tenor - 1.0e-12 {
        let end = (start + step).min(tenor);
        out.push((start, end));
        start = end;
    }

    out
}

fn coupon_annuity(curve: &YieldCurve, tenor: f64, frequency: Frequency) -> f64 {
    coupon_periods(tenor, frequency)
        .iter()
        .map(|(start, end)| (end - start) * curve.discount_factor(*end))
        .sum()
}
