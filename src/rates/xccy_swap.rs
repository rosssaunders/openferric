use crate::rates::YieldCurve;

/// Cross-currency swap: fixed leg in currency 1 vs floating leg in currency 2.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
        if self.notional1 <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let annuity = annual_annuity(ccy1_discount_curve, self.tenor);
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
        if self.notional2 <= 0.0 || self.tenor <= 0.0 {
            return 0.0;
        }

        let mut pv = 0.0;
        for (start, end) in annual_periods(self.tenor) {
            let accrual = end - start;
            if accrual <= 0.0 {
                continue;
            }

            let fwd = ccy2_projection_curve.forward_rate(start, end);
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
        let fixed_leg_ccy1 = self.fixed_leg_pv_ccy1(ccy1_discount_curve);
        let float_leg_ccy2 = self.float_leg_pv_ccy2(ccy2_discount_curve, ccy2_projection_curve);
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
        if self.notional1 <= 0.0 || self.tenor <= 0.0 {
            return f64::NAN;
        }

        let annuity = annual_annuity(ccy1_discount_curve, self.tenor);
        if annuity <= 0.0 {
            return f64::NAN;
        }

        let float_pv_ccy1 =
            self.float_leg_pv_ccy2(ccy2_discount_curve, ccy2_projection_curve) * self.fx_spot;
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
        let mut shifted = *self;
        shifted.fx_spot = current_fx_spot;
        shifted.npv_dual_curve(
            ccy1_discount_curve,
            ccy2_discount_curve,
            ccy2_projection_curve,
            pay_fixed_ccy1,
        )
    }
}

fn annual_periods(tenor: f64) -> Vec<(f64, f64)> {
    if tenor <= 0.0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut start = 0.0;
    while start < tenor - 1.0e-12 {
        let end = (start + 1.0).min(tenor);
        out.push((start, end));
        start = end;
    }

    out
}

fn annual_annuity(curve: &YieldCurve, tenor: f64) -> f64 {
    annual_periods(tenor)
        .iter()
        .map(|(start, end)| (end - start) * curve.discount_factor(*end))
        .sum()
}
