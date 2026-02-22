use crate::rates::YieldCurve;

use super::survival_curve::SurvivalCurve;

/// Standard running-spread CDS contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cds {
    /// Notional amount.
    pub notional: f64,
    /// Running spread in decimal per annum (e.g. 0.01 for 100 bps).
    pub spread: f64,
    /// Contract maturity in years.
    pub maturity: f64,
    /// Recovery rate in `[0, 1)`.
    pub recovery_rate: f64,
    /// Premium payment frequency per year.
    pub payment_freq: usize,
}

impl Cds {
    /// Present value of the premium leg including accrual-on-default.
    pub fn premium_leg_pv(
        &self,
        discount_curve: &YieldCurve,
        survival_curve: &SurvivalCurve,
    ) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        let (coupon_annuity, accrual_annuity, _) = leg_annuity_terms(
            self.maturity,
            self.payment_freq,
            discount_curve,
            survival_curve,
        );

        self.notional * self.spread * (coupon_annuity + accrual_annuity)
    }

    /// Present value of the protection leg.
    pub fn protection_leg_pv(
        &self,
        discount_curve: &YieldCurve,
        survival_curve: &SurvivalCurve,
    ) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        let (_, _, protection_term) = leg_annuity_terms(
            self.maturity,
            self.payment_freq,
            discount_curve,
            survival_curve,
        );

        self.notional * (1.0 - self.recovery_rate) * protection_term
    }

    /// Contract NPV from protection-buyer perspective.
    pub fn npv(&self, discount_curve: &YieldCurve, survival_curve: &SurvivalCurve) -> f64 {
        self.protection_leg_pv(discount_curve, survival_curve)
            - self.premium_leg_pv(discount_curve, survival_curve)
    }

    /// Fair running spread that sets contract NPV to zero.
    pub fn fair_spread(&self, discount_curve: &YieldCurve, survival_curve: &SurvivalCurve) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        let (coupon_annuity, accrual_annuity, protection_term) = leg_annuity_terms(
            self.maturity,
            self.payment_freq,
            discount_curve,
            survival_curve,
        );
        let risky_annuity = coupon_annuity + accrual_annuity;

        if risky_annuity <= 1.0e-14 {
            0.0
        } else {
            (1.0 - self.recovery_rate) * protection_term / risky_annuity
        }
    }

    fn is_valid(&self) -> bool {
        self.notional > 0.0
            && self.spread >= 0.0
            && self.maturity >= 0.0
            && (0.0..1.0).contains(&self.recovery_rate)
            && self.payment_freq > 0
    }
}

fn payment_times(maturity: f64, payment_freq: usize) -> Vec<f64> {
    if maturity <= 0.0 || payment_freq == 0 {
        return vec![];
    }

    let dt = 1.0 / payment_freq as f64;
    let mut t = 0.0;
    let mut times = Vec::new();

    while t + dt < maturity - 1.0e-12 {
        t += dt;
        times.push(t);
    }
    times.push(maturity);

    times
}

fn leg_annuity_terms(
    maturity: f64,
    payment_freq: usize,
    discount_curve: &YieldCurve,
    survival_curve: &SurvivalCurve,
) -> (f64, f64, f64) {
    let times = payment_times(maturity, payment_freq);
    if times.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut coupon_annuity = 0.0;
    let mut accrual_annuity = 0.0;
    let mut protection_term = 0.0;

    let mut t_prev = 0.0;
    for &t in &times {
        let dt = t - t_prev;
        let survival_t = survival_curve.survival_prob(t);

        let df_pay = discount_curve.discount_factor(t);
        coupon_annuity += dt * df_pay * survival_t;

        let default_prob = survival_curve.default_prob(t_prev, t);
        let t_mid = 0.5 * (t_prev + t);
        let df_mid = discount_curve.discount_factor(t_mid);

        // Half-period accrued premium paid if default occurs in-period.
        accrual_annuity += 0.5 * dt * df_mid * default_prob;
        protection_term += df_mid * default_prob;

        t_prev = t;
    }

    (coupon_annuity, accrual_annuity, protection_term)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn fair_spread_matches_flat_hazard_rate_level() {
        let r = 0.03;
        let discount_curve = YieldCurve::new(
            (1..=80)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-r * t).exp())
                })
                .collect(),
        );

        let hazard = 0.02;
        let tenors = (1..=80).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
        let survival_curve =
            SurvivalCurve::from_piecewise_hazard(&tenors, &vec![hazard; tenors.len()]);

        let cds = Cds {
            notional: 10_000_000.0,
            spread: 0.0,
            maturity: 5.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        };

        let fair = cds.fair_spread(&discount_curve, &survival_curve);
        let expected = (1.0 - cds.recovery_rate) * hazard;

        assert_relative_eq!(fair, expected, epsilon = 9e-4);

        let at_fair = Cds {
            spread: fair,
            ..cds.clone()
        };
        assert_relative_eq!(
            at_fair.npv(&discount_curve, &survival_curve),
            0.0,
            epsilon = 1e-8
        );
    }

    #[test]
    fn premium_leg_includes_accrual_on_default() {
        let discount_curve = YieldCurve::new(vec![(1.0, 0.98), (3.0, 0.93), (5.0, 0.88)]);
        let survival_curve = SurvivalCurve::new(vec![(1.0, 0.97), (3.0, 0.90), (5.0, 0.84)]);
        let cds = Cds {
            notional: 1.0,
            spread: 0.01,
            maturity: 5.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        };

        let leg = cds.premium_leg_pv(&discount_curve, &survival_curve);
        let (coupon_annuity, _, _) = leg_annuity_terms(
            cds.maturity,
            cds.payment_freq,
            &discount_curve,
            &survival_curve,
        );
        let coupon_only = cds.notional * cds.spread * coupon_annuity;

        assert!(leg > coupon_only);
    }
}
