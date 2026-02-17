//! CDS options (credit swaptions) priced via Black's model on the CDS spread.

use crate::math::normal_cdf;

/// A CDS option giving the right to enter a CDS at a given spread (strike).
#[derive(Debug, Clone, PartialEq)]
pub struct CdsOption {
    pub notional: f64,
    pub strike_spread: f64,
    pub option_expiry: f64,
    pub cds_maturity: f64,
    pub is_payer: bool,
    pub recovery_rate: f64,
}

impl CdsOption {
    /// Price using Black's model on the CDS spread.
    ///
    /// * `forward_spread` - current fair CDS spread
    /// * `vol` - implied volatility of the CDS spread
    /// * `risky_annuity` - RPV01 of the underlying CDS
    /// * `discount_factor` - risk-free discount factor to option expiry
    pub fn black_price(
        &self,
        forward_spread: f64,
        vol: f64,
        risky_annuity: f64,
        discount_factor: f64,
    ) -> f64 {
        if self.option_expiry <= 0.0 || vol < 0.0 || risky_annuity <= 0.0 {
            return 0.0;
        }

        let f = forward_spread;
        let k = self.strike_spread;

        // Zero vol case: intrinsic value
        if vol <= 1e-14 {
            let intrinsic = if self.is_payer {
                (f - k).max(0.0)
            } else {
                (k - f).max(0.0)
            };
            return self.notional * risky_annuity * intrinsic;
        }

        if f <= 0.0 || k <= 0.0 {
            return 0.0;
        }

        let t = self.option_expiry;
        let sqrt_t = t.sqrt();
        let vol_sqrt_t = vol * sqrt_t;
        let d1 = ((f / k).ln() + 0.5 * vol * vol * t) / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;

        let undiscounted = if self.is_payer {
            f * normal_cdf(d1) - k * normal_cdf(d2)
        } else {
            k * normal_cdf(-d2) - f * normal_cdf(-d1)
        };

        self.notional * risky_annuity * undiscounted
    }
}

/// Compute the risky annuity (RPV01) for a CDS.
///
/// RPV01 = Σ ΔTi * DF(Ti) * Q(Ti)
///
/// where DF is the risk-free discount factor and Q is survival probability.
///
/// * `payment_freq` - number of payments per year (e.g. 4 for quarterly)
/// * `cds_tenor` - total CDS tenor in years
/// * `hazard_rate` - flat hazard rate (continuous)
/// * `risk_free_rate` - flat continuously-compounded risk-free rate
/// * `_recovery` - recovery rate (not used in RPV01 but kept for API consistency)
pub fn risky_annuity(
    payment_freq: u32,
    cds_tenor: f64,
    hazard_rate: f64,
    risk_free_rate: f64,
    _recovery: f64,
) -> f64 {
    if payment_freq == 0 || cds_tenor <= 0.0 {
        return 0.0;
    }
    let dt = 1.0 / payment_freq as f64;
    let n = (cds_tenor * payment_freq as f64).round() as u32;
    let mut rpv01 = 0.0;
    for i in 1..=n {
        let t = i as f64 * dt;
        let df = (-risk_free_rate * t).exp();
        let q = (-hazard_rate * t).exp();
        rpv01 += dt * df * q;
    }
    rpv01
}

/// Compute the fair (par) CDS spread from a flat hazard rate.
///
/// fair_spread = hazard_rate * (1 - recovery) * RPV01_default / RPV01_premium
/// Simplified: for continuous model, fair_spread ≈ hazard_rate * (1 - recovery)
/// But for discrete payments we compute it properly.
pub fn fair_spread_from_hazard(
    payment_freq: u32,
    cds_tenor: f64,
    hazard_rate: f64,
    risk_free_rate: f64,
    recovery: f64,
) -> f64 {
    if payment_freq == 0 || cds_tenor <= 0.0 {
        return 0.0;
    }
    let dt = 1.0 / payment_freq as f64;
    let n = (cds_tenor * payment_freq as f64).round() as u32;

    // Protection leg PV (assuming default at midpoint of each period)
    let mut prot_pv = 0.0;
    let mut premium_pv = 0.0;

    for i in 1..=n {
        let t = i as f64 * dt;
        let t_prev = (i - 1) as f64 * dt;
        let q_prev = (-hazard_rate * t_prev).exp();
        let q = (-hazard_rate * t).exp();
        let default_prob = q_prev - q;
        let t_mid = (t_prev + t) / 2.0;
        let df_mid = (-risk_free_rate * t_mid).exp();
        let df = (-risk_free_rate * t).exp();

        prot_pv += (1.0 - recovery) * default_prob * df_mid;
        premium_pv += dt * df * q;
    }

    if premium_pv <= 0.0 {
        return 0.0;
    }
    prot_pv / premium_pv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atm_put_call_parity() {
        // At ATM, payer and receiver should have equal value
        let forward = 0.01;
        let vol = 0.3;
        let rpv01 = risky_annuity(4, 5.0, 0.01, 0.05, 0.4);

        let payer = CdsOption {
            notional: 1_000_000.0,
            strike_spread: forward,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: true,
            recovery_rate: 0.4,
        };
        let receiver = CdsOption {
            notional: 1_000_000.0,
            strike_spread: forward,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: false,
            recovery_rate: 0.4,
        };

        let p = payer.black_price(forward, vol, rpv01, 1.0);
        let r = receiver.black_price(forward, vol, rpv01, 1.0);
        assert!(
            (p - r).abs() < 1e-8,
            "ATM payer ({p}) != receiver ({r})"
        );
    }

    #[test]
    fn test_quantlib_cached_value() {
        // QuantLib test parameters (from vendor/QuantLib/test-suite/cdsoption.cpp):
        // Eval date: Dec 10, 2007; Hazard rate: 0.001 (flat, Actual/360)
        // Risk-free rate: 0.02 (flat, Actual/360); Recovery: 0.4
        // Option expiry: 9 months => Sep 10, 2008 (275 days)
        // CDS start: 1 month after expiry => Oct 13, 2008 (309 days from eval)
        // CDS maturity: 7 years after start => Oct 13, 2015
        // CDS schedule: quarterly, TARGET calendar, ModifiedFollowing
        // Vol: 0.20, Strike = fair spread
        // Expected NPV: 270.976348

        let hazard_rate = 0.001_f64;
        let risk_free_rate = 0.02_f64;
        let recovery = 0.4_f64;
        let vol = 0.20_f64;
        let notional = 1_000_000.0_f64;

        // Use Actual/360 day count convention as QuantLib does.
        // Eval date: Dec 10, 2007. Expiry: Sep 10, 2008.
        let t_expiry = 275.0 / 360.0; // days from Dec 10 2007 to Sep 10 2008

        // CDS quarterly schedule dates (approximate days from eval date Dec 10, 2007).
        // Start: Oct 13, 2008 (Oct 10 is Friday → ok, but TARGET may adjust).
        // We approximate 28 quarterly dates from start to maturity.
        // CDS start ≈ day 309 from eval.
        let cds_start_days = 309.0_f64;
        // Generate 28 quarterly payment dates (~91.25 days apart)
        // Actual QuantLib dates would use TARGET calendar adjustments.
        // We approximate each quarter as ~91 days for interior, adjusted for actual months.

        // Quarter day counts from start (approximate for actual months):
        // Q1: Jan 12, 2009 (91 days from Oct 13), Q2: Apr 13, 2009 (+91),
        // ... 28 quarters to ~Oct 13, 2015
        let num_periods = 28_u32;

        // Compute coupon leg NPV including accrual on default (QuantLib MidPointCdsEngine style)
        // For a unit spread CDS:
        // couponLegNPV = Σ [S(ti) * Δti * DF(ti) + P(ti-1,ti) * (Δti/2) * DF(mid)]
        // where Δti = accrual fraction (Actual/360), P(a,b) = S(a) - S(b)
        let mut coupon_leg_npv = 0.0;
        let mut prot_leg_npv = 0.0;

        for i in 0..num_periods {
            // Approximate each period as ~91.31 days (365.25/4)
            let period_days = 91.31_f64;
            let t_start_days = cds_start_days + i as f64 * period_days;
            let t_end_days = cds_start_days + (i + 1) as f64 * period_days;
            let t_mid_days = (t_start_days + t_end_days) / 2.0;
            let accrual = period_days / 360.0;

            let t_end = t_end_days / 360.0;
            let t_start = t_start_days / 360.0;
            let t_mid = t_mid_days / 360.0;

            let df_end = (-risk_free_rate * t_end).exp();
            let df_mid = (-risk_free_rate * t_mid).exp();
            let s_start = (-hazard_rate * t_start).exp();
            let s_end = (-hazard_rate * t_end).exp();
            let default_prob = s_start - s_end;

            // Coupon on survival
            coupon_leg_npv += s_end * accrual * df_end;
            // Accrual on default (half period accrued at midpoint)
            coupon_leg_npv += default_prob * (accrual / 2.0) * df_mid;

            // Protection leg
            prot_leg_npv += (1.0 - recovery) * default_prob * df_mid;
        }

        let rpv01 = coupon_leg_npv;
        let fair = prot_leg_npv / coupon_leg_npv;

        let option = CdsOption {
            notional,
            strike_spread: fair,
            option_expiry: t_expiry,
            cds_maturity: 7.0,
            is_payer: true,
            recovery_rate: recovery,
        };

        let price = option.black_price(fair, vol, rpv01, (-risk_free_rate * t_expiry).exp());
        assert!(
            (price - 270.976348).abs() < 1.0,
            "QuantLib cached value mismatch: got {price}, expected 270.976348"
        );
    }

    #[test]
    fn test_zero_vol() {
        let rpv01 = risky_annuity(4, 5.0, 0.01, 0.05, 0.4);
        let forward = 0.02;
        let strike = 0.01;

        let payer = CdsOption {
            notional: 1_000_000.0,
            strike_spread: strike,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: true,
            recovery_rate: 0.4,
        };

        let price = payer.black_price(forward, 0.0, rpv01, 1.0);
        let expected = 1_000_000.0 * rpv01 * (forward - strike);
        assert!(
            (price - expected).abs() < 1e-6,
            "Zero vol payer: got {price}, expected {expected}"
        );

        // OTM receiver with zero vol should be 0
        let receiver = CdsOption {
            notional: 1_000_000.0,
            strike_spread: strike,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: false,
            recovery_rate: 0.4,
        };
        let price_r = receiver.black_price(forward, 0.0, rpv01, 1.0);
        assert!(
            price_r.abs() < 1e-6,
            "Zero vol OTM receiver should be 0, got {price_r}"
        );
    }

    #[test]
    fn test_deep_itm_otm() {
        let rpv01 = risky_annuity(4, 5.0, 0.01, 0.05, 0.4);
        let vol = 0.3;

        // Deep ITM payer: forward >> strike
        let deep_itm = CdsOption {
            notional: 1_000_000.0,
            strike_spread: 0.001,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: true,
            recovery_rate: 0.4,
        };
        let forward = 0.05;
        let price = deep_itm.black_price(forward, vol, rpv01, 1.0);
        let intrinsic = 1_000_000.0 * rpv01 * (forward - 0.001);
        assert!(
            price >= intrinsic * 0.99,
            "Deep ITM payer should be >= intrinsic: {price} vs {intrinsic}"
        );

        // Deep OTM payer: forward << strike
        let deep_otm = CdsOption {
            notional: 1_000_000.0,
            strike_spread: 0.10,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: true,
            recovery_rate: 0.4,
        };
        let forward_low = 0.001;
        let price_otm = deep_otm.black_price(forward_low, vol, rpv01, 1.0);
        assert!(
            price_otm < 1_000_000.0 * rpv01 * 0.001,
            "Deep OTM payer should be near zero, got {price_otm}"
        );
    }
}
