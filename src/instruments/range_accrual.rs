//! Module `instruments::range_accrual`.
//!
//! Implements range accrual abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `RangeAccrual`, `DualRangeAccrual` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
/// Range accrual note instrument definition.
///
/// A range accrual pays a coupon that accrues on each day the reference rate
/// is within a specified range [lower, upper]. Common in rates desks.
/// Single-rate range accrual.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RangeAccrual {
    /// Notional amount.
    pub notional: f64,
    /// Full coupon rate (annualised) paid if rate is always in range.
    pub coupon_rate: f64,
    /// Coupon period year fraction, independent of the payment lag.
    pub accrual_factor: f64,
    /// Lower bound of the accrual range.
    pub lower_bound: f64,
    /// Upper bound of the accrual range.
    pub upper_bound: f64,
    /// Fixing dates as year fractions from valuation date.
    pub fixing_times: Vec<f64>,
    /// Payment date (year fraction).
    pub payment_time: f64,
}

/// Dual-rate range accrual (e.g., CMS spread range accrual).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DualRangeAccrual {
    /// Notional amount.
    pub notional: f64,
    /// Full coupon rate (annualised).
    pub coupon_rate: f64,
    /// Coupon period year fraction, independent of the payment lag.
    pub accrual_factor: f64,
    /// Lower bound of the spread range.
    pub lower_bound: f64,
    /// Upper bound of the spread range.
    pub upper_bound: f64,
    /// Fixing dates as year fractions.
    pub fixing_times: Vec<f64>,
    /// Payment date (year fraction).
    pub payment_time: f64,
}

fn validate_schedule(fixing_times: &[f64], payment_time: f64) -> Result<(), String> {
    if fixing_times.is_empty() {
        return Err("fixing_times must be non-empty".to_string());
    }
    if fixing_times.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return Err("all fixing_times must be finite and > 0".to_string());
    }
    if fixing_times.windows(2).any(|w| w[1] <= w[0]) {
        return Err("fixing_times must be strictly increasing".to_string());
    }
    if !payment_time.is_finite() || payment_time <= 0.0 {
        return Err("payment_time must be finite and > 0".to_string());
    }
    let last_fixing = *fixing_times.last().expect("non-empty fixing_times");
    if payment_time < last_fixing {
        return Err("payment_time must be >= last fixing time".to_string());
    }
    Ok(())
}

impl RangeAccrual {
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate <= 0.0 {
            return Err("coupon_rate must be finite and > 0".to_string());
        }
        if !self.accrual_factor.is_finite() || self.accrual_factor <= 0.0 {
            return Err("accrual_factor must be finite and > 0".to_string());
        }
        if !self.lower_bound.is_finite() || !self.upper_bound.is_finite() {
            return Err("bounds must be finite".to_string());
        }
        if self.lower_bound >= self.upper_bound {
            return Err("lower_bound must be < upper_bound".to_string());
        }
        validate_schedule(&self.fixing_times, self.payment_time)
    }
}

impl DualRangeAccrual {
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate <= 0.0 {
            return Err("coupon_rate must be finite and > 0".to_string());
        }
        if !self.accrual_factor.is_finite() || self.accrual_factor <= 0.0 {
            return Err("accrual_factor must be finite and > 0".to_string());
        }
        if !self.lower_bound.is_finite() || !self.upper_bound.is_finite() {
            return Err("bounds must be finite".to_string());
        }
        if self.lower_bound >= self.upper_bound {
            return Err("lower_bound must be < upper_bound".to_string());
        }
        validate_schedule(&self.fixing_times, self.payment_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_single() -> RangeAccrual {
        RangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            accrual_factor: 1.0,
            lower_bound: 0.02,
            upper_bound: 0.06,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }
    }

    fn valid_dual() -> DualRangeAccrual {
        DualRangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            accrual_factor: 1.0,
            lower_bound: 0.0,
            upper_bound: 0.02,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }
    }

    #[test]
    fn validate_accepts_valid_instruments() {
        assert!(valid_single().validate().is_ok());
        assert!(valid_dual().validate().is_ok());
    }

    #[test]
    fn range_accrual_rejects_bad_schedules() {
        let cases: Vec<(&str, RangeAccrual)> = vec![
            ("empty fixing_times", {
                let mut ra = valid_single();
                ra.fixing_times = vec![];
                ra
            }),
            ("non-finite fixing time", {
                let mut ra = valid_single();
                ra.fixing_times[1] = f64::NAN;
                ra
            }),
            ("non-positive fixing time", {
                let mut ra = valid_single();
                ra.fixing_times[0] = 0.0;
                ra
            }),
            ("unsorted fixing times", {
                let mut ra = valid_single();
                ra.fixing_times = vec![0.5, 0.25, 1.0];
                ra
            }),
            ("duplicate fixing times", {
                let mut ra = valid_single();
                ra.fixing_times = vec![0.25, 0.25, 1.0];
                ra
            }),
            ("NaN payment_time", {
                let mut ra = valid_single();
                ra.payment_time = f64::NAN;
                ra
            }),
            ("non-positive payment_time", {
                let mut ra = valid_single();
                ra.payment_time = 0.0;
                ra
            }),
            ("payment before last fixing", {
                let mut ra = valid_single();
                ra.payment_time = 0.9;
                ra
            }),
            ("NaN bound", {
                let mut ra = valid_single();
                ra.lower_bound = f64::NAN;
                ra
            }),
        ];

        for (label, ra) in cases {
            assert!(ra.validate().is_err(), "{label} must be rejected");
        }
    }

    #[test]
    fn dual_range_accrual_rejects_bad_schedules() {
        let cases: Vec<(&str, DualRangeAccrual)> = vec![
            ("empty fixing_times", {
                let mut ra = valid_dual();
                ra.fixing_times = vec![];
                ra
            }),
            ("non-finite fixing time", {
                let mut ra = valid_dual();
                ra.fixing_times[1] = f64::INFINITY;
                ra
            }),
            ("non-positive fixing time", {
                let mut ra = valid_dual();
                ra.fixing_times[0] = -0.25;
                ra
            }),
            ("unsorted fixing times", {
                let mut ra = valid_dual();
                ra.fixing_times = vec![0.5, 0.25, 1.0];
                ra
            }),
            ("NaN payment_time", {
                let mut ra = valid_dual();
                ra.payment_time = f64::NAN;
                ra
            }),
            ("payment before last fixing", {
                let mut ra = valid_dual();
                ra.payment_time = 0.5;
                ra
            }),
            ("NaN bound", {
                let mut ra = valid_dual();
                ra.upper_bound = f64::NAN;
                ra
            }),
        ];

        for (label, ra) in cases {
            assert!(ra.validate().is_err(), "{label} must be rejected");
        }
    }
}
