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
#[derive(Debug, Clone, PartialEq)]
pub struct RangeAccrual {
    /// Notional amount.
    pub notional: f64,
    /// Full coupon rate (annualised) paid if rate is always in range.
    pub coupon_rate: f64,
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
#[derive(Debug, Clone, PartialEq)]
pub struct DualRangeAccrual {
    /// Notional amount.
    pub notional: f64,
    /// Full coupon rate (annualised).
    pub coupon_rate: f64,
    /// Lower bound of the spread range.
    pub lower_bound: f64,
    /// Upper bound of the spread range.
    pub upper_bound: f64,
    /// Fixing dates as year fractions.
    pub fixing_times: Vec<f64>,
    /// Payment date (year fraction).
    pub payment_time: f64,
}

impl RangeAccrual {
    pub fn validate(&self) -> Result<(), String> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err("notional must be finite and > 0".to_string());
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate <= 0.0 {
            return Err("coupon_rate must be finite and > 0".to_string());
        }
        if !self.lower_bound.is_finite() || !self.upper_bound.is_finite() {
            return Err("bounds must be finite".to_string());
        }
        if self.lower_bound >= self.upper_bound {
            return Err("lower_bound must be < upper_bound".to_string());
        }
        if self.fixing_times.is_empty() {
            return Err("fixing_times must be non-empty".to_string());
        }
        Ok(())
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
        if self.lower_bound >= self.upper_bound {
            return Err("lower_bound must be < upper_bound".to_string());
        }
        if self.fixing_times.is_empty() {
            return Err("fixing_times must be non-empty".to_string());
        }
        Ok(())
    }
}
