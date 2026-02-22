//! Module `instruments::tarf`.
//!
//! Implements tarf abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `TarfType`, `Tarf` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
/// Target Accrual Redemption Forward (TARF) instrument definition.
///
/// A TARF accumulates forward purchases at periodic fixing dates.
/// The structure terminates ("knocks out") once accumulated profit
/// reaches the target level. Leverage is typically applied on the
/// downside (spot below strike).
///
/// References: Wystup, "FX Options and Structured Products" (2nd ed.)
/// TARF product type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TarfType {
    /// Standard TARF: accumulate on upside, leverage on downside.
    Standard,
    /// Decumulator: sell (rather than buy) at each fixing.
    Decumulator,
}

/// Accumulator / TARF instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct Tarf {
    /// Forward strike.
    pub strike: f64,
    /// Notional per fixing period.
    pub notional_per_fixing: f64,
    /// Knock-out barrier (if spot breaches this on upside, structure terminates).
    /// Set to `f64::INFINITY` for no KO barrier.
    pub ko_barrier: f64,
    /// Target profit level for early termination.
    pub target_profit: f64,
    /// Leverage ratio on downside (typically 2x).
    pub downside_leverage: f64,
    /// Fixing dates as year fractions from valuation date.
    pub fixing_times: Vec<f64>,
    /// Product type.
    pub tarf_type: TarfType,
}

impl Tarf {
    pub fn validate(&self) -> Result<(), String> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err("strike must be finite and > 0".to_string());
        }
        if !self.notional_per_fixing.is_finite() || self.notional_per_fixing <= 0.0 {
            return Err("notional_per_fixing must be finite and > 0".to_string());
        }
        if !self.target_profit.is_finite() || self.target_profit <= 0.0 {
            return Err("target_profit must be finite and > 0".to_string());
        }
        if !self.downside_leverage.is_finite() || self.downside_leverage <= 0.0 {
            return Err("downside_leverage must be finite and > 0".to_string());
        }
        if self.fixing_times.is_empty() {
            return Err("fixing_times must be non-empty".to_string());
        }
        if self.fixing_times.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err("all fixing_times must be finite and > 0".to_string());
        }
        Ok(())
    }

    /// Standard TARF with common defaults.
    pub fn standard(
        strike: f64,
        notional_per_fixing: f64,
        ko_barrier: f64,
        target_profit: f64,
        downside_leverage: f64,
        fixing_times: Vec<f64>,
    ) -> Self {
        Self {
            strike,
            notional_per_fixing,
            ko_barrier,
            target_profit,
            downside_leverage,
            fixing_times,
            tarf_type: TarfType::Standard,
        }
    }
}
