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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TarfType {
    /// Standard TARF: accumulate on upside, leverage on downside.
    Standard,
    /// Decumulator: sell (rather than buy) at each fixing.
    Decumulator,
}

/// Accumulator / TARF instrument.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
        if self.ko_barrier.is_nan() || self.ko_barrier <= 0.0 {
            return Err("ko_barrier must be > 0 and not NaN".to_string());
        }
        if self.fixing_times.is_empty() {
            return Err("fixing_times must be non-empty".to_string());
        }
        if self
            .fixing_times
            .iter()
            .any(|t| !t.is_finite() || *t <= 0.0)
        {
            return Err("all fixing_times must be finite and > 0".to_string());
        }
        if self.fixing_times.windows(2).any(|w| w[1] <= w[0]) {
            return Err("fixing_times must be strictly increasing".to_string());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tarf() -> Tarf {
        Tarf::standard(
            100.0,
            1000.0,
            120.0,
            50_000.0,
            2.0,
            vec![0.25, 0.5, 0.75, 1.0],
        )
    }

    #[test]
    fn validate_accepts_valid_tarf() {
        assert!(valid_tarf().validate().is_ok());
    }

    #[test]
    fn validate_rejects_unsorted_or_duplicate_fixing_times() {
        let mut tarf = valid_tarf();
        tarf.fixing_times = vec![0.5, 0.25, 1.0];
        assert!(
            tarf.validate().is_err(),
            "unsorted fixing times must be rejected"
        );

        tarf.fixing_times = vec![0.25, 0.25, 0.5];
        assert!(
            tarf.validate().is_err(),
            "duplicate fixing times must be rejected"
        );
    }

    #[test]
    fn validate_rejects_nan_inputs() {
        let cases: Vec<(&str, Tarf)> = vec![
            ("strike", {
                let mut t = valid_tarf();
                t.strike = f64::NAN;
                t
            }),
            ("notional_per_fixing", {
                let mut t = valid_tarf();
                t.notional_per_fixing = f64::NAN;
                t
            }),
            ("ko_barrier", {
                let mut t = valid_tarf();
                t.ko_barrier = f64::NAN;
                t
            }),
            ("target_profit", {
                let mut t = valid_tarf();
                t.target_profit = f64::NAN;
                t
            }),
            ("downside_leverage", {
                let mut t = valid_tarf();
                t.downside_leverage = f64::NAN;
                t
            }),
            ("fixing_times", {
                let mut t = valid_tarf();
                t.fixing_times[1] = f64::NAN;
                t
            }),
        ];

        for (field, tarf) in cases {
            assert!(tarf.validate().is_err(), "NaN {field} must be rejected");
        }
    }

    #[test]
    fn validate_allows_infinite_ko_barrier() {
        let mut tarf = valid_tarf();
        tarf.ko_barrier = f64::INFINITY;
        assert!(tarf.validate().is_ok());
    }
}
