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
    /// The spot KO barrier is an *upside* barrier: the structure knocks out
    /// when spot fixes at or above `ko_barrier`.
    Standard,
    /// Decumulator: sell (rather than buy) at each fixing, gaining when spot
    /// falls below the strike. The spot KO barrier is a *downside* barrier:
    /// the structure knocks out when spot fixes at or below `ko_barrier`.
    Decumulator,
}

/// Accumulator / TARF instrument.
///
/// # Conventions
///
/// - **KO direction is type-aware.** For [`TarfType::Standard`] the KO barrier
///   is an upside barrier (`spot >= ko_barrier` terminates); for
///   [`TarfType::Decumulator`] it is a downside barrier (`spot <= ko_barrier`
///   terminates). A finite barrier must lie strictly on the correct side of the
///   strike (above for `Standard`, below for `Decumulator`); `validate()`
///   enforces this.
/// - **No-barrier sentinel.** `f64::INFINITY` means "no KO barrier" for *both*
///   types (the pricing loop guards the direction check with `is_finite()`, so
///   an infinite barrier never triggers a decumulator downside KO).
/// - **KO precedes the fixing P&L.** On a KO breach the breaching fixing pays
///   nothing and all remaining fixings are cancelled.
/// - **Full-gain target convention.** On the fixing that reaches
///   `target_profit`, the full uncapped fixing gain is paid before termination
///   (as opposed to no-gain or capped-gain conventions, which are not
///   currently representable).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tarf {
    /// Forward strike.
    pub strike: f64,
    /// Notional per fixing period.
    pub notional_per_fixing: f64,
    /// Spot knock-out barrier; the structure terminates when spot breaches it
    /// at a fixing. Direction depends on `tarf_type`: upside
    /// (`spot >= ko_barrier`) for `Standard`, downside (`spot <= ko_barrier`)
    /// for `Decumulator`. Set to `f64::INFINITY` for no KO barrier (both
    /// types). The breaching fixing itself pays nothing.
    pub ko_barrier: f64,
    /// Target profit level for early termination. The fixing that reaches the
    /// target pays its full uncapped gain (full-gain convention).
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
        // A finite barrier must sit strictly on the KO side of the strike for
        // the product type; `f64::INFINITY` is the "no barrier" sentinel for
        // both types.
        if self.ko_barrier.is_finite() {
            match self.tarf_type {
                TarfType::Standard => {
                    if self.ko_barrier <= self.strike {
                        return Err(
                            "ko_barrier must be > strike for a standard TARF (upside KO); \
                             use f64::INFINITY for no barrier"
                                .to_string(),
                        );
                    }
                }
                TarfType::Decumulator => {
                    if self.ko_barrier >= self.strike {
                        return Err(
                            "ko_barrier must be < strike for a decumulator TARF (downside KO); \
                             use f64::INFINITY for no barrier"
                                .to_string(),
                        );
                    }
                }
            }
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

    /// Standard TARF with common defaults. `ko_barrier` is an upside barrier
    /// and must be above the strike (or `f64::INFINITY` for no barrier).
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

    /// Decumulator TARF with common defaults. `ko_barrier` is a downside
    /// barrier and must be below the strike (or `f64::INFINITY` for no
    /// barrier).
    pub fn decumulator(
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
            tarf_type: TarfType::Decumulator,
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
        // INFINITY is the universal "no KO barrier" sentinel for both types.
        let mut tarf = valid_tarf();
        tarf.ko_barrier = f64::INFINITY;
        assert!(tarf.validate().is_ok());
        tarf.tarf_type = TarfType::Decumulator;
        assert!(tarf.validate().is_ok());
    }

    #[test]
    fn validate_accepts_decumulator_with_downside_barrier() {
        let tarf = Tarf::decumulator(
            100.0,
            1000.0,
            80.0,
            50_000.0,
            2.0,
            vec![0.25, 0.5, 0.75, 1.0],
        );
        assert!(tarf.validate().is_ok());
    }

    #[test]
    fn validate_rejects_barrier_on_wrong_side_of_strike() {
        // Standard KO is an upside barrier: must be strictly above strike.
        let mut standard = valid_tarf();
        standard.ko_barrier = 80.0;
        assert!(
            standard.validate().is_err(),
            "downside barrier must be rejected for a standard TARF"
        );
        standard.ko_barrier = standard.strike;
        assert!(
            standard.validate().is_err(),
            "barrier equal to strike must be rejected for a standard TARF"
        );

        // Decumulator KO is a downside barrier: must be strictly below strike.
        let mut dec = valid_tarf();
        dec.tarf_type = TarfType::Decumulator;
        dec.ko_barrier = 120.0;
        assert!(
            dec.validate().is_err(),
            "upside barrier must be rejected for a decumulator TARF"
        );
        dec.ko_barrier = dec.strike;
        assert!(
            dec.validate().is_err(),
            "barrier equal to strike must be rejected for a decumulator TARF"
        );
    }
}
