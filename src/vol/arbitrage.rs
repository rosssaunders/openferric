//! Module `vol::arbitrage`.
//!
//! Implements arbitrage abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Gatheral (2006), Derman and Kani (1994), static-arbitrage constraints around total variance Eq. (2.2).
//!
//! Key types and purpose: `ArbitrageViolation` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
/// Arbitrage violation types for vol surface validation.
#[derive(Debug, Clone)]
pub enum ArbitrageViolation {
    /// Butterfly arbitrage: negative risk-neutral density.
    Butterfly {
        strike: f64,
        expiry: f64,
        density: f64,
    },
    /// Calendar arbitrage: total variance decreasing in time.
    Calendar {
        strike: f64,
        t1: f64,
        t2: f64,
        dw_dt: f64,
    },
}
