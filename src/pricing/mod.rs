//! Module `pricing::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
pub mod american;
pub mod payoff;
pub mod asian;
pub mod autocallable;
pub mod barrier;
pub mod basket;
pub mod mbs;
pub mod bermudan;
pub mod discrete_div;
pub mod european;
pub mod range_accrual;
pub mod real_option;
pub mod tarf;

pub use crate::core::types::OptionType;
