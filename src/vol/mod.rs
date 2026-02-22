//! Module `vol::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Gatheral (2006), Derman and Kani (1994), static-arbitrage constraints around total variance Eq. (2.2).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
pub mod andreasen_huge;
pub mod arbitrage;
pub mod builder;
pub mod fengler;
pub mod forward;
pub mod implied;
pub mod jaeckel;
pub mod local_vol;
pub mod mixture;
pub mod sabr;
pub mod slice;
pub mod smile;
pub mod surface;

pub use arbitrage::ArbitrageViolation;
