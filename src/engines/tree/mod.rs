//! Module `engines::tree::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13, Cox-Ross-Rubinstein (1979), and backward-induction recursions around Eq. (13.10).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: convergence is first- to second-order in time-step count depending on tree parameterization; deep ITM/OTM nodes may need larger depth.
//!
//! When to use: use trees for early-exercise intuition and lattice diagnostics; use analytic formulas for plain vanillas and Monte Carlo/PDE for richer dynamics.

pub mod bermudan_swaption;
pub mod binomial;
pub mod convertible;
pub mod generalized_binomial;
pub mod swing;
pub mod trinomial;
pub mod two_asset_tree;

pub use bermudan_swaption::BermudanSwaptionEngine;
pub use binomial::BinomialTreeEngine;
pub use convertible::ConvertibleBinomialEngine;
pub use generalized_binomial::GeneralizedBinomialEngine;
pub use swing::SwingTreeEngine;
pub use trinomial::TrinomialTreeEngine;
pub use two_asset_tree::TwoAssetBinomialEngine;
