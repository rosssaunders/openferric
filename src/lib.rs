//! OpenFerric is a quantitative finance toolkit for option pricing, rates, credit, and risk.
//!
//! # What The Crate Provides
//! - Instrument schemas (`instruments`) and shared domain types (`core`).
//! - Pricing engines (`engines`) covering analytic, Monte Carlo, FFT, and PDE-style workflows.
//! - Market/model layers (`market`, `models`, `vol`, `rates`, `credit`) used by the engines.
//! - Portfolio and XVA/risk analytics (`risk`).
//!
//! # Feature Flags
//! - `parallel`: enables rayon-backed parallel Monte Carlo paths/grid calculations.
//! - `simd`: enables x86_64 SIMD-accelerated Monte Carlo kernels where available.
//! - `python`: exports Python bindings through `openferric::python`.
//! - `wasm`: enables WebAssembly-facing bindings in `openferric::wasm`.
//!
//! # Quick Start
//! ```
//! use openferric::core::PricingEngine;
//! use openferric::engines::analytic::BlackScholesEngine;
//! use openferric::instruments::VanillaOption;
//! use openferric::market::Market;
//!
//! let option = VanillaOption::european_call(100.0, 1.0);
//! let market = Market::builder()
//!     .spot(100.0)
//!     .rate(0.05)
//!     .dividend_yield(0.0)
//!     .flat_vol(0.20)
//!     .build()
//!     .expect("valid market");
//!
//! let engine = BlackScholesEngine::new();
//! let result = engine.price(&option, &market).expect("price");
//! assert!(result.price > 0.0);
//! ```
//!
//! # Module Guide
//! - `core`: common enums, errors, diagnostics, and engine traits.
//! - `instruments`: typed product definitions.
//! - `engines`: pricers and Greeks engines.
//! - `market`: market data containers and builders.
//! - `models`: stochastic dynamics and calibration helpers.
//! - `rates`, `credit`, `vol`: domain-specific curve/smile/product math.
//! - `risk`: VaR/XVA/portfolio aggregation utilities.
//! - `prelude`: convenience exports for common workflows.

pub mod core;
pub mod credit;
pub mod engines;
pub mod instruments;
pub mod market;
pub mod math;
pub mod models;
#[cfg(feature = "python")]
pub mod python;
pub mod rates;
pub mod risk;
pub mod vol;
#[cfg(feature = "wasm")]
pub mod wasm;

// Legacy modules kept for compatibility with existing consumers.
pub mod greeks;
pub mod mc;
pub mod pricing;

/// Common imports for ergonomic usage.
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use crate::core::*;
    pub use crate::credit::*;
    pub use crate::engines::analytic::*;
    pub use crate::instruments::*;
    pub use crate::market::*;
    pub use crate::rates::*;
}
