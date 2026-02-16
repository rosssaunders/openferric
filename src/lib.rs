//! OpenFerric: quantitative pricing library with trait-based instrument/engine APIs.

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
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod risk;
pub mod vol;

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
