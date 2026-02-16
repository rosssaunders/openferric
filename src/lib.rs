//! OpenFerric: quantitative pricing library with trait-based instrument/engine APIs.

pub mod core;
pub mod engines;
pub mod instruments;
pub mod market;
pub mod math;
pub mod models;
pub mod python;
pub mod rates;
pub mod vol;

// Legacy modules kept for compatibility with existing consumers.
pub mod greeks;
pub mod mc;
pub mod pricing;

/// Common imports for ergonomic usage.
pub mod prelude {
    pub use crate::core::*;
    pub use crate::engines::analytic::*;
    pub use crate::instruments::*;
    pub use crate::market::*;
    pub use crate::rates::*;
}
