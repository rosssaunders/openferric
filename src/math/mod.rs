//! Module `math::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004) Ch. 5, Joe and Kuo (2008), SIMD and random-sequence implementation details tied to Eq. (5.4).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: approximation regions, branch choices, and machine-precision cancellation near boundaries should be validated with high-precision references.
//!
//! When to use: use these low-level routines in performance-sensitive calibration/pricing loops; use higher-level modules when model semantics matter more than raw numerics.
pub mod arena;
pub mod fast_norm;
pub mod fast_rng;
pub mod functions;
pub mod gamma;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod simd_math;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod simd_neon;
pub mod sobol;

pub use arena::PricingArena;
pub use fast_norm::{
    beasley_springer_moro_inv_cdf, fast_norm_cdf, fast_norm_inv_cdf, fast_norm_pdf, hart_norm_cdf,
};
pub use fast_rng::{FastRng, FastRngKind};
pub use functions::*;
pub use sobol::SobolSequence;
