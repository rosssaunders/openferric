//! Module `engines::gpu::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.

#[cfg(feature = "gpu")]
mod gpu_mc;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub use gpu_mc::mc_european_gpu;

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub use gpu_mc::mc_european_gpu_async;

#[cfg(feature = "gpu")]
pub use gpu_mc::GpuMcResult;
