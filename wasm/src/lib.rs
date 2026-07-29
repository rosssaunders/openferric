mod calibrate;
mod credit;
mod dsl;
mod error;
mod funding;
mod gpu;
mod helpers;
mod heston;
mod payoff;
mod pricing;
mod risk;
mod vol;

/// Initializes the Rayon worker pool for the threaded WebAssembly build.
///
/// Call the generated `initThreadPool(navigator.hardwareConcurrency)` export
/// after loading the module and before invoking parallel pricing operations.
#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(all(test, target_arch = "wasm32"))]
mod abi_tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod contract_tests;
