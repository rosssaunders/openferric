//! GPU-accelerated Monte Carlo pricing via wgpu compute shaders.
//!
//! When enabled via the `gpu` feature flag, this module offloads European
//! option Monte Carlo simulation to the GPU, achieving massive parallelism
//! (thousands of concurrent paths) on any Vulkan/Metal/DX12/WebGPU backend.

#[cfg(feature = "gpu")]
mod gpu_mc;

#[cfg(feature = "gpu")]
pub use gpu_mc::mc_european_gpu;
