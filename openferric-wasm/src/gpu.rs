#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

/// GPU Monte Carlo result exposed to JavaScript via wasm-bindgen.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub struct WasmGpuMcResult {
    pub price: f64,
    pub stderr: f64,
}

/// GPU Monte Carlo European option pricing via WebGPU compute shaders.
///
/// Uses `u32` for `num_paths` and `num_steps` to avoid BigInt in JS.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
#[wasm_bindgen]
pub async fn gpu_mc_price_european(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    num_paths: u32,
    num_steps: u32,
    seed: u32,
    is_call: bool,
) -> Result<WasmGpuMcResult, JsError> {
    let result = openferric::engines::gpu::mc_european_gpu_async(
        spot, strike, rate, vol, expiry, num_paths, num_steps, seed, is_call,
    )
    .await
    .map_err(|e| JsError::new(&e))?;

    Ok(WasmGpuMcResult {
        price: result.price,
        stderr: result.stderr,
    })
}
