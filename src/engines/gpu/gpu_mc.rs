//! Module `engines::gpu::gpu_mc`.
//!
//! Implements gpu mc abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) and standard quantitative-finance references aligned with the concrete algorithms implemented in this module.
//!
//! Key types and purpose: `GpuMcResult` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.

use std::sync::Arc;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Shared types (both native and WASM)
// ---------------------------------------------------------------------------

/// GPU-accelerated parameters matching the WGSL struct layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    spot: f32,
    strike: f32,
    rate: f32,
    vol: f32,
    expiry: f32,
    dt_drift: f32,
    dt_vol: f32,
    discount: f32,
    num_steps: u32,
    num_paths: u32,
    seed: u32,
    is_call: u32,
}

/// Result from GPU Monte Carlo pricing.
#[derive(Debug, Clone)]
pub struct GpuMcResult {
    pub price: f64,
    pub stderr: f64,
}

/// Cached GPU resources that persist across pricing calls.
struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build GpuParams from high-level inputs.
fn build_params(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    num_paths: u32,
    num_steps: u32,
    seed: u32,
    is_call: bool,
) -> GpuParams {
    let dt = expiry / num_steps as f64;
    let dt_drift = (rate - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-rate * expiry).exp();

    GpuParams {
        spot: spot as f32,
        strike: strike as f32,
        rate: rate as f32,
        vol: vol as f32,
        expiry: expiry as f32,
        dt_drift: dt_drift as f32,
        dt_vol: dt_vol as f32,
        discount: discount as f32,
        num_steps,
        num_paths,
        seed,
        is_call: if is_call { 1 } else { 0 },
    }
}

/// Compute MC statistics from raw payoff buffer (single pass, unrolled by 4).
fn compute_mc_statistics(payoffs: &[f32], num_paths: usize, discount: f64) -> GpuMcResult {
    let n = num_paths as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut i = 0;
    while i + 4 <= num_paths {
        let v0 = payoffs[i] as f64;
        let v1 = payoffs[i + 1] as f64;
        let v2 = payoffs[i + 2] as f64;
        let v3 = payoffs[i + 3] as f64;
        sum += v0 + v1 + v2 + v3;
        sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        i += 4;
    }
    while i < num_paths {
        let v = payoffs[i] as f64;
        sum += v;
        sum_sq += v * v;
        i += 1;
    }

    let mean = sum / n;
    let var = if num_paths > 1 {
        (sum_sq - sum * sum / n) / (n - 1.0)
    } else {
        0.0
    };

    GpuMcResult {
        price: discount * mean,
        stderr: discount * (var / n).sqrt(),
    }
}

/// Initialize the GPU context (async — works on both native and WASM).
async fn init_gpu_context() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No GPU adapter found")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("openferric MC"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

    let shader_source = include_str!("mc_shader.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MC shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MC pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MC compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        pipeline,
        bind_group_layout,
    })
}

/// Encode and submit the compute dispatch, returning the staging buffer
/// and the number of bytes to read back.
fn encode_and_submit(ctx: &GpuContext, params: GpuParams, num_paths: usize) -> (wgpu::Buffer, u64) {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let payoff_size = (num_paths * std::mem::size_of::<f32>()) as u64;
    let payoff_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("payoffs"),
        size: payoff_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: payoff_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC bind group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: payoff_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroup_size = 256u32;
    let num_workgroups = (num_paths as u32 + workgroup_size - 1) / workgroup_size;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&payoff_buffer, 0, &staging_buffer, 0, payoff_size);
    queue.submit(std::iter::once(encoder.finish()));

    (staging_buffer, payoff_size)
}

// ===========================================================================
// Native-only (sync via pollster + OnceLock)
// ===========================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::sync::OnceLock;

    /// Global GPU context cache. Initialized on first use, reused thereafter.
    static GPU_CTX: OnceLock<Result<GpuContext, String>> = OnceLock::new();

    fn get_or_init_gpu() -> Result<&'static GpuContext, String> {
        GPU_CTX
            .get_or_init(|| pollster::block_on(init_gpu_context()))
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Readback using blocking poll + mpsc channel (native only).
    fn readback_blocking(
        device: &wgpu::Device,
        staging_buffer: &wgpu::Buffer,
        num_paths: usize,
        discount: f64,
    ) -> Result<GpuMcResult, String> {
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| format!("GPU readback failed: {e}"))?
            .map_err(|e| format!("GPU buffer map failed: {e}"))?;

        let data = buffer_slice.get_mapped_range();
        let payoffs: &[f32] = bytemuck::cast_slice(&data);
        let result = compute_mc_statistics(payoffs, num_paths, discount);
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }

    /// Run Monte Carlo European option pricing on the GPU (synchronous, native only).
    ///
    /// Uses wgpu to dispatch compute shaders that simulate GBM paths in parallel.
    /// Each GPU thread simulates one complete path and computes its payoff.
    /// Statistics are reduced on the CPU after readback.
    ///
    /// The GPU device, queue, and pipeline are cached globally so subsequent calls
    /// skip the expensive initialization (~50-200ms) and only pay the dispatch cost.
    pub fn mc_european_gpu(
        spot: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u32,
        is_call: bool,
    ) -> Result<GpuMcResult, String> {
        let discount = (-rate * expiry).exp();
        let params = build_params(
            spot,
            strike,
            rate,
            vol,
            expiry,
            num_paths as u32,
            num_steps as u32,
            seed,
            is_call,
        );

        let ctx = get_or_init_gpu()?;
        let (staging_buffer, _payoff_size) = encode_and_submit(ctx, params, num_paths);
        readback_blocking(&ctx.device, &staging_buffer, num_paths, discount)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::mc_european_gpu;

// ===========================================================================
// WASM-only (async via browser event loop)
// ===========================================================================

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast;

    thread_local! {
        static GPU_CTX: RefCell<Option<Rc<GpuContext>>> = const { RefCell::new(None) };
    }

    /// Lazily initialize (or reuse) the GPU context on the WASM thread.
    async fn ensure_gpu_ctx() -> Result<Rc<GpuContext>, String> {
        // Check if already initialized — clone Rc out before any await.
        let existing = GPU_CTX.with(|cell| cell.borrow().clone());
        if let Some(ctx) = existing {
            return Ok(ctx);
        }

        // First call — async init.
        let ctx = Rc::new(init_gpu_context().await?);
        GPU_CTX.with(|cell| {
            *cell.borrow_mut() = Some(Rc::clone(&ctx));
        });
        Ok(ctx)
    }

    /// Readback using callback + JsFuture yield loop (WASM only).
    async fn readback_async(
        device: &wgpu::Device,
        staging_buffer: &wgpu::Buffer,
        num_paths: usize,
        discount: f64,
    ) -> Result<GpuMcResult, String> {
        let buffer_slice = staging_buffer.slice(..);

        // Use Cell<Option<bool>> because BufferAsyncError is not Copy.
        // true = map succeeded, false = map failed.
        let done = Rc::new(std::cell::Cell::new(None::<bool>));
        let done_cb = Rc::clone(&done);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            done_cb.set(Some(result.is_ok()));
        });

        // Yield to the browser event loop until the callback fires.
        // Must use setTimeout(0) (macrotask) — NOT Promise.resolve() (microtask) —
        // because the browser needs a real event-loop tick to complete the GPU
        // buffer map and fire our callback.
        loop {
            device.poll(wgpu::Maintain::Poll);
            if done.get().is_some() {
                break;
            }
            let promise = js_sys::Promise::new(&mut |resolve, _| {
                let global = js_sys::global();
                let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into())
                    .expect("setTimeout not found");
                let set_timeout_fn: js_sys::Function = set_timeout.unchecked_into();
                let _ = set_timeout_fn.call2(
                    &wasm_bindgen::JsValue::undefined(),
                    &resolve,
                    &wasm_bindgen::JsValue::from(0),
                );
            });
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|_| "JS yield failed".to_string())?;
        }

        if !done.get().unwrap() {
            return Err("GPU buffer map failed".to_string());
        }

        let data = buffer_slice.get_mapped_range();
        let payoffs: &[f32] = bytemuck::cast_slice(&data);
        let result = compute_mc_statistics(payoffs, num_paths, discount);
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }

    /// Run Monte Carlo European option pricing on the GPU (async, WASM only).
    pub async fn mc_european_gpu_async(
        spot: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
        num_paths: u32,
        num_steps: u32,
        seed: u32,
        is_call: bool,
    ) -> Result<GpuMcResult, String> {
        let discount = (-rate * expiry).exp();
        let params = build_params(
            spot, strike, rate, vol, expiry, num_paths, num_steps, seed, is_call,
        );

        let ctx = ensure_gpu_ctx().await?;
        let (staging_buffer, _payoff_size) = encode_and_submit(&ctx, params, num_paths as usize);
        readback_async(&ctx.device, &staging_buffer, num_paths as usize, discount).await
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::mc_european_gpu_async;
