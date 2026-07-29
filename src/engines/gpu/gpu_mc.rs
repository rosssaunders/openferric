//! Module `engines::gpu::gpu_mc`.
//!
//! GPU Monte Carlo European option pricing with on-device hierarchical reduction.
//!
//! Each GPU thread prices two terminal GBM samples from one Box--Muller pair.
//! A per-workgroup tree reduction in shared memory produces partial sums,
//! so only a tiny summary buffer (2 floats per workgroup) is read back
//! to the host — eliminating the main bandwidth bottleneck.
//!
//! WebGPU's portable numerical baseline is `f32`. Path generation and the
//! workgroup reductions therefore use `f32`; only the final reduction and
//! statistics are evaluated in `f64` on the host. The reported standard error
//! measures Monte Carlo sampling error, not floating-point error. Applications
//! with tight accuracy requirements should compare against a `f64` CPU backend.
//!
//! References: Hull (11th ed.) and standard quantitative-finance references.

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Shared types (both native and WASM)
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 256;
const PATHS_PER_INVOCATION: u32 = 2;

/// GPU-accelerated parameters matching the WGSL struct layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    spot: f32,
    strike: f32,
    terminal_drift: f32,
    terminal_vol: f32,
    num_paths: u32,
    seed: u32,
    is_call: u32,
    _padding: u32,
}

#[derive(Copy, Clone, Debug)]
struct ValidatedGpuRequest {
    params: GpuParams,
    num_paths: usize,
    num_workgroups: u32,
    output_size: u64,
    discount: f64,
}

/// Result from GPU Monte Carlo pricing.
///
/// GPU path generation and partial reductions use portable WebGPU `f32`
/// arithmetic. The host performs the final reduction in `f64`. `stderr`
/// describes sampling uncertainty only and does not include `f32` roundoff.
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
    param_buffer: wgpu::Buffer,
    /// Keeps a shared parameter buffer write adjacent to its submission.
    submission_lock: Mutex<()>,
    /// Completed dispatches return their buffers here for later calls. A call
    /// takes ownership while GPU work or mapping is in flight, so concurrent
    /// callers cannot overwrite one another.
    buffer_pool: Mutex<Vec<DispatchBuffers>>,
}

struct DispatchBuffers {
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u64,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn checked_f32(name: &str, value: f64) -> Result<f32, String> {
    let converted = value as f32;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(format!("{name} is outside the finite f32 range"))
    }
}

fn require_finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

/// Validate high-level inputs and build parameters for exact terminal GBM.
///
/// `num_steps` is retained in the public API for source compatibility with the
/// path-simulation engine. Terminal European pricing is exact under GBM and is
/// intentionally independent of that value, but zero and overflowing values
/// are still rejected as caller errors.
fn build_params(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u32,
    is_call: bool,
) -> Result<ValidatedGpuRequest, String> {
    require_finite("spot", spot)?;
    require_finite("strike", strike)?;
    require_finite("rate", rate)?;
    require_finite("vol", vol)?;
    require_finite("expiry", expiry)?;

    if spot <= 0.0 {
        return Err("spot must be positive".to_string());
    }
    if strike < 0.0 {
        return Err("strike must be non-negative".to_string());
    }
    if vol < 0.0 {
        return Err("vol must be non-negative".to_string());
    }
    if expiry < 0.0 {
        return Err("expiry must be non-negative".to_string());
    }
    if num_paths == 0 {
        return Err("num_paths must be positive".to_string());
    }
    if num_steps == 0 {
        return Err("num_steps must be positive".to_string());
    }

    let num_paths_u32 =
        u32::try_from(num_paths).map_err(|_| "num_paths exceeds the WebGPU u32 limit")?;
    let _num_steps_u32 =
        u32::try_from(num_steps).map_err(|_| "num_steps exceeds the WebGPU u32 limit")?;

    let terminal_drift = (rate - 0.5 * vol * vol) * expiry;
    let terminal_vol = vol * expiry.sqrt();
    let discount = (-rate * expiry).exp();
    if !discount.is_finite() {
        return Err("discount factor is not finite".to_string());
    }

    let invocations = num_paths_u32.div_ceil(PATHS_PER_INVOCATION);
    let num_workgroups = invocations.div_ceil(WORKGROUP_SIZE);
    let output_size = u64::from(num_workgroups)
        .checked_mul(2 * std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| "GPU output buffer size overflow".to_string())?;

    Ok(ValidatedGpuRequest {
        params: GpuParams {
            spot: checked_f32("spot", spot)?,
            strike: checked_f32("strike", strike)?,
            terminal_drift: checked_f32("terminal drift", terminal_drift)?,
            terminal_vol: checked_f32("terminal volatility", terminal_vol)?,
            num_paths: num_paths_u32,
            seed,
            is_call: u32::from(is_call),
            _padding: 0,
        },
        num_paths,
        num_workgroups,
        output_size,
        discount,
    })
}

/// Compute MC statistics from per-workgroup partial sums.
///
/// The GPU shader outputs `[sum_0, sum_sq_0, sum_1, sum_sq_1, ...]`
/// for each workgroup. This function reduces those to the final price
/// and standard error.
fn reduce_partial_sums(
    partial_sums: &[f32],
    num_paths: usize,
    discount: f64,
) -> Result<GpuMcResult, String> {
    if num_paths == 0 {
        return Err("cannot reduce zero GPU paths".to_string());
    }
    if partial_sums.is_empty() || !partial_sums.len().is_multiple_of(2) {
        return Err("GPU partial-sum buffer must contain sum/sum-square pairs".to_string());
    }
    if !discount.is_finite() || discount < 0.0 {
        return Err("discount factor must be finite and non-negative".to_string());
    }

    let n = num_paths as f64;
    let num_workgroups = partial_sums.len() / 2;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut sum_compensation = 0.0_f64;
    let mut sum_sq_compensation = 0.0_f64;

    for wg in 0..num_workgroups {
        let partial_sum = partial_sums[wg * 2] as f64;
        let partial_sum_sq = partial_sums[wg * 2 + 1] as f64;
        if !partial_sum.is_finite()
            || !partial_sum_sq.is_finite()
            || partial_sum < 0.0
            || partial_sum_sq < 0.0
        {
            return Err("GPU produced a non-finite or negative partial sum".to_string());
        }

        // Kahan accumulation reduces loss when many workgroups have very
        // different payoff magnitudes.
        let corrected_sum = partial_sum - sum_compensation;
        let next_sum = sum + corrected_sum;
        sum_compensation = (next_sum - sum) - corrected_sum;
        sum = next_sum;

        let corrected_sum_sq = partial_sum_sq - sum_sq_compensation;
        let next_sum_sq = sum_sq + corrected_sum_sq;
        sum_sq_compensation = (next_sum_sq - sum_sq) - corrected_sum_sq;
        sum_sq = next_sum_sq;
    }

    let mean = sum / n;
    let var = if num_paths > 1 {
        // Workgroup reductions happen in f32, so a mathematically zero
        // variance can arrive a few ulps below zero.
        let estimate = (sum_sq - sum * sum / n) / (n - 1.0);
        if estimate < 0.0 { 0.0 } else { estimate }
    } else {
        0.0
    };

    Ok(GpuMcResult {
        price: discount * mean,
        stderr: discount * (var / n).sqrt(),
    })
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
        .map_err(|e| format!("No GPU adapter found: {e}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("openferric MC"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
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
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MC compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // The parameter buffer is small and fixed-size, so keep it alive with the
    // pipeline and update it for each dispatch.
    let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC params"),
        size: std::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        pipeline,
        bind_group_layout,
        param_buffer,
        submission_lock: Mutex::new(()),
        buffer_pool: Mutex::new(Vec::new()),
    })
}

fn validate_device_limits(ctx: &GpuContext, request: &ValidatedGpuRequest) -> Result<(), String> {
    let limits = ctx.device.limits();
    if request.num_workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(format!(
            "num_paths requires {} workgroups, exceeding this adapter's per-dimension limit of {}",
            request.num_workgroups, limits.max_compute_workgroups_per_dimension
        ));
    }
    if request.output_size > limits.max_buffer_size {
        return Err(format!(
            "GPU output requires {} bytes, exceeding this adapter's buffer limit of {}",
            request.output_size, limits.max_buffer_size
        ));
    }
    if request.output_size > u64::from(limits.max_storage_buffer_binding_size) {
        return Err(format!(
            "GPU output requires {} bytes, exceeding this adapter's storage binding limit of {}",
            request.output_size, limits.max_storage_buffer_binding_size
        ));
    }
    Ok(())
}

/// Encode and submit the compute dispatch, returning the staging buffer
/// and the number of bytes to read back.
///
/// The output buffer holds 2 floats (sum, sum_sq) per workgroup rather
/// than one float per path. Each invocation consumes a Box--Muller pair,
/// so one workgroup represents up to 512 paths.
fn acquire_dispatch_buffers(
    ctx: &GpuContext,
    required_size: u64,
) -> Result<DispatchBuffers, String> {
    let mut pool = ctx
        .buffer_pool
        .lock()
        .map_err(|_| "GPU buffer-pool lock is poisoned".to_string())?;
    if let Some(index) = pool
        .iter()
        .position(|buffers| buffers.capacity >= required_size)
    {
        return Ok(pool.swap_remove(index));
    }
    drop(pool);

    let output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partial_sums"),
        size: required_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: required_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC bind group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ctx.param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });

    Ok(DispatchBuffers {
        output,
        staging,
        bind_group,
        capacity: required_size,
    })
}

fn release_dispatch_buffers(ctx: &GpuContext, buffers: DispatchBuffers) {
    if let Ok(mut pool) = ctx.buffer_pool.lock() {
        // Two reusable sets preserve allocation-free steady-state operation
        // for a small amount of overlapping async work without unbounded
        // retention after bursts.
        if pool.len() < 2 {
            pool.push(buffers);
        }
    }
}

fn encode_and_submit(
    ctx: &GpuContext,
    request: &ValidatedGpuRequest,
) -> Result<DispatchBuffers, String> {
    validate_device_limits(ctx, request)?;

    // Prevent another native caller from inserting a write/submit pair between
    // this request's parameter update and command submission.
    let _submission_guard = ctx
        .submission_lock
        .lock()
        .map_err(|_| "GPU submission lock is poisoned".to_string())?;

    let device = &ctx.device;
    let queue = &ctx.queue;

    let buffers = acquire_dispatch_buffers(ctx, request.output_size)?;
    queue.write_buffer(&ctx.param_buffer, 0, bytemuck::bytes_of(&request.params));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        pass.dispatch_workgroups(request.num_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.staging, 0, request.output_size);
    queue.submit(std::iter::once(encoder.finish()));

    Ok(buffers)
}

// ===========================================================================
// Native-only (sync via pollster + OnceLock)
// ===========================================================================

#[cfg(not(target_family = "wasm"))]
mod native {
    use super::*;
    use std::sync::OnceLock;

    /// Global GPU context cache. Only successful initialization is cached, so a
    /// transient adapter/device failure can be retried by a later call.
    static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

    fn get_or_init_gpu() -> Result<&'static GpuContext, String> {
        if let Some(ctx) = GPU_CTX.get() {
            return Ok(ctx);
        }

        let initialized = pollster::block_on(init_gpu_context())?;
        // Multiple first callers may initialize concurrently. The loser drops
        // its context and uses the successfully cached one.
        let _ = GPU_CTX.set(initialized);
        GPU_CTX
            .get()
            .ok_or_else(|| "GPU context cache initialization failed".to_string())
    }

    /// Readback partial sums using blocking poll + mpsc channel (native only).
    fn readback_blocking(
        device: &wgpu::Device,
        staging_buffer: &wgpu::Buffer,
        output_size: u64,
        num_paths: usize,
        discount: f64,
    ) -> Result<GpuMcResult, String> {
        let buffer_slice = staging_buffer.slice(..output_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        receiver
            .recv()
            .map_err(|e| format!("GPU readback failed: {e}"))?
            .map_err(|e| format!("GPU buffer map failed: {e}"))?;

        let data = buffer_slice.get_mapped_range();
        let partial_sums: &[f32] = bytemuck::cast_slice(&data);
        let result = reduce_partial_sums(partial_sums, num_paths, discount);
        drop(data);
        staging_buffer.unmap();
        result
    }

    /// Run Monte Carlo European option pricing on the GPU (synchronous, native only).
    ///
    /// Uses wgpu to dispatch an exact-terminal GBM compute shader. Each GPU
    /// invocation prices two paths using both normals from one Box--Muller pair.
    /// On-device hierarchical reduction produces per-workgroup partial sums;
    /// only a tiny summary buffer is read back for final reduction on CPU.
    ///
    /// The GPU device, queue, and pipeline are cached globally so subsequent calls
    /// skip initialization. Failed initialization is not cached and can be retried.
    ///
    /// `num_steps` is validated but does not affect the result because terminal
    /// GBM sampling is exact for a European payoff. GPU arithmetic is `f32`;
    /// the returned standard error does not include floating-point error.
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
        let request = build_params(
            spot, strike, rate, vol, expiry, num_paths, num_steps, seed, is_call,
        )?;

        let ctx = get_or_init_gpu()?;
        let buffers = encode_and_submit(ctx, &request)?;
        let result = readback_blocking(
            &ctx.device,
            &buffers.staging,
            request.output_size,
            request.num_paths,
            request.discount,
        );
        if result.is_ok() {
            release_dispatch_buffers(ctx, buffers);
        }
        result
    }
}

#[cfg(not(target_family = "wasm"))]
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

    /// Readback partial sums using callback + JsFuture yield loop (WASM only).
    async fn readback_async(
        device: &wgpu::Device,
        staging_buffer: &wgpu::Buffer,
        output_size: u64,
        num_paths: usize,
        discount: f64,
    ) -> Result<GpuMcResult, String> {
        let buffer_slice = staging_buffer.slice(..output_size);

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
            let _ = device.poll(wgpu::PollType::Poll);
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
        let partial_sums: &[f32] = bytemuck::cast_slice(&data);
        let result = reduce_partial_sums(partial_sums, num_paths, discount);
        drop(data);
        staging_buffer.unmap();
        result
    }

    /// Run Monte Carlo European option pricing on the GPU (async, WASM only).
    ///
    /// Terminal GBM sampling is exact, so `num_steps` is validated for API
    /// consistency but does not affect the result. WebGPU arithmetic is `f32`.
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
        let request = build_params(
            spot,
            strike,
            rate,
            vol,
            expiry,
            num_paths as usize,
            num_steps as usize,
            seed,
            is_call,
        )?;

        let ctx = ensure_gpu_ctx().await?;
        let buffers = encode_and_submit(&ctx, &request)?;
        let result = readback_async(
            &ctx.device,
            &buffers.staging,
            request.output_size,
            request.num_paths,
            request.discount,
        )
        .await;
        if result.is_ok() {
            release_dispatch_buffers(&ctx, buffers);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::mc_european_gpu_async;

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_ARGS: (f64, f64, f64, f64, f64, usize, usize, u32, bool) =
        (100.0, 105.0, 0.03, 0.20, 1.5, 10_001, 252, 42, true);

    fn valid_request() -> ValidatedGpuRequest {
        let (spot, strike, rate, vol, expiry, paths, steps, seed, is_call) = VALID_ARGS;
        build_params(spot, strike, rate, vol, expiry, paths, steps, seed, is_call)
            .expect("standard GPU inputs should validate")
    }

    #[test]
    fn gpu_params_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<GpuParams>(), 32);
        assert_eq!(std::mem::align_of::<GpuParams>(), 4);
    }

    #[test]
    fn build_params_uses_exact_terminal_distribution() {
        let one_step = build_params(100.0, 105.0, 0.03, 0.20, 1.5, 10_001, 1, 42, true)
            .expect("one step should validate");
        let many_steps = valid_request();

        assert_eq!(one_step.params, many_steps.params);
        assert_eq!(one_step.num_workgroups, 20);
        assert_eq!(one_step.output_size, 160);
        assert!((one_step.params.terminal_drift - 0.015).abs() < 1.0e-7);
        assert!((one_step.params.terminal_vol - 0.20 * 1.5_f32.sqrt()).abs() < 1.0e-7);
        assert!((one_step.discount - (-0.045_f64).exp()).abs() < 1.0e-14);
    }

    #[test]
    fn build_params_accepts_expiry_and_zero_vol_boundaries() {
        let request = build_params(100.0, 100.0, -0.01, 0.0, 0.0, 1, 1, 0, false)
            .expect("expiry and volatility may be zero");
        assert_eq!(request.params.terminal_drift, -0.0);
        assert_eq!(request.params.terminal_vol, 0.0);
        assert_eq!(request.params.is_call, 0);
        assert_eq!(request.num_workgroups, 1);
    }

    #[test]
    fn build_params_rejects_invalid_domains() {
        let invalid_cases = [
            ((0.0, 100.0, 0.03, 0.2, 1.0, 10, 1), "spot must be positive"),
            (
                (100.0, -1.0, 0.03, 0.2, 1.0, 10, 1),
                "strike must be non-negative",
            ),
            (
                (100.0, 100.0, 0.03, -0.2, 1.0, 10, 1),
                "vol must be non-negative",
            ),
            (
                (100.0, 100.0, 0.03, 0.2, -1.0, 10, 1),
                "expiry must be non-negative",
            ),
            (
                (100.0, 100.0, 0.03, 0.2, 1.0, 0, 1),
                "num_paths must be positive",
            ),
            (
                (100.0, 100.0, 0.03, 0.2, 1.0, 10, 0),
                "num_steps must be positive",
            ),
        ];

        for ((spot, strike, rate, vol, expiry, paths, steps), expected) in invalid_cases {
            let error =
                build_params(spot, strike, rate, vol, expiry, paths, steps, 0, true).unwrap_err();
            assert_eq!(error, expected);
        }

        let non_finite = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for value in non_finite {
            assert!(build_params(value, 100.0, 0.03, 0.2, 1.0, 10, 1, 0, true).is_err());
            assert!(build_params(100.0, value, 0.03, 0.2, 1.0, 10, 1, 0, true).is_err());
            assert!(build_params(100.0, 100.0, value, 0.2, 1.0, 10, 1, 0, true).is_err());
            assert!(build_params(100.0, 100.0, 0.03, value, 1.0, 10, 1, 0, true).is_err());
            assert!(build_params(100.0, 100.0, 0.03, 0.2, value, 10, 1, 0, true).is_err());
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn build_params_rejects_u32_overflow() {
        let too_many = u32::MAX as usize + 1;
        assert_eq!(
            build_params(100.0, 100.0, 0.03, 0.2, 1.0, too_many, 1, 0, true).unwrap_err(),
            "num_paths exceeds the WebGPU u32 limit"
        );
        assert_eq!(
            build_params(100.0, 100.0, 0.03, 0.2, 1.0, 1, too_many, 0, true).unwrap_err(),
            "num_steps exceeds the WebGPU u32 limit"
        );
    }

    #[test]
    fn reduce_partial_sums_computes_sample_statistics() {
        // Four payoffs with sum=10 and sum of squares=30, split over two
        // workgroups. Sample variance is 5/3.
        let result =
            reduce_partial_sums(&[3.0, 5.0, 7.0, 25.0], 4, 0.5).expect("valid partial sums");
        assert!((result.price - 1.25).abs() < 1.0e-14);
        assert!((result.stderr - 0.5 * (5.0_f64 / 12.0).sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn reduce_partial_sums_handles_single_and_roundoff_cases() {
        let single = reduce_partial_sums(&[7.0, 49.0], 1, 0.9).expect("single path is valid");
        assert!((single.price - 6.3).abs() < 1.0e-14);
        assert_eq!(single.stderr, 0.0);

        // This inconsistent but finite summary yields a slightly negative raw
        // variance; f32 workgroup rounding can do the same for constant payoffs.
        let rounded = reduce_partial_sums(&[2.0, 1.999_999_9], 2, 1.0)
            .expect("small negative roundoff is clamped");
        assert_eq!(rounded.stderr, 0.0);
    }

    #[test]
    fn reduce_partial_sums_rejects_malformed_data() {
        assert!(reduce_partial_sums(&[], 1, 1.0).is_err());
        assert!(reduce_partial_sums(&[1.0], 1, 1.0).is_err());
        assert!(reduce_partial_sums(&[0.0, 0.0], 0, 1.0).is_err());
        assert!(reduce_partial_sums(&[f32::NAN, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_sums(&[-1.0, 1.0], 1, 1.0).is_err());
        assert!(reduce_partial_sums(&[1.0, 1.0], 1, f64::NAN).is_err());
    }

    #[test]
    fn wgsl_shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("mc_shader.wgsl"))
            .expect("GPU MC shader should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("GPU MC shader should validate");
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_price_is_consistent_with_black_scholes_when_adapter_available() {
        let price_once =
            || super::mc_european_gpu(100.0, 100.0, 0.05, 0.20, 1.0, 131_072, 252, 42, true);
        let result = match price_once() {
            Ok(result) => result,
            Err(error) if error.starts_with("No GPU adapter found:") => {
                eprintln!("skipping GPU integration assertion: {error}");
                return;
            }
            Err(error) => panic!("GPU adapter was found but pricing failed: {error}"),
        };

        // Black--Scholes call value for these inputs is approximately 10.4506.
        // Include both sampling uncertainty and a small allowance for portable
        // WebGPU f32 arithmetic.
        let tolerance = 8.0 * result.stderr + 0.05;
        assert!(
            (result.price - 10.450_583_572_185_565).abs() <= tolerance,
            "GPU price {} differed from Black-Scholes by more than {} (stderr {})",
            result.price,
            tolerance,
            result.stderr
        );
        assert!(result.stderr.is_finite() && result.stderr > 0.0);

        // The second call reuses the cached pipeline, parameter buffer, output
        // buffer, staging buffer, and bind group. Fixed path-indexed seeds make
        // the result reproducible.
        let repeated = price_once().expect("cached GPU resources should remain usable");
        assert_eq!(repeated.price.to_bits(), result.price.to_bits());
        assert_eq!(repeated.stderr.to_bits(), result.stderr.to_bits());

        // Reuse the larger pooled buffers for an odd path count. At zero
        // volatility the discounted payoff is deterministic.
        let odd = super::mc_european_gpu(100.0, 100.0, 0.05, 0.0, 1.0, 513, 7, 9, true)
            .expect("odd GPU path count should price successfully");
        let expected = 100.0 - 100.0 * (-0.05_f64).exp();
        assert!((odd.price - expected).abs() < 1.0e-4);
        assert!(odd.stderr < 1.0e-3);
    }
}
