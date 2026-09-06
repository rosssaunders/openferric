//! Module `engines::gpu::gpu_mc`.
//!
//! GPU Monte Carlo European option pricing with on-device hierarchical reduction.
//!
//! Each GPU thread prices two terminal GBM samples from one Box--Muller pair.
//! A per-workgroup tree reduction in shared memory produces centered-moment
//! summaries, so only a tiny buffer (3 floats per workgroup) is read back
//! to the host — eliminating the main bandwidth bottleneck.
//!
//! WebGPU's portable numerical baseline is `f32`. Path generation and the
//! workgroup reductions therefore use `f32`; payoff centering, final reduction,
//! and statistics are evaluated in `f64` on the host. The reported standard error
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
    centered_spot: f32,
    centered_intrinsic: f32,
    num_paths: u32,
    seed_low: u32,
    seed_high: u32,
    is_call: u32,
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

/// Small retryable single-flight state machine used by WASM initialization.
///
/// It is target-independent so its retry and deduplication transitions can be
/// unit-tested without requiring a browser WebGPU implementation.
#[cfg(any(test, target_arch = "wasm32"))]
enum RetryableInitState<C, P> {
    Empty,
    Initializing(P),
    Ready(C),
}

#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum InitSnapshot<C, P> {
    Empty,
    Initializing(P),
    Ready(C),
}

#[cfg(any(test, target_arch = "wasm32"))]
impl<C: Clone, P: Clone> RetryableInitState<C, P> {
    fn snapshot(&self) -> InitSnapshot<C, P> {
        match self {
            Self::Empty => InitSnapshot::Empty,
            Self::Initializing(promise) => InitSnapshot::Initializing(promise.clone()),
            Self::Ready(context) => InitSnapshot::Ready(context.clone()),
        }
    }

    fn begin(&mut self, promise: P) -> bool {
        if matches!(self, Self::Empty) {
            *self = Self::Initializing(promise);
            true
        } else {
            false
        }
    }

    fn succeed(&mut self, context: C) {
        *self = Self::Ready(context);
    }

    fn reset_after_failure(&mut self) {
        if matches!(self, Self::Initializing(_)) {
            *self = Self::Empty;
        }
    }
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
    seed: u64,
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

    let terminal_spot = spot * terminal_drift.exp();
    let terminal_intrinsic = if terminal_drift.abs() <= 0.5 {
        spot.mul_add(terminal_drift.exp_m1(), spot - strike)
    } else {
        terminal_spot - strike
    };
    let (centered_spot, centered_intrinsic) =
        if (terminal_spot as f32).is_finite() && (terminal_intrinsic as f32).is_finite() {
            (terminal_spot as f32, terminal_intrinsic as f32)
        } else {
            (0.0, 0.0)
        };

    let invocations = num_paths_u32.div_ceil(PATHS_PER_INVOCATION);
    let num_workgroups = invocations.div_ceil(WORKGROUP_SIZE);
    let output_size = u64::from(num_workgroups)
        .checked_mul(3 * std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| "GPU output buffer size overflow".to_string())?;

    Ok(ValidatedGpuRequest {
        params: GpuParams {
            spot: checked_f32("spot", spot)?,
            strike: checked_f32("strike", strike)?,
            terminal_drift: checked_f32("terminal drift", terminal_drift)?,
            terminal_vol: checked_f32("terminal volatility", terminal_vol)?,
            centered_spot,
            centered_intrinsic,
            num_paths: num_paths_u32,
            seed_low: seed as u32,
            seed_high: (seed >> 32) as u32,
            is_call: u32::from(is_call),
        },
        num_paths,
        num_workgroups,
        output_size,
        discount,
    })
}

/// Compute MC statistics from per-workgroup centered-moment summaries.
///
/// The GPU shader outputs `[count_0, mean_0, M2_0, count_1, mean_1, M2_1, ...]`.
/// This function uses Chan's parallel variance merge in `f64` to preserve
/// near-deterministic sampling variance without subtracting two large moments.
fn reduce_partial_moments(
    partial_moments: &[f32],
    num_paths: usize,
    discount: f64,
) -> Result<GpuMcResult, String> {
    if num_paths == 0 {
        return Err("cannot reduce zero GPU paths".to_string());
    }
    if partial_moments.is_empty() || !partial_moments.len().is_multiple_of(3) {
        return Err("GPU partial-moment buffer must contain count/mean/M2 triples".to_string());
    }
    if !discount.is_finite() || discount < 0.0 {
        return Err("discount factor must be finite and non-negative".to_string());
    }

    let num_workgroups = partial_moments.len() / 3;
    let mut count = 0_u64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for wg in 0..num_workgroups {
        let partial_count_f32 = partial_moments[wg * 3];
        let partial_mean = partial_moments[wg * 3 + 1] as f64;
        let partial_m2 = partial_moments[wg * 3 + 2] as f64;
        if !partial_count_f32.is_finite()
            || !partial_mean.is_finite()
            || !partial_m2.is_finite()
            || partial_count_f32 <= 0.0
            || partial_count_f32 > (WORKGROUP_SIZE * PATHS_PER_INVOCATION) as f32
            || partial_mean < 0.0
            || partial_m2 < 0.0
            || partial_count_f32.fract() != 0.0
        {
            return Err("GPU produced an invalid centered-moment summary".to_string());
        }

        let partial_count = partial_count_f32 as u64;
        let next_count = count
            .checked_add(partial_count)
            .ok_or_else(|| "GPU path count overflow during reduction".to_string())?;
        if next_count > num_paths as u64 {
            return Err("GPU partial-moment counts exceed the requested path count".to_string());
        }
        let delta = partial_mean - mean;
        let count_f64 = count as f64;
        let partial_count_f64 = partial_count as f64;
        let next_count_f64 = next_count as f64;
        mean += delta * (partial_count_f64 / next_count_f64);
        m2 += partial_m2 + delta * delta * (count_f64 * partial_count_f64 / next_count_f64);
        count = next_count;
    }

    if count != num_paths as u64 {
        return Err(format!(
            "GPU returned summaries for {count} paths; expected {num_paths}"
        ));
    }

    let n = count as f64;
    let var = if count > 1 { m2 / (n - 1.0) } else { 0.0 };

    let price = discount * mean;
    let stderr = discount * (var / n).sqrt();
    if !price.is_finite() || !stderr.is_finite() {
        return Err("discounted GPU statistics are not finite".to_string());
    }

    Ok(GpuMcResult { price, stderr })
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
/// The output buffer holds 3 floats (count, mean, M2) per workgroup rather
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
        label: Some("partial_moments"),
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
    /// Serializes the cold path without caching an initialization error.
    ///
    /// `OnceLock::get_or_init` cannot be used because its closure cannot return
    /// a retryable `Result`. The explicit mutex also prevents concurrent first
    /// callers from constructing and then discarding duplicate devices.
    static GPU_INIT_LOCK: Mutex<()> = Mutex::new(());

    #[cfg(test)]
    pub(super) static SUCCESSFUL_INITIALIZATIONS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn get_or_init_gpu() -> Result<&'static GpuContext, String> {
        if let Some(ctx) = GPU_CTX.get() {
            return Ok(ctx);
        }

        let _guard = GPU_INIT_LOCK
            .lock()
            .map_err(|_| "GPU initialization lock is poisoned".to_string())?;
        if let Some(ctx) = GPU_CTX.get() {
            return Ok(ctx);
        }

        let initialized = pollster::block_on(init_gpu_context())?;
        GPU_CTX
            .set(initialized)
            .map_err(|_| "GPU context cache initialization raced unexpectedly".to_string())?;
        #[cfg(test)]
        SUCCESSFUL_INITIALIZATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GPU_CTX
            .get()
            .ok_or_else(|| "GPU context cache initialization failed".to_string())
    }

    /// Return whether the process already has a successfully initialized GPU
    /// context. This check never performs device discovery or initialization.
    ///
    /// Automatic dispatch can use this to avoid imposing cold-start latency.
    /// Explicit GPU callers should use [`prewarm_gpu`] or call the pricing API.
    pub fn gpu_is_ready() -> bool {
        GPU_CTX.get().is_some()
    }

    /// Initialize and cache the native GPU context without running a pricing
    /// dispatch. Failed attempts are not cached and may be retried later.
    pub fn prewarm_gpu() -> Result<(), String> {
        get_or_init_gpu().map(|_| ())
    }

    /// Read back partial moments using blocking poll + mpsc (native only).
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
        let partial_moments: &[f32] = bytemuck::cast_slice(&data);
        let result = reduce_partial_moments(partial_moments, num_paths, discount);
        drop(data);
        staging_buffer.unmap();
        result
    }

    /// Run Monte Carlo European option pricing on the GPU (synchronous, native only).
    ///
    /// Uses wgpu to dispatch an exact-terminal GBM compute shader. Each GPU
    /// invocation prices two paths using both normals from one Box--Muller pair.
    /// On-device hierarchical reduction produces per-workgroup centered
    /// moments; only a tiny summary buffer is read back for final merging.
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
        seed: u64,
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
pub use native::{gpu_is_ready, mc_european_gpu, prewarm_gpu};

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
        static GPU_INIT_STATE:
            RefCell<RetryableInitState<Rc<GpuContext>, js_sys::Promise>> =
            const { RefCell::new(RetryableInitState::Empty) };
    }

    /// Lazily initialize (or reuse) the GPU context on the WASM worker.
    ///
    /// JavaScript can poll another Rust future while the first caller awaits
    /// adapter/device creation. Store and share that in-flight Promise so those
    /// callers do not create duplicate devices. A rejected Promise resets the
    /// state to `Empty`, preserving retry-after-transient-failure semantics.
    async fn ensure_gpu_ctx() -> Result<Rc<GpuContext>, String> {
        let snapshot = GPU_INIT_STATE.with(|state| state.borrow().snapshot());
        let promise = match snapshot {
            InitSnapshot::Ready(context) => return Ok(context),
            InitSnapshot::Initializing(promise) => promise,
            InitSnapshot::Empty => {
                let promise = wasm_bindgen_futures::future_to_promise(async {
                    match init_gpu_context().await {
                        Ok(context) => {
                            GPU_INIT_STATE.with(|state| {
                                state.borrow_mut().succeed(Rc::new(context));
                            });
                            Ok(wasm_bindgen::JsValue::UNDEFINED)
                        }
                        Err(error) => {
                            GPU_INIT_STATE.with(|state| {
                                state.borrow_mut().reset_after_failure();
                            });
                            Err(wasm_bindgen::JsValue::from_str(&error))
                        }
                    }
                });
                // There is no await between observing Empty and installing the
                // Promise, so another task on this worker cannot interleave.
                let installed =
                    GPU_INIT_STATE.with(|state| state.borrow_mut().begin(promise.clone()));
                debug_assert!(installed);
                promise
            }
        };

        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|error| {
                error
                    .as_string()
                    .unwrap_or_else(|| format!("GPU initialization failed: {error:?}"))
            })?;

        match GPU_INIT_STATE.with(|state| state.borrow().snapshot()) {
            InitSnapshot::Ready(context) => Ok(context),
            InitSnapshot::Empty | InitSnapshot::Initializing(_) => {
                Err("GPU initialization completed without a cached context".to_string())
            }
        }
    }

    /// Return whether this WASM worker already has an initialized WebGPU
    /// context. This check does not start initialization.
    pub fn gpu_is_ready() -> bool {
        matches!(
            GPU_INIT_STATE.with(|state| state.borrow().snapshot()),
            InitSnapshot::Ready(_)
        )
    }

    /// Initialize and cache this worker's WebGPU context without dispatching a
    /// pricing kernel. Concurrent callers share the same in-flight Promise.
    pub async fn prewarm_gpu_async() -> Result<(), String> {
        ensure_gpu_ctx().await.map(|_| ())
    }

    /// Read back partial moments using callback + JsFuture yield loop (WASM).
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
        let partial_moments: &[f32] = bytemuck::cast_slice(&data);
        let result = reduce_partial_moments(partial_moments, num_paths, discount);
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
        seed: u64,
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
pub use wasm::{gpu_is_ready, mc_european_gpu_async, prewarm_gpu_async};

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_ARGS: (f64, f64, f64, f64, f64, usize, usize, u64, bool) =
        (100.0, 105.0, 0.03, 0.20, 1.5, 10_001, 252, 42, true);
    const GPU_F32_PRICE_ABS_BUDGET: f64 = 2.0e-6;
    const GPU_F32_STDERR_RELATIVE_BUDGET: f64 = 2.0e-3;

    fn valid_request() -> ValidatedGpuRequest {
        let (spot, strike, rate, vol, expiry, paths, steps, seed, is_call) = VALID_ARGS;
        build_params(spot, strike, rate, vol, expiry, paths, steps, seed, is_call)
            .expect("standard GPU inputs should validate")
    }

    #[test]
    fn retryable_init_state_deduplicates_and_allows_retry_after_failure() {
        let mut state: RetryableInitState<u32, &str> = RetryableInitState::Empty;
        assert_eq!(state.snapshot(), InitSnapshot::Empty);

        assert!(state.begin("first promise"));
        assert!(!state.begin("duplicate promise"));
        assert_eq!(
            state.snapshot(),
            InitSnapshot::Initializing("first promise")
        );

        state.reset_after_failure();
        assert_eq!(state.snapshot(), InitSnapshot::Empty);
        assert!(state.begin("retry promise"));
        state.succeed(42);
        assert_eq!(state.snapshot(), InitSnapshot::Ready(42));

        // A stale rejection cannot clear an already-ready context.
        state.reset_after_failure();
        assert_eq!(state.snapshot(), InitSnapshot::Ready(42));
    }

    #[test]
    fn gpu_params_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<GpuParams>(), 40);
        assert_eq!(std::mem::align_of::<GpuParams>(), 4);

        let module = naga::front::wgsl::parse_str(include_str!("mc_shader.wgsl"))
            .expect("GPU MC shader should parse");
        let (_, params) = module
            .types
            .iter()
            .find(|(_, definition)| definition.name.as_deref() == Some("Params"))
            .expect("WGSL should define Params");
        let naga::TypeInner::Struct { members, span } = &params.inner else {
            panic!("WGSL Params should be a struct");
        };
        assert_eq!(*span as usize, std::mem::size_of::<GpuParams>());

        let offsets = [
            ("spot", std::mem::offset_of!(GpuParams, spot)),
            ("strike", std::mem::offset_of!(GpuParams, strike)),
            (
                "terminal_drift",
                std::mem::offset_of!(GpuParams, terminal_drift),
            ),
            (
                "terminal_vol",
                std::mem::offset_of!(GpuParams, terminal_vol),
            ),
            (
                "centered_spot",
                std::mem::offset_of!(GpuParams, centered_spot),
            ),
            (
                "centered_intrinsic",
                std::mem::offset_of!(GpuParams, centered_intrinsic),
            ),
            ("num_paths", std::mem::offset_of!(GpuParams, num_paths)),
            ("seed_low", std::mem::offset_of!(GpuParams, seed_low)),
            ("seed_high", std::mem::offset_of!(GpuParams, seed_high)),
            ("is_call", std::mem::offset_of!(GpuParams, is_call)),
        ];
        assert_eq!(members.len(), offsets.len());
        for (member, (name, offset)) in members.iter().zip(offsets) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset for {name}");
        }
    }

    #[test]
    fn build_params_uses_exact_terminal_distribution() {
        let one_step = build_params(100.0, 105.0, 0.03, 0.20, 1.5, 10_001, 1, 42, true)
            .expect("one step should validate");
        let many_steps = valid_request();

        assert_eq!(one_step.params, many_steps.params);
        assert_eq!(one_step.num_workgroups, 20);
        assert_eq!(one_step.output_size, 240);
        // Exact IEEE-754 roundings of the terminal GBM parameters. The f32
        // casts are bit-stable; discounting receives a tight cross-libm budget.
        assert_eq!(one_step.params.terminal_drift.to_bits(), 0x3c75_c28f);
        assert_eq!(one_step.params.terminal_vol.to_bits(), 0x3e7a_d3e7);
        let expected_discount = 0.955_997_481_833_1;
        assert!(
            (one_step.discount - expected_discount).abs() <= 8.0 * f64::EPSILON,
            "discount={} reference={expected_discount:.17}",
            one_step.discount
        );
    }

    #[test]
    fn build_params_accepts_expiry_and_zero_vol_boundaries() {
        let request = build_params(
            100.0,
            100.0,
            -0.01,
            0.0,
            0.0,
            1,
            1,
            0x1122_3344_5566_7788,
            false,
        )
        .expect("expiry and volatility may be zero");
        assert_eq!(request.params.terminal_drift, -0.0);
        assert_eq!(request.params.terminal_vol, 0.0);
        assert_eq!(request.params.seed_low, 0x5566_7788);
        assert_eq!(request.params.seed_high, 0x1122_3344);
        assert_eq!(request.params.is_call, 0);
        assert_eq!(request.num_workgroups, 1);
    }

    #[test]
    fn build_params_centers_payoffs_before_rounding_to_f32() {
        let close_strike = build_params(100.000001, 100.0, 0.0, 0.0, 1.0, 1, 1, 0, true)
            .expect("nearly equal spot and strike should validate");
        assert_eq!(close_strike.params.spot, close_strike.params.strike);
        assert_eq!(close_strike.params.centered_spot, 100.0);
        assert_eq!(
            close_strike.params.centered_intrinsic,
            9.999_999_974_752_427e-7_f64 as f32
        );

        let small_carry = build_params(100.0, 100.0, 1.0e-10, 0.0, 1.0, 1, 1, 0, true)
            .expect("small carry should validate");
        assert_eq!(small_carry.params.centered_spot, 100.0);
        assert_eq!(
            small_carry.params.centered_intrinsic,
            1.000_000_000_05e-8_f64 as f32
        );
    }

    #[test]
    fn build_params_preserves_uncentered_fallback_for_extreme_drifts() {
        for (rate, vol) in [(700.0, 0.0), (0.0, 20.0)] {
            let request = build_params(100.0, 100.0, rate, vol, 1.0, 1, 1, 0, true)
                .expect("finite original shader parameters should remain valid");
            assert_eq!(request.params.centered_spot, 0.0);
            assert!(request.params.centered_intrinsic.is_finite());
            assert!(request.params.terminal_drift.is_finite());
            assert_eq!(request.params.spot, 100.0);
            assert_eq!(request.params.strike, 100.0);
        }
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
    fn reduce_partial_moments_computes_sample_statistics() {
        // Payoffs [1, 2] and [3, 4], represented as two centered-moment
        // summaries. The combined sample variance is 5/3.
        let result = reduce_partial_moments(&[2.0, 1.5, 0.5, 2.0, 3.5, 0.5], 4, 0.5)
            .expect("valid partial moments");
        assert!((result.price - 1.25).abs() < 1.0e-14);
        assert!((result.stderr - 0.5 * (5.0_f64 / 12.0).sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn reduce_partial_moments_handles_single_and_large_offset_cases() {
        let single =
            reduce_partial_moments(&[1.0, 7.0, 0.0], 1, 0.9).expect("single path is valid");
        assert!((single.price - 6.3).abs() < 1.0e-14);
        assert_eq!(single.stderr, 0.0);

        // Centered moments preserve a small variance beside a very large mean;
        // raw sum/sum-square subtraction would lose it.
        let offset =
            reduce_partial_moments(&[2.0, 100_000_000.0, 2.0, 2.0, 100_000_000.0, 2.0], 4, 1.0)
                .expect("large-offset centered moments should remain stable");
        assert_eq!(offset.price, 100_000_000.0);
        assert!((offset.stderr - (1.0_f64 / 3.0).sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn reduce_partial_moments_rejects_malformed_data() {
        assert!(reduce_partial_moments(&[], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[0.0, 0.0, 0.0], 0, 1.0).is_err());
        assert!(reduce_partial_moments(&[0.0, 0.0, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0, f32::NAN, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0, -1.0, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.5, 1.0, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[513.0, 1.0, 0.0], 513, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0, 1.0, -1.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[2.0, 1.0, 0.0], 1, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0, 1.0, 0.0], 2, 1.0).is_err());
        assert!(reduce_partial_moments(&[1.0, 1.0, 0.0], 1, f64::NAN).is_err());
        assert!(
            reduce_partial_moments(&[1.0, f32::MAX, 0.0], 1, f64::MAX).is_err(),
            "overflowing discounted results must not be returned as infinity"
        );
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
    fn concurrent_gpu_prewarm_creates_one_context_when_adapter_available() {
        let outcomes = std::thread::scope(|scope| {
            (0..8)
                .map(|_| scope.spawn(super::prewarm_gpu))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("GPU prewarm thread panicked"))
                .collect::<Vec<_>>()
        });

        if outcomes.iter().all(
            |result| matches!(result, Err(error) if error.starts_with("No GPU adapter found:")),
        ) {
            eprintln!("skipping concurrent prewarm assertion: no GPU adapter");
            return;
        }
        for result in outcomes {
            result.expect("all concurrent prewarm callers should share the initialized context");
        }
        assert!(super::gpu_is_ready());
        assert_eq!(
            super::native::SUCCESSFUL_INITIALIZATIONS.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
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
        let tolerance = 4.0 * result.stderr + GPU_F32_PRICE_ABS_BUDGET;
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
        assert!(super::gpu_is_ready());
        assert_eq!(
            super::native::SUCCESSFUL_INITIALIZATIONS.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "concurrent cold callers must create only one cached GPU context"
        );

        // Reuse the larger pooled buffers for an odd path count. At zero
        // volatility the discounted payoff is deterministic.
        let odd = super::mc_european_gpu(100.0, 100.0, 0.05, 0.0, 1.0, 513, 7, 9, true)
            .expect("odd GPU path count should price successfully");
        let expected = 100.0 - 100.0 * (-0.05_f64).exp();
        assert!(
            (odd.price - expected).abs() <= GPU_F32_PRICE_ABS_BUDGET,
            "zero-volatility GPU price {} differs from exact target {}",
            odd.price,
            expected
        );
        assert_eq!(odd.stderr, 0.0);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_deterministic_payoffs_preserve_small_intrinsic_values_when_adapter_available() {
        if let Err(error) = super::prewarm_gpu() {
            if error.starts_with("No GPU adapter found:") {
                eprintln!("skipping GPU deterministic payoff assertion: {error}");
                return;
            }
            panic!("GPU adapter was found but initialization failed: {error}");
        }

        let references = [
            (100.0, 100.0, 0.05, 1.0, 4.877_057_549_928_599, 0.0),
            (100.0, 100.0, -0.05, 1.0, 0.0, 5.127_109_637_602_405),
            (100.0, 100.0, 0.75, 1.0, 52.763_344_725_898_53, 0.0),
            (100.0, 100.0, -0.75, 1.0, 0.0, 111.700_001_661_267_46),
            (100.000001, 100.0, 0.0, 1.0, 9.999_999_974_752_427e-7, 0.0),
            (100.0, 100.000001, 0.0, 1.0, 0.0, 9.999_999_974_752_427e-7),
            (100.0, 100.0, 1.0e-10, 1.0, 9.999_999_999_500_001e-9, 0.0),
            (100.0, 100.0, -1.0e-10, 1.0, 0.0, 1.000_000_000_050_000_1e-8),
            (100.000001, 100.0, 0.05, 0.0, 9.999_999_974_752_427e-7, 0.0),
            (100.0, 100.000001, 0.05, 0.0, 0.0, 9.999_999_974_752_427e-7),
            (100.0, 0.0, 0.05, 1.0, 100.0, 0.0),
            (100.0, 110.0, -0.05, 1.0, 0.0, 15.639_820_601_362_645),
            (100.0, 100.0, 0.05, 30.0, 77.686_983_985_157_01, 0.0),
        ];
        for (spot, strike, rate, expiry, call_price, put_price) in references {
            for (is_call, expected) in [(true, call_price), (false, put_price)] {
                for paths in [1, 2, 513] {
                    let result = super::mc_european_gpu(
                        spot, strike, rate, 0.0, expiry, paths, 1, 42, is_call,
                    )
                    .expect("deterministic GPU pricing should succeed");
                    assert!(
                        (result.price - expected).abs() <= f64::from(f32::EPSILON) * expected,
                        "spot={spot}, strike={strike}, rate={rate}, expiry={expiry}, \
                         paths={paths}, call={is_call}: price={} reference={expected}",
                        result.price
                    );
                    assert_eq!(result.stderr, 0.0);
                }
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_atm_payoffs_preserve_small_volatility_when_adapter_available() {
        if let Err(error) = super::prewarm_gpu() {
            if error.starts_with("No GPU adapter found:") {
                eprintln!("skipping GPU small-volatility assertion: {error}");
                return;
            }
            panic!("GPU adapter was found but initialization failed: {error}");
        }

        let references = [
            (1.0e-9, 3.989_422_804_014_327e-8),
            (1.0e-6, 3.989_422_804_014_16e-5),
            (0.001, 0.039_894_226_377_883_826),
            (0.02, 0.797_871_262_926_320_6),
            (0.1, 3.987_761_167_674_492_4),
            (0.2, 7.965_567_455_405_796),
        ];
        for (vol, expected) in references {
            for is_call in [true, false] {
                let result =
                    super::mc_european_gpu(100.0, 100.0, 0.0, vol, 1.0, 131_072, 1, 42, is_call)
                        .expect("small-volatility GPU pricing should succeed");
                assert!(result.stderr.is_finite() && result.stderr > 0.0);
                let tolerance = 4.0 * result.stderr + 8.0 * f64::from(f32::EPSILON) * expected;
                assert!(
                    (result.price - expected).abs() <= tolerance,
                    "vol={vol}, call={is_call}: price={} reference={expected}, tolerance={tolerance}",
                    result.price
                );
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_uses_all_seed_bits_without_known_seed_collisions_when_adapter_available() {
        let price_with_seed =
            |seed| super::mc_european_gpu(100.0, 100.0, 0.05, 0.20, 1.0, 131_072, 1, seed, true);
        let base = match price_with_seed(0) {
            Ok(result) => result,
            Err(error) if error.starts_with("No GPU adapter found:") => {
                eprintln!("skipping GPU seed assertion: {error}");
                return;
            }
            Err(error) => panic!("GPU adapter was found but pricing failed: {error}"),
        };

        // The former low^high fold mapped these public u64 seeds to exactly
        // the same shader seed. Adjacent seeds also used to shift and reuse
        // almost every path stream.
        for seed in [1_u64, 0x0000_0001_0000_0001] {
            let other = price_with_seed(seed).expect("cached GPU should price another seed");
            assert_ne!(
                (base.price.to_bits(), base.stderr.to_bits()),
                (other.price.to_bits(), other.stderr.to_bits()),
                "distinct public seed {seed:#018x} produced an identical GPU sample"
            );
        }

        let repeated = price_with_seed(0).expect("same seed should remain repeatable");
        assert_eq!(base.price.to_bits(), repeated.price.to_bits());
        assert_eq!(base.stderr.to_bits(), repeated.stderr.to_bits());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn adjacent_gpu_seeds_have_independent_replication_scale_when_adapter_available() {
        let paths = 65_536;
        let first = match super::mc_european_gpu(100.0, 100.0, 0.05, 0.20, 1.0, paths, 1, 0, true) {
            Ok(result) => result,
            Err(error) if error.starts_with("No GPU adapter found:") => {
                eprintln!("skipping GPU seed-independence assertion: {error}");
                return;
            }
            Err(error) => panic!("GPU adapter was found but pricing failed: {error}"),
        };

        let mut prices = vec![first.price];
        let mut stderr_sum = first.stderr;
        let mut stderr_sq_sum = first.stderr * first.stderr;
        for seed in 1_u64..16 {
            let result =
                super::mc_european_gpu(100.0, 100.0, 0.05, 0.20, 1.0, paths, 1, seed, true)
                    .expect("cached GPU should price adjacent seeds");
            prices.push(result.price);
            stderr_sum += result.stderr;
            stderr_sq_sum += result.stderr * result.stderr;
        }

        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let replicate_variance = prices
            .iter()
            .map(|price| (price - mean) * (price - mean))
            .sum::<f64>()
            / (prices.len() - 1) as f64;
        let replicate_sd = replicate_variance.sqrt();
        let mean_reported_stderr = stderr_sum / prices.len() as f64;
        let ratio = replicate_sd / mean_reported_stderr;

        // Independent replications should disperse on the scale of one
        // reported standard error. The old `global_id + seed` construction
        // reused all but one Box--Muller pair for adjacent seeds, making this
        // ratio roughly three orders of magnitude too small. These bounds are
        // the exact 0.01% and 99.99% chi-square quantiles for 15 degrees of
        // freedom, independently evaluated with SciPy 1.17.1.
        assert!(
            (0.400_681_749_917_726_5..=1.717_813_046_563_862_4).contains(&ratio),
            "adjacent-seed replication SD {replicate_sd} is inconsistent with \
             mean reported stderr {mean_reported_stderr} (ratio {ratio})"
        );

        let aggregate_stderr = stderr_sq_sum.sqrt() / prices.len() as f64;
        let black_scholes = 10.450_583_572_185_565;
        assert!(
            (mean - black_scholes).abs() <= 4.0 * aggregate_stderr + GPU_F32_PRICE_ABS_BUDGET,
            "mean GPU price {mean} differs from independent Black-Scholes \
             reference {black_scholes} (aggregate stderr {aggregate_stderr})"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_centered_moments_preserve_near_deterministic_stderr_when_adapter_available() {
        let paths = 131_072;
        let vol = 1.0e-6;
        let result = match super::mc_european_gpu(
            100.0,
            90.0,
            0.05,
            vol,
            1.0,
            paths,
            1,
            0xfeed_face_cafe_beef,
            true,
        ) {
            Ok(result) => result,
            Err(error) if error.starts_with("No GPU adapter found:") => {
                eprintln!("skipping GPU centered-moment assertion: {error}");
                return;
            }
            Err(error) => panic!("GPU adapter was found but pricing failed: {error}"),
        };

        // This deep-ITM payoff is terminal spot minus a constant to numerical
        // precision. Its discounted analytic standard error is
        // S * sqrt(expm1(vol^2*T) / N).
        let reference = 100.0 * ((vol * vol).exp_m1() / paths as f64).sqrt();
        let sample_sd_relative_budget = 4.0 / (2.0 * (paths - 1) as f64).sqrt();
        let relative_error = (result.stderr / reference - 1.0).abs();
        assert!(
            result.stderr.is_finite()
                && relative_error <= sample_sd_relative_budget + GPU_F32_STDERR_RELATIVE_BUDGET,
            "centered GPU stderr {} differs from analytic reference {} by {:.3}%",
            result.stderr,
            reference,
            100.0 * relative_error
        );

        let exact_price = 100.0 - 90.0 * (-0.05_f64).exp();
        assert!(
            (result.price - exact_price).abs() <= 4.0 * result.stderr + GPU_F32_PRICE_ABS_BUDGET,
            "near-deterministic GPU price {} differs from exact target {}",
            result.price,
            exact_price
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn concurrent_gpu_requests_do_not_alias_parameters_when_adapter_available() {
        if let Err(error) = super::prewarm_gpu() {
            if error.starts_with("No GPU adapter found:") {
                eprintln!("skipping concurrent GPU assertion: {error}");
                return;
            }
            panic!("GPU adapter was found but initialization failed: {error}");
        }

        let requests = [
            (90.0, 0.01, 0.10, 11_u64, true),
            (100.0, 0.03, 0.20, 22_u64, false),
            (110.0, -0.01, 0.30, 33_u64, true),
            (120.0, 0.07, 0.15, 44_u64, false),
        ];
        let run = |(strike, rate, vol, seed, is_call)| {
            super::mc_european_gpu(100.0, strike, rate, vol, 1.25, 65_537, 7, seed, is_call)
                .expect("GPU request should succeed")
        };
        let expected: Vec<_> = requests.iter().copied().map(run).collect();

        let actual = std::thread::scope(|scope| {
            requests
                .iter()
                .copied()
                .map(|request| scope.spawn(move || run(request)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("GPU caller thread panicked"))
                .collect::<Vec<_>>()
        });

        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(
                expected.price.to_bits(),
                actual.price.to_bits(),
                "concurrent request {index} used another request's parameters"
            );
            assert_eq!(
                expected.stderr.to_bits(),
                actual.stderr.to_bits(),
                "concurrent request {index} used another request's moment summary"
            );
        }
    }
}
