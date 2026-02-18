//! GPU Monte Carlo pricing engine using wgpu compute shaders.

use wgpu::util::DeviceExt;

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

/// Run Monte Carlo European option pricing on the GPU.
///
/// Uses wgpu to dispatch compute shaders that simulate GBM paths in parallel.
/// Each GPU thread simulates one complete path and computes its payoff.
/// Statistics are reduced on the CPU after readback.
///
/// # Arguments
/// * `spot` - Current asset price
/// * `strike` - Option strike price
/// * `rate` - Risk-free rate
/// * `vol` - Volatility
/// * `expiry` - Time to expiry in years
/// * `num_paths` - Number of Monte Carlo paths (mapped to GPU threads)
/// * `num_steps` - Number of time steps per path
/// * `seed` - RNG seed
/// * `is_call` - true for call, false for put
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
    let dt = expiry / num_steps as f64;
    let dt_drift = (rate - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-rate * expiry).exp();

    let params = GpuParams {
        spot: spot as f32,
        strike: strike as f32,
        rate: rate as f32,
        vol: vol as f32,
        expiry: expiry as f32,
        dt_drift: dt_drift as f32,
        dt_vol: dt_vol as f32,
        discount: discount as f32,
        num_steps: num_steps as u32,
        num_paths: num_paths as u32,
        seed,
        is_call: if is_call { 1 } else { 0 },
    };

    // Run the GPU pipeline synchronously using pollster.
    pollster::block_on(run_gpu_mc(params, num_paths, discount))
}

async fn run_gpu_mc(
    params: GpuParams,
    num_paths: usize,
    discount: f64,
) -> Result<GpuMcResult, String> {
    // Request GPU adapter and device.
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
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("openferric MC"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        }, None)
        .await
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

    // Load the compute shader.
    let shader_source = include_str!("mc_shader.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MC shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create the payoff output buffer on GPU.
    let payoff_size = (num_paths * std::mem::size_of::<f32>()) as u64;
    let payoff_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("payoffs"),
        size: payoff_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create uniform buffer for params.
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create staging buffer for readback.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: payoff_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create bind group layout and pipeline.
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

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MC compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC bind group"),
        layout: &bind_group_layout,
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

    // Dispatch compute shader.
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
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Copy payoffs to staging buffer.
    encoder.copy_buffer_to_buffer(&payoff_buffer, 0, &staging_buffer, 0, payoff_size);

    queue.submit(std::iter::once(encoder.finish()));

    // Map and read results.
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

    // Compute statistics on CPU (single pass).
    let n = num_paths as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &p in payoffs {
        let v = p as f64;
        sum += v;
        sum_sq += v * v;
    }
    drop(data);
    staging_buffer.unmap();

    let mean = sum / n;
    let var = if num_paths > 1 {
        (sum_sq - sum * sum / n) / (n - 1.0)
    } else {
        0.0
    };

    Ok(GpuMcResult {
        price: discount * mean,
        stderr: discount * (var / n).sqrt(),
    })
}
