use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use openferric::core::{ExecutionPolicy, PricingEngine};
use openferric::engines::gpu::{gpu_is_ready, mc_european_gpu, prewarm_gpu};
use openferric::engines::monte_carlo::MonteCarloPricingEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use std::hint::black_box;
use std::time::Instant;

fn bench_gpu_terminal_european(c: &mut Criterion) {
    // Device creation cannot be repeated in-process because the production
    // context is intentionally cached. Record the real first initialization
    // once, then keep it outside Criterion's warm dispatch measurements.
    assert!(
        !gpu_is_ready(),
        "another benchmark initialized the GPU first"
    );
    let cold_start = Instant::now();
    if let Err(error) = prewarm_gpu() {
        if std::env::var_os("OPENFERRIC_REQUIRE_GPU").is_some() {
            panic!("GPU-labelled benchmark runner could not initialize WebGPU: {error}");
        }
        eprintln!("skipping GPU benchmark on a host without a usable adapter: {error}");
        return;
    }
    eprintln!(
        "openferric_gpu_cold_initialization_seconds={:.9}",
        cold_start.elapsed().as_secs_f64()
    );

    let mut group = c.benchmark_group("gpu_terminal_european");
    group.sample_size(30);
    for num_paths in [100_000_usize, 1_000_000] {
        group.throughput(Throughput::Elements(num_paths as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_paths),
            &num_paths,
            |b, &paths| {
                b.iter(|| {
                    black_box(
                        mc_european_gpu(
                            100.0,
                            100.0,
                            0.05,
                            0.20,
                            1.0,
                            black_box(paths),
                            1,
                            42_u64,
                            true,
                        )
                        .expect("GPU initialized during benchmark setup"),
                    )
                })
            },
        );
    }
    group.finish();

    // Keep a same-process CPU comparator beside the warm GPU result. The GPU
    // workflow enables Rayon; gpu-only local builds retain a scalar fallback.
    // This makes the timing actionable by exposing the host CPU opportunity
    // cost.
    let market = Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.20)
        .build()
        .expect("benchmark market should be valid");
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut cpu_group = c.benchmark_group("gpu_terminal_european_cpu_reference");
    cpu_group.sample_size(30);
    for num_paths in [100_000_usize, 1_000_000] {
        #[cfg(feature = "parallel")]
        let cpu_policy = ExecutionPolicy::Parallel;
        #[cfg(not(feature = "parallel"))]
        let cpu_policy = ExecutionPolicy::Scalar;
        let engine =
            MonteCarloPricingEngine::new(num_paths, 1, 42).with_execution_policy(cpu_policy);
        cpu_group.throughput(Throughput::Elements(num_paths as u64));
        cpu_group.bench_with_input(
            BenchmarkId::from_parameter(num_paths),
            &num_paths,
            |b, _| {
                b.iter(|| {
                    black_box(
                        engine
                            .price(black_box(&option), black_box(&market))
                            .expect("CPU reference pricing should succeed"),
                    )
                })
            },
        );
    }
    cpu_group.finish();
}

criterion_group!(benches, bench_gpu_terminal_european);
criterion_main!(benches);
