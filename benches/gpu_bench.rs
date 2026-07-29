use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use openferric::engines::gpu::mc_european_gpu;
use std::hint::black_box;

fn bench_gpu_terminal_european(c: &mut Criterion) {
    // Initialize outside the timed loop. If this host has no usable adapter,
    // leave the benchmark group empty rather than reporting initialization as
    // pricing throughput.
    if let Err(error) = mc_european_gpu(100.0, 100.0, 0.05, 0.20, 1.0, 1_024, 1, 42, true) {
        eprintln!("skipping GPU benchmark: {error}");
        return;
    }

    let mut group = c.benchmark_group("gpu_terminal_european");
    group.sample_size(10);
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
                            42,
                            true,
                        )
                        .expect("GPU initialized during benchmark setup"),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gpu_terminal_european);
criterion_main!(benches);
