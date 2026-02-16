use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use openferric::core::OptionType;
use openferric::engines::monte_carlo::{
    mc_european_parallel, mc_european_sequential, mc_greeks_grid_parallel,
    mc_greeks_grid_sequential,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::math::fast_norm::fast_norm_cdf;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{ContinuousCDF, Normal};

fn benchmark_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.20)
        .build()
        .expect("benchmark market should be valid")
}

fn bench_mc_european_parallel(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("mc_european_100k");
    group.sample_size(10);

    group.bench_function("single_thread", |b| {
        b.iter(|| {
            let px =
                mc_european_sequential(black_box(&option), black_box(&market), 100_000, 252).price;
            black_box(px)
        })
    });

    for threads in [2_usize, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("rayon_threads", threads),
            &threads,
            |b, &threads| {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool should build");
                b.iter(|| {
                    let px = pool.install(|| {
                        mc_european_parallel(black_box(&option), black_box(&market), 100_000, 252)
                            .price
                    });
                    black_box(px)
                })
            },
        );
    }

    group.finish();
}

fn bench_greeks_grid(c: &mut Criterion) {
    let spots = (0..100).map(|i| 60.0 + i as f64).collect::<Vec<_>>();
    let vols = (0..100)
        .map(|i| 0.05 + 0.0045 * i as f64)
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("greeks_grid_100x100");

    group.bench_function("sequential", |b| {
        b.iter(|| {
            let out = mc_greeks_grid_sequential(
                OptionType::Call,
                100.0,
                0.03,
                0.0,
                1.0,
                black_box(&spots),
                black_box(&vols),
            );
            black_box(out.len())
        })
    });

    group.bench_function("parallel", |b| {
        b.iter(|| {
            let out = mc_greeks_grid_parallel(
                OptionType::Call,
                100.0,
                0.03,
                0.0,
                1.0,
                black_box(&spots),
                black_box(&vols),
            );
            black_box(out.len())
        })
    });

    group.finish();
}

fn bench_fast_norm_cdf(c: &mut Criterion) {
    let xs = (0..1_000_000)
        .map(|i| -8.0 + (16.0 * i as f64 / 999_999.0))
        .collect::<Vec<_>>();
    let normal = Normal::new(0.0, 1.0).expect("normal(0,1) should build");
    let mut group = c.benchmark_group("norm_cdf_1m");

    group.bench_function("fast_norm_cdf", |b| {
        b.iter(|| {
            let sum = xs.iter().map(|&x| fast_norm_cdf(x)).sum::<f64>();
            black_box(sum)
        })
    });

    group.bench_function("statrs_norm_cdf", |b| {
        b.iter(|| {
            let sum = xs.iter().map(|&x| normal.cdf(x)).sum::<f64>();
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    parallel_benches,
    bench_mc_european_parallel,
    bench_greeks_grid,
    bench_fast_norm_cdf
);
criterion_main!(parallel_benches);
