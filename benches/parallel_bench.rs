use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::OptionType;
use openferric::core::PricingEngine;
use openferric::engines::monte_carlo::{
    MonteCarloPricingEngine, mc_european_parallel, mc_european_qmc_with_seed,
    mc_european_sequential, mc_greeks_grid_parallel, mc_greeks_grid_sequential,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::math::fast_norm::fast_norm_cdf;
use openferric::math::fast_rng::{FastRngKind, Pcg64Rng, Xoshiro256Rng};
use openferric::pricing::european::black_scholes_price;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::ThreadPoolBuilder;
use statrs::distribution::{ContinuousCDF, Normal};
use std::hint::black_box;

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

fn bench_rng_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_generation_1m_f64");
    let n = 1_000_000;

    group.bench_function("chacha_std_rng", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(42);
            let sum = (0..n).map(|_| rng.random::<f64>()).sum::<f64>();
            black_box(sum)
        })
    });

    group.bench_function("xoshiro256plusplus", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256Rng::seed_from_u64(42);
            let sum = (0..n).map(|_| rng.next_f64()).sum::<f64>();
            black_box(sum)
        })
    });

    group.bench_function("pcg64", |b| {
        b.iter(|| {
            let mut rng = Pcg64Rng::seed_from_u64(42);
            let sum = (0..n).map(|_| rng.next_f64()).sum::<f64>();
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_mc_european_rng_backends(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("mc_european_50k_rng_backend");
    group.sample_size(10);

    let xoshiro = MonteCarloPricingEngine::new(50_000, 252, 42)
        .with_rng_kind(FastRngKind::Xoshiro256PlusPlus);
    let chacha = MonteCarloPricingEngine::new(50_000, 252, 42).with_rng_kind(FastRngKind::StdRng);

    group.bench_function("xoshiro256plusplus", |b| {
        b.iter(|| {
            let px = xoshiro
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });

    group.bench_function("chacha_std_rng", |b| {
        b.iter(|| {
            let px = chacha
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });

    group.finish();
}

fn bench_qmc_vs_mc_convergence(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
    let mut group = c.benchmark_group("qmc_vs_mc_convergence");
    group.sample_size(10);

    for paths in [10_000_usize, 50_000, 100_000] {
        let engine = MonteCarloPricingEngine::new(paths, 1, 42);

        group.bench_with_input(BenchmarkId::new("mc_abs_error", paths), &paths, |b, _| {
            b.iter(|| {
                let price = engine
                    .price(black_box(&option), black_box(&market))
                    .expect("pricing should succeed")
                    .price;
                black_box((price - bs).abs())
            })
        });

        group.bench_with_input(
            BenchmarkId::new("qmc_abs_error", paths),
            &paths,
            |b, &paths| {
                b.iter(|| {
                    let price = mc_european_qmc_with_seed(
                        black_box(&option),
                        black_box(&market),
                        paths,
                        1,
                        42,
                    )
                    .price;
                    black_box((price - bs).abs())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    parallel_benches,
    bench_mc_european_parallel,
    bench_greeks_grid,
    bench_fast_norm_cdf,
    bench_rng_generation,
    bench_mc_european_rng_backends,
    bench_qmc_vs_mc_convergence
);
criterion_main!(parallel_benches);
