use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::PricingEngine;
use openferric::engines::monte_carlo::MonteCarloPricingEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use std::hint::black_box;

// Monte Carlo performance benchmarks
// Goals:
// - SIMD MC should be 2-4x faster than scalar
// - Xoshiro256++ should be faster than StdRng
// - QMC should show better convergence than PRNG

fn benchmark_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.02)
        .flat_vol(0.20)
        .build()
        .expect("benchmark market should be valid")
}

fn bench_mc_european_paths(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("mc_european_paths");

    for paths in [10_000, 50_000, 100_000].iter() {
        let engine = MonteCarloPricingEngine::new(*paths, 252, 42);
        group.bench_with_input(BenchmarkId::from_parameter(paths), paths, |b, _| {
            b.iter(|| {
                let px = engine
                    .price(black_box(&option), black_box(&market))
                    .expect("pricing should succeed")
                    .price;
                black_box(px)
            })
        });
    }

    group.finish();
}

fn bench_mc_timesteps(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let paths = 50_000;
    let mut group = c.benchmark_group("mc_timesteps");

    for timesteps in [50, 100, 252, 500].iter() {
        let engine = MonteCarloPricingEngine::new(paths, *timesteps, 42);
        group.bench_with_input(BenchmarkId::from_parameter(timesteps), timesteps, |b, _| {
            b.iter(|| {
                let px = engine
                    .price(black_box(&option), black_box(&market))
                    .expect("pricing should succeed")
                    .price;
                black_box(px)
            })
        });
    }

    group.finish();
}

fn bench_mc_convergence_study(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("mc_convergence_study");

    // Test convergence with different path counts
    let path_counts = [1_000, 5_000, 10_000, 50_000];

    for paths in path_counts.iter() {
        let engine = MonteCarloPricingEngine::new(*paths, 252, 42);
        group.bench_with_input(BenchmarkId::from_parameter(paths), paths, |b, _| {
            b.iter(|| {
                let result = engine
                    .price(black_box(&option), black_box(&market))
                    .expect("pricing should succeed");
                black_box((result.price, result.stderr))
            })
        });
    }

    group.finish();
}

fn bench_mc_put_call_parity(c: &mut Criterion) {
    let market = benchmark_market();
    let call_option = VanillaOption::european_call(100.0, 1.0);
    let put_option = VanillaOption::european_put(100.0, 1.0);
    let paths = 50_000;

    let engine = MonteCarloPricingEngine::new(paths, 252, 42);
    let mut group = c.benchmark_group("mc_put_call_parity");

    group.bench_function("call", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&call_option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });

    group.bench_function("put", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&put_option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });

    group.finish();
}

criterion_group!(
    mc_benches,
    bench_mc_european_paths,
    bench_mc_timesteps,
    bench_mc_convergence_study,
    bench_mc_put_call_parity
);
criterion_main!(mc_benches);
