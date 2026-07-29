use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::PricingEngine;
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::pde::AdiHestonEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::models::Heston;
use rayon::ThreadPoolBuilder;

fn market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.20)
        .build()
        .expect("benchmark market")
}

fn thread_counts() -> Vec<usize> {
    let logical = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut counts = vec![1, 2, 4, 8, 16, logical];
    counts.retain(|&threads| threads <= logical);
    counts.sort_unstable();
    counts.dedup();
    counts
}

fn bench_lsm_scaling(c: &mut Criterion) {
    let market = market();
    let option = VanillaOption::american_put(100.0, 1.0);
    let engine = LongstaffSchwartzEngine::new(50_000, 64, 42);
    let mut group = c.benchmark_group("lsm_time_major_50k_x_64");
    group.sample_size(10);

    for threads in thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        group.bench_with_input(
            BenchmarkId::new("rayon_threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    let price = pool
                        .install(|| engine.price(black_box(&option), black_box(&market)))
                        .expect("LSM price")
                        .price;
                    black_box(price)
                })
            },
        );
    }
    group.finish();
}

fn bench_adi_scaling(c: &mut Criterion) {
    let market = market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let model = Heston {
        mu: 0.0,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.25,
        rho: -0.6,
        v0: 0.04,
    };
    let engine = AdiHestonEngine::new(model, 48, 128, 96);
    let mut group = c.benchmark_group("adi_line_parallel_48_x_128_x_96");
    group.sample_size(10);

    for threads in thread_counts() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        group.bench_with_input(
            BenchmarkId::new("rayon_threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    let price = pool
                        .install(|| engine.price(black_box(&option), black_box(&market)))
                        .expect("ADI price")
                        .price;
                    black_box(price)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(solver_benches, bench_lsm_scaling, bench_adi_scaling);
criterion_main!(solver_benches);
