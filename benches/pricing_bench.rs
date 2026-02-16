use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::PricingEngine;
use openferric::engines::analytic::{BarrierAnalyticEngine, BlackScholesEngine, HestonEngine};
use openferric::engines::monte_carlo::MonteCarloPricingEngine;
use openferric::engines::numerical::AmericanBinomialEngine;
use openferric::instruments::{BarrierOption, VanillaOption};
use openferric::market::Market;
use std::hint::black_box;

// Performance goals (guideline, measured on target hardware):
// - Black-Scholes European call: < 100 ns
// - Barrier analytic: < 200 ns
// - American binomial (500 steps): < 1 ms

fn benchmark_market() -> Market {
    Market::builder()
        .spot(100.0)
        .rate(0.05)
        .dividend_yield(0.0)
        .flat_vol(0.20)
        .build()
        .expect("benchmark market should be valid")
}

fn bench_black_scholes_european(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let engine = BlackScholesEngine::new();

    c.bench_function("black_scholes_european_call", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });
}

fn bench_american_binomial_steps(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::american_put(100.0, 1.0);
    let mut group = c.benchmark_group("american_binomial_put");

    for steps in [100_usize, 500, 1000] {
        let engine = AmericanBinomialEngine::new(steps);
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, _| {
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

fn bench_heston_european(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let engine = HestonEngine::new(
        0.04, // v0
        2.0,  // kappa
        0.04, // theta
        0.5,  // sigma_v
        -0.7, // rho
    );

    c.bench_function("heston_european_call", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });
}

fn bench_monte_carlo_european(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("mc_european_call");
    group.sample_size(10);

    for paths in [10_000_usize, 100_000] {
        let engine = MonteCarloPricingEngine::new(paths, 252, 42);
        group.bench_with_input(BenchmarkId::from_parameter(paths), &paths, |b, _| {
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

fn bench_barrier_analytic(c: &mut Criterion) {
    let market = benchmark_market();
    let option = BarrierOption::builder()
        .call()
        .strike(100.0)
        .expiry(1.0)
        .up_and_out(120.0)
        .rebate(0.0)
        .build()
        .expect("barrier option should be valid");
    let engine = BarrierAnalyticEngine::new();

    c.bench_function("barrier_analytic_up_and_out_call", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });
}

criterion_group!(
    pricing_benches,
    bench_black_scholes_european,
    bench_american_binomial_steps,
    bench_heston_european,
    bench_monte_carlo_european,
    bench_barrier_analytic
);
criterion_main!(pricing_benches);
