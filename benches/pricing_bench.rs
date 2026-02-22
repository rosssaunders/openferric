use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::{BarrierDirection, BarrierStyle, PricingEngine};
use openferric::engines::analytic::{BarrierAnalyticEngine, BlackScholesEngine};
use openferric::engines::fft::heston_price_fft;
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

fn bench_black_scholes_aad_single(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let engine = BlackScholesEngine::new();

    c.bench_function("black_scholes_aad_single", |b| {
        b.iter(|| {
            let result = engine
                .price_with_greeks_aad(black_box(&option), black_box(&market))
                .expect("aad pricing should succeed");
            black_box((result.price, result.greeks))
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
    c.bench_function("heston_fft_single", |b| {
        b.iter(|| {
            let strikes = [100.0];
            let prices =
                heston_price_fft(100.0, &strikes, 0.05, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0);
            black_box(prices)
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

fn bench_barrier_all_types(c: &mut Criterion) {
    let market = benchmark_market();
    let engine = BarrierAnalyticEngine::new();
    let mut group = c.benchmark_group("barrier_all_types");

    let barrier_specs = [
        (
            "down_and_out",
            BarrierDirection::Down,
            BarrierStyle::Out,
            90.0,
        ),
        ("up_and_out", BarrierDirection::Up, BarrierStyle::Out, 110.0),
        (
            "down_and_in",
            BarrierDirection::Down,
            BarrierStyle::In,
            90.0,
        ),
        ("up_and_in", BarrierDirection::Up, BarrierStyle::In, 110.0),
    ];

    for (name, direction, style, barrier_level) in barrier_specs.iter() {
        let mut builder = BarrierOption::builder()
            .call()
            .strike(100.0)
            .expiry(1.0)
            .rebate(0.0);

        builder = match (direction, style) {
            (BarrierDirection::Down, BarrierStyle::Out) => builder.down_and_out(*barrier_level),
            (BarrierDirection::Up, BarrierStyle::Out) => builder.up_and_out(*barrier_level),
            (BarrierDirection::Down, BarrierStyle::In) => builder.down_and_in(*barrier_level),
            (BarrierDirection::Up, BarrierStyle::In) => builder.up_and_in(*barrier_level),
        };

        let option = builder.build().expect("barrier option should be valid");

        group.bench_with_input(BenchmarkId::new("barrier", name), &name, |b, _| {
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

fn bench_black_scholes_path_counts(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let mut group = c.benchmark_group("bs_path_counts");

    for paths in [1_000_usize, 10_000, 100_000] {
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

fn bench_heston_fft_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("heston_fft_batch");
    let strikes_32: Vec<f64> = (0..32).map(|i| 70.0 + i as f64 * 2.0).collect();
    let strikes_128: Vec<f64> = (0..128).map(|i| 50.0 + i as f64 * 1.2).collect();

    group.bench_function("heston_fft_32", |b| {
        b.iter(|| {
            let prices = heston_price_fft(
                100.0,
                black_box(&strikes_32),
                0.05,
                0.0,
                0.04,
                2.0,
                0.04,
                0.5,
                -0.7,
                1.0,
            );
            black_box(prices)
        })
    });

    group.bench_function("heston_fft_128", |b| {
        b.iter(|| {
            let prices = heston_price_fft(
                100.0,
                black_box(&strikes_128),
                0.05,
                0.0,
                0.04,
                2.0,
                0.04,
                0.5,
                -0.7,
                1.0,
            );
            black_box(prices)
        })
    });

    group.finish();
}

criterion_group!(
    pricing_benches,
    bench_black_scholes_european,
    bench_black_scholes_aad_single,
    bench_american_binomial_steps,
    bench_heston_european,
    bench_monte_carlo_european,
    bench_barrier_analytic,
    bench_barrier_all_types,
    bench_black_scholes_path_counts,
    bench_heston_fft_batch
);
criterion_main!(pricing_benches);
