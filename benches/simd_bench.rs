use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use openferric::engines::analytic::{bs_price_batch, normal_cdf_batch_approx};
use openferric::engines::monte_carlo::{mc_european_call_soa, mc_european_call_soa_scalar};
use openferric::pricing::{OptionType, european::black_scholes_price};
use statrs::distribution::{ContinuousCDF, Normal};

fn make_spot_strike(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut spots = Vec::with_capacity(n);
    let mut strikes = Vec::with_capacity(n);
    for i in 0..n {
        spots.push(60.0 + (i % 1_000) as f64 * 0.12);
        strikes.push(55.0 + (i % 900) as f64 * 0.11);
    }
    (spots, strikes)
}

fn bench_bs_scalar_vs_simd(c: &mut Criterion) {
    let r: f64 = 0.03;
    let q: f64 = 0.01;
    let vol: f64 = 0.2;
    let t: f64 = 1.0;

    let mut group = c.benchmark_group("bs_price_scalar_vs_simd");
    for n in [1_000usize, 10_000, 100_000] {
        let (spots, strikes) = make_spot_strike(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let adjusted_spot = spots[i] * (-q * t).exp();
                    out.push(black_scholes_price(
                        OptionType::Call,
                        adjusted_spot,
                        strikes[i],
                        r,
                        vol,
                        t,
                    ));
                }
                black_box(out)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd_dispatch", n), &n, |b, _| {
            b.iter(|| black_box(bs_price_batch(&spots, &strikes, r, q, vol, t, true)))
        });
    }
    group.finish();
}

fn bench_mc_scalar_vs_simd(c: &mut Criterion) {
    let s0: f64 = 100.0;
    let k: f64 = 100.0;
    let r: f64 = 0.03;
    let q: f64 = 0.01;
    let vol: f64 = 0.2;
    let t: f64 = 1.0;
    let paths = 100_000usize;
    let steps = 64usize;
    let seed = 42_u64;

    let mut group = c.benchmark_group("mc_european_call_scalar_vs_simd");
    group.throughput(Throughput::Elements(paths as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            black_box(mc_european_call_soa_scalar(
                s0, k, r, q, vol, t, paths, steps, seed,
            ))
        })
    });

    group.bench_function("simd_dispatch", |b| {
        b.iter(|| {
            black_box(mc_european_call_soa(
                s0, k, r, q, vol, t, paths, steps, seed,
            ))
        })
    });

    group.finish();
}

fn bench_normal_cdf_scalar_vs_simd(c: &mut Criterion) {
    let n = 1_000_000usize;
    let normal = Normal::new(0.0, 1.0).expect("normal distribution should be valid");
    let mut xs = Vec::with_capacity(n);
    for i in 0..n {
        xs.push(-6.0 + 12.0 * (i as f64) / (n as f64));
    }

    let mut group = c.benchmark_group("normal_cdf_scalar_vs_simd");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("scalar_statrs", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(n);
            for &x in &xs {
                out.push(normal.cdf(x));
            }
            black_box(out)
        })
    });

    group.bench_function("simd_dispatch_as", |b| {
        b.iter(|| black_box(normal_cdf_batch_approx(&xs)))
    });

    group.finish();
}

criterion_group!(
    simd_benches,
    bench_bs_scalar_vs_simd,
    bench_mc_scalar_vs_simd,
    bench_normal_cdf_scalar_vs_simd
);
criterion_main!(simd_benches);
