use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use openferric::engines::analytic::{bs_price_batch, normal_cdf_batch_approx};
use openferric::engines::monte_carlo::{mc_european_call_soa, mc_european_call_soa_scalar};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use openferric::math::simd_math::{exp_f64x4, ln_f64x4, load_f64x4, store_f64x4};
use openferric::pricing::{OptionType, european::black_scholes_price};
use statrs::distribution::{ContinuousCDF, Normal};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::hint::black_box;

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

        // "before/after" view for the SIMD path now that ln/exp are vectorized.
        group.bench_with_input(
            BenchmarkId::new("simd_dispatch_vec_ln_exp", n),
            &n,
            |b, _| b.iter(|| black_box(bs_price_batch(&spots, &strikes, r, q, vol, t, true))),
        );
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln_batch_avx2(xs: &[f64], out: &mut [f64]) {
    let mut i = 0usize;
    while i + 4 <= xs.len() {
        // SAFETY: loop guarantees in-bounds 4-wide accesses.
        let x = unsafe { load_f64x4(xs, i) };
        // SAFETY: target feature is enabled by this function.
        let y = unsafe { ln_f64x4(x) };
        // SAFETY: loop guarantees in-bounds 4-wide accesses.
        unsafe { store_f64x4(out, i, y) };
        i += 4;
    }
    while i < xs.len() {
        out[i] = xs[i].ln();
        i += 1;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_batch_avx2(xs: &[f64], out: &mut [f64]) {
    let mut i = 0usize;
    while i + 4 <= xs.len() {
        // SAFETY: loop guarantees in-bounds 4-wide accesses.
        let x = unsafe { load_f64x4(xs, i) };
        // SAFETY: target feature is enabled by this function.
        let y = unsafe { exp_f64x4(x) };
        // SAFETY: loop guarantees in-bounds 4-wide accesses.
        unsafe { store_f64x4(out, i, y) };
        i += 4;
    }
    while i < xs.len() {
        out[i] = xs[i].exp();
        i += 1;
    }
}

fn bench_ln_exp_scalar_vs_simd(c: &mut Criterion) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let n = 100_000usize;
        let mut xs_ln = Vec::with_capacity(n);
        let mut xs_exp = Vec::with_capacity(n);
        for i in 0..n {
            let p = -300.0 + 600.0 * (i as f64) / ((n - 1) as f64);
            xs_ln.push(10f64.powf(p));
            xs_exp.push(-700.0 + 1_400.0 * (i as f64) / ((n - 1) as f64));
        }

        let mut group = c.benchmark_group("ln_exp_scalar_vs_simd");
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function("ln_scalar", |b| {
            b.iter(|| {
                let out: Vec<f64> = xs_ln.iter().map(|x| x.ln()).collect();
                black_box(out)
            })
        });

        group.bench_function("exp_scalar", |b| {
            b.iter(|| {
                let out: Vec<f64> = xs_exp.iter().map(|x| x.exp()).collect();
                black_box(out)
            })
        });

        group.bench_function("ln_simd_avx2", |b| {
            let mut out = vec![0.0_f64; n];
            b.iter(|| {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: runtime feature check above.
                    unsafe { ln_batch_avx2(&xs_ln, &mut out) };
                } else {
                    for (dst, &x) in out.iter_mut().zip(xs_ln.iter()) {
                        *dst = x.ln();
                    }
                }
                black_box(&out)
            })
        });

        group.bench_function("exp_simd_avx2", |b| {
            let mut out = vec![0.0_f64; n];
            b.iter(|| {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: runtime feature check above.
                    unsafe { exp_batch_avx2(&xs_exp, &mut out) };
                } else {
                    for (dst, &x) in out.iter_mut().zip(xs_exp.iter()) {
                        *dst = x.exp();
                    }
                }
                black_box(&out)
            })
        });

        group.finish();
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        let _ = c;
    }
}

criterion_group!(
    simd_benches,
    bench_bs_scalar_vs_simd,
    bench_mc_scalar_vs_simd,
    bench_normal_cdf_scalar_vs_simd,
    bench_ln_exp_scalar_vs_simd
);
criterion_main!(simd_benches);
