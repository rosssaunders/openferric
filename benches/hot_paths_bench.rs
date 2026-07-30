//! Benchmarks for recently optimized hot paths.
//!
//! Covers, in order:
//! - SVI vol-surface queries (total-variance interpolation across expiries),
//! - analytic Black-Scholes price + Greeks (baseline reference),
//! - Monte Carlo European pricing: standard engine and the SoA SIMD path,
//! - ADI Heston PDE pricing on a moderate grid,
//! - Longstaff-Schwartz American put,
//! - autocallable price-only vs price-with-Greeks (the ~9x split),
//! - CDS survival-curve bootstrap (5 pillars),
//! - bounded Levenberg-Marquardt Heston calibration (1 LM iteration),
//! - rolling historical VaR (5k returns, window 250),
//! - gamma ladder on a 10-pillar yield curve.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use openferric::calibration::{Calibrator, HestonCalibrator, LmOptions, OptionVolQuote};
use openferric::core::{ExecutionPolicy, PricingEngine};
use openferric::credit::bootstrap_survival_curve_from_cds_spreads;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::engines::lsm::LongstaffSchwartzEngine;
use openferric::engines::monte_carlo::{
    MonteCarloPricingEngine, VarianceReduction, mc_european_call_soa, mc_european_call_soa_scalar,
};
use openferric::engines::pde::{AdiHestonEngine, AdiScheme};
use openferric::instruments::{Autocallable, VanillaOption};
use openferric::market::Market;
use openferric::math::approx_tier::AccuracyTier;
use openferric::models::stochastic::Heston;
use openferric::pricing::autocallable::{price_autocallable, price_autocallable_with_greeks};
use openferric::rates::YieldCurve;
use openferric::risk::sensitivities::{CurveBumpConfig, gamma_ladder};
use openferric::risk::var::rolling_historical_var_from_prices;
use openferric::vol::surface::{SviParams, VolSurface};
#[cfg(feature = "parallel")]
use openferric::{
    mc::{CpuExecutionPolicy, GbmPathGenerator, MonteCarloEngine as GenericMonteCarloEngine},
    models::Gbm,
};
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

/// Multi-expiry SVI surface; queries exercise the total-variance
/// interpolation path between expiry knots.
fn benchmark_vol_surface() -> VolSurface {
    let expiries = [0.1, 0.25, 0.5, 1.0, 2.0];
    let slices: Vec<(f64, SviParams)> = expiries
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let scale = 1.0 + 0.1 * i as f64;
            (
                t,
                SviParams {
                    a: 0.015 * t * scale,
                    b: 0.10 * scale,
                    rho: -0.3 - 0.05 * i as f64,
                    m: 0.01 * i as f64,
                    sigma: 0.25 + 0.02 * i as f64,
                },
            )
        })
        .collect();
    VolSurface::new(slices, 100.0).expect("benchmark surface should be valid")
}

fn bench_vol_surface_query(c: &mut Criterion) {
    let surface = benchmark_vol_surface();

    // Strided (K, T) grid: strikes 60..140, expiries deliberately off the
    // knot grid so every query interpolates between two SVI slices.
    let strikes: Vec<f64> = (0..16).map(|i| 60.0 + i as f64 * 5.0).collect();
    let times: Vec<f64> = (0..8).map(|j| 0.15 + j as f64 * 0.22).collect();

    c.bench_function("vol_surface_query_128", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &t in &times {
                for &k in &strikes {
                    acc += surface.vol(black_box(k), black_box(t));
                }
            }
            black_box(acc)
        })
    });
}

fn bench_black_scholes_price_greeks(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let engine = BlackScholesEngine::new();

    // Analytic BS returns price + full Greeks in one call; this is the
    // reference baseline for the other engines below.
    c.bench_function("bs_analytic_price_greeks", |b| {
        b.iter(|| {
            let result = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed");
            black_box((result.price, result.greeks))
        })
    });
}

fn bench_mc_european_50k(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);

    let mut group = c.benchmark_group("mc_european_50k");
    group.sample_size(10);

    let engine = MonteCarloPricingEngine::new(50_000, 64, 42);
    group.bench_function("standard_engine", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });

    // SoA scalar reference path.
    group.bench_function("soa_scalar", |b| {
        b.iter(|| {
            black_box(mc_european_call_soa_scalar(
                100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 50_000, 64, 42,
            ))
        })
    });

    // Runtime-dispatched SoA path; uses AVX2/NEON when built with the
    // `simd` feature and supported by the host.
    #[cfg(feature = "simd")]
    group.bench_function("soa_simd_dispatch", |b| {
        b.iter(|| {
            black_box(mc_european_call_soa(
                100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 50_000, 64, 42,
            ))
        })
    });
    #[cfg(not(feature = "simd"))]
    let _ = mc_european_call_soa;

    group.finish();
}

/// Keeps the exact-terminal Auto crossover calibrated. The implementation has
/// no SIMD-only allocation pass, so the cost model selects SIMD once one full
/// native vector is available. These sizes cover likely NEON, AVX2, and
/// AVX-512 boundaries plus the former 512-path cutoff.
fn bench_mc_exact_terminal_auto_crossover(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let simd_probe = MonteCarloPricingEngine::new(8, 1, 42)
        .with_execution_policy(ExecutionPolicy::Simd)
        .price(&option, &market);
    if simd_probe.is_err() {
        return;
    }

    let mut group = c.benchmark_group("mc_exact_terminal_auto_crossover");
    group.sample_size(20);
    for paths in [1_usize, 2, 3, 4, 7, 8, 15, 16, 31, 32, 255, 256, 511, 512] {
        group.throughput(Throughput::Elements(paths as u64));
        for (name, policy) in [
            ("scalar", ExecutionPolicy::Scalar),
            ("simd", ExecutionPolicy::Simd),
            ("auto", ExecutionPolicy::Auto),
        ] {
            let engine = MonteCarloPricingEngine::new(paths, 1, 42).with_execution_policy(policy);
            group.bench_with_input(BenchmarkId::new(name, paths), &paths, |b, _| {
                b.iter(|| {
                    black_box(
                        engine
                            .price(black_box(&option), black_box(&market))
                            .expect("selected backend should be available")
                            .price,
                    )
                })
            });
        }
        let antithetic = MonteCarloPricingEngine::new(paths, 1, 42)
            .with_variance_reduction(VarianceReduction::Antithetic);
        group.bench_with_input(
            BenchmarkId::new("auto-antithetic", paths),
            &paths,
            |b, _| {
                b.iter(|| {
                    black_box(
                        antithetic
                            .price(black_box(&option), black_box(&market))
                            .expect("Auto antithetic pricing should succeed")
                            .price,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_mc_exact_terminal_accuracy_tier(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let probe = MonteCarloPricingEngine::new(4, 1, 42)
        .with_execution_policy(ExecutionPolicy::Simd)
        .price(&option, &market);
    if probe.is_err() {
        return;
    }

    let paths = 100_000;
    let mut group = c.benchmark_group("mc_exact_terminal_accuracy_tier");
    group.sample_size(20);
    group.throughput(Throughput::Elements(paths as u64));
    for tier in [AccuracyTier::High, AccuracyTier::Fast] {
        let engine = MonteCarloPricingEngine::new(paths, 1, 42)
            .with_execution_policy(ExecutionPolicy::Simd)
            .with_accuracy_tier(tier);
        group.bench_function(format!("{tier:?}").to_lowercase(), |b| {
            b.iter(|| {
                black_box(
                    engine
                        .price(black_box(&option), black_box(&market))
                        .expect("SIMD backend should be available")
                        .price,
                )
            })
        });
    }
    group.finish();
}

/// Measures the Rayon crossover separately from the native-SIMD crossover.
/// The exact-terminal implementation schedules fixed 4,096-sample chunks, so
/// the matrix straddles every early chunk boundary for 1-, 2-, and host-sized
/// pools.
#[cfg(feature = "parallel")]
fn bench_mc_exact_terminal_rayon_crossover(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let serial_policy = if MonteCarloPricingEngine::new(8, 1, 42)
        .with_execution_policy(ExecutionPolicy::Simd)
        .price(&option, &market)
        .is_ok()
    {
        ExecutionPolicy::Simd
    } else {
        ExecutionPolicy::Scalar
    };
    let host_threads = rayon::current_num_threads();
    let mut pool_sizes = vec![1_usize, 2, host_threads];
    pool_sizes.sort_unstable();
    pool_sizes.dedup();

    let mut group = c.benchmark_group("mc_exact_terminal_rayon_crossover");
    group.sample_size(20);
    for threads in pool_sizes {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("benchmark thread pool");
        for paths in [
            4_095_usize,
            4_096,
            4_097,
            5_119,
            5_120,
            5_121,
            6_144,
            7_168,
            8_191,
            8_192,
            8_193,
            12_288,
            16_384,
            32_768,
        ] {
            group.throughput(Throughput::Elements(paths as u64));
            for (name, policy) in [
                ("serial", serial_policy),
                ("parallel", ExecutionPolicy::Parallel),
                ("auto", ExecutionPolicy::Auto),
            ] {
                let engine =
                    MonteCarloPricingEngine::new(paths, 1, 42).with_execution_policy(policy);
                group.bench_with_input(
                    BenchmarkId::new(format!("{threads}t-{name}"), paths),
                    &paths,
                    |b, _| {
                        b.iter(|| {
                            black_box(pool.install(|| {
                                engine
                                    .price(black_box(&option), black_box(&market))
                                    .expect("selected backend should be available")
                                    .price
                            }))
                        })
                    },
                );
            }
        }

        // Requested path counts map to 5,119 and 5,120 effective antithetic
        // samples, immediately below and at the calibrated Rayon boundary.
        for effective_samples in [5_119_usize, 5_120] {
            let requested_paths = 2 * effective_samples - 1;
            group.throughput(Throughput::Elements(requested_paths as u64));
            for (name, policy) in [
                ("serial-antithetic", serial_policy),
                ("parallel-antithetic", ExecutionPolicy::Parallel),
                ("auto-antithetic", ExecutionPolicy::Auto),
            ] {
                let engine = MonteCarloPricingEngine::new(requested_paths, 1, 42)
                    .with_execution_policy(policy)
                    .with_variance_reduction(VarianceReduction::Antithetic);
                group.bench_with_input(
                    BenchmarkId::new(format!("{threads}t-{name}"), requested_paths),
                    &requested_paths,
                    |b, _| {
                        b.iter(|| {
                            black_box(pool.install(|| {
                                engine
                                    .price(black_box(&option), black_box(&market))
                                    .expect("selected backend should be available")
                                    .price
                            }))
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_mc_exact_terminal_rayon_crossover(_: &mut Criterion) {}

/// Measures path-stepped callback simulation independently of exact-terminal
/// pricing. Each row has the same total `(paths × steps)` work at different
/// path lengths, exposing scheduling and scratch-buffer costs that a single
/// global cutoff cannot model.
#[cfg(feature = "parallel")]
fn bench_mc_path_rayon_crossover(c: &mut Criterion) {
    let host_threads = rayon::current_num_threads();
    let mut pool_sizes = vec![1_usize, 2, host_threads];
    pool_sizes.sort_unstable();
    pool_sizes.dedup();

    let mut group = c.benchmark_group("mc_path_rayon_crossover");
    group.sample_size(20);
    for threads in pool_sizes {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("benchmark thread pool");
        for work_items in [
            512_usize, 768, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536,
        ] {
            for steps in [1_usize, 8, 32] {
                let paths = work_items / steps;
                let generator = GbmPathGenerator {
                    model: Gbm {
                        mu: 0.05,
                        sigma: 0.2,
                    },
                    s0: 100.0,
                    maturity: 1.0,
                    steps,
                };
                group.throughput(Throughput::Elements(work_items as u64));
                for (name, policy) in [
                    ("serial", CpuExecutionPolicy::Scalar),
                    ("parallel", CpuExecutionPolicy::Parallel),
                    ("auto", CpuExecutionPolicy::Auto),
                ] {
                    let engine =
                        GenericMonteCarloEngine::new(paths, 42).with_execution_policy(policy);
                    group.bench_with_input(
                        BenchmarkId::new(format!("{threads}t-{name}-steps{steps}"), work_items),
                        &work_items,
                        |b, _| {
                            b.iter(|| {
                                black_box(pool.install(|| {
                                    engine.run(
                                        &generator,
                                        |path| (path[path.len() - 1] - 100.0).max(0.0),
                                        (-0.05_f64).exp(),
                                    )
                                }))
                            })
                        },
                    );
                }
            }
        }

        for effective_samples in [511_usize, 512, 767, 768] {
            let requested_paths = 2 * effective_samples - 1;
            let generator = GbmPathGenerator {
                model: Gbm {
                    mu: 0.05,
                    sigma: 0.2,
                },
                s0: 100.0,
                maturity: 1.0,
                steps: 1,
            };
            group.throughput(Throughput::Elements(requested_paths as u64));
            for (name, policy) in [
                ("serial-antithetic", CpuExecutionPolicy::Scalar),
                ("parallel-antithetic", CpuExecutionPolicy::Parallel),
                ("auto-antithetic", CpuExecutionPolicy::Auto),
            ] {
                let engine = GenericMonteCarloEngine::new(requested_paths, 42)
                    .with_antithetic(true)
                    .with_execution_policy(policy);
                group.bench_with_input(
                    BenchmarkId::new(format!("{threads}t-{name}-steps1"), requested_paths),
                    &requested_paths,
                    |b, _| {
                        b.iter(|| {
                            black_box(pool.install(|| {
                                engine.run(
                                    &generator,
                                    |path| (path[path.len() - 1] - 100.0).max(0.0),
                                    (-0.05_f64).exp(),
                                )
                            }))
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_mc_path_rayon_crossover(_: &mut Criterion) {}

fn bench_adi_heston_pde(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::european_call(100.0, 1.0);
    let model = Heston {
        mu: 0.05,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.3,
        rho: -0.7,
        v0: 0.04,
    };
    // Moderate grid: 50 time steps on a 100 x 50 (S, v) mesh.
    let engine = AdiHestonEngine::new(model, 50, 100, 50).with_scheme(AdiScheme::CraigSneyd);

    let mut group = c.benchmark_group("adi_heston_pde");
    group.sample_size(20);
    group.bench_function("craig_sneyd_50x100x50", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });
    group.finish();
}

fn bench_lsm_american_put(c: &mut Criterion) {
    let market = benchmark_market();
    let option = VanillaOption::american_put(100.0, 1.0);
    let engine = LongstaffSchwartzEngine::new(20_000, 50, 42);

    let mut group = c.benchmark_group("lsm_american_put");
    group.sample_size(10);
    group.bench_function("paths_20k_steps_50", |b| {
        b.iter(|| {
            let px = engine
                .price(black_box(&option), black_box(&market))
                .expect("pricing should succeed")
                .price;
            black_box(px)
        })
    });
    group.finish();
}

fn benchmark_autocallable() -> Autocallable {
    Autocallable {
        underlyings: vec![0, 1],
        notional: 100.0,
        autocall_dates: vec![0.5, 1.0, 1.5, 2.0],
        autocall_barrier: 1.0,
        coupon_rate: 0.08,
        ki_barrier: 0.6,
        ki_strike: 1.0,
        maturity: 2.0,
    }
}

fn bench_autocallable_price_vs_greeks(c: &mut Criterion) {
    let autocall = benchmark_autocallable();
    let spots = [100.0, 95.0];
    let vols = [0.25, 0.30];
    let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let (n_paths, n_steps) = (10_000usize, 100usize);

    let mut group = c.benchmark_group("autocallable");
    group.sample_size(10);

    // Price-only path: no bump-and-reprice ladder.
    group.bench_function("price_only", |b| {
        b.iter(|| {
            let result = price_autocallable(
                black_box(&autocall),
                black_box(&spots),
                black_box(&vols),
                black_box(&corr),
                0.03,
                0.0,
                n_paths,
                n_steps,
            );
            black_box(result.price)
        })
    });

    // Bump-and-reprice Greeks ladder: roughly 9x the price-only cost.
    group.bench_function("price_with_greeks", |b| {
        b.iter(|| {
            let result = price_autocallable_with_greeks(
                black_box(&autocall),
                black_box(&spots),
                black_box(&vols),
                black_box(&corr),
                0.03,
                0.0,
                n_paths,
                n_steps,
            );
            black_box((result.price, result.greeks))
        })
    });

    group.finish();
}

fn benchmark_discount_curve() -> YieldCurve {
    // Flat 2% continuously compounded zero curve on 6 nodes.
    let tenors: Vec<(f64, f64)> = [0.5_f64, 1.0, 2.0, 3.0, 5.0, 10.0]
        .iter()
        .map(|&t| (t, (-0.02 * t).exp()))
        .collect();
    YieldCurve::new(tenors)
}

fn bench_cds_bootstrap(c: &mut Criterion) {
    let discount_curve = benchmark_discount_curve();
    let spreads = [
        (1.0, 0.008),
        (3.0, 0.011),
        (5.0, 0.014),
        (7.0, 0.016),
        (10.0, 0.018),
    ];

    c.bench_function("cds_bootstrap_5_pillars", |b| {
        b.iter(|| {
            let curve = bootstrap_survival_curve_from_cds_spreads(
                black_box(&spreads),
                0.40,
                4,
                black_box(&discount_curve),
            );
            black_box(curve.survival_prob(10.0))
        })
    });
}

fn bench_heston_lm_calibration_one_iter(c: &mut Criterion) {
    // Small single-expiry quote set with LM capped at one iteration: this
    // bounds the bench to the residual + finite-difference Jacobian cost
    // (the dominant per-iteration work) without a full calibration run.
    let cal = HestonCalibrator {
        lm_options: LmOptions {
            max_iterations: 1,
            max_stagnation: 1,
            ..LmOptions::default()
        },
        ..HestonCalibrator::default()
    };

    let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
    let vols = [0.25, 0.23, 0.215, 0.205, 0.20, 0.205, 0.22];
    let quotes: Vec<OptionVolQuote> = strikes
        .iter()
        .zip(vols.iter())
        .enumerate()
        .map(|(i, (&k, &v))| OptionVolQuote::new(format!("Q{i}"), k, 1.0, v))
        .collect();

    let mut group = c.benchmark_group("heston_lm_calibration");
    group.sample_size(10);
    group.bench_function("one_iteration_7_quotes", |b| {
        b.iter(|| {
            let result = cal
                .calibrate(black_box(&quotes))
                .expect("bounded calibration should succeed");
            black_box(result.objective)
        })
    });
    group.finish();
}

/// Deterministic pseudo-random price series (xorshift; no external RNG dep).
fn benchmark_price_series(n: usize) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut prices = Vec::with_capacity(n);
    let mut p = 100.0_f64;
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Uniform in [-1, 1), scaled to ~1% daily moves.
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        p *= 1.0 + 0.01 * (2.0 * u - 1.0);
        prices.push(p);
    }
    prices
}

fn bench_rolling_historical_var(c: &mut Criterion) {
    // 5001 prices -> 5000 returns; 250-day window -> 4750 forecasts.
    let prices = benchmark_price_series(5_001);

    let mut group = c.benchmark_group("rolling_historical_var");
    group.sample_size(20);
    group.bench_function("returns_5k_window_250", |b| {
        b.iter(|| {
            let forecasts = rolling_historical_var_from_prices(black_box(&prices), 250, 0.99, true);
            black_box(forecasts)
        })
    });
    group.finish();
}

fn bench_gamma_ladder(c: &mut Criterion) {
    // 10-pillar curve; pricer values a 20-cashflow annuity off the curve.
    let tenors: Vec<(f64, f64)> = (1..=10)
        .map(|i| {
            let t = i as f64;
            (t, (-0.02 * t).exp())
        })
        .collect();
    let curve = YieldCurve::new(tenors);
    let cashflow_times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();

    c.bench_function("gamma_ladder_10_pillars", |b| {
        b.iter(|| {
            let ladder = gamma_ladder(black_box(&curve), CurveBumpConfig::default(), |c| {
                cashflow_times
                    .iter()
                    .map(|&t| 2.5 * c.discount_factor(t))
                    .sum::<f64>()
            });
            black_box(ladder)
        })
    });
}

criterion_group!(
    hot_path_benches,
    bench_vol_surface_query,
    bench_black_scholes_price_greeks,
    bench_mc_european_50k,
    bench_mc_exact_terminal_auto_crossover,
    bench_mc_exact_terminal_accuracy_tier,
    bench_mc_exact_terminal_rayon_crossover,
    bench_mc_path_rayon_crossover,
    bench_adi_heston_pde,
    bench_lsm_american_put,
    bench_autocallable_price_vs_greeks,
    bench_cds_bootstrap,
    bench_heston_lm_calibration_one_iter,
    bench_rolling_historical_var,
    bench_gamma_ladder
);
criterion_main!(hot_path_benches);
