use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::dsl::{AssetMarketData, DslMonteCarloEngine, MultiAssetMarket, parse_and_compile};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use openferric::dsl::{JitProductEvaluator, eval::evaluate_product};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

fn load_example(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("examples").join("dsl").join(name);
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn single_asset_market() -> MultiAssetMarket {
    MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02)
}

fn three_asset_market() -> MultiAssetMarket {
    MultiAssetMarket {
        assets: vec![
            AssetMarketData::Equity {
                spot: 100.0,
                vol: 0.20,
                dividend_yield: 0.02,
            },
            AssetMarketData::Equity {
                spot: 100.0,
                vol: 0.22,
                dividend_yield: 0.03,
            },
            AssetMarketData::Equity {
                spot: 100.0,
                vol: 0.25,
                dividend_yield: 0.01,
            },
        ],
        correlation: vec![
            vec![1.0, 0.7, 0.5],
            vec![0.7, 1.0, 0.6],
            vec![0.5, 0.6, 1.0],
        ],
        rate: 0.03,
    }
}

fn bench_dsl_parse_compile(c: &mut Criterion) {
    let zero_coupon = load_example("01_zero_coupon.of");
    let autocall = load_example("06_autocallable_worst_of.of");
    let mut group = c.benchmark_group("dsl_parse_compile");

    for (name, source) in [
        ("zero_coupon", zero_coupon.as_str()),
        ("worst_of_autocall", autocall.as_str()),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let product =
                    parse_and_compile(black_box(source)).expect("DSL parse/compile should succeed");
                black_box(product.schedules.len())
            })
        });
    }

    group.finish();
}

fn bench_dsl_price(c: &mut Criterion) {
    let zero_coupon =
        parse_and_compile(&load_example("01_zero_coupon.of")).expect("zero coupon should compile");
    let autocall = parse_and_compile(&load_example("06_autocallable_worst_of.of"))
        .expect("autocall should compile");
    let single_market = single_asset_market();
    let basket_market = three_asset_market();
    let mut group = c.benchmark_group("dsl_price");
    group.sample_size(10);

    for (name, product, market, paths, steps) in [
        (
            "zero_coupon",
            &zero_coupon,
            &single_market,
            20_000usize,
            100usize,
        ),
        ("autocall", &autocall, &basket_market, 20_000usize, 100usize),
        ("autocall", &autocall, &basket_market, 50_000usize, 252usize),
    ] {
        let engine = DslMonteCarloEngine::new(paths, steps, 42);
        let case = format!("{paths}p_{steps}s");
        group.bench_with_input(BenchmarkId::new(name, case), &engine, |b, engine| {
            b.iter(|| {
                let result = engine
                    .price_multi_asset(black_box(product), black_box(market))
                    .expect("DSL pricing should succeed");
                black_box(result.price)
            })
        });
    }

    group.finish();
}

fn bench_dsl_jit(c: &mut Criterion) {
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    {
        let product = parse_and_compile(&load_example("06_autocallable_worst_of.of"))
            .expect("autocall should compile");
        let num_steps = 252;
        let rate = 0.03;
        let initial_spots = vec![100.0; product.num_underlyings];
        let path = vec![initial_spots.clone(); num_steps + 1];
        let evaluator =
            JitProductEvaluator::compile(&product, num_steps, rate).expect("JIT should compile");
        let mut scratch = evaluator.new_scratch();
        let mut group = c.benchmark_group("dsl_jit");

        group.bench_function("compile_execution_plan", |b| {
            b.iter(|| {
                black_box(
                    JitProductEvaluator::compile(
                        black_box(&product),
                        black_box(num_steps),
                        black_box(rate),
                    )
                    .expect("JIT should compile"),
                )
            })
        });
        group.bench_function("warm_path_evaluation", |b| {
            b.iter(|| {
                black_box(
                    evaluator
                        .evaluate_path_with_scratch(
                            black_box(&path),
                            black_box(&initial_spots),
                            &mut scratch,
                        )
                        .expect("JIT path evaluation should succeed"),
                )
            })
        });
        group.bench_function("interpreter_plan_and_path_evaluation", |b| {
            b.iter(|| {
                black_box(
                    evaluate_product(
                        black_box(&product),
                        black_box(&path),
                        black_box(&initial_spots),
                        black_box(num_steps),
                        black_box(rate),
                    )
                    .expect("interpreter path evaluation should succeed"),
                )
            })
        });

        group.finish();
    }

    #[cfg(not(all(feature = "jit", not(target_family = "wasm"))))]
    let _ = c;
}

criterion_group!(
    dsl_benches,
    bench_dsl_parse_compile,
    bench_dsl_price,
    bench_dsl_jit
);
criterion_main!(dsl_benches);
