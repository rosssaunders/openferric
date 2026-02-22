use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use openferric::math::{ExtrapolationMode, Interpolator, LogLinearInterpolator};
use openferric::rates::{
    YieldCurve, YieldCurveInterpolationMethod, YieldCurveInterpolationSettings,
};
use std::hint::black_box;

// Target guideline:
// - interpolation latency < 100ns per point for core methods on target hardware.

fn sample_curve_points() -> Vec<(f64, f64)> {
    vec![
        (0.25, 0.995),
        (0.50, 0.989),
        (1.0, 0.978),
        (2.0, 0.955),
        (3.0, 0.932),
        (5.0, 0.886),
        (7.0, 0.844),
        (10.0, 0.786),
        (20.0, 0.620),
        (30.0, 0.510),
    ]
}

fn bench_log_linear_df(c: &mut Criterion) {
    let points = sample_curve_points();
    let x: Vec<f64> = points.iter().map(|(t, _)| *t).collect();
    let y: Vec<f64> = points.iter().map(|(_, df)| *df).collect();
    let itp = LogLinearInterpolator::new(x, y, ExtrapolationMode::Linear).unwrap();

    let queries: Vec<f64> = (1..=10_000).map(|i| 30.0 * i as f64 / 10_000.0).collect();
    let mut group = c.benchmark_group("interp_log_linear_df");
    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("value", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for t in &queries {
                acc += itp.value(black_box(*t)).unwrap();
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_yield_curve_methods(c: &mut Criterion) {
    let points = sample_curve_points();
    let queries: Vec<f64> = (1..=10_000).map(|i| 30.0 * i as f64 / 10_000.0).collect();

    let methods = [
        YieldCurveInterpolationMethod::LogLinearDiscount,
        YieldCurveInterpolationMethod::MonotoneConvex,
        YieldCurveInterpolationMethod::HermiteMonotone,
        YieldCurveInterpolationMethod::LogCubicMonotone,
    ];

    let mut group = c.benchmark_group("yield_curve_interpolation");
    group.throughput(Throughput::Elements(queries.len() as u64));

    for method in methods {
        let settings = YieldCurveInterpolationSettings {
            method,
            extrapolation: ExtrapolationMode::Linear,
        };
        let curve = YieldCurve::new_with_settings(points.clone(), settings).unwrap();
        group.bench_function(format!("{:?}", method), |b| {
            b.iter(|| {
                let mut acc = 0.0;
                for t in &queries {
                    acc += curve.discount_factor(black_box(*t));
                }
                black_box(acc)
            })
        });
    }

    group.finish();
}

criterion_group!(
    interpolation_benches,
    bench_log_linear_df,
    bench_yield_curve_methods
);
criterion_main!(interpolation_benches);
