use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use openferric::core::PricingEngine;
use openferric::engines::analytic::HestonEngine;
use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, carr_madan_fft, carr_madan_fft_strikes, heston_price_fft,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::pricing::OptionType;
use openferric::pricing::european::black_scholes_price;
use std::hint::black_box;

fn log_spaced_strikes(min_k: f64, max_k: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![min_k];
    }
    let log_min = min_k.ln();
    let log_step = (max_k.ln() - log_min) / (n as f64 - 1.0);
    (0..n)
        .map(|i| (log_min + i as f64 * log_step).exp())
        .collect()
}

fn bench_heston_gauss_laguerre_vs_fft(c: &mut Criterion) {
    let spot = 100.0;
    let rate = 0.02;
    let q = 0.0;
    let maturity = 1.0;

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .flat_vol(0.2)
        .build()
        .expect("benchmark market should be valid");

    let heston = HestonEngine::new(0.04, 1.8, 0.04, 0.5, -0.7);
    let single_strikes: Vec<f64> = (0..100).map(|i| 60.0 + i as f64 * 0.8).collect();
    let fft_strikes = log_spaced_strikes(20.0, 500.0, 4096);

    let mut group = c.benchmark_group("heston_gauss_vs_fft");

    group.bench_function("gauss_laguerre_100_single_strikes", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for k in &single_strikes {
                let option = VanillaOption::european_call(*k, maturity);
                sum += heston
                    .price(black_box(&option), black_box(&market))
                    .expect("heston price")
                    .price;
            }
            black_box(sum)
        })
    });

    group.bench_function("fft_4096_strikes", |b| {
        b.iter(|| {
            let out = heston_price_fft(
                spot,
                black_box(&fft_strikes),
                rate,
                q,
                0.04,
                1.8,
                0.04,
                0.5,
                -0.7,
                maturity,
            );
            black_box(out[2048].1)
        })
    });

    group.finish();
}

fn bench_bs_fft_vs_analytic(c: &mut Criterion) {
    let spot = 100.0;
    let rate = 0.03;
    let maturity = 1.0;
    let vol = 0.2;

    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);
    let strikes: Vec<f64> = (0..10).map(|i| 80.0 + i as f64 * 5.0).collect();

    let fft_prices = carr_madan_fft_strikes(
        &cf,
        rate,
        maturity,
        spot,
        &strikes,
        CarrMadanParams::default(),
    )
    .expect("fft bs prices");

    for (k, fft_px) in &fft_prices {
        let bs = black_scholes_price(OptionType::Call, spot, *k, rate, vol, maturity);
        assert!(
            (fft_px - bs).abs() < 1e-4,
            "BS FFT mismatch at K={k}: fft={fft_px} bs={bs}"
        );
    }

    c.bench_function("bs_fft_analytic_consistency", |b| {
        b.iter(|| {
            let out = carr_madan_fft(
                black_box(&cf),
                rate,
                maturity,
                spot,
                CarrMadanParams::default(),
            )
            .expect("fft bs slice");
            black_box(out[2000].1)
        })
    });
}

fn bench_carr_madan_scaling(c: &mut Criterion) {
    let spot = 100.0;
    let rate = 0.02;
    let maturity = 1.0;
    let vol = 0.25;
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);

    let mut group = c.benchmark_group("carr_madan_scaling");

    for n in [1024_usize, 4096, 8192] {
        let params = CarrMadanParams {
            n,
            eta: 0.25,
            alpha: 1.5,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let out = carr_madan_fft(black_box(&cf), rate, maturity, spot, params)
                    .expect("fft slice");
                black_box(out[n / 2].1)
            })
        });
    }

    group.finish();
}

criterion_group!(
    fft_benches,
    bench_heston_gauss_laguerre_vs_fft,
    bench_bs_fft_vs_analytic,
    bench_carr_madan_scaling
);
criterion_main!(fft_benches);
