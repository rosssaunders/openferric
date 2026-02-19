use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use openferric::core::PricingEngine;
use openferric::engines::analytic::{HestonEngine, BlackScholesEngine};
use openferric::engines::fft::{
    BlackScholesCharFn, CarrMadanParams, carr_madan_fft, carr_madan_fft_complex,
    carr_madan_fft_strikes, heston_price_fft,
};
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::core::OptionType;
use openferric::pricing::european::black_scholes_price;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
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

    for n in [1024_usize, 4096, 8192, 16384] {
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

fn bench_carr_madan_4096_complex_vs_dispatch(c: &mut Criterion) {
    let spot = 100.0;
    let rate = 0.02;
    let maturity = 1.0;
    let vol = 0.20;
    let params = CarrMadanParams {
        n: 4096,
        eta: 0.25,
        alpha: 1.5,
    };
    let cf = BlackScholesCharFn::new(spot, rate, 0.0, vol, maturity);

    let mut group = c.benchmark_group("carr_madan_4096_complex_vs_dispatch");
    group.bench_function("complex_only", |b| {
        b.iter(|| {
            let out = carr_madan_fft_complex(black_box(&cf), rate, maturity, spot, params)
                .expect("fft slice");
            black_box(out[params.n / 2].1)
        })
    });
    group.bench_function("dispatch_auto", |b| {
        b.iter(|| {
            let out =
                carr_madan_fft(black_box(&cf), rate, maturity, spot, params).expect("fft slice");
            black_box(out[params.n / 2].1)
        })
    });
    group.finish();
}

fn bench_fft_4096_rustfft_vs_realfft(c: &mut Criterion) {
    let n = 4096_usize;
    let real_input: Vec<f64> = (0..n)
        .map(|i| (0.03 * i as f64).sin() + 0.5 * (0.07 * i as f64).cos())
        .collect();

    let mut rust_planner = FftPlanner::<f64>::new();
    let rust_forward = rust_planner.plan_fft_forward(n);
    let rust_scratch_len = rust_forward.get_inplace_scratch_len();
    let mut rust_scratch = vec![Complex::new(0.0, 0.0); rust_scratch_len];

    let mut real_planner = RealFftPlanner::<f64>::new();
    let real_forward = real_planner.plan_fft_forward(n);
    let real_scratch_len = real_forward.get_scratch_len();
    let mut real_scratch = vec![Complex::new(0.0, 0.0); real_scratch_len];
    let mut real_output = real_forward.make_output_vec();

    let mut group = c.benchmark_group("fft_4096_rustfft_vs_realfft");
    group.bench_function("rustfft_complex", |b| {
        b.iter(|| {
            let mut input: Vec<Complex<f64>> = real_input
                .iter()
                .copied()
                .map(|x| Complex::new(x, 0.0))
                .collect();
            rust_forward.process_with_scratch(&mut input, &mut rust_scratch);
            black_box(input[n / 2].re)
        })
    });
    group.bench_function("realfft_r2c", |b| {
        b.iter(|| {
            let mut input = real_input.clone();
            real_forward
                .process_with_scratch(&mut input, &mut real_output, &mut real_scratch)
                .expect("real FFT process");
            black_box(real_output[n / 4].re)
        })
    });
    group.finish();
}

fn bench_heston_fft_different_strikes(c: &mut Criterion) {
    let mut group = c.benchmark_group("heston_fft_strikes");
    
    let spot = 100.0;
    let rate = 0.05;
    let dividend = 0.0;
    let maturity = 1.0;
    let v0 = 0.04;
    let kappa = 2.0;
    let theta = 0.04;
    let sigma_v = 0.5;
    let rho = -0.7;
    
    let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
    
    for strike in strikes.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(strike), strike, |b, _| {
            b.iter(|| {
                let strike_array = [*strike];
                let prices = heston_price_fft(
                    spot,
                    &strike_array,
                    rate,
                    dividend,
                    v0,
                    kappa,
                    theta,
                    sigma_v,
                    rho,
                    maturity,
                );
                black_box(prices)
            })
        });
    }
    
    group.finish();
}

fn bench_heston_fft_vs_bs_analytic(c: &mut Criterion) {
    let mut group = c.benchmark_group("heston_vs_bs");
    
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let dividend = 0.0;
    let maturity = 1.0;
    let v0 = 0.04;
    let kappa = 2.0;
    let theta = 0.04;
    let sigma_v = 0.3;
    let rho = -0.5;
    
    group.bench_function("heston_fft", |b| {
        b.iter(|| {
            let strike_array = [strike];
            let prices = heston_price_fft(
                spot,
                &strike_array,
                rate,
                dividend,
                v0,
                kappa,
                theta,
                sigma_v,
                rho,
                maturity,
            );
            black_box(prices)
        })
    });
    
    // Black-Scholes analytic for comparison
    let option = VanillaOption::european_call(strike, maturity);
    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend)
        .flat_vol(theta.sqrt()) // Use theta as vol for comparison
        .build()
        .expect("valid market");
    let bs_engine = BlackScholesEngine::new();
    
    group.bench_function("bs_analytic", |b| {
        b.iter(|| {
            let price = bs_engine
                .price(&option, &market)
                .expect("pricing should succeed")
                .price;
            black_box(price)
        })
    });
    
    group.finish();
}

fn bench_heston_fft_parameter_regimes(c: &mut Criterion) {
    let mut group = c.benchmark_group("heston_fft_regimes");
    
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let dividend = 0.0;
    let maturity = 1.0;
    
    // Low volatility regime
    group.bench_function("low_vol_regime", |b| {
        b.iter(|| {
            let strike_array = [strike];
            let prices = heston_price_fft(
                spot,
                &strike_array,
                rate,
                dividend,
                0.01,  // v0: low initial variance
                3.0,   // kappa: fast mean reversion
                0.01,  // theta: low long-term variance
                0.1,   // sigma_v: low vol-of-vol
                -0.3,  // rho: mild negative correlation
                maturity,
            );
            black_box(prices)
        })
    });
    
    // High volatility regime
    group.bench_function("high_vol_regime", |b| {
        b.iter(|| {
            let strike_array = [strike];
            let prices = heston_price_fft(
                spot,
                &strike_array,
                rate,
                dividend,
                0.16,  // v0: high initial variance
                0.5,   // kappa: slow mean reversion
                0.16,  // theta: high long-term variance
                0.8,   // sigma_v: high vol-of-vol
                -0.8,  // rho: strong negative correlation
                maturity,
            );
            black_box(prices)
        })
    });
    
    // Standard regime
    group.bench_function("standard_regime", |b| {
        b.iter(|| {
            let strike_array = [strike];
            let prices = heston_price_fft(
                spot,
                &strike_array,
                rate,
                dividend,
                0.04,  // v0
                2.0,   // kappa
                0.04,  // theta
                0.5,   // sigma_v
                -0.7,  // rho
                maturity,
            );
            black_box(prices)
        })
    });
    
    group.finish();
}

criterion_group!(
    fft_benches,
    bench_heston_gauss_laguerre_vs_fft,
    bench_bs_fft_vs_analytic,
    bench_carr_madan_scaling,
    bench_carr_madan_4096_complex_vs_dispatch,
    bench_fft_4096_rustfft_vs_realfft,
    bench_heston_fft_different_strikes,
    bench_heston_fft_vs_bs_analytic,
    bench_heston_fft_parameter_regimes
);
criterion_main!(fft_benches);
