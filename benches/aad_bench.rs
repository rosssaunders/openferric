use criterion::{Criterion, criterion_group, criterion_main};
use openferric::core::PricingEngine;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;
use openferric::math::Tape;
use std::hint::black_box;

fn bench_bs_aad_single(c: &mut Criterion) {
    let engine = BlackScholesEngine::new();
    let option = VanillaOption::european_call(100.0, 1.0);
    let market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.0)
        .flat_vol(0.2)
        .build()
        .expect("benchmark market should be valid");

    c.bench_function("aad_bs_single_trade", |b| {
        b.iter(|| {
            let out = engine
                .price_with_greeks_aad(black_box(&option), black_box(&market))
                .expect("aad pricing should succeed");
            black_box(out.price)
        })
    });
}

fn bench_portfolio_aad_reverse_gradient(c: &mut Criterion) {
    const N_TRADES: usize = 10_000;
    const N_FACTORS: usize = 500;
    const FACTORS_PER_TRADE: usize = 6;

    let mut exposures = vec![[(0usize, 0.0f64); FACTORS_PER_TRADE]; N_TRADES];
    for (t, row) in exposures.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            let idx = (t * 29 + j * 71) % N_FACTORS;
            let w = 0.01 + (1 + ((t + j * 17) % 31)) as f64 * 1e-3;
            *entry = (idx, w);
        }
    }

    c.bench_function("aad_portfolio_10k_trades_500_factors", |b| {
        b.iter(|| {
            let mut tape = Tape::with_capacity(N_TRADES * 32 + N_FACTORS * 4);
            let factors: Vec<_> = (0..N_FACTORS)
                .map(|i| tape.input(1.0 + i as f64 * 1e-3))
                .collect();

            let mut portfolio = tape.constant(0.0);
            for row in &exposures {
                let mut trade_underlying = tape.constant(0.0);
                for &(idx, weight) in row {
                    let term = tape.mul_const(factors[idx], weight);
                    trade_underlying = tape.add(trade_underlying, term);
                }
                let intrinsic = tape.sub_const(trade_underlying, 1.0);
                let payoff = tape.positive_part(intrinsic);
                portfolio = tape.add(portfolio, payoff);
            }

            let gradient = tape.gradient(portfolio, &factors);
            black_box((tape.value(portfolio), gradient[0], gradient[N_FACTORS - 1]))
        })
    });
}

criterion_group!(
    aad_benches,
    bench_bs_aad_single,
    bench_portfolio_aad_reverse_gradient
);
criterion_main!(aad_benches);
