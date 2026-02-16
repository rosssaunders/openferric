# OpenFerric — Rust Quantitative Finance Library

[![Rust](https://img.shields.io/badge/Rust-Edition%202024-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

OpenFerric is a high-performance quantitative finance library in Rust, designed around composable instrument and engine traits.
It targets production-style pricing workflows for derivatives, fixed income, credit, and volatility modeling.

## Why OpenFerric

- Trait-based architecture: `Instrument` + `PricingEngine` gives explicit model/instrument pairings.
- Broad product coverage: vanilla, barrier, digital, Asian, spread, rainbow, FX, variance, rates, and credit.
- Multi-engine stack: analytic, numerical, tree, PDE, and Monte Carlo.
- Python interoperability: optional `pyo3` bindings for integration into research stacks.
- Performance-oriented implementation with predictable memory behavior in hot paths.

## Feature Coverage by Hull Chapters

| Hull chapter (topic) | Coverage in OpenFerric | Key modules |
|---|---|---|
| Ch. 12-13 (binomial trees / numerical methods) | Implemented | `engines::numerical`, `engines::tree`, `pricing::american` |
| Ch. 14 (Black-Scholes-Merton) | Implemented | `engines::analytic::black_scholes`, `pricing::european` |
| Ch. 17-18 (FX and currency options) | Implemented | `instruments::fx`, `engines::analytic::fx` |
| Ch. 19 (Greeks) | Implemented (analytic + MC support) | `core::Greeks`, `greeks`, `engines::monte_carlo::mc_greeks` |
| Ch. 20 (volatility smiles/surfaces) | Implemented | `vol::surface`, `vol::implied`, `vol::sabr`, `vol::local_vol` |
| Ch. 21 (Monte Carlo valuation) | Implemented | `engines::monte_carlo`, `mc` |
| Ch. 25-26 (credit derivatives) | Implemented | `credit::cds`, `credit::survival_curve`, `credit::bootstrap`, `credit::copula` |
| Ch. 7 (swaps and rates products) | Implemented | `rates::swap`, `rates::yield_curve`, `rates::ois`, `rates::inflation` |
| Ch. 26+ (exotics: barrier, Asian, digital, rainbow) | Implemented | `instruments::*`, `engines::analytic::*`, `pricing::*` |

## Quick Start

### 1) European Call (Black-Scholes)

```rust
use openferric::core::PricingEngine;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

let market = Market::builder()
    .spot(100.0)
    .rate(0.05)
    .dividend_yield(0.0)
    .flat_vol(0.20)
    .build()?;

let option = VanillaOption::european_call(100.0, 1.0);
let result = BlackScholesEngine::new().price(&option, &market)?;
println!("price = {:.6}", result.price);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 2) American Put (CRR Binomial)

```rust
use openferric::core::PricingEngine;
use openferric::engines::numerical::AmericanBinomialEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

let market = Market::builder()
    .spot(100.0)
    .rate(0.03)
    .dividend_yield(0.0)
    .flat_vol(0.25)
    .build()?;

let option = VanillaOption::american_put(100.0, 1.0);
let result = AmericanBinomialEngine::new(500).price(&option, &market)?;
println!("price = {:.6}", result.price);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 3) CDS (Fair Spread / NPV)

```rust
use openferric::credit::{Cds, SurvivalCurve};
use openferric::rates::YieldCurve;

let discount_curve = YieldCurve::new(vec![(1.0, 0.97), (3.0, 0.91), (5.0, 0.85)]);
let survival_curve = SurvivalCurve::new(vec![(1.0, 0.98), (3.0, 0.93), (5.0, 0.88)]);

let cds = Cds {
    notional: 10_000_000.0,
    spread: 0.0100,
    maturity: 5.0,
    recovery_rate: 0.40,
    payment_freq: 4,
};

let npv = cds.npv(&discount_curve, &survival_curve);
let fair = cds.fair_spread(&discount_curve, &survival_curve);
println!("npv = {:.2}, fair_spread = {:.6}", npv, fair);
```

### 4) IRS (Par Rate / DV01)

```rust
use chrono::NaiveDate;
use openferric::rates::{
    DayCountConvention, Frequency, InterestRateSwap, YieldCurve,
};

let curve = YieldCurve::new(vec![(1.0, 0.97), (2.0, 0.94), (5.0, 0.86), (10.0, 0.74)]);

let swap = InterestRateSwap::builder()
    .notional(100_000_000.0)
    .fixed_rate(0.035)
    .start_date(NaiveDate::from_ymd_opt(2025, 1, 1).unwrap())
    .end_date(NaiveDate::from_ymd_opt(2030, 1, 1).unwrap())
    .fixed_freq(Frequency::SemiAnnual)
    .float_freq(Frequency::Quarterly)
    .fixed_day_count(DayCountConvention::Thirty360)
    .float_day_count(DayCountConvention::Act360)
    .build();

let par = swap.par_rate(&curve);
let dv01 = swap.dv01(&curve);
println!("par_rate = {:.6}, dv01 = {:.2}", par, dv01);
```

### 5) Barrier Option (Analytic)

```rust
use openferric::core::PricingEngine;
use openferric::engines::analytic::BarrierAnalyticEngine;
use openferric::instruments::BarrierOption;
use openferric::market::Market;

let market = Market::builder()
    .spot(100.0)
    .rate(0.02)
    .dividend_yield(0.00)
    .flat_vol(0.20)
    .build()?;

let option = BarrierOption::builder()
    .call()
    .strike(100.0)
    .expiry(1.0)
    .up_and_out(120.0)
    .rebate(0.0)
    .build()?;

let result = BarrierAnalyticEngine::new().price(&option, &market)?;
println!("barrier price = {:.6}", result.price);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Architecture

```text
+--------------------+      +-----------------------+      +----------------+      +----------------------+
|    Instruments     | ---> |       Engines         | ---> |     Market     | ---> |       Results        |
| Vanilla/Barrier/...|      | Analytic/Tree/MC/PDE |      | Spot/Rate/Vol  |      | Price/Greeks/Diag    |
+--------------------+      +-----------------------+      +----------------+      +----------------------+
```

Core flow:

1. Define an instrument (`VanillaOption`, `BarrierOption`, `InterestRateSwap`, `Cds`, ...).
2. Select an engine (`BlackScholesEngine`, `AmericanBinomialEngine`, `HestonEngine`, ...).
3. Build a `Market` snapshot (spot, curve/rate assumptions, vol source).
4. Call `engine.price(&instrument, &market)` and consume `PricingResult`.

## Modules

| Module | Description |
|---|---|
| `core` | Traits (`Instrument`, `PricingEngine`) and shared result/error types |
| `instruments` | Equity/FX/exotic instrument definitions and builders |
| `engines` | Analytic, numerical, tree, PDE, LSM, and Monte Carlo engines |
| `market` | Market data container and volatility-source abstraction |
| `rates` | Curves, schedules, swaps, OIS, inflation, FRA, cap/floor, bonds |
| `credit` | Survival curves, CDS analytics, CDO/copula tooling, bootstrapping |
| `vol` | Vol surfaces, SABR, local vol, implied vol routines |
| `models` | Model components (e.g., GBM and short-rate models) |
| `math` | Numerical helpers (`normal_cdf`, distributions, utilities) |
| `python` | Optional Python bindings through `pyo3` (`python` feature flag) |
| `greeks`, `pricing`, `mc` | Compatibility modules and standalone pricing utilities |

## Python Bindings

OpenFerric ships optional Python bindings behind the `python` feature.

```bash
cargo build --release --features python
```

If you want a Python extension module build, a standard `pyo3` workflow is:

```bash
maturin develop --release --features python
```

The bindings expose pricing helpers for Black-Scholes, barrier options, American binomial, Heston, FX, spread options, and CDS metrics.

## Performance

- Hot-path pricing engines are implemented with low-overhead scalar math and tight loops.
- The library prefers stack-local temporaries and avoids unnecessary allocations in critical pricing routines where possible.
- Market/instrument data can be reused across repeated evaluations to reduce setup overhead.
- Monte Carlo path generation supports variance-reduction modes (antithetic/control variate).

Benchmark suite lives in `benches/pricing_bench.rs` (Criterion):

| Benchmark | Target latency |
|---|---|
| Black-Scholes European call | `< 100 ns` |
| Barrier analytic | `< 200 ns` |
| American binomial (500 steps) | `< 1 ms` |

Run benchmarks locally when needed:

```bash
cargo bench --bench pricing_bench
```

## Comparison

| Capability | OpenFerric | QuantLib | RustQuant |
|---|---|---|---|
| Primary language | Rust | C++ | Rust |
| Core API style | Trait-based (`Instrument` + `PricingEngine`) | OO pricing framework | Rust-native modular toolkit |
| Analytic vanilla/barrier/Heston | Yes | Yes | Partial / evolving |
| Rates + credit primitives | Yes | Yes (very broad) | Partial / evolving |
| Python support | Optional `pyo3` module | Mature SWIG/Python bindings | Varies by crate/workflow |
| Focus | Lean, composable quant engine in Rust | Industry-standard broad coverage | Rust quant ecosystem and research |

## Testing

```bash
cargo test -q
```

## References

- John C. Hull, *Options, Futures, and Other Derivatives*.
- Espen Gaarder Haug, *The Complete Guide to Option Pricing Formulas*.
- QuantLib project documentation and source.
- Patrick S. Hagan et al., SABR volatility model papers.
- René Stulz, option and risk-management references.
- Steven Shreve, *Stochastic Calculus for Finance II*.
- Peter Jaeckel, *Monte Carlo Methods in Finance*.
