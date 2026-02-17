# OpenFerric

**High-performance quantitative finance in Rust.** Derivatives pricing, risk analytics, and live market tools — covering the full scope of Hull's *Options, Futures, and Other Derivatives* and beyond.

[![Rust](https://img.shields.io/badge/Rust-nightly-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-316%20passing-brightgreen)](#testing)
[![Lines](https://img.shields.io/badge/lines-36K%2B-blue)](#)

---

## Highlights

- **36,600+ lines** of Rust across 149 source files
- **316 tests** — all passing, validated against QuantLib, Haug, and Alan Lewis reference values
- **Trait-based architecture**: `Instrument` + `PricingEngine` — composable, extensible
- **SIMD-accelerated**: AVX2 vectorized Black-Scholes (69M options/sec)
- **FFT pricing**: Carr-Madan for entire strike grids in O(N log N)
- **Live market tools**: Real-time Deribit vol surface dashboard with SVI calibration
- **Python bindings**: Optional PyO3 module for research integration
- **SQL integration**: Designed as extension for [OpenAssay](https://github.com/rosssaunders/openassay) (`openferric.*` schema)

## Performance

| Benchmark | Throughput |
|---|---|
| Black-Scholes (single) | **88 ns** / 11.3M ops/sec |
| Black-Scholes SIMD batch | **69M options/sec** (1.8x scalar) |
| Normal CDF (SIMD) | **283M evals/sec** (2.1x scalar) |
| Barrier analytic | **168 ns** / 6M ops/sec |
| Heston semi-analytic | **3.9 µs** / 256K ops/sec |
| American binomial (500 steps) | **1.9 ms** |
| Monte Carlo 100K paths | **406 ms** |
| Vol surface recalibration | **< 50 ms** (20 expiry slices) |

```bash
cargo bench
```

## Coverage

### Equity Derivatives
| Model/Product | Module | Reference |
|---|---|---|
| Black-Scholes-Merton | `engines::analytic::black_scholes` | Hull Ch 13 |
| Greeks (Δ, Γ, V, Θ, ρ, vanna, volga) | `greeks` | Hull Ch 19 |
| American options (CRR binomial) | `engines::numerical::american_binomial` | Hull Ch 18 |
| Barrier options (8 types) | `engines::analytic::barrier_analytic` | Hull Ch 15, Haug |
| Asian options (geometric + arithmetic MC) | `engines::analytic::asian_geometric`, `engines::monte_carlo` | Hull Ch 17 |
| Lookback (fixed + floating strike) | `engines::analytic::exotic` | Haug 2007 |
| Digital / binary options | `engines::analytic::digital` | Haug 2007 |
| Double barrier (Ikeda-Kunitomo) | `engines::analytic::double_barrier` | Haug 2007 |
| Rainbow (best/worst of two, Stulz) | `engines::analytic::rainbow` | Stulz 1982 |
| Power options | `engines::analytic::power` | Haug 2007 |
| Compound options | `engines::analytic::exotic` | Hull Ch 23 |
| Chooser options | `engines::analytic::exotic` | Hull Ch 23 |
| Quanto options | `engines::analytic::exotic` | Hull Ch 23 |
| Forward start / cliquet | `instruments::cliquet` | Rubinstein 1991 |
| Variance / volatility swaps | `engines::analytic::variance_swap` | Hull Ch 33 |
| Employee stock options | `instruments::employee_stock_option` | Hull Ch 14 |
| Convertible bonds | `engines::tree::convertible` | Hull Ch 23 |
| Discrete dividend BSM | `pricing::discrete_div` | Escrowed dividend |
| Spread options (Kirk + Margrabe) | `engines::analytic::spread` | Kirk, Margrabe |

### Volatility
| Model | Module | Reference |
|---|---|---|
| Heston stochastic vol | `engines::analytic::heston` | Gatheral formulation |
| SABR (Hagan 2002) | `vol::sabr` | Hagan et al. |
| Local vol (Dupire) | `vol::local_vol` | Dupire 1994 |
| SVI parameterization | `vol::surface` | Gatheral 2004 |
| Vol smile (sticky strike/delta) | `vol::smile` | Hull Ch 10 |
| Vanna-volga method | `vol::smile` | Castagna & Mercurio |
| Mixture of lognormals | `vol::mixture` | |
| Implied vol solver (Newton-Raphson) | `vol::implied` | Jäckel |
| Vol surface builder | `vol::builder` | Market → SVI surface |

### Rates & Fixed Income
| Product | Module | Reference |
|---|---|---|
| Yield curve bootstrapping | `rates::yield_curve` | Hull Ch 4 |
| Bond pricing (dirty/clean, duration, convexity, YTM) | `rates::bond` | Hull Ch 4 |
| Interest rate swaps (NPV, par rate, DV01) | `rates::swap` | Hull Ch 7 |
| FRAs | `rates::fra` | Hull Ch 4 |
| Caps / floors | `rates::capfloor` | Hull Ch 9 |
| Swaptions (Black) | `rates::swaption` | Hull Ch 9 |
| Cross-currency swaps | `rates::xccy_swap` | Hull Ch 24 |
| OIS / basis swaps | `rates::ois` | Hull Ch 24 |
| Inflation swaps (ZC + YoY) | `rates::inflation` | Hull Ch 24 |
| Futures pricing | `rates::futures` | Hull Ch 3 |
| Convexity / timing / quanto adjustments | `rates::adjustments` | Hull Ch 34 |
| Day count conventions (ACT/360, ACT/365, 30/360, ACT/ACT) | `rates::day_count` | |

### FX
| Product | Module | Reference |
|---|---|---|
| Garman-Kohlhagen | `engines::analytic::fx` | Hull Ch 26 |
| FX Greeks (domestic + foreign rho) | `engines::analytic::fx` | |
| Black-76 (futures options) | `engines::analytic::black76` | Hull Ch 18 |
| Bachelier / normal model | `engines::analytic::bachelier` | |

### Credit
| Model | Module | Reference |
|---|---|---|
| CDS pricing (NPV, fair spread) | `credit::cds` | Hull Ch 20 |
| Survival curves | `credit::survival_curve` | |
| Hazard rate bootstrap | `credit::bootstrap` | |
| ISDA standard model | `credit::isda` | |
| CDS index pricing | `credit::cds_index` | |
| Nth-to-default (Gaussian copula) | `credit::cds_index` | Hull Ch 20 |
| CDO tranche pricing (LHP) | `credit::cdo` | Hull Ch 20 |
| Copula simulation | `credit::copula` | |

### Risk
| Measure | Module | Reference |
|---|---|---|
| Historical VaR | `risk::var` | Hull Ch 22 |
| Parametric / delta-normal VaR | `risk::var` | |
| Cornish-Fisher VaR | `risk::var` | |
| Expected Shortfall (CVaR) | `risk::var` | |
| CVA / DVA | `risk::xva` | Hull Ch 24 |
| Portfolio Greeks aggregation | `risk::portfolio` | |
| Scenario analysis | `risk::portfolio` | |

### Numerical Engines
| Engine | Module | Notes |
|---|---|---|
| Analytic (closed-form) | `engines::analytic` | 15+ engines |
| CRR binomial tree | `engines::numerical` | Up to 1000 steps |
| Trinomial tree | `engines::tree::trinomial` | European + American |
| Bermudan swaption tree | `engines::tree::bermudan_swaption` | Early exercise |
| Crank-Nicolson PDE | `engines::pde::crank_nicolson` | European + American |
| Longstaff-Schwartz LSM | `engines::lsm` | American MC |
| Monte Carlo (GBM, Heston) | `engines::monte_carlo` | Antithetic + control variate |
| MC Greeks (pathwise + likelihood ratio) | `engines::monte_carlo::mc_greeks` | |
| SIMD Monte Carlo | `engines::monte_carlo::mc_simd` | AVX2 vectorized GBM |
| Parallel Monte Carlo (Rayon) | `engines::monte_carlo::mc_parallel` | Behind `parallel` feature |
| FFT Carr-Madan | `engines::fft::carr_madan` | O(N log N) strike grid |
| Fractional FFT | `engines::fft::frft` | Non-uniform strikes |
| Swing option (DP tree) | `engines::tree::swing` | Energy derivatives |
| Convertible bond tree | `engines::tree::convertible` | Call/put provisions |

### Stochastic Models
| Model | Module | Reference |
|---|---|---|
| GBM | `models` | |
| Heston | `models` | Gatheral |
| SABR | `models` | Hagan 2002 |
| Hull-White (1-factor) | `models::short_rate` | Hull Ch 31 |
| Vasicek | `models::short_rate` | Hull Ch 31 |
| Cox-Ingersoll-Ross | `models::short_rate` | Hull Ch 31 |
| HW calibration (swaption vols) | `models::hw_calibration` | |
| HJM (single + multi-factor) | `models::hjm` | Hull Ch 32/35 |
| LIBOR Market Model (BGM) | `models::lmm` | Hull Ch 35 |
| Schwartz (commodity) | `models::commodity` | Hull Ch 27 |
| Variance Gamma | `models::variance_gamma` | Madan et al. |

### Other
| Feature | Module | Reference |
|---|---|---|
| Energy / commodity derivatives | `instruments::commodity`, `models::commodity` | Hull Ch 27 |
| Weather derivatives (HDD/CDD) | `instruments::weather` | Hull Ch 29 |
| Catastrophe bonds | `instruments::weather` | Hull Ch 29 |
| Real options (defer/expand/abandon) | `instruments::real_option`, `pricing::real_option` | Hull Ch 28 |
| FFT characteristic functions (BS, Heston, VG, CGMY) | `engines::fft::char_fn` | Carr-Madan 1999 |
| Fast normal CDF (Hart) | `math::fast_norm` | |
| BSM inverse CDF | `math::fast_norm` | Beasley-Springer-Moro |
| Bivariate normal CDF | `math` | |
| Cubic spline interpolation | `math` | |

### Live Market Tools
| Tool | Binary | Description |
|---|---|---|
| Deribit vol surface snapshot | `deribit_vol_surface` | REST fetch → SVI calibration → 3D Plotly HTML |
| Live vol surface dashboard | `vol_dashboard` | WebSocket stream → real-time recalibration → browser dashboard |

```bash
# Snapshot
cargo run --features deribit --bin deribit_vol_surface --release

# Live dashboard (http://localhost:3000)
cargo run --features deribit --bin vol_dashboard --release
```

## Quick Start

### European Call (Black-Scholes)

```rust
use openferric::core::PricingEngine;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

let market = Market::builder()
    .spot(100.0).rate(0.05).dividend_yield(0.0).flat_vol(0.20)
    .build()?;

let option = VanillaOption::european_call(100.0, 1.0);
let result = BlackScholesEngine::new().price(&option, &market)?;
println!("price = {:.4}, delta = {:.4}", result.price, result.greeks.delta);
```

### Heston via FFT (4096 strikes at once)

```rust
use openferric::engines::fft::carr_madan::heston_price_fft;

let prices = heston_price_fft(
    100.0,           // spot
    &strike_grid,    // Vec<f64>
    0.03,            // rate
    0.0,             // dividend
    0.04, 1.5, 0.04, 0.3, -0.7,  // v0, kappa, theta, sigma_v, rho
    1.0,             // maturity
)?;
// prices: Vec<(f64, f64)> — (strike, call_price) for all 4096 strikes
```

### CDS Fair Spread

```rust
use openferric::credit::{Cds, SurvivalCurve};
use openferric::rates::YieldCurve;

let cds = Cds { notional: 10e6, spread: 0.01, maturity: 5.0, recovery_rate: 0.40, payment_freq: 4 };
let fair = cds.fair_spread(&discount_curve, &survival_curve);
```

## Architecture

```
Instruments ─→ Engines ─→ Market ─→ PricingResult
                 │
    ┌────────────┼────────────┐
    │            │            │
 Analytic    Tree/PDE     Monte Carlo
 (closed)   (numerical)   (simulation)
    │            │            │
    └────────────┼────────────┘
                 │
            FFT / SIMD
          (acceleration)
```

## Feature Flags

| Feature | Description |
|---|---|
| `python` | PyO3 bindings for Python integration |
| `parallel` | Rayon-parallelized Monte Carlo |
| `deribit` | Live market binaries (reqwest, tokio, axum) |

## Testing

Test vectors are externally validated — not self-generated. The `tests/quantlib_reference.rs` suite includes:

- **43 European option cases** from Haug's *Option Pricing Formulas* (pag. 2-8, 24, 27)
- **10 Heston model cases** from Alan Lewis's 12-digit reference prices ([wilmott.com](http://wilmott.com/messageview.cfm?catid=34&threadid=90957))
- **4 Heston cached regression cases** from QuantLib's `hestonmodel.cpp`
- **72 barrier option cases** from Haug pag. 72 (all 4 barrier types × calls/puts × 2 vol levels)

QuantLib's C++ test suite is included as a git submodule at `vendor/QuantLib/` for reference.

```bash
cargo test                              # all 316 tests
cargo test --test quantlib_reference    # reference suite only
cargo test --features parallel          # include parallel MC tests
cargo bench                             # Criterion benchmarks
```

## References

### Textbooks
- Hull, J.C. — *Options, Futures, and Other Derivatives* (11th ed.)
- Haug, E.G. — *The Complete Guide to Option Pricing Formulas* (2nd ed.)
- Gatheral, J. — *The Volatility Surface: A Practitioner's Guide*
- Jäckel, P. — *Monte Carlo Methods in Finance*
- Shreve, S. — *Stochastic Calculus for Finance II: Continuous-Time Models*
- Glasserman, P. — *Monte Carlo Methods in Financial Engineering*

### Papers
- Carr, P. & Madan, D. — *Option Valuation Using the FFT* (1999)
- Hagan, P. et al. — *Managing Smile Risk* (Wilmott, 2002)
- Dupire, B. — *Pricing with a Smile* (Risk, 1994)
- Stulz, R. — *Options on the Maximum or Minimum of Two Risky Assets* (1982)
- Madan, D., Carr, P. & Chang, E. — *The Variance Gamma Process and Option Pricing* (1998)
- Carr, P., Geman, H., Madan, D. & Yor, M. — *The Fine Structure of Asset Returns: An Empirical Investigation* (CGMY, 2002)
- Barndorff-Nielsen, O. — *Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling* (NIG, 1997)
- Hagan, P. — *Convexity Conundrums: Pricing CMS Swaps, Caps, and Floors* (Wilmott, 2003)
- Lewis, A. — *A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes* (2001)
- Ikeda, M. & Kunitomo, N. — *Pricing Options with Curved Boundaries* (1992)
- Rubinstein, M. — *Pay Now, Choose Later* (1991)
- Castagna, A. & Mercurio, F. — *The Vanna-Volga Method for Implied Volatilities* (Risk, 2007)

## Acknowledgements

- **[QuantLib](https://www.quantlib.org/)** — Reference test vectors extracted from the QuantLib C++ test suite (BSD-3-Clause license). QuantLib is included as a git submodule at `vendor/QuantLib/` for reproducibility.
- **[Alan Lewis](http://wilmott.com/)** — High-precision Heston model reference prices used for validation.
- **E.G. Haug** — Extensive option pricing test cases from *The Complete Guide to Option Pricing Formulas*.
- **J.C. Hull** — Structural coverage follows *Options, Futures, and Other Derivatives*.

## License

MIT
