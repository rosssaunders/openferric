# OpenFerric

**High-performance quantitative finance in Rust.** Derivatives pricing, risk analytics, and live market tools.

[![Rust](https://img.shields.io/badge/Rust-nightly-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-366%20passing-brightgreen)](#testing)
[![Coverage](https://codecov.io/gh/rosssaunders/openferric/graph/badge.svg)](https://codecov.io/gh/rosssaunders/openferric)
[![Lines](https://img.shields.io/badge/lines-40K%2B-blue)](#)

---

## Highlights

- **40,300+ lines** of Rust across 160 source files
- **366 tests** — validated against QuantLib, Haug, Alan Lewis, and Fabozzi reference values
- **Trait-based architecture**: `Instrument` + `PricingEngine` — composable, extensible
- **SIMD-accelerated**: AVX2 vectorized Black-Scholes (69M options/sec)
- **FFT pricing**: Carr-Madan for entire strike grids in O(N log N)
- **Live market tools**: Deribit vol surface snapshot tooling plus rich browser dashboard (WASM/GitHub Pages)
- **Python bindings**: Optional PyO3 module for research integration
- **SQL integration**: Designed as extension for [OpenAssay](https://github.com/rosssaunders/openassay)

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

## Quick Start

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

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for more — Heston FFT, CDS pricing, vol surface calibration.

## Coverage

Full coverage details in [docs/COVERAGE.md](docs/COVERAGE.md). Summary:

**Equity & FX** — Black-Scholes, barriers, Asians, lookbacks, digitals, double barriers, rainbows, power options, compound/chooser/quanto, cliquets, variance swaps, convertibles, spread options, employee stock options

**Volatility** — Heston, SABR, local vol (Dupire), SVI, Andreasen-Huge (arb-free), Fengler, vanna-volga, mixture of lognormals, Jäckel implied vol

**Rates & Fixed Income** — Yield curves, bonds, swaps, FRAs, caps/floors, swaptions, cross-currency, OIS, inflation, CMS, multi-curve OIS framework

**Credit** — CDS, survival curves, hazard rate bootstrap, ISDA model, CDS index, nth-to-default, CDO tranches, CDS options

**Structured Products** — TARFs, range accruals, autocallables, MBS pass-throughs (PSA/CPR), IO/PO strips

**Risk** — VaR, Expected Shortfall, CVA/DVA/FVA/MVA/KVA, wrong-way risk, SA-CCR, portfolio Greeks

**Stochastic Models** — GBM, Heston, SABR, Hull-White, Vasicek, CIR, HJM, LMM/BGM, Variance Gamma, CGMY, NIG, rBergomi, stochastic local vol

**Numerical Engines** — Analytic (15+), binomial/trinomial trees, explicit/implicit/Crank-Nicolson/Hopscotch FD, Longstaff-Schwartz, Monte Carlo (antithetic, control variate, SIMD, parallel), FFT (Carr-Madan, FRFT), generalized & two-asset trees

**Live Market** — Deribit vol surface snapshot (REST + SVI calibration + 3D Plotly) and browser dashboard

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
| `deribit` | Deribit snapshot binary (`deribit_vol_surface`) using reqwest + tokio |

## Testing

Test vectors are externally validated — not self-generated. The reference suite includes:

- **43 European option cases** from Haug's *Option Pricing Formulas*
- **10 Heston model cases** from Alan Lewis's 12-digit precision reference prices
- **72 barrier option cases** from Haug
- **4 Heston cached regression cases** from QuantLib

QuantLib's C++ test suite is included as a git submodule at `vendor/QuantLib/` for reproducibility.

```bash
cargo test                              # all 366 tests
cargo test --test quantlib_reference    # reference suite only
cargo test --features parallel          # include parallel MC tests
cargo llvm-cov --workspace --features parallel --summary-only  # local coverage metric
cargo bench                             # Criterion benchmarks
```

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for the full list. Key sources:

- Hull — *Options, Futures, and Other Derivatives*
- Haug — *The Complete Guide to Option Pricing Formulas*
- Gatheral — *The Volatility Surface*
- Carr & Madan — *Option Valuation Using the FFT*
- Hagan et al. — *Managing Smile Risk* (SABR)

## Acknowledgements

- **[QuantLib](https://www.quantlib.org/)** — Reference test vectors (BSD-3-Clause). Included as submodule at `vendor/QuantLib/`.
- **Alan Lewis** — High-precision Heston reference prices.
- **E.G. Haug** — Extensive option pricing test cases.
- **F.J. Fabozzi** — MBS/ABS reference values.

## License

MIT
