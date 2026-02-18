# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                                # debug build
cargo build --release                      # optimized build (LTO, stripped)
cargo test                                 # all tests (~370 lib + integration)
cargo test test_name                       # run a single test by name
cargo test --test quantlib_reference       # run one integration test file
cargo test --features parallel             # include parallel MC tests
cargo clippy                               # lint (expect ~17 too_many_arguments warnings, intentionally suppressed)
cargo bench                                # Criterion benchmarks (pricing, SIMD, FFT, parallel)
cargo bench --bench pricing_bench          # single benchmark suite
```

**Feature flags:** `parallel` (rayon MC), `simd` (AVX2/NEON), `python` (PyO3 bindings), `wasm` (wasm-bindgen), `gpu` (wgpu WebGPU), `deribit` (live market tools)

**WASM build:** `wasm-pack build --target web --features wasm`

**Python wheel:** `pip install .` (requires `python` feature)

## Architecture

### Core Pricing Flow

Every pricing operation follows: **Instrument + Market + Engine → PricingResult**

```
core::Instrument (trait)     →  instruments/*  (27+ types: vanilla, barrier, Asian, etc.)
core::PricingEngine<I> (trait) →  engines/*     (analytic, MC, PDE, FFT, LSM, GPU)
market::Market               →  spot + rate + dividend_yield + VolSource
core::PricingResult          →  price + stderr + Greeks + Diagnostics
```

`PricingEngine::price(&self, instrument: &I, market: &Market) -> Result<PricingResult, PricingError>` is the universal entry point. Engines are generic over instrument type.

### Module Map

| Module | Purpose |
|---|---|
| `core/` | Traits (`Instrument`, `PricingEngine`), `Greeks`, `PricingResult`, `PricingError`, `Diagnostics` (fixed 8-entry inline array), `DiagKey` enum |
| `core/types.rs` | Domain enums: `OptionType`, `ExerciseStyle`, `BarrierDirection`, `BarrierStyle`, `Averaging`, `StrikeType` |
| `market/` | `Market` (builder pattern), `VolSource` (Flat or Surface), `VolSurface` trait |
| `instruments/` | Concrete instrument structs, each implementing `Instrument` trait. Includes `validate()` methods |
| `engines/analytic/` | Closed-form pricers: Black-Scholes, Heston semi-analytic, barrier, digital, exotic, rainbow, spread, SIMD batch |
| `engines/monte_carlo/` | MC engine with variance reduction, QMC (Sobol), SIMD SoA layout, parallel (rayon) |
| `engines/pde/` | Finite difference: implicit (backward Euler), Crank-Nicolson with tridiagonal solver |
| `engines/tree/` | Binomial (CRR), trinomial, generalized, two-asset, swing |
| `engines/fft/` | Carr-Madan FFT for Lévy processes (Heston, VG, CGMY, NIG) |
| `engines/lsm/` | Longstaff-Schwartz American option pricing |
| `engines/gpu/` | WebGPU compute shader MC (feature-gated) |
| `models/` | Stochastic processes: GBM, Heston, SABR, Hull-White, Vasicek, CIR, rough Bergomi, HJM, LMM, Schwartz commodity |
| `vol/` | Vol surface models: SVI, SABR calibration, local vol (Dupire), Andreasen-Huge, Fengler, vanna-volga, Jäckel implied vol |
| `rates/` | Yield curves, swaps, swaptions, bonds, FRA, cap/floor, OIS, CMS, inflation, multi-curve framework |
| `credit/` | CDS, survival curves, CDO tranches, Gaussian copula, ISDA conventions |
| `risk/` | VaR/ES, XVA (CVA/DVA/FVA/MVA/KVA), SA-CCR, portfolio aggregation |
| `math/` | Fast normal CDF/PDF, SIMD math, Sobol QMC, fast RNG (Xoshiro256++, PCG64), bump allocator |
| `python/` | PyO3 bindings (feature-gated), 55+ exported functions |
| `wasm/` | wasm-bindgen bindings (feature-gated), 40+ exported functions |
| `greeks/`, `mc/`, `pricing/` | Legacy modules kept for backward compatibility |

### Key Patterns

- **Market builder:** `Market::builder().spot(100.0).rate(0.05).flat_vol(0.20).build().unwrap()`
- **Instrument constructors:** `VanillaOption::european_call(strike, expiry)`, `VanillaOption::american_put(strike, expiry)`
- **Diagnostics:** Use `DiagKey` enum variants (not raw strings) on hot paths via `diagnostics.insert_key(DiagKey::Vol, val)`. String-based `diagnostics.insert("vol", val)` parses to `DiagKey` and panics on unknown keys — add new keys to the `DiagKey` enum + both `as_str()` and `FromStr` impls.
- **VolSurface trait:** Implement `VolSurface + Clone + Debug + Send + Sync` for custom surfaces. The `VolSurfaceClone` helper enables `Box<dyn VolSurface>` cloning.
- **Arbitrage checking:** Vol surfaces can detect butterfly (negative density) and calendar (decreasing variance) violations via `ArbitrageViolation` enum.

### Testing

Integration tests in `tests/` are validated against QuantLib (vendored as git submodule at `vendor/QuantLib/`). Key test files:
- `european_quantlib.rs` — 43 European cases from Haug's formulas
- `heston_quantlib.rs` — 10 high-precision Heston cases from Alan Lewis
- `barrier_quantlib.rs` — 72 barrier option test cases
- `quantlib_reference.rs` — cross-engine reference suite

Most unit tests are colocated in `#[cfg(test)] mod tests` within each source file.

## Conventions

- Rust 2024 edition. Crate types: `rlib` + `cdylib`.
- Release profile: LTO fat, 1 codegen-unit, opt-level 3, panic=abort, stripped symbols.
- Time is in year fractions (f64). Rates are continuously compounded.
- `#[inline]` and `#[inline(always)]` are used extensively on hot-path math functions.
- `too_many_arguments` clippy warnings are intentionally accepted (pricing functions naturally have many parameters).
- SIMD code lives behind `#[cfg(feature = "simd")]` with separate modules for AVX2 (`simd_math.rs`) and NEON (`simd_neon.rs`).
