# OpenFerric

**High-performance quantitative finance in Rust.** Derivatives pricing, funding and rates analytics, credit, risk, structured products, and tooling built around a shared Rust core.

[![Crates.io](https://img.shields.io/crates/v/openferric.svg)](https://crates.io/crates/openferric)
[![docs.rs](https://docs.rs/openferric/badge.svg)](https://docs.rs/openferric)
[![Rust](https://img.shields.io/badge/Rust-stable-blue?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/rosssaunders/openferric/graph/badge.svg)](https://codecov.io/gh/rosssaunders/openferric)

OpenFerric is a Rust library first, but this repository also ships Python bindings, WebAssembly bindings, an Excel add-in, a VS Code extension and LSP for the OpenFerric DSL, and an MCP server for tool-driven integrations.

## Highlights

- **Trait-based pricing core** built around `Instrument + Market + PricingEngine -> PricingResult`.
- **Broad product coverage** across equity, FX, rates, credit, volatility, structured products, and portfolio risk.
- **Funding-rate analytics** for Boros/Pendle-style swaps, including multi-venue curves, rolling stats, liquidation and margin simulation, and piecewise-linear or piecewise-constant interpolation.
- **Structured product DSL** with `.of` examples, a VS Code pricing dashboard, and an LSP-backed editing experience.
- **Multiple delivery surfaces** from one Rust core: crate, Python package, WebAssembly, Excel add-in, and MCP server.
- **Performance-oriented execution** with SIMD, parallel Monte Carlo, optional WebGPU, and optional Cranelift JIT support.
- **Reference-validated tests** against QuantLib, Haug, Alan Lewis, Fabozzi, and other external sources.

## Demo and Examples

- Example hosted demo: [openferric.netlify.app](https://openferric.netlify.app/) (example only, not the primary supported distribution surface)
- Rust examples: [`examples/`](examples) and [docs/EXAMPLES.md](docs/EXAMPLES.md)
- Python funding walkthrough: [examples/boros_funding_swap.py](examples/boros_funding_swap.py)
- DSL guide: [docs/dsl.md](docs/dsl.md)
- DSL sample products: [`examples/dsl/`](examples/dsl)

## Quick Start

```bash
cargo add openferric
```

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

println!("price = {:.4}", result.price);
println!("delta = {:.4}", result.greeks.delta);
```

## Repo Components

| Component | Purpose |
|---|---|
| `src/` | Core Rust library for pricing, rates, credit, volatility, calibration, models, DSL, and risk |
| `python/` | PyO3 bindings for Python workflows and notebooks |
| `wasm/` | `wasm-bindgen` bindings for browser and WebAssembly consumers |
| `excel-addin/` | Excel add-in powered by the WASM build |
| `vscode-ext/` + `lsp/` | VS Code extension and language server for `.of` DSL files with a pricing dashboard |
| `mcp/` | MCP server exposing OpenFerric functionality as callable tools |

## Coverage

Full coverage notes live in [docs/COVERAGE.md](docs/COVERAGE.md). The current repo covers:

- **Equity and FX**: Black-Scholes, barriers, digitals, Asians, lookbacks, spreads, baskets, rainbows, power options, convertibles, employee stock options, and more.
- **Volatility and calibration**: Heston, SABR, SVI, local vol, Andreasen-Huge, Fengler, vanna-volga, forward variance, mixture models, and implied vol tooling.
- **Rates and funding**: yield curves, bonds, swaps, FRAs, caps/floors, swaptions, OIS, inflation, CMS, cross-currency products, funding-rate curves, and funding-rate swaps.
- **Credit and XVA**: survival curves, CDS, CDS options, CDS indices, nth-to-default baskets, CDO tranches, Gaussian copula, and XVA analytics.
- **Structured products and DSL**: autocallables, phoenix notes, range accruals, TARFs, MBS pass-throughs, IO/PO strips, and a dedicated product DSL.
- **Risk and scenarios**: VaR, Expected Shortfall, SA-CCR, portfolio aggregation, margin simulation, liquidation simulation, and stress workflows.
- **Numerical engines**: analytic pricers, trees, PDEs, Monte Carlo, FFT, SIMD paths, GPU support, and optional JIT-backed DSL execution.

## Performance

Pricing engines support explicit `ExecutionPolicy` selection and an `Auto`
policy that chooses between scalar, SIMD, Rayon, GPU, and prepared JIT
implementations according to workload and target capabilities. The selected
backend, vector width, and thread count are returned in pricing diagnostics.

Criterion results are produced by the scheduled benchmark workflow on a
labelled, stable runner. Each artifact includes the commit, compiler version,
hardware metadata, and Criterion's timing distributions and confidence intervals; benchmark
numbers are intentionally not copied into this README without that provenance.
See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for backend eligibility,
determinism, precision, build variants, and reproducible benchmark commands.

## Feature Flags

| Feature | Description |
|---|---|
| `accelerated-native` | Portable native build with runtime SIMD dispatch and Rayon |
| `parallel` | Rayon-parallel Monte Carlo and parallel-enabled benchmark/test paths |
| `simd` | Runtime-dispatched AVX2, AVX-512, and NEON kernels plus opt-in WASM SIMD128 analytic batches |
| `gpu` | WebGPU exact-terminal Monte Carlo with pooled asynchronous resources |
| `jit` | Native-only Cranelift JIT support for prepared DSL execution |

## Build and Validation

Core Rust workflow:

```bash
cargo build
cargo build --release
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features
cargo test --locked --workspace --features accelerated-native
cargo bench --locked --features accelerated-native
```

TypeScript and Python linting:

```bash
npm ci
npm run typecheck
npm run lint
ruff check python/ examples/
ruff format --check python/ examples/
```

Python package:

```bash
maturin build --locked --release -m python/Cargo.toml
pip install target/wheels/*.whl
pytest python/tests/ -v
```

WASM build:

```bash
./scripts/build-wasm.sh baseline
./scripts/build-wasm.sh simd
./scripts/build-wasm.sh threads
wasm-pack test --node wasm --locked
```

Coverage helpers:

```bash
make install-coverage
make coverage-test
make coverage-bench
make coverage-bench-parallel
make coverage-all
make coverage-lcov
```

## Testing and References

OpenFerric uses externally validated references rather than self-generated assertions wherever practical.

- QuantLib is vendored as a git submodule at [`vendor/QuantLib/`](vendor/QuantLib) for reproducible cross-checks.
- The test suite includes reference cases from Haug, Alan Lewis, Fabozzi, and other published sources.
- Integration tests live in [`tests/`](tests), while module-level unit tests stay close to implementation code.

See [docs/REFERENCES.md](docs/REFERENCES.md) for source material and [docs/EXAMPLES.md](docs/EXAMPLES.md) for end-to-end usage examples.

## Related Surfaces

- [Python bindings README](python/README.md)
- [Excel add-in README](excel-addin/README.md)
- [OpenAssay](https://github.com/rosssaunders/openassay) for SQL-oriented integration work

## License

MIT
