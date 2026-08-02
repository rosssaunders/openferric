# OpenFerric Agent Guide

Project-level instructions for coding agents working in this repository. Keep this file factual, shared, and tool-neutral.

## Scope

- `AGENTS.md` is the canonical agent guide for this repository.
- Keep Claude-specific behavior in `CLAUDE.md`; keep shared project instructions here.
- Prefer hard cutovers. Do not add backward-compatibility shims unless the user explicitly asks for them.

## Repo Overview

- Core crate: `openferric` in the repository root.
- Workspace crates: `wasm/`, `python/`, `lsp/`, `mcp/`.
- TypeScript workspaces: `excel-addin/`, `vscode-ext/` managed from the root `package.json`.
- Reference source: `vendor/QuantLib/` is a citation-only git submodule. Tests
  use committed reference literals and never build, import, or read the
  submodule at runtime.

## Toolchain

- Rust edition: 2024.
- Pinned Rust toolchain: `1.94.0` via `rust-toolchain.toml`.
- Core crate type: `rlib`.
- Binding crates produce `cdylib`.

## Common Commands

### Rust

```bash
cargo build
cargo build --release
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features
cargo test --locked --workspace --features accelerated-native
cargo test test_name
cargo test --test quantlib_reference
cargo bench
cargo bench --bench pricing_bench
```

`quantlib_reference` is one of roughly 15 QuantLib-derived integration suites,
not a complete QuantLib validation run; use the workspace test command for the
full set.

### Coverage

```bash
make install-coverage
make coverage-test
make coverage-bench
make coverage-bench-parallel
make coverage-all
make coverage-lcov
make coverage-clean
```

Coverage reports are written to `target/llvm-cov/html/index.html`.

### Python

```bash
maturin build --release -m python/Cargo.toml
pip install target/wheels/*.whl
pytest python/tests -v
```

### WASM

```bash
./scripts/build-wasm.sh baseline
./scripts/build-wasm.sh simd
./scripts/build-wasm.sh threads
wasm-pack test --node wasm --locked
```

### TypeScript

```bash
npm run lint -- excel-addin/src vscode-ext/src
npm run typecheck
```

## CI Baseline

Current CI in `.github/workflows/ci.yml` runs:

1. `cargo fmt --all --check`
2. `cargo clippy --locked --workspace --all-targets --all-features`
3. `cargo test --locked --workspace --features accelerated-native`
4. ARM64 NEON SIMD regression tests on `ubuntu-24.04-arm`
5. Python wheel build plus `pytest python/tests -v`
6. Portable, SIMD128, and threaded WASM builds plus Node ABI tests
7. WebGPU WASM compile check
8. Portable WASM binary size budget check for `www/pkg/openferric_wasm_bg.wasm`

Prefer validating the narrowest relevant subset locally, but make sure changes are consistent with the CI commands above before opening a PR.

## Feature Flags

- `parallel`: Rayon-parallel Monte Carlo.
- `simd`: AVX2, AVX-512, NEON, and opt-in WASM SIMD128 paths.
- `accelerated-native`: enables both `parallel` and `simd`.
- `gpu`: WebGPU support.
- `jit`: Cranelift-backed DSL JIT.

When changing feature-gated code, check the corresponding bindings, docs, and tests.

## Architecture

Most pricing flows follow:

```text
Instrument + Market + Engine -> PricingResult
```

Key contracts:

- `core::Instrument`
- `core::PricingEngine<I>`
- `market::Market`
- `core::PricingResult`

Universal entry point:

```rust
PricingEngine::price(&self, instrument: &I, market: &Market) -> Result<PricingResult, PricingError>
```

## Repo Map

- `src/core/`: core traits, result types, diagnostics, domain enums.
- `src/instruments/`: instrument definitions and validation.
- `src/engines/`: analytic, Monte Carlo, PDE, tree, FFT, LSM, GPU engines.
- `src/market/`: market data model and builders.
- `src/models/`: stochastic and rate models.
- `src/vol/`: volatility surfaces, arbitrage checks, calibration.
- `src/rates/`, `src/credit/`, `src/risk/`: rates, credit, and risk analytics.
- `src/dsl/`: DSL parser, compiler, evaluator, and optional JIT.
- `wasm/`: wasm-bindgen bindings.
- `python/`: PyO3 bindings.
- `lsp/`: language server for `.of` DSL files.
- `mcp/`: MCP server binary.
- `excel-addin/`, `vscode-ext/`: TypeScript clients and tooling.

## Project Conventions

- Time is represented in year fractions as `f64`.
- Rates are continuously compounded unless a module explicitly states otherwise.
- Hot paths use `#[inline]` and `#[inline(always)]` aggressively; keep performance-sensitive changes tight.
- `too_many_arguments` and `type_complexity` are intentionally allowed at the workspace level. Do not spend time “fixing” those by default.
- Prefer `DiagKey` enum variants on hot paths, for example `diagnostics.insert_key(DiagKey::Vol, value)`.
- If a new diagnostic key is needed, add it to the `DiagKey` enum and keep its string conversions in sync.
- `Market::builder()` and instrument constructors such as `VanillaOption::european_call(...)` are the standard API shape used across docs, tests, and bindings.

## Testing Notes

- Many integration tests are validated against external references rather than self-generated values.
- QuantLib-derived values are generated offline and committed as literals with
  provenance. Tests do not require `vendor/QuantLib/`, QuantLib libraries, or
  Python packages to be present.
- Keep tests close to the code when adding unit coverage; integration coverage lives in `tests/`.
- Do not hardcode repository-wide test counts in docs unless you are updating them deliberately.

## Change Checklist

- Update the most relevant tests for behavioral changes.
- Update bindings when public Rust APIs used by `python/`, `wasm/`, `lsp/`, or `mcp/` change.
- Update docs when adding modules, feature flags, or build steps.
- Keep instructions here concise; if this file starts growing into deep procedures, split those into narrower docs instead of expanding this guide indefinitely.
