# Full Derivatives Pricing in 700KB of WASM

*How we built a complete quantitative finance library in Rust that runs in your browser.*

---

You can price a European option in a spreadsheet. You can price an exotic autocallable in QuantLib. But can you price both — plus barriers, Heston stochastic vol, FFT strike grids, credit default swaps, and a custom DSL for structured products — in **700KB of WebAssembly** that loads in your browser?

That's [OpenFerric](https://github.com/rosssaunders/openferric).

## The Numbers

| Metric | Value |
|--------|-------|
| Rust source | 72,000+ lines across 201 files |
| Tests | 501 (validated against QuantLib, Haug, Lewis, Fabozzi) |
| WASM binary | **719KB** (272KB gzipped) |
| Black-Scholes throughput | 11.3M options/sec (88ns each) |
| SIMD batch pricing | 69M options/sec (AVX2) |
| Monte Carlo 50K paths × 252 steps | 365ms |
| Vol surface calibration | < 50ms for 20 expiry slices |

That last row matters. The [live Vol Terminal](https://rosssaunders.github.io/openferric/) calibrates Deribit's entire options surface in real-time, in your browser tab, using the same WASM binary.

## Why Rust → WASM?

The derivatives pricing world is dominated by C++ (QuantLib) and Python (QuantLib-Python, or bespoke numpy). Both have problems for modern deployment:

**C++** gives you performance but not portability. Compiling QuantLib for a browser? Good luck. And distributing native binaries to end users means dealing with platform-specific builds, system dependencies, and ABI compatibility.

**Python** gives you reach but not performance. A Monte Carlo in numpy is 10-100x slower than optimised native code. You can use Cython or numba, but then you're writing two languages and still can't ship to a browser.

**Rust → WASM** gives you both:
- Near-native performance (the WASM JIT compiles to real machine instructions)
- Universal deployment (any browser, any OS, no installation)
- Memory safety without GC pauses
- `no_std` core means the same code runs on servers, in browsers, and on embedded devices

## Architecture

OpenFerric follows a trait-based design borrowed from QuantLib's architecture but idiomatic to Rust:

```rust
pub trait Instrument {
    fn payoff(&self, spot: f64) -> f64;
}

pub trait PricingEngine<I: Instrument> {
    fn price(&self, instrument: &I, market: &Market) -> Result<PricingResult>;
}
```

Engines and instruments are independent. A `BlackScholesEngine` can price any `VanillaOption`. A `MonteCarloEngine` can price anything that implements `Instrument`. This composability is what lets us pack so many models into a small binary — shared infrastructure, no duplication.

### What's Inside

**Analytic engines:** Black-Scholes-Merton, Bachelier, Black-76, Bjerksund-Stensland (American), Barone-Adesi-Whaley, barrier options (all 8 types), binary/digital options with full Greeks.

**Semi-analytic:** Heston model via Gauss-Laguerre quadrature, Carr-Madan FFT for pricing entire strike grids in O(N log N).

**Monte Carlo:** Multi-asset correlated GBM with Cholesky decomposition, variance reduction, path-dependent payoffs. SIMD-accelerated with AVX2 batch evaluation.

**Fixed income & credit:** CDS pricing with hazard rates, yield curve bootstrapping, survival curves, bond analytics.

**Structured Product DSL:** A text-based language for defining exotic products without writing Rust:

```text
product "Worst-of Autocallable"
    notional: 1_000_000
    maturity: 3.0
    underlyings
        SPX = asset(0)
        EUROSTOXX = asset(1)
    schedule quarterly from 0.25 to 3.0
        let worst = min(perf(SPX), perf(EUROSTOXX))
        if worst >= 1.0 then
            redeem notional * 1.05
        if worst < 0.6 then
            redeem notional * worst
```

This gets compiled to bytecode and executed inside the Monte Carlo engine. The bytecode uses packed 4-byte instructions with a raw f64 stack — no enum dispatch overhead, no allocations on the hot path.

## The WASM Compilation

Getting from Rust to a 700KB WASM binary requires a few tricks:

**1. Feature gates.** The WASM crate (`wasm/`) only exposes what the browser needs. Server-only functionality (file I/O, Python bindings, GPU compute) is gated behind features.

**2. wasm-bindgen.** We use `wasm-bindgen` to generate JavaScript bindings. The API surface is deliberately small — a handful of functions that accept JSON configs and return JSON results. All the complexity stays in Rust.

**3. No allocator bloat.** Rust's default allocator adds surprisingly little to WASM. The bulk of the binary is actual pricing logic, not runtime overhead.

**4. LTO + opt-level.** Link-time optimisation across the entire crate graph, `opt-level = "z"` for size, and `wasm-opt` post-processing shave another 20-30%.

```toml
[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
```

## Performance: WASM vs Native

WASM runs at roughly 60-80% of native speed for numerical workloads. Here's what we measured:

| Benchmark | Native (x86-64) | WASM (V8) | Ratio |
|-----------|-----------------|-----------|-------|
| Black-Scholes single | 88ns | ~140ns | 0.63x |
| MC autocall 20K paths | 59ms | ~95ms | 0.62x |
| Vol surface calibration | 48ms | ~75ms | 0.64x |

That 60-65% ratio is consistent and predictable. For interactive applications (the Vol Terminal recalibrates on every tick), it's more than enough. Users don't notice 75ms vs 48ms.

The real win is **deployment cost**: zero. No servers, no containers, no cold starts. The user's browser does the compute. For a SaaS product, that's the difference between paying for GPU instances and paying for a CDN.

## Live Demo: The Vol Terminal

The [Ferric Terminal](https://rosssaunders.github.io/openferric/) is a single-page app that:

1. Connects to Deribit's WebSocket API for real-time options data
2. Calibrates a volatility surface using the WASM pricing engine
3. Renders 3D surface plots, skew charts, term structure, and Greeks
4. Updates in real-time as new ticks arrive

Everything runs client-side. The "server" is GitHub Pages serving static files. The WASM binary handles all the quantitative heavy lifting — surface fitting, interpolation, Greek computation, scenario analysis.

## What's Next

We're working towards:
- **crates.io + npm** — `cargo add openferric` and `npm install @openferric/wasm`
- **Cloud API** — OpenFerric Pro for server-side pricing at scale
- **Live vol surfaces as a data product** — calibrated surfaces via API
- **Enterprise licensing** — on-prem deployment for regulated firms

## Try It

```bash
# Clone and run the Vol Terminal locally
git clone https://github.com/rosssaunders/openferric
cd openferric
wasm-pack build --target web --features wasm
cp -r pkg www/pkg
python3 -m http.server 3001 --directory www
# Open http://localhost:3001
```

Or just visit the [live demo](https://rosssaunders.github.io/openferric/).

The entire codebase is MIT-licensed. Star the repo, file issues, send PRs. We're building the quant library that should have existed ten years ago — fast, portable, and open.

---

*[OpenFerric](https://github.com/rosssaunders/openferric) — high-performance quantitative finance in Rust.*
