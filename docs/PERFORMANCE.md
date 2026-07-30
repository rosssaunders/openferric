# Performance and hardware execution

OpenFerric keeps the portable scalar implementation available and adds
hardware acceleration behind explicit Cargo features. Acceleration is selected
at runtime where the target permits it; enabling a feature does not guarantee
that every engine or instrument can use that backend.

This document describes the behavior implemented in the current source. The
thresholds are deliberately conservative implementation details rather than
API guarantees. Re-run the benchmarks before changing them.

## Feature flags and build profiles

| Feature | What it enables |
|---|---|
| none | Portable scalar core |
| `parallel` | Rayon-backed CPU parallelism |
| `simd` | Runtime-dispatched x86-64 AVX2/FMA and AVX-512 kernels, AArch64 NEON kernels, and compile-time WASM SIMD128 analytic kernels |
| `accelerated-native` | Convenience alias for `parallel,simd` |
| `gpu` | `wgpu` exact-terminal European Monte Carlo on native WebGPU and through the WASM async wrapper |
| `jit` | Native-only Cranelift compilation for DSL product evaluation |

The normal portable and accelerated builds are:

```bash
cargo build --locked --release -p openferric --no-default-features
cargo build --locked --release -p openferric --features accelerated-native
cargo build --locked --release -p openferric --features accelerated-native,gpu,jit
```

The declared MSRV is Rust 1.92, which is the minimum supported by the current
`wgpu` dependency used by the full hardware feature set. CI checks every core
feature on that compiler and uses the repository-pinned toolchain for normal
development and releases.

Despite its name, `accelerated-native` does not set
`-C target-cpu=native`. The repository has no global native-CPU rustflag:
distributed binaries can therefore use runtime dispatch and still run on
older CPUs of the same target architecture. For a binary that will run only
on the machine that built it, local compiler tuning is opt-in:

```bash
RUSTFLAGS="-C target-cpu=native" \
  cargo build --locked --release -p openferric --features accelerated-native
```

Do not distribute that locally tuned artifact to machines with an unknown CPU
feature set.

The release profile uses optimization level 3, fat LTO, one codegen unit, and
symbol stripping. The benchmark profile uses optimization level 3, thin LTO,
and one codegen unit.

### WebAssembly distributions

The `openferric-wasm` crate keeps its default package portable and exposes
three separately deployable CPU distributions through the build helper:

```bash
./scripts/build-wasm.sh baseline
./scripts/build-wasm.sh simd
./scripts/build-wasm.sh threads
./scripts/build-wasm.sh all
```

- `baseline` uses the stable toolchain and writes `www/pkg`.
- `simd` enables the WASM crate's `simd` feature and the `simd128` target
  feature, writing `www/pkg-simd`.
- `threads` enables Rayon through `wasm-bindgen-rayon`, atomics, bulk memory,
  and shared memory. It writes `www/pkg-threads` and uses the pinned
  `nightly-2025-11-15` toolchain by default because it builds the WASM standard
  library with the required features.
- `all` builds those three independent packages; it does not make one package
  that requires every capability.

The SIMD package exposes uniform-parameter Black-Scholes price and Greek
batches that dispatch to an explicit `f64x2` SIMD128 kernel. WebAssembly has
no vector `ln` or `exp`, so those transcendental operations remain
lane-scalar; division, d1/d2, CDF/PDF polynomial work, discounting, payoff,
and Greek arithmetic are vectorized. Node differential tests assert the
SIMD128 backend is selected and exercise odd tails and degenerate inputs,
while the binary verifier confirms the required SIMD opcodes are present.

At runtime, feature-detect SIMD and threads before selecting a package and
always retain the baseline `pkg` package as the fallback. A threaded package requires
`SharedArrayBuffer` in a cross-origin-isolated page, normally provided with:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

After loading the threaded module, await its exported
`initThreadPool(navigator.hardwareConcurrency)` before invoking work that can
enter Rayon. The default and SIMD-only packages do not require those headers
or worker initialization.

The WASM `gpu` feature is independent of these CPU packages. It exposes the
asynchronous WebGPU Monte Carlo wrapper. CI compile-checks that target; it does
not claim a browser WebGPU runtime test. Cranelift `ExecutionPolicy::Jit` is
native-only; it is not part of any WASM distribution.

## Execution policies and diagnostics

Hardware-aware engines accept an `ExecutionPolicy`:

| Policy | Meaning |
|---|---|
| `Auto` | Select an eligible backend from the compiled features, runtime capabilities, thread count, and workload size |
| `Scalar` | Force the portable scalar implementation |
| `Simd` | Request a single-threaded SIMD implementation |
| `Parallel` | Request Rayon; the parallel kernel may also use SIMD |
| `Gpu` | Request the engine's GPU implementation |
| `Jit` | Request the engine's prepared JIT implementation |

An explicit request through a high-level pricing engine returns an error when
that backend is not compiled, is unsupported by the CPU, or cannot preserve
the instrument's semantics. Policies are engine-specific: for example,
`MonteCarloPricingEngine` implements GPU but not JIT, while
`DslMonteCarloEngine` implements JIT but not GPU. The low-level generic
`mc::MonteCarloEngine` has a tuple-returning API and accepts the narrower
`CpuExecutionPolicy` (`Auto`, `Scalar`, and feature-gated `Parallel`), so it
cannot silently accept SIMD, GPU, or JIT requests that it cannot execute.

`PricingResult::diagnostics` reports what actually ran:

- `execution_backend`: `0` scalar, `1` SIMD, `2` parallel, `3` GPU, or `4` JIT.
- `vector_width`: the number of `f64` lanes used by the selected CPU path.
  A parallel backend can report a width greater than one.
- `num_threads`: the Rayon pool size for parallel CPU work, `1` for
  single-threaded CPU work, and `0` for GPU execution.

Use the diagnostics instead of inferring the backend from enabled Cargo
features.

## Monte Carlo backend selection

`MonteCarloPricingEngine` has a fast exact-terminal path for European vanilla
options without discrete dividends when variance reduction is either `None`
or `Antithetic`. It samples terminal GBM directly, so `num_steps` does not add
simulation work and the result reports one step.

For `ExecutionPolicy::Auto`, backend selection is ordered as follows:

1. With native `gpu` support, an already-prewarmed GPU context, at least
   **1,000,000 paths**, and GPU-eligible inputs, use GPU execution.
2. With `parallel`, more than one Rayon worker, and enough estimated work
   removed from the serial critical path to repay scheduling and pool wake-up,
   use Rayon. Exact-terminal and time-stepped paths use separate fixed-chunk
   cost models described below.
3. For an exact-terminal option with `simd`, an available SIMD kernel, and at
   least one full native vector (**8 AVX-512, 4 AVX2/FMA, or 2 NEON effective
   samples**), use SIMD.
4. Otherwise use scalar execution.

All CPU decisions use the number of estimator samples actually executed:
`num_paths` for ordinary sampling and `ceil(num_paths / 2)` for antithetic
sampling. This prevents `Auto` from reporting SIMD or Rayon when antithetic
pairing leaves only a scalar tail or too little parallel work.

`Auto` never performs cold GPU initialization on a pricing request. Call
`engines::gpu::prewarm_gpu()` during application startup; readiness can be
queried without side effects through `engines::gpu::gpu_is_ready()`. Without a
ready context, `Auto` chooses the best eligible CPU backend. An explicit `Gpu`
request still initializes on demand and reports any failure.

The public exact-terminal CPU path uses fixed **4,096-sample chunks**, one RNG
stream per chunk, reusable normal storage inside SIMD kernels, and an ordered
host reduction. The generic path engine uses fixed **256-sample chunks**, one
RNG per chunk, and worker-local reusable normal and path buffers. Generic
`Auto` estimates the critical-path saving from the actual chunk/tail balance
and current Rayon pool size:

- Exact-terminal execution selects Rayon after it can remove at least **1,024
  effective samples** from the critical path. For both 2-worker and 24-worker
  pools this first occurs at **5,120 effective samples**; a 1-worker pool
  remains serial.
- Generic path execution weights each sample by
  `max(steps, 1) * max(normal_streams, 1)` and includes a bounded larger-pool
  wake-up allowance. A one-step GBM crosses at **512 effective samples** in a
  2-worker pool and **768 effective samples** in the measured 24-worker pool.
  Longer paths and two-normal-stream models can cross with smaller tail chunks
  because each sample performs more work.

These are cost-model calibration points, not public API guarantees. The model
continues to account for non-uniform tails and useful workers above each
boundary rather than comparing requested paths with a single global cutoff.

An explicit SIMD request currently requires the exact-terminal vanilla path.
An explicit parallel request requires the `parallel` feature but bypasses the
automatic workload threshold. An explicit GPU request is narrower than the
CPU exact-terminal path: it requires a European vanilla option, no discrete
dividends, no variance reduction, reproducible streams, the
`Xoshiro256PlusPlus` RNG selection, and a path count that fits in `u32`.

On x86-64 and AArch64, exact-terminal SIMD Monte Carlo defaults to
`AccuracyTier::High`. Path count cannot bound approximation bias when payoff
variance is small or variance reduction is effective. Callers may explicitly
select `AccuracyTier::Fast` after establishing an application-specific error
budget. Fast uses a degree-7 vector exponential with a dense-tested
relative-error bound of `8e-9` over `[-700, 700]`; High uses degree 11 with a
`2e-14` bound. Scalar math continues to use `f64` standard-library functions.

## CPU SIMD and parallel kernels

Runtime dispatch prevents unsupported instructions from being called:

- Batch Black-Scholes pricing, Greeks, and normal CDF choose AVX-512
  (8 `f64` lanes), AVX2/FMA (4 lanes), AArch64 NEON (2 lanes), WASM SIMD128
  (2 lanes in a SIMD-enabled build), or scalar. Native detection is cached
  once per process; WASM selection is fixed by the distribution at compile
  time.
- The allocation-returning batch APIs delegate to `*_into` variants.
  Reuse caller-owned output slices in hot loops to avoid repeated allocations.
- Analytic batches use scalar execution below one full vector and for a short
  partial-vector tail until four vector widths; measured crossover benchmarks
  cover every boundary from 1 through 64 elements. Very small total
  volatility (`sigma*sqrt(T) <= 1e-3`) uses a cancellation-resistant scalar
  CDF-interval formula so time value is not replaced by approximation error.
- Exact-terminal Monte Carlo uses 8-lane AVX-512, 4-lane AVX2/FMA, or 2-lane
  NEON, with scalar tail handling.
- Structure-of-arrays GBM simulation used by vanilla Longstaff-Schwartz
  dispatches to AVX-512 or AVX2/FMA on x86-64, NEON on AArch64, and otherwise
  uses scalar generation. Time-major storage keeps each exercise-date scan
  contiguous.

Rayon acceleration is also used outside the policy-driven engines:

- Vanilla Longstaff-Schwartz parallelizes fixed-chunk regression aggregation
  and independent exercise updates at **8,192 paths** or more when the pool
  has multiple workers.
- The Heston ADI solver parallelizes independent spot rows and variance
  columns when `spot_steps * variance_steps >= 8,192`. Thomas-solver scratch
  arrays and the variance-column staging buffer are allocated once and reused
  across sweeps.

For controlled scaling experiments, install work in a local Rayon pool:

```rust
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()?;
let result = pool.install(|| engine.price(&instrument, &market));
```

This avoids conflating the algorithm with the host's default logical-CPU
count.

The vector exp and log kernels preserve IEEE-754 subnormal results through a
rare-lane scalar repair. This adds one predictable exceptional-case check to
the normal path and avoids silently flushing exp results or mis-scaling
positive subnormal log inputs. Inverse-normal SIMD no longer clamps valid
probabilities below `1e-300`.

### Python hot paths

The Python extension enables `accelerated-native`. Its NumPy `bs_price_batch`
entry point validates contiguous `float64` arrays. Below 4,096 elements it
borrows them under the GIL to avoid a small-input copy. At 4,096 elements and
above it snapshots both arrays into owned Rust buffers, releases the GIL, and
runs the dispatched batch kernel; the returned array is newly allocated. The
measured threshold step was about 1.67 microseconds (4.2%) on the implementation
host. Pure-Rust Monte Carlo methods similarly convert Python objects first and
then release the GIL while Rust owns the simulation loop. Prefer
`McEngine.vanilla_price` for a vanilla terminal payoff because it evaluates the
payoff entirely in Rust. A custom Python payoff passed to `run_gbm` or
`run_heston` re-enters Python for each evaluation, runs in deterministic scalar
callback order, and will usually dominate the simulation cost.

## DSL SIMD, parallel, and JIT behavior

`DslMonteCarloEngine` has independent policy rules because its workload
includes multi-asset path generation and product-program evaluation:

- x86-64 SIMD evaluates 4 paths together when AVX2/FMA is available;
  AArch64 NEON evaluates 2 paths together.
- SIMD is considered only from four complete vectors: **16 paths** on x86-64
  or **8 paths** on AArch64.
- Rayon starts at **8,192 paths** with more than one worker and uses chunks of
  at least **4,096 paths**.
- On a native `jit` build without an eligible SIMD path, `Auto` selects JIT
  from **4,096 paths**. At 8,192 paths or more, a JIT evaluation can also
  distribute path chunks through Rayon when `parallel` is enabled.
- When SIMD is eligible, `Auto` prefers Rayon at 8,192 paths and SIMD below
  that threshold; it does not automatically select scalar JIT.

An explicit `Jit` request is available at any path count on native targets
when the feature is enabled. Cranelift compiles a prepared evaluator for the
product's observation programs. Evaluation reuses a scratch workspace, and
the engine keeps a 16-entry, mutex-protected LRU cache keyed by the compiled
product, step count, and rate. Alternating products therefore do not force
continuous recompilation, while the fixed capacity bounds executable-memory
retention. JIT dependencies and exports are excluded from WASM builds.

## GPU execution and precision

The WebGPU kernel prices exact-terminal GBM European calls and puts. Each
invocation uses both normals from one Box-Muller pair, so it handles two paths.
A 256-invocation workgroup therefore covers up to 512 paths. Workgroup-local
tree reduction writes only `(count, mean, M2)` as three `f32` values per
workgroup; the host does not read back one payoff per path. The host merges
these summaries in fixed workgroup order using Chan's centered-moment
recurrence in `f64`, avoiding cancellation between raw first and second
moments for near-deterministic samples.

Successful native initialization caches the device, queue, compute pipeline,
layout, and parameter buffer. Cold initialization is serialized so concurrent
first callers do not construct duplicate devices. Initialization failures are
not cached, so a later call can retry. Completed output/staging/bind-group sets
return to a bounded two-entry pool for allocation-free steady-state
dispatches. The native API blocks for readback; the WASM WebGPU wrapper is
asynchronous. Concurrent first WASM Promises on one worker share an in-flight
initialization Promise; rejection resets the worker-local state for retry.
Separate WASM workers retain separate contexts because WebGPU handles are not
shared through Rust thread-local state. JavaScript can call
`gpu_mc_prewarm()` during startup and query `gpu_mc_is_ready()` without
starting initialization.

WebGPU's portable arithmetic baseline is `f32`. Path generation, payoff
evaluation, and workgroup reduction are all `f32`. The host converts
workgroup summaries to `f64` for the final centered-moment merge, but that
cannot recover precision already lost on the device.
`GpuMcResult::stderr` is sampling uncertainty only; it does **not** include
floating-point roundoff.

For tight tolerances, risk reports, or validation, compare GPU output with an
`f64` CPU backend. GPU normals come from keyed, counter-based Threefry2x32-20:
the invocation id is the counter and both halves of the public `u64` seed are
the key. The compatibility WASM export accepts a `u32` seed; its `seed64`
variant accepts low and high `u32` halves without requiring JavaScript
`BigInt`. A fixed seed makes a GPU run repeatable in a stable environment, but
the GPU uses a different RNG implementation from the CPU and portable WebGPU
transcendental results need not be bit-identical across adapters or drivers.

The GPU API validates finite `f32` conversion, positive counts, `u32` count
limits, and adapter workgroup/buffer limits. `num_steps` remains in the API and
must be positive, but it does not affect exact-terminal GBM.

## Determinism contract

Seeded CPU parallelism uses fixed chunk boundaries, chunk-derived streams, an
indexed collection, and a fixed-order final reduction. For the same compiled
backend this makes results independent of Rayon scheduling and pool size.
Regression tests exercise this property for generic Monte Carlo,
exact-terminal Monte Carlo, vanilla Longstaff-Schwartz, and ADI line solves.

This does not promise bitwise identity between different backends:

- Scalar and SIMD can use different exponential approximations and reduction
  groupings.
- GPU uses `f32`, a different RNG, and a device-side reduction.
- `with_randomized_streams()` and thread-local RNG mode are intentionally
  non-reproducible.
- Compiler flags, CPU architectures, and math-library implementations can
  change the last bits even when the statistical estimator is equivalent.

Compare different hardware backends with model-appropriate tolerances and
standard errors. Use a fixed seed, fixed feature set, compiler version, and
accuracy tier when exact reproducibility matters.

## Benchmarking and provenance

Criterion benchmarks cover public Monte Carlo policies, scalar/SIMD batch
analytics, RNGs, DSL interpretation and JIT compilation/warm evaluation,
FFT/interpolation routines, LSM and ADI thread scaling, and GPU terminal
pricing. The `mc_exact_terminal_rayon_crossover` and
`mc_path_rayon_crossover` groups exercise scalar, explicit Rayon, and `Auto`
immediately around the calibrated boundaries in 1-, 2-, and host-sized pools,
including ordinary and antithetic sampling. Useful local commands are:

```bash
RUSTFLAGS="" cargo bench --locked -p openferric --no-default-features -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric \
  --features accelerated-native -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric --bench parallel_bench \
  --features accelerated-native -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric --bench simd_bench \
  --features simd -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric --bench solver_bench \
  --features parallel -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric --bench dsl_bench \
  --features parallel,simd,jit -- --noplot
RUSTFLAGS="" cargo bench --locked -p openferric --bench gpu_bench \
  --features gpu -- --noplot
```

The GPU benchmark records real process-cold initialization once, then measures
warm dispatch/readback and a same-process CPU reference at 100,000 and
1,000,000 paths. The scheduled GPU job enables Rayon for the CPU comparison;
a local `gpu`-only build uses scalar execution. Local hosts without an adapter
skip cleanly. The labelled GPU workflow sets `OPENFERRIC_REQUIRE_GPU=1`,
verifies that both Criterion groups and the cold-start record were emitted,
and fails if the adapter is unusable or Criterion reports an artifact/runtime
error even when its process exit status is zero.
It first runs `origin/main` in a detached worktree on the same host, reports
all warm-median changes, and fails when either warm GPU median regresses by
more than 25%. Cold initialization is recorded for visibility but excluded
from that steady-state gate. The dependency-free gate self-tests recursive
benchmark discovery, missing baselines, and an injected regression before the
GPU benchmark starts.

On the RTX 3080 implementation host, switching from additive per-invocation
Xoshiro seeding and raw moments to full-width keyed Threefry and centered
moments changed same-host warm medians by **+20.91% at 100,000 paths** and
**+14.18% at 1,000,000 paths**. That is the measured cost of removing seed
overlap/collisions and stabilizing standard errors. The 25% gate accepts this
intentional correctness cost, while leaving only about four percentage points
of additional headroom at the smaller workload.

The CPU job likewise runs `origin/main` first on the same labelled host and
then the candidate. It uses the candidate's SIMD and exact-terminal crossover
benchmark harness against both library revisions so a newly added threshold
group receives a like-for-like baseline immediately. Warm Criterion medians
for analytic SIMD, elementary SIMD math, exact-terminal SIMD/Auto and accuracy
tiering, exact-terminal/path Rayon crossover, and DSL JIT groups fail above a
20% regression. On the implementation host, the analytic crossover selected
SIMD at complete vectors (4, 8, 16, and
larger AVX2-aligned batches) while retaining scalar execution for costly short
partial tails; the exact-terminal crossover selected AVX2 from four paths,
where it measured 1.49x faster, and reached 2.30x at 511 paths.

On the 24-thread i9-12900K implementation host, exact-terminal `Auto` at the
5,120-effective-sample Rayon boundary measured **54.7 µs** versus **66.6 µs**
serial (about **18% lower latency**); immediately below the boundary it
remained serial. For one-step generic GBM, the 2-worker 512-sample boundary
measured **9.9 µs** under `Auto` versus **16.4 µs** serial (about **40% lower**).
The 24-worker 768-sample boundary measured **21.1 µs** under `Auto` versus
**24.2 µs** serial (about **13% lower**). Re-run the named Criterion groups
before recalibrating these implementation constants on another architecture.

The CPU and solver scaling benchmarks create explicit Rayon pools across
common thread counts up to the host's available parallelism.

Never publish a timing without at least:

```bash
git rev-parse HEAD
rustc -Vv
lscpu
# GPU runs:
lspci -nn | grep -Ei 'vga|3d|display'
vulkaninfo --summary  # when available
```

Also record the Cargo features, `RUSTFLAGS`, power mode, runner load, and
whether initialization is inside the timed region. Vendor-specific tools such
as `nvidia-smi` or `rocm-smi` are useful supplementary metadata when present,
but are not prerequisites. Criterion output is stored under
`target/criterion`.

The scheduled `Benchmarks` workflow uses labelled self-hosted CPU and GPU
runners, a pinned Rust toolchain, `RUSTFLAGS=""`, and `--locked`. It uploads
the Criterion history together with commit, compiler, CPU, and GPU metadata
for 90 days. Prefer those artifacts over isolated workstation numbers.

## Validation, coverage, fuzzing, and mutation testing

The normal local verification matrix is:

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features
cargo test --locked -p openferric --no-default-features
cargo test --locked -p openferric --no-default-features --features parallel,simd
cargo test --locked -p openferric --all-features
cargo test --locked -p openferric --test property_invariants --no-default-features
cargo test --locked -p openferric --test simd_test --features simd
cargo test --locked -p openferric --test dsl_examples --features jit
```

The deterministic `property_invariants` suite uses fixed-seed parameter sweeps
for put-call parity, no-arbitrage bounds, monotonicity, homogeneity, analytic
Greeks, barrier in/out parity, and curve/surface invariants. Each failure is
reproducible from its seed and reported parameters. SIMD tests cover empty
inputs, unaligned slices, every short vector tail, special floating-point
values, caller-owned workspace guards, and scalar differential checks. GPU
tests validate request limits, host reduction, WGSL parsing and Naga
validation; the native live-adapter comparison skips only when no adapter is
available. DSL tests compare every example product against the interpreter
when JIT is enabled.

CI runs the full scalar and all-feature core suites on Linux. Parallel-only
and SIMD-only Linux matrix entries are compile checks; Windows and macOS run
the complete accelerated native tests so architecture-specific JIT and
dispatch behavior cannot be hidden by a successful cross-platform build.
Native ARM64 separately executes the NEON/Rayon tests. The WASM job executes the
portable ABI and the same ABI compiled with required SIMD128 support under
Node, including differential tests of the explicit uniform Black-Scholes
price and Greek kernels. It also initializes a threaded Rayon pool and prices
a DSL product in headless Chrome, preserves the portable size budget, and
compile-checks (but does not runtime-test) WASM WebGPU.

Branch coverage merges three Rust configurations plus execution through the
Python bindings:

```bash
cargo +nightly llvm-cov clean --workspace
cargo +nightly llvm-cov --locked --branch --no-report \
  -p openferric --no-default-features
cargo +nightly llvm-cov --locked --branch --no-report \
  -p openferric --no-default-features --features parallel,simd
cargo +nightly llvm-cov --locked --branch --no-report \
  -p openferric --no-default-features --features jit
cargo +nightly llvm-cov report --branch --codecov \
  --output-path codecov.json \
  --ignore-filename-regex '/(tests|benches|examples|src/bin)/'
```

Codecov enforces an 80% patch target. The coverage workflow additionally
builds an instrumented Python wheel and runs `python/tests`.

Scheduled deep verification runs three libFuzzer targets, Miri, and mutation
testing. The equivalent focused commands are:

```bash
cargo +nightly fuzz run dsl_pipeline -- -max_total_time=120
cargo +nightly fuzz run trade_json -- -max_total_time=120
cargo +nightly fuzz run compiled_product_json -- -max_total_time=120
cargo +nightly miri test -p openferric --no-default-features --lib math::
cargo mutants --package openferric --features accelerated-native \
  --file src/core/engine.rs \
  --file src/dsl/ir.rs \
  --file src/pricing/european.rs \
  --file src/engines/analytic/bs_simd.rs \
  --timeout-multiplier 3 --jobs 2 \
  --cargo-test-arg=--lib
```

The fuzzers cover the complete DSL parse/diagnostic/compile pipeline,
serialized `TradeInstrument`/`Portfolio` inputs, and validation of deserialized
`CompiledProduct` IR. Miri targets scalar math
where unsupported SIMD instructions do not obscure memory-safety checks;
platform SIMD is covered by the native x86-64 and ARM64 test matrix.
