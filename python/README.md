# OpenFerric Python Bindings

Python bindings to the native Rust pricing, market, model, calibration, risk,
DSL, and numerical APIs. Calculations delegate to Rust; Python does not contain
a separate implementation of the pricing formulas.

## Build and install

Run these commands from the repository root with Python 3.9+, NumPy, the pinned
Rust toolchain in `rust-toolchain.toml`, and maturin installed:

```bash
python -m pip install maturin pytest
maturin build --locked --release -m python/Cargo.toml
python -m pip install --force-reinstall target/wheels/openferric-*.whl
python -m pytest python/tests -v
```

For development, activate a virtual environment and run:

```bash
maturin develop --locked -m python/Cargo.toml
```

Default wheels include `parallel`, `simd`, `jit`, and `gpu`. Building without
optional backends is supported:

```bash
maturin build --locked --release --no-default-features -m python/Cargo.toml
```

Inspect compiled capabilities with `openferric.build_features()`. Compiled GPU
support does not guarantee a working GPU adapter. JIT support is platform
dependent. Explicit unsupported execution policies return errors rather than
silently selecting another backend. `Auto` permits native backend selection.

## Typed pricing

```python
from openferric.engines import BlackScholesEngine, BinomialTreeEngine
from openferric.instruments import VanillaOption
from openferric.market import Market

market = Market.builder().spot(100).rate(0.05).dividend_yield(0).flat_vol(0.2).build()
option = VanillaOption.european_call(100, 1)

result = BlackScholesEngine().price(option, market)
assert abs(result.price - 10.450583572185565) < 1e-12
tree_result = BinomialTreeEngine(400).price(option, market)
greeks_result = BlackScholesEngine().price_with_greeks_aad(option, market)
```

Engines accept their supported native instrument types, not arbitrary Python
objects. Results expose price, optional standard error, optional Greeks, and
diagnostics. Rates and time use the core API's units: year fractions and
continuously compounded rates unless the particular module states otherwise.

Root-level classes and functions remain available. Functions exported with
`py_` prefixes or `_py` suffixes also have clean aliases when the name is not
already occupied. Prefer keyword arguments when using scalar helpers, whose
parameter ordering follows their existing Rust adapters:

```python
import openferric as of

price = of.bs_price(spot=100, strike=100, expiry=1, vol=0.2, rate=0.05, option_type="call")
```

## API namespaces

| Namespace | Native functionality |
|-----------|----------------------|
| `core` | Domain types, pricing results, diagnostics, execution policies |
| `instruments` | All trade variants, builders, exotics, callable and coupon notes, variance/volatility swaps, weather, catastrophe, MBS, real options |
| `engines.analytic` | Black-Scholes, Black-76, FX, digital, barrier, Asian, rainbow, spread, variance, exotic engines and kernels |
| `engines.tree` | Binomial, trinomial, generalized and two-asset trees, swing, convertible, Bermudan swaption |
| `engines.pde` | Crank-Nicolson, implicit/explicit finite differences, hopscotch, ADI Heston, exercise boundaries |
| `engines.lsm` | Longstaff-Schwartz, GBM/local-vol/Heston dynamics, Bermudan exercise boundaries |
| `engines.numerical` | American binomial and reusable pricing arenas |
| `mc`, `engines.monte_carlo` | Native pricing, custom payoffs, variance reduction, pathwise/LR/AAD Greeks, Sobol QMC, owned paths, parallel grids |
| `engines.gpu` | GPU readiness, prewarming and native GPU Monte Carlo when compiled |
| `fft`, `engines.fft` | Characteristic functions, Carr-Madan contexts, FFT/FRFT grids, derivatives, complex values |
| `models` | GBM, Heston, Hull-White, CIR, Vasicek, HJM, LMM, SLV, rough volatility, Levy and commodity models |
| `calibration` | Heston, Hull-White, SABR, SVI, LM/DE/Nelder-Mead optimizers, constraints and diagnostics |
| `market` | Markets, snapshots, dividends, FX conventions/curves/smiles, sampled and parametric surfaces |
| `rates`, `funding` | Curves, schedules/calendars, bonds, swaps, FRA, futures, cap/floor, CMS, inflation, multicurve and funding analytics |
| `credit` | CDS/ISDA, indices, baskets, CDO, default simulation and survival bootstrapping |
| `risk` | Portfolio risk, margin, liquidation, scenarios, VaR/ES, XVA, SA-CCR and SIMM |
| `vol` | Implied/local volatility, SABR/SVI, surfaces, forward variance, VIX, arbitrage and calibration utilities |
| `greeks` | Analytic sensitivities, Python pricing callbacks, owned and in-place batch Greeks |
| `math` | AAD/dual numbers, RNGs, Sobol, quadrature, root finding, interpolation, copulas/correlation |
| `timeseries`, `math.timeseries` | Returns, moments, distributions, correlation/shrinkage and VaR backtesting |
| `dsl` | Lexer, AST, compiler, validated IR, language analysis, evaluator, multi-asset markets, Monte Carlo and JIT |

`instruments.Portfolio` is the serializable trade portfolio; `risk.Portfolio`
is the risk portfolio. The root exports these as `InstrumentPortfolio` and
`Portfolio`, respectively. `rates.cms_convexity_adjustment_simple` exposes the
three-scalar adjustment; `rates.cms_convexity_adjustment` takes `CmsConvexityParams`.

## DSL and serialization

```python
import openferric as of

source = '''product "Redemption"
    notional: 100
    maturity: 1
    underlyings
        SPX = asset(0)
    schedule annual from 1 to 1
        redeem notional
'''
product = of.dsl.CompiledProduct(source)
restored = of.dsl.CompiledProduct.from_json(product.to_json())
result = of.dsl.DslMonteCarloEngine(256, 4, 42).price_multi_asset(
    restored, of.dsl.MultiAssetMarket.single(100, 0.2, 0.05), of.ExecutionPolicy.Scalar
)

trade = of.Trade(of.TradeMetadata("trade-1", 1, 0), of.VanillaOption.european_call(100, 1))
portfolio = of.instruments.Portfolio("book", [trade])
roundtrip = of.instruments.Portfolio.from_dict(portfolio.to_dict())
```

`TradeInstrument.from_instrument(contract)` accepts every native trade variant.
`TradeInstrument(kind, payload)` and `from_dict` accept the native serde schema
(`{"type": ..., "data": ...}`). Deserialization is a data operation; price-time
validation still applies. Compiled DSL products are validated on deserialization.

AST/IR records support `to_dict`, `from_dict`, and keyword construction, e.g.
`dsl.Span(start=0, end=7)`. Their fields are readable as attributes. Payload enums
use native serde tagged data: `dsl.Value({"F64": 3.5})`; unit variants use their
names, e.g. `dsl.ScheduleFreq("Quarterly")`. Nested wrapper values are accepted
when constructing records. Returned lists and serialized dictionaries are owned
copies, not mutable views into Rust objects.

## Python callbacks and buffers

- Custom Monte Carlo payoffs receive path lists. `MonteCarloEngine.run` accepts
  native GBM/Heston generators; `run_fallible` preserves Python exceptions.
  Scalar execution is the default for callback ordering. Parallel execution can
  be explicitly selected with `CpuExecutionPolicy`.
- FFT contexts accept native characteristic functions or Python callables
  returning complex values. Optimizers, root finders, quadrature, and finite
  difference Greeks accept Python callbacks and propagate their exceptions.
- `Market.builder().vol_surface(surface)` accepts native surfaces or a Python
  callable/object with `vol(strike, expiry)`. Arbitrary Python surfaces are
  sampled once onto the native `SampledVolSurface` grid, not called per pricing
  step. Native SVI parametric surfaces are retained exactly.
- Rust-owned long-running calculations release the GIL. Python callbacks
  re-enter Python; NumPy in-place buffer operations retain the GIL while borrowed.
- GPU calculations use portable WebGPU `float32` arithmetic, with final
  statistics reduced in `float64`. GPU standard errors measure sampling error,
  not floating-point error; use CPU engines for tight binary64 tolerances.
- `bs_price_batch` uses contiguous one-dimensional `float64` NumPy inputs.
  The `*_into` functions require contiguous, writable `float64` output arrays
  of matching length. Overlapping Greek outputs and read-only outputs raise
  ordinary Python errors. Inputs are copied before in-place output writes.
- `PricingArena` owns reusable native scratch storage. Python buffer/slice
  getters return copies; assign the buffer property to change its contents.
- Calendar dates use `YYYY-MM-DD`; funding timestamps use ISO 8601 UTC strings,
  e.g. `2026-03-18T00:00:00Z`.

## Coverage contract

[`api_manifest.json`](api_manifest.json) maps the reviewed public Rust symbols,
inherent methods, fields, and enum variants to concrete Python entry points.
Tests verify the installed extension against this manifest and round-trip every
trade variant against committed Rust-generated serialization fixtures. CI also
checks all-feature Rust exports and runs both default and portable Python wheels.

Rust traits map to concrete Python implementations or callback protocols. Raw
architecture-specific SIMD registers/pointers are intentionally not exported;
use the safe runtime-dispatched batch APIs. Raw `JitCompiledProgram` execution is
likewise Rust-only; Python uses validated `JitProductEvaluator` and owned scratch.
These exceptions and their alternatives are explicit in the manifest. Nothing
exposes unchecked native pointers to Python.

On x86-64, check Rust/Python parity after public API changes:

```bash
rustup toolchain install nightly-2026-03-10 --profile minimal
cargo +nightly-2026-03-10 rustdoc --locked -p openferric --all-features --lib -- -Z unstable-options --output-format json
python python/scripts/check_api.py --rustdoc target/doc/openferric.json
```

Update the manifest and appropriate behavioral tests when adding public Rust
APIs; a name-presence check alone is not evidence of numerical correctness.

The complete funding-rate swap walkthrough is in
[`examples/boros_funding_swap.py`](../examples/boros_funding_swap.py).
