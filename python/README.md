# OpenFerric Python Bindings

Python interface to the [OpenFerric](https://github.com/rosssaunders/openferric) quantitative finance library, built with [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/).

## Quick Start

### Prerequisites

- Python 3.9+
- Rust toolchain (stable)
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

### Install (development)

```bash
cd python
pip install maturin
maturin develop
```

This compiles the Rust library and installs `openferric` as an editable Python package.

### Install (release wheel)

```bash
cd python
maturin build --release
pip install target/wheels/openferric-*.whl
```

## Usage

```python
import openferric as of

# Black-Scholes pricing
price = of.bs_price(100, 100, 0.25, 0.05, 0.0, 0.2, "call")
greeks = of.bs_greeks(100, 100, 0.25, 0.05, 0.0, 0.2, "call")

# Implied volatility
iv = of.implied_vol(10.45, 100, 100, 0.25, 0.05, 0.0, "call")

# SABR vol
vol = of.sabr_vol(100, 100, 0.25, 0.3, 0.8, -0.4, 0.5)

# Heston pricing (analytic + FFT)
price = of.heston_price(100, 100, 0.25, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call")
prices = of.heston_fft_prices(100, [90, 95, 100, 105, 110], 0.25, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)

# VaR / Expected Shortfall
var = of.historical_var([-0.02, -0.01, 0.005, 0.01, -0.03], 0.95)
es = of.historical_es([-0.02, -0.01, 0.005, 0.01, -0.03], 0.95)
```

## Modules

| Module | What's covered |
|--------|---------------|
| **pricing** | Black-Scholes, barriers, American, Heston, FX, digital, spread, lookback, Asian, autocallable, basket, Bermudan, range accrual, TARF, real options |
| **vol** | Implied vol, SABR, SVI, vol surfaces, Fengler, Andreasen-Huge, forward variance, ATM skew |
| **rates** | Yield curves, bonds (dirty/clean price, duration, convexity, YTM), calendars, cap/floor, CMS, FRA, IRS, swaptions, funding rate curves |
| **credit** | CDS NPV, survival curves, CDO tranches, CDS indices, nth-to-default baskets, Gaussian copula |
| **risk** | CVA/DVA/FVA/MVA/KVA, SA-CCR EAD, VaR, ES, SIMM, portfolio risk, margin, liquidation simulation, stress testing |
| **instruments** | Options, barriers, baskets, exotics, structured notes, autocallables, funding swaps |
| **models** | Heston, Hull-White, CIR, Vasicek, SABR, GBM, Schwartz commodity, LMM/BGM, rBergomi |
| **calibration** | Heston calibrator, Hull-White calibrator, diagnostics, vol quotes |
| **market** | Market data, dividends, FX pairs, FX forward curves, FX vol surfaces |
| **engines** | Analytic, PDE, Monte Carlo pricing engines |
| **math** | AAD (automatic differentiation), interpolation, copulas, correlation models |
| **mc** | GBM/Heston path generators, Monte Carlo engine, control variates |
| **fft** | Heston FFT, Variance Gamma, CGMY, NIG |

## Examples

### Boros Funding Rate Swap

A complete Pendle Boros funding-rate swap walkthrough — builds curves from exchange data, prices swaps, computes risk, and runs liquidation simulations:

```bash
# With synthetic data (no dependencies):
python examples/boros_funding_swap.py

# With live Binance/Bybit data:
pip install requests
python examples/boros_funding_swap.py --live
```

See [`examples/boros_funding_swap.py`](../examples/boros_funding_swap.py) for the full source.

## Running Tests

```bash
cd python
maturin develop
python -m pytest tests/ -v
```

## Architecture

The Python package is a thin PyO3 wrapper around the Rust `openferric` core library. No finance logic is reimplemented in Python — every calculation calls directly into optimised Rust code.

```
openferric (Python)
  └── PyO3 FFI layer (python/src/*.rs)
       └── openferric core (Rust library)
```

DateTime values are passed as ISO 8601 strings (`"2026-03-18T00:00:00Z"`).
Enums are constructed via class methods (e.g. `StressScenario.baseline()`).
