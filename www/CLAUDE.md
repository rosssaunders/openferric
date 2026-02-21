# www/ — Volatility Terminal Frontend

## Golden Rule

**ALL heavy computation runs in WASM via the Web Worker. JavaScript does ZERO math beyond trivial data marshalling (array slicing, Map lookups, simple aggregation). If it involves a formula, it belongs in Rust/WASM.**

## Architecture (3 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Main Thread (index.html)                      │
│  - REST/WS data ingestion (Deribit)                     │
│  - DOM manipulation, Plotly rendering, Dockview layout   │
│  - IndexedDB persistence                                │
│  - Dispatches data to worker, renders results            │
│  - NO MATH. NO PRICING. NO CALIBRATION.                  │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Web Worker (worker.js)                        │
│  - Owns the single WASM instance                        │
│  - Orchestrates compute pipeline per render cycle        │
│  - Packs/unpacks Float64Array for WASM calls             │
│  - Tiered cadence: surface/5, greeks/2, scanner/3       │
│  - Posts structured results back to main thread          │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Rust/WASM (pkg/openferric_wasm_bg.wasm)       │
│  - SVI/SABR calibration                                 │
│  - BSM pricing + Greeks (batch vectorized)               │
│  - IV evaluation (iv_grid, batch_slice_iv)               │
│  - 25-delta strike search (Newton's method)              │
│  - Realized vol, fit diagnostics, forward vol            │
│  - Heston FFT, barrier pricing, VaR                      │
└─────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose | Touches WASM? |
|---|---|---|
| `index.html` | Full SPA — CSS, markup, main `<script>` | NO (except optional GPU test at startup) |
| `worker.js` | Compute orchestrator on Web Worker thread | YES — all WASM calls go through here |
| `pkg/` | wasm-pack output (`wasm-build` target web) | Generated — do not edit |

## Message Protocol (main ↔ worker)

### Main → Worker

| `type` | When | Payload |
|---|---|---|
| `config-update` | Model/greek/filter change | `{ activeModel, activeGreek, edgeThreshold, edgeSideFilter, isLinear }` |
| `market-update` | 1s debounce after tick | `{ chainEntries, spotPrice, spotHistory, renderCycle }` |
| `strategy-compute` | Strategy leg/slider edit | `{ legs, spotPrice, spotShock, volShock, timeShock }` |

### Worker → Main

| `type` | When | Payload |
|---|---|---|
| `ready` | WASM init done | — |
| `compute-result` | Pipeline complete | `{ calibratedSlices, calibTimeUs, realizedVol, d25, atmIv, termRr25, termBf25, surface, greeks, scanner, forwardVol }` |
| `strategy-result` | Strategy done | `{ spotAxis, pnlAtExpiry, pnlBeforeExpiry, netGreeks, totalCost }` |

## WASM Functions (all called from worker.js only)

| Function | Purpose |
|---|---|
| `calibrate_slice_wasm` | Per-expiry calibration: filtering, log-moneyness, SVI/SABR/VV fit, diagnostics — single call |
| `log_moneyness_batch_wasm` | Batch `ln(K/F)` computation (broadcasts single forward) |
| `log_returns_batch_wasm` | Batch `ln(p[i]/p[i-1])` from raw price series |
| `calibrate_svi_wasm` | SVI gradient descent (3000 iters) — used internally by `calibrate_slice_wasm` |
| `fit_sabr_wasm` | SABR calibration — used internally by `calibrate_slice_wasm` |
| `slice_fit_diagnostics` | Per-slice RMSE, skew, kurtosis, fitted IVs |
| `iv_grid` | IV surface mesh (n_slices × n_k, row-major) |
| `batch_slice_iv` | Irregular per-option IV lookups |
| `find_25d_strikes_batch` | Newton's method 25-delta strikes |
| `bs_price_batch_wasm` | Batch BS pricing |
| `bsm_greeks_batch_wasm` | Batch Greeks (delta,gamma,vega,theta,rho,vanna,volga) |
| `realized_vol` | Annualized RVol from log returns |

## Data Flow

```
Deribit REST+WS → index.html (parse, normalize, chain Map)
    → postMessage('market-update') → worker.js
    → WASM calibrate → WASM compute surface/greeks/scanner/fwdvol
    → postMessage('compute-result') → index.html
    → Plotly render
```

## Build

```bash
# From repo root
wasm-pack build openferric-wasm --target web --out-dir ../www/pkg
```

## Conventions

- **Year fractions** for all time values (f64)
- **Continuously compounded** rates
- **Log-moneyness** `k = ln(K/F)` for smile/surface axes
- **IV as percentage** (e.g., 45.2 not 0.452) in WASM return values
- **Float64Array** for all numeric data crossing the WASM boundary
- **Uint8Array** for boolean arrays (1=call, 0=put) — wasm-bindgen limitation
- Slice packing: headers = `[model_type, T, forward, param_offset]` per slice; params = concatenated model params

## What JS May Do

- Parse WebSocket/REST JSON
- Maintain the `chain` Map and `spotPrice`
- Build `Float64Array` / `Uint8Array` for WASM input
- Render Plotly charts from worker results
- DOM manipulation (tables, panels, sparklines)
- IndexedDB read/write
- Trivial array iteration for rendering (e.g., sparkline SVG polyline points)

## What JS Must NOT Do

- Black-Scholes pricing or Greeks
- Vol surface calibration (SVI, SABR, or any model)
- IV interpolation or evaluation
- Newton solvers, gradient descent, optimization
- Realized vol computation
- Any formula that appears in a quant textbook
- Forward vol computation from total variance

## Adding New Computation

1. Implement in Rust under `openferric-wasm/src/` with `#[wasm_bindgen]`
2. Rebuild: `wasm-pack build openferric-wasm --target web --out-dir ../www/pkg`
3. Call from `worker.js` — never from `index.html`
4. Post results back via `postMessage`
