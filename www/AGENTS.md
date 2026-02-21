# www/ Agent Guidelines

## The One Rule

**Every calculation goes through WASM. No exceptions.**

JavaScript gathers data (REST, WebSocket, user input) and renders results (Plotly, DOM). The Web Worker owns the WASM instance and orchestrates all computation. The main thread never calls WASM directly (except the optional WebGPU probe at startup).

## Before You Write Code

1. **Is it a formula?** → Write it in Rust (`src/wasm/`), expose via `#[wasm_bindgen]`, call from `worker.js`
2. **Is it data ingestion or rendering?** → Write it in `index.html`
3. **Is it compute orchestration?** → Write it in `worker.js`

## Architecture Invariants

These must never be violated:

- **Main thread does no math.** It parses JSON, maintains state (`chain` Map, `spotPrice`), dispatches to the worker, and renders Plotly charts. That's it.
- **Worker owns the WASM instance.** All `wasm.*` calls happen in `worker.js`. The worker receives raw market data via `postMessage`, runs the full pipeline (calibrate → compute → post results), and returns structured data.
- **WASM is pure math.** Calibration solvers, BS pricing, Greeks, IV evaluation, Newton solvers, realized vol — all live in Rust. No DOM, no fetch, no state.
- **No shared memory.** Main thread and worker communicate exclusively via `postMessage` with structured clone. No `SharedArrayBuffer`.

## Compute Pipeline (worker.js)

Every `market-update` message triggers this sequence:

```
1. runCalibration     — WASM: fit_sabr / calibrate_svi + slice_fit_diagnostics
2. computeFrameCache  — WASM: find_25d_strikes_batch + iv_grid (ATM)
3. computeSurfaceData — WASM: iv_grid (15-point k-grid)     [every 5 cycles]
4. computeGreeksData  — WASM: batch_slice_iv + bsm_greeks_batch  [every 2 cycles]
5. computeScannerData — WASM: batch_slice_iv + bs_price_batch    [every 3 cycles]
6. computeForwardVol  — WASM: iv_grid (7-point k-grid)          [every cycle]
7. computeRealizedVol — WASM: realized_vol                       [every cycle]
```

Results posted back as `{ type: 'compute-result', payload: {...} }`.

## WASM Interface Conventions

- All numeric arrays cross the boundary as `Float64Array`
- Booleans as `Uint8Array` (1=call, 0=put) — wasm-bindgen limitation
- Slice data packed as headers (`[model_type, T, forward, param_offset]` × N) + params (concatenated)
- IV returned as percentage (45.2 not 0.452)
- Time as year fractions, rates continuously compounded

## Adding a New Feature

### New WASM function
```
1. src/wasm/mod.rs  → implement with #[wasm_bindgen]
2. wasm-pack build --target web --features wasm
3. worker.js        → call wasm.new_function(...) in appropriate pipeline stage
4. worker.js        → include results in postMessage payload
5. index.html       → consume in render function
```

### New panel / visualization
```
1. index.html → add Dockview panel registration + render function
2. worker.js  → if new data needed, add compute stage (use WASM for any math)
3. index.html → handle new fields in applyComputeResult()
```

### New data source
```
1. index.html → add REST seed or WS subscription
2. index.html → normalize and include in market-update payload
3. worker.js  → consume in pipeline
```

## Common Mistakes to Avoid

- Computing IV edge, dollar edge, or any pricing formula in JS — use WASM
- Adding `Math.exp`, `Math.log`, `Math.sqrt` sequences that implement BS or Greeks — use WASM
- Calling WASM from `index.html` main thread — route through worker
- Doing rolling vol, percentile, or statistical computation in JS — add a WASM function
- Duplicating Rust logic in JS "for convenience" — never duplicate, always call WASM

## File Map

| File | Role | Lines |
|---|---|---|
| `index.html` | Full SPA: CSS + HTML + main thread JS | ~2,600 |
| `worker.js` | Web Worker: WASM owner + compute pipeline | ~700 |
| `pkg/` | wasm-pack output (generated, do not edit) | — |

## Testing

- WASM functions are tested via `cargo test` in the Rust crate
- Frontend behavior tested by running the app against Deribit testnet
- Build: `wasm-pack build --target web --features wasm` from repo root
