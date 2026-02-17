# OpenFerric Excel Add-in

Quantitative finance functions in Excel — powered by Rust + WebAssembly. Cross-platform (Windows, Mac, Excel Online).

## Functions

| Function | Description |
|---|---|
| `=OPENFERRIC.BS_PRICE(S, K, r, q, σ, T, call?)` | Black-Scholes European option price |
| `=OPENFERRIC.BS_IMPLIED_VOL(price, S, K, r, q, T, call?)` | Implied volatility |
| `=OPENFERRIC.BS_GREEKS(S, K, r, q, σ, T, call?)` | Delta, gamma, vega, theta, rho, vanna, volga |
| `=OPENFERRIC.HESTON_PRICE(S, K, r, q, v₀, κ, θ, σᵥ, ρ, T, call?)` | Heston stochastic vol price |
| `=OPENFERRIC.BARRIER_PRICE(S, K, H, r, q, σ, T, type, call?)` | Barrier option (up-in/out, down-in/out) |
| `=OPENFERRIC.BOND_PRICE(face, coupon, T, yield, freq)` | Fixed-rate bond price |
| `=OPENFERRIC.CDS_SPREAD(notional, T, recovery, λ, r)` | CDS fair spread |
| `=OPENFERRIC.VAR(returns, confidence)` | Historical Value-at-Risk |
| `=OPENFERRIC.SABR_VOL(F, K, T, α, β, ρ, ν)` | SABR implied volatility |

## Quick Start

```bash
# 1. Build WASM package (from repo root)
wasm-pack build --target web --out-dir excel-addin/pkg --release

# 2. Start the dev server
cd excel-addin
node serve.js

# 3. Sideload in Excel
#    Excel Online: Insert → Office Add-ins → Upload My Add-in → manifest.xml
#    Excel Desktop: https://learn.microsoft.com/office/dev/add-ins/testing/sideload-office-add-ins
```

## Architecture

```
Excel ──→ Office.js ──→ functions.js ──→ WASM (openferric)
                                            │
                                    Rust compiled to WebAssembly
                                    All computation client-side
```

No server required. The WASM binary contains the full OpenFerric pricing library.

## Development

The dev server (`serve.js`) generates a self-signed certificate for HTTPS (required by Office Add-ins). You'll need to accept the certificate warning in your browser the first time.

### Adding new functions

1. Add the `#[wasm_bindgen]` export in `src/wasm/mod.rs`
2. Add metadata to `functions.json`
3. Add the JS wrapper in `functions.js`
4. Register with `CustomFunctions.associate()`
5. Rebuild WASM: `wasm-pack build --target web --out-dir excel-addin/pkg --release`

## Requirements

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) for building WASM
- Node.js for the dev server
- OpenSSL for self-signed certificates
- Excel 2016+ or Excel Online
