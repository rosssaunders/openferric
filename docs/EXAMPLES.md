# Examples

## European Call (Black-Scholes)

```rust
use openferric::core::PricingEngine;
use openferric::engines::analytic::BlackScholesEngine;
use openferric::instruments::VanillaOption;
use openferric::market::Market;

let market = Market::builder()
    .spot(100.0).rate(0.05).dividend_yield(0.0).flat_vol(0.20)
    .build()?;

let option = VanillaOption::european_call(100.0, 1.0);
let result = BlackScholesEngine::new().price(&option, &market)?;
println!("price = {:.4}, delta = {:.4}", result.price, result.greeks.delta);
```

## Heston via FFT (4096 strikes at once)

```rust
use openferric::engines::fft::carr_madan::heston_price_fft;

let prices = heston_price_fft(
    100.0,           // spot
    &strike_grid,    // Vec<f64>
    0.03,            // rate
    0.0,             // dividend
    0.04, 1.5, 0.04, 0.3, -0.7,  // v0, kappa, theta, sigma_v, rho
    1.0,             // maturity
)?;
// prices: Vec<(f64, f64)> — (strike, call_price) for all 4096 strikes
```

## CDS Fair Spread

```rust
use openferric::credit::{Cds, SurvivalCurve};
use openferric::rates::YieldCurve;

let cds = Cds {
    notional: 10e6,
    spread: 0.01,
    maturity: 5.0,
    recovery_rate: 0.40,
    payment_freq: 4,
};
let fair = cds.fair_spread(&discount_curve, &survival_curve);
```

## Deribit Vol Surface Tools

```bash
# One-shot snapshot (writes vol_surface.html)
cargo run --features deribit --bin deribit_vol_surface --release

# Rich dashboard (same UI as GitHub Pages)
wasm-pack build --target web --features wasm
cp -r pkg www/pkg
python3 -m http.server 3001 --bind 127.0.0.1 --directory www
# open http://127.0.0.1:3001
```

## Benchmarks

```bash
cargo bench
```
