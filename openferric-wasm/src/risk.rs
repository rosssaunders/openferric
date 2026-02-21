use wasm_bindgen::prelude::*;

use openferric::risk::var::historical_var;

/// Historical Value-at-Risk from a flat array of P&L returns.
#[wasm_bindgen]
pub fn var_historical(returns_flat: &[f64], confidence: f64) -> f64 {
    if returns_flat.is_empty() {
        return f64::NAN;
    }
    historical_var(returns_flat, confidence)
}
