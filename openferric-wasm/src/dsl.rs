use wasm_bindgen::prelude::*;

use openferric::dsl::{DslMonteCarloEngine, MultiAssetMarket, parse_and_compile};

/// Parse and compile a DSL source string.
///
/// Returns JSON: `{"ok": <compiled_product>}` or
/// `{"err": {"kind": "...", "message": "...", "span_start": N, "span_end": N}}`.
#[wasm_bindgen]
pub fn dsl_parse_and_compile(source: &str) -> String {
    match parse_and_compile(source) {
        Ok(product) => {
            let product_json = serde_json::to_value(&product).unwrap();
            serde_json::to_string(&serde_json::json!({ "ok": product_json })).unwrap()
        }
        Err(e) => {
            let (kind, message, span_start, span_end) = match &e {
                openferric::dsl::DslError::LexError { message, span } => {
                    ("lex", message.clone(), span.start, span.end)
                }
                openferric::dsl::DslError::ParseError { message, span } => {
                    ("parse", message.clone(), span.start, span.end)
                }
                openferric::dsl::DslError::CompileError { message, span } => {
                    let (s, e) = span.map_or((0, 0), |sp| (sp.start, sp.end));
                    ("compile", message.clone(), s, e)
                }
                openferric::dsl::DslError::EvalError(msg) => ("eval", msg.clone(), 0, 0),
            };
            serde_json::to_string(&serde_json::json!({
                "err": {
                    "kind": kind,
                    "message": message,
                    "span_start": span_start,
                    "span_end": span_end,
                }
            }))
            .unwrap()
        }
    }
}

/// Price a compiled DSL product.
///
/// `product_json`: JSON string of `CompiledProduct` (from `dsl_parse_and_compile`'s `ok` field).
/// `market_json`: JSON string of `MultiAssetMarket`.
///
/// Returns JSON: `{"price": N, "stderr": N}` or `{"err": "..."}`.
#[wasm_bindgen]
pub fn dsl_price(
    product_json: &str,
    market_json: &str,
    num_paths: u32,
    num_steps: u32,
    seed: u64,
) -> String {
    let product = match serde_json::from_str(product_json) {
        Ok(p) => p,
        Err(e) => {
            return serde_json::to_string(
                &serde_json::json!({"err": format!("invalid product JSON: {e}")}),
            )
            .unwrap();
        }
    };
    let market: MultiAssetMarket = match serde_json::from_str(market_json) {
        Ok(m) => m,
        Err(e) => {
            return serde_json::to_string(
                &serde_json::json!({"err": format!("invalid market JSON: {e}")}),
            )
            .unwrap();
        }
    };

    let engine = DslMonteCarloEngine::new(num_paths as usize, num_steps as usize, seed);
    match engine.price_multi_asset(&product, &market) {
        Ok(result) => serde_json::to_string(&serde_json::json!({
            "price": result.price,
            "stderr": result.stderr.unwrap_or(0.0),
        }))
        .unwrap(),
        Err(e) => serde_json::to_string(&serde_json::json!({"err": e.to_string()})).unwrap(),
    }
}

/// Compute Greeks for a compiled DSL product via bump-and-reprice.
///
/// Returns JSON: `{"delta": N, "gamma": N, "vega": N, "rho": N}` or `{"err": "..."}`.
#[wasm_bindgen]
pub fn dsl_greeks(
    product_json: &str,
    market_json: &str,
    num_paths: u32,
    num_steps: u32,
    seed: u64,
    asset_index: u32,
) -> String {
    let product = match serde_json::from_str(product_json) {
        Ok(p) => p,
        Err(e) => {
            return serde_json::to_string(
                &serde_json::json!({"err": format!("invalid product JSON: {e}")}),
            )
            .unwrap();
        }
    };
    let market: MultiAssetMarket = match serde_json::from_str(market_json) {
        Ok(m) => m,
        Err(e) => {
            return serde_json::to_string(
                &serde_json::json!({"err": format!("invalid market JSON: {e}")}),
            )
            .unwrap();
        }
    };

    let engine = DslMonteCarloEngine::new(num_paths as usize, num_steps as usize, seed);
    match engine.greeks_multi_asset(&product, &market, asset_index as usize) {
        Ok(greeks) => serde_json::to_string(&serde_json::json!({
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "vega": greeks.vega,
            "rho": greeks.rho,
        }))
        .unwrap(),
        Err(e) => serde_json::to_string(&serde_json::json!({"err": e.to_string()})).unwrap(),
    }
}
