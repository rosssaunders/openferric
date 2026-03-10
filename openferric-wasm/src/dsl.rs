use wasm_bindgen::prelude::*;

use openferric::dsl::analysis;
use openferric::dsl::ir::CompiledProduct;
use openferric::dsl::market::AssetMarketData;
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

// =========================================================================
// Analysis endpoints for the Monaco editor
// =========================================================================

/// Full analysis: parse + diagnose + semantic tokens + compile.
///
/// Returns JSON:
/// ```json
/// {
///   "diagnostics": [{"severity": "error", "message": "...", "start": N, "end": N}],
///   "semanticTokens": [[deltaLine, deltaStart, length, tokenType, modifiers], ...],
///   "compiled": <CompiledProduct JSON or null>
/// }
/// ```
#[wasm_bindgen]
pub fn dsl_analyze(source: &str) -> String {
    let (ast, product, diags) = analysis::parse_and_diagnose(source);

    let symbols = ast
        .as_ref()
        .map(|a| analysis::build_symbol_table(a, source))
        .unwrap_or_default();

    let semantic_tokens = analysis::semantic_token_data(source, &symbols);

    let diags_json: Vec<serde_json::Value> = diags
        .iter()
        .map(|d| {
            serde_json::json!({
                "severity": match d.severity {
                    analysis::DiagnosticSeverity::Error => "error",
                    analysis::DiagnosticSeverity::Warning => "warning",
                },
                "message": d.message,
                "start": d.start,
                "end": d.end,
            })
        })
        .collect();

    let tokens_json: Vec<Vec<u32>> = semantic_tokens
        .iter()
        .map(|t| {
            vec![
                t.delta_line,
                t.delta_start,
                t.length,
                t.token_type,
                t.modifiers,
            ]
        })
        .collect();

    let compiled_json = product.map(|p| serde_json::to_value(&p).unwrap());

    serde_json::to_string(&serde_json::json!({
        "diagnostics": diags_json,
        "semanticTokens": tokens_json,
        "compiled": compiled_json,
    }))
    .unwrap()
}

/// Hover information at a byte offset.
///
/// Returns JSON: `{"markdown": "...", "start": N, "end": N}` or `"null"`.
#[wasm_bindgen]
pub fn dsl_hover(source: &str, offset: u32) -> String {
    let symbols = build_symbols_from_source(source);
    match analysis::hover_info(source, &symbols, offset as usize) {
        Some(info) => serde_json::to_string(&serde_json::json!({
            "markdown": info.markdown,
            "start": info.start,
            "end": info.end,
        }))
        .unwrap(),
        None => "null".to_string(),
    }
}

/// Context-aware completions at a byte offset.
///
/// Returns JSON array of `{label, kind, detail, documentation}`.
#[wasm_bindgen]
pub fn dsl_completions(source: &str, offset: u32) -> String {
    let symbols = build_symbols_from_source(source);
    let candidates = analysis::completions(source, &symbols, offset as usize);
    serde_json::to_string(&candidates).unwrap()
}

/// Go to definition at a byte offset.
///
/// Returns JSON: `{"start": N, "end": N}` or `"null"`.
#[wasm_bindgen]
pub fn dsl_goto_definition(source: &str, offset: u32) -> String {
    let symbols = build_symbols_from_source(source);
    match analysis::goto_definition(source, &symbols, offset as usize) {
        Some(span) => serde_json::to_string(&serde_json::json!({
            "start": span.start,
            "end": span.end,
        }))
        .unwrap(),
        None => "null".to_string(),
    }
}

/// Full pricing: price + extended greeks + cross-greeks + payoff profile.
///
/// `product_json`: JSON string of `CompiledProduct`.
/// `market_json`: JSON string of `MultiAssetMarket` (or "null" for defaults).
#[wasm_bindgen]
pub fn dsl_price_full(
    product_json: &str,
    market_json: &str,
    num_paths: u32,
    num_steps: u32,
    seed: u64,
) -> String {
    let product: CompiledProduct = match serde_json::from_str(product_json) {
        Ok(p) => p,
        Err(e) => {
            return serde_json::to_string(
                &serde_json::json!({"error": format!("invalid product JSON: {e}")}),
            )
            .unwrap();
        }
    };

    let market = if market_json == "null" || market_json.is_empty() {
        build_default_market(product.num_underlyings)
    } else {
        match serde_json::from_str::<MultiAssetMarket>(market_json) {
            Ok(mut m) => {
                pad_market(&mut m, product.num_underlyings);
                m
            }
            Err(_) => build_default_market(product.num_underlyings),
        }
    };

    let underlying_names: Vec<String> =
        product.underlyings.iter().map(|u| u.name.clone()).collect();

    let engine = DslMonteCarloEngine::new(num_paths as usize, num_steps as usize, seed);

    // Price
    let (price, stderr, error) = match engine.price_multi_asset(&product, &market) {
        Ok(result) => (result.price, result.stderr, None),
        Err(e) => (0.0, None, Some(format!("{e}"))),
    };

    // Extended greeks per underlying
    let mut greeks = Vec::new();
    for i in 0..product.num_underlyings {
        let name = product
            .underlyings
            .get(i)
            .map(|u| u.name.clone())
            .unwrap_or_else(|| format!("Asset {i}"));
        if let Ok(g) = engine.extended_greeks_multi_asset(&product, &market, i, price) {
            greeks.push(serde_json::json!({
                "asset": name,
                "delta": g.delta,
                "gamma": g.gamma,
                "vega": g.vega,
                "theta": g.theta,
                "rho": g.rho,
                "vanna": g.vanna,
                "volga": g.volga,
            }));
        }
    }

    // Cross-greeks
    let mut cross_greeks = Vec::new();
    for i in 0..product.num_underlyings {
        for j in (i + 1)..product.num_underlyings {
            let name_i = product
                .underlyings
                .get(i)
                .map(|u| u.name.clone())
                .unwrap_or_else(|| format!("Asset {i}"));
            let name_j = product
                .underlyings
                .get(j)
                .map(|u| u.name.clone())
                .unwrap_or_else(|| format!("Asset {j}"));
            if let Ok(cg) = engine.cross_greeks_multi_asset(&product, &market, i, j, price) {
                cross_greeks.push(serde_json::json!({
                    "assetI": name_i,
                    "assetJ": name_j,
                    "crossGamma": cg.cross_gamma,
                    "corrSens": cg.corr_sens,
                }));
            }
        }
    }

    // Payoff profile: 21 spot levels from 50% to 150%
    let payoff_paths = (num_paths as usize).min(5_000);
    let payoff_engine = DslMonteCarloEngine::new(payoff_paths, num_steps as usize, seed);
    let mut payoff_profile = Vec::with_capacity(21);
    for i in 0..=20 {
        let pct = 50.0 + (i as f64) * 5.0;
        let scale = pct / 100.0;
        let mut bumped_market = market.clone();
        for asset in &mut bumped_market.assets {
            let base_val = asset.initial_value();
            let bump = base_val * (scale - 1.0);
            *asset = asset.with_spot_bump(bump);
        }
        let pv = payoff_engine
            .price_multi_asset(&product, &bumped_market)
            .map(|r| r.price)
            .unwrap_or(0.0);
        payoff_profile.push(serde_json::json!({ "spotPct": pct, "pv": pv }));
    }

    // Market snapshot
    let market_snapshot = serde_json::json!({
        "rate": market.rate,
        "assets": product.underlyings.iter().zip(market.assets.iter()).map(|(u, a)| {
            serde_json::json!({
                "name": u.name,
                "spot": a.initial_value(),
                "vol": a.vol(),
                "underlyingType": format!("{:?}", u.underlying_type),
            })
        }).collect::<Vec<_>>(),
    });

    serde_json::to_string(&serde_json::json!({
        "productName": product.name,
        "notional": product.notional,
        "maturity": product.maturity,
        "underlyings": underlying_names,
        "price": price,
        "stderr": stderr,
        "greeks": greeks,
        "crossGreeks": cross_greeks,
        "payoffProfile": payoff_profile,
        "market": market_snapshot,
        "error": error,
    }))
    .unwrap()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_symbols_from_source(source: &str) -> analysis::SymbolTable {
    let (ast, _, _) = analysis::parse_and_diagnose(source);
    ast.as_ref()
        .map(|a| analysis::build_symbol_table(a, source))
        .unwrap_or_default()
}

fn build_default_market(num_underlyings: usize) -> MultiAssetMarket {
    let assets: Vec<AssetMarketData> = (0..num_underlyings)
        .map(|_| AssetMarketData::Equity {
            spot: 100.0,
            vol: 0.20,
            dividend_yield: 0.02,
        })
        .collect();
    let correlation = identity_correlation(num_underlyings);
    MultiAssetMarket {
        assets,
        correlation,
        rate: 0.05,
    }
}

fn pad_market(market: &mut MultiAssetMarket, num_underlyings: usize) {
    while market.assets.len() < num_underlyings {
        market.assets.push(AssetMarketData::Equity {
            spot: 100.0,
            vol: 0.20,
            dividend_yield: 0.02,
        });
    }
    let n = market.assets.len();
    if market.correlation.len() < n {
        market.correlation = identity_correlation(n);
    }
}

fn identity_correlation(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}
