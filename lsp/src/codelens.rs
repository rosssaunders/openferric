use openferric::dsl::engine::DslMonteCarloEngine;
use openferric::dsl::ir::CompiledProduct;
use openferric::dsl::market::{AssetMarketData, MultiAssetMarket};
use tower_lsp::lsp_types::*;

use crate::backend::PricingConfig;
use crate::diagnostics::offset_to_position;

/// Generate CodeLens items showing live price and Greeks.
///
/// Takes decomposed document state (not a reference to DocumentState) so the
/// caller can release the document mutex before pricing runs.
pub fn code_lenses(
    product: &CompiledProduct,
    product_span_start: usize,
    source: &str,
    pricing_cfg: &PricingConfig,
    market_json: Option<&serde_json::Value>,
) -> Vec<CodeLens> {
    let product_line_pos = offset_to_position(source, product_span_start);
    let range = Range {
        start: product_line_pos,
        end: product_line_pos,
    };

    // Build market data. A malformed market config must surface as an error
    // lens rather than silently pricing against defaults.
    let market = match build_market(product.num_underlyings, market_json) {
        Ok(m) => m,
        Err(e) => {
            return vec![CodeLens {
                range,
                command: Some(Command {
                    title: format!("Market data error: {e}"),
                    command: String::new(),
                    arguments: None,
                }),
                data: None,
            }];
        }
    };

    let engine = DslMonteCarloEngine::new(
        pricing_cfg.num_paths as usize,
        pricing_cfg.num_steps as usize,
        pricing_cfg.seed,
    );

    let mut lenses = Vec::new();

    // Price.
    match engine.price_multi_asset(product, &market) {
        Ok(result) => {
            let stderr_str = result
                .stderr
                .map(|s| format!(" | StdErr: {s:.4}"))
                .unwrap_or_default();
            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: format!("Price: {:.4}{stderr_str}", result.price),
                    command: String::new(),
                    arguments: None,
                }),
                data: None,
            });
        }
        Err(e) => {
            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: format!("Pricing error: {e}"),
                    command: String::new(),
                    arguments: None,
                }),
                data: None,
            });
        }
    }

    // Greeks for asset 0.
    if product.num_underlyings > 0
        && let Ok(greeks) = engine.greeks_multi_asset(product, &market, 0)
    {
        lenses.push(CodeLens {
            range,
            command: Some(Command {
                title: format!(
                    "Delta: {:.4} | Gamma: {:.4} | Vega: {:.4} | Rho: {:.4}",
                    greeks.delta, greeks.gamma, greeks.vega, greeks.rho,
                ),
                command: String::new(),
                arguments: None,
            }),
            data: None,
        });
    }

    lenses
}

/// Build a `MultiAssetMarket` from the configured market JSON, or defaults
/// when no market is configured.
///
/// Returns `Err` when a market config is present but cannot be parsed, so
/// callers can surface the problem instead of silently pricing on defaults.
pub fn build_market(
    num_underlyings: usize,
    market_json: Option<&serde_json::Value>,
) -> Result<MultiAssetMarket, String> {
    if let Some(json) = market_json {
        let mut market = parse_market_json(json)?;
        pad_market(&mut market, num_underlyings);
        return Ok(market);
    }

    // Default market.
    Ok(default_market(num_underlyings))
}

fn default_market(num_underlyings: usize) -> MultiAssetMarket {
    let assets: Vec<AssetMarketData> = (0..num_underlyings)
        .map(|_| default_equity_asset())
        .collect();
    let correlation = identity_correlation(num_underlyings);

    MultiAssetMarket {
        assets,
        correlation,
        rate: 0.05,
    }
}

fn default_equity_asset() -> AssetMarketData {
    AssetMarketData::Equity {
        spot: 100.0,
        vol: 0.20,
        dividend_yield: 0.02,
    }
}

/// Parse a market JSON value, accepting both the serde-tagged form
/// (`{"type": "Equity", ...}` assets) and untagged equity-shaped assets
/// (`{"spot": ..., "vol": ..., "dividend_yield": ...}`) for compatibility
/// with simple clients.
fn parse_market_json(json: &serde_json::Value) -> Result<MultiAssetMarket, String> {
    let tagged_err = match serde_json::from_value::<MultiAssetMarket>(json.clone()) {
        Ok(market) => return Ok(market),
        Err(e) => e,
    };

    // Fallback: manual parse of untagged equity-shaped assets.
    let market_obj = json
        .as_object()
        .ok_or_else(|| "market must be a JSON object".to_string())?;

    let assets_val = market_obj
        .get("assets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("invalid market JSON: {tagged_err}"))?;

    let mut assets = Vec::with_capacity(assets_val.len());
    for asset in assets_val {
        let a = asset
            .as_object()
            .ok_or_else(|| "market.assets entries must be objects".to_string())?;
        let spot = a
            .get("spot")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| "market asset `spot` must be numeric".to_string())?;
        let vol = a
            .get("vol")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| "market asset `vol` must be numeric".to_string())?;
        let dividend_yield = a
            .get("dividend_yield")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);

        assets.push(AssetMarketData::Equity {
            spot,
            vol,
            dividend_yield,
        });
    }

    if assets.is_empty() {
        return Err("market.assets cannot be empty".to_string());
    }

    let n = assets.len();
    let correlation = if let Some(rows) = market_obj
        .get("correlation")
        .and_then(serde_json::Value::as_array)
    {
        rows.iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| "market.correlation must be a matrix".to_string())?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| "market.correlation entries must be numbers".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        identity_correlation(n)
    };

    let rate = market_obj
        .get("rate")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.05);

    Ok(MultiAssetMarket {
        assets,
        correlation,
        rate,
    })
}

/// Pad assets and the correlation matrix up to `num_underlyings`, preserving
/// any user-provided correlation entries in the top-left block.
fn pad_market(market: &mut MultiAssetMarket, num_underlyings: usize) {
    while market.assets.len() < num_underlyings {
        market.assets.push(default_equity_asset());
    }
    let n = market.assets.len();
    if market.correlation.len() < n {
        market.correlation = padded_correlation(&market.correlation, n);
    }
}

/// Expand `existing` into an `n x n` correlation matrix: the top-left block is
/// preserved, all new entries are identity (1 on the diagonal, 0 elsewhere).
fn padded_correlation(existing: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| match existing.get(i).and_then(|row| row.get(j)) {
                    Some(&v) => v,
                    None => {
                        if i == j {
                            1.0
                        } else {
                            0.0
                        }
                    }
                })
                .collect()
        })
        .collect()
}

fn identity_correlation(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}
