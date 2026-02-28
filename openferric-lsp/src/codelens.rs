use openferric::dsl::engine::DslMonteCarloEngine;
use openferric::dsl::ir::CompiledProduct;
use openferric::dsl::market::{AssetData, MultiAssetMarket};
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
    // Build market data.
    let market = build_market(product.num_underlyings, market_json);
    let market = match market {
        Some(m) => m,
        None => return vec![],
    };

    let engine = DslMonteCarloEngine::new(
        pricing_cfg.num_paths as usize,
        pricing_cfg.num_steps as usize,
        pricing_cfg.seed,
    );

    let mut lenses = Vec::new();
    let product_line_pos = offset_to_position(source, product_span_start);
    let range = Range {
        start: product_line_pos,
        end: product_line_pos,
    };

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
                    greeks.delta,
                    greeks.gamma,
                    greeks.vega,
                    greeks.rho,
                ),
                command: String::new(),
                arguments: None,
            }),
            data: None,
        });
    }

    lenses
}

fn build_market(num_underlyings: usize, market_json: Option<&serde_json::Value>) -> Option<MultiAssetMarket> {
    // Try to parse from config.
    if let Some(json) = market_json
        && let Ok(mut market) = serde_json::from_value::<MultiAssetMarket>(json.clone())
    {
        // Pad assets if the product needs more.
        while market.assets.len() < num_underlyings {
            market.assets.push(AssetData {
                spot: 100.0,
                vol: 0.20,
                dividend_yield: 0.02,
            });
        }
        // Pad correlation matrix.
        let n = market.assets.len();
        if market.correlation.len() < n {
            market.correlation = identity_correlation(n);
        }
        return Some(market);
    }

    // Default market.
    let assets: Vec<AssetData> = (0..num_underlyings)
        .map(|_| AssetData {
            spot: 100.0,
            vol: 0.20,
            dividend_yield: 0.02,
        })
        .collect();
    let correlation = identity_correlation(num_underlyings);

    Some(MultiAssetMarket {
        assets,
        correlation,
        rate: 0.05,
    })
}

fn identity_correlation(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}
