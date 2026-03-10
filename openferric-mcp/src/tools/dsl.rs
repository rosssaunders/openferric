use std::fs;
use std::path::PathBuf;

use openferric::dsl::{DslMonteCarloEngine, MultiAssetMarket, parse_and_compile};
use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, obj, opt_usize, req_str};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "dsl_price",
            description: "Compile DSL source and price it via multi-asset Monte Carlo.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string" },
                    "market": {
                        "type": "object",
                        "properties": {
                            "assets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "spot": { "type": "number" },
                                        "vol": { "type": "number" },
                                        "dividend_yield": { "type": "number" }
                                    },
                                    "required": ["spot", "vol"]
                                }
                            },
                            "correlation": { "type": "array" },
                            "rate": { "type": "number" }
                        }
                    },
                    "num_paths": { "type": "integer", "minimum": 1 },
                    "seed": { "type": "integer", "minimum": 0 }
                },
                "required": ["source"]
            }),
        },
        ToolSpec {
            name: "dsl_greeks",
            description: "Compile DSL source and compute Greeks for one underlying.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string" },
                    "market": { "type": "object" },
                    "asset_index": { "type": "integer", "minimum": 0 },
                    "num_paths": { "type": "integer", "minimum": 1 },
                    "seed": { "type": "integer", "minimum": 0 }
                },
                "required": ["source"]
            }),
        },
        ToolSpec {
            name: "dsl_compile",
            description: "Compile DSL source into IR JSON and report compilation errors.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string" }
                },
                "required": ["source"]
            }),
        },
        ToolSpec {
            name: "dsl_examples",
            description: "List bundled DSL examples from examples/dsl/.",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "dsl_price" => Some(dsl_price(args)),
        "dsl_greeks" => Some(dsl_greeks(args)),
        "dsl_compile" => Some(dsl_compile(args)),
        "dsl_examples" => Some(dsl_examples(args)),
        _ => None,
    }
}

fn dsl_price(args: &Value) -> ToolCallResult {
    let source = req_str(args, "source")?;
    let product = parse_and_compile(source).map_err(|e| e.to_string())?;

    let market = parse_market(args)?;
    let num_paths = opt_usize(args, "num_paths", 20_000)?;
    let seed = opt_usize(args, "seed", 42)? as u64;

    let engine = DslMonteCarloEngine::new(num_paths, 252, seed);
    let price = engine
        .price_multi_asset(&product, &market)
        .map_err(|e| e.to_string())?;

    let greeks = engine
        .greeks_multi_asset(&product, &market, 0)
        .map(|g| {
            json!({
                "delta": g.delta,
                "gamma": g.gamma,
                "vega": g.vega,
                "theta": g.theta,
                "rho": g.rho
            })
        })
        .unwrap_or_else(|_| Value::Null);

    Ok(json!({
        "price": price.price,
        "stderr": price.stderr,
        "greeks": greeks
    }))
}

fn dsl_greeks(args: &Value) -> ToolCallResult {
    let source = req_str(args, "source")?;
    let product = parse_and_compile(source).map_err(|e| e.to_string())?;

    let market = parse_market(args)?;
    let asset_index = opt_usize(args, "asset_index", 0)?;
    let num_paths = opt_usize(args, "num_paths", 20_000)?;
    let seed = opt_usize(args, "seed", 42)? as u64;

    let engine = DslMonteCarloEngine::new(num_paths, 252, seed);
    let greeks = engine
        .greeks_multi_asset(&product, &market, asset_index)
        .map_err(|e| e.to_string())?;

    Ok(json!({
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega": greeks.vega,
        "theta": greeks.theta,
        "rho": greeks.rho
    }))
}

fn dsl_compile(args: &Value) -> ToolCallResult {
    let source = req_str(args, "source")?;

    match parse_and_compile(source) {
        Ok(product) => Ok(json!({
            "product_json": serde_json::to_value(product).unwrap_or(Value::Null),
            "errors": []
        })),
        Err(e) => Ok(json!({
            "product_json": Value::Null,
            "errors": [e.to_string()]
        })),
    }
}

fn dsl_examples(_args: &Value) -> ToolCallResult {
    let mut dir = PathBuf::from("examples");
    dir.push("dsl");

    let mut files = fs::read_dir(&dir)
        .map_err(|e| format!("failed to read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|x| x.to_str()) == Some("of"))
        .collect::<Vec<_>>();

    files.sort_by_key(|entry| entry.path());

    let examples = files
        .into_iter()
        .map(|entry| {
            let path = entry.path();
            let source = fs::read_to_string(&path).unwrap_or_default();
            let name = path
                .file_stem()
                .and_then(|x| x.to_str())
                .unwrap_or("example")
                .to_string();

            let description = source
                .lines()
                .find(|line| line.trim_start().starts_with("product "))
                .map(|line| line.trim().to_string())
                .unwrap_or_else(|| "DSL product example".to_string());

            json!({
                "name": name,
                "source": source,
                "description": description
            })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(examples))
}

fn parse_market(args: &Value) -> Result<MultiAssetMarket, String> {
    let arg_obj = obj(args, "arguments")?;

    let Some(market_val) = arg_obj.get("market") else {
        return Ok(MultiAssetMarket::single(100.0, 0.2, 0.03, 0.0));
    };

    let market_obj = market_val
        .as_object()
        .ok_or_else(|| "market must be an object".to_string())?;

    let assets_val = market_obj
        .get("assets")
        .and_then(Value::as_array)
        .ok_or_else(|| "market.assets must be an array".to_string())?;

    if assets_val.is_empty() {
        return Err("market.assets cannot be empty".to_string());
    }

    let mut assets = Vec::with_capacity(assets_val.len());
    for asset in assets_val {
        let a = asset
            .as_object()
            .ok_or_else(|| "market.assets entries must be objects".to_string())?;

        let spot = a
            .get("spot")
            .and_then(Value::as_f64)
            .ok_or_else(|| "market asset `spot` must be numeric".to_string())?;
        let vol = a
            .get("vol")
            .and_then(Value::as_f64)
            .ok_or_else(|| "market asset `vol` must be numeric".to_string())?;
        let dividend_yield = a
            .get("dividend_yield")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);

        assets.push(openferric::dsl::AssetMarketData::Equity {
            spot,
            vol,
            dividend_yield,
        });
    }

    let n = assets.len();
    let correlation = if let Some(rows) = market_obj.get("correlation").and_then(Value::as_array) {
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
        let mut corr = vec![vec![0.0; n]; n];
        for (i, row) in corr.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        corr
    };

    let rate = market_obj
        .get("rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.03);

    Ok(MultiAssetMarket {
        assets,
        correlation,
        rate,
    })
}
