use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, opt_f64, opt_str, opt_usize, req_str};

const API_BASE: &str = "https://www.deribit.com/api/v2/public";

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "get_spot",
            description: "Fetch Deribit index spot for a currency.",
            input_schema: json!({
                "type":"object",
                "properties": {"currency": {"type":"string"}},
                "required": ["currency"]
            }),
        },
        ToolSpec {
            name: "get_ticker",
            description: "Fetch Deribit ticker for a specific instrument.",
            input_schema: json!({
                "type":"object",
                "properties": {"instrument_name": {"type":"string"}},
                "required": ["instrument_name"]
            }),
        },
        ToolSpec {
            name: "list_instruments",
            description: "List Deribit instruments for a currency and kind.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "currency": {"type":"string"},
                    "kind": {"type":"string"}
                },
                "required": ["currency"]
            }),
        },
        ToolSpec {
            name: "get_orderbook",
            description: "Fetch Deribit order book snapshot for instrument.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "instrument_name": {"type":"string"},
                    "depth": {"type":"integer", "minimum": 1}
                },
                "required": ["instrument_name"]
            }),
        },
        ToolSpec {
            name: "get_historical_volatility",
            description: "Fetch Deribit historical volatility series.",
            input_schema: json!({
                "type":"object",
                "properties": {"currency": {"type":"string"}},
                "required": ["currency"]
            }),
        },
        ToolSpec {
            name: "get_vol_index",
            description: "Fetch Deribit volatility index OHLC data.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "currency": {"type":"string"},
                    "start": {},
                    "end": {},
                    "resolution": {"type":"string"}
                },
                "required": ["currency"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "get_spot" => Some(get_spot(args)),
        "get_ticker" => Some(get_ticker(args)),
        "list_instruments" => Some(list_instruments(args)),
        "get_orderbook" => Some(get_orderbook(args)),
        "get_historical_volatility" => Some(get_historical_volatility(args)),
        "get_vol_index" => Some(get_vol_index(args)),
        _ => None,
    }
}

fn get_spot(args: &Value) -> ToolCallResult {
    let currency = req_str(args, "currency")?.to_ascii_lowercase();
    let index_name = format!("{currency}_usd");
    let url = format!("{API_BASE}/get_index_price?index_name={index_name}");
    let payload = fetch_json(&url)?;
    let result = payload
        .get("result")
        .ok_or_else(|| "missing `result` in Deribit response".to_string())?;

    Ok(json!({
        "spot": result.get("index_price").and_then(Value::as_f64).unwrap_or(f64::NAN),
        "timestamp": result
            .get("timestamp")
            .and_then(Value::as_i64)
            .or_else(|| payload.get("usOut").and_then(Value::as_i64))
    }))
}

fn get_ticker(args: &Value) -> ToolCallResult {
    let instrument_name = req_str(args, "instrument_name")?;
    let url = format!("{API_BASE}/ticker?instrument_name={instrument_name}");
    let payload = fetch_json(&url)?;
    let result = payload
        .get("result")
        .ok_or_else(|| "missing `result` in Deribit response".to_string())?;

    Ok(json!({
        "mark_price": result.get("mark_price").and_then(Value::as_f64),
        "iv": result.get("mark_iv").and_then(Value::as_f64),
        "greeks": result.get("greeks").cloned().unwrap_or(Value::Null),
        "underlying_price": result.get("underlying_price").and_then(Value::as_f64),
        "volume": result.get("stats").and_then(|s| s.get("volume")).and_then(Value::as_f64)
            .or_else(|| result.get("volume").and_then(Value::as_f64))
    }))
}

fn list_instruments(args: &Value) -> ToolCallResult {
    let currency = req_str(args, "currency")?;
    let kind = opt_str(args, "kind").unwrap_or("option");

    let url = format!(
        "{API_BASE}/get_instruments?currency={}&kind={}&expired=false",
        currency.to_ascii_uppercase(),
        kind
    );

    let payload = fetch_json(&url)?;
    let list = payload
        .get("result")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing `result` array in Deribit response".to_string())?;

    let instruments = list
        .iter()
        .map(|inst| {
            json!({
                "name": inst.get("instrument_name").and_then(Value::as_str),
                "strike": inst.get("strike").and_then(Value::as_f64),
                "expiry": inst.get("expiration_timestamp").and_then(Value::as_i64),
                "type": inst.get("option_type").and_then(Value::as_str)
                    .or_else(|| inst.get("kind").and_then(Value::as_str))
            })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(instruments))
}

fn get_orderbook(args: &Value) -> ToolCallResult {
    let instrument_name = req_str(args, "instrument_name")?;
    let depth = opt_usize(args, "depth", 20)?;

    let url = format!("{API_BASE}/get_order_book?instrument_name={instrument_name}&depth={depth}");
    let payload = fetch_json(&url)?;
    let result = payload
        .get("result")
        .ok_or_else(|| "missing `result` in Deribit response".to_string())?;

    Ok(json!({
        "bids": result.get("bids").cloned().unwrap_or(Value::Array(Vec::new())),
        "asks": result.get("asks").cloned().unwrap_or(Value::Array(Vec::new())),
        "mark_price": result.get("mark_price").and_then(Value::as_f64)
    }))
}

fn get_historical_volatility(args: &Value) -> ToolCallResult {
    let currency = req_str(args, "currency")?;
    let url = format!(
        "{API_BASE}/get_historical_volatility?currency={}",
        currency.to_ascii_uppercase()
    );

    let payload = fetch_json(&url)?;
    let data = payload
        .get("result")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing `result` array in Deribit response".to_string())?;

    let out = data
        .iter()
        .filter_map(|row| {
            row.as_array().and_then(|arr| {
                if arr.len() >= 2 {
                    Some(json!({
                        "timestamp": arr[0].as_i64(),
                        "vol": arr[1].as_f64()
                    }))
                } else {
                    None
                }
            })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(out))
}

fn get_vol_index(args: &Value) -> ToolCallResult {
    let currency = req_str(args, "currency")?;
    let start = opt_f64(args, "start", 0.0)?;
    let end = opt_f64(args, "end", 0.0)?;
    let resolution = opt_str(args, "resolution").unwrap_or("1D");

    let mut url = format!(
        "{API_BASE}/get_volatility_index_data?currency={}&resolution={resolution}",
        currency.to_ascii_uppercase()
    );
    if start > 0.0 {
        url.push_str(&format!("&start_timestamp={}", start as i64));
    }
    if end > 0.0 {
        url.push_str(&format!("&end_timestamp={}", end as i64));
    }

    let payload = fetch_json(&url)?;
    let result = payload
        .get("result")
        .ok_or_else(|| "missing `result` in Deribit response".to_string())?;

    let rows = if let Some(arr) = result.as_array() {
        arr.clone()
    } else if let Some(arr) = result.get("data").and_then(Value::as_array) {
        arr.clone()
    } else {
        Vec::new()
    };

    let out = rows
        .iter()
        .filter_map(|row| {
            row.as_array().and_then(|arr| {
                if arr.len() >= 5 {
                    Some(json!({
                        "timestamp": arr[0].as_i64(),
                        "open": arr[1].as_f64(),
                        "high": arr[2].as_f64(),
                        "low": arr[3].as_f64(),
                        "close": arr[4].as_f64()
                    }))
                } else {
                    None
                }
            })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(out))
}

fn fetch_json(url: &str) -> Result<Value, String> {
    let mut response = ureq::get(url)
        .call()
        .map_err(|e| format!("request failed for `{url}`: {e}"))?;

    let payload: Value = response
        .body_mut()
        .read_json()
        .map_err(|e| format!("invalid JSON response from `{url}`: {e}"))?;

    if let Some(error) = payload.get("error") {
        return Err(format!("Deribit error: {error}"));
    }

    Ok(payload)
}
