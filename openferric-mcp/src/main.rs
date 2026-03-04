use serde_json::{Map, Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};

mod tools;

const SERVER_NAME: &str = "openferric-mcp";
const SERVER_VERSION: &str = "0.1.0";
const PROTOCOL_VERSION: &str = "2024-11-05";

#[tokio::main]
async fn main() -> io::Result<()> {
    eprintln!("{SERVER_NAME} starting on stdio");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut lines = BufReader::new(stdin).lines();
    let mut out = BufWriter::new(stdout);

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parsed = match serde_json::from_str::<Value>(line) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("invalid JSON: {err}");
                let resp = error_response(Value::Null, -32700, "parse error", None);
                write_response(&mut out, &resp).await?;
                continue;
            }
        };

        let responses = handle_incoming(parsed);
        for response in responses {
            write_response(&mut out, &response).await?;
        }
    }

    eprintln!("{SERVER_NAME} shutting down");
    Ok(())
}

fn handle_incoming(message: Value) -> Vec<Value> {
    if let Some(batch) = message.as_array() {
        if batch.is_empty() {
            return vec![error_response(
                Value::Null,
                -32600,
                "invalid request: empty batch",
                None,
            )];
        }

        let mut out = Vec::new();
        for item in batch {
            if let Some(resp) = handle_request(item) {
                out.push(resp);
            }
        }
        return out;
    }

    handle_request(&message).into_iter().collect()
}

fn handle_request(request: &Value) -> Option<Value> {
    let obj = match request.as_object() {
        Some(v) => v,
        None => {
            return Some(error_response(
                Value::Null,
                -32600,
                "invalid request: expected object",
                None,
            ));
        }
    };

    let id = obj.get("id").cloned();
    let method = match obj.get("method").and_then(Value::as_str) {
        Some(m) => m,
        None => {
            return id.map(|idv| {
                error_response(idv, -32600, "invalid request: missing string method", None)
            });
        }
    };

    let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));
    let response = match dispatch_method(method, &params) {
        Ok(result) => success_response(id.clone().unwrap_or(Value::Null), result),
        Err((code, message, data)) => {
            error_response(id.clone().unwrap_or(Value::Null), code, message, data)
        }
    };

    if id.is_none() { None } else { Some(response) }
}

fn dispatch_method(
    method: &str,
    params: &Value,
) -> Result<Value, (i64, &'static str, Option<Value>)> {
    match method {
        "initialize" => Ok(handle_initialize(params)),
        "tools/list" => Ok(handle_tools_list()),
        "tools/call" => handle_tools_call(params),
        "resources/list" => Ok(handle_resources_list()),
        "resources/read" => handle_resources_read(params),
        _ => Err((
            -32601,
            "method not found",
            Some(json!({ "method": method })),
        )),
    }
}

fn handle_initialize(_params: &Value) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "subscribe": false, "listChanged": false }
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "instructions": "OpenFerric quant MCP server exposing pricing, rates, credit, volatility, calibration, risk, models, DSL, and market data tools."
    })
}

fn handle_tools_list() -> Value {
    let tools = tools::all_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema
            })
        })
        .collect::<Vec<_>>();

    json!({ "tools": tools })
}

fn handle_tools_call(params: &Value) -> Result<Value, (i64, &'static str, Option<Value>)> {
    let params_obj = params
        .as_object()
        .ok_or((-32602, "invalid params: expected object", None))?;

    let name = params_obj.get("name").and_then(Value::as_str).ok_or((
        -32602,
        "invalid params: missing `name`",
        None,
    ))?;

    let arguments = params_obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    if !arguments.is_object() {
        return Err((
            -32602,
            "invalid params: `arguments` must be an object",
            None,
        ));
    }

    match tools::call_tool(name, &arguments) {
        Ok(value) => Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
            }],
            "structuredContent": value,
            "isError": false
        })),
        Err(message) => Ok(json!({
            "content": [{
                "type": "text",
                "text": message
            }],
            "isError": true
        })),
    }
}

fn handle_resources_list() -> Value {
    json!({
        "resources": [
            {
                "uri": "openferric://tools/catalog",
                "name": "Tool Catalog",
                "description": "JSON catalog of all OpenFerric MCP tools and schemas.",
                "mimeType": "application/json"
            },
            {
                "uri": "openferric://dsl/examples",
                "name": "DSL Examples",
                "description": "Bundled OpenFerric DSL example contracts.",
                "mimeType": "application/json"
            },
            {
                "uri": "openferric://server/info",
                "name": "Server Info",
                "description": "OpenFerric MCP server metadata and protocol info.",
                "mimeType": "application/json"
            }
        ]
    })
}

fn handle_resources_read(params: &Value) -> Result<Value, (i64, &'static str, Option<Value>)> {
    let params_obj = params
        .as_object()
        .ok_or((-32602, "invalid params: expected object", None))?;
    let uri = params_obj.get("uri").and_then(Value::as_str).ok_or((
        -32602,
        "invalid params: missing `uri`",
        None,
    ))?;

    let payload = match uri {
        "openferric://tools/catalog" => handle_tools_list(),
        "openferric://dsl/examples" => {
            let examples = tools::call_tool("dsl_examples", &json!({})).map_err(|e| {
                (
                    -32603,
                    "failed to load DSL examples",
                    Some(json!({ "error": e })),
                )
            })?;
            json!({ "examples": examples })
        }
        "openferric://server/info" => json!({
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocolVersion": PROTOCOL_VERSION
        }),
        _ => {
            return Err((
                -32602,
                "invalid params: unknown resource uri",
                Some(json!({ "uri": uri })),
            ));
        }
    };

    let text = serde_json::to_string_pretty(&payload)
        .map_err(|_| (-32603, "internal error: could not serialize resource", None))?;

    Ok(json!({
        "contents": [{
            "uri": uri,
            "mimeType": "application/json",
            "text": text
        }]
    }))
}

fn success_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn error_response(id: Value, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut err = Map::new();
    err.insert("code".to_string(), json!(code));
    err.insert("message".to_string(), json!(message));
    if let Some(data) = data {
        err.insert("data".to_string(), data);
    }

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": Value::Object(err)
    })
}

async fn write_response(out: &mut BufWriter<io::Stdout>, response: &Value) -> io::Result<()> {
    let text = serde_json::to_string(response).unwrap_or_else(|_| {
        r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"internal serialization error"}}"#
            .to_string()
    });
    out.write_all(text.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await
}
