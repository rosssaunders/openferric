use openferric::pricing::european::{black_scholes_greeks, black_scholes_price};
use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, parse_option_type, req_bool, req_f64, req_matrix_f64};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "correlation_matrix",
            description: "Compute sample Pearson correlation matrix from returns matrix.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": { "type": "number" }
                        }
                    }
                },
                "required": ["returns"]
            }),
        },
        ToolSpec {
            name: "black_scholes",
            description: "Black-Scholes option price and Greeks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "is_call": {"type": "boolean"}
                },
                "required": ["spot", "strike", "rate", "vol", "time", "is_call"]
            }),
        },
        ToolSpec {
            name: "normal_cdf",
            description: "Standard normal cumulative distribution function.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                },
                "required": ["x"]
            }),
        },
        ToolSpec {
            name: "normal_pdf",
            description: "Standard normal probability density function.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                },
                "required": ["x"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "correlation_matrix" => Some(correlation_matrix(args)),
        "black_scholes" => Some(black_scholes(args)),
        "normal_cdf" => Some(normal_cdf(args)),
        "normal_pdf" => Some(normal_pdf(args)),
        _ => None,
    }
}

fn correlation_matrix(args: &Value) -> ToolCallResult {
    let returns = req_matrix_f64(args, "returns")?;
    if returns.is_empty() {
        return Err("returns cannot be empty".to_string());
    }

    let n_obs = returns[0].len();
    if n_obs < 2 {
        return Err("each returns series must contain at least two observations".to_string());
    }
    if returns.iter().any(|row| row.len() != n_obs) {
        return Err("all returns series must have the same length".to_string());
    }

    let n_assets = returns.len();
    let means = returns
        .iter()
        .map(|series| series.iter().sum::<f64>() / n_obs as f64)
        .collect::<Vec<_>>();

    let mut matrix = vec![vec![0.0; n_assets]; n_assets];
    for i in 0..n_assets {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n_assets {
            let mut cov = 0.0;
            let mut var_i = 0.0;
            let mut var_j = 0.0;
            for (ri, rj) in returns[i].iter().zip(returns[j].iter()) {
                let di = *ri - means[i];
                let dj = *rj - means[j];
                cov += di * dj;
                var_i += di * di;
                var_j += dj * dj;
            }

            let denom = (var_i * var_j).sqrt();
            let rho = if denom > 0.0 { cov / denom } else { 0.0 }.clamp(-1.0, 1.0);
            matrix[i][j] = rho;
            matrix[j][i] = rho;
        }
    }

    Ok(json!({ "matrix": matrix }))
}

fn black_scholes(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let option_type = parse_option_type(req_bool(args, "is_call")?);

    let price = black_scholes_price(option_type, spot, strike, rate, vol, time);
    let greeks = black_scholes_greeks(option_type, spot, strike, rate, vol, time);

    Ok(json!({
        "price": price,
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega": greeks.vega,
        "theta": greeks.theta,
        "rho": greeks.rho
    }))
}

fn normal_cdf(args: &Value) -> ToolCallResult {
    let x = req_f64(args, "x")?;
    Ok(json!({ "value": openferric::math::normal_cdf(x) }))
}

fn normal_pdf(args: &Value) -> ToolCallResult {
    let x = req_f64(args, "x")?;
    Ok(json!({ "value": openferric::math::normal_pdf(x) }))
}
