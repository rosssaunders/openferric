use openferric::calibration::{
    Calibrator, HestonCalibrator, OptionVolQuote, SabrCalibrator, SviCalibrationParams,
    SviCalibrator, SviParameterization, SwaptionVolQuote,
};
use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, curve_from_value, req_array_f64, req_f64, req_value};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "calibrate_heston",
            description: "Calibrate Heston parameters to option vol grid.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "rate": {"type":"number"},
                    "market_vols": {"type":"array"},
                    "strikes": {"type":"array", "items": {"type":"number"}},
                    "expiries": {"type":"array", "items": {"type":"number"}}
                },
                "required": ["spot","rate","market_vols","strikes","expiries"]
            }),
        },
        ToolSpec {
            name: "calibrate_sabr",
            description: "Calibrate SABR parameters to one maturity smile.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "forward": {"type":"number"},
                    "expiry": {"type":"number"},
                    "strikes": {"type":"array", "items": {"type":"number"}},
                    "vols": {"type":"array", "items": {"type":"number"}}
                },
                "required": ["forward","expiry","strikes","vols"]
            }),
        },
        ToolSpec {
            name: "calibrate_hull_white",
            description: "Calibrate Hull-White (a, sigma) to swaption vol quotes.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "yield_curve": {"type":"object"},
                    "swaption_vols": {
                        "type":"array",
                        "items": {
                            "type":"object",
                            "properties": {
                                "expiry": {"type":"number"},
                                "tenor": {"type":"number"},
                                "vol": {"type":"number"}
                            },
                            "required": ["expiry","tenor","vol"]
                        }
                    }
                },
                "required": ["yield_curve","swaption_vols"]
            }),
        },
        ToolSpec {
            name: "calibrate_svi",
            description: "Calibrate SVI raw parameters from total-variance slice.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "forward": {"type":"number"},
                    "expiry": {"type":"number"},
                    "strikes": {"type":"array", "items": {"type":"number"}},
                    "total_variances": {"type":"array", "items": {"type":"number"}}
                },
                "required": ["forward","expiry","strikes","total_variances"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "calibrate_heston" => Some(calibrate_heston(args)),
        "calibrate_sabr" => Some(calibrate_sabr(args)),
        "calibrate_hull_white" => Some(calibrate_hull_white(args)),
        "calibrate_svi" => Some(calibrate_svi(args)),
        _ => None,
    }
}

fn calibrate_heston(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let rate = req_f64(args, "rate")?;
    let strikes = req_array_f64(args, "strikes")?;
    let expiries = req_array_f64(args, "expiries")?;
    let market_vols = req_value(args, "market_vols")?;

    if strikes.is_empty() || expiries.is_empty() {
        return Err("strikes and expiries cannot be empty".to_string());
    }

    let matrix = parse_vol_matrix(market_vols, expiries.len(), strikes.len())?;

    let mut quotes = Vec::with_capacity(expiries.len() * strikes.len());
    for (ei, t) in expiries.iter().enumerate() {
        for (si, k) in strikes.iter().enumerate() {
            quotes.push(OptionVolQuote::new(
                format!("T{ei}_K{si}"),
                *k,
                *t,
                matrix[ei][si],
            ));
        }
    }

    let calibrator = HestonCalibrator {
        spot,
        rate,
        dividend_yield: 0.0,
        ..HestonCalibrator::default()
    };

    let result = calibrator.calibrate(&quotes).map_err(|e| e.to_string())?;
    Ok(json!({
        "v0": result.params.v0,
        "kappa": result.params.kappa,
        "theta": result.params.theta,
        "sigma": result.params.sigma_v,
        "rho": result.params.rho,
        "fit_error": result.diagnostics.fit_quality.rmse
    }))
}

fn calibrate_sabr(args: &Value) -> ToolCallResult {
    let forward = req_f64(args, "forward")?;
    let expiry = req_f64(args, "expiry")?;
    let strikes = req_array_f64(args, "strikes")?;
    let vols = req_array_f64(args, "vols")?;

    if strikes.len() != vols.len() || strikes.is_empty() {
        return Err("strikes and vols must be non-empty and equal-length".to_string());
    }

    let quotes = strikes
        .iter()
        .zip(vols.iter())
        .enumerate()
        .map(|(i, (k, v))| OptionVolQuote::new(format!("q{i}"), *k, expiry, *v))
        .collect::<Vec<_>>();

    let calibrator = SabrCalibrator {
        forward,
        maturity: expiry,
        beta_pin: Some(0.5),
        use_global_search: false,
        ..SabrCalibrator::default()
    };

    let result = calibrator.calibrate(&quotes).map_err(|e| e.to_string())?;
    Ok(json!({
        "alpha": result.params.alpha,
        "beta": result.params.beta,
        "rho": result.params.rho,
        "nu": result.params.nu
    }))
}

fn calibrate_hull_white(args: &Value) -> ToolCallResult {
    let _curve = curve_from_value(req_value(args, "yield_curve")?)?;

    let quotes_val = req_value(args, "swaption_vols")?
        .as_array()
        .ok_or_else(|| "swaption_vols must be an array".to_string())?;

    let mut quotes = Vec::with_capacity(quotes_val.len());
    for (i, q) in quotes_val.iter().enumerate() {
        let obj_map = q
            .as_object()
            .ok_or_else(|| "swaption_vol entries must be objects".to_string())?;
        let expiry = obj_map
            .get("expiry")
            .and_then(Value::as_f64)
            .ok_or_else(|| "swaption_vol expiry must be numeric".to_string())?;
        let tenor = obj_map
            .get("tenor")
            .and_then(Value::as_f64)
            .ok_or_else(|| "swaption_vol tenor must be numeric".to_string())?;
        let vol = obj_map
            .get("vol")
            .and_then(Value::as_f64)
            .ok_or_else(|| "swaption_vol vol must be numeric".to_string())?;
        quotes.push(SwaptionVolQuote::new(format!("swp{i}"), expiry, tenor, vol));
    }

    let calibrator = openferric::calibration::HullWhiteCalibrator::default();
    let result = calibrator.calibrate(&quotes).map_err(|e| e.to_string())?;

    Ok(json!({
        "a": result.params.a,
        "sigma": result.params.sigma,
        "fit_error": result.diagnostics.fit_quality.rmse
    }))
}

fn calibrate_svi(args: &Value) -> ToolCallResult {
    let forward = req_f64(args, "forward")?;
    let expiry = req_f64(args, "expiry")?;
    let strikes = req_array_f64(args, "strikes")?;
    let total_variances = req_array_f64(args, "total_variances")?;

    if strikes.len() != total_variances.len() || strikes.is_empty() {
        return Err("strikes and total_variances must be non-empty and equal-length".to_string());
    }

    let vols = total_variances
        .iter()
        .map(|w| (w.max(1.0e-12) / expiry.max(1.0e-12)).sqrt())
        .collect::<Vec<_>>();

    let quotes = strikes
        .iter()
        .zip(vols.iter())
        .enumerate()
        .map(|(i, (k, v))| OptionVolQuote::new(format!("svi{i}"), *k, expiry, *v))
        .collect::<Vec<_>>();

    let calibrator = SviCalibrator {
        forward,
        maturity: expiry,
        parameterization: SviParameterization::Raw,
        use_global_search: false,
        ..SviCalibrator::default()
    };

    let result = calibrator.calibrate(&quotes).map_err(|e| e.to_string())?;

    match result.params {
        SviCalibrationParams::Raw(raw) => Ok(json!({
            "a": raw.a,
            "b": raw.b,
            "rho": raw.rho,
            "m": raw.m,
            "sigma": raw.sigma
        })),
        SviCalibrationParams::JumpWings(_) => {
            Err("unexpected SVI jump-wings result when raw requested".to_string())
        }
    }
}

fn parse_vol_matrix(
    value: &Value,
    n_expiries: usize,
    n_strikes: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if let Some(rows) = value.as_array() {
        if rows.len() == n_expiries && rows.first().and_then(Value::as_array).is_some() {
            let matrix = rows
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or_else(|| "market_vols must be a matrix".to_string())?
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| "market_vols entries must be numeric".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;

            if matrix.iter().any(|row| row.len() != n_strikes) {
                return Err("market_vols row length must match strikes length".to_string());
            }
            return Ok(matrix);
        }

        if rows.len() == n_expiries * n_strikes {
            let flat = rows
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| "market_vols entries must be numeric".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut out = vec![vec![0.0; n_strikes]; n_expiries];
            for e in 0..n_expiries {
                for s in 0..n_strikes {
                    out[e][s] = flat[e * n_strikes + s];
                }
            }
            return Ok(out);
        }
    }

    Err(
        "market_vols must be a matrix [expiry][strike] or flat vector of length expiries*strikes"
            .to_string(),
    )
}
