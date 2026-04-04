use openferric::credit::cds_option::{CdsOption, risky_annuity};
use openferric::credit::{CdoTranche, Cds, CdsIndex, SurvivalCurve, SyntheticCdo};
use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, curve_from_value, req_array_f64, req_f64, req_value};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "cds_price",
            description: "Price a single-name CDS from spread, recovery, and discount curve.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spread": {"type": "number"},
                    "maturity": {"type": "number"},
                    "recovery": {"type": "number"},
                    "yield_curve": {"type": "object"},
                    "notional": {"type": "number"}
                },
                "required": ["spread", "maturity", "recovery", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "cds_index_price",
            description: "Price a CDS index as weighted single-name constituents.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spreads": {"type": "array", "items": {"type": "number"}},
                    "weights": {"type": "array", "items": {"type": "number"}},
                    "maturity": {"type": "number"},
                    "recovery": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["spreads", "weights", "maturity", "recovery", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "cds_option_price",
            description: "Price a CDS spread option via Black model on forward spread.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "forward_spread": {"type": "number"},
                    "strike": {"type": "number"},
                    "vol": {"type": "number"},
                    "maturity": {"type": "number"},
                    "recovery": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["forward_spread", "strike", "vol", "maturity", "recovery", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "survival_curve_build",
            description: "Bootstrap survival curve from tenor-spread CDS quotes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tenors": {"type": "array"},
                    "spreads": {"type": "array", "items": {"type": "number"}},
                    "recovery": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["tenors", "spreads", "recovery", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "cdo_tranche_price",
            description: "Price a synthetic CDO tranche under LHP Gaussian model.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "attachment": {"type": "number"},
                    "detachment": {"type": "number"},
                    "spreads": {"type": "array", "items": {"type": "number"}},
                    "correlation": {"type": "number"},
                    "maturity": {"type": "number"},
                    "recovery": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["attachment", "detachment", "spreads", "correlation", "maturity", "recovery", "yield_curve"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "cds_price" => Some(cds_price(args)),
        "cds_index_price" => Some(cds_index_price(args)),
        "cds_option_price" => Some(cds_option_price(args)),
        "survival_curve_build" => Some(survival_curve_build(args)),
        "cdo_tranche_price" => Some(cdo_tranche_price(args)),
        _ => None,
    }
}

fn cds_price(args: &Value) -> ToolCallResult {
    let spread = req_f64(args, "spread")?;
    let maturity = req_f64(args, "maturity")?;
    let recovery = req_f64(args, "recovery")?;
    let curve = curve_from_value(req_value(args, "yield_curve")?)?;
    let notional = super::opt_f64(args, "notional", 1_000_000.0)?;

    let hazard = if (1.0 - recovery).abs() > 1.0e-12 {
        (spread / (1.0 - recovery)).max(0.0)
    } else {
        0.0
    };
    let survival = SurvivalCurve::from_piecewise_hazard(&[maturity], &[hazard]);

    let cds = Cds {
        notional,
        spread,
        maturity,
        recovery_rate: recovery,
        payment_freq: 4,
    };

    let price = cds.npv(&curve, &survival);
    let bump = 1.0e-4;
    let cds_up = Cds {
        spread: spread + bump,
        ..cds.clone()
    };
    let cds_dn = Cds {
        spread: (spread - bump).max(0.0),
        ..cds.clone()
    };
    let dv01 = (cds_up.npv(&curve, &survival) - cds_dn.npv(&curve, &survival)) / 2.0;
    let spread_duration = if notional.abs() > 1.0e-12 {
        -dv01 / notional
    } else {
        0.0
    };

    Ok(json!({
        "price": price,
        "dv01": dv01,
        "spread_duration": spread_duration
    }))
}

fn cds_index_price(args: &Value) -> ToolCallResult {
    let spreads = req_array_f64(args, "spreads")?;
    let weights = req_array_f64(args, "weights")?;
    let maturity = req_f64(args, "maturity")?;
    let recovery = req_f64(args, "recovery")?;
    let curve = curve_from_value(req_value(args, "yield_curve")?)?;

    if spreads.len() != weights.len() {
        return Err("spreads and weights length mismatch".to_string());
    }
    if spreads.is_empty() {
        return Err("spreads cannot be empty".to_string());
    }

    let constituents = spreads
        .iter()
        .map(|s| Cds {
            notional: 1.0,
            spread: *s,
            maturity,
            recovery_rate: recovery,
            payment_freq: 4,
        })
        .collect::<Vec<_>>();

    let curves = spreads
        .iter()
        .map(|s| {
            let h = if (1.0 - recovery).abs() > 1.0e-12 {
                (*s / (1.0 - recovery)).max(0.0)
            } else {
                0.0
            };
            SurvivalCurve::from_piecewise_hazard(&[maturity], &[h])
        })
        .collect::<Vec<_>>();

    let index = CdsIndex {
        constituents,
        weights,
    };

    Ok(json!({ "price": index.npv(&curve, &curves) }))
}

fn cds_option_price(args: &Value) -> ToolCallResult {
    let forward_spread = req_f64(args, "forward_spread")?;
    let strike = req_f64(args, "strike")?;
    let vol = req_f64(args, "vol")?;
    let maturity = req_f64(args, "maturity")?;
    let recovery = req_f64(args, "recovery")?;
    let curve = curve_from_value(req_value(args, "yield_curve")?)?;

    let rf = curve.zero_rate(maturity.max(1.0e-6));
    let hazard = if (1.0 - recovery).abs() > 1.0e-12 {
        (forward_spread / (1.0 - recovery)).max(0.0)
    } else {
        0.0
    };
    let rpv01 = risky_annuity(4, maturity, hazard, rf, recovery);

    let option = CdsOption {
        notional: 1.0,
        strike_spread: strike,
        option_expiry: maturity,
        cds_maturity: maturity,
        is_payer: true,
        recovery_rate: recovery,
    };

    let price = option.black_price(forward_spread, vol, rpv01);
    Ok(json!({ "price": price }))
}

fn survival_curve_build(args: &Value) -> ToolCallResult {
    let tenors = super::tenors_from_value(args, "tenors")?;
    let spreads = req_array_f64(args, "spreads")?;
    let recovery = req_f64(args, "recovery")?;
    let curve = curve_from_value(req_value(args, "yield_curve")?)?;

    if tenors.len() != spreads.len() {
        return Err("tenors/spreads length mismatch".to_string());
    }

    let quotes = tenors.into_iter().zip(spreads).collect::<Vec<_>>();
    let survival = SurvivalCurve::bootstrap_from_cds_spreads(&quotes, recovery, 4, &curve);

    let probs = survival.tenors.iter().map(|(_, p)| *p).collect::<Vec<_>>();
    Ok(json!({
        "survival_probs": probs,
        "curve_json": serde_json::to_value(survival).unwrap_or(Value::Null)
    }))
}

fn cdo_tranche_price(args: &Value) -> ToolCallResult {
    let attachment = req_f64(args, "attachment")?;
    let detachment = req_f64(args, "detachment")?;
    let spreads = req_array_f64(args, "spreads")?;
    let correlation = req_f64(args, "correlation")?;
    let maturity = req_f64(args, "maturity")?;
    let recovery = req_f64(args, "recovery")?;
    let curve = curve_from_value(req_value(args, "yield_curve")?)?;

    if spreads.is_empty() {
        return Err("spreads cannot be empty".to_string());
    }

    let avg_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;

    let model = SyntheticCdo {
        num_names: spreads.len(),
        pool_spread: avg_spread.max(0.0),
        recovery_rate: recovery,
        correlation,
        risk_free_rate: curve.zero_rate(maturity.max(1.0e-6)),
        maturity,
        payment_freq: 4,
    };

    let tranche = CdoTranche {
        attachment,
        detachment,
        notional: (detachment - attachment).max(0.0),
        spread: avg_spread,
    };

    Ok(json!({ "price": model.npv(&tranche) }))
}
