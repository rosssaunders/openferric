use openferric::core::Greeks;
use openferric::pricing::european::black_scholes_price;
use openferric::risk::{
    Portfolio, Position, XvaCalculator, delta_normal_var, historical_expected_shortfall,
    historical_var,
};
use serde_json::{Value, json};

use super::{
    ToolCallResult, ToolSpec, curve_from_value, obj, req_array, req_array_f64, req_f64, req_value,
    survival_curve_from_value,
};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "var_historical",
            description: "Historical VaR/CVaR/ES on return or PnL samples.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "returns": {"type":"array", "items": {"type":"number"}},
                    "confidence": {"type":"number"},
                    "horizon": {"type":"number"}
                },
                "required": ["returns", "confidence"]
            }),
        },
        ToolSpec {
            name: "var_parametric",
            description: "Parametric Gaussian VaR from mean/std/confidence.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "mean": {"type":"number"},
                    "std": {"type":"number"},
                    "confidence": {"type":"number"},
                    "horizon": {"type":"number"}
                },
                "required": ["mean", "std", "confidence"]
            }),
        },
        ToolSpec {
            name: "portfolio_risk",
            description: "Aggregate portfolio Greeks and simple delta-normal VaR.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "positions": {"type":"array"},
                    "market": {"type":"object"}
                },
                "required": ["positions", "market"]
            }),
        },
        ToolSpec {
            name: "scenario_analysis",
            description: "Run Greek-based scenario PnL over scenario set.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "positions": {"type":"array"},
                    "scenarios": {"type":"array"}
                },
                "required": ["positions", "scenarios"]
            }),
        },
        ToolSpec {
            name: "sensitivity_ladder",
            description: "Price/delta ladder for a European option over spot bumps.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "instrument": {"type":"object"},
                    "market": {"type":"object"},
                    "bump_sizes": {"type":"array", "items": {"type":"number"}}
                },
                "required": ["instrument", "market", "bump_sizes"]
            }),
        },
        ToolSpec {
            name: "cva",
            description: "Compute CVA from exposure profile and survival curve.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "exposure_profile": {"type":"object"},
                    "survival_curve": {"type":"object"},
                    "recovery": {"type":"number"}
                },
                "required": ["exposure_profile", "survival_curve", "recovery"]
            }),
        },
        ToolSpec {
            name: "dva",
            description: "Compute DVA from exposure profile and own survival curve.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "exposure_profile": {"type":"object"},
                    "own_survival": {"type":"object"},
                    "recovery": {"type":"number"}
                },
                "required": ["exposure_profile", "own_survival", "recovery"]
            }),
        },
        ToolSpec {
            name: "fva",
            description: "Compute FVA from exposure profile and funding curve.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "exposure_profile": {"type":"object"},
                    "funding_curve": {"type":"object"}
                },
                "required": ["exposure_profile", "funding_curve"]
            }),
        },
        ToolSpec {
            name: "xva_all",
            description: "Compute CVA/DVA/FVA and total XVA.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "exposure_profile": {"type":"object"},
                    "counterparty_curve": {"type":"object"},
                    "own_curve": {"type":"object"},
                    "funding_curve": {"type":"object"},
                    "recovery": {"type":"number"}
                },
                "required": ["exposure_profile", "counterparty_curve", "own_curve", "funding_curve", "recovery"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "var_historical" => Some(var_historical(args)),
        "var_parametric" => Some(var_parametric(args)),
        "portfolio_risk" => Some(portfolio_risk(args)),
        "scenario_analysis" => Some(scenario_analysis(args)),
        "sensitivity_ladder" => Some(sensitivity_ladder(args)),
        "cva" => Some(cva(args)),
        "dva" => Some(dva(args)),
        "fva" => Some(fva(args)),
        "xva_all" => Some(xva_all(args)),
        _ => None,
    }
}

fn var_historical(args: &Value) -> ToolCallResult {
    let returns = req_array_f64(args, "returns")?;
    let confidence = req_f64(args, "confidence")?;
    let horizon = super::opt_f64(args, "horizon", 1.0)?;

    let var = historical_var(&returns, confidence) * horizon.sqrt();
    let es = historical_expected_shortfall(&returns, confidence) * horizon.sqrt();

    Ok(json!({
        "var": var,
        "cvar": es,
        "es": es
    }))
}

fn var_parametric(args: &Value) -> ToolCallResult {
    let mean = req_f64(args, "mean")?;
    let std = req_f64(args, "std")?;
    let confidence = req_f64(args, "confidence")?;
    let horizon = super::opt_f64(args, "horizon", 1.0)?;

    let z = openferric::math::normal_inv_cdf(confidence);
    let sigma_h = std.abs() * horizon.sqrt();
    let var = (-mean * horizon + z * sigma_h).max(0.0);

    Ok(json!({ "var": var }))
}

fn portfolio_risk(args: &Value) -> ToolCallResult {
    let positions = parse_positions(req_value(args, "positions")?)?;
    let market_obj = obj(req_value(args, "market")?, "market")?;
    let annual_vol = market_obj
        .get("annual_volatility")
        .and_then(Value::as_f64)
        .unwrap_or(0.2);
    let confidence = market_obj
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.99);
    let horizon = market_obj
        .get("horizon")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);

    let portfolio = Portfolio::new(positions);
    let total_delta = portfolio.total_delta();
    let total_gamma = portfolio.total_gamma();
    let total_vega = portfolio.total_vega();
    let var = delta_normal_var(total_delta, annual_vol, confidence, horizon);

    Ok(json!({
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "total_vega": total_vega,
        "var": var
    }))
}

fn scenario_analysis(args: &Value) -> ToolCallResult {
    let positions = parse_positions(req_value(args, "positions")?)?;
    let scenarios = req_array(args, "scenarios")?;

    let portfolio = Portfolio::new(positions);
    let agg = portfolio.aggregate_greeks();

    let rows = scenarios
        .iter()
        .map(|scenario| {
            let s = scenario
                .as_object()
                .ok_or_else(|| "scenario entries must be objects".to_string())?;
            let name = s
                .get("scenario")
                .or_else(|| s.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("scenario")
                .to_string();
            let spot_shock = s
                .get("spot_shock_pct")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let vol_shock = s
                .get("vol_shock_pct")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let horizon = s
                .get("horizon_years")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);

            let pnl = portfolio.scenario_pnl_with_horizon(spot_shock, vol_shock, horizon);
            Ok(json!({
                "scenario": name,
                "pnl": pnl,
                "greeks": {
                    "delta": agg.delta,
                    "gamma": agg.gamma,
                    "vega": agg.vega,
                    "theta": agg.theta
                }
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(Value::Array(rows))
}

fn sensitivity_ladder(args: &Value) -> ToolCallResult {
    let instrument = obj(req_value(args, "instrument")?, "instrument")?;
    let bump_sizes = req_array_f64(args, "bump_sizes")?;

    let spot = instrument
        .get("spot")
        .and_then(Value::as_f64)
        .ok_or_else(|| "instrument.spot must be numeric".to_string())?;
    let strike = instrument
        .get("strike")
        .and_then(Value::as_f64)
        .ok_or_else(|| "instrument.strike must be numeric".to_string())?;
    let rate = instrument
        .get("rate")
        .and_then(Value::as_f64)
        .ok_or_else(|| "instrument.rate must be numeric".to_string())?;
    let vol = instrument
        .get("vol")
        .and_then(Value::as_f64)
        .ok_or_else(|| "instrument.vol must be numeric".to_string())?;
    let time = instrument
        .get("time")
        .and_then(Value::as_f64)
        .ok_or_else(|| "instrument.time must be numeric".to_string())?;
    let is_call = instrument
        .get("is_call")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let option_type = if is_call {
        openferric::core::OptionType::Call
    } else {
        openferric::core::OptionType::Put
    };

    let ladder = bump_sizes
        .iter()
        .map(|bump| {
            let bumped_spot = (spot * (1.0 + bump)).max(1.0e-8);
            let price = black_scholes_price(option_type, bumped_spot, strike, rate, vol, time);

            let ds = (0.01 * bumped_spot).max(1.0e-6);
            let p_up = black_scholes_price(option_type, bumped_spot + ds, strike, rate, vol, time);
            let p_dn = black_scholes_price(
                option_type,
                (bumped_spot - ds).max(1.0e-8),
                strike,
                rate,
                vol,
                time,
            );
            let delta = (p_up - p_dn) / (2.0 * ds);

            json!({ "bump": bump, "price": price, "delta": delta })
        })
        .collect::<Vec<_>>();

    Ok(Value::Array(ladder))
}

fn cva(args: &Value) -> ToolCallResult {
    let (times, values) = parse_profile(req_value(args, "exposure_profile")?)?;
    let curve = survival_curve_from_value(req_value(args, "survival_curve")?)?;
    let recovery = req_f64(args, "recovery")?;

    let discount = unit_discount_curve(&times);
    let calc = XvaCalculator::new(
        discount,
        curve.clone(),
        curve,
        (1.0 - recovery).clamp(0.0, 1.0),
        (1.0 - recovery).clamp(0.0, 1.0),
    );

    Ok(json!({ "cva": calc.cva_from_expected_exposure(&times, &values) }))
}

fn dva(args: &Value) -> ToolCallResult {
    let (times, values) = parse_profile(req_value(args, "exposure_profile")?)?;
    let own_curve = survival_curve_from_value(req_value(args, "own_survival")?)?;
    let recovery = req_f64(args, "recovery")?;

    let discount = unit_discount_curve(&times);
    let calc = XvaCalculator::new(
        discount,
        own_curve.clone(),
        own_curve,
        (1.0 - recovery).clamp(0.0, 1.0),
        (1.0 - recovery).clamp(0.0, 1.0),
    );

    Ok(json!({ "dva": calc.dva_from_negative_expected_exposure(&times, &values) }))
}

fn fva(args: &Value) -> ToolCallResult {
    let (times, values) = parse_profile(req_value(args, "exposure_profile")?)?;
    let funding_curve = curve_from_value(req_value(args, "funding_curve")?)?;

    let spreads = times
        .iter()
        .map(|t| funding_curve.zero_rate(*t).max(0.0))
        .collect::<Vec<_>>();

    let fva_value = openferric::risk::fva_from_profile(&times, &values, &spreads, &funding_curve);
    Ok(json!({ "fva": fva_value }))
}

fn xva_all(args: &Value) -> ToolCallResult {
    let (times, values) = parse_profile(req_value(args, "exposure_profile")?)?;
    let counterparty = survival_curve_from_value(req_value(args, "counterparty_curve")?)?;
    let own = survival_curve_from_value(req_value(args, "own_curve")?)?;
    let funding_curve = curve_from_value(req_value(args, "funding_curve")?)?;
    let recovery = req_f64(args, "recovery")?;

    let discount = unit_discount_curve(&times);
    let lgd = (1.0 - recovery).clamp(0.0, 1.0);
    let calc = XvaCalculator::new(discount, counterparty, own, lgd, lgd);

    let cva = calc.cva_from_expected_exposure(&times, &values);
    let dva = calc.dva_from_negative_expected_exposure(&times, &values);
    let spreads = times
        .iter()
        .map(|t| funding_curve.zero_rate(*t).max(0.0))
        .collect::<Vec<_>>();
    let fva = openferric::risk::fva_from_profile(&times, &values, &spreads, &funding_curve);

    Ok(json!({
        "cva": cva,
        "dva": dva,
        "fva": fva,
        "total": cva + dva + fva
    }))
}

fn parse_positions(value: &Value) -> Result<Vec<Position<String>>, String> {
    let arr = value
        .as_array()
        .ok_or_else(|| "positions must be an array".to_string())?;

    let mut out = Vec::with_capacity(arr.len());
    for (i, p) in arr.iter().enumerate() {
        let obj_map = p
            .as_object()
            .ok_or_else(|| "position entries must be objects".to_string())?;

        let quantity = obj_map
            .get("quantity")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let spot = obj_map.get("spot").and_then(Value::as_f64).unwrap_or(100.0);
        let implied_vol = obj_map
            .get("implied_vol")
            .and_then(Value::as_f64)
            .unwrap_or(0.2);

        let greeks_obj = obj_map
            .get("greeks")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        let delta = greeks_obj
            .get("delta")
            .and_then(Value::as_f64)
            .or_else(|| obj_map.get("delta").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let gamma = greeks_obj
            .get("gamma")
            .and_then(Value::as_f64)
            .or_else(|| obj_map.get("gamma").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let vega = greeks_obj
            .get("vega")
            .and_then(Value::as_f64)
            .or_else(|| obj_map.get("vega").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let theta = greeks_obj
            .get("theta")
            .and_then(Value::as_f64)
            .or_else(|| obj_map.get("theta").and_then(Value::as_f64))
            .unwrap_or(0.0);
        let rho = greeks_obj
            .get("rho")
            .and_then(Value::as_f64)
            .or_else(|| obj_map.get("rho").and_then(Value::as_f64))
            .unwrap_or(0.0);

        let greeks = Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        };

        out.push(Position::new(
            format!("position_{i}"),
            quantity,
            greeks,
            spot,
            implied_vol,
        ));
    }

    Ok(out)
}

fn parse_profile(value: &Value) -> Result<(Vec<f64>, Vec<f64>), String> {
    if let Some(obj_map) = value.as_object() {
        let times = obj_map
            .get("times")
            .or_else(|| obj_map.get("tenors"))
            .and_then(Value::as_array)
            .ok_or_else(|| "exposure_profile.times must be an array".to_string())?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| "exposure times must be numeric".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let values = obj_map
            .get("values")
            .or_else(|| obj_map.get("exposures"))
            .and_then(Value::as_array)
            .ok_or_else(|| "exposure_profile.values must be an array".to_string())?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| "exposure values must be numeric".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        if times.len() != values.len() {
            return Err("exposure times/values length mismatch".to_string());
        }

        return Ok((times, values));
    }

    if let Some(values_arr) = value.as_array() {
        let values = values_arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| "exposure profile values must be numeric".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let times = (1..=values.len()).map(|i| i as f64).collect::<Vec<_>>();
        return Ok((times, values));
    }

    Err("exposure_profile must be an object with times/values or a numeric array".to_string())
}

fn unit_discount_curve(times: &[f64]) -> openferric::rates::YieldCurve {
    let points = times
        .iter()
        .map(|t| (t.max(1.0e-8), 1.0))
        .collect::<Vec<_>>();
    openferric::rates::YieldCurve::new(points)
}
