use openferric::core::OptionType;
use openferric::vol::arbitrage::ArbitrageViolation;
use openferric::vol::fengler::FenglerSurface;
use openferric::vol::forward::{ForwardVarianceCurve, ForwardVarianceSource};
use openferric::vol::implied::implied_vol;
use openferric::vol::local_vol::{ImpliedVolSurface as LocalVolSurface, dupire_local_vol};
use openferric::vol::sabr::fit_sabr;
use openferric::vol::surface::{SviParams, calibrate_svi};
use serde_json::{Value, json};

use super::{
    ToolCallResult, ToolSpec, obj, opt_f64, req_array_f64, req_f64, req_matrix_f64, req_value,
};

#[derive(Debug, Clone)]
struct SimpleSurface {
    forward: f64,
    strikes: Vec<f64>,
    expiries: Vec<f64>,
    vols: Vec<Vec<f64>>,
}

impl SimpleSurface {
    fn vol_at(&self, strike: f64, expiry: f64) -> f64 {
        let (ei0, ei1, ew) = locate_bounds(&self.expiries, expiry);
        let (si0, si1, sw) = locate_bounds(&self.strikes, strike);

        if ei0 == ei1 && si0 == si1 {
            return self.vols[ei0][si0];
        }
        if ei0 == ei1 {
            let v0 = self.vols[ei0][si0];
            let v1 = self.vols[ei0][si1];
            return v0 + (v1 - v0) * sw;
        }
        if si0 == si1 {
            let v0 = self.vols[ei0][si0];
            let v1 = self.vols[ei1][si0];
            return v0 + (v1 - v0) * ew;
        }

        let v00 = self.vols[ei0][si0];
        let v01 = self.vols[ei0][si1];
        let v10 = self.vols[ei1][si0];
        let v11 = self.vols[ei1][si1];

        let v0 = v00 + (v01 - v00) * sw;
        let v1 = v10 + (v11 - v10) * sw;
        v0 + (v1 - v0) * ew
    }
}

impl LocalVolSurface for SimpleSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.vol_at(strike, expiry)
    }
}

impl ForwardVarianceSource for SimpleSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.vol_at(strike, expiry)
    }

    fn forward_price(&self, _expiry: f64) -> f64 {
        self.forward
    }

    fn expiries(&self) -> &[f64] {
        &self.expiries
    }
}

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "implied_vol_calc",
            description: "Compute Black-Scholes implied volatility from option price.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "price": {"type":"number"},
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "rate": {"type":"number"},
                    "time": {"type":"number"},
                    "is_call": {"type":"boolean"}
                },
                "required": ["price","spot","strike","rate","time","is_call"]
            }),
        },
        ToolSpec {
            name: "vol_surface_build",
            description: "Build a volatility surface JSON payload from strike/expiry vol grid.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "strikes": {"type":"array", "items": {"type":"number"}},
                    "expiries": {"type":"array", "items": {"type":"number"}},
                    "vols": {"type":"array"},
                    "forward": {"type":"number"}
                },
                "required": ["strikes","expiries","vols"]
            }),
        },
        ToolSpec {
            name: "sabr_calibrate",
            description: "Calibrate SABR parameters to a strike-vol slice.",
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
            name: "svi_calibrate",
            description: "Calibrate SVI parameters to a strike-vol slice.",
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
            name: "local_vol",
            description: "Compute Dupire local vol from a surface JSON.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "expiry": {"type":"number"},
                    "surface": {"type":"object"}
                },
                "required": ["spot","strike","expiry","surface"]
            }),
        },
        ToolSpec {
            name: "forward_vol",
            description: "Compute forward vol between two maturities from a surface.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "t1": {"type":"number"},
                    "t2": {"type":"number"},
                    "surface": {"type":"object"}
                },
                "required": ["t1","t2","surface"]
            }),
        },
        ToolSpec {
            name: "vol_smile",
            description: "Extract smile points at a given expiry from a surface.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "expiry": {"type":"number"},
                    "surface": {"type":"object"}
                },
                "required": ["expiry","surface"]
            }),
        },
        ToolSpec {
            name: "vol_arbitrage_check",
            description: "Run static-arbitrage diagnostics on a surface.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "surface": {"type":"object"}
                },
                "required": ["surface"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "implied_vol_calc" => Some(implied_vol_calc(args)),
        "vol_surface_build" => Some(vol_surface_build(args)),
        "sabr_calibrate" => Some(sabr_calibrate(args)),
        "svi_calibrate" => Some(svi_calibrate(args)),
        "local_vol" => Some(local_vol_tool(args)),
        "forward_vol" => Some(forward_vol_tool(args)),
        "vol_smile" => Some(vol_smile_tool(args)),
        "vol_arbitrage_check" => Some(vol_arbitrage_check(args)),
        _ => None,
    }
}

fn implied_vol_calc(args: &Value) -> ToolCallResult {
    let price = req_f64(args, "price")?;
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let time = req_f64(args, "time")?;
    let is_call = super::req_bool(args, "is_call")?;

    let iv = implied_vol(
        if is_call {
            OptionType::Call
        } else {
            OptionType::Put
        },
        spot,
        strike,
        rate,
        time,
        price,
        1.0e-10,
        128,
    )
    .map_err(|e| e.to_string())?;

    Ok(json!({ "iv": iv }))
}

fn vol_surface_build(args: &Value) -> ToolCallResult {
    let strikes = req_array_f64(args, "strikes")?;
    let expiries = req_array_f64(args, "expiries")?;
    let vols = req_matrix_f64(args, "vols")?;
    let forward = opt_f64(args, "forward", strikes[strikes.len() / 2])?;

    validate_grid(&strikes, &expiries, &vols)?;

    let mut svi_slices = Vec::with_capacity(expiries.len());
    for (i, t) in expiries.iter().enumerate() {
        let row = &vols[i];
        let points = strikes
            .iter()
            .zip(row.iter())
            .map(|(k, v)| ((k / forward).ln(), v * v * t.max(1.0e-10)))
            .collect::<Vec<_>>();

        let init = SviParams {
            a: 0.01,
            b: 0.2,
            rho: -0.2,
            m: 0.0,
            sigma: 0.3,
        };
        let fit = calibrate_svi(&points, init, 200, 0.0);

        svi_slices.push(json!({
            "expiry": t,
            "a": fit.a,
            "b": fit.b,
            "rho": fit.rho,
            "m": fit.m,
            "sigma": fit.sigma
        }));
    }

    Ok(json!({
        "surface_json": {
            "forward": forward,
            "strikes": strikes,
            "expiries": expiries,
            "vols": vols,
            "svi_slices": svi_slices
        }
    }))
}

fn sabr_calibrate(args: &Value) -> ToolCallResult {
    let forward = req_f64(args, "forward")?;
    let expiry = req_f64(args, "expiry")?;
    let strikes = req_array_f64(args, "strikes")?;
    let vols = req_array_f64(args, "vols")?;

    if strikes.len() != vols.len() || strikes.is_empty() {
        return Err("strikes and vols must be non-empty and have equal length".to_string());
    }

    let params = fit_sabr(forward, &strikes, &vols, expiry, 0.5);
    let mse = strikes
        .iter()
        .zip(vols.iter())
        .map(|(k, v)| {
            let e = params.implied_vol(forward, *k, expiry) - *v;
            e * e
        })
        .sum::<f64>()
        / strikes.len() as f64;

    Ok(json!({
        "alpha": params.alpha,
        "beta": params.beta,
        "rho": params.rho,
        "nu": params.nu,
        "fit_error": mse.sqrt()
    }))
}

fn svi_calibrate(args: &Value) -> ToolCallResult {
    let forward = req_f64(args, "forward")?;
    let expiry = req_f64(args, "expiry")?;
    let strikes = req_array_f64(args, "strikes")?;
    let vols = req_array_f64(args, "vols")?;

    if strikes.len() != vols.len() || strikes.is_empty() {
        return Err("strikes and vols must be non-empty and have equal length".to_string());
    }

    let points = strikes
        .iter()
        .zip(vols.iter())
        .map(|(k, v)| ((k / forward).ln(), v * v * expiry.max(1.0e-10)))
        .collect::<Vec<_>>();

    let init = SviParams {
        a: 0.01,
        b: 0.2,
        rho: -0.2,
        m: 0.0,
        sigma: 0.25,
    };
    let fit = calibrate_svi(&points, init, 250, 0.0);

    let mse = points
        .iter()
        .map(|(k, w)| {
            let e = fit.total_variance(*k) - *w;
            e * e
        })
        .sum::<f64>()
        / points.len() as f64;

    Ok(json!({
        "a": fit.a,
        "b": fit.b,
        "rho": fit.rho,
        "m": fit.m,
        "sigma": fit.sigma,
        "fit_error": mse.sqrt()
    }))
}

fn local_vol_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let expiry = req_f64(args, "expiry")?;
    let surface = parse_surface(req_value(args, "surface")?, Some(spot))?;

    let lv = dupire_local_vol(surface.clone(), surface.forward.max(1.0e-8), strike, expiry);
    Ok(json!({ "local_vol": lv }))
}

fn forward_vol_tool(args: &Value) -> ToolCallResult {
    let t1 = req_f64(args, "t1")?;
    let t2 = req_f64(args, "t2")?;
    let surface = parse_surface(req_value(args, "surface")?, None)?;

    let fvc = ForwardVarianceCurve::from_surface(&surface, &surface.expiries)
        .map_err(|e| e.to_string())?;
    let fv = fvc.forward_vol(t1, t2).map_err(|e| e.to_string())?;

    Ok(json!({ "forward_vol": fv }))
}

fn vol_smile_tool(args: &Value) -> ToolCallResult {
    let expiry = req_f64(args, "expiry")?;
    let surface = parse_surface(req_value(args, "surface")?, None)?;

    let vols = surface
        .strikes
        .iter()
        .map(|k| surface.vol_at(*k, expiry))
        .collect::<Vec<_>>();

    Ok(json!({
        "strikes": surface.strikes,
        "vols": vols
    }))
}

fn vol_arbitrage_check(args: &Value) -> ToolCallResult {
    let surface = parse_surface(req_value(args, "surface")?, None)?;

    let mut quotes = Vec::new();
    for (ei, t) in surface.expiries.iter().enumerate() {
        for (si, k) in surface.strikes.iter().enumerate() {
            quotes.push((*k, *t, surface.vols[ei][si]));
        }
    }

    let forwards = surface
        .expiries
        .iter()
        .map(|t| (*t, surface.forward_price(*t)))
        .collect::<Vec<_>>();

    let fengler = FenglerSurface::new(&quotes, &forwards);
    let violations = fengler.check_arbitrage();

    let details = violations
        .iter()
        .map(arbitrage_violation_to_json)
        .collect::<Vec<_>>();

    Ok(json!({
        "has_arbitrage": !violations.is_empty(),
        "details": details
    }))
}

fn parse_surface(value: &Value, forward_fallback: Option<f64>) -> Result<SimpleSurface, String> {
    let obj_map = obj(value, "surface")?;

    let strikes = obj_map
        .get("strikes")
        .and_then(Value::as_array)
        .ok_or_else(|| "surface.strikes must be an array".to_string())?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "surface.strikes must be numeric".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let expiries = obj_map
        .get("expiries")
        .and_then(Value::as_array)
        .ok_or_else(|| "surface.expiries must be an array".to_string())?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "surface.expiries must be numeric".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let vols = obj_map
        .get("vols")
        .and_then(Value::as_array)
        .ok_or_else(|| "surface.vols must be a 2D array".to_string())?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| "surface.vols must be a 2D array".to_string())?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| "surface.vols entries must be numeric".to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    validate_grid(&strikes, &expiries, &vols)?;

    let forward = obj_map
        .get("forward")
        .and_then(Value::as_f64)
        .or(forward_fallback)
        .unwrap_or_else(|| strikes[strikes.len() / 2]);

    Ok(SimpleSurface {
        forward: forward.max(1.0e-8),
        strikes,
        expiries,
        vols,
    })
}

fn validate_grid(strikes: &[f64], expiries: &[f64], vols: &[Vec<f64>]) -> Result<(), String> {
    if strikes.is_empty() || expiries.is_empty() {
        return Err("surface strike/expiry grids cannot be empty".to_string());
    }
    if strikes.windows(2).any(|w| w[1] <= w[0]) {
        return Err("surface strikes must be strictly increasing".to_string());
    }
    if expiries.windows(2).any(|w| w[1] <= w[0]) {
        return Err("surface expiries must be strictly increasing".to_string());
    }
    if vols.len() != expiries.len() {
        return Err("surface vols row count must match expiries".to_string());
    }
    if vols.iter().any(|row| row.len() != strikes.len()) {
        return Err("surface vol row length must match strikes".to_string());
    }
    Ok(())
}

fn locate_bounds(grid: &[f64], x: f64) -> (usize, usize, f64) {
    if x <= grid[0] {
        return (0, 0, 0.0);
    }
    let last = grid.len() - 1;
    if x >= grid[last] {
        return (last, last, 0.0);
    }

    let mut lo = 0usize;
    for i in 0..last {
        if x >= grid[i] && x <= grid[i + 1] {
            lo = i;
            break;
        }
    }

    let hi = lo + 1;
    let w = (x - grid[lo]) / (grid[hi] - grid[lo]);
    (lo, hi, w)
}

fn arbitrage_violation_to_json(v: &ArbitrageViolation) -> Value {
    match v {
        ArbitrageViolation::Butterfly {
            strike,
            expiry,
            density,
        } => json!({
            "type": "butterfly",
            "strike": strike,
            "expiry": expiry,
            "metric": density
        }),
        ArbitrageViolation::Calendar {
            strike,
            t1,
            t2,
            dw_dt,
        } => json!({
            "type": "calendar",
            "strike": strike,
            "t1": t1,
            "t2": t2,
            "metric": dw_dt
        }),
    }
}
