use openferric::core::OptionType;
use openferric::engines::fft::carr_madan::{CarrMadanParams, try_heston_price_fft};
use openferric::models::{HullWhite, VarianceGamma, rbergomi_european_mc};
use openferric::pricing::european::black_scholes_price;
use openferric::vol::local_vol::{ImpliedVolSurface as LocalVolSurface, dupire_local_vol};
use openferric::vol::sabr::SabrParams;
use serde_json::{Value, json};

use super::{ToolCallResult, ToolSpec, obj, opt_bool, opt_usize, req_f64, req_value};

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
            return self.vols[ei0][si0] + (self.vols[ei0][si1] - self.vols[ei0][si0]) * sw;
        }
        if si0 == si1 {
            return self.vols[ei0][si0] + (self.vols[ei1][si0] - self.vols[ei0][si0]) * ew;
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

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "heston_price",
            description: "Price an option under Heston via FFT.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "rate": {"type":"number"},
                    "v0": {"type":"number"},
                    "kappa": {"type":"number"},
                    "theta": {"type":"number"},
                    "sigma": {"type":"number"},
                    "rho": {"type":"number"},
                    "time": {"type":"number"},
                    "is_call": {"type":"boolean"}
                },
                "required": ["spot","strike","rate","v0","kappa","theta","sigma","rho","time","is_call"]
            }),
        },
        ToolSpec {
            name: "sabr_vol",
            description: "Compute SABR implied volatility.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "forward": {"type":"number"},
                    "strike": {"type":"number"},
                    "expiry": {"type":"number"},
                    "alpha": {"type":"number"},
                    "rho": {"type":"number"},
                    "nu": {"type":"number"},
                    "beta": {"type":"number"}
                },
                "required": ["forward","strike","expiry","alpha","rho","nu"]
            }),
        },
        ToolSpec {
            name: "hull_white_price",
            description: "Compute Hull-White ZCB price from flat initial curve.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "rate": {"type":"number"},
                    "a": {"type":"number"},
                    "sigma": {"type":"number"},
                    "maturity": {"type":"number"}
                },
                "required": ["rate","a","sigma","maturity"]
            }),
        },
        ToolSpec {
            name: "rough_bergomi_price",
            description: "Monte Carlo European call pricing under rough Bergomi.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "time": {"type":"number"},
                    "H": {"type":"number"},
                    "eta": {"type":"number"},
                    "rho": {"type":"number"},
                    "xi": {"type":"number"},
                    "num_paths": {"type":"integer", "minimum": 1}
                },
                "required": ["spot","strike","time","H","eta","rho","xi"]
            }),
        },
        ToolSpec {
            name: "variance_gamma_price",
            description: "Price option under Variance-Gamma via FFT.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "rate": {"type":"number"},
                    "time": {"type":"number"},
                    "sigma": {"type":"number"},
                    "theta_vg": {"type":"number"},
                    "nu": {"type":"number"},
                    "is_call": {"type":"boolean"}
                },
                "required": ["spot","strike","rate","time","sigma","theta_vg","nu","is_call"]
            }),
        },
        ToolSpec {
            name: "local_vol_price",
            description: "Approximate local-vol European call price using Dupire local vol level.",
            input_schema: json!({
                "type":"object",
                "properties": {
                    "spot": {"type":"number"},
                    "strike": {"type":"number"},
                    "time": {"type":"number"},
                    "surface": {"type":"object"},
                    "num_paths": {"type":"integer", "minimum": 1}
                },
                "required": ["spot","strike","time","surface"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "heston_price" => Some(heston_price(args)),
        "sabr_vol" => Some(sabr_vol(args)),
        "hull_white_price" => Some(hull_white_price(args)),
        "rough_bergomi_price" => Some(rough_bergomi_price(args)),
        "variance_gamma_price" => Some(variance_gamma_price(args)),
        "local_vol_price" => Some(local_vol_price(args)),
        _ => None,
    }
}

fn heston_price(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let v0 = req_f64(args, "v0")?;
    let kappa = req_f64(args, "kappa")?;
    let theta = req_f64(args, "theta")?;
    let sigma = req_f64(args, "sigma")?;
    let rho = req_f64(args, "rho")?;
    let time = req_f64(args, "time")?;
    let is_call = opt_bool(args, "is_call", true)?;

    let priced = try_heston_price_fft(
        spot,
        &[strike],
        rate,
        0.0,
        v0,
        kappa,
        theta,
        sigma,
        rho,
        time,
    )
    .map_err(|e| e.to_string())?;

    let call = priced
        .first()
        .map(|(_, p)| *p)
        .ok_or_else(|| "Heston pricing returned no values".to_string())?;

    let price = if is_call {
        call
    } else {
        call - spot + strike * (-rate * time).exp()
    };

    Ok(json!({ "price": price }))
}

fn sabr_vol(args: &Value) -> ToolCallResult {
    let forward = req_f64(args, "forward")?;
    let strike = req_f64(args, "strike")?;
    let expiry = req_f64(args, "expiry")?;
    let alpha = req_f64(args, "alpha")?;
    let rho = req_f64(args, "rho")?;
    let nu = req_f64(args, "nu")?;
    let beta = super::opt_f64(args, "beta", 0.5)?;

    let params = SabrParams {
        alpha,
        beta,
        rho,
        nu,
    };

    Ok(json!({ "vol": params.implied_vol(forward, strike, expiry) }))
}

fn hull_white_price(args: &Value) -> ToolCallResult {
    let rate = req_f64(args, "rate")?;
    let a = req_f64(args, "a")?;
    let sigma = req_f64(args, "sigma")?;
    let maturity = req_f64(args, "maturity")?;

    let curve =
        openferric::rates::YieldCurve::new(vec![(maturity.max(1.0e-8), (-rate * maturity).exp())]);
    let mut model = HullWhite::new(a, sigma);
    let times = (0..=20)
        .map(|i| maturity.max(1.0e-8) * i as f64 / 20.0)
        .collect::<Vec<_>>();
    model.calibrate_theta(&curve, &times);

    let r0 = HullWhite::instantaneous_forward(&curve, 0.0);
    let zcb = model.bond_price(0.0, maturity, r0, &curve);

    Ok(json!({ "zcb_price": zcb }))
}

fn rough_bergomi_price(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let time = req_f64(args, "time")?;
    let hurst = req_f64(args, "H")?;
    let eta = req_f64(args, "eta")?;
    let rho = req_f64(args, "rho")?;
    let xi = req_f64(args, "xi")?;
    let num_paths = opt_usize(args, "num_paths", 20_000)?;

    let n_steps = ((time * 252.0).round() as usize).clamp(16, 512);
    let result = rbergomi_european_mc(
        spot, strike, 0.0, 0.0, time, hurst, eta, rho, xi, num_paths, n_steps,
    );

    Ok(json!({ "price": result.price, "stderr": result.stderr }))
}

fn variance_gamma_price(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let time = req_f64(args, "time")?;
    let sigma = req_f64(args, "sigma")?;
    let theta_vg = req_f64(args, "theta_vg")?;
    let nu = req_f64(args, "nu")?;
    let is_call = req_value(args, "is_call").and_then(|v| {
        v.as_bool()
            .ok_or_else(|| "parameter `is_call` must be boolean".to_string())
    })?;

    let vg = VarianceGamma {
        sigma,
        theta: theta_vg,
        nu,
    };

    let priced = vg
        .european_calls_fft(spot, &[strike], rate, 0.0, time, CarrMadanParams::default())
        .map_err(|e| e.to_string())?;

    let call = priced
        .first()
        .map(|(_, p)| *p)
        .ok_or_else(|| "VG pricing returned no values".to_string())?;

    let price = if is_call {
        call
    } else {
        call - spot + strike * (-rate * time).exp()
    };

    Ok(json!({ "price": price }))
}

fn local_vol_price(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let time = req_f64(args, "time")?;
    let surface = parse_surface(req_value(args, "surface")?, Some(spot))?;

    let lv = dupire_local_vol(surface.clone(), surface.forward, strike, time);
    let num_paths = opt_usize(args, "num_paths", 0)?;

    let price = if num_paths == 0 {
        black_scholes_price(OptionType::Call, spot, strike, 0.0, lv, time)
    } else {
        mc_call_price(spot, strike, time, lv, num_paths, 42)
    };

    Ok(json!({ "price": price }))
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

    if strikes.is_empty() || expiries.is_empty() {
        return Err("surface grids cannot be empty".to_string());
    }
    if vols.len() != expiries.len() || vols.iter().any(|row| row.len() != strikes.len()) {
        return Err("surface vols shape mismatch".to_string());
    }

    let forward = obj_map
        .get("forward")
        .and_then(Value::as_f64)
        .or(forward_fallback)
        .unwrap_or_else(|| strikes[strikes.len() / 2]);

    Ok(SimpleSurface {
        forward,
        strikes,
        expiries,
        vols,
    })
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

fn mc_call_price(spot: f64, strike: f64, time: f64, vol: f64, num_paths: usize, seed: u64) -> f64 {
    use openferric::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};

    let mut rng = FastRng::from_seed(FastRngKind::Philox4x32, seed);
    let drift = -0.5 * vol * vol * time;
    let diff = vol * time.sqrt();

    let mut sum = 0.0;
    for _ in 0..num_paths {
        let z = sample_standard_normal(&mut rng);
        let st = spot * (drift + diff * z).exp();
        sum += (st - strike).max(0.0);
    }

    sum / num_paths as f64
}
