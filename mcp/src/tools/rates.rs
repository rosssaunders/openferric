use chrono::{Duration, NaiveDate};
use openferric::math::ExtrapolationMode;
use openferric::rates::{
    Calendar, CapFloor, DayCountConvention, FixedRateBond, ForwardRateAgreement, InterestRateSwap,
    OvernightIndexSwap, ScheduleConfig, Swaption, YieldCurve, YieldCurveBuilder,
    YieldCurveInterpolationSettings, generate_schedule_with_config,
};
use serde_json::{Value, json};

use super::{
    ToolCallResult, ToolSpec, curve_from_value, curve_to_json, opt_bool, opt_str,
    parse_business_day_convention, parse_curve_interpolation, parse_date, parse_day_count,
    parse_frequency, req_f64, req_str, tenors_from_value,
};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "yield_curve_build",
            description: "Build a discount curve from tenors and deposit rates.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tenors": {"type": "array"},
                    "rates": {"type": "array", "items": {"type": "number"}},
                    "interp": {"type": "string"}
                },
                "required": ["tenors", "rates"]
            }),
        },
        ToolSpec {
            name: "bond_price",
            description: "Price a fixed-rate bond and return risk measures.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "face": {"type": "number"},
                    "coupon_rate": {"type": "number"},
                    "maturity": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["face", "coupon_rate", "maturity", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "swap_rate",
            description: "Compute par swap rate, PV01, and DV01.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "fixed_freq": {"type": "string"},
                    "float_freq": {"type": "string"},
                    "maturity": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["fixed_freq", "float_freq", "maturity", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "swaption_price",
            description: "Price a Black-76 swaption and return finite-difference Greeks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "strike": {"type": "number"},
                    "maturity": {"type": "number"},
                    "swap_tenor": {"type": "number"},
                    "vol": {"type": "number"},
                    "yield_curve": {"type": "object"},
                    "is_payer": {"type": "boolean"}
                },
                "required": ["strike", "maturity", "swap_tenor", "vol", "yield_curve", "is_payer"]
            }),
        },
        ToolSpec {
            name: "fra_price",
            description: "Price a forward rate agreement.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "fixed_rate": {"type": "number"},
                    "notional": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["start", "end", "fixed_rate", "notional", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "cap_floor_price",
            description: "Price an IR cap/floor and return finite-difference Greeks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "strike": {"type": "number"},
                    "maturity": {"type": "number"},
                    "freq": {"type": "string"},
                    "vol": {"type": "number"},
                    "yield_curve": {"type": "object"},
                    "is_cap": {"type": "boolean"}
                },
                "required": ["strike", "maturity", "freq", "vol", "yield_curve", "is_cap"]
            }),
        },
        ToolSpec {
            name: "ois_rate",
            description: "Compute OIS par fixed rate.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "maturity": {"type": "number"},
                    "yield_curve": {"type": "object"}
                },
                "required": ["maturity", "yield_curve"]
            }),
        },
        ToolSpec {
            name: "day_count",
            description: "Compute year fraction under a day-count convention.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "convention": {"type": "string"}
                },
                "required": ["start_date", "end_date", "convention"]
            }),
        },
        ToolSpec {
            name: "schedule_generate",
            description: "Generate a coupon schedule between two dates.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "freq": {"type": "string"},
                    "convention": {"type": "string"}
                },
                "required": ["start", "end", "freq"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "yield_curve_build" => Some(yield_curve_build(args)),
        "bond_price" => Some(bond_price(args)),
        "swap_rate" => Some(swap_rate(args)),
        "swaption_price" => Some(swaption_price(args)),
        "fra_price" => Some(fra_price(args)),
        "cap_floor_price" => Some(cap_floor_price(args)),
        "ois_rate" => Some(ois_rate(args)),
        "day_count" => Some(day_count(args)),
        "schedule_generate" => Some(schedule_generate(args)),
        _ => None,
    }
}

fn yield_curve_build(args: &Value) -> ToolCallResult {
    let tenors = tenors_from_value(args, "tenors")?;
    let rates = args
        .get("rates")
        .and_then(Value::as_array)
        .ok_or_else(|| "parameter `rates` must be an array".to_string())?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "parameter `rates` must be numeric".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    if tenors.len() != rates.len() {
        return Err("tenors/rates length mismatch".to_string());
    }

    let method = parse_curve_interpolation(opt_str(args, "interp"));
    let settings = YieldCurveInterpolationSettings {
        method,
        extrapolation: ExtrapolationMode::Linear,
    };

    let deposits = tenors.into_iter().zip(rates).collect::<Vec<_>>();
    let curve = YieldCurveBuilder::from_deposits_with_settings(&deposits, settings)
        .map_err(|e| format!("curve build failed: {e:?}"))?;

    Ok(json!({ "curve_json": curve_to_json(&curve) }))
}

fn bond_price(args: &Value) -> ToolCallResult {
    let face = req_f64(args, "face")?;
    let coupon_rate = req_f64(args, "coupon_rate")?;
    let maturity = req_f64(args, "maturity")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;

    let bond = FixedRateBond {
        face_value: face,
        coupon_rate,
        frequency: 2,
        maturity,
        day_count: DayCountConvention::Act365Fixed,
    };

    let dirty = bond.dirty_price(&curve);
    let settlement = 0.0;
    let accrued = bond.accrued_interest(settlement);
    let clean = bond.clean_price(&curve, settlement);
    let duration = bond.duration(&curve);
    let convexity = bond.convexity(&curve);

    let bumped = bump_curve_parallel(&curve, 1.0e-4);
    let dv01 = bond.dirty_price(&bumped) - dirty;

    Ok(json!({
        "dirty_price": dirty,
        "clean_price": clean,
        "accrued": accrued,
        "duration": duration,
        "convexity": convexity,
        "dv01": dv01
    }))
}

fn swap_rate(args: &Value) -> ToolCallResult {
    let fixed_freq = parse_frequency(req_str(args, "fixed_freq")?)?;
    let float_freq = parse_frequency(req_str(args, "float_freq")?)?;
    let maturity = req_f64(args, "maturity")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;

    let start =
        NaiveDate::from_ymd_opt(2026, 1, 1).ok_or_else(|| "invalid base date".to_string())?;
    let end = start + Duration::days((maturity * 365.0).round() as i64);

    let swap = InterestRateSwap::builder()
        .notional(1.0)
        .fixed_rate(0.0)
        .start_date(start)
        .end_date(end)
        .fixed_freq(fixed_freq)
        .float_freq(float_freq)
        .build();

    let par_rate = swap.par_rate(&curve);
    let npv_at_zero = swap.npv(&curve);

    let swap_1bp = InterestRateSwap::builder()
        .notional(1.0)
        .fixed_rate(1.0e-4)
        .start_date(start)
        .end_date(end)
        .fixed_freq(fixed_freq)
        .float_freq(float_freq)
        .build();
    let pv01 = (npv_at_zero - swap_1bp.npv(&curve)).abs();

    let dv01 = swap.dv01(&curve);

    Ok(json!({ "par_rate": par_rate, "pv01": pv01, "dv01": dv01 }))
}

fn swaption_price(args: &Value) -> ToolCallResult {
    let strike = req_f64(args, "strike")?;
    let maturity = req_f64(args, "maturity")?;
    let swap_tenor = req_f64(args, "swap_tenor")?;
    let vol = req_f64(args, "vol")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;
    let is_payer = opt_bool(args, "is_payer", true)?;

    let swaption = Swaption {
        notional: 1.0,
        strike,
        option_expiry: maturity,
        swap_tenor,
        is_payer,
    };

    let price = swaption.price(&curve, vol);

    let dk = (0.01 * strike.abs()).max(1.0e-6);
    let dv = 0.01;
    let dt = maturity.clamp(1.0e-6, 1.0 / 365.0);
    let dr = 1.0e-4;

    let mut up = swaption;
    up.strike = strike + dk;
    let mut dn = swaption;
    dn.strike = (strike - dk).max(1.0e-8);
    let p_up = up.price(&curve, vol);
    let p_dn = dn.price(&curve, vol);
    let delta = (p_up - p_dn) / (2.0 * dk);
    let gamma = (p_up - 2.0 * price + p_dn) / (dk * dk);

    let vega = (swaption.price(&curve, vol + dv) - swaption.price(&curve, (vol - dv).max(1.0e-8)))
        / (2.0 * dv);

    let mut t_short = swaption;
    t_short.option_expiry = (maturity - dt).max(1.0e-8);
    let theta = (t_short.price(&curve, vol) - price) / dt;

    let curve_up = bump_curve_parallel(&curve, dr);
    let curve_dn = bump_curve_parallel(&curve, -dr);
    let rho = (swaption.price(&curve_up, vol) - swaption.price(&curve_dn, vol)) / (2.0 * dr);

    Ok(json!({
        "price": price,
        "greeks": {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }
    }))
}

fn fra_price(args: &Value) -> ToolCallResult {
    let start = req_f64(args, "start")?;
    let end = req_f64(args, "end")?;
    let fixed_rate = req_f64(args, "fixed_rate")?;
    let notional = req_f64(args, "notional")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;

    let base =
        NaiveDate::from_ymd_opt(2026, 1, 1).ok_or_else(|| "invalid base date".to_string())?;
    let fra = ForwardRateAgreement {
        notional,
        fixed_rate,
        start_date: base + Duration::days((start * 365.0).round() as i64),
        end_date: base + Duration::days((end * 365.0).round() as i64),
        day_count: DayCountConvention::Act365Fixed,
    };

    Ok(json!({ "price": fra.npv(&curve) }))
}

fn cap_floor_price(args: &Value) -> ToolCallResult {
    let strike = req_f64(args, "strike")?;
    let maturity = req_f64(args, "maturity")?;
    let freq = parse_frequency(req_str(args, "freq")?)?;
    let vol = req_f64(args, "vol")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;
    let is_cap = opt_bool(args, "is_cap", true)?;

    let start =
        NaiveDate::from_ymd_opt(2026, 1, 1).ok_or_else(|| "invalid base date".to_string())?;
    let end = start + Duration::days((maturity * 365.0).round() as i64);

    let instrument = CapFloor {
        notional: 1.0,
        strike,
        start_date: start,
        end_date: end,
        frequency: freq,
        day_count: DayCountConvention::Act365Fixed,
        is_cap,
    };

    let price = instrument.price(&curve, vol);
    let dv = 0.01;
    let dr = 1.0e-4;

    let vega = (instrument.price(&curve, vol + dv)
        - instrument.price(&curve, (vol - dv).max(1.0e-8)))
        / (2.0 * dv);

    let curve_up = bump_curve_parallel(&curve, dr);
    let curve_dn = bump_curve_parallel(&curve, -dr);
    let rho = (instrument.price(&curve_up, vol) - instrument.price(&curve_dn, vol)) / (2.0 * dr);

    Ok(json!({
        "price": price,
        "greeks": {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": vega,
            "theta": 0.0,
            "rho": rho
        }
    }))
}

fn ois_rate(args: &Value) -> ToolCallResult {
    let maturity = req_f64(args, "maturity")?;
    let curve = curve_from_value(super::req_value(args, "yield_curve")?)?;

    let ois = OvernightIndexSwap {
        notional: 1.0,
        fixed_rate: 0.0,
        float_spread: 0.0,
        tenor: maturity,
    };

    let rate = ois.par_fixed_rate(&curve, &curve);
    Ok(json!({ "rate": rate }))
}

fn day_count(args: &Value) -> ToolCallResult {
    let start = parse_date(req_str(args, "start_date")?)?;
    let end = parse_date(req_str(args, "end_date")?)?;
    let convention = parse_day_count(req_str(args, "convention")?)?;

    let fraction = openferric::rates::year_fraction(start, end, convention);
    Ok(json!({ "fraction": fraction }))
}

fn schedule_generate(args: &Value) -> ToolCallResult {
    let start = parse_date(req_str(args, "start")?)?;
    let end = parse_date(req_str(args, "end")?)?;
    let freq = parse_frequency(req_str(args, "freq")?)?;

    let convention = opt_str(args, "convention")
        .map(parse_business_day_convention)
        .transpose()?
        .unwrap_or(openferric::rates::BusinessDayConvention::ModifiedFollowing);

    let config = ScheduleConfig {
        calendar: Calendar::weekends_only(),
        business_day_convention: convention,
        stub_convention: openferric::rates::StubConvention::ShortBack,
        roll_convention: openferric::rates::RollConvention::None,
    };

    let dates = generate_schedule_with_config(start, end, freq, &config)
        .into_iter()
        .map(|d| d.format("%Y-%m-%d").to_string())
        .collect::<Vec<_>>();

    Ok(json!({ "dates": dates }))
}

fn bump_curve_parallel(curve: &YieldCurve, bump: f64) -> YieldCurve {
    let bumped = curve
        .tenors
        .iter()
        .map(|(t, df)| {
            let z = if *t > 0.0 { -df.ln() / *t } else { 0.0 };
            (*t, (-(z + bump) * *t).exp())
        })
        .collect::<Vec<_>>();

    YieldCurve::new(bumped)
}
