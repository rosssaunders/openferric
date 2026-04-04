use openferric::core::PricingEngine;
use openferric::engines::analytic::digital::DigitalAnalyticEngine;
use openferric::instruments::autocallable::Autocallable;
use openferric::instruments::basket::{BasketOption, BasketType};
use openferric::instruments::digital::CashOrNothingOption;
use openferric::instruments::range_accrual::RangeAccrual;
use openferric::instruments::tarf::Tarf;
use openferric::market::Market;
use openferric::pricing::american::crr_binomial_american;
use openferric::pricing::asian::{AsianStrike, arithmetic_asian_price_mc};
use openferric::pricing::autocallable::price_autocallable;
use openferric::pricing::barrier::{BarrierDirection, BarrierStyle, barrier_price_mc};
use openferric::pricing::basket::price_basket_mc;
use openferric::pricing::bermudan::longstaff_schwartz_bermudan;
use openferric::pricing::european::{black_scholes_greeks, black_scholes_price};
use openferric::pricing::range_accrual::range_accrual_mc_price;
use openferric::pricing::tarf::tarf_mc_price;
use serde_json::{Value, json};

use super::{
    ToolCallResult, ToolSpec, frequency_to_per_year, opt_usize, parse_fixing_frequency_per_year,
    parse_option_type, req_array_f64, req_bool, req_f64, req_matrix_f64, req_str,
};

pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "price_european",
            description: "Black-Scholes European option price and Greeks.",
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
            name: "price_american",
            description: "CRR-binomial American option price with finite-difference Greeks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "is_call": {"type": "boolean"},
                    "steps": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "strike", "rate", "vol", "time", "is_call"]
            }),
        },
        ToolSpec {
            name: "price_asian",
            description: "Arithmetic Asian option Monte Carlo pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "is_call": {"type": "boolean"},
                    "fixing_freq": {"type": "string"},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "strike", "rate", "vol", "time", "is_call", "fixing_freq"]
            }),
        },
        ToolSpec {
            name: "price_barrier",
            description: "Barrier option MC pricing (up/down, in/out).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "barrier": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "is_call": {"type": "boolean"},
                    "barrier_type": {"type": "string"},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "strike", "barrier", "rate", "vol", "time", "is_call", "barrier_type"]
            }),
        },
        ToolSpec {
            name: "price_autocallable",
            description: "Single-underlying worst-of autocallable pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "coupon_barrier": {"type": "number"},
                    "ki_barrier": {"type": "number"},
                    "coupon_rate": {"type": "number"},
                    "time": {"type": "number"},
                    "freq": {"type": ["string", "integer"]},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "coupon_barrier", "ki_barrier", "coupon_rate", "time", "freq", "rate", "vol"]
            }),
        },
        ToolSpec {
            name: "price_basket",
            description: "Correlated basket option MC pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spots": {"type": "array", "items": {"type": "number"}},
                    "vols": {"type": "array", "items": {"type": "number"}},
                    "correlation": {"type": "array"},
                    "strike": {"type": "number"},
                    "time": {"type": "number"},
                    "rate": {"type": "number"},
                    "is_call": {"type": "boolean"},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spots", "vols", "correlation", "strike", "time", "rate", "is_call"]
            }),
        },
        ToolSpec {
            name: "price_bermudan",
            description: "Longstaff-Schwartz Bermudan option pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "is_call": {"type": "boolean"},
                    "exercise_dates": {"type": "array", "items": {"type": "number"}},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "strike", "rate", "vol", "time", "is_call", "exercise_dates"]
            }),
        },
        ToolSpec {
            name: "price_digital",
            description: "Cash-or-nothing digital option analytic pricing.",
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
            name: "price_range_accrual",
            description: "Single-rate range accrual MC pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "lower": {"type": "number"},
                    "upper": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "coupon": {"type": "number"},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "lower", "upper", "rate", "vol", "time", "coupon"]
            }),
        },
        ToolSpec {
            name: "price_tarf",
            description: "TARF Monte Carlo pricing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "spot": {"type": "number"},
                    "strike": {"type": "number"},
                    "rate": {"type": "number"},
                    "vol": {"type": "number"},
                    "time": {"type": "number"},
                    "leverage": {"type": "number"},
                    "ki_barrier": {"type": "number"},
                    "num_fixings": {"type": "integer", "minimum": 1},
                    "num_paths": {"type": "integer", "minimum": 1}
                },
                "required": ["spot", "strike", "rate", "vol", "time", "leverage", "ki_barrier", "num_fixings"]
            }),
        },
    ]
}

pub fn call(name: &str, args: &Value) -> Option<ToolCallResult> {
    match name {
        "price_european" => Some(price_european(args)),
        "price_american" => Some(price_american(args)),
        "price_asian" => Some(price_asian(args)),
        "price_barrier" => Some(price_barrier(args)),
        "price_autocallable" => Some(price_autocallable_tool(args)),
        "price_basket" => Some(price_basket_tool(args)),
        "price_bermudan" => Some(price_bermudan_tool(args)),
        "price_digital" => Some(price_digital_tool(args)),
        "price_range_accrual" => Some(price_range_accrual_tool(args)),
        "price_tarf" => Some(price_tarf_tool(args)),
        _ => None,
    }
}

fn price_european(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;
    let option_type = parse_option_type(is_call);

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

fn price_american(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;
    let steps = opt_usize(args, "steps", 400)?;

    let option_type = parse_option_type(is_call);

    let price_fn = |s: f64, k: f64, r: f64, v: f64, t: f64| -> f64 {
        crr_binomial_american(option_type, s, k, r, v, t.max(1.0e-8), steps)
    };

    let price = price_fn(spot, strike, rate, vol, time);

    let ds = (0.01 * spot.abs()).max(1.0e-6);
    let dv = 0.01;
    let dr = 1.0e-4;
    let dt = time.clamp(1.0e-6, 1.0 / 365.0);

    let p_up = price_fn(spot + ds, strike, rate, vol, time);
    let p_dn = price_fn((spot - ds).max(1.0e-8), strike, rate, vol, time);
    let delta = (p_up - p_dn) / (2.0 * ds);
    let gamma = (p_up - 2.0 * price + p_dn) / (ds * ds);

    let p_vu = price_fn(spot, strike, rate, vol + dv, time);
    let p_vd = price_fn(spot, strike, rate, (vol - dv).max(1.0e-8), time);
    let vega = (p_vu - p_vd) / (2.0 * dv);

    let p_ru = price_fn(spot, strike, rate + dr, vol, time);
    let p_rd = price_fn(spot, strike, rate - dr, vol, time);
    let rho = (p_ru - p_rd) / (2.0 * dr);

    let p_tu = price_fn(spot, strike, rate, vol, (time - dt).max(1.0e-8));
    let theta = (p_tu - price) / dt;

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

fn price_asian(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;
    let fixing_freq = req_str(args, "fixing_freq")?;
    let num_paths = opt_usize(args, "num_paths", 50_000)?;

    let option_type = parse_option_type(is_call);
    let per_year = parse_fixing_frequency_per_year(fixing_freq)?;
    let steps = ((time.max(1.0e-6) * per_year as f64).round() as usize).max(1);

    let (price, stderr) = arithmetic_asian_price_mc(
        option_type,
        AsianStrike::Fixed(strike),
        spot,
        rate,
        vol,
        time,
        steps,
        num_paths,
        42,
    );

    Ok(json!({ "price": price, "stderr": stderr }))
}

fn price_barrier(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let barrier = req_f64(args, "barrier")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;
    let barrier_type = req_str(args, "barrier_type")?;
    let num_paths = opt_usize(args, "num_paths", 50_000)?;

    let option_type = parse_option_type(is_call);
    let (style, direction) = parse_barrier_type(barrier_type)?;
    let steps = ((time.max(1.0e-6) * 252.0).round() as usize).max(8);

    let (price, stderr) = barrier_price_mc(
        option_type,
        style,
        direction,
        spot,
        strike,
        barrier,
        rate,
        vol,
        time,
        steps,
        num_paths,
        42,
    );

    Ok(json!({ "price": price, "stderr": stderr }))
}

fn price_autocallable_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let coupon_barrier = req_f64(args, "coupon_barrier")?;
    let ki_barrier = req_f64(args, "ki_barrier")?;
    let coupon_rate = req_f64(args, "coupon_rate")?;
    let time = req_f64(args, "time")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let num_paths = opt_usize(args, "num_paths", 40_000)?;

    let freq_per_year = parse_freq_per_year(args)?;
    let dates = schedule_times(time, freq_per_year);

    let instrument = Autocallable {
        underlyings: vec![0],
        notional: 1.0,
        autocall_dates: dates,
        autocall_barrier: coupon_barrier,
        coupon_rate,
        ki_barrier,
        ki_strike: 1.0,
        maturity: time,
    };

    let corr = vec![vec![1.0]];
    let n_steps = ((time.max(1.0e-6) * 252.0).round() as usize).max(8);
    let priced = price_autocallable(
        &instrument,
        &[spot],
        &[vol],
        &corr,
        rate,
        0.0,
        num_paths,
        n_steps,
    );

    Ok(json!({ "price": priced.price }))
}

fn price_basket_tool(args: &Value) -> ToolCallResult {
    let spots = req_array_f64(args, "spots")?;
    let vols = req_array_f64(args, "vols")?;
    let corr = req_matrix_f64(args, "correlation")?;
    let strike = req_f64(args, "strike")?;
    let time = req_f64(args, "time")?;
    let rate = req_f64(args, "rate")?;
    let is_call = req_bool(args, "is_call")?;
    let num_paths = opt_usize(args, "num_paths", 60_000)?;

    if spots.len() != vols.len() {
        return Err("spots and vols length mismatch".to_string());
    }

    let n = spots.len();
    if n == 0 {
        return Err("spots cannot be empty".to_string());
    }

    let weights = vec![1.0 / n as f64; n];
    let dividends = vec![0.0; n];

    let instrument = BasketOption {
        weights,
        strike,
        maturity: time,
        is_call,
        basket_type: BasketType::Average,
    };

    let priced = price_basket_mc(
        &instrument,
        &spots,
        &vols,
        &corr,
        rate,
        &dividends,
        num_paths,
    );

    Ok(json!({ "price": priced.price, "stderr": priced.stderr }))
}

fn price_bermudan_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;
    let exercise_dates = req_array_f64(args, "exercise_dates")?;
    let num_paths = opt_usize(args, "num_paths", 50_000)?;

    let option_type = parse_option_type(is_call);
    let steps = ((time.max(1.0e-6) * 252.0).round() as usize).max(16);

    let exercise_steps = exercise_dates
        .iter()
        .map(|t| ((*t / time.max(1.0e-8) * steps as f64).round() as usize).min(steps))
        .collect::<Vec<_>>();

    let price = longstaff_schwartz_bermudan(
        option_type,
        spot,
        strike,
        rate,
        vol,
        time,
        steps,
        &exercise_steps,
        num_paths,
        42,
    );

    Ok(json!({ "price": price }))
}

fn price_digital_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let is_call = req_bool(args, "is_call")?;

    let market = Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(0.0)
        .flat_vol(vol)
        .build()
        .map_err(|e| e.to_string())?;

    let instrument = CashOrNothingOption::new(parse_option_type(is_call), strike, 1.0, time);
    let result = DigitalAnalyticEngine::new()
        .price(&instrument, &market)
        .map_err(|e| e.to_string())?;

    Ok(json!({ "price": result.price }))
}

fn price_range_accrual_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let lower = req_f64(args, "lower")?;
    let upper = req_f64(args, "upper")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let coupon = req_f64(args, "coupon")?;
    let num_paths = opt_usize(args, "num_paths", 20_000)?;

    let fixings = ((time * 252.0).round() as usize).max(1);
    let fixing_times = (1..=fixings)
        .map(|i| i as f64 * time / fixings as f64)
        .collect::<Vec<_>>();

    let instrument = RangeAccrual {
        notional: 1.0,
        coupon_rate: coupon,
        lower_bound: lower,
        upper_bound: upper,
        fixing_times,
        payment_time: time,
    };

    let result = range_accrual_mc_price(&instrument, spot, 1.0, spot, vol, rate, num_paths, 42)
        .map_err(|e| e.to_string())?;

    Ok(json!({ "price": result.price, "stderr": result.std_error }))
}

fn price_tarf_tool(args: &Value) -> ToolCallResult {
    let spot = req_f64(args, "spot")?;
    let strike = req_f64(args, "strike")?;
    let rate = req_f64(args, "rate")?;
    let vol = req_f64(args, "vol")?;
    let time = req_f64(args, "time")?;
    let leverage = req_f64(args, "leverage")?;
    let ki_barrier = req_f64(args, "ki_barrier")?;
    let num_fixings = opt_usize(args, "num_fixings", 52)?;
    let num_paths = opt_usize(args, "num_paths", 20_000)?;

    let fixing_times = (1..=num_fixings)
        .map(|i| i as f64 * time / num_fixings as f64)
        .collect::<Vec<_>>();

    let tarf = Tarf::standard(
        strike,
        1.0,
        ki_barrier,
        1.0,
        leverage.max(0.0),
        fixing_times,
    );

    let result =
        tarf_mc_price(&tarf, spot, rate, 0.0, vol, num_paths, 42).map_err(|e| e.to_string())?;

    Ok(json!({ "price": result.price, "stderr": result.std_error }))
}

fn parse_barrier_type(value: &str) -> Result<(BarrierStyle, BarrierDirection), String> {
    match value.to_ascii_lowercase().as_str() {
        "up_in" | "up-and-in" | "ui" => Ok((BarrierStyle::In, BarrierDirection::Up)),
        "up_out" | "up-and-out" | "uo" => Ok((BarrierStyle::Out, BarrierDirection::Up)),
        "down_in" | "down-and-in" | "di" => Ok((BarrierStyle::In, BarrierDirection::Down)),
        "down_out" | "down-and-out" | "do" => Ok((BarrierStyle::Out, BarrierDirection::Down)),
        _ => Err(format!(
            "unsupported barrier_type `{value}` (expected up_in|up_out|down_in|down_out)"
        )),
    }
}

fn parse_freq_per_year(args: &Value) -> Result<usize, String> {
    let freq_value = super::req_value(args, "freq")?;
    if let Some(v) = freq_value.as_u64() {
        return usize::try_from(v).map_err(|_| "freq is too large".to_string());
    }

    let freq = freq_value
        .as_str()
        .ok_or_else(|| "freq must be a string or integer".to_string())?;

    let per_year = match freq.to_ascii_lowercase().as_str() {
        "annual" | "yearly" => 1,
        "semiannual" | "semi-annual" | "6m" => 2,
        "quarterly" | "3m" => 4,
        "monthly" | "1m" => 12,
        "weekly" => 52,
        _ => {
            let f = super::parse_frequency(freq)?;
            frequency_to_per_year(f)
        }
    };

    Ok(per_year)
}

fn schedule_times(maturity: f64, freq_per_year: usize) -> Vec<f64> {
    let n = ((maturity * freq_per_year as f64).round() as usize).max(1);
    (1..=n)
        .map(|i| i as f64 * maturity / n as f64)
        .collect::<Vec<_>>()
}
