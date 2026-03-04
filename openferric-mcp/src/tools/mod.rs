use chrono::NaiveDate;
use openferric::core::OptionType;
use openferric::credit::SurvivalCurve;
use openferric::rates::{
    BusinessDayConvention, DayCountConvention, Frequency, YieldCurve, YieldCurveBuilder,
};
use serde_json::{Map, Value, json};

pub mod calibration;
pub mod credit;
pub mod dsl;
pub mod market_data;
pub mod math;
pub mod models;
pub mod pricing;
pub mod rates;
pub mod risk;
pub mod vol;

#[derive(Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
}

pub type ToolCallResult = Result<Value, String>;

pub fn all_specs() -> Vec<ToolSpec> {
    let mut out = Vec::new();
    out.extend(dsl::specs());
    out.extend(pricing::specs());
    out.extend(rates::specs());
    out.extend(credit::specs());
    out.extend(vol::specs());
    out.extend(calibration::specs());
    out.extend(risk::specs());
    out.extend(models::specs());
    out.extend(market_data::specs());
    out.extend(math::specs());
    out
}

pub fn call_tool(name: &str, args: &Value) -> ToolCallResult {
    let modules: [fn(&str, &Value) -> Option<ToolCallResult>; 10] = [
        dsl::call,
        pricing::call,
        rates::call,
        credit::call,
        vol::call,
        calibration::call,
        risk::call,
        models::call,
        market_data::call,
        math::call,
    ];

    for module in modules {
        if let Some(result) = module(name, args) {
            return result;
        }
    }

    Err(format!("unknown tool: {name}"))
}

pub fn obj<'a>(value: &'a Value, ctx: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{ctx} must be a JSON object"))
}

pub fn req_value<'a>(args: &'a Value, key: &str) -> Result<&'a Value, String> {
    let map = obj(args, "arguments")?;
    map.get(key)
        .ok_or_else(|| format!("missing required parameter: {key}"))
}

pub fn req_f64(args: &Value, key: &str) -> Result<f64, String> {
    req_value(args, key)?
        .as_f64()
        .ok_or_else(|| format!("parameter `{key}` must be a number"))
}

pub fn opt_f64(args: &Value, key: &str, default: f64) -> Result<f64, String> {
    let map = obj(args, "arguments")?;
    match map.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .ok_or_else(|| format!("parameter `{key}` must be a number")),
    }
}

pub fn opt_bool(args: &Value, key: &str, default: bool) -> Result<bool, String> {
    let map = obj(args, "arguments")?;
    match map.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("parameter `{key}` must be a boolean")),
    }
}

pub fn req_bool(args: &Value, key: &str) -> Result<bool, String> {
    req_value(args, key)?
        .as_bool()
        .ok_or_else(|| format!("parameter `{key}` must be a boolean"))
}

pub fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    req_value(args, key)?
        .as_str()
        .ok_or_else(|| format!("parameter `{key}` must be a string"))
}

pub fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    obj(args, "arguments")
        .ok()
        .and_then(|m| m.get(key))
        .and_then(Value::as_str)
}

pub fn opt_usize(args: &Value, key: &str, default: usize) -> Result<usize, String> {
    let map = obj(args, "arguments")?;
    match map.get(key) {
        None => Ok(default),
        Some(v) => {
            if let Some(u) = v.as_u64() {
                usize::try_from(u).map_err(|_| format!("parameter `{key}` is too large"))
            } else {
                Err(format!("parameter `{key}` must be a non-negative integer"))
            }
        }
    }
}

pub fn req_array<'a>(args: &'a Value, key: &str) -> Result<&'a [Value], String> {
    req_value(args, key)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("parameter `{key}` must be an array"))
}

pub fn req_array_f64(args: &Value, key: &str) -> Result<Vec<f64>, String> {
    req_array(args, key)?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("parameter `{key}` must be an array of numbers"))
        })
        .collect()
}

pub fn req_matrix_f64(args: &Value, key: &str) -> Result<Vec<Vec<f64>>, String> {
    req_array(args, key)?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("parameter `{key}` must be a matrix"))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("parameter `{key}` must be a matrix of numbers"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

pub fn parse_option_type(is_call: bool) -> OptionType {
    if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    }
}

pub fn parse_frequency(value: &str) -> Result<Frequency, String> {
    match value.to_ascii_lowercase().as_str() {
        "annual" | "1y" | "yearly" => Ok(Frequency::Annual),
        "semiannual" | "semi_annual" | "semi-annual" | "6m" => Ok(Frequency::SemiAnnual),
        "quarterly" | "3m" => Ok(Frequency::Quarterly),
        "monthly" | "1m" => Ok(Frequency::Monthly),
        _ => Err(format!(
            "unsupported frequency `{value}` (expected annual|semiannual|quarterly|monthly)"
        )),
    }
}

pub fn frequency_to_per_year(freq: Frequency) -> usize {
    match freq {
        Frequency::Annual => 1,
        Frequency::SemiAnnual => 2,
        Frequency::Quarterly => 4,
        Frequency::Monthly => 12,
    }
}

pub fn parse_fixing_frequency_per_year(value: &str) -> Result<usize, String> {
    match value.to_ascii_lowercase().as_str() {
        "daily" => Ok(252),
        "weekly" => Ok(52),
        "monthly" => Ok(12),
        "quarterly" => Ok(4),
        "annual" | "yearly" => Ok(1),
        other => Err(format!(
            "unsupported fixing_freq `{other}` (expected daily|weekly|monthly|quarterly|annual)"
        )),
    }
}

pub fn parse_day_count(value: &str) -> Result<DayCountConvention, String> {
    match value.to_ascii_lowercase().as_str() {
        "act360" | "act/360" => Ok(DayCountConvention::Act360),
        "act365" | "act/365" | "act365fixed" => Ok(DayCountConvention::Act365Fixed),
        "actact" | "act/act" | "actactisda" => Ok(DayCountConvention::ActActISDA),
        "30/360" | "thirty360" => Ok(DayCountConvention::Thirty360),
        "30e/360" | "thirtye360" => Ok(DayCountConvention::ThirtyE360),
        _ => Err(format!("unsupported day count convention `{value}`")),
    }
}

pub fn parse_business_day_convention(value: &str) -> Result<BusinessDayConvention, String> {
    match value.to_ascii_lowercase().as_str() {
        "following" => Ok(BusinessDayConvention::Following),
        "modifiedfollowing" | "modified_following" | "modified-following" => {
            Ok(BusinessDayConvention::ModifiedFollowing)
        }
        "preceding" => Ok(BusinessDayConvention::Preceding),
        "modifiedpreceding" | "modified_preceding" | "modified-preceding" => {
            Ok(BusinessDayConvention::ModifiedPreceding)
        }
        "unadjusted" => Ok(BusinessDayConvention::Unadjusted),
        "nearest" => Ok(BusinessDayConvention::Nearest),
        _ => Err(format!("unsupported business-day convention `{value}`")),
    }
}

pub fn parse_date(s: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map_err(|e| format!("invalid date `{s}` (expected YYYY-MM-DD): {e}"))
}

pub fn tenor_to_years(input: &Value) -> Result<f64, String> {
    if let Some(v) = input.as_f64() {
        return Ok(v);
    }

    let s = input
        .as_str()
        .ok_or_else(|| "tenor must be a number or tenor string (e.g. `6M`, `5Y`)".to_string())?;

    let s = s.trim().to_ascii_uppercase();
    if s.is_empty() {
        return Err("empty tenor string".to_string());
    }

    let (num_part, unit) = s.split_at(s.len() - 1);
    let n: f64 = num_part
        .parse()
        .map_err(|_| format!("invalid tenor `{s}`"))?;

    let years = match unit {
        "D" => n / 365.0,
        "W" => n / 52.0,
        "M" => n / 12.0,
        "Y" => n,
        _ => return Err(format!("unsupported tenor unit `{unit}` in `{s}`")),
    };

    Ok(years)
}

pub fn tenors_from_value(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("parameter `{key}` must be an array"))?
        .iter()
        .map(tenor_to_years)
        .collect()
}

pub fn parse_curve_interpolation(
    name: Option<&str>,
) -> openferric::rates::YieldCurveInterpolationMethod {
    use openferric::rates::YieldCurveInterpolationMethod;

    match name.unwrap_or("log_linear").to_ascii_lowercase().as_str() {
        "linear_zero" | "linear_zero_rate" => YieldCurveInterpolationMethod::LinearZeroRate,
        "monotone_convex" => YieldCurveInterpolationMethod::MonotoneConvex,
        "tension_spline" => YieldCurveInterpolationMethod::TensionSpline { tension: 0.5 },
        "hermite_monotone" => YieldCurveInterpolationMethod::HermiteMonotone,
        "log_cubic_monotone" => YieldCurveInterpolationMethod::LogCubicMonotone,
        "nelson_siegel" => YieldCurveInterpolationMethod::NelsonSiegel,
        "nelson_siegel_svensson" => YieldCurveInterpolationMethod::NelsonSiegelSvensson,
        "smith_wilson" => YieldCurveInterpolationMethod::SmithWilson {
            ufr: 0.03,
            alpha: 0.1,
        },
        _ => YieldCurveInterpolationMethod::LogLinearDiscount,
    }
}

pub fn curve_to_json(curve: &YieldCurve) -> Value {
    serde_json::to_value(curve).unwrap_or_else(|_| json!({ "tenors": [] }))
}

pub fn curve_from_value(input: &Value) -> Result<YieldCurve, String> {
    if let Some(curve_json) = input.get("curve_json") {
        return curve_from_value(curve_json);
    }

    if let Ok(curve) = serde_json::from_value::<YieldCurve>(input.clone()) {
        return Ok(curve);
    }

    if let Some(obj_map) = input.as_object() {
        if let Some(points) = obj_map.get("points").and_then(Value::as_array) {
            let parsed = parse_curve_points(points)?;
            return Ok(YieldCurve::new(parsed));
        }

        if let Some(tenors_val) = obj_map.get("tenors") {
            if let Some(rates_val) = obj_map.get("rates") {
                let tenors = tenors_val
                    .as_array()
                    .ok_or_else(|| "yield_curve.tenors must be an array".to_string())?
                    .iter()
                    .map(tenor_to_years)
                    .collect::<Result<Vec<_>, _>>()?;
                let rates = rates_val
                    .as_array()
                    .ok_or_else(|| "yield_curve.rates must be an array".to_string())?
                    .iter()
                    .map(|v| {
                        v.as_f64().ok_or_else(|| {
                            "yield_curve.rates must be an array of numbers".to_string()
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                if tenors.len() != rates.len() {
                    return Err(
                        "yield_curve.tenors and yield_curve.rates length mismatch".to_string()
                    );
                }

                let deposits = tenors.into_iter().zip(rates).collect::<Vec<_>>();
                return Ok(YieldCurveBuilder::from_deposits(&deposits));
            }

            let points = tenors_val
                .as_array()
                .ok_or_else(|| "yield_curve.tenors must be an array".to_string())?;
            let parsed = parse_curve_points(points)?;
            return Ok(YieldCurve::new(parsed));
        }
    }

    if let Some(arr) = input.as_array() {
        let parsed = parse_curve_points(arr)?;
        return Ok(YieldCurve::new(parsed));
    }

    Err(
        "unable to parse yield curve; expected serialized curve or {tenors,rates} style object"
            .to_string(),
    )
}

fn parse_curve_points(points: &[Value]) -> Result<Vec<(f64, f64)>, String> {
    let mut out = Vec::with_capacity(points.len());

    for point in points {
        if let Some(arr) = point.as_array() {
            if arr.len() != 2 {
                return Err("curve points must be [tenor, discount_factor] pairs".to_string());
            }
            let t = tenor_to_years(&arr[0])?;
            let df = arr[1]
                .as_f64()
                .ok_or_else(|| "curve point discount factor must be a number".to_string())?;
            out.push((t, df));
            continue;
        }

        if let Some(obj) = point.as_object() {
            let t = if let Some(v) = obj.get("tenor") {
                tenor_to_years(v)?
            } else if let Some(v) = obj.get("t") {
                tenor_to_years(v)?
            } else {
                return Err("curve point object must contain `tenor`".to_string());
            };

            let df = obj
                .get("discount_factor")
                .or_else(|| obj.get("df"))
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    "curve point object must contain numeric `discount_factor`".to_string()
                })?;
            out.push((t, df));
            continue;
        }

        return Err("curve point must be [tenor, discount_factor] or object".to_string());
    }

    Ok(out)
}

pub fn survival_curve_from_value(input: &Value) -> Result<SurvivalCurve, String> {
    if let Ok(curve) = serde_json::from_value::<SurvivalCurve>(input.clone()) {
        return Ok(curve);
    }

    let obj_map = input
        .as_object()
        .ok_or_else(|| "survival_curve must be an object".to_string())?;

    if let Some(points) = obj_map.get("tenors").and_then(Value::as_array) {
        let mut out = Vec::with_capacity(points.len());
        for point in points {
            if let Some(arr) = point.as_array() {
                if arr.len() != 2 {
                    return Err("survival_curve.tenors entries must be [tenor, prob]".to_string());
                }
                let t = tenor_to_years(&arr[0])?;
                let p = arr[1]
                    .as_f64()
                    .ok_or_else(|| "survival probability must be numeric".to_string())?;
                out.push((t, p));
            } else {
                return Err("survival_curve.tenors entries must be [tenor, prob]".to_string());
            }
        }
        return Ok(SurvivalCurve::new(out));
    }

    let tenors = obj_map
        .get("times")
        .or_else(|| obj_map.get("tenor_times"))
        .and_then(Value::as_array)
        .ok_or_else(|| "survival_curve must contain `tenors` or `times`".to_string())?;

    let probs = obj_map
        .get("survival_probs")
        .or_else(|| obj_map.get("probs"))
        .and_then(Value::as_array)
        .ok_or_else(|| "survival_curve must contain `survival_probs`".to_string())?;

    if tenors.len() != probs.len() {
        return Err("survival curve times/probabilities length mismatch".to_string());
    }

    let mut points = Vec::with_capacity(tenors.len());
    for (t, p) in tenors.iter().zip(probs.iter()) {
        let tt = tenor_to_years(t)?;
        let pp = p
            .as_f64()
            .ok_or_else(|| "survival probability must be numeric".to_string())?;
        points.push((tt, pp));
    }

    Ok(SurvivalCurve::new(points))
}
