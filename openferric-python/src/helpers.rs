use chrono::{DateTime, NaiveDateTime, SecondsFormat, Utc};
use openferric_core::core::{BarrierDirection, BarrierStyle, OptionType};
use openferric_core::market::Market;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::panic::{self, UnwindSafe};

pub(crate) fn parse_option_type(value: &str) -> Option<OptionType> {
    match value.to_ascii_lowercase().as_str() {
        "call" => Some(OptionType::Call),
        "put" => Some(OptionType::Put),
        _ => None,
    }
}

pub(crate) fn parse_barrier_style(value: &str) -> Option<BarrierStyle> {
    match value.to_ascii_lowercase().as_str() {
        "in" => Some(BarrierStyle::In),
        "out" => Some(BarrierStyle::Out),
        _ => None,
    }
}

pub(crate) fn parse_barrier_direction(value: &str) -> Option<BarrierDirection> {
    match value.to_ascii_lowercase().as_str() {
        "up" => Some(BarrierDirection::Up),
        "down" => Some(BarrierDirection::Down),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DigitalKind {
    CashOrNothing,
    AssetOrNothing,
}

pub(crate) fn parse_digital_kind(value: &str) -> Option<DigitalKind> {
    match value.to_ascii_lowercase().as_str() {
        "cash" | "cash-or-nothing" | "cash_or_nothing" => Some(DigitalKind::CashOrNothing),
        "asset" | "asset-or-nothing" | "asset_or_nothing" => Some(DigitalKind::AssetOrNothing),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SpreadMethod {
    Kirk,
    Margrabe,
}

pub(crate) fn parse_spread_method(value: &str) -> Option<SpreadMethod> {
    match value.to_ascii_lowercase().as_str() {
        "kirk" => Some(SpreadMethod::Kirk),
        "margrabe" => Some(SpreadMethod::Margrabe),
        _ => None,
    }
}

pub(crate) fn build_market(spot: f64, rate: f64, div_yield: f64, vol: f64) -> Option<Market> {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(div_yield)
        .flat_vol(vol.max(1e-8))
        .build()
        .ok()
}

pub(crate) fn tenor_grid(maturity: f64, payment_freq: usize) -> Vec<f64> {
    if maturity <= 0.0 || payment_freq == 0 {
        return vec![];
    }

    let dt = 1.0 / payment_freq as f64;
    let mut times = Vec::new();
    let mut t = 0.0;

    while t + dt < maturity - 1e-12 {
        t += dt;
        times.push(t);
    }
    times.push(maturity);
    times
}

#[inline]
pub(crate) fn intrinsic_from_option_type(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
pub(crate) fn option_price_from_call(
    option_type: OptionType,
    call_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    expiry: f64,
) -> f64 {
    match option_type {
        OptionType::Call => call_price,
        OptionType::Put => {
            call_price - spot * (-div_yield * expiry).exp() + strike * (-rate * expiry).exp()
        }
    }
}

pub(crate) fn parse_datetime(value: &str) -> PyResult<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Ok(dt.with_timezone(&Utc));
    }

    for format in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(value, format) {
            return Ok(DateTime::from_naive_utc_and_offset(naive, Utc));
        }
    }

    Err(PyValueError::new_err(format!(
        "invalid UTC datetime '{value}'; use ISO 8601 like 2026-03-27T00:00:00Z"
    )))
}

pub(crate) fn format_datetime(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Secs, true)
}

pub(crate) fn panic_to_pyerr(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return PyValueError::new_err((*message).to_string());
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return PyValueError::new_err(message.clone());
    }
    PyValueError::new_err("operation failed")
}

pub(crate) fn catch_unwind_py<T, F>(f: F) -> PyResult<T>
where
    F: FnOnce() -> T + UnwindSafe,
{
    panic::catch_unwind(f).map_err(panic_to_pyerr)
}
