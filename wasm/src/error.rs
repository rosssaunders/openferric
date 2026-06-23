use wasm_bindgen::prelude::*;

#[inline]
pub(crate) fn js_error(message: impl Into<String>) -> JsValue {
    JsValue::from_str(&message.into())
}

pub(crate) fn batch_length_error(
    primary_name: &str,
    primary_len: usize,
    others: &[(&str, usize)],
) -> Result<(), String> {
    for (name, len) in others {
        if *len != primary_len {
            return Err(format!(
                "batch input length mismatch: `{name}` has {len} entries, expected {primary_len} to match `{primary_name}`"
            ));
        }
    }
    Ok(())
}

pub(crate) fn check_batch_lengths(
    primary_name: &str,
    primary_len: usize,
    others: &[(&str, usize)],
) -> Result<(), JsValue> {
    batch_length_error(primary_name, primary_len, others).map_err(js_error)
}

#[inline]
pub(crate) fn require_finite(name: &str, value: f64) -> Result<(), JsValue> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(js_error(format!("`{name}` must be finite")))
    }
}

#[inline]
pub(crate) fn require_positive(name: &str, value: f64) -> Result<(), JsValue> {
    require_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(js_error(format!("`{name}` must be positive")))
    }
}

#[inline]
pub(crate) fn require_non_negative(name: &str, value: f64) -> Result<(), JsValue> {
    require_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(js_error(format!("`{name}` must be non-negative")))
    }
}

#[inline]
pub(crate) fn require_probability(name: &str, value: f64) -> Result<(), JsValue> {
    require_finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(js_error(format!("`{name}` must be between 0 and 1")))
    }
}

#[inline]
pub(crate) fn require_open_probability(name: &str, value: f64) -> Result<(), JsValue> {
    require_finite(name, value)?;
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(js_error(format!(
            "`{name}` must be greater than 0 and less than 1"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_length_error_reports_mismatched_input() {
        let err = batch_length_error("spots", 2, &[("strikes", 1)]).unwrap_err();
        assert!(err.contains("strikes"));
        assert!(err.contains("spots"));
    }

    #[test]
    fn batch_length_error_accepts_matching_inputs() {
        assert!(batch_length_error("spots", 2, &[("strikes", 2), ("vols", 2)]).is_ok());
    }
}
