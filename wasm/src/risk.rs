use wasm_bindgen::prelude::*;

use crate::error::{js_error, require_open_probability};
use openferric::risk::var::historical_var;

/// Historical Value-at-Risk from a flat array of P&L returns.
#[wasm_bindgen]
pub fn var_historical(returns_flat: &[f64], confidence: f64) -> Result<f64, JsValue> {
    if returns_flat.is_empty() {
        return Err(js_error("`returns_flat` must not be empty"));
    }
    require_open_probability("confidence", confidence)?;
    Ok(historical_var(returns_flat, confidence))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_historical_matches_exact_interpolated_order_statistic() {
        let returns = [
            -0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
        ];
        let var = var_historical(&returns, 0.95).unwrap();
        // Sorted losses have 0.05 and 0.10 at ranks 8 and 9. The 95th
        // percentile is rank 8.55, hence 0.05 + 0.55 * (0.10 - 0.05).
        // Binary representation of 0.95 makes the interpolation miss the
        // decimal result by 4.17e-17 (three ulps at this scale).
        assert!((var - 0.0775).abs() <= 5.0e-17);
    }

    #[test]
    fn var_historical_confidence_levels_match_exact_order_statistics() {
        let returns = [
            -0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
        ];
        let var_90 = var_historical(&returns, 0.90).unwrap();
        let var_99 = var_historical(&returns, 0.99).unwrap();
        assert!((var_90 - 0.055).abs() <= 2.0e-17);
        assert!((var_99 - 0.0955).abs() <= 2.0e-17);
    }

    #[test]
    fn var_historical_all_positive() {
        let returns = [0.01, 0.02, 0.03, 0.04, 0.05];
        let var = var_historical(&returns, 0.95).unwrap();
        assert_eq!(var, 0.0);
    }
}
