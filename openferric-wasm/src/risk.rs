use wasm_bindgen::prelude::*;

use openferric::risk::var::historical_var;

/// Historical Value-at-Risk from a flat array of P&L returns.
#[wasm_bindgen]
pub fn var_historical(returns_flat: &[f64], confidence: f64) -> f64 {
    if returns_flat.is_empty() {
        return f64::NAN;
    }
    historical_var(returns_flat, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_historical_positive_for_losses() {
        let returns = [
            -0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
        ];
        let var = var_historical(&returns, 0.95);
        assert!(var > 0.0);
    }

    #[test]
    fn var_historical_higher_confidence_higher_var() {
        let returns = [
            -0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05,
        ];
        let var_90 = var_historical(&returns, 0.90);
        let var_99 = var_historical(&returns, 0.99);
        assert!(var_99 >= var_90);
    }

    #[test]
    fn var_historical_empty_returns_nan() {
        assert!(var_historical(&[], 0.95).is_nan());
    }

    #[test]
    fn var_historical_all_positive() {
        let returns = [0.01, 0.02, 0.03, 0.04, 0.05];
        let var = var_historical(&returns, 0.95);
        assert!(var <= 0.01);
    }
}
