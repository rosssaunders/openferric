use wasm_bindgen::prelude::*;

/// Strategy intrinsic PnL at expiry for a set of option legs across a spot axis.
#[wasm_bindgen]
pub fn strategy_intrinsic_pnl_wasm(
    spot_axis: &[f64],
    strikes: &[f64],
    quantities: &[f64],
    is_calls: &[u8],
    total_cost: f64,
) -> Vec<f64> {
    openferric::pricing::payoff::strategy_intrinsic_pnl(
        spot_axis, strikes, quantities, is_calls, total_cost,
    )
}
