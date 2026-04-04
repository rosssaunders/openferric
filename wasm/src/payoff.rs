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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_call_pnl() {
        // Long 1 call at K=100, cost=5
        let spots = [80.0, 90.0, 100.0, 110.0, 120.0];
        let pnl = strategy_intrinsic_pnl_wasm(&spots, &[100.0], &[1.0], &[1u8], 5.0);
        assert_eq!(pnl.len(), 5);
        // Below strike: PnL = -cost
        assert!((pnl[0] - (-5.0)).abs() < 1e-10);
        assert!((pnl[1] - (-5.0)).abs() < 1e-10);
        assert!((pnl[2] - (-5.0)).abs() < 1e-10);
        // Above strike: PnL = intrinsic - cost
        assert!((pnl[3] - 5.0).abs() < 1e-10); // 110-100-5
        assert!((pnl[4] - 15.0).abs() < 1e-10); // 120-100-5
    }

    #[test]
    fn long_put_pnl() {
        // Long 1 put at K=100, cost=5
        let spots = [80.0, 90.0, 100.0, 110.0, 120.0];
        let pnl = strategy_intrinsic_pnl_wasm(&spots, &[100.0], &[1.0], &[0u8], 5.0);
        assert_eq!(pnl.len(), 5);
        assert!((pnl[0] - 15.0).abs() < 1e-10); // 100-80-5
        assert!((pnl[1] - 5.0).abs() < 1e-10); // 100-90-5
        assert!((pnl[2] - (-5.0)).abs() < 1e-10); // 0-5
        assert!((pnl[3] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn bull_call_spread() {
        // Long call K=100, short call K=110, net debit=3
        let spots = [90.0, 100.0, 105.0, 110.0, 120.0];
        let pnl =
            strategy_intrinsic_pnl_wasm(&spots, &[100.0, 110.0], &[1.0, -1.0], &[1u8, 1u8], 3.0);
        assert_eq!(pnl.len(), 5);
        // Below both strikes: max loss = -cost
        assert!((pnl[0] - (-3.0)).abs() < 1e-10);
        // Between strikes: partial gain
        assert!((pnl[2] - 2.0).abs() < 1e-10); // 5-3
        // Above both: max gain = spread width - cost = 10 - 3 = 7
        assert!((pnl[4] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn empty_spot_axis() {
        let pnl = strategy_intrinsic_pnl_wasm(&[], &[100.0], &[1.0], &[1u8], 5.0);
        assert!(pnl.is_empty());
    }
}
