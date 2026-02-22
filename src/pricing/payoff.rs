//! Module `pricing::payoff`.
//!
//! Implements payoff workflows with concrete routines such as `strategy_intrinsic_pnl`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: free functions `strategy_intrinsic_pnl`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.

/// Strategy intrinsic PnL at expiry for a set of option legs across a spot axis.
///
/// - `spot_axis`: spot prices to evaluate
/// - `strikes`: strike per leg
/// - `quantities`: signed quantity per leg (+buy, -sell)
/// - `is_calls`: 1 = call, 0 = put per leg
/// - `total_cost`: net premium paid (subtracted from payoff)
///
/// Returns PnL for each spot point.
pub fn strategy_intrinsic_pnl(
    spot_axis: &[f64],
    strikes: &[f64],
    quantities: &[f64],
    is_calls: &[u8],
    total_cost: f64,
) -> Vec<f64> {
    let n_legs = strikes.len();
    let mut out = Vec::with_capacity(spot_axis.len());

    for &s in spot_axis {
        let mut pnl = 0.0;
        for j in 0..n_legs {
            let payoff = if is_calls[j] == 1 {
                (s - strikes[j]).max(0.0)
            } else {
                (strikes[j] - s).max(0.0)
            };
            pnl += quantities[j] * payoff;
        }
        out.push(pnl - total_cost);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_call() {
        let spots = vec![90.0, 100.0, 110.0, 120.0];
        let strikes = vec![100.0];
        let quantities = vec![1.0];
        let is_calls = vec![1u8];
        let premium = 5.0;

        let pnl = strategy_intrinsic_pnl(&spots, &strikes, &quantities, &is_calls, premium);
        assert_eq!(pnl.len(), 4);
        assert!((pnl[0] - (-5.0)).abs() < 1e-12); // OTM: 0 - 5
        assert!((pnl[1] - (-5.0)).abs() < 1e-12); // ATM: 0 - 5
        assert!((pnl[2] - 5.0).abs() < 1e-12); // ITM: 10 - 5
        assert!((pnl[3] - 15.0).abs() < 1e-12); // deep ITM: 20 - 5
    }

    #[test]
    fn test_long_put() {
        let spots = vec![80.0, 90.0, 100.0, 110.0];
        let strikes = vec![100.0];
        let quantities = vec![1.0];
        let is_calls = vec![0u8];
        let premium = 5.0;

        let pnl = strategy_intrinsic_pnl(&spots, &strikes, &quantities, &is_calls, premium);
        assert!((pnl[0] - 15.0).abs() < 1e-12); // deep ITM: 20 - 5
        assert!((pnl[1] - 5.0).abs() < 1e-12); // ITM: 10 - 5
        assert!((pnl[2] - (-5.0)).abs() < 1e-12); // ATM: 0 - 5
        assert!((pnl[3] - (-5.0)).abs() < 1e-12); // OTM: 0 - 5
    }

    #[test]
    fn test_bull_call_spread() {
        // Buy 100 call, sell 110 call
        let spots = vec![90.0, 100.0, 105.0, 110.0, 120.0];
        let strikes = vec![100.0, 110.0];
        let quantities = vec![1.0, -1.0];
        let is_calls = vec![1u8, 1u8];
        let premium = 3.0;

        let pnl = strategy_intrinsic_pnl(&spots, &strikes, &quantities, &is_calls, premium);
        assert!((pnl[0] - (-3.0)).abs() < 1e-12); // both OTM: 0 - 3
        assert!((pnl[1] - (-3.0)).abs() < 1e-12); // both ATM/OTM: 0 - 3
        assert!((pnl[2] - 2.0).abs() < 1e-12); // long ITM: 5 - 3
        assert!((pnl[3] - 7.0).abs() < 1e-12); // max: 10 - 3
        assert!((pnl[4] - 7.0).abs() < 1e-12); // capped: 10 - 3
    }
}
