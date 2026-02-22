//! Risk analytics for Portfolio.
//!
//! Module openferric::risk::portfolio provides portfolio-level measures and adjustments.

use crate::core::Greeks;

/// Aggregated portfolio Greeks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AggregatedGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
}

/// Position wrapper storing instrument, quantity, and risk metadata.
#[derive(Debug, Clone)]
pub struct Position<I> {
    pub instrument: I,
    pub quantity: f64,
    pub greeks: Greeks,
    pub spot: f64,
    pub implied_vol: f64,
}

impl<I> Position<I> {
    pub fn new(instrument: I, quantity: f64, greeks: Greeks, spot: f64, implied_vol: f64) -> Self {
        assert!(
            spot.is_finite() && spot > 0.0,
            "spot must be finite and > 0"
        );
        assert!(
            implied_vol.is_finite() && implied_vol >= 0.0,
            "implied_vol must be finite and >= 0"
        );
        Self {
            instrument,
            quantity,
            greeks,
            spot,
            implied_vol,
        }
    }
}

/// Portfolio container for risk aggregation and scenario P&L.
#[derive(Debug, Clone, Default)]
pub struct Portfolio<I> {
    pub positions: Vec<Position<I>>,
}

impl<I> Portfolio<I> {
    pub fn new(positions: Vec<Position<I>>) -> Self {
        Self { positions }
    }

    pub fn add_position(&mut self, position: Position<I>) {
        self.positions.push(position);
    }

    pub fn total_delta(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.quantity * p.greeks.delta)
            .sum()
    }

    pub fn total_gamma(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.quantity * p.greeks.gamma)
            .sum()
    }

    pub fn total_vega(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.quantity * p.greeks.vega)
            .sum()
    }

    pub fn total_theta(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.quantity * p.greeks.theta)
            .sum()
    }

    pub fn aggregate_greeks(&self) -> AggregatedGreeks {
        AggregatedGreeks {
            delta: self.total_delta(),
            gamma: self.total_gamma(),
            vega: self.total_vega(),
            theta: self.total_theta(),
        }
    }

    /// Scenario P&L approximation using Delta/Gamma/Vega:
    /// dS = spot * spot_shock_pct, dVol = implied_vol * vol_shock_pct.
    pub fn scenario_pnl(&self, spot_shock_pct: f64, vol_shock_pct: f64) -> f64 {
        self.scenario_pnl_with_horizon(spot_shock_pct, vol_shock_pct, 0.0)
    }

    /// Scenario P&L with optional horizon term for theta carry (in years).
    pub fn scenario_pnl_with_horizon(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        horizon_years: f64,
    ) -> f64 {
        self.positions
            .iter()
            .map(|p| {
                let ds = p.spot * spot_shock_pct;
                let dvol = p.implied_vol * vol_shock_pct;
                let pnl = p.greeks.delta * ds
                    + 0.5 * p.greeks.gamma * ds * ds
                    + p.greeks.vega * dvol
                    + p.greeks.theta * horizon_years;
                p.quantity * pnl
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn greeks(delta: f64, gamma: f64, vega: f64, theta: f64) -> Greeks {
        Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho: 0.0,
        }
    }

    #[test]
    fn aggregates_delta_for_long_and_short_options() {
        let long_calls = Position::new("call", 100.0, greeks(0.6, 0.0, 0.0, 0.0), 100.0, 0.2);
        let short_puts = Position::new("put", -50.0, greeks(-0.4, 0.0, 0.0, 0.0), 100.0, 0.2);

        let portfolio = Portfolio::new(vec![long_calls, short_puts]);
        assert_relative_eq!(portfolio.total_delta(), 80.0, epsilon = 1.0e-12);
    }

    #[test]
    fn scenario_pnl_uses_delta_gamma_vega_and_theta() {
        let position = Position::new("option", 10.0, greeks(2.0, 1.0, 5.0, -1.0), 100.0, 0.2);
        let portfolio = Portfolio::new(vec![position]);

        let pnl = portfolio.scenario_pnl_with_horizon(0.01, 0.10, 1.0 / 252.0);
        assert_relative_eq!(pnl, 25.960_317_460_3, epsilon = 1.0e-10);
    }
}
