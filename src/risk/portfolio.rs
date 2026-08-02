//! Portfolio container and Greek-based scenario P&L aggregation.
//!
//! Core types:
//! - [`Position<I>`]: instrument payload, quantity, Greeks, spot, and implied vol.
//! - [`Portfolio<I>`]: vector of positions with additive risk aggregation.
//! - [`AggregatedGreeks`]: compact totals for delta/gamma/vega/theta.
//!
//! P&L is approximated with a second-order Greek expansion:
//! `dV ~= Delta*dS + 0.5*Gamma*dS^2 + Vega*dVol + Theta*dt`,
//! where `dS = spot * spot_shock_pct` and `dVol = implied_vol * vol_shock_pct`.
//!
//! Numerical notes: this is a local approximation around current risk inputs; large shocks,
//! jump risk, and higher-order terms are not represented. Quantities may be signed, so short
//! positions naturally invert contributions.
//!
//! References:
//! - Hull, *Options, Futures, and Other Derivatives*, Greek-based P&L approximations.
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
        assert!(quantity.is_finite(), "quantity must be finite");
        assert!(
            [greeks.delta, greeks.gamma, greeks.vega, greeks.theta]
                .iter()
                .all(|value| value.is_finite()),
            "delta, gamma, vega, and theta must be finite"
        );
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
        assert!(spot_shock_pct.is_finite(), "spot_shock_pct must be finite");
        assert!(vol_shock_pct.is_finite(), "vol_shock_pct must be finite");
        assert!(horizon_years.is_finite(), "horizon_years must be finite");
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

    #[test]
    fn add_position_and_aggregate_greeks_include_every_signed_contribution() {
        let mut portfolio = Portfolio::default();
        assert_eq!(portfolio.aggregate_greeks(), AggregatedGreeks::default());

        portfolio.add_position(Position::new(
            "first",
            2.0,
            greeks(1.5, -0.25, 4.0, -3.0),
            80.0,
            0.25,
        ));
        portfolio.add_position(Position::new(
            "second",
            -4.0,
            greeks(-0.5, 0.125, -2.0, 0.75),
            120.0,
            0.30,
        ));

        assert_eq!(portfolio.positions[0].instrument, "first");
        assert_eq!(
            portfolio.aggregate_greeks(),
            AggregatedGreeks {
                delta: 5.0,
                gamma: -1.0,
                vega: 16.0,
                theta: -9.0,
            }
        );
    }

    #[test]
    fn scenario_pnl_aggregates_position_specific_spot_and_vol_scales_exactly() {
        let portfolio = Portfolio::new(vec![
            Position::new("long", 2.0, greeks(0.5, 0.02, 3.0, -4.0), 100.0, 0.20),
            Position::new("short", -3.0, greeks(-0.25, -0.01, 2.0, 1.0), 80.0, 0.40),
        ]);

        // Position 1: 2 * (0.5*10 + 0.5*0.02*10^2 + 3*0.02 - 4*0.25) = 10.12.
        // Position 2: -3 * (-0.25*8 + 0.5*-0.01*8^2 + 2*0.04 + 1*0.25) = 5.97.
        let expected = 16.09;
        assert_relative_eq!(
            portfolio.scenario_pnl_with_horizon(0.10, 0.10, 0.25),
            expected,
            epsilon = 16.0 * f64::EPSILON
        );

        assert_eq!(
            portfolio.scenario_pnl(-0.05, -0.20),
            portfolio.scenario_pnl_with_horizon(-0.05, -0.20, 0.0)
        );
    }

    #[test]
    fn position_and_scenario_reject_non_finite_risk_inputs() {
        fn position_panics(quantity: f64, greeks: Greeks, spot: f64, implied_vol: f64) -> bool {
            std::panic::catch_unwind(|| {
                Position::new("invalid", quantity, greeks, spot, implied_vol)
            })
            .is_err()
        }

        let valid = greeks(0.5, 0.1, 2.0, -1.0);
        assert!(position_panics(f64::NAN, valid, 100.0, 0.2));
        assert!(position_panics(1.0, valid, 0.0, 0.2));
        assert!(position_panics(1.0, valid, f64::INFINITY, 0.2));
        assert!(position_panics(1.0, valid, 100.0, -f64::EPSILON));
        assert!(position_panics(1.0, valid, 100.0, f64::NAN));

        for invalid_greeks in [
            greeks(f64::NAN, 0.1, 2.0, -1.0),
            greeks(0.5, f64::INFINITY, 2.0, -1.0),
            greeks(0.5, 0.1, f64::NEG_INFINITY, -1.0),
            greeks(0.5, 0.1, 2.0, f64::NAN),
        ] {
            assert!(position_panics(1.0, invalid_greeks, 100.0, 0.2));
        }

        // Rho is not consumed by this portfolio's aggregation or scenario
        // approximation, so an unavailable rho must not reject usable risk.
        assert!(!position_panics(
            1.0,
            Greeks {
                rho: f64::NAN,
                ..valid
            },
            100.0,
            0.2,
        ));

        let portfolio = Portfolio::new(vec![Position::new("valid", 1.0, valid, 100.0, 0.2)]);
        for (spot_shock, vol_shock, horizon) in [
            (f64::NAN, 0.0, 0.0),
            (0.0, f64::INFINITY, 0.0),
            (0.0, 0.0, f64::NEG_INFINITY),
        ] {
            assert!(
                std::panic::catch_unwind(|| {
                    portfolio.scenario_pnl_with_horizon(spot_shock, vol_shock, horizon)
                })
                .is_err()
            );
        }
    }
}
