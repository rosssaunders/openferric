//! Margin, liquidation-threshold, and inherent-leverage utilities for Boros-style rate swaps.
//!
//! The formulas here assume isolated margin and linear mark-to-market in the funding rate:
//! `PnL = size * (entry_rate - current_rate)`.
//! Positive `size` therefore loses money when funding rises; negative `size` loses money when
//! funding falls.

const EPSILON: f64 = 1.0e-12;

/// Margin parameters for Boros-style funding-rate positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginParams {
    pub initial_margin_ratio: f64,
    pub maintenance_margin_ratio: f64,
    /// Annualised funding-rate volatility.
    pub funding_rate_vol: f64,
    /// Remaining time to maturity in years.
    pub time_to_maturity: f64,
    /// Minimum funding-rate increment.
    pub tick_size: f64,
}

/// Margin calculator for isolated Boros-style positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MarginCalculator;

impl MarginCalculator {
    /// Computes the initial-margin requirement using a volatility-time scaling term.
    pub fn initial_margin(notional: f64, params: &MarginParams) -> f64 {
        validate_margin_params(params);
        notional.abs() * params.initial_margin_ratio * margin_scalar(params)
    }

    /// Computes the maintenance-margin requirement using a simplified vol-time rule.
    pub fn maintenance_margin(notional: f64, params: &MarginParams) -> f64 {
        validate_margin_params(params);
        notional.abs() * params.maintenance_margin_ratio * margin_scalar(params)
    }

    /// Equity divided by maintenance margin.
    pub fn health_ratio(
        collateral: f64,
        notional: f64,
        unrealized_pnl: f64,
        params: &MarginParams,
    ) -> f64 {
        assert!(
            collateral.is_finite(),
            "collateral must be finite for health-ratio calculation"
        );
        assert!(
            notional.is_finite(),
            "notional must be finite for health-ratio calculation"
        );
        assert!(
            unrealized_pnl.is_finite(),
            "unrealized_pnl must be finite for health-ratio calculation"
        );

        let maintenance_margin = Self::maintenance_margin(notional, params);
        let equity = collateral + unrealized_pnl;

        if maintenance_margin <= EPSILON {
            if equity >= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            equity / maintenance_margin
        }
    }

    /// Returns `true` when liquidation should trigger.
    #[inline]
    pub fn is_liquidatable(health_ratio: f64) -> bool {
        health_ratio <= 1.0
    }

    /// Funding rate that triggers liquidation for a signed position.
    ///
    /// Positive notional is harmed by higher funding. Negative notional is harmed by lower
    /// funding. The returned threshold is snapped to the next adverse tick.
    pub fn liquidation_rate(
        entry_rate: f64,
        collateral: f64,
        notional: f64,
        params: &MarginParams,
    ) -> f64 {
        assert!(
            entry_rate.is_finite(),
            "entry_rate must be finite for liquidation-rate calculation"
        );
        assert!(
            collateral.is_finite(),
            "collateral must be finite for liquidation-rate calculation"
        );
        assert!(
            notional.is_finite() && notional.abs() > EPSILON,
            "notional must be finite and non-zero for liquidation-rate calculation"
        );

        let maintenance_margin = Self::maintenance_margin(notional, params);
        let raw_rate = entry_rate + (collateral - maintenance_margin) / notional;

        adverse_tick_round(raw_rate, notional.signum(), params.tick_size)
    }
}

/// Simple leverage helpers for Yield Unit positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InherentLeverage;

impl InherentLeverage {
    /// Notional exposure divided by upfront Yield Unit cost.
    pub fn leverage(notional: f64, yu_cost: f64) -> f64 {
        assert!(
            notional.is_finite(),
            "notional must be finite for leverage calculation"
        );
        assert!(
            yu_cost.is_finite() && yu_cost > 0.0,
            "yu_cost must be finite and > 0 for leverage calculation"
        );
        notional.abs() / yu_cost
    }

    /// Returns linear rate-move return scaled by the embedded leverage.
    #[inline]
    pub fn leveraged_return(rate_move: f64, leverage: f64) -> f64 {
        assert!(
            rate_move.is_finite(),
            "rate_move must be finite for leveraged-return calculation"
        );
        assert!(
            leverage.is_finite() && leverage >= 0.0,
            "leverage must be finite and >= 0 for leveraged-return calculation"
        );
        rate_move * leverage
    }
}

fn margin_scalar(params: &MarginParams) -> f64 {
    (params.funding_rate_vol * params.time_to_maturity).sqrt()
}

fn adverse_tick_round(rate: f64, position_sign: f64, tick_size: f64) -> f64 {
    if !tick_size.is_finite() || tick_size <= EPSILON {
        return rate;
    }

    let ticks = rate / tick_size;
    if position_sign >= 0.0 {
        ticks.ceil() * tick_size
    } else {
        ticks.floor() * tick_size
    }
}

fn validate_margin_params(params: &MarginParams) {
    assert!(
        params.initial_margin_ratio.is_finite() && params.initial_margin_ratio >= 0.0,
        "initial_margin_ratio must be finite and >= 0"
    );
    assert!(
        params.maintenance_margin_ratio.is_finite() && params.maintenance_margin_ratio >= 0.0,
        "maintenance_margin_ratio must be finite and >= 0"
    );
    assert!(
        params.funding_rate_vol.is_finite() && params.funding_rate_vol >= 0.0,
        "funding_rate_vol must be finite and >= 0"
    );
    assert!(
        params.time_to_maturity.is_finite() && params.time_to_maturity >= 0.0,
        "time_to_maturity must be finite and >= 0"
    );
    assert!(
        params.tick_size.is_finite() && params.tick_size >= 0.0,
        "tick_size must be finite and >= 0"
    );
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn sample_params() -> MarginParams {
        MarginParams {
            initial_margin_ratio: 0.20,
            maintenance_margin_ratio: 0.10,
            funding_rate_vol: 0.16,
            time_to_maturity: 0.25,
            tick_size: 0.0001,
        }
    }

    #[test]
    fn health_ratio_equals_collateral_over_maintenance_margin_without_pnl() {
        let params = sample_params();
        let notional = 100.0;
        let collateral = 4.0;

        let maintenance_margin = MarginCalculator::maintenance_margin(notional, &params);
        let health_ratio = MarginCalculator::health_ratio(collateral, notional, 0.0, &params);

        assert_relative_eq!(
            health_ratio,
            collateral / maintenance_margin,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn liquidation_threshold_implies_unit_health_ratio() {
        let params = sample_params();
        let notional = 100.0;
        let collateral = 4.0;
        let entry_rate = 0.05;
        let liquidation_rate =
            MarginCalculator::liquidation_rate(entry_rate, collateral, notional, &params);
        let pnl = notional * (entry_rate - liquidation_rate);
        let health_ratio = MarginCalculator::health_ratio(collateral, notional.abs(), pnl, &params);

        assert!(health_ratio <= 1.0 + 1.0e-12);
        assert!(health_ratio >= 1.0 - 5.0e-3);
    }

    #[test]
    fn inherent_leverage_matches_notional_over_cost() {
        assert_relative_eq!(
            InherentLeverage::leverage(100.0, 5.0),
            20.0,
            epsilon = 1.0e-12
        );
    }
}
