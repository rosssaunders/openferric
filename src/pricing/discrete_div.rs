//! Module `pricing::discrete_div`.
//!
//! Discrete-dividend pricing helpers and bootstrap wrappers.
//!
//! This module complements [`crate::market::dividends`] by exposing
//! convenience free functions for pricing workflows.

use crate::core::OptionType;
use crate::market::{
    DividendCurveBootstrap, DividendEvent, DividendSchedule, PutCallParityQuote,
    bootstrap_dividend_curve_from_put_call_parity,
};
use crate::pricing::european::black_scholes_price;

/// Escrowed-dividend spot adjustment for a pure cash schedule.
///
/// Equivalent to the legacy API and kept for backward compatibility.
pub fn escrowed_dividend_adjusted_spot(
    spot: f64,
    rate: f64,
    expiry: f64,
    dividends: &[(f64, f64)],
) -> f64 {
    let mut events = Vec::with_capacity(dividends.len());
    for (time, amount) in dividends {
        if let Ok(event) = DividendEvent::cash(*time, *amount) {
            events.push(event);
        }
    }
    let schedule = DividendSchedule::new(events).unwrap_or_else(|_| DividendSchedule::empty());
    schedule.escrowed_spot_adjustment(spot, rate, expiry)
}

/// Escrowed-dividend spot adjustment for a mixed schedule.
#[inline]
pub fn escrowed_dividend_adjusted_spot_mixed(
    spot: f64,
    rate: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    schedule.escrowed_spot_adjustment(spot, rate, expiry)
}

/// Forward price with continuous yield plus mixed discrete dividends.
#[inline]
pub fn forward_price_discrete_div(
    spot: f64,
    rate: f64,
    continuous_dividend_yield: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    schedule.forward_price(spot, rate, continuous_dividend_yield, expiry)
}

/// Equivalent continuous dividend yield implied by mixed discrete dividends.
#[inline]
pub fn effective_dividend_yield_discrete(
    spot: f64,
    rate: f64,
    continuous_dividend_yield: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    schedule.effective_dividend_yield(spot, rate, continuous_dividend_yield, expiry)
}

/// European option price under Black-Scholes with escrowed spot adjustment.
///
/// The schedule is transformed to prepaid-forward spot and priced as vanilla
/// with zero carry. This is the classic escrowed-dividend approximation.
pub fn european_price_discrete_div(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    dividends: &[(f64, f64)],
) -> f64 {
    let adjusted_spot = escrowed_dividend_adjusted_spot(spot, rate, expiry, dividends);
    if adjusted_spot <= 0.0 {
        return 0.0;
    }
    black_scholes_price(OptionType::Call, adjusted_spot, strike, rate, vol, expiry)
}

/// European option price with mixed discrete schedule using escrowed model.
pub fn european_price_discrete_div_mixed(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    schedule: &DividendSchedule,
) -> f64 {
    let adjusted_spot = schedule.escrowed_spot_adjustment(spot, rate, expiry);
    if adjusted_spot <= 0.0 {
        return 0.0;
    }
    black_scholes_price(option_type, adjusted_spot, strike, rate, vol, expiry)
}

/// Bootstraps an implied dividend curve from put-call parity observations.
#[inline]
pub fn bootstrap_dividend_curve(
    spot: f64,
    rate: f64,
    parity_quotes: &[PutCallParityQuote],
) -> Result<DividendCurveBootstrap, String> {
    bootstrap_dividend_curve_from_put_call_parity(spot, rate, parity_quotes)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::market::{DividendKind, DividendSchedule};

    #[test]
    fn escrowed_discrete_dividend_adjusts_spot_and_call_price() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let vol = 0.20;
        let expiry = 1.0;
        let dividends = [(0.5, 2.0)];

        let adjusted_spot = escrowed_dividend_adjusted_spot(spot, rate, expiry, &dividends);
        let call = european_price_discrete_div(spot, strike, rate, vol, expiry, &dividends);

        assert_relative_eq!(adjusted_spot, 98.0494, epsilon = 2e-4);
        assert_relative_eq!(call, 9.2447, epsilon = 2e-4);
    }

    #[test]
    fn mixed_schedule_forward_consistency() {
        let schedule = DividendSchedule::new(vec![
            DividendEvent {
                time: 0.25,
                kind: DividendKind::Cash(1.0),
            },
            DividendEvent {
                time: 0.5,
                kind: DividendKind::Proportional(0.01),
            },
            DividendEvent {
                time: 0.75,
                kind: DividendKind::Cash(0.5),
            },
        ])
        .expect("valid mixed schedule");

        let fwd = forward_price_discrete_div(100.0, 0.03, 0.0, 1.0, &schedule);
        let prepaid = schedule.prepaid_forward_spot(100.0, 0.03, 0.0, 1.0);
        assert_relative_eq!(prepaid * (0.03_f64).exp(), fwd, epsilon = 1e-12);
    }
}
