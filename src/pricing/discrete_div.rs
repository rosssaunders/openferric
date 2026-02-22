//! Module `pricing::discrete_div`.
//!
//! Implements discrete div workflows with concrete routines such as `escrowed_dividend_adjusted_spot`, `european_price_discrete_div`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: free functions `escrowed_dividend_adjusted_spot`, `european_price_discrete_div`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::core::OptionType;
use crate::pricing::european::black_scholes_price;

/// Escrowed-dividend spot adjustment: subtract PV of dividends paid before expiry.
///
/// Parameters:
/// - `spot`: current stock spot.
/// - `rate`: continuously compounded discount rate.
/// - `expiry`: option maturity in years.
/// - `dividends`: `(time, amount)` cash dividends.
///
/// Edge cases:
/// - Dividends with non-positive payment times or after expiry are ignored.
///
/// # Examples
/// ```rust
/// use openferric::pricing::discrete_div::escrowed_dividend_adjusted_spot;
///
/// let adj = escrowed_dividend_adjusted_spot(100.0, 0.05, 1.0, &[(0.5, 2.0), (1.5, 2.0)]);
/// assert!(adj < 100.0);
/// ```
pub fn escrowed_dividend_adjusted_spot(
    spot: f64,
    rate: f64,
    expiry: f64,
    dividends: &[(f64, f64)],
) -> f64 {
    let pv_dividends = dividends
        .iter()
        .filter(|(time, _)| *time > 0.0 && *time <= expiry)
        .map(|(time, amount)| amount * (-rate * *time).exp())
        .sum::<f64>();

    spot - pv_dividends
}

/// European call under Black-Scholes-Merton with escrowed discrete dividends.
///
/// The spot is adjusted by PV of future discrete dividends, then priced as vanilla BSM.
///
/// Parameters:
/// - `spot`, `strike`, `r`, `vol`, `t`: standard BSM inputs.
/// - `dividends`: discrete dividend schedule before expiry.
///
/// Edge cases:
/// - Returns `0.0` if the adjusted spot is non-positive.
///
/// # Examples
/// ```rust
/// use openferric::pricing::discrete_div::european_price_discrete_div;
///
/// let px = european_price_discrete_div(100.0, 100.0, 0.03, 0.20, 1.0, &[(0.25, 1.0), (0.75, 1.0)]);
/// assert!(px.is_finite() && px > 0.0);
/// ```
pub fn european_price_discrete_div(
    spot: f64,
    strike: f64,
    r: f64,
    vol: f64,
    t: f64,
    dividends: &[(f64, f64)],
) -> f64 {
    let adjusted_spot = escrowed_dividend_adjusted_spot(spot, r, t, dividends);
    if adjusted_spot <= 0.0 {
        return 0.0;
    }

    black_scholes_price(OptionType::Call, adjusted_spot, strike, r, vol, t)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

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
}
