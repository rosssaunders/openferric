use std::cmp::Ordering;

use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::variance_swap::{VarianceOptionQuote, VarianceSwap, VolatilitySwap};
use crate::market::Market;

/// Analytic replication-based engine for variance and volatility swaps.
#[derive(Debug, Clone, Default)]
pub struct VarianceSwapEngine;

impl VarianceSwapEngine {
    /// Creates a variance/volatility swap engine.
    pub fn new() -> Self {
        Self
    }
}

fn quote_sort_cmp(lhs: &VarianceOptionQuote, rhs: &VarianceOptionQuote) -> Ordering {
    lhs.strike
        .partial_cmp(&rhs.strike)
        .unwrap_or(Ordering::Equal)
}

/// Fair annualized variance strike from static OTM option replication.
///
/// Uses
/// `K_var = (2 * exp(rT) / T) * sum_i delta_k_i * Q_otm(K_i) / K_i^2`.
pub fn fair_variance_strike_from_quotes(
    expiry: f64,
    rate: f64,
    spot: f64,
    dividend_yield: f64,
    quotes: &[VarianceOptionQuote],
) -> Result<f64, PricingError> {
    if !expiry.is_finite() || expiry <= 0.0 {
        return Err(PricingError::InvalidInput(
            "variance replication expiry must be finite and > 0".to_string(),
        ));
    }
    if !spot.is_finite() || spot <= 0.0 {
        return Err(PricingError::InvalidInput(
            "variance replication spot must be finite and > 0".to_string(),
        ));
    }
    if !rate.is_finite() || !dividend_yield.is_finite() {
        return Err(PricingError::InvalidInput(
            "variance replication rates must be finite".to_string(),
        ));
    }
    if quotes.len() < 2 {
        return Err(PricingError::InvalidInput(
            "variance replication requires at least two option quotes".to_string(),
        ));
    }

    let mut sorted_quotes = quotes.to_vec();
    for quote in &sorted_quotes {
        quote.validate()?;
    }
    sorted_quotes.sort_by(quote_sort_cmp);

    let forward = spot * ((rate - dividend_yield) * expiry).exp();
    let mut replicated_sum = 0.0;

    for i in 0..sorted_quotes.len() {
        let quote = sorted_quotes[i];
        let delta_k = if i == 0 {
            sorted_quotes[1].strike - sorted_quotes[0].strike
        } else if i + 1 == sorted_quotes.len() {
            sorted_quotes[i].strike - sorted_quotes[i - 1].strike
        } else {
            0.5 * (sorted_quotes[i + 1].strike - sorted_quotes[i - 1].strike)
        };

        if delta_k <= 0.0 {
            return Err(PricingError::InvalidInput(
                "variance replication strikes must be strictly increasing".to_string(),
            ));
        }

        let otm_price = if quote.strike < forward {
            quote.put_price
        } else if quote.strike > forward {
            quote.call_price
        } else {
            0.5 * (quote.call_price + quote.put_price)
        }
        .max(0.0);

        replicated_sum += delta_k * otm_price / (quote.strike * quote.strike);
    }

    let fair_variance = (2.0 * (rate * expiry).exp() / expiry) * replicated_sum;
    if !fair_variance.is_finite() {
        return Err(PricingError::NumericalError(
            "variance replication produced a non-finite fair variance".to_string(),
        ));
    }

    Ok(fair_variance.max(0.0))
}

/// Fair volatility strike with first-order convexity adjustment.
///
/// `K_vol ≈ sqrt(K_var) - var_of_var / (8 * K_var^(3/2))`
pub fn fair_volatility_strike_from_variance(
    fair_variance: f64,
    var_of_var: f64,
) -> Result<f64, PricingError> {
    if !fair_variance.is_finite() || fair_variance <= 0.0 {
        return Err(PricingError::InvalidInput(
            "fair_variance must be finite and > 0".to_string(),
        ));
    }
    if !var_of_var.is_finite() || var_of_var < 0.0 {
        return Err(PricingError::InvalidInput(
            "var_of_var must be finite and >= 0".to_string(),
        ));
    }

    let sqrt_k_var = fair_variance.sqrt();
    let adjustment = var_of_var / (8.0 * fair_variance.powf(1.5));
    Ok((sqrt_k_var - adjustment).max(0.0))
}

/// Mark-to-market for a variance swap using vega-notional convention.
pub fn variance_swap_mtm(
    instrument: &VarianceSwap,
    fair_variance: f64,
    rate: f64,
) -> Result<f64, PricingError> {
    instrument.validate()?;
    if !fair_variance.is_finite() || fair_variance < 0.0 {
        return Err(PricingError::InvalidInput(
            "fair_variance must be finite and >= 0".to_string(),
        ));
    }
    if !rate.is_finite() {
        return Err(PricingError::InvalidInput(
            "rate must be finite".to_string(),
        ));
    }

    let expected_realized_var = instrument.observed_realized_var.unwrap_or(fair_variance);
    let variance_notional = instrument.notional_vega / (2.0 * instrument.strike_vol);
    let payoff = variance_notional * (expected_realized_var - instrument.strike_vol.powi(2));

    Ok((-rate * instrument.expiry).exp() * payoff)
}

/// Mark-to-market for a volatility swap.
pub fn volatility_swap_mtm(
    instrument: &VolatilitySwap,
    fair_variance: f64,
    fair_volatility: f64,
    rate: f64,
) -> Result<f64, PricingError> {
    instrument.validate()?;
    if !fair_variance.is_finite() || fair_variance < 0.0 {
        return Err(PricingError::InvalidInput(
            "fair_variance must be finite and >= 0".to_string(),
        ));
    }
    if !fair_volatility.is_finite() || fair_volatility < 0.0 {
        return Err(PricingError::InvalidInput(
            "fair_volatility must be finite and >= 0".to_string(),
        ));
    }
    if !rate.is_finite() {
        return Err(PricingError::InvalidInput(
            "rate must be finite".to_string(),
        ));
    }

    let expected_realized_vol = instrument
        .observed_realized_var
        .unwrap_or(fair_variance)
        .sqrt();
    let payoff = instrument.notional_vega * (expected_realized_vol - instrument.strike_vol);

    Ok((-rate * instrument.expiry).exp() * payoff)
}

impl PricingEngine<VarianceSwap> for VarianceSwapEngine {
    fn price(
        &self,
        instrument: &VarianceSwap,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        let fair_variance = fair_variance_strike_from_quotes(
            instrument.expiry,
            market.rate,
            market.spot,
            market.dividend_yield,
            &instrument.option_quotes,
        )?;
        let fair_volatility = fair_variance.sqrt();
        let price = variance_swap_mtm(instrument, fair_variance, market.rate)?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("fair_variance", fair_variance);
        diagnostics.insert("fair_volatility", fair_volatility);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<VolatilitySwap> for VarianceSwapEngine {
    fn price(
        &self,
        instrument: &VolatilitySwap,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        let fair_variance = fair_variance_strike_from_quotes(
            instrument.expiry,
            market.rate,
            market.spot,
            market.dividend_yield,
            &instrument.option_quotes,
        )?;
        let fair_volatility =
            fair_volatility_strike_from_variance(fair_variance, instrument.var_of_var)?;
        let price = volatility_swap_mtm(instrument, fair_variance, fair_volatility, market.rate)?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("fair_variance", fair_variance);
        diagnostics.insert("fair_volatility", fair_volatility);
        diagnostics.insert("var_of_var", instrument.var_of_var);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::core::OptionType;
    use crate::pricing::european::black_scholes_price;

    fn flat_surface_quotes(
        spot: f64,
        rate: f64,
        vol: f64,
        expiry: f64,
    ) -> Vec<VarianceOptionQuote> {
        let mut quotes = Vec::new();
        for strike in 1..=500 {
            let strike = strike as f64;
            let call_price = black_scholes_price(OptionType::Call, spot, strike, rate, vol, expiry);
            let put_price = black_scholes_price(OptionType::Put, spot, strike, rate, vol, expiry);
            quotes.push(VarianceOptionQuote::new(strike, call_price, put_price));
        }
        quotes
    }

    #[test]
    fn fair_variance_and_volatility_match_flat_surface_vol() {
        let spot = 100.0;
        let rate = 0.01;
        let dividend_yield = 0.0;
        let vol = 0.20;
        let expiry = 1.0;

        let quotes = flat_surface_quotes(spot, rate, vol, expiry);
        let fair_variance =
            fair_variance_strike_from_quotes(expiry, rate, spot, dividend_yield, &quotes).unwrap();
        let fair_volatility = fair_volatility_strike_from_variance(fair_variance, 0.0).unwrap();

        assert_relative_eq!(fair_variance, 0.04, epsilon = 2e-4);
        assert_relative_eq!(fair_volatility, 0.20, epsilon = 5e-4);
    }
}
