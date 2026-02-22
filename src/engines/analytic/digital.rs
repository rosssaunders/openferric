//! Module `engines::analytic::digital`.
//!
//! Implements digital abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `DigitalAnalyticEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
use crate::market::Market;
use crate::math::normal_cdf;

/// Analytic Black-Scholes style engine for digital/binary options.
#[derive(Debug, Clone, Default)]
pub struct DigitalAnalyticEngine;

impl DigitalAnalyticEngine {
    /// Creates a digital analytic engine.
    pub fn new() -> Self {
        Self
    }
}

#[inline]
fn d1_d2(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> (f64, f64) {
    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 =
        ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;
    (d1, d2)
}

#[inline]
fn cash_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64, cash: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => cash,
        OptionType::Put if spot < strike => cash,
        _ => 0.0,
    }
}

#[inline]
fn asset_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => spot,
        OptionType::Put if spot < strike => spot,
        _ => 0.0,
    }
}

#[inline]
fn gap_expiry(option_type: OptionType, spot: f64, payoff_strike: f64, trigger_strike: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > trigger_strike => spot - payoff_strike,
        OptionType::Put if spot < trigger_strike => payoff_strike - spot,
        _ => 0.0,
    }
}

impl PricingEngine<CashOrNothingOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &CashOrNothingOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: cash_or_nothing_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.strike,
                    instrument.cash,
                ),
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let (_, d2) = d1_d2(
            market.spot,
            instrument.strike,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.expiry,
        );
        let df_r = (-market.rate * instrument.expiry).exp();

        // Compute N(d2) once; derive put price via N(-d2) = 1 - N(d2).
        let nd2 = normal_cdf(d2);
        let price = match instrument.option_type {
            OptionType::Call => instrument.cash * df_r * nd2,
            OptionType::Put => instrument.cash * df_r * (1.0 - nd2),
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d2", d2);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<AssetOrNothingOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &AssetOrNothingOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: asset_or_nothing_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.strike,
                ),
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let (d1, _) = d1_d2(
            market.spot,
            instrument.strike,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.expiry,
        );
        let df_q = (-market.dividend_yield * instrument.expiry).exp();

        // Compute N(d1) once; derive put price via N(-d1) = 1 - N(d1).
        let nd1 = normal_cdf(d1);
        let price = match instrument.option_type {
            OptionType::Call => market.spot * df_q * nd1,
            OptionType::Put => market.spot * df_q * (1.0 - nd1),
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d1", d1);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<GapOption> for DigitalAnalyticEngine {
    fn price(
        &self,
        instrument: &GapOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.expiry <= 0.0 {
            return Ok(PricingResult {
                price: gap_expiry(
                    instrument.option_type,
                    market.spot,
                    instrument.payoff_strike,
                    instrument.trigger_strike,
                ),
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.vol_for(instrument.trigger_strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let (d1, d2) = d1_d2(
            market.spot,
            instrument.trigger_strike,
            market.rate,
            market.dividend_yield,
            vol,
            instrument.expiry,
        );
        let df_r = (-market.rate * instrument.expiry).exp();
        let df_q = (-market.dividend_yield * instrument.expiry).exp();

        // Compute N(d1), N(d2) once; derive put via N(-d) = 1 - N(d).
        let nd1 = normal_cdf(d1);
        let nd2 = normal_cdf(d2);
        let call = market.spot.mul_add(df_q * nd1, -(instrument.payoff_strike * df_r * nd2));
        let price = match instrument.option_type {
            OptionType::Call => call,
            OptionType::Put => call - market.spot * df_q + instrument.payoff_strike * df_r,
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("vol", vol);
        diagnostics.insert("d1", d1);
        diagnostics.insert("d2", d2);

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

    #[test]
    fn cash_or_nothing_matches_haug_reference() {
        let instrument = CashOrNothingOption::new(OptionType::Call, 80.0, 10.0, 0.75);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.06)
            .dividend_yield(0.06)
            .flat_vol(0.35)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(price, 6.9358, epsilon = 5e-2);
    }

    #[test]
    fn asset_or_nothing_put_matches_haug_reference_value() {
        let instrument = AssetOrNothingOption::new(OptionType::Put, 65.0, 0.50);
        let market = Market::builder()
            .spot(70.0)
            .rate(0.07)
            .dividend_yield(0.05)
            .flat_vol(0.27)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(price, 20.2069, epsilon = 2e-4);
    }

    #[test]
    fn gap_call_matches_haug_reference() {
        let instrument = GapOption::new(OptionType::Call, 57.0, 50.0, 0.50);
        let market = Market::builder()
            .spot(50.0)
            .rate(0.09)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();
        let price = DigitalAnalyticEngine::new()
            .price(&instrument, &market)
            .unwrap()
            .price;

        assert_relative_eq!(price, -0.0053, epsilon = 2e-4);
    }
}
