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
        ((spot / strike).ln() + (rate - dividend_yield + 0.5 * vol * vol) * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;
    (d1, d2)
}

fn cash_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64, cash: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => cash,
        OptionType::Put if spot < strike => cash,
        _ => 0.0,
    }
}

fn asset_or_nothing_expiry(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call if spot > strike => spot,
        OptionType::Put if spot < strike => spot,
        _ => 0.0,
    }
}

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

        let price = match instrument.option_type {
            OptionType::Call => instrument.cash * df_r * normal_cdf(d2),
            OptionType::Put => instrument.cash * df_r * normal_cdf(-d2),
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

        let price = match instrument.option_type {
            OptionType::Call => market.spot * df_q * normal_cdf(d1),
            OptionType::Put => market.spot * df_q * normal_cdf(-d1),
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

        let price = match instrument.option_type {
            OptionType::Call => {
                market.spot * df_q * normal_cdf(d1)
                    - instrument.payoff_strike * df_r * normal_cdf(d2)
            }
            OptionType::Put => {
                instrument.payoff_strike * df_r * normal_cdf(-d2)
                    - market.spot * df_q * normal_cdf(-d1)
            }
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
