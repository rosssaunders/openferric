//! Instrument definition for Commodity contracts.
//!
//! Module openferric::instruments::commodity contains payoff parameters and validation logic.

use crate::core::{Instrument, OptionType, PricingError};
use crate::engines::analytic::{black76_price, kirk_spread_price};
use crate::instruments::{FuturesOption, SpreadOption};

/// Commodity forward contract priced with continuous carry and convenience yield.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommodityForward {
    pub spot: f64,
    pub strike: f64,
    pub notional: f64,
    pub risk_free_rate: f64,
    pub storage_cost: f64,
    pub convenience_yield: f64,
    pub maturity: f64,
    pub is_long: bool,
}

impl CommodityForward {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.spot.is_finite() || self.spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity forward spot must be finite and > 0".to_string(),
            ));
        }
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity forward strike must be finite and > 0".to_string(),
            ));
        }
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity forward notional must be finite and > 0".to_string(),
            ));
        }
        if !self.risk_free_rate.is_finite()
            || !self.storage_cost.is_finite()
            || !self.convenience_yield.is_finite()
        {
            return Err(PricingError::InvalidInput(
                "commodity forward rates/carry fields must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity forward maturity must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Theoretical forward level `F0 = S0 * exp((r + u - y) * T)`.
    pub fn theoretical_forward_price(&self) -> f64 {
        self.spot
            * ((self.risk_free_rate + self.storage_cost - self.convenience_yield) * self.maturity)
                .exp()
    }

    /// Present value of a newly priced contract versus its strike.
    pub fn present_value(&self) -> Result<f64, PricingError> {
        self.validate()?;

        let sign = if self.is_long { 1.0 } else { -1.0 };
        let df = (-self.risk_free_rate * self.maturity).exp();
        Ok(sign * self.notional * df * (self.theoretical_forward_price() - self.strike))
    }

    /// Mark-to-market from an observed forward level for the same maturity.
    pub fn mark_to_market(&self, market_forward: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !market_forward.is_finite() || market_forward <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market_forward must be finite and > 0".to_string(),
            ));
        }

        let sign = if self.is_long { 1.0 } else { -1.0 };
        let df = (-self.risk_free_rate * self.maturity).exp();
        Ok(sign * self.notional * df * (market_forward - self.strike))
    }
}

impl Instrument for CommodityForward {
    fn instrument_type(&self) -> &str {
        "CommodityForward"
    }
}

/// Commodity futures contract marked-to-market daily.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommodityFutures {
    pub contract_price: f64,
    pub contract_size: f64,
    pub is_long: bool,
}

impl CommodityFutures {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.contract_price.is_finite() || self.contract_price <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity futures contract_price must be finite and > 0".to_string(),
            ));
        }
        if !self.contract_size.is_finite() || self.contract_size <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity futures contract_size must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// PnL for a given mark price.
    pub fn value(&self, mark_price: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !mark_price.is_finite() || mark_price <= 0.0 {
            return Err(PricingError::InvalidInput(
                "mark_price must be finite and > 0".to_string(),
            ));
        }

        let sign = if self.is_long { 1.0 } else { -1.0 };
        Ok(sign * self.contract_size * (mark_price - self.contract_price))
    }
}

impl Instrument for CommodityFutures {
    fn instrument_type(&self) -> &str {
        "CommodityFutures"
    }
}

/// Commodity option on a forward/futures level priced with Black-76.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommodityOption {
    pub forward: f64,
    pub strike: f64,
    pub vol: f64,
    pub risk_free_rate: f64,
    pub maturity: f64,
    pub notional: f64,
    pub option_type: OptionType,
}

impl CommodityOption {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "commodity option notional must be finite and > 0".to_string(),
            ));
        }

        let black_input = self.as_futures_option();
        black_input.validate()
    }

    pub fn as_futures_option(&self) -> FuturesOption {
        FuturesOption::new(
            self.forward,
            self.strike,
            self.vol,
            self.risk_free_rate,
            self.maturity,
            self.option_type,
        )
    }

    pub fn price_black76(&self) -> Result<f64, PricingError> {
        self.validate()?;
        let unit_price = black76_price(
            self.option_type,
            self.forward,
            self.strike,
            self.risk_free_rate,
            self.vol,
            self.maturity,
        )?;
        Ok(self.notional * unit_price)
    }
}

impl Instrument for CommodityOption {
    fn instrument_type(&self) -> &str {
        "CommodityOption"
    }
}

/// Commodity spread option priced with Kirk approximation.
///
/// Payoff convention: `max(q1 * F1 - q2 * F2 - K, 0)` for calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommoditySpreadOption {
    pub option_type: OptionType,
    pub forward_1: f64,
    pub forward_2: f64,
    pub strike: f64,
    pub quantity_1: f64,
    pub quantity_2: f64,
    pub vol_1: f64,
    pub vol_2: f64,
    pub rho: f64,
    pub risk_free_rate: f64,
    pub maturity: f64,
    pub notional: f64,
}

impl CommoditySpreadOption {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.forward_1.is_finite() || self.forward_1 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "forward_1 must be finite and > 0".to_string(),
            ));
        }
        if !self.forward_2.is_finite() || self.forward_2 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "forward_2 must be finite and > 0".to_string(),
            ));
        }
        if !self.strike.is_finite() || self.strike < 0.0 {
            return Err(PricingError::InvalidInput(
                "strike must be finite and >= 0".to_string(),
            ));
        }
        if !self.quantity_1.is_finite() || self.quantity_1 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "quantity_1 must be finite and > 0".to_string(),
            ));
        }
        if !self.quantity_2.is_finite() || self.quantity_2 <= 0.0 {
            return Err(PricingError::InvalidInput(
                "quantity_2 must be finite and > 0".to_string(),
            ));
        }
        if !self.vol_1.is_finite()
            || self.vol_1 < 0.0
            || !self.vol_2.is_finite()
            || self.vol_2 < 0.0
        {
            return Err(PricingError::InvalidInput(
                "vol_1 and vol_2 must be finite and >= 0".to_string(),
            ));
        }
        if !self.rho.is_finite() || !(-1.0..=1.0).contains(&self.rho) {
            return Err(PricingError::InvalidInput(
                "rho must be finite and in [-1, 1]".to_string(),
            ));
        }
        if !self.risk_free_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "risk_free_rate must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "maturity must be finite and >= 0".to_string(),
            ));
        }
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "notional must be finite and > 0".to_string(),
            ));
        }

        Ok(())
    }

    fn as_kirk_input(&self) -> SpreadOption {
        SpreadOption {
            s1: self.quantity_1 * self.forward_1,
            s2: self.quantity_2 * self.forward_2,
            k: self.strike,
            vol1: self.vol_1,
            vol2: self.vol_2,
            rho: self.rho,
            // For options on forwards/futures, using q = r keeps forwards unchanged.
            q1: self.risk_free_rate,
            q2: self.risk_free_rate,
            r: self.risk_free_rate,
            t: self.maturity,
        }
    }

    pub fn price_kirk(&self) -> Result<f64, PricingError> {
        self.validate()?;

        let spread = self.as_kirk_input();
        let call = kirk_spread_price(&spread)?;

        let unit_price = match self.option_type {
            OptionType::Call => call,
            OptionType::Put => {
                let df = (-self.risk_free_rate * self.maturity).exp();
                call - df * (spread.s1 - spread.s2 - spread.k)
            }
        };

        Ok(self.notional * unit_price)
    }

    /// Crack spread constructor: `max(refined_ratio * refined - crude_ratio * crude - K, 0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn crack_spread(
        option_type: OptionType,
        refined_forward: f64,
        crude_forward: f64,
        strike: f64,
        refined_ratio: f64,
        crude_ratio: f64,
        vol_refined: f64,
        vol_crude: f64,
        rho: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
    ) -> Self {
        Self {
            option_type,
            forward_1: refined_forward,
            forward_2: crude_forward,
            strike,
            quantity_1: refined_ratio,
            quantity_2: crude_ratio,
            vol_1: vol_refined,
            vol_2: vol_crude,
            rho,
            risk_free_rate,
            maturity,
            notional,
        }
    }

    /// Spark spread constructor: `max(power - heat_rate * gas - K, 0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn spark_spread(
        option_type: OptionType,
        power_forward: f64,
        gas_forward: f64,
        strike: f64,
        heat_rate: f64,
        vol_power: f64,
        vol_gas: f64,
        rho: f64,
        risk_free_rate: f64,
        maturity: f64,
        notional: f64,
    ) -> Self {
        Self {
            option_type,
            forward_1: power_forward,
            forward_2: gas_forward,
            strike,
            quantity_1: 1.0,
            quantity_2: heat_rate,
            vol_1: vol_power,
            vol_2: vol_gas,
            rho,
            risk_free_rate,
            maturity,
            notional,
        }
    }
}

impl Instrument for CommoditySpreadOption {
    fn instrument_type(&self) -> &str {
        "CommoditySpreadOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commodity_option_black76_price_is_positive() {
        let option = CommodityOption {
            forward: 100.0,
            strike: 95.0,
            vol: 0.30,
            risk_free_rate: 0.03,
            maturity: 1.0,
            notional: 10_000.0,
            option_type: OptionType::Call,
        };

        let price = option.price_black76().unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn crack_spread_prices_with_kirk() {
        let spread = CommoditySpreadOption::crack_spread(
            OptionType::Call,
            95.0,
            88.0,
            2.0,
            2.0,
            1.0,
            0.30,
            0.25,
            0.6,
            0.03,
            0.75,
            1_000.0,
        );

        let price = spread.price_kirk().unwrap();
        assert!(price > 0.0);
    }
}
