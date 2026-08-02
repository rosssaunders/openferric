//! Module `instruments::weather`.
//!
//! Implements weather workflows with concrete routines such as `hdd_day`, `cdd_day`, `cumulative_hdd`, `cumulative_cdd`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `DegreeDayType`, `WeatherSwap`, `WeatherOption`, `CatastropheBond` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Degree-day index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DegreeDayType {
    /// Heating degree days: `max(base - temp, 0)`.
    HDD,
    /// Cooling degree days: `max(temp - base, 0)`.
    CDD,
}

/// Daily HDD contribution.
pub fn hdd_day(average_temperature: f64, base_temperature: f64) -> f64 {
    if !average_temperature.is_finite() || !base_temperature.is_finite() {
        return f64::NAN;
    }
    (base_temperature - average_temperature).max(0.0)
}

/// Daily CDD contribution.
pub fn cdd_day(average_temperature: f64, base_temperature: f64) -> f64 {
    if !average_temperature.is_finite() || !base_temperature.is_finite() {
        return f64::NAN;
    }
    (average_temperature - base_temperature).max(0.0)
}

/// Cumulative HDD over a temperature series.
pub fn cumulative_hdd(temperatures: &[f64], base_temperature: f64) -> f64 {
    if temperatures.is_empty() {
        return 0.0;
    }
    temperatures
        .iter()
        .map(|&t| hdd_day(t, base_temperature))
        .sum()
}

/// Cumulative CDD over a temperature series.
pub fn cumulative_cdd(temperatures: &[f64], base_temperature: f64) -> f64 {
    if temperatures.is_empty() {
        return 0.0;
    }
    temperatures
        .iter()
        .map(|&t| cdd_day(t, base_temperature))
        .sum()
}

/// Cumulative degree days for a given index type.
pub fn cumulative_degree_days(
    temperatures: &[f64],
    base_temperature: f64,
    index_type: DegreeDayType,
) -> f64 {
    match index_type {
        DegreeDayType::HDD => cumulative_hdd(temperatures, base_temperature),
        DegreeDayType::CDD => cumulative_cdd(temperatures, base_temperature),
    }
}

/// Weather swap with linear payoff on cumulative HDD/CDD.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeatherSwap {
    pub index_type: DegreeDayType,
    pub strike: f64,
    pub tick_size: f64,
    pub notional: f64,
    /// If true, receives `(index - strike)`; otherwise receives `(strike - index)`.
    pub is_payer: bool,
    pub discount_rate: f64,
    pub maturity: f64,
}

impl WeatherSwap {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather swap strike must be finite and >= 0".to_string(),
            ));
        }
        if !self.tick_size.is_finite() || self.tick_size < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather swap tick_size must be finite and >= 0".to_string(),
            ));
        }
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "weather swap notional must be finite and > 0".to_string(),
            ));
        }
        if !self.discount_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "weather swap discount_rate must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather swap maturity must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn payoff(&self, realized_index: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !realized_index.is_finite() || realized_index < 0.0 {
            return Err(PricingError::InvalidInput(
                "realized_index must be finite and >= 0".to_string(),
            ));
        }

        let signed_spread = if self.is_payer {
            realized_index - self.strike
        } else {
            self.strike - realized_index
        };
        Ok(self.notional * self.tick_size * signed_spread)
    }

    pub fn price_from_expected_index(&self, expected_index: f64) -> Result<f64, PricingError> {
        let payoff = self.payoff(expected_index)?;
        Ok((-self.discount_rate * self.maturity).exp() * payoff)
    }

    pub fn price_from_historical_indices(
        &self,
        historical_indices: &[f64],
    ) -> Result<f64, PricingError> {
        self.validate()?;
        if historical_indices.is_empty() {
            return Err(PricingError::InvalidInput(
                "historical_indices cannot be empty".to_string(),
            ));
        }
        if historical_indices
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(PricingError::InvalidInput(
                "historical_indices must contain finite values >= 0".to_string(),
            ));
        }

        let mean_index = historical_indices.iter().sum::<f64>() / historical_indices.len() as f64;
        self.price_from_expected_index(mean_index)
    }
}

impl Instrument for WeatherSwap {
    fn instrument_type(&self) -> &str {
        "WeatherSwap"
    }
}

/// Weather option on cumulative HDD/CDD priced by burn analysis.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeatherOption {
    pub index_type: DegreeDayType,
    pub option_type: OptionType,
    pub strike: f64,
    pub tick_size: f64,
    pub notional: f64,
    pub discount_rate: f64,
    pub maturity: f64,
}

impl WeatherOption {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather option strike must be finite and >= 0".to_string(),
            ));
        }
        if !self.tick_size.is_finite() || self.tick_size < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather option tick_size must be finite and >= 0".to_string(),
            ));
        }
        if !self.notional.is_finite() || self.notional <= 0.0 {
            return Err(PricingError::InvalidInput(
                "weather option notional must be finite and > 0".to_string(),
            ));
        }
        if !self.discount_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "weather option discount_rate must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "weather option maturity must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn payoff(&self, realized_index: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !realized_index.is_finite() || realized_index < 0.0 {
            return Err(PricingError::InvalidInput(
                "realized_index must be finite and >= 0".to_string(),
            ));
        }

        let intrinsic = match self.option_type {
            OptionType::Call => (realized_index - self.strike).max(0.0),
            OptionType::Put => (self.strike - realized_index).max(0.0),
        };

        Ok(self.notional * self.tick_size * intrinsic)
    }

    /// Burn analysis price from historical index realizations.
    pub fn price_burn_analysis(&self, historical_indices: &[f64]) -> Result<f64, PricingError> {
        self.validate()?;
        if historical_indices.is_empty() {
            return Err(PricingError::InvalidInput(
                "historical_indices cannot be empty".to_string(),
            ));
        }
        if historical_indices
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(PricingError::InvalidInput(
                "historical_indices must contain finite values >= 0".to_string(),
            ));
        }

        let mean_payoff = historical_indices
            .iter()
            .map(|&idx| self.payoff(idx))
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum::<f64>()
            / historical_indices.len() as f64;

        Ok((-self.discount_rate * self.maturity).exp() * mean_payoff)
    }

    /// Burn analysis price directly from historical daily temperature samples.
    pub fn price_burn_from_temperature_history(
        &self,
        historical_temperature_paths: &[Vec<f64>],
        base_temperature: f64,
    ) -> Result<f64, PricingError> {
        self.validate()?;
        if !base_temperature.is_finite() {
            return Err(PricingError::InvalidInput(
                "base_temperature must be finite".to_string(),
            ));
        }
        if historical_temperature_paths.is_empty() {
            return Err(PricingError::InvalidInput(
                "historical_temperature_paths cannot be empty".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(historical_temperature_paths.len());
        for temps in historical_temperature_paths {
            if temps.is_empty() || temps.iter().any(|t| !t.is_finite()) {
                return Err(PricingError::InvalidInput(
                    "each temperature path must be non-empty with finite values".to_string(),
                ));
            }
            indices.push(cumulative_degree_days(
                temps,
                base_temperature,
                self.index_type,
            ));
        }

        self.price_burn_analysis(&indices)
    }
}

impl Instrument for WeatherOption {
    fn instrument_type(&self) -> &str {
        "WeatherOption"
    }
}

/// Catastrophe bond with coupon and principal at risk under a Poisson loss model.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CatastropheBond {
    pub principal: f64,
    pub coupon_rate: f64,
    pub risk_free_rate: f64,
    pub maturity: f64,
    pub coupon_frequency: usize,
    /// Poisson intensity `lambda` (events per year).
    pub loss_intensity: f64,
    /// Expected principal loss fraction per event in `[0, 1]`.
    pub expected_loss_per_event: f64,
}

impl CatastropheBond {
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.principal.is_finite() || self.principal <= 0.0 {
            return Err(PricingError::InvalidInput(
                "cat bond principal must be finite and > 0".to_string(),
            ));
        }
        if !self.coupon_rate.is_finite() || self.coupon_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "cat bond coupon_rate must be finite and >= 0".to_string(),
            ));
        }
        if !self.risk_free_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "cat bond risk_free_rate must be finite".to_string(),
            ));
        }
        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "cat bond maturity must be finite and > 0".to_string(),
            ));
        }
        if self.coupon_frequency == 0 {
            return Err(PricingError::InvalidInput(
                "cat bond coupon_frequency must be > 0".to_string(),
            ));
        }
        if !self.loss_intensity.is_finite() || self.loss_intensity < 0.0 {
            return Err(PricingError::InvalidInput(
                "cat bond loss_intensity must be finite and >= 0".to_string(),
            ));
        }
        if !self.expected_loss_per_event.is_finite()
            || !(0.0..=1.0).contains(&self.expected_loss_per_event)
        {
            return Err(PricingError::InvalidInput(
                "cat bond expected_loss_per_event must be finite in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }

    /// Expected surviving principal fraction at time `t`.
    fn expected_survival_fraction(&self, t: f64) -> f64 {
        (-self.loss_intensity * self.expected_loss_per_event * t).exp()
    }

    /// Present value under expected-loss approximation from Poisson jumps.
    pub fn price(&self) -> Result<f64, PricingError> {
        self.validate()?;

        let freq = self.coupon_frequency as f64;
        let dt = 1.0 / freq;
        let n_coupons = (self.maturity * freq).round() as usize;

        let mut pv = 0.0;
        for i in 1..=n_coupons {
            let t = (i as f64 * dt).min(self.maturity);
            let survival = self.expected_survival_fraction(t);
            let expected_principal = self.principal * survival;
            let coupon = expected_principal * self.coupon_rate * dt;
            let df = (-self.risk_free_rate * t).exp();
            pv += df * coupon;
        }

        let principal_survival = self.expected_survival_fraction(self.maturity);
        let principal_redemption = self.principal * principal_survival;
        pv += (-self.risk_free_rate * self.maturity).exp() * principal_redemption;

        Ok(pv)
    }
}

impl Instrument for CatastropheBond {
    fn instrument_type(&self) -> &str {
        "CatastropheBond"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdd_cdd_daily_values_are_non_negative() {
        assert_eq!(hdd_day(40.0, 65.0), 25.0);
        assert_eq!(hdd_day(70.0, 65.0), 0.0);
        assert_eq!(cdd_day(80.0, 65.0), 15.0);
        assert_eq!(cdd_day(50.0, 65.0), 0.0);
    }

    #[test]
    fn cat_bond_price_decreases_with_intensity() {
        let low = CatastropheBond {
            principal: 100.0,
            coupon_rate: 0.08,
            risk_free_rate: 0.03,
            maturity: 3.0,
            coupon_frequency: 4,
            loss_intensity: 0.2,
            expected_loss_per_event: 0.5,
        }
        .price()
        .unwrap();

        let high = CatastropheBond {
            principal: 100.0,
            coupon_rate: 0.08,
            risk_free_rate: 0.03,
            maturity: 3.0,
            coupon_frequency: 4,
            loss_intensity: 1.0,
            expected_loss_per_event: 0.5,
        }
        .price()
        .unwrap();

        assert!(high < low);
    }

    #[test]
    fn cumulative_degree_day_dispatch_matches_exact_daily_sums() {
        let temperatures = [60.0, 65.0, 70.0, 80.0];
        assert_eq!(cumulative_hdd(&temperatures, 65.0), 5.0);
        assert_eq!(cumulative_cdd(&temperatures, 65.0), 20.0);
        assert_eq!(
            cumulative_degree_days(&temperatures, 65.0, DegreeDayType::HDD),
            5.0
        );
        assert_eq!(
            cumulative_degree_days(&temperatures, 65.0, DegreeDayType::CDD),
            20.0
        );
        assert_eq!(cumulative_hdd(&[], 65.0), 0.0);
        assert_eq!(cumulative_cdd(&[], 65.0), 0.0);
    }

    #[test]
    fn weather_swap_payer_receiver_and_historical_prices_match_cashflows() {
        let payer = WeatherSwap {
            index_type: DegreeDayType::CDD,
            strike: 100.0,
            tick_size: 20.0,
            notional: 5.0,
            is_payer: true,
            discount_rate: 0.03,
            maturity: 0.5,
        };
        let receiver = WeatherSwap {
            is_payer: false,
            ..payer
        };
        assert_eq!(payer.payoff(112.0).unwrap(), 1_200.0);
        assert_eq!(receiver.payoff(112.0).unwrap(), -1_200.0);

        let historical = [90.0, 100.0, 125.0];
        let expected_index = historical.iter().sum::<f64>() / historical.len() as f64;
        let expected = 5.0 * 20.0 * (expected_index - 100.0) * (-0.03_f64 * 0.5).exp();
        assert!(
            (payer.price_from_historical_indices(&historical).unwrap() - expected).abs()
                <= 8.0 * f64::EPSILON * expected.abs().max(1.0)
        );
        assert_eq!(payer.instrument_type(), "WeatherSwap");
    }

    #[test]
    fn weather_put_burn_analysis_matches_temperature_path_cashflows() {
        let option = WeatherOption {
            index_type: DegreeDayType::CDD,
            option_type: OptionType::Put,
            strike: 12.0,
            tick_size: 10.0,
            notional: 2.0,
            discount_rate: 0.04,
            maturity: 0.25,
        };
        assert_eq!(option.payoff(7.0).unwrap(), 100.0);
        assert_eq!(option.payoff(15.0).unwrap(), 0.0);

        let paths = vec![vec![70.0, 68.0], vec![80.0, 60.0]];
        // CDD indices versus 65F are 8 and 15, hence put payoffs 80 and 0.
        let expected = 40.0 * (-0.04_f64 * 0.25).exp();
        let actual = option
            .price_burn_from_temperature_history(&paths, 65.0)
            .unwrap();
        assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
        assert_eq!(option.instrument_type(), "WeatherOption");
    }

    #[test]
    fn zero_loss_cat_bond_matches_exact_discounted_coupon_strip() {
        let bond = CatastropheBond {
            principal: 1_000.0,
            coupon_rate: 0.08,
            risk_free_rate: 0.03,
            maturity: 2.0,
            coupon_frequency: 2,
            loss_intensity: 0.0,
            expected_loss_per_event: 0.75,
        };
        let coupon = 1_000.0 * 0.08 / 2.0;
        let expected = (1..=4)
            .map(|period| coupon * (-0.03 * period as f64 / 2.0).exp())
            .sum::<f64>()
            + 1_000.0 * (-0.03_f64 * 2.0).exp();
        let actual = bond.price().unwrap();
        assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
        assert_eq!(bond.instrument_type(), "CatastropheBond");
    }

    #[test]
    fn weather_products_reject_empty_and_non_finite_histories() {
        let swap = WeatherSwap {
            index_type: DegreeDayType::HDD,
            strike: 100.0,
            tick_size: 1.0,
            notional: 1.0,
            is_payer: true,
            discount_rate: 0.0,
            maturity: 1.0,
        };
        assert!(swap.price_from_historical_indices(&[]).is_err());
        assert!(swap.price_from_historical_indices(&[f64::NAN]).is_err());

        let option = WeatherOption {
            index_type: DegreeDayType::CDD,
            option_type: OptionType::Call,
            strike: 100.0,
            tick_size: 1.0,
            notional: 1.0,
            discount_rate: 0.0,
            maturity: 1.0,
        };
        assert!(option.price_burn_analysis(&[]).is_err());
        assert!(
            option
                .price_burn_from_temperature_history(&[vec![]], 65.0)
                .is_err()
        );
        assert!(
            option
                .price_burn_from_temperature_history(&[vec![60.0]], f64::NAN)
                .is_err()
        );
    }
}
