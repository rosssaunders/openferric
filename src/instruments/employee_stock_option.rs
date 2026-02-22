//! Module `instruments::employee_stock_option`.
//!
//! Implements employee stock option abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `EmployeeStockOption` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.
use crate::core::{Instrument, OptionType, PricingError};

/// Employee stock option valued with a binomial tree and Hull-White style assumptions.
#[derive(Debug, Clone, PartialEq)]
pub struct EmployeeStockOption {
    /// Call/put side (ESOs are typically calls but both are supported).
    pub option_type: OptionType,
    /// Contract strike.
    pub strike: f64,
    /// Contractual maturity in years.
    pub maturity: f64,
    /// Vesting period in years (no exercise before vesting).
    pub vesting_period: f64,
    /// Expected life in years used to truncate contractual maturity.
    pub expected_life: f64,
    /// Optional forced exercise boundary multiple `M` where exercise occurs when `S > M*K`.
    pub early_exercise_multiple: Option<f64>,
    /// Annualized forfeiture/turnover intensity.
    pub forfeiture_rate: f64,
    /// Shares outstanding (`N`) for dilution adjustment.
    pub shares_outstanding: f64,
    /// Options granted (`M`) for dilution adjustment.
    pub options_granted: f64,
}

impl EmployeeStockOption {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        option_type: OptionType,
        strike: f64,
        maturity: f64,
        vesting_period: f64,
        expected_life: f64,
        early_exercise_multiple: Option<f64>,
        forfeiture_rate: f64,
        shares_outstanding: f64,
        options_granted: f64,
    ) -> Self {
        Self {
            option_type,
            strike,
            maturity,
            vesting_period,
            expected_life,
            early_exercise_multiple,
            forfeiture_rate,
            shares_outstanding,
            options_granted,
        }
    }

    pub fn validate(&self) -> Result<(), PricingError> {
        if self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "eso strike must be > 0".to_string(),
            ));
        }
        if self.maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "eso maturity must be >= 0".to_string(),
            ));
        }
        if self.vesting_period < 0.0 || self.vesting_period > self.maturity {
            return Err(PricingError::InvalidInput(
                "eso vesting_period must be in [0, maturity]".to_string(),
            ));
        }
        if self.expected_life <= 0.0 {
            return Err(PricingError::InvalidInput(
                "eso expected_life must be > 0".to_string(),
            ));
        }
        if self.forfeiture_rate < 0.0 {
            return Err(PricingError::InvalidInput(
                "eso forfeiture_rate must be >= 0".to_string(),
            ));
        }
        if self.shares_outstanding <= 0.0 {
            return Err(PricingError::InvalidInput(
                "eso shares_outstanding must be > 0".to_string(),
            ));
        }
        if self.options_granted < 0.0 {
            return Err(PricingError::InvalidInput(
                "eso options_granted must be >= 0".to_string(),
            ));
        }
        if self.early_exercise_multiple.is_some_and(|m| m <= 0.0) {
            return Err(PricingError::InvalidInput(
                "eso early_exercise_multiple must be > 0 when provided".to_string(),
            ));
        }

        Ok(())
    }

    pub fn effective_maturity(&self) -> f64 {
        if self.maturity <= 0.0 {
            return 0.0;
        }

        let truncated = self.maturity.min(self.expected_life.max(1e-8));
        truncated.max(self.vesting_period)
    }

    pub fn dilution_factor(&self) -> f64 {
        let denom = self.shares_outstanding + self.options_granted;
        if denom <= 0.0 {
            1.0
        } else {
            (self.shares_outstanding / denom).clamp(0.0, 1.0)
        }
    }

    pub fn attrition_factor(&self) -> f64 {
        (-self.forfeiture_rate * self.effective_maturity()).exp()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn price_binomial(
        &self,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        vol: f64,
        steps: usize,
    ) -> Result<f64, PricingError> {
        self.validate()?;

        if spot <= 0.0 {
            return Err(PricingError::InvalidInput("spot must be > 0".to_string()));
        }
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput("vol must be > 0".to_string()));
        }
        if steps == 0 {
            return Err(PricingError::InvalidInput("steps must be > 0".to_string()));
        }

        let t = self.effective_maturity();
        if t <= 0.0 {
            return Ok(intrinsic(self.option_type, spot, self.strike) * self.dilution_factor());
        }

        let dt = t / steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let growth = ((rate - dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);

        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }

        let disc = (-rate * dt).exp();
        let vesting_step = if self.vesting_period <= 0.0 {
            0
        } else {
            ((self.vesting_period / t) * steps as f64).ceil() as usize
        };

        let mut values = vec![0.0_f64; steps + 1];
        for (j, value) in values.iter_mut().enumerate() {
            let st = spot * u.powf(j as f64) * d.powf((steps - j) as f64);
            *value = intrinsic(self.option_type, st, self.strike);
        }

        for i in (0..steps).rev() {
            for j in 0..=i {
                let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
                let st = spot * u.powf(j as f64) * d.powf((i - j) as f64);
                let exercise = intrinsic(self.option_type, st, self.strike);

                let can_exercise = i >= vesting_step;
                values[j] = if !can_exercise {
                    continuation
                } else {
                    match self.early_exercise_multiple {
                        Some(multiple) => {
                            let boundary_hit = match self.option_type {
                                OptionType::Call => st > multiple * self.strike,
                                OptionType::Put => st < self.strike / multiple,
                            };

                            if boundary_hit { exercise } else { continuation }
                        }
                        None => continuation.max(exercise),
                    }
                };
            }
        }

        let diluted = values[0] * self.dilution_factor();
        Ok(diluted * self.attrition_factor())
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

impl Instrument for EmployeeStockOption {
    fn instrument_type(&self) -> &str {
        "EmployeeStockOption"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilution_and_attrition_reduce_value() {
        let base = EmployeeStockOption::new(
            OptionType::Call,
            100.0,
            5.0,
            1.0,
            4.0,
            Some(1.8),
            0.0,
            1000.0,
            0.0,
        );

        let diluted = EmployeeStockOption::new(
            OptionType::Call,
            100.0,
            5.0,
            1.0,
            4.0,
            Some(1.8),
            0.08,
            1000.0,
            200.0,
        );

        let v_base = base.price_binomial(100.0, 0.03, 0.0, 0.3, 600).unwrap();
        let v_diluted = diluted.price_binomial(100.0, 0.03, 0.0, 0.3, 600).unwrap();

        assert!(v_diluted < v_base);
    }
}
