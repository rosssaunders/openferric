use std::collections::HashMap;
use std::sync::Arc;

use crate::core::{
    Averaging, BarrierStyle, ExerciseStyle, Instrument, OptionType, PricingEngine, PricingError,
    PricingResult, StrikeType,
};
use crate::instruments::{AsianOption, BarrierOption, VanillaOption};
use crate::market::Market;
use crate::mc::{ControlVariate, GbmPathGenerator, MonteCarloEngine};
use crate::models::Gbm;
use crate::pricing::asian::geometric_asian_discrete_fixed_closed_form;
use crate::pricing::european::black_scholes_price;

/// Variance reduction scheme.
#[derive(Debug, Clone)]
pub enum VarianceReduction {
    /// No variance reduction.
    None,
    /// Antithetic variates.
    Antithetic,
    /// Built-in control variates:
    /// - European vanilla uses Black-Scholes.
    /// - Arithmetic fixed-strike Asian uses geometric Asian.
    ControlVariate,
}

/// Instrument interface required by the generic Monte Carlo engine.
pub trait MonteCarloInstrument: Instrument {
    /// Validates instrument fields for Monte Carlo pricing.
    fn validate_for_mc(&self) -> Result<(), PricingError>;
    /// Returns maturity in years.
    fn maturity(&self) -> f64;
    /// Returns a strike-like value for volatility lookup.
    fn reference_strike(&self, spot: f64) -> f64;
    /// Computes path payoff.
    fn payoff_from_path(&self, path: &[f64]) -> f64;
    /// Optional control variate for this instrument.
    fn control_variate(&self, _market: &Market, _vol: f64) -> Option<ControlVariate> {
        None
    }
}

fn vanilla_payoff(option_type: crate::core::OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        crate::core::OptionType::Call => (spot - strike).max(0.0),
        crate::core::OptionType::Put => (strike - spot).max(0.0),
    }
}

fn black_scholes_price_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    let adjusted_spot = spot * (-dividend_yield * expiry).exp();
    black_scholes_price(option_type, adjusted_spot, strike, rate, vol, expiry)
}

fn path_hits_barrier(path: &[f64], barrier: f64, direction: crate::core::BarrierDirection) -> bool {
    match direction {
        crate::core::BarrierDirection::Up => path.iter().any(|&s| s >= barrier),
        crate::core::BarrierDirection::Down => path.iter().any(|&s| s <= barrier),
    }
}

fn average_for_observations(
    path: &[f64],
    maturity: f64,
    observation_times: &[f64],
    averaging: Averaging,
) -> f64 {
    if observation_times.is_empty() || maturity <= 0.0 {
        return path[path.len() - 1];
    }

    let last_idx = path.len().saturating_sub(1) as f64;

    match averaging {
        Averaging::Arithmetic => {
            let sum = observation_times
                .iter()
                .map(|&t| {
                    let idx = ((t / maturity) * last_idx).round() as usize;
                    path[idx.min(path.len() - 1)]
                })
                .sum::<f64>();
            sum / observation_times.len() as f64
        }
        Averaging::Geometric => {
            let mean_log = observation_times
                .iter()
                .map(|&t| {
                    let idx = ((t / maturity) * last_idx).round() as usize;
                    path[idx.min(path.len() - 1)].max(1e-12).ln()
                })
                .sum::<f64>()
                / observation_times.len() as f64;
            mean_log.exp()
        }
    }
}

impl MonteCarloInstrument for VanillaOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, _spot: f64) -> f64 {
        self.strike
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        vanilla_payoff(self.option_type, path[path.len() - 1], self.strike)
    }

    fn control_variate(&self, market: &Market, vol: f64) -> Option<ControlVariate> {
        if !matches!(self.exercise, ExerciseStyle::European) {
            return None;
        }

        let expected_discounted = black_scholes_price_with_dividend(
            self.option_type,
            market.spot,
            self.strike,
            market.rate,
            market.dividend_yield,
            vol,
            self.expiry,
        );
        let discount_factor = (-market.rate * self.expiry).exp();
        let option_type = self.option_type;
        let strike = self.strike;

        Some(ControlVariate {
            expected: expected_discounted / discount_factor,
            evaluator: Arc::new(move |path: &[f64]| {
                vanilla_payoff(option_type, path[path.len() - 1], strike)
            }),
        })
    }
}

impl MonteCarloInstrument for BarrierOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, _spot: f64) -> f64 {
        self.strike
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        let hit = path_hits_barrier(path, self.barrier.level, self.barrier.direction);
        let active = match self.barrier.style {
            BarrierStyle::In => hit,
            BarrierStyle::Out => !hit,
        };

        if active {
            vanilla_payoff(self.option_type, path[path.len() - 1], self.strike)
        } else {
            self.barrier.rebate
        }
    }
}

impl MonteCarloInstrument for AsianOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, spot: f64) -> f64 {
        match self.asian.strike_type {
            StrikeType::Fixed => self.strike,
            StrikeType::Floating => spot,
        }
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        let avg = average_for_observations(
            path,
            self.expiry,
            &self.asian.observation_times,
            self.asian.averaging,
        );
        let st = path[path.len() - 1];
        match self.asian.strike_type {
            StrikeType::Fixed => vanilla_payoff(self.option_type, avg, self.strike),
            StrikeType::Floating => vanilla_payoff(self.option_type, st, avg),
        }
    }

    fn control_variate(&self, market: &Market, vol: f64) -> Option<ControlVariate> {
        if self.asian.averaging != Averaging::Arithmetic
            || self.asian.strike_type != StrikeType::Fixed
        {
            return None;
        }

        let expected_discounted = geometric_asian_discrete_fixed_closed_form(
            self.option_type,
            market.spot,
            self.strike,
            market.rate,
            market.dividend_yield,
            vol,
            &self.asian.observation_times,
        );
        let discount_factor = (-market.rate * self.expiry).exp();
        let option_type = self.option_type;
        let strike = self.strike;
        let expiry = self.expiry;
        let observation_times = self.asian.observation_times.clone();

        Some(ControlVariate {
            expected: expected_discounted / discount_factor,
            evaluator: Arc::new(move |path: &[f64]| {
                let geometric_avg = average_for_observations(
                    path,
                    expiry,
                    &observation_times,
                    Averaging::Geometric,
                );
                vanilla_payoff(option_type, geometric_avg, strike)
            }),
        })
    }
}

/// Generic Monte Carlo pricing engine.
#[derive(Debug, Clone)]
pub struct MonteCarloPricingEngine {
    /// Number of simulated paths.
    pub num_paths: usize,
    /// Number of time steps per path.
    pub num_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Variance reduction configuration.
    pub variance_reduction: VarianceReduction,
}

impl MonteCarloPricingEngine {
    /// Creates an engine with explicit path and time-step counts.
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            variance_reduction: VarianceReduction::None,
        }
    }

    /// Sets the variance reduction scheme.
    pub fn with_variance_reduction(mut self, variance_reduction: VarianceReduction) -> Self {
        self.variance_reduction = variance_reduction;
        self
    }
}

impl<T> PricingEngine<T> for MonteCarloPricingEngine
where
    T: MonteCarloInstrument + Sync,
{
    fn price(&self, instrument: &T, market: &Market) -> Result<PricingResult, PricingError> {
        instrument.validate_for_mc()?;

        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }
        if self.num_steps == 0 {
            return Err(PricingError::InvalidInput(
                "num_steps must be > 0".to_string(),
            ));
        }

        let maturity = instrument.maturity();
        if maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "instrument maturity must be >= 0".to_string(),
            ));
        }

        if maturity == 0.0 {
            let payoff = instrument.payoff_from_path(&[market.spot]);
            return Ok(PricingResult {
                price: payoff,
                stderr: Some(0.0),
                greeks: None,
                diagnostics: HashMap::new(),
            });
        }

        let ref_strike = instrument.reference_strike(market.spot).max(1e-12);
        let vol = market.vol_for(ref_strike, maturity);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let generator = GbmPathGenerator {
            model: Gbm {
                mu: market.rate - market.dividend_yield,
                sigma: vol,
            },
            s0: market.spot,
            maturity,
            steps: self.num_steps,
        };

        let base = MonteCarloEngine::new(self.num_paths, self.seed);
        let base = match &self.variance_reduction {
            VarianceReduction::Antithetic => base.with_antithetic(true),
            _ => base.with_antithetic(false),
        };

        let engine = match &self.variance_reduction {
            VarianceReduction::ControlVariate => {
                if let Some(cv) = instrument.control_variate(market, vol) {
                    base.with_control_variate(cv)
                } else {
                    base
                }
            }
            _ => base,
        };

        let discount_factor = (-market.rate * maturity).exp();
        let (price, stderr) = engine.run(
            &generator,
            |path| instrument.payoff_from_path(path),
            discount_factor,
        );

        let mut diagnostics = HashMap::new();
        diagnostics.insert("num_paths".to_string(), self.num_paths as f64);
        diagnostics.insert("num_steps".to_string(), self.num_steps as f64);
        diagnostics.insert("vol".to_string(), vol);
        diagnostics.insert(
            "variance_reduction".to_string(),
            match self.variance_reduction {
                VarianceReduction::None => 0.0,
                VarianceReduction::Antithetic => 1.0,
                VarianceReduction::ControlVariate => 2.0,
            },
        );

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PricingEngine;
    use crate::instruments::VanillaOption;

    #[test]
    fn mc_european_call_matches_black_scholes_within_one_percent() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let result = MonteCarloPricingEngine::new(100_000, 252, 42)
            .price(&option, &market)
            .expect("mc pricing succeeds");

        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        let rel_err = ((result.price - bs) / bs).abs();
        assert!(
            rel_err <= 0.01,
            "MC/BS relative error too high: mc={} bs={} rel_err={}",
            result.price,
            bs,
            rel_err
        );
    }

    #[test]
    fn mc_antithetic_has_lower_stderr_than_plain_mc() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let plain = MonteCarloPricingEngine::new(100_000, 252, 42)
            .price(&option, &market)
            .expect("plain MC succeeds");
        let antithetic = MonteCarloPricingEngine::new(100_000, 252, 42)
            .with_variance_reduction(VarianceReduction::Antithetic)
            .price(&option, &market)
            .expect("antithetic MC succeeds");

        assert!(
            antithetic.stderr.expect("stderr present") < plain.stderr.expect("stderr present"),
            "expected antithetic stderr < plain stderr"
        );
    }

    #[test]
    fn mc_european_control_variate_is_within_half_percent_of_bs() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let result = MonteCarloPricingEngine::new(100_000, 252, 42)
            .with_variance_reduction(VarianceReduction::ControlVariate)
            .price(&option, &market)
            .expect("control-variate MC succeeds");

        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        let rel_err = ((result.price - bs) / bs).abs();
        assert!(
            rel_err <= 0.005,
            "MC/BS relative error too high with control variate: mc={} bs={} rel_err={}",
            result.price,
            bs,
            rel_err
        );
    }
}
