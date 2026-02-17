use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Generalized cost-of-carry CRR binomial tree engine.
///
/// Uses cost-of-carry parameter `b` to handle different underlying types:
/// - Equity with continuous dividend: `b = r - q`
/// - Currency options: `b = r_d - r_f`
/// - Futures options: `b = 0` (Black-76 equivalent)
/// - Commodity with storage cost: `b = r - q + c`
#[derive(Debug, Clone)]
pub struct GeneralizedBinomialEngine {
    /// Number of tree steps.
    pub steps: usize,
    /// Cost-of-carry parameter b.
    pub cost_of_carry: f64,
}

impl GeneralizedBinomialEngine {
    /// Creates a generalized binomial engine with the given steps and cost-of-carry.
    pub fn new(steps: usize, cost_of_carry: f64) -> Self {
        Self {
            steps,
            cost_of_carry,
        }
    }

    /// Creates an engine for futures options (b = 0).
    pub fn futures(steps: usize) -> Self {
        Self::new(steps, 0.0)
    }

    /// Creates an engine for currency options (b = r_d - r_f).
    pub fn currency(steps: usize, domestic_rate: f64, foreign_rate: f64) -> Self {
        Self::new(steps, domestic_rate - foreign_rate)
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

impl PricingEngine<VanillaOption> for GeneralizedBinomialEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "binomial steps must be > 0".to_string(),
            ));
        }

        if instrument.expiry == 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
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

        let dt = instrument.expiry / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let growth = (self.cost_of_carry * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-market.rate * dt).exp();

        let mut values = vec![0.0_f64; self.steps + 1];
        for (j, value) in values.iter_mut().enumerate().take(self.steps + 1) {
            let st = market.spot * u.powf(j as f64) * d.powf((self.steps - j) as f64);
            *value = intrinsic(instrument.option_type, st, instrument.strike);
        }

        for i in (0..self.steps).rev() {
            for j in 0..=i {
                let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);

                let can_exercise = match &instrument.exercise {
                    ExerciseStyle::European => false,
                    ExerciseStyle::American => true,
                    ExerciseStyle::Bermudan { .. } => false,
                };

                if can_exercise {
                    let st = market.spot * u.powf(j as f64) * d.powf((i - j) as f64);
                    let exercise = intrinsic(instrument.option_type, st, instrument.strike);
                    values[j] = continuation.max(exercise);
                } else {
                    values[j] = continuation;
                }
            }
        }

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("vol", vol);

        Ok(PricingResult {
            price: values[0],
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{OptionType, PricingEngine};
    use crate::engines::analytic::black76::black76_price;
    use crate::engines::analytic::fx::GarmanKohlhagenEngine;
    use crate::engines::tree::binomial::BinomialTreeEngine;
    use crate::instruments::fx::FxOption;
    use crate::instruments::vanilla::VanillaOption;
    use crate::market::Market;
    use approx::assert_relative_eq;

    #[test]
    fn futures_option_matches_black76() {
        // Futures option: b = 0, spot = forward price
        let forward = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let vol = 0.20;
        let t = 1.0;

        let analytic = black76_price(OptionType::Call, forward, strike, r, vol, t).unwrap();

        let market = Market::builder()
            .spot(forward)
            .rate(r)
            .dividend_yield(0.0)
            .flat_vol(vol)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(strike, t);
        let tree_price = GeneralizedBinomialEngine::futures(500)
            .price(&option, &market)
            .unwrap()
            .price;

        assert_relative_eq!(tree_price, analytic, epsilon = 0.05);
    }

    #[test]
    fn currency_option_matches_garman_kohlhagen() {
        let spot = 1.25;
        let strike = 1.30;
        let r_d = 0.05;
        let r_f = 0.03;
        let vol = 0.10;
        let t = 0.5;

        let fx_option = FxOption::new(OptionType::Call, r_d, r_f, spot, strike, vol, t);
        let dummy_market = Market::builder()
            .spot(1.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let analytic = GarmanKohlhagenEngine::new()
            .price(&fx_option, &dummy_market)
            .unwrap()
            .price;

        // For the tree: treat spot_fx as spot, r_d as rate, b = r_d - r_f
        let market = Market::builder()
            .spot(spot)
            .rate(r_d)
            .dividend_yield(0.0)
            .flat_vol(vol)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(strike, t);
        let tree_price = GeneralizedBinomialEngine::currency(500, r_d, r_f)
            .price(&option, &market)
            .unwrap()
            .price;

        assert_relative_eq!(tree_price, analytic, epsilon = 0.01);
    }

    #[test]
    fn equity_with_dividends_matches_standard_binomial() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.25)
            .build()
            .unwrap();
        let option = VanillaOption::american_put(100.0, 1.0);
        let steps = 300;

        let standard = BinomialTreeEngine::new(steps)
            .price(&option, &market)
            .unwrap()
            .price;

        // b = r - q for equity with continuous dividend
        let generalized =
            GeneralizedBinomialEngine::new(steps, market.rate - market.dividend_yield)
                .price(&option, &market)
                .unwrap()
                .price;

        assert_relative_eq!(standard, generalized, epsilon = 1e-10);
    }
}
