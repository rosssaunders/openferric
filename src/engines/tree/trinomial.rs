use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Recombining trinomial tree engine for vanilla options.
#[derive(Debug, Clone)]
pub struct TrinomialTreeEngine {
    /// Number of tree steps.
    pub steps: usize,
}

impl TrinomialTreeEngine {
    /// Creates a trinomial tree engine with the provided number of steps.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

fn bermudan_exercise_steps(dates: &[f64], expiry: f64, steps: usize) -> Vec<bool> {
    let mut flags = vec![false; steps + 1];
    for &t in dates {
        if expiry <= 0.0 {
            continue;
        }
        let idx = ((t / expiry) * steps as f64).round() as usize;
        flags[idx.min(steps)] = true;
    }
    flags[steps] = true;
    flags
}

impl PricingEngine<VanillaOption> for TrinomialTreeEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "trinomial steps must be > 0".to_string(),
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
        let u = (vol * (2.0 * dt).sqrt()).exp();
        let d = 1.0 / u;
        let m = 1.0;

        let nu = market.rate - market.dividend_yield;
        let a = (nu * dt / 2.0).exp();
        let b = (vol * (dt / 2.0).sqrt()).exp();
        let denom = b - 1.0 / b;
        if denom.abs() <= 1.0e-14 {
            return Err(PricingError::NumericalError(
                "trinomial denominator is near zero".to_string(),
            ));
        }

        let pu = ((a - 1.0 / b) / denom).powi(2);
        let pd = ((b - a) / denom).powi(2);
        let pm = 1.0 - pu - pd;

        if pu < -1.0e-12
            || pm < -1.0e-12
            || pd < -1.0e-12
            || !pu.is_finite()
            || !pm.is_finite()
            || !pd.is_finite()
        {
            return Err(PricingError::NumericalError(
                "trinomial probabilities are invalid".to_string(),
            ));
        }

        let pu = pu.max(0.0);
        let pm = pm.max(0.0);
        let pd = pd.max(0.0);
        let disc = (-market.rate * dt).exp();

        let bermudan_flags = match &instrument.exercise {
            ExerciseStyle::Bermudan { dates } => Some(bermudan_exercise_steps(
                dates,
                instrument.expiry,
                self.steps,
            )),
            _ => None,
        };

        let mut values = vec![0.0_f64; 2 * self.steps + 1];
        for j in -(self.steps as isize)..=(self.steps as isize) {
            let idx = (j + self.steps as isize) as usize;
            let st = market.spot * u.powf(j as f64);
            values[idx] = intrinsic(instrument.option_type, st, instrument.strike);
        }

        for i in (0..self.steps).rev() {
            let mut next_values = vec![0.0_f64; 2 * i + 1];
            for j in -(i as isize)..=(i as isize) {
                let next_shift = i + 1;
                let up_idx = (j + next_shift as isize + 1) as usize;
                let mid_idx = (j + next_shift as isize) as usize;
                let down_idx = (j + next_shift as isize - 1) as usize;

                let continuation =
                    disc * (pu * values[up_idx] + pm * m * values[mid_idx] + pd * values[down_idx]);

                let can_exercise = match &instrument.exercise {
                    ExerciseStyle::European => false,
                    ExerciseStyle::American => true,
                    ExerciseStyle::Bermudan { .. } => {
                        bermudan_flags.as_ref().is_some_and(|flags| flags[i])
                    }
                };

                let idx = (j + i as isize) as usize;
                if can_exercise {
                    let st = market.spot * u.powf(j as f64);
                    let exercise = intrinsic(instrument.option_type, st, instrument.strike);
                    next_values[idx] = continuation.max(exercise);
                } else {
                    next_values[idx] = continuation;
                }
            }
            values = next_values;
        }

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("vol", vol);
        diagnostics.insert("u", u);
        diagnostics.insert("d", d);
        diagnostics.insert("pu", pu);
        diagnostics.insert("pm", pm);
        diagnostics.insert("pd", pd);

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
    use crate::core::PricingEngine;
    use crate::engines::tree::binomial::BinomialTreeEngine;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn american_put_matches_binomial_within_tolerance() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.30)
            .build()
            .unwrap();
        let option = VanillaOption::american_put(100.0, 1.0);

        let tri = TrinomialTreeEngine::new(200)
            .price(&option, &market)
            .unwrap();
        let bin = BinomialTreeEngine::new(200)
            .price(&option, &market)
            .unwrap();

        assert!(
            (tri.price - bin.price).abs() <= 0.05,
            "trinomial/binomial mismatch: tri={} bin={}",
            tri.price,
            bin.price
        );
    }

    #[test]
    fn european_call_converges_faster_than_binomial_at_same_steps() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(100.0, 1.0);
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);

        let steps = 50;
        let tri = TrinomialTreeEngine::new(steps)
            .price(&option, &market)
            .unwrap();
        let bin = BinomialTreeEngine::new(steps)
            .price(&option, &market)
            .unwrap();

        let tri_err = (tri.price - bs).abs();
        let bin_err = (bin.price - bs).abs();
        assert!(
            tri_err <= bin_err,
            "expected trinomial error <= binomial error, tri_err={} bin_err={}",
            tri_err,
            bin_err
        );
    }
}
