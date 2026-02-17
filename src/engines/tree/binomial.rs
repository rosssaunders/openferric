use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Cox-Ross-Rubinstein binomial tree engine.
#[derive(Debug, Clone)]
pub struct BinomialTreeEngine {
    /// Number of tree steps.
    pub steps: usize,
}

impl BinomialTreeEngine {
    /// Creates a tree engine with the given number of steps.
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

impl PricingEngine<VanillaOption> for BinomialTreeEngine {
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
        let growth = ((market.rate - market.dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-market.rate * dt).exp();

        let bermudan_flags = match &instrument.exercise {
            ExerciseStyle::Bermudan { dates } => Some(bermudan_exercise_steps(
                dates,
                instrument.expiry,
                self.steps,
            )),
            _ => None,
        };

        // Multiplicative recurrence replaces O(steps^2) powf() calls with multiplications.
        // spot * u^j * d^(steps-j) = spot * d^steps * (u/d)^j
        let ratio = u / d;
        // Pre-fuse disc*p and disc*(1-p) to save one multiply per inner-loop iteration.
        let disc_p = disc * p;
        let disc_1mp = disc * (1.0 - p);

        let mut values = vec![0.0_f64; self.steps + 1];
        {
            let mut st = market.spot * d.powi(self.steps as i32);
            for j in 0..=self.steps {
                values[j] = intrinsic(instrument.option_type, st, instrument.strike);
                st *= ratio;
            }
        }

        let mut base = market.spot * d.powi((self.steps - 1) as i32);
        for i in (0..self.steps).rev() {
            let can_exercise = match &instrument.exercise {
                ExerciseStyle::European => false,
                ExerciseStyle::American => true,
                ExerciseStyle::Bermudan { .. } => {
                    bermudan_flags.as_ref().is_some_and(|flags| flags[i])
                }
            };

            if can_exercise {
                let mut st = base;
                for j in 0..=i {
                    let continuation = disc_p * values[j + 1] + disc_1mp * values[j];
                    let exercise = intrinsic(instrument.option_type, st, instrument.strike);
                    values[j] = continuation.max(exercise);
                    st *= ratio;
                }
            } else {
                for j in 0..=i {
                    values[j] = disc_p * values[j + 1] + disc_1mp * values[j];
                }
            }
            base *= u;
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
