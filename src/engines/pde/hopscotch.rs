use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Hopscotch finite-difference engine for Black-Scholes PDE.
///
/// Alternates explicit and implicit updates in a checkerboard pattern.
/// At points where (i + n) is even: explicit update.
/// At points where (i + n) is odd: implicit (local 1-point solve).
/// Unconditionally stable, no tridiagonal solve needed.
#[derive(Debug, Clone)]
pub struct HopscotchEngine {
    pub time_steps: usize,
    pub space_steps: usize,
    pub s_max_multiplier: f64,
}

impl Default for HopscotchEngine {
    fn default() -> Self {
        Self {
            time_steps: 200,
            space_steps: 200,
            s_max_multiplier: 4.0,
        }
    }
}

impl HopscotchEngine {
    pub fn new(time_steps: usize, space_steps: usize) -> Self {
        Self {
            time_steps,
            space_steps,
            ..Self::default()
        }
    }

    pub fn with_s_max_multiplier(mut self, s_max_multiplier: f64) -> Self {
        self.s_max_multiplier = s_max_multiplier.max(1.0);
        self
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

fn boundary_values(
    option_type: OptionType,
    is_american: bool,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    s_max: f64,
    tau: f64,
) -> (f64, f64) {
    match (option_type, is_american) {
        (OptionType::Call, false) => {
            let lower = 0.0;
            let upper =
                (s_max * (-dividend_yield * tau).exp() - strike * (-rate * tau).exp()).max(0.0);
            (lower, upper)
        }
        (OptionType::Put, false) => {
            let lower = strike * (-rate * tau).exp();
            let upper = 0.0;
            (lower, upper)
        }
        (OptionType::Call, true) => {
            let lower = 0.0;
            let upper = (s_max - strike).max(0.0);
            (lower, upper)
        }
        (OptionType::Put, true) => {
            let lower = strike;
            let upper = 0.0;
            (lower, upper)
        }
    }
}

impl PricingEngine<VanillaOption> for HopscotchEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.time_steps == 0 || self.space_steps < 2 {
            return Err(PricingError::InvalidInput(
                "time_steps must be > 0 and space_steps must be >= 2".to_string(),
            ));
        }
        if self.s_max_multiplier <= 0.0 || !self.s_max_multiplier.is_finite() {
            return Err(PricingError::InvalidInput(
                "s_max_multiplier must be finite and > 0".to_string(),
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
        if vol <= 0.0 || !vol.is_finite() {
            return Err(PricingError::InvalidInput(
                "market volatility must be finite and > 0".to_string(),
            ));
        }

        let n_t = self.time_steps;
        let n_s = self.space_steps;
        let dt = instrument.expiry / n_t as f64;
        let s_max = self.s_max_multiplier * instrument.strike;
        let ds = s_max / n_s as f64;

        let is_american = matches!(instrument.exercise, ExerciseStyle::American);
        let bermudan_flags = match &instrument.exercise {
            ExerciseStyle::Bermudan { dates } => {
                Some(bermudan_exercise_steps(dates, instrument.expiry, n_t))
            }
            _ => None,
        };

        let mut values = vec![0.0_f64; n_s + 1];
        for (i, v) in values.iter_mut().enumerate() {
            let s = i as f64 * ds;
            *v = intrinsic(instrument.option_type, s, instrument.strike);
        }

        // Precompute coefficients for each interior point
        // L V_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}
        let interior_n = n_s - 1;
        let mut a_coeff = vec![0.0_f64; interior_n];
        let mut b_coeff = vec![0.0_f64; interior_n];
        let mut c_coeff = vec![0.0_f64; interior_n];

        for k in 0..interior_n {
            let i = k + 1;
            let s = i as f64 * ds;
            let alpha = 0.5 * vol * vol * s * s / (ds * ds);
            let beta = (market.rate - market.dividend_yield) * s / (2.0 * ds);

            a_coeff[k] = alpha - beta;
            b_coeff[k] = -2.0 * alpha - market.rate;
            c_coeff[k] = alpha + beta;
        }

        // Pre-allocate double-buffer to eliminate per-timestep clone + allocation.
        let mut next_values = vec![0.0_f64; n_s + 1];

        for n in (0..n_t).rev() {
            let tau_new = instrument.expiry - n as f64 * dt;
            let (lower_bv, upper_bv) = boundary_values(
                instrument.option_type,
                is_american,
                instrument.strike,
                market.rate,
                market.dividend_yield,
                s_max,
                tau_new,
            );

            // Copy old values then apply boundary conditions.
            next_values.copy_from_slice(&values);
            next_values[0] = lower_bv;
            next_values[n_s] = upper_bv;

            // First pass: explicit updates where (i + n) is even
            for k in 0..interior_n {
                let i = k + 1;
                if (i + n) % 2 == 0 {
                    let l_v = a_coeff[k] * values[i - 1]
                        + b_coeff[k] * values[i]
                        + c_coeff[k] * values[i + 1];
                    next_values[i] = values[i] + dt * l_v;
                }
            }

            // Second pass: implicit updates where (i + n) is odd
            for k in 0..interior_n {
                let i = k + 1;
                if (i + n) % 2 != 0 {
                    let rhs = values[i]
                        + dt * (a_coeff[k] * next_values[i - 1]
                            + c_coeff[k] * next_values[i + 1]);
                    next_values[i] = rhs / (1.0 - dt * b_coeff[k]);
                }
            }

            let can_exercise = match &instrument.exercise {
                ExerciseStyle::European => false,
                ExerciseStyle::American => true,
                ExerciseStyle::Bermudan { .. } => {
                    bermudan_flags.as_ref().is_some_and(|flags| flags[n])
                }
            };
            if can_exercise {
                for (i, v) in next_values.iter_mut().enumerate() {
                    let s = i as f64 * ds;
                    *v = v.max(intrinsic(instrument.option_type, s, instrument.strike));
                }
            }

            values.copy_from_slice(&next_values);
        }

        let price = if market.spot <= 0.0 {
            values[0]
        } else if market.spot >= s_max {
            values[n_s]
        } else {
            let x = market.spot / ds;
            let i = x.floor() as usize;
            let w = x - i as f64;
            (1.0 - w) * values[i] + w * values[i + 1]
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_time_steps", n_t as f64);
        diagnostics.insert("num_space_steps", n_s as f64);
        diagnostics.insert("s_max", s_max);
        diagnostics.insert("vol", vol);

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
    use super::*;
    use crate::core::PricingEngine;
    use crate::engines::pde::crank_nicolson::CrankNicolsonEngine;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn european_call_matches_black_scholes() {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let pde = HopscotchEngine::new(400, 400)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);

        assert!(
            (pde.price - bs).abs() <= 0.01,
            "PDE/BS mismatch: pde={} bs={}",
            pde.price,
            bs
        );
    }

    #[test]
    fn european_put_matches_black_scholes() {
        let option = VanillaOption::european_put(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let pde = HopscotchEngine::new(400, 400)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let bs = black_scholes_price(OptionType::Put, 100.0, 100.0, 0.05, 0.20, 1.0);

        assert!(
            (pde.price - bs).abs() <= 0.01,
            "PDE/BS mismatch: pde={} bs={}",
            pde.price,
            bs
        );
    }

    #[test]
    fn american_put_matches_crank_nicolson() {
        let option = VanillaOption::american_put(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.30)
            .build()
            .unwrap();

        let pde = HopscotchEngine::new(200, 200)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let cn = CrankNicolsonEngine::new(200, 200)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();

        assert!(
            (pde.price - cn.price).abs() <= 0.05,
            "Hopscotch/CN mismatch: hopscotch={} cn={}",
            pde.price,
            cn.price
        );
    }
}
