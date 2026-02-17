use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Crank-Nicolson finite-difference engine for Black-Scholes PDE.
#[derive(Debug, Clone)]
pub struct CrankNicolsonEngine {
    /// Number of time steps.
    pub time_steps: usize,
    /// Number of space steps.
    pub space_steps: usize,
    /// Spot grid upper bound multiplier, `S_max = s_max_multiplier * K`.
    pub s_max_multiplier: f64,
}

impl Default for CrankNicolsonEngine {
    fn default() -> Self {
        Self {
            time_steps: 200,
            space_steps: 200,
            s_max_multiplier: 4.0,
        }
    }
}

impl CrankNicolsonEngine {
    /// Creates a Crank-Nicolson engine with explicit grid sizes.
    pub fn new(time_steps: usize, space_steps: usize) -> Self {
        Self {
            time_steps,
            space_steps,
            ..Self::default()
        }
    }

    /// Sets `S_max = multiplier * K`.
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

/// In-place tridiagonal solve using pre-allocated scratch buffers.
/// Writes solution into `x`. `c_star` and `d_star` are scratch space.
fn solve_tridiagonal_inplace(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
    c_star: &mut [f64],
    d_star: &mut [f64],
    x: &mut [f64],
) -> Result<(), PricingError> {
    let n = diag.len();

    let denom0 = diag[0];
    if denom0.abs() <= 1.0e-14 || !denom0.is_finite() {
        return Err(PricingError::NumericalError(
            "tridiagonal solver singular matrix".to_string(),
        ));
    }
    c_star[0] = if n > 1 { upper[0] / denom0 } else { 0.0 };
    d_star[0] = rhs[0] / denom0;

    for i in 1..n {
        let denom = diag[i] - lower[i] * c_star[i - 1];
        if denom.abs() <= 1.0e-14 || !denom.is_finite() {
            return Err(PricingError::NumericalError(
                "tridiagonal solver singular matrix".to_string(),
            ));
        }
        c_star[i] = if i < n - 1 { upper[i] / denom } else { 0.0 };
        d_star[i] = (rhs[i] - lower[i] * d_star[i - 1]) / denom;
    }

    x[n - 1] = d_star[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }
    Ok(())
}

impl PricingEngine<VanillaOption> for CrankNicolsonEngine {
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

        let interior_n = n_s - 1;
        let mut lhs_lower = vec![0.0_f64; interior_n];
        let mut lhs_diag = vec![0.0_f64; interior_n];
        let mut lhs_upper = vec![0.0_f64; interior_n];
        let mut rhs_lower = vec![0.0_f64; interior_n];
        let mut rhs_diag = vec![0.0_f64; interior_n];
        let mut rhs_upper = vec![0.0_f64; interior_n];

        for k in 0..interior_n {
            let i = k + 1;
            let s = i as f64 * ds;
            let alpha = 0.5 * vol * vol * s * s / (ds * ds);
            let beta = (market.rate - market.dividend_yield) * s / (2.0 * ds);

            let a = alpha - beta;
            let b = -2.0 * alpha - market.rate;
            let c = alpha + beta;

            lhs_lower[k] = -0.5 * dt * a;
            lhs_diag[k] = 1.0 - 0.5 * dt * b;
            lhs_upper[k] = -0.5 * dt * c;

            rhs_lower[k] = 0.5 * dt * a;
            rhs_diag[k] = 1.0 + 0.5 * dt * b;
            rhs_upper[k] = 0.5 * dt * c;
        }

        // Pre-allocate all scratch buffers once to eliminate per-timestep allocations.
        // Previous code allocated 6+ vectors per timestep (rhs, lower clone, upper clone,
        // c_star, d_star, x, next_values). For 200 timesteps that was 1400+ heap allocs.
        let mut rhs_buf = vec![0.0_f64; interior_n];
        let mut solve_lower = vec![0.0_f64; interior_n];
        let mut solve_upper = vec![0.0_f64; interior_n];
        let mut c_star = vec![0.0_f64; interior_n];
        let mut d_star = vec![0.0_f64; interior_n];
        let mut interior = vec![0.0_f64; interior_n];
        let mut next_values = vec![0.0_f64; n_s + 1];

        // Pre-copy the LHS bands with zeroed boundary entries (they never change).
        solve_lower.copy_from_slice(&lhs_lower);
        solve_lower[0] = 0.0;
        solve_upper.copy_from_slice(&lhs_upper);
        solve_upper[interior_n - 1] = 0.0;

        for n in (0..n_t).rev() {
            let tau_new = instrument.expiry - n as f64 * dt;
            let (lower_new, upper_new) = boundary_values(
                instrument.option_type,
                is_american,
                instrument.strike,
                market.rate,
                market.dividend_yield,
                s_max,
                tau_new,
            );

            for k in 0..interior_n {
                let i = k + 1;
                rhs_buf[k] = rhs_lower[k] * values[i - 1]
                    + rhs_diag[k] * values[i]
                    + rhs_upper[k] * values[i + 1];
            }

            rhs_buf[0] -= lhs_lower[0] * lower_new;
            rhs_buf[interior_n - 1] -= lhs_upper[interior_n - 1] * upper_new;

            solve_tridiagonal_inplace(
                &solve_lower,
                &lhs_diag,
                &solve_upper,
                &rhs_buf,
                &mut c_star,
                &mut d_star,
                &mut interior,
            )?;

            next_values[0] = lower_new;
            next_values[n_s] = upper_new;
            for (k, v) in interior.iter().enumerate() {
                next_values[k + 1] = *v;
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
    use crate::engines::tree::binomial::BinomialTreeEngine;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn european_call_matches_black_scholes_to_cent() {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let pde = CrankNicolsonEngine::new(200, 200)
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
    fn american_put_matches_binomial_within_ten_cents() {
        let option = VanillaOption::american_put(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.30)
            .build()
            .unwrap();

        let pde = CrankNicolsonEngine::new(200, 200)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let bin = BinomialTreeEngine::new(600)
            .price(&option, &market)
            .unwrap();

        assert!(
            (pde.price - bin.price).abs() <= 0.10,
            "PDE/binomial mismatch: pde={} bin={}",
            pde.price,
            bin.price
        );
    }
}
