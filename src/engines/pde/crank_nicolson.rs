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

#[inline(always)]
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
#[inline(always)]
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

    let inv_denom0 = 1.0 / diag[0];
    if !inv_denom0.is_finite() {
        return Err(PricingError::NumericalError(
            "tridiagonal solver singular matrix".to_string(),
        ));
    }
    c_star[0] = if n > 1 { upper[0] * inv_denom0 } else { 0.0 };
    d_star[0] = rhs[0] * inv_denom0;

    for i in 1..n {
        // denom = diag[i] - lower[i] * c_star[i-1]  →  FMA
        let denom = (-lower[i]).mul_add(c_star[i - 1], diag[i]);
        if denom.abs() <= 1.0e-14 {
            return Err(PricingError::NumericalError(
                "tridiagonal solver singular matrix".to_string(),
            ));
        }
        let inv_denom = 1.0 / denom;
        c_star[i] = if i < n - 1 { upper[i] * inv_denom } else { 0.0 };
        // (rhs[i] - lower[i] * d_star[i-1]) / denom  →  FMA + mul
        d_star[i] = (-lower[i]).mul_add(d_star[i - 1], rhs[i]) * inv_denom;
    }

    x[n - 1] = d_star[n - 1];
    for i in (0..(n - 1)).rev() {
        // d_star[i] - c_star[i] * x[i+1]  →  FMA
        x[i] = (-c_star[i]).mul_add(x[i + 1], d_star[i]);
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
        // Pre-compute exercise parameters outside the time loop so the inner
        // exercise loop is branchless (compiles to maxsd instructions).
        let is_call = matches!(instrument.option_type, OptionType::Call);
        let strike = instrument.strike;
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

        // Pre-compute reciprocals to replace per-iteration divisions with multiplications.
        let inv_ds2 = 1.0 / (ds * ds);
        let inv_2ds = 1.0 / (2.0 * ds);
        let half_vol2 = 0.5 * vol * vol;
        let drift = market.rate - market.dividend_yield;
        let half_dt = 0.5 * dt;

        for k in 0..interior_n {
            let i = k + 1;
            let s = i as f64 * ds;
            let alpha = half_vol2 * s * s * inv_ds2;
            let beta = drift * s * inv_2ds;

            let a = alpha - beta;
            let b = -2.0 * alpha - market.rate;
            let c = alpha + beta;

            lhs_lower[k] = -half_dt * a;
            lhs_diag[k] = 1.0 - half_dt * b;
            lhs_upper[k] = -half_dt * c;

            rhs_lower[k] = half_dt * a;
            rhs_diag[k] = 1.0 + half_dt * b;
            rhs_upper[k] = half_dt * c;
        }

        // Pre-allocate all scratch buffers once to eliminate per-timestep allocations.
        // Previous code allocated 6+ vectors per timestep (rhs, lower clone, upper clone,
        // c_star, d_star, x, next_values). For 200 timesteps that was 1400+ heap allocs.
        let mut rhs_buf = vec![0.0_f64; interior_n];
        let mut solve_lower = vec![0.0_f64; interior_n];
        let mut solve_upper = vec![0.0_f64; interior_n];
        let mut c_star = vec![0.0_f64; interior_n];
        let mut d_star = vec![0.0_f64; interior_n];
        // Eliminated separate `interior` buffer: tridiagonal solver writes directly
        // into next_values[1..n_s], saving one allocation and one copy_from_slice.
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

            // RHS = B * values; use FMA chains for the tridiagonal multiply.
            // SIMD path when available, otherwise scalar FMA.
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if interior_n >= 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        pde_rhs_avx2(
                            &rhs_lower, &rhs_diag, &rhs_upper,
                            &values, &mut rhs_buf, interior_n,
                        );
                    }
                } else {
                    for k in 0..interior_n {
                        let i = k + 1;
                        rhs_buf[k] = rhs_diag[k].mul_add(
                            values[i],
                            rhs_lower[k].mul_add(values[i - 1], rhs_upper[k] * values[i + 1]),
                        );
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for k in 0..interior_n {
                    let i = k + 1;
                    rhs_buf[k] = rhs_diag[k].mul_add(
                        values[i],
                        rhs_lower[k].mul_add(values[i - 1], rhs_upper[k] * values[i + 1]),
                    );
                }
            }

            rhs_buf[0] -= lhs_lower[0] * lower_new;
            rhs_buf[interior_n - 1] -= lhs_upper[interior_n - 1] * upper_new;

            next_values[0] = lower_new;
            next_values[n_s] = upper_new;

            // Solve directly into next_values[1..n_s], eliminating a separate
            // interior buffer and the copy_from_slice that followed.
            solve_tridiagonal_inplace(
                &solve_lower,
                &lhs_diag,
                &solve_upper,
                &rhs_buf,
                &mut c_star,
                &mut d_star,
                &mut next_values[1..n_s],
            )?;

            let can_exercise = match &instrument.exercise {
                ExerciseStyle::European => false,
                ExerciseStyle::American => true,
                ExerciseStyle::Bermudan { .. } => {
                    bermudan_flags.as_ref().is_some_and(|flags| flags[n])
                }
            };
            if can_exercise {
                // Branchless early-exercise: the match on is_call is hoisted
                // out of the loop by the compiler; inner body is just maxsd ops.
                for (i, v) in next_values.iter_mut().enumerate() {
                    let s = i as f64 * ds;
                    let exercise_value = if is_call {
                        (s - strike).max(0.0)
                    } else {
                        (strike - s).max(0.0)
                    };
                    *v = v.max(exercise_value);
                }
            }

            // Swap instead of copy — eliminates full memcpy per timestep.
            std::mem::swap(&mut values, &mut next_values);
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
        diagnostics.insert_key(crate::core::DiagKey::NumTimeSteps, n_t as f64);
        diagnostics.insert_key(crate::core::DiagKey::NumSpaceSteps, n_s as f64);
        diagnostics.insert_key(crate::core::DiagKey::SMax, s_max);
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

/// AVX2+FMA tridiagonal RHS multiply: rhs[k] = diag[k]*v[k+1] + lower[k]*v[k] + upper[k]*v[k+2]
/// Processes 4 elements at a time.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn pde_rhs_avx2(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    values: &[f64],
    rhs: &mut [f64],
    n: usize,
) {
    use std::arch::x86_64::*;

    let mut k = 0usize;
    while k + 4 <= n {
        let i = k + 1; // values index offset
        unsafe {
            let l = _mm256_loadu_pd(lower.as_ptr().add(k));
            let d = _mm256_loadu_pd(diag.as_ptr().add(k));
            let u = _mm256_loadu_pd(upper.as_ptr().add(k));

            let v_lo = _mm256_loadu_pd(values.as_ptr().add(i - 1)); // values[k..k+4]
            let v_mid = _mm256_loadu_pd(values.as_ptr().add(i));    // values[k+1..k+5]
            let v_hi = _mm256_loadu_pd(values.as_ptr().add(i + 1)); // values[k+2..k+6]

            // rhs[k] = diag[k] * v_mid + lower[k] * v_lo + upper[k] * v_hi
            let result = _mm256_fmadd_pd(d, v_mid, _mm256_fmadd_pd(l, v_lo, _mm256_mul_pd(u, v_hi)));
            _mm256_storeu_pd(rhs.as_mut_ptr().add(k), result);
        }
        k += 4;
    }

    // Scalar remainder
    while k < n {
        let i = k + 1;
        rhs[k] = diag[k].mul_add(
            values[i],
            lower[k].mul_add(values[i - 1], upper[k] * values[i + 1]),
        );
        k += 1;
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
