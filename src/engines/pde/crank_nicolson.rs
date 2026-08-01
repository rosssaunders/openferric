//! Module `engines::pde::crank_nicolson`.
//!
//! Implements crank nicolson abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 20, Tavella and Randall (2000), finite-difference stencils around Eq. (20.11)-(20.13).
//!
//! Key types and purpose: `CrankNicolsonEngine` define the core data contracts for this module.
//!
//! Numerical considerations: grid spacing, time-step size, and boundary handling dominate stability/consistency; check monotonicity and CFL-style limits where explicit terms appear.
//!
//! When to use: use PDE engines for early-exercise and barrier-style boundary problems in low dimensions; switch to Monte Carlo or trees for high-dimensional state spaces.
use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::{BermudanOption, vanilla::VanillaOption};
use crate::market::Market;

use super::fd_common::{EscrowedStep, escrowed_root_spot, escrowed_steps};

/// Exercise-boundary estimate at a Bermudan decision time from Crank-Nicolson.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PdeExerciseBoundaryPoint {
    /// Exercise date in year fractions.
    pub time: f64,
    /// Strike at this exercise date.
    pub strike: f64,
    /// Estimated boundary `S*` where exercise and continuation are equal.
    pub boundary_spot: Option<f64>,
}

/// Bermudan Crank-Nicolson output including price and boundary diagnostics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BermudanPdeOutput {
    /// Standard engine result payload.
    pub result: PricingResult,
    /// Exercise boundary points in chronological order.
    pub exercise_boundary: Vec<PdeExerciseBoundaryPoint>,
}

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

#[allow(clippy::too_many_arguments)]
fn bermudan_boundary_values(
    option_type: OptionType,
    step: usize,
    step_schedule: &[Option<(f64, f64)>],
    dt: f64,
    rate: f64,
    dividend_yield: f64,
    s_max: f64,
    div_steps: Option<&[EscrowedStep]>,
) -> (f64, f64) {
    let future = step_schedule
        .iter()
        .enumerate()
        .skip(step)
        .filter_map(|(j, e)| e.map(|(_, k)| (j, k)));

    // With discrete dividends the grid carries the escrowed spot S*, so a
    // future exercise at step j pays intrinsic on the reconstructed spot:
    // strike becomes k*P_j - A_j scaled by 1/P_j (identity without a schedule).
    match option_type {
        OptionType::Put => {
            let mut lower = 0.0_f64;
            for (j, k) in future {
                let tau = (j - step) as f64 * dt;
                let (k_ex, ex_scale) = match div_steps {
                    Some(adj) => (adj[j].adjusted_strike(k), adj[j].scale()),
                    None => (k, 1.0),
                };
                lower = lower.max(k_ex.max(0.0) * ex_scale * (-rate * tau).exp());
            }
            (lower, 0.0)
        }
        OptionType::Call => {
            let mut upper = 0.0_f64;
            for (j, k) in future {
                let tau = (j - step) as f64 * dt;
                let (k_ex, ex_scale) = match div_steps {
                    Some(adj) => (adj[j].adjusted_strike(k), adj[j].scale()),
                    None => (k, 1.0),
                };
                upper = upper.max(
                    ((s_max * (-dividend_yield * tau).exp() - k_ex * (-rate * tau).exp())
                        * ex_scale)
                        .max(0.0),
                );
            }
            (0.0, upper)
        }
    }
}

fn estimate_exercise_boundary(option_type: OptionType, diff: &[f64], ds: f64) -> Option<f64> {
    let eps = 1.0e-12;
    if diff.is_empty() {
        return None;
    }
    match option_type {
        OptionType::Put => {
            let mut last_ex_idx = None;
            for (i, &d) in diff.iter().enumerate() {
                if d >= -eps {
                    last_ex_idx = Some(i);
                } else {
                    break;
                }
            }
            let i = last_ex_idx?;
            if i + 1 >= diff.len() {
                return Some(i as f64 * ds);
            }
            let d0 = diff[i];
            let d1 = diff[i + 1];
            let w = if (d0 - d1).abs() > eps {
                (d0 / (d0 - d1)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some((i as f64 + w) * ds)
        }
        OptionType::Call => {
            let first_ex_idx = diff.iter().position(|&d| d >= -eps)?;
            if first_ex_idx == 0 {
                return Some(0.0);
            }
            let d0 = diff[first_ex_idx - 1];
            let d1 = diff[first_ex_idx];
            let w = if (d0 - d1).abs() > eps {
                (d0 / (d0 - d1)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            Some(((first_ex_idx - 1) as f64 + w) * ds)
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

impl CrankNicolsonEngine {
    /// Prices a Bermudan option with Crank-Nicolson and returns an exercise boundary.
    ///
    /// Supports time-varying strikes by applying the date-specific intrinsic value
    /// during the discrete exercise projection at each scheduled date.
    ///
    /// The local volatility is sampled from `market.vol_for(S, t)` for each
    /// `(spot-grid, time-step)` pair, so flat and smile/surface inputs are both
    /// supported in one implementation.
    pub fn price_bermudan_with_boundary(
        &self,
        instrument: &BermudanOption,
        market: &Market,
    ) -> Result<BermudanPdeOutput, PricingError> {
        instrument.validate()?;
        market.validate()?;

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

        let schedule = instrument.effective_schedule()?;
        let terminal_strike = schedule.last().map(|(_, k)| *k).ok_or_else(|| {
            PricingError::InvalidInput("bermudan schedule cannot be empty".to_string())
        })?;

        let n_t = self.time_steps;
        let n_s = self.space_steps;
        let dt = instrument.expiry / n_t as f64;
        // Escrowed discrete-dividend model: the grid carries S*, diffused with
        // the true continuous yield only; exercise projections reconstruct the
        // observed spot per decision date.
        let spot0 = escrowed_root_spot(market, instrument.expiry)?;
        let div_steps = escrowed_steps(market, instrument.expiry, n_t);
        let max_strike = schedule.iter().map(|(_, k)| *k).fold(0.0_f64, f64::max);
        let s_anchor = spot0.max(max_strike).max(1.0e-8);
        let s_max = self.s_max_multiplier * s_anchor;
        let ds = s_max / n_s as f64;

        let mut step_schedule = vec![None::<(f64, f64)>; n_t + 1];
        for &(t, k) in &schedule {
            let idx = (((t / instrument.expiry) * n_t as f64).round() as usize).clamp(1, n_t);
            step_schedule[idx] = Some((t, k));
        }
        step_schedule[n_t] = Some((instrument.expiry, terminal_strike));

        let mut values = vec![0.0_f64; n_s + 1];
        for (i, v) in values.iter_mut().enumerate() {
            let s = i as f64 * ds;
            *v = intrinsic(instrument.option_type, s, terminal_strike);
        }

        let interior_n = n_s - 1;
        let mut lhs_lower = vec![0.0_f64; interior_n];
        let mut lhs_diag = vec![0.0_f64; interior_n];
        let mut lhs_upper = vec![0.0_f64; interior_n];
        let mut rhs_lower = vec![0.0_f64; interior_n];
        let mut rhs_diag = vec![0.0_f64; interior_n];
        let mut rhs_upper = vec![0.0_f64; interior_n];

        let mut rhs_buf = vec![0.0_f64; interior_n];
        let mut solve_lower = vec![0.0_f64; interior_n];
        let mut solve_upper = vec![0.0_f64; interior_n];
        let mut c_star = vec![0.0_f64; interior_n];
        let mut d_star = vec![0.0_f64; interior_n];
        let mut next_values = vec![0.0_f64; n_s + 1];
        let mut diff_buf = vec![0.0_f64; n_s + 1];

        let half_dt = 0.5 * dt;
        let inv_2ds = 1.0 / (2.0 * ds);
        let inv_ds2 = 1.0 / (ds * ds);
        let dividend_yield = market.dividend_yield;
        let drift = market.rate - dividend_yield;

        let mut boundary_rev = vec![PdeExerciseBoundaryPoint {
            time: instrument.expiry,
            strike: terminal_strike,
            boundary_spot: Some(terminal_strike),
        }];

        for n in (0..n_t).rev() {
            let t_node = (n as f64 * dt).max(1.0e-8);
            let (lower_new, upper_new) = bermudan_boundary_values(
                instrument.option_type,
                n,
                &step_schedule,
                dt,
                market.rate,
                dividend_yield,
                s_max,
                div_steps.as_deref(),
            );

            for k in 0..interior_n {
                let i = k + 1;
                let s = i as f64 * ds;
                let sigma = market.checked_vol_for(s.max(1.0e-8), t_node)?;

                let alpha = 0.5 * sigma * sigma * s * s * inv_ds2;
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

            solve_lower.copy_from_slice(&lhs_lower);
            solve_lower[0] = 0.0;
            solve_upper.copy_from_slice(&lhs_upper);
            solve_upper[interior_n - 1] = 0.0;

            for k in 0..interior_n {
                let i = k + 1;
                rhs_buf[k] = rhs_diag[k].mul_add(
                    values[i],
                    rhs_lower[k].mul_add(values[i - 1], rhs_upper[k] * values[i + 1]),
                );
            }

            rhs_buf[0] -= lhs_lower[0] * lower_new;
            rhs_buf[interior_n - 1] -= lhs_upper[interior_n - 1] * upper_new;

            next_values[0] = lower_new;
            next_values[n_s] = upper_new;
            solve_tridiagonal_inplace(
                &solve_lower,
                &lhs_diag,
                &solve_upper,
                &rhs_buf,
                &mut c_star,
                &mut d_star,
                &mut next_values[1..n_s],
            )?;

            if let Some((time, strike)) = step_schedule[n] {
                // Escrowed model: exercise compares against the reconstructed
                // spot, i.e. adjusted strike K*P - A and payoff scale 1/P.
                let (ex_strike, ex_scale) = match &div_steps {
                    Some(adj) => (adj[n].adjusted_strike(strike), adj[n].scale()),
                    None => (strike, 1.0),
                };
                for (i, v) in next_values.iter_mut().enumerate() {
                    let s = i as f64 * ds;
                    let continuation = *v;
                    let exercise = intrinsic(instrument.option_type, s, ex_strike) * ex_scale;
                    diff_buf[i] = exercise - continuation;
                    *v = continuation.max(exercise);
                }
                // The estimated boundary lives on the S* grid; report it in
                // observed-spot units via S = (S* + A) / P.
                let boundary_spot =
                    estimate_exercise_boundary(instrument.option_type, &diff_buf, ds).map(|b| {
                        match &div_steps {
                            Some(adj) => (b + adj[n].cash) / adj[n].prop,
                            None => b,
                        }
                    });
                boundary_rev.push(PdeExerciseBoundaryPoint {
                    time,
                    strike,
                    boundary_spot,
                });
            }

            std::mem::swap(&mut values, &mut next_values);
        }

        let price = if spot0 <= 0.0 {
            values[0]
        } else if spot0 >= s_max {
            values[n_s]
        } else {
            let x = spot0 / ds;
            let i = x.floor() as usize;
            let w = x - i as f64;
            (1.0 - w) * values[i] + w * values[i + 1]
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::NumTimeSteps, n_t as f64);
        diagnostics.insert_key(crate::core::DiagKey::NumSpaceSteps, n_s as f64);
        diagnostics.insert_key(crate::core::DiagKey::SMax, s_max);
        diagnostics.insert_key(crate::core::DiagKey::ExerciseDates, schedule.len() as f64);

        boundary_rev.reverse();
        Ok(BermudanPdeOutput {
            result: PricingResult {
                price,
                stderr: None,
                greeks: None,
                diagnostics,
            },
            exercise_boundary: boundary_rev,
        })
    }
}

impl PricingEngine<VanillaOption> for CrankNicolsonEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
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

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;

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
        // Escrowed discrete-dividend model: the grid carries S*, diffused
        // with the true continuous yield only; exercise obstacles are
        // reconstructed per step below.
        let spot0 = escrowed_root_spot(market, instrument.expiry)?;
        let div_steps = escrowed_steps(market, instrument.expiry, n_t);
        let dividend_yield = market.dividend_yield;
        let drift = market.rate - dividend_yield;
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
            let (mut lower_new, mut upper_new) = boundary_values(
                instrument.option_type,
                is_american,
                instrument.strike,
                market.rate,
                dividend_yield,
                s_max,
                tau_new,
            );
            if is_american && let Some(adj) = &div_steps {
                let sa = &adj[n];
                match instrument.option_type {
                    OptionType::Put => {
                        lower_new =
                            sa.put_floor_at_zero(instrument.strike, market.rate, n as f64 * dt);
                    }
                    OptionType::Call => {
                        upper_new = upper_new.max(sa.exercise_value(
                            OptionType::Call,
                            s_max,
                            instrument.strike,
                        ));
                    }
                }
            }

            // RHS = B * values; use FMA chains for the tridiagonal multiply.
            // SIMD path when available, otherwise scalar FMA.
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if interior_n >= 4
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma")
                {
                    unsafe {
                        pde_rhs_avx2(
                            &rhs_lower,
                            &rhs_diag,
                            &rhs_upper,
                            &values,
                            &mut rhs_buf,
                            interior_n,
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
                // Escrowed model: intrinsic((S*+A)/P, K) == intrinsic(S*, K*P-A)/P,
                // so the per-step adjustment is a strike shift plus a scale.
                let (ex_strike, ex_scale) = match &div_steps {
                    Some(adj) => (adj[n].adjusted_strike(strike), adj[n].scale()),
                    None => (strike, 1.0),
                };
                for (i, v) in next_values.iter_mut().enumerate() {
                    let s = i as f64 * ds;
                    let exercise_value = if is_call {
                        (s - ex_strike).max(0.0) * ex_scale
                    } else {
                        (ex_strike - s).max(0.0) * ex_scale
                    };
                    *v = v.max(exercise_value);
                }
            }

            // Swap instead of copy — eliminates full memcpy per timestep.
            std::mem::swap(&mut values, &mut next_values);
        }

        let price = if spot0 <= 0.0 {
            values[0]
        } else if spot0 >= s_max {
            values[n_s]
        } else {
            let x = spot0 / ds;
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

impl PricingEngine<BermudanOption> for CrankNicolsonEngine {
    fn price(
        &self,
        instrument: &BermudanOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        self.price_bermudan_with_boundary(instrument, market)
            .map(|out| out.result)
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
            let v_mid = _mm256_loadu_pd(values.as_ptr().add(i)); // values[k+1..k+5]
            let v_hi = _mm256_loadu_pd(values.as_ptr().add(i + 1)); // values[k+2..k+6]

            // rhs[k] = diag[k] * v_mid + lower[k] * v_lo + upper[k] * v_hi
            let result =
                _mm256_fmadd_pd(d, v_mid, _mm256_fmadd_pd(l, v_lo, _mm256_mul_pd(u, v_hi)));
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
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn european_call_converges_to_exact_black_scholes() {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let coarse = CrankNicolsonEngine::new(100, 100)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let mid = CrankNicolsonEngine::new(200, 200)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let fine = CrankNicolsonEngine::new(400, 400)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);

        let mid_grid_reference = 10.440_691_508_529_95;
        let mid_roundoff = 8.0 * f64::EPSILON * mid_grid_reference;
        assert!(
            (mid.price - mid_grid_reference).abs() <= mid_roundoff,
            "200x200 Crank-Nicolson finite-grid value changed: price={} reference={} budget={}",
            mid.price,
            mid_grid_reference,
            mid_roundoff,
        );

        // Crank-Nicolson is second order on this smooth, aligned grid.
        let coarse_extrapolated = (4.0 * mid.price - coarse.price) / 3.0;
        let fine_extrapolated = (4.0 * fine.price - mid.price) / 3.0;
        let measured_refinement = (fine_extrapolated - coarse_extrapolated).abs();
        let error = (fine_extrapolated - bs).abs();
        let roundoff = 64.0 * f64::EPSILON * bs;
        assert!(
            error <= measured_refinement + roundoff,
            "Crank-Nicolson Richardson={fine_extrapolated:.15}, exact BSM={bs:.15}, \
             error={error:.3e}, measured refinement={measured_refinement:.3e}"
        );
    }

    #[test]
    fn american_put_converges_to_quantlib_crr_reference() {
        let option = VanillaOption::american_put(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.30)
            .build()
            .unwrap();

        let coarse = CrankNicolsonEngine::new(100, 100)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let mid = CrankNicolsonEngine::new(200, 200)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();
        let fine = CrankNicolsonEngine::new(400, 400)
            .with_s_max_multiplier(4.0)
            .price(&option, &market)
            .unwrap();

        let mid_grid_reference = 10.461_951_016_351_836;
        let mid_roundoff = 8.0 * f64::EPSILON * mid_grid_reference;
        assert!(
            (mid.price - mid_grid_reference).abs() <= mid_roundoff,
            "200x200 American Crank-Nicolson finite-grid value changed: price={} reference={} budget={}",
            mid.price,
            mid_grid_reference,
            mid_roundoff,
        );

        // QuantLib 1.43 BinomialVanillaEngine("crr"), 20,000 steps.  The
        // 10k->20k change is 7.9720731037e-5.
        let reference = 10.471_179_159_642_21;
        let ql_refinement = 7.9720731037e-5;
        let coarse_extrapolated = (4.0 * mid.price - coarse.price) / 3.0;
        let fine_extrapolated = (4.0 * fine.price - mid.price) / 3.0;
        let measured_refinement = (fine_extrapolated - coarse_extrapolated).abs();
        let error = (fine_extrapolated - reference).abs();
        let roundoff = 64.0 * f64::EPSILON * reference;
        assert!(
            error <= measured_refinement + ql_refinement + roundoff,
            "Crank-Nicolson Richardson={fine_extrapolated:.15}, QuantLib={reference:.15}, \
             error={error:.3e}, measured refinement={measured_refinement:.3e}, \
             QuantLib refinement={ql_refinement:.3e}"
        );
    }
}
