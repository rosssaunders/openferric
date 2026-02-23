//! Backward-Euler finite-difference solver for Black-Scholes vanilla options.
//!
//! This module implements a fully implicit time discretization and solves the
//! tridiagonal linear system at each step with the Thomas algorithm.

use crate::core::{DiagKey, ExerciseStyle, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

use super::fd_common::{
    bermudan_exercise_steps, boundary_values, build_operator_coefficients,
    build_stretched_spot_grid, interpolate_on_grid, intrinsic, solve_tridiagonal_inplace,
};

/// Backward-Euler implicit finite-difference engine for the Black-Scholes PDE.
///
/// The scheme is unconditionally stable in time and supports European,
/// American, and Bermudan exercise through projection onto intrinsic value.
#[derive(Debug, Clone)]
pub struct ImplicitFdEngine {
    /// Number of time steps.
    pub time_steps: usize,
    /// Number of spot intervals; the grid has `space_steps + 1` nodes.
    pub space_steps: usize,
    /// Spot truncation multiplier: `S_max = s_max_multiplier * max(spot, strike)`.
    pub s_max_multiplier: f64,
    /// Stretching parameter for the non-uniform spot grid.
    ///
    /// Smaller values concentrate more points around strike.
    pub grid_stretch: f64,
}

impl Default for ImplicitFdEngine {
    fn default() -> Self {
        Self {
            time_steps: 320,
            space_steps: 180,
            s_max_multiplier: 4.0,
            grid_stretch: 0.15,
        }
    }
}

impl ImplicitFdEngine {
    /// Creates an implicit-FD engine with custom time/space resolution.
    pub fn new(time_steps: usize, space_steps: usize) -> Self {
        Self {
            time_steps,
            space_steps,
            ..Self::default()
        }
    }

    /// Sets `S_max = multiplier * max(spot, strike)`.
    pub fn with_s_max_multiplier(mut self, s_max_multiplier: f64) -> Self {
        self.s_max_multiplier = s_max_multiplier;
        self
    }

    /// Sets the non-uniform grid stretching parameter.
    pub fn with_grid_stretch(mut self, grid_stretch: f64) -> Self {
        self.grid_stretch = grid_stretch;
        self
    }
}

impl PricingEngine<VanillaOption> for ImplicitFdEngine {
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
        if self.grid_stretch <= 0.0 || !self.grid_stretch.is_finite() {
            return Err(PricingError::InvalidInput(
                "grid_stretch must be finite and > 0".to_string(),
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

        let s_anchor = market.spot.max(instrument.strike).max(1.0e-8);
        let s_max = self.s_max_multiplier * s_anchor;
        let grid = build_stretched_spot_grid(n_s, s_max, instrument.strike, self.grid_stretch)?;

        let dividend_yield = market.effective_dividend_yield(instrument.expiry);
        let (a, b, c) = build_operator_coefficients(&grid, market.rate, dividend_yield, vol);

        let is_american = matches!(instrument.exercise, ExerciseStyle::American);
        let bermudan_flags = match &instrument.exercise {
            ExerciseStyle::Bermudan { dates } => {
                Some(bermudan_exercise_steps(dates, instrument.expiry, n_t))
            }
            _ => None,
        };

        let interior_n = n_s - 1;
        let mut lhs_lower = vec![0.0_f64; interior_n];
        let mut lhs_diag = vec![0.0_f64; interior_n];
        let mut lhs_upper = vec![0.0_f64; interior_n];
        for k in 0..interior_n {
            let i = k + 1;
            lhs_lower[k] = -dt * a[i];
            lhs_diag[k] = (-dt).mul_add(b[i], 1.0);
            lhs_upper[k] = -dt * c[i];
        }

        let mut solve_lower = lhs_lower.clone();
        let mut solve_upper = lhs_upper.clone();
        solve_lower[0] = 0.0;
        solve_upper[interior_n - 1] = 0.0;

        let mut values = grid
            .iter()
            .map(|&s| intrinsic(instrument.option_type, s, instrument.strike))
            .collect::<Vec<_>>();
        let mut next_values = vec![0.0_f64; n_s + 1];
        let mut rhs = vec![0.0_f64; interior_n];
        let mut c_star = vec![0.0_f64; interior_n];
        let mut d_star = vec![0.0_f64; interior_n];
        let mut interior = vec![0.0_f64; interior_n];

        for n in (0..n_t).rev() {
            let tau_new = instrument.expiry - n as f64 * dt;
            let (lower_bv, upper_bv) = boundary_values(
                instrument.option_type,
                is_american,
                instrument.strike,
                market.rate,
                dividend_yield,
                s_max,
                tau_new,
            );

            rhs.copy_from_slice(&values[1..n_s]);
            rhs[0] -= lhs_lower[0] * lower_bv;
            rhs[interior_n - 1] -= lhs_upper[interior_n - 1] * upper_bv;

            solve_tridiagonal_inplace(
                &solve_lower,
                &lhs_diag,
                &solve_upper,
                &rhs,
                &mut c_star,
                &mut d_star,
                &mut interior,
            )?;

            next_values[0] = lower_bv;
            next_values[n_s] = upper_bv;
            next_values[1..n_s].copy_from_slice(&interior);

            let can_exercise = match &instrument.exercise {
                ExerciseStyle::European => false,
                ExerciseStyle::American => true,
                ExerciseStyle::Bermudan { .. } => {
                    bermudan_flags.as_ref().is_some_and(|flags| flags[n])
                }
            };
            if can_exercise {
                for (i, v) in next_values.iter_mut().enumerate() {
                    *v = v.max(intrinsic(
                        instrument.option_type,
                        grid[i],
                        instrument.strike,
                    ));
                }
            }

            std::mem::swap(&mut values, &mut next_values);
        }

        let price = interpolate_on_grid(market.spot, &grid, &values);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumTimeSteps, n_t as f64);
        diagnostics.insert_key(DiagKey::NumSpaceSteps, n_s as f64);
        diagnostics.insert_key(DiagKey::SMax, s_max);
        diagnostics.insert_key(DiagKey::Vol, vol);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
