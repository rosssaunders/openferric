//! Alternating-Direction Implicit solvers for the two-factor Heston PDE.
//!
//! This module provides Douglas-Rachford and Craig-Sneyd ADI schemes for
//! European vanilla options on a uniform `(S, v)` grid.

use crate::core::{DiagKey, ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::models::stochastic::Heston;

use super::fd_common::{intrinsic, solve_tridiagonal_inplace};

#[inline]
fn idx(i: usize, j: usize, n_s: usize) -> usize {
    j * (n_s + 1) + i
}

/// ADI splitting variant for the Heston PDE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdiScheme {
    /// Douglas-Rachford two-sweep ADI splitting.
    DouglasRachford,
    /// Craig-Sneyd splitting with an extra mixed-derivative correction sweep.
    CraigSneyd,
}

/// ADI finite-difference engine for European vanilla options under Heston dynamics.
#[derive(Debug, Clone)]
pub struct AdiHestonEngine {
    /// Heston model parameters.
    pub model: Heston,
    /// ADI splitting scheme.
    pub scheme: AdiScheme,
    /// Number of time steps.
    pub time_steps: usize,
    /// Number of spot intervals.
    pub spot_steps: usize,
    /// Number of variance intervals.
    pub variance_steps: usize,
    /// Spot truncation multiplier: `S_max = s_max_multiplier * max(spot, strike)`.
    pub s_max_multiplier: f64,
    /// Variance truncation multiplier: `v_max = v_max_multiplier * max(theta, v0, 0.04)`.
    pub v_max_multiplier: f64,
    /// ADI theta parameter used by implicit directional sweeps.
    pub theta_adi: f64,
    /// If `true`, reject model parameters that violate `2*kappa*theta >= xi^2`.
    pub enforce_feller: bool,
}

impl AdiHestonEngine {
    /// Creates an ADI Heston engine with explicit grid sizes.
    pub fn new(model: Heston, time_steps: usize, spot_steps: usize, variance_steps: usize) -> Self {
        Self {
            model,
            scheme: AdiScheme::CraigSneyd,
            time_steps,
            spot_steps,
            variance_steps,
            s_max_multiplier: 4.0,
            v_max_multiplier: 5.0,
            theta_adi: 0.5,
            enforce_feller: true,
        }
    }

    /// Selects the splitting scheme.
    pub fn with_scheme(mut self, scheme: AdiScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Sets `S_max = multiplier * max(spot, strike)`.
    pub fn with_s_max_multiplier(mut self, s_max_multiplier: f64) -> Self {
        self.s_max_multiplier = s_max_multiplier;
        self
    }

    /// Sets `v_max = multiplier * max(theta, v0, 0.04)`.
    pub fn with_v_max_multiplier(mut self, v_max_multiplier: f64) -> Self {
        self.v_max_multiplier = v_max_multiplier;
        self
    }

    /// Sets the ADI theta parameter.
    pub fn with_theta_adi(mut self, theta_adi: f64) -> Self {
        self.theta_adi = theta_adi;
        self
    }

    /// Enables or disables strict Feller-condition enforcement.
    pub fn with_enforce_feller(mut self, enforce_feller: bool) -> Self {
        self.enforce_feller = enforce_feller;
        self
    }
}

struct HestonPdeCoefficients {
    a0: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    total: Vec<f64>,
}

struct HestonGrid {
    n_s: usize,
    n_v: usize,
    ds: f64,
    dv: f64,
    s_max: f64,
}

impl HestonGrid {
    fn n_points(&self) -> usize {
        (self.n_s + 1) * (self.n_v + 1)
    }
}

fn apply_spot_boundaries(
    values: &mut [f64],
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    tau: f64,
    grid: &HestonGrid,
) {
    let lower = match option_type {
        OptionType::Call => 0.0,
        OptionType::Put => strike * (-rate * tau).exp(),
    };
    let upper = match option_type {
        OptionType::Call => {
            (grid.s_max * (-dividend_yield * tau).exp() - strike * (-rate * tau).exp()).max(0.0)
        }
        OptionType::Put => 0.0,
    };

    for j in 0..=grid.n_v {
        values[idx(0, j, grid.n_s)] = lower;
        values[idx(grid.n_s, j, grid.n_s)] = upper;
    }
}

fn apply_variance_neumann_boundaries(values: &mut [f64], grid: &HestonGrid) {
    for i in 1..grid.n_s {
        values[idx(i, 0, grid.n_s)] = values[idx(i, 1, grid.n_s)];
        values[idx(i, grid.n_v, grid.n_s)] = values[idx(i, grid.n_v - 1, grid.n_s)];
    }
}

fn apply_boundaries(
    values: &mut [f64],
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    tau: f64,
    grid: &HestonGrid,
) {
    apply_variance_neumann_boundaries(values, grid);
    apply_spot_boundaries(values, option_type, strike, rate, dividend_yield, tau, grid);
}

fn compute_heston_operators(
    values: &[f64],
    rate: f64,
    dividend_yield: f64,
    model: &Heston,
    grid: &HestonGrid,
    out: &mut HestonPdeCoefficients,
) {
    out.a0.fill(0.0);
    out.a1.fill(0.0);
    out.a2.fill(0.0);
    out.total.fill(0.0);

    let ds = grid.ds;
    let dv = grid.dv;
    let inv_2ds = 0.5 / ds;
    let inv_2dv = 0.5 / dv;
    let inv_ds2 = 1.0 / (ds * ds);
    let inv_dv2 = 1.0 / (dv * dv);
    let inv_4dsdv = 0.25 / (ds * dv);

    for j in 1..grid.n_v {
        let v = j as f64 * dv;
        for i in 1..grid.n_s {
            let s = i as f64 * ds;
            let p = idx(i, j, grid.n_s);

            let v_c = values[p];
            let v_s_m = values[idx(i - 1, j, grid.n_s)];
            let v_s_p = values[idx(i + 1, j, grid.n_s)];
            let v_v_m = values[idx(i, j - 1, grid.n_s)];
            let v_v_p = values[idx(i, j + 1, grid.n_s)];

            let dss = (v_s_p - 2.0 * v_c + v_s_m) * inv_ds2;
            let d_s = (v_s_p - v_s_m) * inv_2ds;
            let dvv = (v_v_p - 2.0 * v_c + v_v_m) * inv_dv2;
            let d_v = (v_v_p - v_v_m) * inv_2dv;

            let cross = (values[idx(i + 1, j + 1, grid.n_s)]
                - values[idx(i + 1, j - 1, grid.n_s)]
                - values[idx(i - 1, j + 1, grid.n_s)]
                + values[idx(i - 1, j - 1, grid.n_s)])
                * inv_4dsdv;

            let a0 = model.rho * model.xi * v * s * cross;
            let a1 = 0.5 * v * s * s * dss + (rate - dividend_yield) * s * d_s - rate * v_c;
            let a2 = 0.5 * model.xi * model.xi * v * dvv + model.kappa * (model.theta - v) * d_v;

            out.a0[p] = a0;
            out.a1[p] = a1;
            out.a2[p] = a2;
            out.total[p] = a0 + a1 + a2;
        }
    }
}

struct SolveCtx<'a> {
    rate: f64,
    dividend_yield: f64,
    theta_dt: f64,
    model: &'a Heston,
    grid: &'a HestonGrid,
}

fn solve_s_direction(
    rhs_seed: &[f64],
    a1_old: &[f64],
    option_type: OptionType,
    strike: f64,
    tau: f64,
    ctx: &SolveCtx<'_>,
    out: &mut [f64],
) -> Result<(), PricingError> {
    let m = ctx.grid.n_s - 1;
    let mut lower = vec![0.0_f64; m];
    let mut diag = vec![0.0_f64; m];
    let mut upper = vec![0.0_f64; m];
    let mut solve_lower = vec![0.0_f64; m];
    let mut solve_upper = vec![0.0_f64; m];
    let mut rhs = vec![0.0_f64; m];
    let mut c_star = vec![0.0_f64; m];
    let mut d_star = vec![0.0_f64; m];
    let mut interior = vec![0.0_f64; m];

    apply_spot_boundaries(
        out,
        option_type,
        strike,
        ctx.rate,
        ctx.dividend_yield,
        tau,
        ctx.grid,
    );

    let ds = ctx.grid.ds;
    let inv_2ds = 0.5 / ds;
    let inv_ds2 = 1.0 / (ds * ds);

    for j in 1..ctx.grid.n_v {
        let v = j as f64 * ctx.grid.dv;
        let lo_bv = out[idx(0, j, ctx.grid.n_s)];
        let hi_bv = out[idx(ctx.grid.n_s, j, ctx.grid.n_s)];

        for k in 0..m {
            let i = k + 1;
            let s = i as f64 * ds;

            let a = 0.5 * v * s * s * inv_ds2 - (ctx.rate - ctx.dividend_yield) * s * inv_2ds;
            let b = -v * s * s * inv_ds2 - ctx.rate;
            let c = 0.5 * v * s * s * inv_ds2 + (ctx.rate - ctx.dividend_yield) * s * inv_2ds;

            lower[k] = -ctx.theta_dt * a;
            diag[k] = 1.0 - ctx.theta_dt * b;
            upper[k] = -ctx.theta_dt * c;

            let p = idx(i, j, ctx.grid.n_s);
            rhs[k] = rhs_seed[p] - ctx.theta_dt * a1_old[p];
        }

        rhs[0] -= lower[0] * lo_bv;
        rhs[m - 1] -= upper[m - 1] * hi_bv;

        solve_lower.copy_from_slice(&lower);
        solve_upper.copy_from_slice(&upper);
        solve_lower[0] = 0.0;
        solve_upper[m - 1] = 0.0;

        solve_tridiagonal_inplace(
            &solve_lower,
            &diag,
            &solve_upper,
            &rhs,
            &mut c_star,
            &mut d_star,
            &mut interior,
        )?;

        for (k, val) in interior[..m].iter().enumerate() {
            let i = k + 1;
            out[idx(i, j, ctx.grid.n_s)] = *val;
        }
    }

    apply_boundaries(
        out,
        option_type,
        strike,
        ctx.rate,
        ctx.dividend_yield,
        tau,
        ctx.grid,
    );
    Ok(())
}

fn solve_v_direction(
    rhs_seed: &[f64],
    a2_old: &[f64],
    option_type: OptionType,
    strike: f64,
    tau: f64,
    ctx: &SolveCtx<'_>,
    out: &mut [f64],
) -> Result<(), PricingError> {
    let m = ctx.grid.n_v - 1;
    let mut lower = vec![0.0_f64; m];
    let mut diag = vec![0.0_f64; m];
    let mut upper = vec![0.0_f64; m];
    let mut solve_lower = vec![0.0_f64; m];
    let mut solve_upper = vec![0.0_f64; m];
    let mut rhs = vec![0.0_f64; m];
    let mut c_star = vec![0.0_f64; m];
    let mut d_star = vec![0.0_f64; m];
    let mut interior = vec![0.0_f64; m];

    let dv = ctx.grid.dv;
    let inv_2dv = 0.5 / dv;
    let inv_dv2 = 1.0 / (dv * dv);

    for i in 1..ctx.grid.n_s {
        let lo_bv = rhs_seed[idx(i, 1, ctx.grid.n_s)];
        let hi_bv = rhs_seed[idx(i, ctx.grid.n_v - 1, ctx.grid.n_s)];

        for k in 0..m {
            let j = k + 1;
            let v = j as f64 * dv;

            let a = 0.5 * ctx.model.xi * ctx.model.xi * v * inv_dv2
                - ctx.model.kappa * (ctx.model.theta - v) * inv_2dv;
            let b = -ctx.model.xi * ctx.model.xi * v * inv_dv2;
            let c = 0.5 * ctx.model.xi * ctx.model.xi * v * inv_dv2
                + ctx.model.kappa * (ctx.model.theta - v) * inv_2dv;

            lower[k] = -ctx.theta_dt * a;
            diag[k] = 1.0 - ctx.theta_dt * b;
            upper[k] = -ctx.theta_dt * c;

            let p = idx(i, j, ctx.grid.n_s);
            rhs[k] = rhs_seed[p] - ctx.theta_dt * a2_old[p];
        }

        rhs[0] -= lower[0] * lo_bv;
        rhs[m - 1] -= upper[m - 1] * hi_bv;

        solve_lower.copy_from_slice(&lower);
        solve_upper.copy_from_slice(&upper);
        solve_lower[0] = 0.0;
        solve_upper[m - 1] = 0.0;

        solve_tridiagonal_inplace(
            &solve_lower,
            &diag,
            &solve_upper,
            &rhs,
            &mut c_star,
            &mut d_star,
            &mut interior,
        )?;

        for (k, val) in interior[..m].iter().enumerate() {
            let j = k + 1;
            out[idx(i, j, ctx.grid.n_s)] = *val;
        }
    }

    apply_boundaries(
        out,
        option_type,
        strike,
        ctx.rate,
        ctx.dividend_yield,
        tau,
        ctx.grid,
    );
    Ok(())
}

impl PricingEngine<VanillaOption> for AdiHestonEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "AdiHestonEngine supports European exercise only".to_string(),
            ));
        }
        if self.time_steps == 0 || self.spot_steps < 3 || self.variance_steps < 3 {
            return Err(PricingError::InvalidInput(
                "time_steps must be > 0 and spot_steps/variance_steps must be >= 3".to_string(),
            ));
        }
        if self.s_max_multiplier <= 0.0
            || !self.s_max_multiplier.is_finite()
            || self.v_max_multiplier <= 0.0
            || !self.v_max_multiplier.is_finite()
        {
            return Err(PricingError::InvalidInput(
                "s_max_multiplier and v_max_multiplier must be finite and > 0".to_string(),
            ));
        }
        if self.theta_adi <= 0.0 || self.theta_adi > 1.0 || !self.theta_adi.is_finite() {
            return Err(PricingError::InvalidInput(
                "theta_adi must be finite and in (0, 1]".to_string(),
            ));
        }
        if !self.model.validate() {
            return Err(PricingError::InvalidInput(
                "invalid Heston parameters".to_string(),
            ));
        }

        let feller_lhs = 2.0 * self.model.kappa * self.model.theta;
        let feller_rhs = self.model.xi * self.model.xi;
        if self.enforce_feller && feller_lhs + 1.0e-12 < feller_rhs {
            return Err(PricingError::InvalidInput(format!(
                "Feller condition violated: 2*kappa*theta={feller_lhs:.6e} < xi^2={feller_rhs:.6e}",
            )));
        }

        if instrument.expiry == 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let s_anchor = market.spot.max(instrument.strike).max(1.0e-8);
        let s_max = self.s_max_multiplier * s_anchor;
        let v_anchor = self.model.theta.max(self.model.v0).max(0.04);
        let v_max = self.v_max_multiplier * v_anchor;

        let grid = HestonGrid {
            n_s: self.spot_steps,
            n_v: self.variance_steps,
            ds: s_max / self.spot_steps as f64,
            dv: v_max / self.variance_steps as f64,
            s_max,
        };

        let n_t = self.time_steps;
        let dt = instrument.expiry / n_t as f64;
        let theta_dt = self.theta_adi * dt;
        let rate = market.rate;
        let dividend_yield = market.effective_dividend_yield(instrument.expiry);

        let mut u = vec![0.0_f64; grid.n_points()];
        for j in 0..=grid.n_v {
            for i in 0..=grid.n_s {
                let s = i as f64 * grid.ds;
                u[idx(i, j, grid.n_s)] = intrinsic(instrument.option_type, s, instrument.strike);
            }
        }
        apply_boundaries(
            &mut u,
            instrument.option_type,
            instrument.strike,
            rate,
            dividend_yield,
            0.0,
            &grid,
        );

        let mut ops = HestonPdeCoefficients {
            a0: vec![0.0; grid.n_points()],
            a1: vec![0.0; grid.n_points()],
            a2: vec![0.0; grid.n_points()],
            total: vec![0.0; grid.n_points()],
        };
        let mut ops_tmp = HestonPdeCoefficients {
            a0: vec![0.0; grid.n_points()],
            a1: vec![0.0; grid.n_points()],
            a2: vec![0.0; grid.n_points()],
            total: vec![0.0; grid.n_points()],
        };

        let mut y0 = vec![0.0_f64; grid.n_points()];
        let mut y1 = vec![0.0_f64; grid.n_points()];
        let mut y2 = vec![0.0_f64; grid.n_points()];
        let mut z0 = vec![0.0_f64; grid.n_points()];
        let mut z1 = vec![0.0_f64; grid.n_points()];
        let mut z2 = vec![0.0_f64; grid.n_points()];

        let ctx = SolveCtx {
            rate,
            dividend_yield,
            theta_dt,
            model: &self.model,
            grid: &grid,
        };

        for step in 0..n_t {
            let tau_new = (step + 1) as f64 * dt;

            compute_heston_operators(&u, rate, dividend_yield, &self.model, &grid, &mut ops);

            y0.copy_from_slice(&u);
            for j in 1..grid.n_v {
                for i in 1..grid.n_s {
                    let p = idx(i, j, grid.n_s);
                    y0[p] = dt.mul_add(ops.total[p], u[p]);
                }
            }
            apply_boundaries(
                &mut y0,
                instrument.option_type,
                instrument.strike,
                rate,
                dividend_yield,
                tau_new,
                &grid,
            );

            solve_s_direction(
                &y0,
                &ops.a1,
                instrument.option_type,
                instrument.strike,
                tau_new,
                &ctx,
                &mut y1,
            )?;
            solve_v_direction(
                &y1,
                &ops.a2,
                instrument.option_type,
                instrument.strike,
                tau_new,
                &ctx,
                &mut y2,
            )?;

            match self.scheme {
                AdiScheme::DouglasRachford => {
                    u.copy_from_slice(&y2);
                }
                AdiScheme::CraigSneyd => {
                    compute_heston_operators(
                        &y2,
                        rate,
                        dividend_yield,
                        &self.model,
                        &grid,
                        &mut ops_tmp,
                    );

                    z0.copy_from_slice(&y0);
                    for j in 1..grid.n_v {
                        for i in 1..grid.n_s {
                            let p = idx(i, j, grid.n_s);
                            z0[p] = y0[p] + 0.5 * dt * (ops_tmp.a0[p] - ops.a0[p]);
                        }
                    }
                    apply_boundaries(
                        &mut z0,
                        instrument.option_type,
                        instrument.strike,
                        rate,
                        dividend_yield,
                        tau_new,
                        &grid,
                    );

                    solve_s_direction(
                        &z0,
                        &ops.a1,
                        instrument.option_type,
                        instrument.strike,
                        tau_new,
                        &ctx,
                        &mut z1,
                    )?;
                    solve_v_direction(
                        &z1,
                        &ops.a2,
                        instrument.option_type,
                        instrument.strike,
                        tau_new,
                        &ctx,
                        &mut z2,
                    )?;
                    u.copy_from_slice(&z2);
                }
            }

            apply_boundaries(
                &mut u,
                instrument.option_type,
                instrument.strike,
                rate,
                dividend_yield,
                tau_new,
                &grid,
            );
        }

        let s = market.spot.clamp(0.0, s_max);
        let v = self.model.v0.clamp(0.0, v_max);

        let i_hi = (s / grid.ds).floor() as usize;
        let j_hi = (v / grid.dv).floor() as usize;
        let i = i_hi.min(grid.n_s - 1);
        let j = j_hi.min(grid.n_v - 1);

        let s0 = i as f64 * grid.ds;
        let s1 = (i + 1) as f64 * grid.ds;
        let v0 = j as f64 * grid.dv;
        let v1 = (j + 1) as f64 * grid.dv;

        let ws = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        let wv = if v1 > v0 { (v - v0) / (v1 - v0) } else { 0.0 };

        let p00 = u[idx(i, j, grid.n_s)];
        let p10 = u[idx(i + 1, j, grid.n_s)];
        let p01 = u[idx(i, j + 1, grid.n_s)];
        let p11 = u[idx(i + 1, j + 1, grid.n_s)];

        let price = (1.0 - ws) * (1.0 - wv) * p00
            + ws * (1.0 - wv) * p10
            + (1.0 - ws) * wv * p01
            + ws * wv * p11;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumTimeSteps, self.time_steps as f64);
        diagnostics.insert_key(DiagKey::NumSpaceSteps, self.spot_steps as f64);
        diagnostics.insert_key(DiagKey::NumSteps, self.variance_steps as f64);
        diagnostics.insert_key(DiagKey::SMax, s_max);
        diagnostics.insert_key(DiagKey::VarOfVar, self.model.xi * self.model.xi);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}
