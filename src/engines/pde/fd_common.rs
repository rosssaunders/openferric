use crate::core::{OptionType, PricingError};
use crate::market::Market;

pub(super) fn ensure_spot_inside_grid(spot: f64, upper: f64) -> Result<(), PricingError> {
    if !upper.is_finite() || upper <= spot {
        return Err(PricingError::InvalidInput(
            "spot grid upper bound must be finite and exceed the pricing spot; increase s_max_multiplier".to_string(),
        ));
    }
    Ok(())
}

#[inline]
pub(super) fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

/// Per-time-step escrowed discrete-dividend data for PDE engines.
///
/// With a discrete schedule the finite-difference grid carries the escrowed
/// spot `S*`; the observed spot at time `t` is `S = (S* + cash) / prop`, so
/// exercise payoffs on grid nodes use `intrinsic(S*, K*prop - cash) / prop`.
#[derive(Debug, Clone, Copy)]
pub(super) struct EscrowedStep {
    /// Proportional factor `P(t)` of the remaining schedule events.
    pub prop: f64,
    /// Additive cash adjustment `A(t)` of the remaining schedule events.
    pub cash: f64,
    /// Ex-date of the last remaining discrete event, if any.
    pub last_event_time: Option<f64>,
}

impl EscrowedStep {
    /// Strike equivalent on the escrowed grid: `K * P(t) - A(t)`.
    #[inline]
    pub fn adjusted_strike(&self, strike: f64) -> f64 {
        strike.mul_add(self.prop, -self.cash)
    }

    /// Payoff scale `1 / P(t)`.
    #[inline]
    pub fn scale(&self) -> f64 {
        1.0 / self.prop
    }

    /// Exercise payoff at escrowed grid level `s_star`.
    #[inline]
    pub fn exercise_value(&self, option_type: OptionType, s_star: f64, strike: f64) -> f64 {
        intrinsic(option_type, s_star, self.adjusted_strike(strike)) / self.prop
    }

    /// American put Dirichlet value at `S* = 0`.
    ///
    /// Compares immediate exercise with waiting until the last remaining
    /// ex-date or maturity. Waiting until maturity is essential at negative
    /// rates; without remaining dividends the boundary is `K * max(1, DF)`.
    #[inline]
    pub fn put_floor_at_zero(&self, strike: f64, rate: f64, time: f64, maturity: f64) -> f64 {
        let exercise_now = self.adjusted_strike(strike).max(0.0) / self.prop;
        let exercise_before_maturity = match self.last_event_time {
            Some(t_last) if t_last > time => {
                exercise_now.max(strike * (-rate * (t_last - time)).exp())
            }
            _ => exercise_now,
        };
        exercise_before_maturity.max(strike * (-rate * (maturity - time)).exp())
    }
}

/// Builds the per-step escrowed adjustments on a uniform time grid, or `None`
/// when the market has no discrete dividends (identity, hot path untouched).
pub(super) fn escrowed_steps(
    market: &Market,
    expiry: f64,
    time_steps: usize,
) -> Option<Vec<EscrowedStep>> {
    if !market.has_discrete_dividends() {
        return None;
    }
    let events = market.dividends().events();
    Some(
        (0..=time_steps)
            .map(|n| {
                let t = expiry * n as f64 / time_steps as f64;
                let (prop, cash) = market.escrowed_reconstruction(t, expiry);
                let last_event_time = events
                    .iter()
                    .rev()
                    .find(|ev| ev.time <= expiry && ev.time > t)
                    .map(|ev| ev.time);
                EscrowedStep {
                    prop,
                    cash,
                    last_event_time,
                }
            })
            .collect(),
    )
}

/// Escrowed-model root spot `S*(0)`; errors when dividends exceed spot value.
pub(super) fn escrowed_root_spot(market: &Market, expiry: f64) -> Result<f64, PricingError> {
    if !market.has_discrete_dividends() {
        return Ok(market.spot);
    }
    let s_star = market.escrowed_spot(expiry);
    if !s_star.is_finite() || s_star <= 0.0 {
        return Err(PricingError::InvalidInput(
            "discrete dividend schedule exceeds spot under the escrowed model".to_string(),
        ));
    }
    Ok(s_star)
}

pub(super) fn bermudan_exercise_steps(dates: &[f64], expiry: f64, steps: usize) -> Vec<bool> {
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

pub(super) fn boundary_values(
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
            (lower, 0.0)
        }
        (OptionType::Call, true) => {
            let discounted_exercise = |time: f64| {
                (s_max * (-dividend_yield * time).exp() - strike * (-rate * time).exp()).max(0.0)
            };
            let mut upper = discounted_exercise(0.0).max(discounted_exercise(tau));
            if rate * dividend_yield > 0.0 && rate != dividend_yield {
                let stationary_time =
                    ((rate / dividend_yield) * (strike / s_max)).ln() / (rate - dividend_yield);
                if stationary_time > 0.0 && stationary_time < tau {
                    upper = upper.max(discounted_exercise(stationary_time));
                }
            }
            (0.0, upper)
        }
        (OptionType::Put, true) => (strike * (-rate * tau).exp().max(1.0), 0.0),
    }
}

pub(super) fn build_stretched_spot_grid(
    space_steps: usize,
    s_max: f64,
    strike: f64,
    stretch: f64,
) -> Result<Vec<f64>, PricingError> {
    if space_steps < 2 {
        return Err(PricingError::InvalidInput(
            "space_steps must be >= 2".to_string(),
        ));
    }
    if s_max <= 0.0 || !s_max.is_finite() {
        return Err(PricingError::InvalidInput(
            "s_max must be finite and > 0".to_string(),
        ));
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err(PricingError::InvalidInput(
            "strike must be finite and > 0".to_string(),
        ));
    }
    if !stretch.is_finite() || stretch <= 0.0 {
        return Err(PricingError::InvalidInput(
            "grid_stretch must be finite and > 0".to_string(),
        ));
    }

    let anchor = (strike / s_max).clamp(1.0e-8, 1.0 - 1.0e-8);
    let alpha = stretch.max(1.0e-6);
    let y_lo = (-anchor / alpha).asinh();
    let y_hi = ((1.0 - anchor) / alpha).asinh();
    let y_span = y_hi - y_lo;

    let mut grid = vec![0.0_f64; space_steps + 1];
    for (i, s) in grid.iter_mut().enumerate() {
        let x = i as f64 / space_steps as f64;
        let y = y_lo + y_span * x;
        let z = anchor + alpha * y.sinh();
        *s = (s_max * z).clamp(0.0, s_max);
    }
    grid[0] = 0.0;
    grid[space_steps] = s_max;

    if grid
        .windows(2)
        .any(|w| !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0])
    {
        return Err(PricingError::NumericalError(
            "failed to build a strictly increasing stretched grid".to_string(),
        ));
    }

    Ok(grid)
}

pub(super) fn build_operator_coefficients(
    grid: &[f64],
    rate: f64,
    dividend_yield: f64,
    vol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_s = grid.len() - 1;
    let mut a = vec![0.0_f64; n_s + 1];
    let mut b = vec![0.0_f64; n_s + 1];
    let mut c = vec![0.0_f64; n_s + 1];

    for i in 1..n_s {
        let s = grid[i];
        let h_m = grid[i] - grid[i - 1];
        let h_p = grid[i + 1] - grid[i];

        let d1_m = -h_p / (h_m * (h_m + h_p));
        let d1_0 = (h_p - h_m) / (h_m * h_p);
        let d1_p = h_m / (h_p * (h_m + h_p));

        let d2_m = 2.0 / (h_m * (h_m + h_p));
        let d2_0 = -2.0 / (h_m * h_p);
        let d2_p = 2.0 / (h_p * (h_m + h_p));

        let diffusion = 0.5 * vol * vol * s * s;
        let drift = (rate - dividend_yield) * s;

        a[i] = diffusion * d2_m + drift * d1_m;
        b[i] = diffusion * d2_0 + drift * d1_0 - rate;
        c[i] = diffusion * d2_p + drift * d1_p;
    }

    (a, b, c)
}

/// Maximum stable forward-Euler time step for the explicit scheme
/// `u_new_i = dt*a_i*u_{i-1} + (1 + dt*b_i)*u_i + dt*c_i*u_{i+1}`.
///
/// Positivity (monotonicity) of the update needs all three coefficients
/// non-negative:
/// - `1 + dt*b_i >= 0` bounds `dt <= -1/b_i` on the diagonal;
/// - `dt*a_i >= 0` and `dt*c_i >= 0` are dt-independent: on a stretched grid,
///   drift-dominated cells can make `a_i` or `c_i` negative regardless of
///   `dt`. Fixing that properly is an upwinding (spatial discretization)
///   question, not a time-step question. For such rows this function clamps
///   the dt-dependent part with the tighter bound
///   `dt <= 1 / (|a_i| + |c_i| - b_i)`, which keeps `1 + dt*b_i >= 0` and
///   limits the per-step l-inf amplification from the sign-indefinite row
///   (row sum of magnitudes <= 1 + 2*dt*(negative off-diagonal mass)).
pub(super) fn explicit_cfl_dt_max(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    cfl_safety_factor: f64,
) -> Result<f64, PricingError> {
    if !cfl_safety_factor.is_finite() || cfl_safety_factor <= 0.0 {
        return Err(PricingError::InvalidInput(
            "cfl_safety_factor must be finite and > 0".to_string(),
        ));
    }
    if a.len() != b.len() || c.len() != b.len() {
        return Err(PricingError::InvalidInput(
            "operator coefficient lengths must match".to_string(),
        ));
    }

    let mut dt_max = f64::INFINITY;
    let interior = 1..b.len().saturating_sub(1);
    for i in interior {
        let (ai, bi, ci) = (a[i], b[i], c[i]);
        if ai < -1.0e-14 || ci < -1.0e-14 {
            // Drift-dominated row: off-diagonal positivity cannot be restored
            // by any dt (upwinding question); clamp the dt-dependent part.
            let denom = ai.abs() + ci.abs() - bi;
            if denom > 1.0e-14 {
                dt_max = dt_max.min((1.0 / denom) * cfl_safety_factor);
            }
        } else if bi < -1.0e-14 {
            dt_max = dt_max.min((-1.0 / bi) * cfl_safety_factor);
        }
    }
    if !dt_max.is_finite() || dt_max <= 0.0 {
        return Err(PricingError::NumericalError(
            "unable to compute a positive CFL bound".to_string(),
        ));
    }
    Ok(dt_max)
}

pub(super) fn interpolate_on_grid(spot: f64, grid: &[f64], values: &[f64]) -> f64 {
    debug_assert_eq!(grid.len(), values.len());

    if spot <= grid[0] {
        return values[0];
    }
    let n = grid.len() - 1;
    if spot >= grid[n] {
        return values[n];
    }

    let hi = grid.partition_point(|&x| x < spot).clamp(1, n);
    let lo = hi - 1;
    let w = (spot - grid[lo]) / (grid[hi] - grid[lo]);
    (1.0 - w) * values[lo] + w * values[hi]
}

pub(super) fn solve_tridiagonal_inplace(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
    c_star: &mut [f64],
    d_star: &mut [f64],
    out: &mut [f64],
) -> Result<(), PricingError> {
    let n = diag.len();
    if n == 0 {
        return Ok(());
    }
    if lower.len() != n
        || upper.len() != n
        || rhs.len() != n
        || c_star.len() != n
        || d_star.len() != n
        || out.len() != n
    {
        return Err(PricingError::InvalidInput(
            "tridiagonal input lengths must match".to_string(),
        ));
    }

    if diag[0].abs() <= 1.0e-14 {
        return Err(PricingError::NumericalError(
            "tridiagonal solver singular matrix".to_string(),
        ));
    }

    c_star[0] = if n > 1 { upper[0] / diag[0] } else { 0.0 };
    d_star[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i] * c_star[i - 1];
        if denom.abs() <= 1.0e-14 {
            return Err(PricingError::NumericalError(
                "tridiagonal solver singular matrix".to_string(),
            ));
        }
        c_star[i] = if i < n - 1 { upper[i] / denom } else { 0.0 };
        d_star[i] = (rhs[i] - lower[i] * d_star[i - 1]) / denom;
    }

    out[n - 1] = d_star[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = d_star[i] - c_star[i] * out[i + 1];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfl_bound_tightens_for_drift_dominated_coefficients() {
        // High rate, tiny vol: the convection term dominates diffusion on the
        // stretched grid and flips off-diagonal coefficients negative.
        let grid = build_stretched_spot_grid(200, 400.0, 100.0, 0.25).expect("grid builds");
        let (a, b, c) = build_operator_coefficients(&grid, 0.30, 0.0, 0.05);

        let n = b.len();
        let has_negative_offdiag = (1..n - 1).any(|i| a[i] < -1.0e-14 || c[i] < -1.0e-14);
        assert!(
            has_negative_offdiag,
            "parameter set should be drift-dominated (negative off-diagonal)"
        );

        let dt_max = explicit_cfl_dt_max(&a, &b, &c, 1.0).expect("cfl bound exists");

        // Diagonal-only bound (the old check).
        let mut dt_diag = f64::INFINITY;
        for &bi in b.iter().skip(1).take(n - 2) {
            if bi < -1.0e-14 {
                dt_diag = dt_diag.min(-1.0 / bi);
            }
        }

        assert!(
            dt_max < dt_diag,
            "off-diagonal-aware bound {dt_max} should be tighter than diagonal-only {dt_diag}"
        );

        // The bound satisfies every row constraint.
        for i in 1..n - 1 {
            assert!(1.0 + dt_max * b[i] >= -1.0e-12, "diagonal row {i} violated");
            if a[i] < -1.0e-14 || c[i] < -1.0e-14 {
                let denom = a[i].abs() + c[i].abs() - b[i];
                assert!(
                    dt_max <= 1.0 / denom + 1.0e-12,
                    "drift-dominated row {i} violated"
                );
            }
        }
    }

    #[test]
    fn cfl_bound_is_diagonal_bound_when_offdiagonals_are_nonnegative() {
        // Diffusion-dominated rows (a, c >= 0): the bound is exactly -1/b.
        let a = vec![0.0, 1.0, 2.0, 0.0];
        let b = vec![0.0, -4.0, -5.0, 0.0];
        let c = vec![0.0, 1.5, 2.5, 0.0];

        let dt_max = explicit_cfl_dt_max(&a, &b, &c, 1.0).expect("cfl bound exists");
        assert!((dt_max - 0.2).abs() <= 1.0e-15, "dt_max={dt_max}");
    }
}
