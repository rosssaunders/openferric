use crate::core::{OptionType, PricingError};

#[inline]
pub(super) fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
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
        (OptionType::Call, true) => (0.0, (s_max - strike).max(0.0)),
        (OptionType::Put, true) => (strike, 0.0),
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

pub(super) fn explicit_cfl_dt_max(b: &[f64], cfl_safety_factor: f64) -> Result<f64, PricingError> {
    if !cfl_safety_factor.is_finite() || cfl_safety_factor <= 0.0 {
        return Err(PricingError::InvalidInput(
            "cfl_safety_factor must be finite and > 0".to_string(),
        ));
    }

    let mut dt_max = f64::INFINITY;
    for &bi in b.iter().skip(1).take(b.len().saturating_sub(2)) {
        if bi < -1.0e-14 {
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
