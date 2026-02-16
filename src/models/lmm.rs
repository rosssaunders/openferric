use crate::math::normal_cdf;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Parameters for a lognormal LIBOR market model.
#[derive(Debug, Clone, PartialEq)]
pub struct LmmParams {
    /// Volatility per forward rate.
    pub volatilities: Vec<f64>,
    /// Correlation matrix across forwards.
    pub correlation: Vec<Vec<f64>>,
    /// Tenor grid `T_0, T_1, ..., T_n` (years).
    pub tenors: Vec<f64>,
}

impl LmmParams {
    /// Validates dimensions and basic numerical assumptions.
    pub fn validate(&self) -> Result<(), String> {
        if self.volatilities.is_empty() {
            return Err("volatilities cannot be empty".to_string());
        }

        let n = self.volatilities.len();
        if self.tenors.len() != n + 1 {
            return Err("tenors length must equal volatilities length + 1".to_string());
        }
        if self.correlation.len() != n || self.correlation.iter().any(|row| row.len() != n) {
            return Err("correlation matrix dimensions do not match volatilities".to_string());
        }
        if self.volatilities.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("volatilities must be finite and >= 0".to_string());
        }

        for w in self.tenors.windows(2) {
            if !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0] {
                return Err("tenors must be finite and strictly increasing".to_string());
            }
        }

        for i in 0..n {
            if !self.correlation[i][i].is_finite() || (self.correlation[i][i] - 1.0).abs() > 1.0e-8
            {
                return Err("correlation matrix diagonal entries must be 1".to_string());
            }

            for j in 0..n {
                let cij = self.correlation[i][j];
                if !cij.is_finite() || !(-1.0..=1.0).contains(&cij) {
                    return Err("correlation entries must be finite in [-1, 1]".to_string());
                }
                if (cij - self.correlation[j][i]).abs() > 1.0e-8 {
                    return Err("correlation matrix must be symmetric".to_string());
                }
            }
        }

        Ok(())
    }

    fn taus(&self) -> Vec<f64> {
        self.tenors.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// One-factor lognormal LIBOR market model (BGM style) with MC pricing utilities.
#[derive(Debug, Clone, PartialEq)]
pub struct LmmModel {
    pub params: LmmParams,
}

impl LmmModel {
    /// Creates an LMM model after validating inputs.
    pub fn new(params: LmmParams) -> Result<Self, String> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Simulates terminal forwards at `horizon` under a spot-measure drift approximation.
    pub fn simulate_terminal_forwards(
        &self,
        initial_forwards: &[f64],
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>, String> {
        if initial_forwards.len() != self.params.volatilities.len() {
            return Err("initial_forwards length must match model dimension".to_string());
        }
        if initial_forwards.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err("initial forwards must be finite and > 0".to_string());
        }
        if !horizon.is_finite() || horizon <= 0.0 || num_steps == 0 || num_paths == 0 {
            return Err("horizon, num_steps and num_paths must be > 0".to_string());
        }

        let chol = cholesky_lower(&self.params.correlation)
            .ok_or_else(|| "correlation matrix is not positive semidefinite".to_string())?;
        let taus = self.params.taus();
        let dt = horizon / num_steps as f64;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut terminal = Vec::with_capacity(num_paths);
        for _ in 0..num_paths {
            let mut forwards = initial_forwards.to_vec();
            self.evolve_forwards_path(&mut forwards, &taus, &chol, dt, num_steps, &mut rng);
            terminal.push(forwards);
        }

        Ok(terminal)
    }

    /// Prices a European swaption with Monte Carlo under the LMM.
    #[allow(clippy::too_many_arguments)]
    pub fn price_european_swaption_mc(
        &self,
        initial_forwards: &[f64],
        strike: f64,
        expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> Result<f64, String> {
        if !strike.is_finite()
            || strike <= 0.0
            || !expiry.is_finite()
            || expiry <= 0.0
            || !swap_start.is_finite()
            || !swap_end.is_finite()
            || swap_start < expiry - 1.0e-10
            || swap_end <= swap_start
            || !notional.is_finite()
            || notional <= 0.0
            || num_paths == 0
            || num_steps == 0
        {
            return Err("invalid swaption or simulation inputs".to_string());
        }
        if initial_forwards.len() != self.params.volatilities.len() {
            return Err("initial_forwards length must match model dimension".to_string());
        }

        let start_idx = tenor_index(&self.params.tenors, swap_start)
            .ok_or_else(|| "swap_start must match a tenor-grid point".to_string())?;
        let end_idx = tenor_index(&self.params.tenors, swap_end)
            .ok_or_else(|| "swap_end must match a tenor-grid point".to_string())?;
        if end_idx <= start_idx {
            return Err("swap_end must be greater than swap_start".to_string());
        }

        let chol = cholesky_lower(&self.params.correlation)
            .ok_or_else(|| "correlation matrix is not positive semidefinite".to_string())?;
        let taus = self.params.taus();
        let dt = expiry / num_steps as f64;
        let discount_to_expiry = initial_discount_to(initial_forwards, &self.params.tenors, expiry)
            .ok_or_else(|| "failed to compute initial discount factor to expiry".to_string())?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut payoff_sum = 0.0;
        for _ in 0..num_paths {
            let mut forwards = initial_forwards.to_vec();
            self.evolve_forwards_path(&mut forwards, &taus, &chol, dt, num_steps, &mut rng);

            let (swap_rate, annuity) =
                swap_rate_annuity_from_forwards(&forwards, &taus, start_idx, end_idx);
            let intrinsic = if is_payer {
                (swap_rate - strike).max(0.0)
            } else {
                (strike - swap_rate).max(0.0)
            };
            payoff_sum += notional * annuity * intrinsic;
        }

        Ok(discount_to_expiry * payoff_sum / num_paths as f64)
    }

    fn evolve_forwards_path(
        &self,
        forwards: &mut [f64],
        taus: &[f64],
        chol: &[Vec<f64>],
        dt: f64,
        num_steps: usize,
        rng: &mut StdRng,
    ) {
        let n = forwards.len();
        let sqrt_dt = dt.sqrt();
        let mut indep = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        let mut drifts = vec![0.0_f64; n];

        for step in 0..num_steps {
            let t = step as f64 * dt;
            let active = first_active_forward_index(&self.params.tenors, t);

            for zi in &mut indep {
                *zi = StandardNormal.sample(rng);
            }
            correlate_normals(chol, &indep, &mut z);

            for i in active..n {
                let mut drift = 0.0;
                for k in active..=i {
                    let denom = 1.0 + taus[k] * forwards[k];
                    if denom > 1.0e-12 {
                        drift += self.params.volatilities[i]
                            * self.params.correlation[i][k]
                            * self.params.volatilities[k]
                            * taus[k]
                            * forwards[k]
                            / denom;
                    }
                }
                drifts[i] = drift;
            }

            for i in active..n {
                let vol = self.params.volatilities[i];
                let diffusion = vol * sqrt_dt * z[i];
                let drift_term = (drifts[i] - 0.5 * vol * vol) * dt;
                forwards[i] = (forwards[i] * (drift_term + diffusion).exp()).max(1.0e-12);
            }
        }
    }
}

/// Black swaption price from forward swap rate and annuity.
#[allow(clippy::too_many_arguments)]
pub fn black_swaption_price(
    notional: f64,
    forward_swap_rate: f64,
    strike: f64,
    annuity: f64,
    vol: f64,
    expiry: f64,
    is_payer: bool,
) -> f64 {
    if notional <= 0.0 || forward_swap_rate <= 0.0 || strike <= 0.0 || annuity <= 0.0 {
        return f64::NAN;
    }

    let scale = notional * annuity;
    if vol <= 0.0 || expiry <= 0.0 {
        let intrinsic = if is_payer {
            (forward_swap_rate - strike).max(0.0)
        } else {
            (strike - forward_swap_rate).max(0.0)
        };
        return scale * intrinsic;
    }

    let sig_sqrt_t = vol * expiry.sqrt();
    let d1 = ((forward_swap_rate / strike).ln() + 0.5 * vol * vol * expiry) / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let option_value = if is_payer {
        forward_swap_rate * normal_cdf(d1) - strike * normal_cdf(d2)
    } else {
        strike * normal_cdf(-d2) - forward_swap_rate * normal_cdf(-d1)
    };
    scale * option_value
}

/// Computes initial swap rate and annuity from a forward curve on the tenor grid.
pub fn initial_swap_rate_annuity(
    initial_forwards: &[f64],
    tenors: &[f64],
    swap_start: f64,
    swap_end: f64,
) -> Option<(f64, f64)> {
    if tenors.len() != initial_forwards.len() + 1 {
        return None;
    }
    let taus = tenors.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    let start_idx = tenor_index(tenors, swap_start)?;
    let end_idx = tenor_index(tenors, swap_end)?;
    if end_idx <= start_idx {
        return None;
    }

    let p0_start = initial_discount_to(initial_forwards, tenors, swap_start)?;
    let p0_end = initial_discount_to(initial_forwards, tenors, swap_end)?;

    let mut annuity = 0.0;
    for i in start_idx..end_idx {
        let pay_time = tenors[i + 1];
        let p = initial_discount_to(initial_forwards, tenors, pay_time)?;
        annuity += taus[i] * p;
    }
    if annuity <= 0.0 {
        return None;
    }

    let forward_swap_rate = (p0_start - p0_end) / annuity;
    Some((forward_swap_rate, annuity))
}

fn swap_rate_annuity_from_forwards(
    forwards: &[f64],
    taus: &[f64],
    start_idx: usize,
    end_idx: usize,
) -> (f64, f64) {
    let mut p = 1.0;
    let mut annuity = 0.0;
    for i in start_idx..end_idx {
        p /= 1.0 + taus[i] * forwards[i];
        annuity += taus[i] * p;
    }

    if annuity <= 0.0 {
        return (0.0, 0.0);
    }
    ((1.0 - p) / annuity, annuity)
}

fn initial_discount_to(initial_forwards: &[f64], tenors: &[f64], t: f64) -> Option<f64> {
    if t < 0.0 {
        return None;
    }
    if t <= 0.0 {
        return Some(1.0);
    }

    let mut df = 1.0;
    for i in 0..initial_forwards.len() {
        let t0 = tenors[i];
        let t1 = tenors[i + 1];
        if t <= t0 {
            break;
        }

        let dt = (t.min(t1) - t0).max(0.0);
        if dt <= 0.0 {
            continue;
        }
        let denom = 1.0 + initial_forwards[i] * dt;
        if denom <= 1.0e-12 {
            return None;
        }
        df /= denom;
        if t <= t1 {
            break;
        }
    }
    Some(df)
}

fn tenor_index(tenors: &[f64], t: f64) -> Option<usize> {
    tenors.iter().position(|ti| (*ti - t).abs() <= 1.0e-10)
}

fn first_active_forward_index(tenors: &[f64], t: f64) -> usize {
    tenors
        .windows(2)
        .position(|w| w[1] > t + 1.0e-12)
        .unwrap_or(tenors.len().saturating_sub(2))
}

fn correlate_normals(chol: &[Vec<f64>], indep: &[f64], out: &mut [f64]) {
    for i in 0..chol.len() {
        let mut v = 0.0;
        for (j, lij) in chol[i].iter().enumerate().take(i + 1) {
            v += *lij * indep[j];
        }
        out[i] = v;
    }
}

fn cholesky_lower(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut l = vec![vec![0.0_f64; n]; n];
    let tol = 1.0e-12;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum < -tol {
                    return None;
                }
                l[i][j] = sum.max(tol).sqrt();
            } else if l[j][j].abs() > tol {
                l[i][j] = sum / l[j][j];
            } else {
                l[i][j] = 0.0;
            }
        }
    }

    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lmm_swaption_mc_matches_black_within_two_percent() {
        let tenors = (0..=10).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
        let n = tenors.len() - 1;
        let vols = vec![0.15; n];
        let corr = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 1.0 } else { 0.999 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let params = LmmParams {
            volatilities: vols,
            correlation: corr,
            tenors: tenors.clone(),
        };
        let model = LmmModel::new(params).unwrap();

        let initial_forwards = vec![0.05; n];
        let notional = 1_000_000.0;
        let strike = 0.05;
        let expiry = 2.0;
        let swap_start = 2.0;
        let swap_end = 5.0;

        let mc = model
            .price_european_swaption_mc(
                &initial_forwards,
                strike,
                expiry,
                swap_start,
                swap_end,
                true,
                notional,
                50_000,
                80,
                7,
            )
            .unwrap();

        let (forward_swap_rate, annuity) =
            initial_swap_rate_annuity(&initial_forwards, &tenors, swap_start, swap_end).unwrap();
        let black = black_swaption_price(
            notional,
            forward_swap_rate,
            strike,
            annuity,
            0.15,
            expiry,
            true,
        );

        let rel_err = (mc - black).abs() / black.max(1.0e-12);
        assert!(
            rel_err <= 0.02,
            "mc={} black={} rel_err={}",
            mc,
            black,
            rel_err
        );
    }
}
