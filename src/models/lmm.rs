//! Module `models::lmm`.
//!
//! Implements lmm workflows with concrete routines such as `black_swaption_price`, `initial_swap_rate_annuity`.
//!
//! References: Brace, Gatarek, Musiela (1997), Brigo and Mercurio (2006) Ch. 6, lognormal forward-rate dynamics around Eq. (6.16).
//!
//! Key types and purpose: `LmmParams`, `LmmModel` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
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
    ///
    /// Forwards evolve under the spot-LIBOR measure, and each path payoff is
    /// deflated by the realized rolling bank account
    /// `B(T_e) = prod_k (1 + tau_k F_k(T_k))` over the accrual periods covering
    /// `[T_0, T_e]`, so the numeraire matches the simulation measure. With the
    /// module convention of no discounting before the first tenor date,
    /// `B(0) = 1` and the price is `mean(payoff / B(T_e))`.
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
        self.price_european_swaption_mc_with_stderr(
            initial_forwards,
            strike,
            expiry,
            swap_start,
            swap_end,
            is_payer,
            notional,
            num_paths,
            num_steps,
            seed,
        )
        .map(|(price, _stderr)| price)
    }

    /// As [`Self::price_european_swaption_mc`], but also returns the Monte
    /// Carlo standard error.
    #[allow(clippy::too_many_arguments)]
    pub fn price_european_swaption_mc_with_stderr(
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
    ) -> Result<(f64, f64), String> {
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

        let mut rng = StdRng::seed_from_u64(seed);
        let mut forwards = vec![0.0_f64; initial_forwards.len()];
        let mut payoff_sum = 0.0;
        let mut payoff_sq_sum = 0.0;
        for _ in 0..num_paths {
            forwards.copy_from_slice(initial_forwards);
            self.evolve_forwards_path(&mut forwards, &taus, &chol, dt, num_steps, &mut rng);

            let (swap_rate, annuity) =
                swap_rate_annuity_from_forwards(&forwards, &taus, start_idx, end_idx);
            let intrinsic = if is_payer {
                (swap_rate - strike).max(0.0)
            } else {
                (strike - swap_rate).max(0.0)
            };

            // Realized rolling bank account B(T_e) under the spot-LIBOR
            // measure. Forwards freeze at their reset dates during evolution,
            // so `forwards[k]` holds F_k(T_k) for expired forwards.
            let bank = rolling_bank_account_to(&forwards, &self.params.tenors, expiry);
            // The annuity from swap_rate_annuity_from_forwards is expressed in
            // time-T_start money (its discounting is anchored at start_idx).
            // For forward-start swaptions (swap_start > expiry) the payoff is
            // observed at T_e, so bring the annuity back from T_start to T_e
            // with the pathwise discount built from the forwards seen at T_e.
            // The swap rate itself needs no adjustment: its anchor cancels.
            let stub_df = pathwise_discount_between(
                &forwards,
                &self.params.tenors,
                expiry,
                self.params.tenors[start_idx],
            );
            let deflated = notional * annuity * stub_df * intrinsic / bank;
            payoff_sum += deflated;
            payoff_sq_sum += deflated * deflated;
        }

        let m = num_paths as f64;
        let mean = payoff_sum / m;
        let stderr = if num_paths > 1 {
            ((payoff_sq_sum - m * mean * mean).max(0.0) / (m - 1.0) / m).sqrt()
        } else {
            0.0
        };
        Ok((mean, stderr))
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
            // Forwards freeze at their reset date T_i: F_i is alive while t < T_i.
            // The spot-LIBOR drift sums over alive forwards only (Brigo &
            // Mercurio Eq. 6.16 with beta(t) = first index with T_k >= t).
            let alive = first_alive_forward_index(&self.params.tenors, t);

            for zi in &mut indep {
                *zi = StandardNormal.sample(rng);
            }
            correlate_normals(chol, &indep, &mut z);

            for (i, drift_i) in drifts.iter_mut().enumerate().take(n).skip(alive) {
                let mut drift = 0.0;
                for k in alive..=i {
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
                *drift_i = drift;
            }

            for i in alive..n {
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

/// First forward index whose reset date `T_i = tenors[i]` lies strictly after `t`.
fn first_alive_forward_index(tenors: &[f64], t: f64) -> usize {
    let n = tenors.len().saturating_sub(1);
    tenors[..n]
        .iter()
        .position(|&tk| tk > t + 1.0e-12)
        .unwrap_or(n)
}

/// Realized spot-LIBOR rolling bank account `B(t)` accumulated to `t` from the
/// realized forwards at their reset dates (`forwards[k] = F_k(T_k)` for expired
/// forwards). Accrual periods fully before `t` compound at the full period
/// rate; a period straddling `t` accrues the full period at its frozen reset
/// rate and is then discounted from the period end back to `t` at that same
/// rate. With the module convention of no discounting before `tenors[0]`,
/// `B(0) = 1`.
fn rolling_bank_account_to(forwards: &[f64], tenors: &[f64], t: f64) -> f64 {
    let mut bank = 1.0;
    for k in 0..forwards.len() {
        let t_start = tenors[k];
        let t_end = tenors[k + 1];
        if t_end <= t + 1.0e-12 {
            bank *= 1.0 + (t_end - t_start) * forwards[k];
        } else if t_start < t - 1.0e-12 {
            bank *= (1.0 + (t_end - t_start) * forwards[k]) / (1.0 + (t_end - t) * forwards[k]);
        } else {
            break;
        }
    }
    bank
}

/// Pathwise discount factor `P(t0, t1)` built from the forwards observed at
/// `t0` (`forwards[k]` is frozen at `F_k(T_k)` for expired forwards). Mirrors
/// the convention of [`rolling_bank_account_to`]: a period straddling `t0`
/// accrues over `(t0, T_{k+1}]` at its frozen reset rate, and fully contained
/// periods discount at the full period rate. Returns 1 when `t1 <= t0`.
fn pathwise_discount_between(forwards: &[f64], tenors: &[f64], t0: f64, t1: f64) -> f64 {
    if t1 <= t0 + 1.0e-12 {
        return 1.0;
    }
    let mut df = 1.0;
    for k in 0..forwards.len() {
        let t_start = tenors[k];
        let t_end = tenors[k + 1];
        if t_end <= t0 + 1.0e-12 {
            continue;
        }
        if t_start >= t1 - 1.0e-12 {
            break;
        }
        let accrual_start = t_start.max(t0);
        df /= 1.0 + (t_end - accrual_start) * forwards[k];
    }
    df
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
            for (&lik, &ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
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
    use approx::assert_relative_eq;

    use super::*;

    /// A one-period swaption is a caplet, which is priced exactly by Black-76
    /// in the lognormal LMM: F_m is driftless lognormal with volatility
    /// `sigma_m` under its forward measure. The spot-measure simulation with
    /// pathwise bank-account deflation must reproduce it within MC error.
    #[test]
    fn lmm_one_period_swaption_matches_black76_within_mc_error() {
        let tenors = (0..=6).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
        let n = tenors.len() - 1;
        let vol = 0.20;
        // Correlated forwards so the rolling bank account correlates with the
        // payoff: this gives the test power against numeraire/measure mixing.
        let corr = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 1.0 } else { 0.9 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let params = LmmParams {
            volatilities: vec![vol; n],
            correlation: corr,
            tenors: tenors.clone(),
        };
        let model = LmmModel::new(params).unwrap();

        let initial_forwards = vec![0.05; n];
        let notional = 1_000_000.0;
        let strike = 0.05;
        let expiry = 2.0;
        let swap_start = 2.0;
        let swap_end = 2.5;

        let (mc, stderr) = model
            .price_european_swaption_mc_with_stderr(
                &initial_forwards,
                strike,
                expiry,
                swap_start,
                swap_end,
                true,
                notional,
                120_000,
                80,
                11,
            )
            .unwrap();

        // SciPy 1.17.1 `special.ndtr` Black-76 target.  For the flat 5%
        // semiannual curve P(0,2.5)=1/1.025^5 and the one-period annuity is
        // 0.5*P(0,2.5)=0.44192714380475867.
        let exact_annuity = 0.441_927_143_804_758_67;
        let black_reference = 2_485.020_762_995_754;
        let black = black_swaption_price(notional, 0.05, strike, exact_annuity, vol, expiry, true);
        assert_relative_eq!(black, black_reference, epsilon = 2.0e-11);

        assert!(stderr > 0.0 && stderr.is_finite());
        assert!(
            (mc - black_reference).abs() <= 4.0 * stderr,
            "mc={mc} black={black_reference} stderr={stderr}"
        );
    }

    /// With all volatilities zero the simulation is deterministic: forwards
    /// stay at their initial values, the payoff is the intrinsic value, and
    /// the price must equal the time-0 annuity-discounted intrinsic exactly.
    /// This pins down the pathwise discounting of forward-start swaptions
    /// (swap_start > expiry): without the (T_e, T_start] stub discount the
    /// price overstates by 1 / P(T_e, T_start).
    #[test]
    fn lmm_forward_start_swaption_zero_vol_matches_closed_form() {
        let tenors = (0..=6).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
        let n = tenors.len() - 1;
        let corr = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 1.0 } else { 0.5 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let params = LmmParams {
            volatilities: vec![0.0; n],
            correlation: corr,
            tenors: tenors.clone(),
        };
        let model = LmmModel::new(params).unwrap();

        let initial_forwards = vec![0.05; n];
        let notional = 1_000_000.0;
        let strike = 0.04; // in the money for a payer
        let expiry = 1.0;
        let swap_start = 2.0;
        let swap_end = 3.0;

        let mc = model
            .price_european_swaption_mc(
                &initial_forwards,
                strike,
                expiry,
                swap_start,
                swap_end,
                true,
                notional,
                16,
                8,
                3,
            )
            .unwrap();

        let (forward_swap_rate, annuity0) =
            initial_swap_rate_annuity(&initial_forwards, &tenors, swap_start, swap_end).unwrap();
        // Independent 60-digit Decimal sums for the flat semiannual curve:
        // annuity = .5*(1.025^-5 + 1.025^-6), forward = 5%.
        assert_relative_eq!(forward_swap_rate, 0.05, epsilon = 3.0e-16);
        assert_relative_eq!(annuity0, 0.873_075_576_785_010_6, epsilon = 6.0e-16);
        let expected = 8_730.755_767_850_106;

        assert_relative_eq!(mc, expected, epsilon = 2.0e-10);
    }

    /// Independent SciPy/NumPy scrambled-Sobol reference for an ATM swaption
    /// whose four payment forwards have heterogeneous volatilities and a
    /// fully non-diagonal correlation matrix.  The oracle independently
    /// implemented the spot-LIBOR drift, simultaneous log-Euler update,
    /// rolling bank-account deflator, and swap payoff over all 144 Gaussian
    /// dimensions (6 forwards x 24 time steps).
    #[test]
    fn correlated_multi_forward_swaption_matches_scipy_sobol_oracle() {
        let tenors = (0..=6).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
        let n = tenors.len() - 1;
        let correlation = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| 0.65_f64.powi(i.abs_diff(j) as i32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let model = LmmModel::new(LmmParams {
            volatilities: vec![0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
            correlation,
            tenors: tenors.clone(),
        })
        .unwrap();
        let initial_forwards = [0.031, 0.032, 0.033, 0.034, 0.035, 0.036];
        let strike = 0.034_478_320_403_160_47;

        // SciPy 1.17.1 `qmc.Sobol(scramble=True)` and NumPy 2.4.3, 32
        // independent scrambles of 2^17 paths.  The reference is the mean of
        // the scramble estimates; its 0.5732039509018008 standard error is the
        // sample SD across scrambles divided by sqrt(32), not the less useful
        // within-path pseudo-MC error.
        let reference = 3_877.826_327_256_506_4;
        let reference_stderr = 0.573_203_950_901_800_8;
        let (mc, mc_stderr) = model
            .price_european_swaption_mc_with_stderr(
                &initial_forwards,
                strike,
                1.0,
                1.0,
                3.0,
                true,
                1_000_000.0,
                160_000,
                24,
                1_618_033,
            )
            .unwrap();

        let combined_stderr = mc_stderr.hypot(reference_stderr);
        assert!(mc_stderr > 0.0 && mc_stderr.is_finite());
        assert!(
            (mc - reference).abs() <= 4.0 * combined_stderr,
            "mc={mc} reference={reference} mc_stderr={mc_stderr} reference_stderr={reference_stderr}"
        );
    }
}
