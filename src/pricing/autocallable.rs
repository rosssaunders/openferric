//! Module `pricing::autocallable`.
//!
//! Implements autocallable workflows with concrete routines such as `price_autocallable`, `autocallable_sensitivities`, `price_phoenix_autocallable`, `phoenix_autocallable_sensitivities`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `AutocallableSensitivities` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use std::collections::BTreeMap;

use crate::core::{Greeks, PricingError, PricingResult};
use crate::instruments::{Autocallable, PhoenixAutocallable};
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};

const MC_SEED: u64 = 94_210;
const SPOT_BUMP_REL: f64 = 0.01;
const VOL_BUMP_ABS: f64 = 0.01;
const CORR_BUMP: f64 = 0.01;

/// Multi-asset sensitivities from bump-and-reprice.
#[derive(Debug, Clone)]
pub struct AutocallableSensitivities {
    /// Delta per underlying in contract order.
    pub delta: Vec<f64>,
    /// Parallel volatility sensitivity for the contract underlyings.
    pub vega: f64,
    /// Correlation sensitivity (cega).
    pub cega: f64,
}

#[derive(Debug, Clone)]
struct PreparedAutocallable {
    underlyings: Vec<usize>,
    initial_spots: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    maturity: f64,
    notional: f64,
    observation_schedule: Vec<(usize, f64)>,
    autocall_barrier: f64,
    coupon_rate: f64,
    ki_barrier: f64,
    ki_strike: f64,
    coupon_barrier: Option<f64>,
    memory: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn price_autocallable(
    autocall: &Autocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    let Ok((price, stderr)) = price_standard_for_inputs(
        autocall,
        spots,
        vols,
        corr_matrix,
        r,
        q,
        n_paths,
        n_steps,
        MC_SEED,
    ) else {
        return invalid_result();
    };

    let greeks =
        autocallable_sensitivities(autocall, spots, vols, corr_matrix, r, q, n_paths, n_steps)
            .ok()
            .map(|s| Greeks {
                delta: s.delta.iter().sum::<f64>(),
                gamma: 0.0,
                vega: s.vega,
                theta: 0.0,
                rho: s.cega,
            });

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("observation_count", autocall.autocall_dates.len() as f64);

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks,
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn autocallable_sensitivities(
    autocall: &Autocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> Result<AutocallableSensitivities, PricingError> {
    let prepared = prepare_standard(autocall, spots, vols, corr_matrix, n_steps)?;
    bump_and_reprice_sensitivities(
        &prepared.underlyings,
        spots,
        vols,
        corr_matrix,
        |bumped_spots, bumped_vols, bumped_corr| {
            price_standard_for_inputs(
                autocall,
                bumped_spots,
                bumped_vols,
                bumped_corr,
                r,
                q,
                n_paths,
                n_steps,
                MC_SEED,
            )
            .map(|(p, _)| p)
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn price_phoenix_autocallable(
    phoenix: &PhoenixAutocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    let Ok((price, stderr)) = price_phoenix_for_inputs(
        phoenix,
        spots,
        vols,
        corr_matrix,
        r,
        q,
        n_paths,
        n_steps,
        MC_SEED,
    ) else {
        return invalid_result();
    };

    let greeks = phoenix_autocallable_sensitivities(
        phoenix,
        spots,
        vols,
        corr_matrix,
        r,
        q,
        n_paths,
        n_steps,
    )
    .ok()
    .map(|s| Greeks {
        delta: s.delta.iter().sum::<f64>(),
        gamma: 0.0,
        vega: s.vega,
        theta: 0.0,
        rho: s.cega,
    });

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("observation_count", phoenix.autocall_dates.len() as f64);

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks,
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn phoenix_autocallable_sensitivities(
    phoenix: &PhoenixAutocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> Result<AutocallableSensitivities, PricingError> {
    let prepared = prepare_phoenix(phoenix, spots, vols, corr_matrix, n_steps)?;
    bump_and_reprice_sensitivities(
        &prepared.underlyings,
        spots,
        vols,
        corr_matrix,
        |bumped_spots, bumped_vols, bumped_corr| {
            price_phoenix_for_inputs(
                phoenix,
                bumped_spots,
                bumped_vols,
                bumped_corr,
                r,
                q,
                n_paths,
                n_steps,
                MC_SEED,
            )
            .map(|(p, _)| p)
        },
    )
}

fn invalid_result() -> PricingResult {
    PricingResult {
        price: f64::NAN,
        stderr: None,
        greeks: None,
        diagnostics: crate::core::Diagnostics::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn price_standard_for_inputs(
    autocall: &Autocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<(f64, f64), PricingError> {
    let prepared = prepare_standard(autocall, spots, vols, corr_matrix, n_steps)?;
    simulate_autocallable_paths(&prepared, r, q, n_paths, n_steps, seed)
}

#[allow(clippy::too_many_arguments)]
fn price_phoenix_for_inputs(
    phoenix: &PhoenixAutocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<(f64, f64), PricingError> {
    let prepared = prepare_phoenix(phoenix, spots, vols, corr_matrix, n_steps)?;
    simulate_autocallable_paths(&prepared, r, q, n_paths, n_steps, seed)
}

fn prepare_standard(
    autocall: &Autocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    n_steps: usize,
) -> Result<PreparedAutocallable, PricingError> {
    autocall.validate()?;
    prepare_common(
        &autocall.underlyings,
        autocall.notional,
        &autocall.autocall_dates,
        autocall.autocall_barrier,
        autocall.coupon_rate,
        autocall.ki_barrier,
        autocall.ki_strike,
        autocall.maturity,
        None,
        false,
        spots,
        vols,
        corr_matrix,
        n_steps,
    )
}

fn prepare_phoenix(
    phoenix: &PhoenixAutocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    n_steps: usize,
) -> Result<PreparedAutocallable, PricingError> {
    phoenix.validate()?;
    prepare_common(
        &phoenix.underlyings,
        phoenix.notional,
        &phoenix.autocall_dates,
        phoenix.autocall_barrier,
        phoenix.coupon_rate,
        phoenix.ki_barrier,
        phoenix.ki_strike,
        phoenix.maturity,
        Some(phoenix.coupon_barrier),
        phoenix.memory,
        spots,
        vols,
        corr_matrix,
        n_steps,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_common(
    underlyings: &[usize],
    notional: f64,
    autocall_dates: &[f64],
    autocall_barrier: f64,
    coupon_rate: f64,
    ki_barrier: f64,
    ki_strike: f64,
    maturity: f64,
    coupon_barrier: Option<f64>,
    memory: bool,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    n_steps: usize,
) -> Result<PreparedAutocallable, PricingError> {
    if n_steps == 0 {
        return Err(PricingError::InvalidInput(
            "autocallable n_steps must be > 0".to_string(),
        ));
    }
    validate_market_inputs(spots, vols, corr_matrix)?;

    if underlyings.iter().any(|&i| i >= spots.len()) {
        return Err(PricingError::InvalidInput(
            "autocallable underlying index out of range".to_string(),
        ));
    }

    let mut initial_spots = Vec::with_capacity(underlyings.len());
    let mut selected_vols = Vec::with_capacity(underlyings.len());
    for &idx in underlyings {
        initial_spots.push(spots[idx]);
        selected_vols.push(vols[idx]);
    }

    let selected_corr = underlyings
        .iter()
        .map(|&i| {
            underlyings
                .iter()
                .map(|&j| corr_matrix[i][j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let observation_schedule = observation_schedule(autocall_dates, maturity, n_steps);

    Ok(PreparedAutocallable {
        underlyings: underlyings.to_vec(),
        initial_spots,
        vols: selected_vols,
        corr_matrix: selected_corr,
        maturity,
        notional,
        observation_schedule,
        autocall_barrier,
        coupon_rate,
        ki_barrier,
        ki_strike,
        coupon_barrier,
        memory,
    })
}

#[allow(clippy::too_many_arguments)]
fn bump_and_reprice_sensitivities<F>(
    underlyings: &[usize],
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    mut price_fn: F,
) -> Result<AutocallableSensitivities, PricingError>
where
    F: FnMut(&[f64], &[f64], &[Vec<f64>]) -> Result<f64, PricingError>,
{
    let mut delta = Vec::with_capacity(underlyings.len());
    for &idx in underlyings {
        let bump = (spots[idx].abs() * SPOT_BUMP_REL).max(1.0e-4);
        let mut up_spots = spots.to_vec();
        let mut dn_spots = spots.to_vec();
        up_spots[idx] += bump;
        dn_spots[idx] = (dn_spots[idx] - bump).max(1.0e-8);

        let up = price_fn(&up_spots, vols, corr_matrix)?;
        let dn = price_fn(&dn_spots, vols, corr_matrix)?;
        delta.push((up - dn) / (2.0 * bump));
    }

    let mut up_vols = vols.to_vec();
    let mut dn_vols = vols.to_vec();
    for &idx in underlyings {
        up_vols[idx] += VOL_BUMP_ABS;
        dn_vols[idx] = (dn_vols[idx] - VOL_BUMP_ABS).max(1.0e-6);
    }
    let vega_up = price_fn(spots, &up_vols, corr_matrix)?;
    let vega_dn = price_fn(spots, &dn_vols, corr_matrix)?;
    let vega = (vega_up - vega_dn) / (2.0 * VOL_BUMP_ABS);

    let mut cega = f64::NAN;
    let mut corr_bump = CORR_BUMP;
    for _ in 0..6 {
        let up_corr = bump_corr_matrix(corr_matrix, corr_bump);
        let dn_corr = bump_corr_matrix(corr_matrix, -corr_bump);
        let up = price_fn(spots, vols, &up_corr);
        let dn = price_fn(spots, vols, &dn_corr);
        if let (Ok(up_p), Ok(dn_p)) = (up, dn) {
            cega = (up_p - dn_p) / (2.0 * corr_bump);
            break;
        }
        corr_bump *= 0.5;
    }

    Ok(AutocallableSensitivities { delta, vega, cega })
}

#[allow(clippy::too_many_arguments)]
fn simulate_autocallable_paths(
    prepared: &PreparedAutocallable,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<(f64, f64), PricingError> {
    if n_paths == 0 {
        return Err(PricingError::InvalidInput(
            "autocallable n_paths must be > 0".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(PricingError::InvalidInput(
            "autocallable n_steps must be > 0".to_string(),
        ));
    }

    let n_assets = prepared.initial_spots.len();
    let dt = prepared.maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();

    let chol = cholesky_lower(&prepared.corr_matrix).ok_or_else(|| {
        PricingError::InvalidInput("autocallable correlation matrix is not PSD".to_string())
    })?;

    let drift = prepared
        .vols
        .iter()
        .map(|sigma| (r - q - 0.5 * sigma * sigma) * dt)
        .collect::<Vec<_>>();
    let vol_dt = prepared
        .vols
        .iter()
        .map(|v| v * sqrt_dt)
        .collect::<Vec<_>>();

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut indep = vec![0.0_f64; n_assets];
    let mut corr = vec![0.0_f64; n_assets];
    let mut state = prepared.initial_spots.clone();
    let mut discounted_payoffs = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        state.copy_from_slice(&prepared.initial_spots);

        let mut obs_idx = 0usize;
        let mut called = false;
        let mut ki_breached = false;
        let mut worst_final = 1.0_f64;
        let mut pv = 0.0_f64;

        // Phoenix coupon accounting.
        let mut pending_coupon = 0.0_f64;
        let mut prev_obs_time = 0.0_f64;

        for step in 1..=n_steps {
            for z in &mut indep {
                *z = sample_standard_normal(&mut rng);
            }
            correlate_normals(&chol, &indep, &mut corr);

            for i in 0..n_assets {
                state[i] *= (drift[i] + vol_dt[i] * corr[i]).exp();
                state[i] = state[i].max(1.0e-12);
            }

            let worst_ratio = worst_of_ratio(&state, &prepared.initial_spots);
            worst_final = worst_ratio;
            if worst_ratio <= prepared.ki_barrier {
                ki_breached = true;
            }

            while obs_idx < prepared.observation_schedule.len()
                && prepared.observation_schedule[obs_idx].0 == step
            {
                let obs_time = prepared.observation_schedule[obs_idx].1;

                if let Some(coupon_barrier) = prepared.coupon_barrier {
                    let accrual = prepared.notional
                        * prepared.coupon_rate
                        * (obs_time - prev_obs_time).max(0.0);
                    if worst_ratio >= coupon_barrier {
                        let due = if prepared.memory {
                            pending_coupon + accrual
                        } else {
                            accrual
                        };
                        pv += (-r * obs_time).exp() * due;
                        pending_coupon = 0.0;
                    } else if prepared.memory {
                        pending_coupon += accrual;
                    }
                    prev_obs_time = obs_time;
                }

                if worst_ratio >= prepared.autocall_barrier {
                    if prepared.coupon_barrier.is_none() {
                        let payoff = prepared.notional * (1.0 + prepared.coupon_rate * obs_time);
                        pv = (-r * obs_time).exp() * payoff;
                    } else {
                        pv += (-r * obs_time).exp() * prepared.notional;
                    }
                    called = true;
                    break;
                }

                obs_idx += 1;
            }

            if called {
                break;
            }
        }

        if !called {
            let redemption = if ki_breached {
                prepared.notional * (worst_final / prepared.ki_strike)
            } else if prepared.coupon_barrier.is_none() {
                prepared.notional * (1.0 + prepared.coupon_rate * prepared.maturity)
            } else {
                prepared.notional
            };
            pv += (-r * prepared.maturity).exp() * redemption;
        }

        discounted_payoffs.push(pv);
    }

    let n = n_paths as f64;
    let mean = discounted_payoffs.iter().sum::<f64>() / n;
    let variance = if n_paths > 1 {
        discounted_payoffs
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };

    Ok((mean, (variance / n).sqrt()))
}

fn observation_schedule(dates: &[f64], maturity: f64, n_steps: usize) -> Vec<(usize, f64)> {
    let mut map = BTreeMap::<usize, f64>::new();
    for &t in dates {
        let raw = ((t / maturity) * n_steps as f64).round();
        let step = (raw as usize).clamp(1, n_steps);
        map.entry(step)
            .and_modify(|existing| *existing = existing.max(t))
            .or_insert(t);
    }
    map.into_iter().collect()
}

fn worst_of_ratio(state: &[f64], initial: &[f64]) -> f64 {
    state
        .iter()
        .zip(initial.iter())
        .map(|(s, s0)| s / s0)
        .fold(f64::INFINITY, f64::min)
}

fn validate_market_inputs(
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
) -> Result<(), PricingError> {
    if spots.is_empty() {
        return Err(PricingError::InvalidInput(
            "autocallable spots cannot be empty".to_string(),
        ));
    }
    if vols.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "autocallable spots and vols lengths must match".to_string(),
        ));
    }
    if spots.iter().any(|s| *s <= 0.0) {
        return Err(PricingError::InvalidInput(
            "autocallable spots must be > 0".to_string(),
        ));
    }
    if vols.iter().any(|v| *v < 0.0) {
        return Err(PricingError::InvalidInput(
            "autocallable vols must be >= 0".to_string(),
        ));
    }
    validate_correlation_matrix(corr_matrix, spots.len())
}

fn validate_correlation_matrix(
    corr_matrix: &[Vec<f64>],
    n_assets: usize,
) -> Result<(), PricingError> {
    if corr_matrix.len() != n_assets || corr_matrix.iter().any(|row| row.len() != n_assets) {
        return Err(PricingError::InvalidInput(
            "autocallable correlation matrix dimensions must match assets".to_string(),
        ));
    }

    for i in 0..n_assets {
        if (corr_matrix[i][i] - 1.0).abs() > 1.0e-10 {
            return Err(PricingError::InvalidInput(
                "autocallable correlation matrix diagonal must be 1".to_string(),
            ));
        }
        for j in 0..n_assets {
            let rho = corr_matrix[i][j];
            if !(-1.0..=1.0).contains(&rho) {
                return Err(PricingError::InvalidInput(
                    "autocallable correlation entries must be in [-1, 1]".to_string(),
                ));
            }
            if (rho - corr_matrix[j][i]).abs() > 1.0e-10 {
                return Err(PricingError::InvalidInput(
                    "autocallable correlation matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn correlate_normals(chol: &[Vec<f64>], indep: &[f64], out: &mut [f64]) {
    for i in 0..chol.len() {
        let mut sum = 0.0;
        for (j, lij) in chol[i].iter().enumerate().take(i + 1) {
            sum += *lij * indep[j];
        }
        out[i] = sum;
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
            } else if l[j][j] > tol {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Some(l)
}

fn bump_corr_matrix(corr_matrix: &[Vec<f64>], bump: f64) -> Vec<Vec<f64>> {
    let n = corr_matrix.len();
    let mut out = corr_matrix.to_vec();
    for i in 0..n {
        out[i][i] = 1.0;
        for j in (i + 1)..n {
            let bumped = (corr_matrix[i][j] + bump).clamp(-0.999, 0.999);
            out[i][j] = bumped;
            out[j][i] = bumped;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_note(coupon_rate: f64, autocall_barrier: f64) -> Autocallable {
        Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier,
            coupon_rate,
            ki_barrier: 0.2,
            ki_strike: 1.0,
            maturity: 1.0,
        }
    }

    #[test]
    fn autocallable_price_decreases_with_lower_autocall_barrier() {
        let high_barrier = standard_note(0.08, 1.0);
        let low_barrier = standard_note(0.08, 0.8);
        let spots = [100.0, 100.0];
        let vols = [0.12, 0.12];
        let corr = [vec![1.0, 0.4], vec![0.4, 1.0]];

        let high = price_standard_for_inputs(
            &high_barrier,
            &spots,
            &vols,
            &corr,
            0.01,
            0.0,
            16_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;
        let low = price_standard_for_inputs(
            &low_barrier,
            &spots,
            &vols,
            &corr,
            0.01,
            0.0,
            16_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;

        assert!(
            low < high,
            "expected lower barrier to lower value: low={} high={}",
            low,
            high
        );
    }

    #[test]
    fn autocallable_price_increases_with_higher_coupon() {
        let low_coupon = standard_note(0.02, 1.0);
        let high_coupon = standard_note(0.10, 1.0);
        let spots = [100.0, 100.0];
        let vols = [0.20, 0.20];
        let corr = [vec![1.0, 0.3], vec![0.3, 1.0]];

        let low = price_standard_for_inputs(
            &low_coupon,
            &spots,
            &vols,
            &corr,
            0.01,
            0.0,
            16_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;
        let high = price_standard_for_inputs(
            &high_coupon,
            &spots,
            &vols,
            &corr,
            0.01,
            0.0,
            16_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;

        assert!(
            high > low,
            "expected higher coupon to increase value: high={} low={}",
            high,
            low
        );
    }

    #[test]
    fn autocallable_single_underlying_reduces_to_single_stock_case() {
        let one_asset = Autocallable {
            underlyings: vec![0],
            notional: 100.0,
            autocall_dates: vec![0.5, 1.0],
            autocall_barrier: 1.0,
            coupon_rate: 0.07,
            ki_barrier: 0.6,
            ki_strike: 1.0,
            maturity: 1.0,
        };
        let one_asset_shifted_index = Autocallable {
            underlyings: vec![1],
            notional: one_asset.notional,
            autocall_dates: one_asset.autocall_dates.clone(),
            autocall_barrier: one_asset.autocall_barrier,
            coupon_rate: one_asset.coupon_rate,
            ki_barrier: one_asset.ki_barrier,
            ki_strike: one_asset.ki_strike,
            maturity: one_asset.maturity,
        };

        let single = price_standard_for_inputs(
            &one_asset,
            &[100.0],
            &[0.22],
            &[vec![1.0]],
            0.01,
            0.0,
            12_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;
        let indexed = price_standard_for_inputs(
            &one_asset_shifted_index,
            &[80.0, 100.0],
            &[0.35, 0.22],
            &[vec![1.0, -0.2], vec![-0.2, 1.0]],
            0.01,
            0.0,
            12_000,
            64,
            MC_SEED,
        )
        .unwrap()
        .0;

        assert!(
            (single - indexed).abs() < 1.0e-10,
            "single={} indexed={}",
            single,
            indexed
        );
    }

    #[test]
    fn phoenix_coupon_feature_increases_value_vs_standard() {
        let standard = Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 1.05,
            coupon_rate: 0.12,
            ki_barrier: 0.9,
            ki_strike: 1.0,
            maturity: 1.0,
        };
        let phoenix = PhoenixAutocallable {
            underlyings: standard.underlyings.clone(),
            notional: standard.notional,
            autocall_dates: standard.autocall_dates.clone(),
            autocall_barrier: standard.autocall_barrier,
            coupon_barrier: 0.7,
            coupon_rate: standard.coupon_rate,
            memory: true,
            ki_barrier: standard.ki_barrier,
            ki_strike: standard.ki_strike,
            maturity: standard.maturity,
        };
        let spots = [100.0, 100.0];
        let vols = [0.35, 0.35];
        let corr = [vec![1.0, 0.2], vec![0.2, 1.0]];

        let standard_price = price_standard_for_inputs(
            &standard, &spots, &vols, &corr, 0.01, 0.0, 20_000, 80, MC_SEED,
        )
        .unwrap()
        .0;
        let phoenix_price = price_phoenix_for_inputs(
            &phoenix, &spots, &vols, &corr, 0.01, 0.0, 20_000, 80, MC_SEED,
        )
        .unwrap()
        .0;

        assert!(
            phoenix_price > standard_price,
            "phoenix={} standard={}",
            phoenix_price,
            standard_price
        );
    }
}
