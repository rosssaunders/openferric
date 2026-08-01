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

use crate::core::{DiagKey, Greeks, PricingError, PricingResult};
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
    /// Simulation start values (pricing spots). Spot bumps for delta apply
    /// here only.
    pricing_spots: Vec<f64>,
    /// Initial fixings: performance denominator and barrier reference. These
    /// stay fixed under spot bumps because barriers/strikes are struck at the
    /// initial fixing, not at the pricing spot.
    initial_fixings: Vec<f64>,
    vols: Vec<f64>,
    corr_matrix: Vec<Vec<f64>>,
    /// Lower Cholesky factor of `corr_matrix`, computed once at preparation
    /// time so spot/vol bumps can reuse it without re-factorizing.
    chol: Vec<Vec<f64>>,
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

/// Prices a standard autocallable note (no Greeks).
///
/// Use [`price_autocallable_with_greeks`] when the bump-and-reprice Greeks
/// ladder is required; it costs roughly 9x a plain price call.
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
    let Ok(prepared) = prepare_standard(autocall, spots, vols, corr_matrix, n_steps) else {
        return invalid_result();
    };
    price_prepared(
        &prepared,
        autocall.autocall_dates.len(),
        r,
        q,
        n_paths,
        n_steps,
        false,
    )
}

/// Prices a standard autocallable note including bump-and-reprice Greeks.
#[allow(clippy::too_many_arguments)]
pub fn price_autocallable_with_greeks(
    autocall: &Autocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    let Ok(prepared) = prepare_standard(autocall, spots, vols, corr_matrix, n_steps) else {
        return invalid_result();
    };
    price_prepared(
        &prepared,
        autocall.autocall_dates.len(),
        r,
        q,
        n_paths,
        n_steps,
        true,
    )
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
    bump_and_reprice_sensitivities(&prepared, r, q, n_paths, n_steps, MC_SEED)
}

/// Prices a phoenix autocallable note (no Greeks).
///
/// Use [`price_phoenix_autocallable_with_greeks`] when the bump-and-reprice
/// Greeks ladder is required; it costs roughly 9x a plain price call.
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
    let Ok(prepared) = prepare_phoenix(phoenix, spots, vols, corr_matrix, n_steps) else {
        return invalid_result();
    };
    price_prepared(
        &prepared,
        phoenix.autocall_dates.len(),
        r,
        q,
        n_paths,
        n_steps,
        false,
    )
}

/// Prices a phoenix autocallable note including bump-and-reprice Greeks.
#[allow(clippy::too_many_arguments)]
pub fn price_phoenix_autocallable_with_greeks(
    phoenix: &PhoenixAutocallable,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    let Ok(prepared) = prepare_phoenix(phoenix, spots, vols, corr_matrix, n_steps) else {
        return invalid_result();
    };
    price_prepared(
        &prepared,
        phoenix.autocall_dates.len(),
        r,
        q,
        n_paths,
        n_steps,
        true,
    )
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
    bump_and_reprice_sensitivities(&prepared, r, q, n_paths, n_steps, MC_SEED)
}

#[allow(clippy::too_many_arguments)]
fn price_prepared(
    prepared: &PreparedAutocallable,
    observation_count: usize,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    with_greeks: bool,
) -> PricingResult {
    let Ok((price, stderr)) =
        simulate_autocallable_paths(prepared, r, q, n_paths, n_steps, MC_SEED)
    else {
        return invalid_result();
    };

    let greeks = if with_greeks {
        bump_and_reprice_sensitivities(prepared, r, q, n_paths, n_steps, MC_SEED)
            .ok()
            .map(|s| Greeks {
                delta: s.delta.iter().sum::<f64>(),
                gamma: 0.0,
                vega: s.vega,
                theta: 0.0,
                rho: s.cega,
            })
    } else {
        None
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert_key(DiagKey::NumPaths, n_paths as f64);
    diagnostics.insert_key(DiagKey::NumSteps, n_steps as f64);
    diagnostics.insert_key(DiagKey::ObservationCount, observation_count as f64);

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks,
        diagnostics,
    }
}

fn invalid_result() -> PricingResult {
    PricingResult {
        price: f64::NAN,
        stderr: None,
        greeks: None,
        diagnostics: crate::core::Diagnostics::new(),
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

    let mut selected_spots = Vec::with_capacity(underlyings.len());
    let mut selected_vols = Vec::with_capacity(underlyings.len());
    for &idx in underlyings {
        selected_spots.push(spots[idx]);
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

    let chol = cholesky_lower(&selected_corr).ok_or_else(|| {
        PricingError::InvalidInput("autocallable correlation matrix is not PSD".to_string())
    })?;

    Ok(PreparedAutocallable {
        pricing_spots: selected_spots.clone(),
        initial_fixings: selected_spots,
        vols: selected_vols,
        corr_matrix: selected_corr,
        chol,
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

/// Bump-and-reprice sensitivities computed in prepared (contract) space.
///
/// Spot and vol bumps reuse the Cholesky factor computed once at preparation
/// time; only correlation bumps (cega) re-factorize the bumped matrix.
fn bump_and_reprice_sensitivities(
    prepared: &PreparedAutocallable,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<AutocallableSensitivities, PricingError> {
    let n_assets = prepared.pricing_spots.len();

    let mut delta = Vec::with_capacity(n_assets);
    for k in 0..n_assets {
        let bump = (prepared.pricing_spots[k].abs() * SPOT_BUMP_REL).max(1.0e-4);
        // Bump the pricing spot only: barriers and performance ratios remain
        // struck at the original initial fixings.
        let mut up = prepared.clone();
        up.pricing_spots[k] += bump;
        let mut dn = prepared.clone();
        dn.pricing_spots[k] = (dn.pricing_spots[k] - bump).max(1.0e-8);

        let up_p = simulate_autocallable_paths(&up, r, q, n_paths, n_steps, seed)?.0;
        let dn_p = simulate_autocallable_paths(&dn, r, q, n_paths, n_steps, seed)?.0;
        delta.push((up_p - dn_p) / (2.0 * bump));
    }

    let mut up_vols = prepared.clone();
    let mut dn_vols = prepared.clone();
    for k in 0..n_assets {
        up_vols.vols[k] += VOL_BUMP_ABS;
        dn_vols.vols[k] = (dn_vols.vols[k] - VOL_BUMP_ABS).max(1.0e-6);
    }
    let vega_up = simulate_autocallable_paths(&up_vols, r, q, n_paths, n_steps, seed)?.0;
    let vega_dn = simulate_autocallable_paths(&dn_vols, r, q, n_paths, n_steps, seed)?.0;
    let vega = (vega_up - vega_dn) / (2.0 * VOL_BUMP_ABS);

    let mut cega = f64::NAN;
    let mut corr_bump = CORR_BUMP;
    for _ in 0..6 {
        let up_corr = bump_corr_matrix(&prepared.corr_matrix, corr_bump);
        let dn_corr = bump_corr_matrix(&prepared.corr_matrix, -corr_bump);

        if let (Some(up_chol), Some(dn_chol)) = (cholesky_lower(&up_corr), cholesky_lower(&dn_corr))
        {
            let mut up = prepared.clone();
            up.corr_matrix = up_corr;
            up.chol = up_chol;
            let mut dn = prepared.clone();
            dn.corr_matrix = dn_corr;
            dn.chol = dn_chol;

            let up_p = simulate_autocallable_paths(&up, r, q, n_paths, n_steps, seed)?.0;
            let dn_p = simulate_autocallable_paths(&dn, r, q, n_paths, n_steps, seed)?.0;
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

    let n_assets = prepared.pricing_spots.len();
    let dt = prepared.maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();

    let chol = &prepared.chol;

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
    let mut state = prepared.pricing_spots.clone();
    let mut discounted_payoffs = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        state.copy_from_slice(&prepared.pricing_spots);

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
            correlate_normals(chol, &indep, &mut corr);

            for i in 0..n_assets {
                state[i] *= (drift[i] + vol_dt[i] * corr[i]).exp();
                state[i] = state[i].max(1.0e-12);
            }

            let worst_ratio = worst_of_ratio(&state, &prepared.initial_fixings);
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
                // Knock-in put: holder bears downside but redemption is capped at par
                // even if the worst-of recovers above the KI strike by maturity.
                prepared.notional * (worst_final / prepared.ki_strike).min(1.0)
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

    for (i, row_i) in corr_matrix.iter().enumerate().take(n_assets) {
        if (row_i[i] - 1.0).abs() > 1.0e-10 {
            return Err(PricingError::InvalidInput(
                "autocallable correlation matrix diagonal must be 1".to_string(),
            ));
        }
        for (j, rho) in row_i.iter().copied().enumerate().take(n_assets) {
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
            for (&lik, &ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
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
    fn knock_in_redemption_capped_at_par() {
        // KI barrier above all paths (always breached), autocall barrier never hit,
        // zero coupon, zero rate: redemption is min(worst/ki_strike, 1)*notional,
        // so the price can never exceed par even when worst recovers above ki_strike.
        let note = Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 10.0,
            coupon_rate: 0.0,
            ki_barrier: 5.0,
            ki_strike: 0.5,
            maturity: 1.0,
        };
        let spots = [100.0, 100.0];
        // Zero volatility and r=q=0 make every path deterministic.  The KI is
        // breached, worst/strike=2, and the contractual cap therefore pays
        // exactly par at maturity.
        let vols = [0.0, 0.0];
        let corr = [vec![1.0, 0.3], vec![0.3, 1.0]];

        let (price, stderr) =
            price_standard_for_inputs(&note, &spots, &vols, &corr, 0.0, 0.0, 16_000, 64, MC_SEED)
                .unwrap();

        let roundoff = 32.0 * f64::EPSILON * note.notional;
        assert!(
            (price - note.notional).abs() <= 4.0 * stderr + roundoff,
            "deterministic capped KI redemption mismatch: price={price} exact={} stderr={stderr}",
            note.notional
        );
    }

    #[test]
    fn zero_vol_standard_and_phoenix_autocall_at_first_observation_exactly() {
        let standard = standard_note(0.08, 1.0);
        let phoenix = PhoenixAutocallable {
            underlyings: standard.underlyings.clone(),
            notional: standard.notional,
            autocall_dates: standard.autocall_dates.clone(),
            autocall_barrier: 1.0,
            coupon_barrier: 0.5,
            coupon_rate: standard.coupon_rate,
            memory: true,
            ki_barrier: standard.ki_barrier,
            ki_strike: standard.ki_strike,
            maturity: standard.maturity,
        };
        let spots = [100.0, 100.0];
        let vols = [0.0, 0.0];
        let corr = [vec![1.0, 0.3], vec![0.3, 1.0]];
        let rate = 0.03;
        let first_observation = standard.autocall_dates[0];

        let (standard_price, standard_se) = price_standard_for_inputs(
            &standard, &spots, &vols, &corr, rate, rate, 256, 64, MC_SEED,
        )
        .unwrap();
        let standard_exact = (-rate * first_observation).exp()
            * standard.notional
            * (1.0 + standard.coupon_rate * first_observation);
        let (phoenix_price, phoenix_se) =
            price_phoenix_for_inputs(&phoenix, &spots, &vols, &corr, rate, rate, 256, 64, MC_SEED)
                .unwrap();
        let phoenix_exact = (-rate * first_observation).exp()
            * phoenix.notional
            * (1.0 + phoenix.coupon_rate * first_observation);

        for (label, value, exact, stderr) in [
            ("standard", standard_price, standard_exact, standard_se),
            ("phoenix", phoenix_price, phoenix_exact, phoenix_se),
        ] {
            let roundoff = 64.0 * f64::EPSILON * exact.abs().max(1.0);
            assert!(
                (value - exact).abs() <= 4.0 * stderr + roundoff,
                "zero-vol {label} first-call mismatch: value={value} exact={exact} stderr={stderr}"
            );
        }
    }

    #[test]
    fn immediate_call_autocallable_has_exact_zero_vega_and_cega() {
        // Every simulated state is strictly positive (and floored at 1e-12),
        // so this tiny barrier guarantees redemption at the first observation
        // for every volatility and correlation bump. The discounted contractual
        // cash flow is therefore independent of both parameters.
        let standard = Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 1.0e-15,
            coupon_rate: 0.08,
            ki_barrier: 0.5,
            ki_strike: 1.0,
            maturity: 1.0,
        };
        let phoenix = PhoenixAutocallable {
            underlyings: standard.underlyings.clone(),
            notional: standard.notional,
            autocall_dates: standard.autocall_dates.clone(),
            autocall_barrier: standard.autocall_barrier,
            coupon_barrier: 1.0e-15,
            coupon_rate: standard.coupon_rate,
            memory: true,
            ki_barrier: standard.ki_barrier,
            ki_strike: standard.ki_strike,
            maturity: standard.maturity,
        };
        let spots = [100.0, 95.0];
        let vols = [0.20, 0.30];
        let corr = [vec![1.0, 0.35], vec![0.35, 1.0]];
        let standard_sens =
            autocallable_sensitivities(&standard, &spots, &vols, &corr, 0.03, 0.01, 512, 16)
                .unwrap();
        let phoenix_sens =
            phoenix_autocallable_sensitivities(&phoenix, &spots, &vols, &corr, 0.03, 0.01, 512, 16)
                .unwrap();

        for (label, sensitivities) in [("standard", standard_sens), ("phoenix", phoenix_sens)] {
            assert_eq!(sensitivities.vega, 0.0, "{label} immediate-call vega");
            assert_eq!(sensitivities.cega, 0.0, "{label} immediate-call cega");
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
    fn deep_ki_worst_of_delta_matches_deterministic_reduction() {
        // KI barrier far above any path (breach is certain), autocall barrier
        // unreachable, zero coupon, zero rates: the payoff is effectively
        // notional * min(worst_T / ki_strike, 1). Bumping the pricing spots up
        // raises worst_T while the initial fixings (performance denominator)
        // stay struck, so delta must be significantly positive. A structurally
        // zero delta (the old bug, where bumps scaled numerator and
        // denominator identically) fails this test.
        let note = Autocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 10.0,
            coupon_rate: 0.0,
            ki_barrier: 5.0,
            ki_strike: 1.0,
            maturity: 1.0,
        };
        let spots = [100.0, 100.0];
        let vols = [0.0, 0.0];
        let corr = [vec![1.0, 0.3], vec![0.3, 1.0]];
        let dividend_yield = 0.20;

        let sens =
            autocallable_sensitivities(&note, &spots, &vols, &corr, 0.0, dividend_yield, 1_024, 64)
                .unwrap();

        let delta_sum: f64 = sens.delta.iter().sum();
        // With deterministic S_T/S_0 = exp(-qT), the min-of-two central bump
        // gives half the redemption slope to each identical underlying.
        let steps = 64_usize;
        let step_growth = (-dividend_yield * note.maturity / steps as f64).exp();
        let terminal = (0..steps).fold(spots[0], |state, _| state * step_growth);
        let simulated_growth = terminal / spots[0];
        let expected_sum = simulated_growth * note.notional / spots[0] / note.ki_strike;
        // The deterministic path value is summed over 1,024 identical paths
        // before the central difference; account for that accumulation only.
        let roundoff = 32.0 * 1_024.0 * f64::EPSILON * expected_sum.abs().max(1.0);
        assert!(
            (delta_sum - expected_sum).abs() <= roundoff,
            "deterministic deep-KI delta mismatch: deltas={:?} sum={delta_sum} exact={expected_sum}",
            sens.delta
        );
        for (k, d) in sens.delta.iter().enumerate() {
            assert!(d.is_finite(), "delta[{k}] must be finite: {d}");
        }
    }

    #[test]
    fn deep_ki_phoenix_delta_matches_deterministic_reduction() {
        // Same construction as the standard deep-KI note, phoenix variant:
        // coupon barrier unreachable so the payoff reduces to the knock-in
        // redemption leg, which is monotone in the pricing spots.
        let phoenix = PhoenixAutocallable {
            underlyings: vec![0, 1],
            notional: 100.0,
            autocall_dates: vec![0.25, 0.5, 0.75, 1.0],
            autocall_barrier: 10.0,
            coupon_barrier: 10.0,
            coupon_rate: 0.08,
            memory: false,
            ki_barrier: 5.0,
            ki_strike: 1.0,
            maturity: 1.0,
        };
        let spots = [100.0, 100.0];
        let vols = [0.0, 0.0];
        let corr = [vec![1.0, 0.3], vec![0.3, 1.0]];
        let dividend_yield = 0.20;

        let sens = phoenix_autocallable_sensitivities(
            &phoenix,
            &spots,
            &vols,
            &corr,
            0.0,
            dividend_yield,
            1_024,
            64,
        )
        .unwrap();

        let delta_sum: f64 = sens.delta.iter().sum();
        let steps = 64_usize;
        let step_growth = (-dividend_yield * phoenix.maturity / steps as f64).exp();
        let terminal = (0..steps).fold(spots[0], |state, _| state * step_growth);
        let simulated_growth = terminal / spots[0];
        let expected_sum = simulated_growth * phoenix.notional / spots[0] / phoenix.ki_strike;
        let roundoff = 32.0 * 1_024.0 * f64::EPSILON * expected_sum.abs().max(1.0);
        assert!(
            (delta_sum - expected_sum).abs() <= roundoff,
            "deterministic deep-KI phoenix delta mismatch: deltas={:?} sum={delta_sum} exact={expected_sum}",
            sens.delta
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

        let (standard_price, standard_stderr) = price_standard_for_inputs(
            &standard, &spots, &vols, &corr, 0.01, 0.0, 20_000, 80, MC_SEED,
        )
        .unwrap();
        let (phoenix_price, phoenix_stderr) = price_phoenix_for_inputs(
            &phoenix, &spots, &vols, &corr, 0.01, 0.0, 20_000, 80, MC_SEED,
        )
        .unwrap();

        // Independent SciPy Brownian-bridge Sobol references for these exact
        // 80-step, two-asset contracts (64 Owen scrambles x 2^15 paths).
        // The replicate uncertainty is combined with each MC-reported SE.
        for (label, value, stderr, reference, reference_stderr) in [
            (
                "standard",
                standard_price,
                standard_stderr,
                82.666_147_186_488_99,
                2.953_048_739_249_488_2e-3,
            ),
            (
                "phoenix",
                phoenix_price,
                phoenix_stderr,
                88.513_296_306_547_33,
                2.102_584_152_148_88e-3,
            ),
        ] {
            let combined_stderr = stderr.hypot(reference_stderr);
            let roundoff = 32.0 * f64::EPSILON * reference;
            assert!(
                (value - reference).abs() <= 4.0 * combined_stderr + roundoff,
                "{label} autocall mismatch: mc={value} reference={reference} mc_stderr={stderr} reference_stderr={reference_stderr}"
            );
        }

        // Supplemental contractual invariant: conditional phoenix coupons
        // add value relative to the otherwise matched standard note.
        assert!(
            phoenix_price > standard_price,
            "phoenix={} standard={}",
            phoenix_price,
            standard_price
        );
    }

    #[test]
    fn stochastic_autocall_greeks_match_independent_scipy_sobol_references() {
        const N_REPLICATES: usize = 12;
        const PATHS_PER_REPLICATE: usize = 30_000;
        const N_STEPS: usize = 12;

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
        let vols = [0.35, 0.30];
        let corr = [vec![1.0, 0.2], vec![0.2, 1.0]];
        let standard_prepared = prepare_standard(&standard, &spots, &vols, &corr, N_STEPS).unwrap();
        let phoenix_prepared = prepare_phoenix(&phoenix, &spots, &vols, &corr, N_STEPS).unwrap();

        let mut standard_samples = [[0.0_f64; 4]; N_REPLICATES];
        let mut phoenix_samples = [[0.0_f64; 4]; N_REPLICATES];
        for replicate in 0..N_REPLICATES {
            let seed = 8_101 + 104_729 * replicate as u64;
            for (prepared, samples) in [
                (&standard_prepared, &mut standard_samples),
                (&phoenix_prepared, &mut phoenix_samples),
            ] {
                let result = bump_and_reprice_sensitivities(
                    prepared,
                    0.01,
                    0.0,
                    PATHS_PER_REPLICATE,
                    N_STEPS,
                    seed,
                )
                .unwrap();
                samples[replicate] = [result.delta[0], result.delta[1], result.vega, result.cega];
            }
        }

        fn mean_and_se<const N: usize>(samples: &[[f64; 4]; N], index: usize) -> (f64, f64) {
            let mean = samples.iter().map(|row| row[index]).sum::<f64>() / N as f64;
            let variance = samples
                .iter()
                .map(|row| (row[index] - mean).powi(2))
                .sum::<f64>()
                / (N - 1) as f64;
            (mean, (variance / N as f64).sqrt())
        }

        // Independent SciPy 1.17.1 inverse-normal Sobol integration of the
        // exact 12-step contracts: 24 Owen scrambles x 2^16 paths.  Greeks use
        // the public API's central bumps (1% spot, 1 vol point, 1 correlation
        // point) with common Sobol paths.  Replicate standard errors below are
        // combined with the implementation's independently seeded MC error.
        let references = [
            (
                "standard",
                &standard_samples,
                [
                    (0.410_612_541_152_064_9, 1.397_510_588_775_907e-3),
                    (0.410_986_285_283_688_5, 1.252_746_121_642_196_2e-3),
                    (-50.262_723_149_640_89, 5.541_320_952_644_099e-2),
                    (6.475_731_286_827_004, 5.225_070_338_970_829e-2),
                ],
            ),
            (
                "phoenix",
                &phoenix_samples,
                [
                    (0.349_123_514_024_741_73, 1.001_627_942_994_87e-3),
                    (0.319_378_694_272_530_8, 9.405_057_986_520_466e-4),
                    (-61.475_090_367_831_66, 6.074_592_022_665_935e-2),
                    (4.391_667_654_829_028, 3.792_074_108_317_604e-2),
                ],
            ),
        ];
        let labels = ["delta[0]", "delta[1]", "vega", "cega"];

        for (contract, samples, targets) in references {
            for (index, label) in labels.iter().enumerate() {
                let (actual, implementation_se) = mean_and_se(samples, index);
                let (target, reference_se) = targets[index];
                let combined_se = implementation_se.hypot(reference_se);
                assert!(
                    (actual - target).abs() <= 4.0 * combined_se,
                    "{contract} {label}: actual={actual} target={target} implementation_se={implementation_se} reference_se={reference_se}"
                );
            }
        }
    }
}
