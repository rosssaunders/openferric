//! Module `pricing::basket`.
//!
//! Implements basket workflows with concrete routines such as `price_basket_mc`, `basket_sensitivities`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `BasketSensitivities` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::core::{Greeks, PricingError, PricingResult};
use crate::instruments::{BasketOption, BasketType};
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};

const MC_SEED: u64 = 3_148_159;
const SPOT_BUMP_REL: f64 = 0.01;
const VOL_BUMP_ABS: f64 = 0.01;
const CORR_BUMP: f64 = 0.01;

/// Multi-asset basket sensitivities from bump-and-reprice.
#[derive(Debug, Clone)]
pub struct BasketSensitivities {
    /// Delta per underlying spot in input order.
    pub delta: Vec<f64>,
    /// Parallel volatility sensitivity.
    pub vega: f64,
    /// Correlation sensitivity (cega).
    pub cega: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn price_basket_mc(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
) -> PricingResult {
    if let Err(_e) = validate_inputs(basket, spots, vols, corr_matrix, dividends, n_paths) {
        return invalid_result();
    }

    let Ok((price, stderr)) = simulate_basket_once(
        basket,
        spots,
        vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
    ) else {
        return invalid_result();
    };

    let greeks = basket_sensitivities(basket, spots, vols, corr_matrix, r, dividends, n_paths)
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
    diagnostics.insert("rho", average_off_diagonal(corr_matrix).unwrap_or(0.0));

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks,
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn basket_sensitivities(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
) -> Result<BasketSensitivities, PricingError> {
    validate_inputs(basket, spots, vols, corr_matrix, dividends, n_paths)?;

    let mut delta = vec![0.0_f64; spots.len()];
    for i in 0..spots.len() {
        let bump = (spots[i].abs() * SPOT_BUMP_REL).max(1.0e-4);
        let mut up_spots = spots.to_vec();
        let mut dn_spots = spots.to_vec();
        up_spots[i] += bump;
        dn_spots[i] = (dn_spots[i] - bump).max(1.0e-8);

        let up = simulate_basket_once(
            basket,
            &up_spots,
            vols,
            corr_matrix,
            r,
            dividends,
            n_paths,
            MC_SEED,
        )?
        .0;
        let dn = simulate_basket_once(
            basket,
            &dn_spots,
            vols,
            corr_matrix,
            r,
            dividends,
            n_paths,
            MC_SEED,
        )?
        .0;

        delta[i] = (up - dn) / (2.0 * bump);
    }

    let mut up_vols = vols.to_vec();
    let mut dn_vols = vols.to_vec();
    for i in 0..vols.len() {
        up_vols[i] += VOL_BUMP_ABS;
        dn_vols[i] = (dn_vols[i] - VOL_BUMP_ABS).max(1.0e-6);
    }

    let vega_up = simulate_basket_once(
        basket,
        spots,
        &up_vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
    )?
    .0;
    let vega_dn = simulate_basket_once(
        basket,
        spots,
        &dn_vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
    )?
    .0;
    let vega = (vega_up - vega_dn) / (2.0 * VOL_BUMP_ABS);

    let cega = correlation_sensitivity(basket, spots, vols, corr_matrix, r, dividends, n_paths);

    Ok(BasketSensitivities { delta, vega, cega })
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
fn correlation_sensitivity(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
) -> f64 {
    let mut bump = CORR_BUMP;
    for _ in 0..6 {
        let up_corr = bump_corr_matrix(corr_matrix, bump);
        let dn_corr = bump_corr_matrix(corr_matrix, -bump);

        let up = simulate_basket_once(
            basket, spots, vols, &up_corr, r, dividends, n_paths, MC_SEED,
        );
        let dn = simulate_basket_once(
            basket, spots, vols, &dn_corr, r, dividends, n_paths, MC_SEED,
        );

        if let (Ok((up_p, _)), Ok((dn_p, _))) = (up, dn) {
            return (up_p - dn_p) / (2.0 * bump);
        }

        bump *= 0.5;
    }

    f64::NAN
}

#[allow(clippy::too_many_arguments)]
fn simulate_basket_once(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
    seed: u64,
) -> Result<(f64, f64), PricingError> {
    validate_inputs(basket, spots, vols, corr_matrix, dividends, n_paths)?;

    if basket.maturity <= 0.0 {
        let payoff = terminal_payoff(basket, spots, spots);
        return Ok((payoff, 0.0));
    }

    let n_assets = spots.len();
    let dt = basket.maturity;
    let sqrt_dt = dt.sqrt();
    let discount = (-r * basket.maturity).exp();
    let chol = cholesky_lower(corr_matrix).ok_or_else(|| {
        PricingError::InvalidInput("basket correlation matrix is not PSD".to_string())
    })?;

    let drift = vols
        .iter()
        .zip(dividends.iter())
        .map(|(vol, q)| (r - *q - 0.5 * vol * vol) * dt)
        .collect::<Vec<_>>();
    let vol_dt = vols.iter().map(|v| v * sqrt_dt).collect::<Vec<_>>();

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut indep = vec![0.0_f64; n_assets];
    let mut corr = vec![0.0_f64; n_assets];
    let mut terminal = vec![0.0_f64; n_assets];
    let mut payoffs = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        for z in &mut indep {
            *z = sample_standard_normal(&mut rng);
        }
        correlate_normals(&chol, &indep, &mut corr);

        for i in 0..n_assets {
            terminal[i] = spots[i] * (drift[i] + vol_dt[i] * corr[i]).exp();
        }

        payoffs.push(terminal_payoff(basket, spots, &terminal));
    }

    let n = n_paths as f64;
    let mean = payoffs.iter().sum::<f64>() / n;
    let variance = if n_paths > 1 {
        payoffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    Ok((discount * mean, discount * (variance / n).sqrt()))
}

fn terminal_payoff(basket: &BasketOption, initial: &[f64], terminal: &[f64]) -> f64 {
    let underlying = match basket.basket_type {
        BasketType::Average => basket
            .weights
            .iter()
            .zip(terminal.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>(),
        BasketType::BestOf => terminal
            .iter()
            .zip(initial.iter())
            .map(|(st, s0)| st / s0)
            .fold(f64::NEG_INFINITY, f64::max),
        BasketType::WorstOf => terminal
            .iter()
            .zip(initial.iter())
            .map(|(st, s0)| st / s0)
            .fold(f64::INFINITY, f64::min),
    };

    if basket.is_call {
        (underlying - basket.strike).max(0.0)
    } else {
        (basket.strike - underlying).max(0.0)
    }
}

fn validate_inputs(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
    basket.validate()?;
    if n_paths == 0 {
        return Err(PricingError::InvalidInput(
            "basket n_paths must be > 0".to_string(),
        ));
    }
    if spots.is_empty() {
        return Err(PricingError::InvalidInput(
            "basket spots cannot be empty".to_string(),
        ));
    }
    if vols.len() != spots.len() || dividends.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "basket spots/vols/dividends lengths must match".to_string(),
        ));
    }
    if matches!(basket.basket_type, BasketType::Average) && basket.weights.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "average basket weights length must match number of assets".to_string(),
        ));
    }
    if spots.iter().any(|s| *s <= 0.0) {
        return Err(PricingError::InvalidInput(
            "basket spots must be > 0".to_string(),
        ));
    }
    if vols.iter().any(|v| *v < 0.0) {
        return Err(PricingError::InvalidInput(
            "basket vols must be >= 0".to_string(),
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
            "basket correlation matrix dimensions must match asset count".to_string(),
        ));
    }

    for i in 0..n_assets {
        if (corr_matrix[i][i] - 1.0).abs() > 1.0e-10 {
            return Err(PricingError::InvalidInput(
                "basket correlation matrix diagonal must be 1".to_string(),
            ));
        }
        for j in 0..n_assets {
            let rho = corr_matrix[i][j];
            if !(-1.0..=1.0).contains(&rho) {
                return Err(PricingError::InvalidInput(
                    "basket correlation entries must be in [-1, 1]".to_string(),
                ));
            }
            if (rho - corr_matrix[j][i]).abs() > 1.0e-10 {
                return Err(PricingError::InvalidInput(
                    "basket correlation matrix must be symmetric".to_string(),
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

fn average_off_diagonal(corr_matrix: &[Vec<f64>]) -> Option<f64> {
    if corr_matrix.len() < 2 {
        return Some(0.0);
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..corr_matrix.len() {
        for j in (i + 1)..corr_matrix.len() {
            sum += corr_matrix[i][j];
            count += 1;
        }
    }

    (count > 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::OptionType;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn basket_with_one_asset_reduces_to_vanilla_option() {
        let basket = BasketOption {
            weights: vec![1.0],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        };
        let spots = vec![100.0];
        let vols = vec![0.2];
        let dividends = vec![0.0];
        let corr = vec![vec![1.0]];

        let result = price_basket_mc(&basket, &spots, &vols, &corr, 0.03, &dividends, 120_000);
        let bs = black_scholes_price(
            OptionType::Call,
            spots[0],
            basket.strike,
            0.03,
            vols[0],
            basket.maturity,
        );

        let tol = 3.0 * result.stderr.unwrap_or(0.0) + 0.15;
        assert!(
            (result.price - bs).abs() <= tol,
            "basket={} bs={} tol={}",
            result.price,
            bs,
            tol
        );
    }

    #[test]
    fn worst_of_put_is_more_expensive_than_single_asset_put() {
        let two_asset = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: false,
            basket_type: BasketType::WorstOf,
        };
        let single_asset = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: false,
            basket_type: BasketType::WorstOf,
        };

        let two = price_basket_mc(
            &two_asset,
            &[100.0, 100.0],
            &[0.25, 0.25],
            &[vec![1.0, 0.2], vec![0.2, 1.0]],
            0.01,
            &[0.0, 0.0],
            120_000,
        )
        .price;
        let one = price_basket_mc(
            &single_asset,
            &[100.0],
            &[0.25],
            &[vec![1.0]],
            0.01,
            &[0.0],
            120_000,
        )
        .price;

        assert!(two > one, "worst-of two-asset={} single-asset={}", two, one);
    }

    #[test]
    fn best_of_call_is_more_expensive_than_single_asset_call() {
        let two_asset = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::BestOf,
        };
        let single_asset = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::BestOf,
        };

        let two = price_basket_mc(
            &two_asset,
            &[100.0, 100.0],
            &[0.25, 0.25],
            &[vec![1.0, 0.2], vec![0.2, 1.0]],
            0.01,
            &[0.0, 0.0],
            120_000,
        )
        .price;
        let one = price_basket_mc(
            &single_asset,
            &[100.0],
            &[0.25],
            &[vec![1.0]],
            0.01,
            &[0.0],
            120_000,
        )
        .price;

        assert!(two > one, "best-of two-asset={} single-asset={}", two, one);
    }
}
