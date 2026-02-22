//! Basket option pricing for multi-asset products.
//!
//! Supported capabilities include:
//! - N-asset correlated Monte Carlo with matrix or factor-model dependence,
//! - Gaussian and Student-t copula simulation,
//! - correlation stress scenarios with PSD repair,
//! - fast moment-matching approximations (Levy, Gentle-style).
//!
//! References:
//! - Levy, E. (1992): arithmetic average approximations.
//! - Gentle, J. (1993): higher-moment style refinements.
//! - Higham, N. (2002): nearest correlation matrix projection.

use crate::core::{OptionType, PricingError, PricingResult};
use crate::instruments::{
    BasketOption, BasketType, OutperformanceBasketOption, QuantoBasketOption,
};
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};
use crate::math::normal_cdf;
use crate::math::{
    CopulaFamily, CorrelationStressScenario, FactorCorrelationModel, PsdProjectionConfig,
    apply_correlation_stress, cholesky_lower_psd, copula_uniforms_to_normals, correlate_normals,
    sample_copula_uniforms_from_cholesky, sample_copula_uniforms_from_factor_model,
    validate_correlation_matrix, validate_or_repair_correlation_matrix,
};

const MC_SEED: u64 = 3_148_159;
const SPOT_BUMP_REL: f64 = 0.01;
const VOL_BUMP_ABS: f64 = 0.01;
const CORR_BUMP: f64 = 0.01;

/// Copula choice for basket Monte Carlo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasketCopula {
    Gaussian,
    StudentT { degrees_of_freedom: u32 },
}

impl BasketCopula {
    fn to_core(self) -> CopulaFamily {
        match self {
            Self::Gaussian => CopulaFamily::Gaussian,
            Self::StudentT { degrees_of_freedom } => CopulaFamily::StudentT { degrees_of_freedom },
        }
    }
}

/// Moment-matching method for weighted-average baskets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasketMomentMatchingMethod {
    /// Two-moment lognormal fit.
    Levy,
    /// Three-moment adjusted lognormal fit.
    Gentle,
}

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

#[derive(Debug, Clone, Copy, Default)]
struct RunningStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(self) -> f64 {
        self.mean
    }

    fn sample_variance(self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n as f64 - 1.0)
        } else {
            0.0
        }
    }
}

/// Baseline basket Monte Carlo under Gaussian copula.
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
    price_basket_mc_with_copula(
        basket,
        spots,
        vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        BasketCopula::Gaussian,
    )
}

/// Basket Monte Carlo with explicit copula family.
#[allow(clippy::too_many_arguments)]
pub fn price_basket_mc_with_copula(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
    copula: BasketCopula,
) -> PricingResult {
    if let Err(_e) = validate_inputs_matrix(basket, spots, vols, corr_matrix, dividends, n_paths) {
        return invalid_result();
    }

    let Ok((price, stderr)) = simulate_basket_once_matrix(
        basket,
        spots,
        vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
        copula,
    ) else {
        return invalid_result();
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("rho", average_off_diagonal(corr_matrix).unwrap_or(0.0));

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks: None,
        diagnostics,
    }
}

/// Basket Monte Carlo using a factor correlation model (recommended for large baskets).
#[allow(clippy::too_many_arguments)]
pub fn price_basket_mc_with_factor_model(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    factor_model: &FactorCorrelationModel,
    r: f64,
    dividends: &[f64],
    n_paths: usize,
    copula: BasketCopula,
) -> PricingResult {
    if let Err(_e) = validate_inputs_factor(basket, spots, vols, factor_model, dividends, n_paths) {
        return invalid_result();
    }

    let Ok((price, stderr)) = simulate_basket_once_factor(
        basket,
        spots,
        vols,
        factor_model,
        r,
        dividends,
        n_paths,
        MC_SEED,
        copula,
    ) else {
        return invalid_result();
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("rho", 0.0);

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks: None,
        diagnostics,
    }
}

/// Fast indicative basket pricing via moment-matching approximation.
///
/// Currently supported for `BasketType::Average` only.
#[allow(clippy::too_many_arguments)]
pub fn price_basket_moment_matching(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    method: BasketMomentMatchingMethod,
) -> Result<f64, PricingError> {
    validate_inputs_matrix(basket, spots, vols, corr_matrix, dividends, 1)?;

    if !matches!(basket.basket_type, BasketType::Average) {
        return Err(PricingError::InvalidInput(
            "moment matching is only implemented for arithmetic weighted baskets".to_string(),
        ));
    }

    if basket.maturity <= 0.0 {
        let intrinsic = terminal_payoff(basket, spots, spots);
        return Ok(intrinsic);
    }

    let (corr, _repaired) = validate_or_repair_correlation_matrix(
        corr_matrix,
        spots.len(),
        PsdProjectionConfig::default(),
    )
    .map_err(PricingError::InvalidInput)?;

    let (m1, m2, m3) = weighted_basket_moments(
        spots,
        &basket.weights,
        vols,
        &corr,
        r,
        dividends,
        basket.maturity,
    )?;
    if m1 <= 0.0 || !m1.is_finite() {
        return Err(PricingError::InvalidInput(
            "invalid first moment in basket moment matching".to_string(),
        ));
    }

    let sigma2_levy = ((m2 / (m1 * m1)).max(1.0 + 1.0e-14)).ln();
    let sigma2 = match method {
        BasketMomentMatchingMethod::Levy => sigma2_levy,
        BasketMomentMatchingMethod::Gentle => {
            let sigma2_m3 = ((m3 / m1.powi(3)).max(1.0 + 1.0e-14)).ln() / 3.0;
            0.5 * (sigma2_levy + sigma2_m3).max(1.0e-14)
        }
    };

    let sigma = sigma2.sqrt();
    let k = basket.strike;
    let discount = (-r * basket.maturity).exp();

    let call_undiscounted = if k <= 0.0 {
        m1
    } else {
        let d1 = ((m1 / k).ln() + 0.5 * sigma2) / sigma;
        let d2 = d1 - sigma;
        m1 * normal_cdf(d1) - k * normal_cdf(d2)
    };

    let put_undiscounted = if k <= 0.0 {
        0.0
    } else {
        let d1 = ((m1 / k).ln() + 0.5 * sigma2) / sigma;
        let d2 = d1 - sigma;
        k * normal_cdf(-d2) - m1 * normal_cdf(-d1)
    };

    Ok(if basket.is_call {
        discount * call_undiscounted
    } else {
        discount * put_undiscounted
    })
}

/// Monte Carlo pricing of outperformance basket options.
#[allow(clippy::too_many_arguments)]
pub fn price_outperformance_basket_mc(
    option: &OutperformanceBasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
) -> PricingResult {
    if let Err(_e) =
        validate_outperformance_inputs(option, spots, vols, corr_matrix, dividends, n_paths)
    {
        return invalid_result();
    }

    let Ok((corr, _)) = validate_or_repair_correlation_matrix(
        corr_matrix,
        spots.len(),
        PsdProjectionConfig::default(),
    ) else {
        return invalid_result();
    };

    let Some(chol) = cholesky_lower_psd(&corr, 1.0e-12) else {
        return invalid_result();
    };

    let t = option.maturity;
    if t <= 0.0 {
        let payoff = outperformance_payoff(option, spots);
        return PricingResult {
            price: payoff,
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, MC_SEED);
    let mut indep = vec![0.0; spots.len()];
    let mut corr_z = vec![0.0; spots.len()];
    let mut terminal = vec![0.0; spots.len()];

    let drift = vols
        .iter()
        .zip(dividends.iter())
        .map(|(v, q)| (r - *q - 0.5 * v * v) * t)
        .collect::<Vec<_>>();
    let vol_t = vols.iter().map(|v| v * t.sqrt()).collect::<Vec<_>>();

    let mut stats = RunningStats::default();
    for _ in 0..n_paths {
        for z in &mut indep {
            *z = sample_standard_normal(&mut rng);
        }
        correlate_normals(&chol, &indep, &mut corr_z);

        for i in 0..spots.len() {
            terminal[i] = spots[i] * (drift[i] + vol_t[i] * corr_z[i]).exp();
        }

        stats.push(outperformance_payoff(option, &terminal));
    }

    let discount = (-r * t).exp();
    let mean = stats.mean();
    let stderr = (stats.sample_variance() / n_paths as f64).sqrt();

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * stderr),
        greeks: None,
        diagnostics: crate::core::Diagnostics::new(),
    }
}

/// Monte Carlo pricing of quanto baskets with fixed FX conversion.
#[allow(clippy::too_many_arguments)]
pub fn price_quanto_basket_mc(
    option: &QuantoBasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    dividends: &[f64],
    n_paths: usize,
) -> PricingResult {
    if let Err(_e) = validate_quanto_inputs(option, spots, vols, corr_matrix, dividends, n_paths) {
        return invalid_result();
    }

    let Ok((corr, _)) = validate_or_repair_correlation_matrix(
        corr_matrix,
        spots.len(),
        PsdProjectionConfig::default(),
    ) else {
        return invalid_result();
    };

    let Some(chol) = cholesky_lower_psd(&corr, 1.0e-12) else {
        return invalid_result();
    };

    let t = option.basket.maturity;
    if t <= 0.0 {
        let payoff = option.fx_rate * terminal_payoff(&option.basket, spots, spots);
        return PricingResult {
            price: payoff,
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, MC_SEED);
    let mut indep = vec![0.0; spots.len()];
    let mut corr_z = vec![0.0; spots.len()];
    let mut terminal = vec![0.0; spots.len()];

    let drift = vols
        .iter()
        .zip(dividends.iter())
        .zip(option.asset_fx_corr.iter())
        .map(|((v, q), rho_fx)| {
            let quanto_adj = rho_fx * *v * option.fx_vol;
            (option.foreign_rate - *q - quanto_adj - 0.5 * v * v) * t
        })
        .collect::<Vec<_>>();
    let vol_t = vols.iter().map(|v| v * t.sqrt()).collect::<Vec<_>>();

    let mut stats = RunningStats::default();
    for _ in 0..n_paths {
        for z in &mut indep {
            *z = sample_standard_normal(&mut rng);
        }
        correlate_normals(&chol, &indep, &mut corr_z);

        for i in 0..spots.len() {
            terminal[i] = spots[i] * (drift[i] + vol_t[i] * corr_z[i]).exp();
        }

        let payoff = option.fx_rate * terminal_payoff(&option.basket, spots, &terminal);
        stats.push(payoff);
    }

    let discount = (-option.domestic_rate * t).exp();
    let mean = stats.mean();
    let stderr = (stats.sample_variance() / n_paths as f64).sqrt();

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * stderr),
        greeks: None,
        diagnostics: crate::core::Diagnostics::new(),
    }
}

/// Applies stressed-correlation scenarios and returns a PSD-repaired matrix.
pub fn stressed_correlation_matrix(
    corr_matrix: &[Vec<f64>],
    scenarios: &[CorrelationStressScenario],
) -> Result<Vec<Vec<f64>>, PricingError> {
    apply_correlation_stress(corr_matrix, scenarios, true, PsdProjectionConfig::default())
        .map_err(PricingError::InvalidInput)
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
    validate_inputs_matrix(basket, spots, vols, corr_matrix, dividends, n_paths)?;

    let mut delta = vec![0.0_f64; spots.len()];
    for i in 0..spots.len() {
        let bump = (spots[i].abs() * SPOT_BUMP_REL).max(1.0e-4);
        let mut up_spots = spots.to_vec();
        let mut dn_spots = spots.to_vec();
        up_spots[i] += bump;
        dn_spots[i] = (dn_spots[i] - bump).max(1.0e-8);

        let up = simulate_basket_once_matrix(
            basket,
            &up_spots,
            vols,
            corr_matrix,
            r,
            dividends,
            n_paths,
            MC_SEED,
            BasketCopula::Gaussian,
        )?
        .0;
        let dn = simulate_basket_once_matrix(
            basket,
            &dn_spots,
            vols,
            corr_matrix,
            r,
            dividends,
            n_paths,
            MC_SEED,
            BasketCopula::Gaussian,
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

    let vega_up = simulate_basket_once_matrix(
        basket,
        spots,
        &up_vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
        BasketCopula::Gaussian,
    )?
    .0;
    let vega_dn = simulate_basket_once_matrix(
        basket,
        spots,
        &dn_vols,
        corr_matrix,
        r,
        dividends,
        n_paths,
        MC_SEED,
        BasketCopula::Gaussian,
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

        let up = simulate_basket_once_matrix(
            basket,
            spots,
            vols,
            &up_corr,
            r,
            dividends,
            n_paths,
            MC_SEED,
            BasketCopula::Gaussian,
        );
        let dn = simulate_basket_once_matrix(
            basket,
            spots,
            vols,
            &dn_corr,
            r,
            dividends,
            n_paths,
            MC_SEED,
            BasketCopula::Gaussian,
        );

        if let (Ok((up_p, _)), Ok((dn_p, _))) = (up, dn) {
            return (up_p - dn_p) / (2.0 * bump);
        }

        bump *= 0.5;
    }

    f64::NAN
}

#[allow(clippy::too_many_arguments)]
fn simulate_basket_once_matrix(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    n_paths: usize,
    seed: u64,
    copula: BasketCopula,
) -> Result<(f64, f64), PricingError> {
    validate_inputs_matrix(basket, spots, vols, corr_matrix, dividends, n_paths)?;

    if basket.maturity <= 0.0 {
        let payoff = terminal_payoff(basket, spots, spots);
        return Ok((payoff, 0.0));
    }

    let n_assets = spots.len();
    let t = basket.maturity;
    let sqrt_t = t.sqrt();
    let discount = (-r * t).exp();
    let (corr_matrix, _was_repaired) = validate_or_repair_correlation_matrix(
        corr_matrix,
        n_assets,
        PsdProjectionConfig::default(),
    )
    .map_err(PricingError::InvalidInput)?;
    let chol = cholesky_lower_psd(&corr_matrix, 1.0e-12).ok_or_else(|| {
        PricingError::InvalidInput("basket correlation matrix is not PSD".to_string())
    })?;

    let drift = vols
        .iter()
        .zip(dividends.iter())
        .map(|(vol, q)| (r - *q - 0.5 * vol * vol) * t)
        .collect::<Vec<_>>();
    let vol_t = vols.iter().map(|v| v * sqrt_t).collect::<Vec<_>>();

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut indep = vec![0.0_f64; n_assets];
    let mut corr = vec![0.0_f64; n_assets];
    let mut uniforms = vec![0.0_f64; n_assets];
    let mut normals = vec![0.0_f64; n_assets];
    let mut terminal = vec![0.0_f64; n_assets];

    let mut stats = RunningStats::default();

    for _ in 0..n_paths {
        match copula {
            BasketCopula::Gaussian => {
                for z in &mut indep {
                    *z = sample_standard_normal(&mut rng);
                }
                correlate_normals(&chol, &indep, &mut corr);
                normals.copy_from_slice(&corr);
            }
            BasketCopula::StudentT { .. } => {
                sample_copula_uniforms_from_cholesky(
                    &chol,
                    copula.to_core(),
                    &mut rng,
                    &mut uniforms,
                )
                .map_err(PricingError::InvalidInput)?;
                copula_uniforms_to_normals(&uniforms, &mut normals)
                    .map_err(PricingError::InvalidInput)?;
            }
        }

        for i in 0..n_assets {
            terminal[i] = spots[i] * (drift[i] + vol_t[i] * normals[i]).exp();
        }

        stats.push(terminal_payoff(basket, spots, &terminal));
    }

    let mean = stats.mean();
    let variance = stats.sample_variance();

    Ok((
        discount * mean,
        discount * (variance / n_paths as f64).sqrt(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn simulate_basket_once_factor(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    factor_model: &FactorCorrelationModel,
    r: f64,
    dividends: &[f64],
    n_paths: usize,
    seed: u64,
    copula: BasketCopula,
) -> Result<(f64, f64), PricingError> {
    validate_inputs_factor(basket, spots, vols, factor_model, dividends, n_paths)?;

    if basket.maturity <= 0.0 {
        let payoff = terminal_payoff(basket, spots, spots);
        return Ok((payoff, 0.0));
    }

    let n_assets = spots.len();
    let t = basket.maturity;
    let sqrt_t = t.sqrt();
    let discount = (-r * t).exp();

    let drift = vols
        .iter()
        .zip(dividends.iter())
        .map(|(vol, q)| (r - *q - 0.5 * vol * vol) * t)
        .collect::<Vec<_>>();
    let vol_t = vols.iter().map(|v| v * sqrt_t).collect::<Vec<_>>();

    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut normals = vec![0.0_f64; n_assets];
    let mut uniforms = vec![0.0_f64; n_assets];
    let mut terminal = vec![0.0_f64; n_assets];

    let mut stats = RunningStats::default();

    for _ in 0..n_paths {
        match copula {
            BasketCopula::Gaussian => {
                factor_model
                    .sample_correlated_normals(&mut rng, &mut normals)
                    .map_err(PricingError::InvalidInput)?;
            }
            BasketCopula::StudentT { .. } => {
                sample_copula_uniforms_from_factor_model(
                    factor_model,
                    copula.to_core(),
                    &mut rng,
                    &mut uniforms,
                )
                .map_err(PricingError::InvalidInput)?;
                copula_uniforms_to_normals(&uniforms, &mut normals)
                    .map_err(PricingError::InvalidInput)?;
            }
        }

        for i in 0..n_assets {
            terminal[i] = spots[i] * (drift[i] + vol_t[i] * normals[i]).exp();
        }

        stats.push(terminal_payoff(basket, spots, &terminal));
    }

    let mean = stats.mean();
    let variance = stats.sample_variance();

    Ok((
        discount * mean,
        discount * (variance / n_paths as f64).sqrt(),
    ))
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

fn outperformance_payoff(option: &OutperformanceBasketOption, terminal: &[f64]) -> f64 {
    let lagger = option
        .lagger_weights
        .iter()
        .zip(terminal.iter())
        .map(|(w, s)| w * s)
        .sum::<f64>()
        .max(1.0e-12);
    let ratio = terminal[option.leader_index] / lagger;

    match option.option_type {
        OptionType::Call => (ratio - option.strike).max(0.0),
        OptionType::Put => (option.strike - ratio).max(0.0),
    }
}

fn validate_market_inputs(
    spots: &[f64],
    vols: &[f64],
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
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
    if spots.iter().any(|s| *s <= 0.0 || !s.is_finite()) {
        return Err(PricingError::InvalidInput(
            "basket spots must be finite and > 0".to_string(),
        ));
    }
    if vols.iter().any(|v| *v < 0.0 || !v.is_finite()) {
        return Err(PricingError::InvalidInput(
            "basket vols must be finite and >= 0".to_string(),
        ));
    }
    if dividends.iter().any(|q| !q.is_finite()) {
        return Err(PricingError::InvalidInput(
            "basket dividends must be finite".to_string(),
        ));
    }

    Ok(())
}

fn validate_inputs_matrix(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
    basket.validate()?;
    validate_market_inputs(spots, vols, dividends, n_paths)?;

    if matches!(basket.basket_type, BasketType::Average) && basket.weights.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "average basket weights length must match number of assets".to_string(),
        ));
    }

    validate_correlation_matrix(corr_matrix, spots.len()).map_err(PricingError::InvalidInput)
}

fn validate_inputs_factor(
    basket: &BasketOption,
    spots: &[f64],
    vols: &[f64],
    factor_model: &FactorCorrelationModel,
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
    basket.validate()?;
    validate_market_inputs(spots, vols, dividends, n_paths)?;
    factor_model
        .validate()
        .map_err(PricingError::InvalidInput)?;

    if factor_model.n_assets() != spots.len() {
        return Err(PricingError::InvalidInput(
            "factor-model asset count must match spots length".to_string(),
        ));
    }

    if matches!(basket.basket_type, BasketType::Average) && basket.weights.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "average basket weights length must match number of assets".to_string(),
        ));
    }

    Ok(())
}

fn validate_outperformance_inputs(
    option: &OutperformanceBasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
    option.validate()?;
    validate_market_inputs(spots, vols, dividends, n_paths)?;
    validate_correlation_matrix(corr_matrix, spots.len()).map_err(PricingError::InvalidInput)?;

    if option.lagger_weights.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "outperformance lagger weights length must match assets".to_string(),
        ));
    }
    if option.leader_index >= spots.len() {
        return Err(PricingError::InvalidInput(
            "outperformance leader_index out of bounds".to_string(),
        ));
    }

    Ok(())
}

fn validate_quanto_inputs(
    option: &QuantoBasketOption,
    spots: &[f64],
    vols: &[f64],
    corr_matrix: &[Vec<f64>],
    dividends: &[f64],
    n_paths: usize,
) -> Result<(), PricingError> {
    option.validate()?;
    validate_market_inputs(spots, vols, dividends, n_paths)?;
    validate_correlation_matrix(corr_matrix, spots.len()).map_err(PricingError::InvalidInput)?;

    if option.asset_fx_corr.len() != spots.len() {
        return Err(PricingError::InvalidInput(
            "quanto asset_fx_corr length must match assets".to_string(),
        ));
    }
    if matches!(option.basket.basket_type, BasketType::Average)
        && option.basket.weights.len() != spots.len()
    {
        return Err(PricingError::InvalidInput(
            "quanto average basket weights length must match assets".to_string(),
        ));
    }

    Ok(())
}

fn weighted_basket_moments(
    spots: &[f64],
    weights: &[f64],
    vols: &[f64],
    corr: &[Vec<f64>],
    r: f64,
    dividends: &[f64],
    t: f64,
) -> Result<(f64, f64, f64), PricingError> {
    if spots.len() != weights.len()
        || spots.len() != vols.len()
        || spots.len() != dividends.len()
        || corr.len() != spots.len()
    {
        return Err(PricingError::InvalidInput(
            "moment matching dimensions are inconsistent".to_string(),
        ));
    }

    let n = spots.len();

    let mut m1 = 0.0;
    for i in 0..n {
        let es = spots[i] * ((r - dividends[i]) * t).exp();
        m1 += weights[i] * es;
    }

    let mut m2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let e_sisj = spots[i]
                * spots[j]
                * ((2.0 * r - dividends[i] - dividends[j] + corr[i][j] * vols[i] * vols[j]) * t)
                    .exp();
            m2 += weights[i] * weights[j] * e_sisj;
        }
    }

    let log_spot = spots.iter().map(|s| s.ln()).collect::<Vec<_>>();
    let mu_ln = (0..n)
        .map(|i| log_spot[i] + (r - dividends[i] - 0.5 * vols[i] * vols[i]) * t)
        .collect::<Vec<_>>();
    let cov_ln = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| corr[i][j] * vols[i] * vols[j] * t)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut m3 = 0.0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut counts = vec![0_u32; n];
                counts[i] += 1;
                counts[j] += 1;
                counts[k] += 1;

                let mut mean_term = 0.0;
                for a in 0..n {
                    mean_term += counts[a] as f64 * mu_ln[a];
                }

                let mut cov_term = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        cov_term += counts[a] as f64 * counts[b] as f64 * cov_ln[a][b];
                    }
                }

                let e_triple = (mean_term + 0.5 * cov_term).exp();
                m3 += weights[i] * weights[j] * weights[k] * e_triple;
            }
        }
    }

    Ok((m1, m2, m3))
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
    for (i, row) in corr_matrix.iter().enumerate() {
        for rho in row.iter().skip(i + 1) {
            sum += *rho;
            count += 1;
        }
    }

    (count > 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::math::{
        CorrelationStressScenario, FactorCorrelationModel, is_positive_semidefinite,
        validate_or_repair_correlation_matrix,
    };
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
    fn worst_of_call_with_three_assets_prices_finite() {
        let basket = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::WorstOf,
        };

        let out = price_basket_mc(
            &basket,
            &[100.0, 98.0, 102.0],
            &[0.25, 0.22, 0.2],
            &[
                vec![1.0, 0.4, 0.2],
                vec![0.4, 1.0, 0.3],
                vec![0.2, 0.3, 1.0],
            ],
            0.01,
            &[0.0, 0.0, 0.0],
            120_000,
        );

        assert!(out.price.is_finite() && out.price >= 0.0);
    }

    #[test]
    fn best_of_call_with_three_assets_prices_finite() {
        let basket = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::BestOf,
        };

        let out = price_basket_mc(
            &basket,
            &[100.0, 98.0, 102.0],
            &[0.25, 0.22, 0.2],
            &[
                vec![1.0, 0.4, 0.2],
                vec![0.4, 1.0, 0.3],
                vec![0.2, 0.3, 1.0],
            ],
            0.01,
            &[0.0, 0.0, 0.0],
            120_000,
        );

        assert!(out.price.is_finite() && out.price >= 0.0);
    }

    #[test]
    fn five_asset_mc_matches_moment_matching_within_two_percent() {
        let basket = BasketOption {
            weights: vec![0.25, 0.2, 0.2, 0.2, 0.15],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        };
        let spots = vec![100.0, 98.0, 102.0, 95.0, 105.0];
        let vols = vec![0.2, 0.21, 0.19, 0.23, 0.18];
        let dividends = vec![0.01; 5];
        let corr = vec![
            vec![1.0, 0.35, 0.30, 0.20, 0.15],
            vec![0.35, 1.0, 0.25, 0.30, 0.10],
            vec![0.30, 0.25, 1.0, 0.28, 0.22],
            vec![0.20, 0.30, 0.28, 1.0, 0.26],
            vec![0.15, 0.10, 0.22, 0.26, 1.0],
        ];

        let mc = price_basket_mc(&basket, &spots, &vols, &corr, 0.02, &dividends, 250_000).price;
        let mm = price_basket_moment_matching(
            &basket,
            &spots,
            &vols,
            &corr,
            0.02,
            &dividends,
            BasketMomentMatchingMethod::Gentle,
        )
        .unwrap();

        let rel = ((mc - mm) / mm.max(1.0e-12)).abs();
        assert!(
            rel <= 0.02,
            "mc={} mm={} rel_err={} exceeds 2%",
            mc,
            mm,
            rel
        );
    }

    #[test]
    fn psd_repair_validation_produces_valid_matrix() {
        let bad = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, -0.95],
            vec![0.95, -0.95, 1.0],
        ];

        let (repaired, was_repaired) =
            validate_or_repair_correlation_matrix(&bad, 3, PsdProjectionConfig::default()).unwrap();
        assert!(was_repaired);
        assert!(is_positive_semidefinite(&repaired, 1.0e-8));
    }

    #[test]
    fn stressed_correlation_is_repaired_to_psd() {
        let base = vec![
            vec![1.0, 0.7, 0.6],
            vec![0.7, 1.0, 0.65],
            vec![0.6, 0.65, 1.0],
        ];

        let stressed = stressed_correlation_matrix(
            &base,
            &[CorrelationStressScenario::AdditiveShift { shift: 0.35 }],
        )
        .unwrap();

        assert!(is_positive_semidefinite(&stressed, 1.0e-8));
    }

    #[test]
    fn factor_model_mc_prices_finite() {
        let basket = BasketOption {
            weights: vec![0.3, 0.25, 0.2, 0.15, 0.1],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        };
        let model = FactorCorrelationModel::MultiFactor {
            loadings: vec![
                vec![0.6, 0.1],
                vec![0.5, -0.1],
                vec![0.4, 0.2],
                vec![0.35, -0.2],
                vec![0.25, 0.15],
            ],
        };

        let out = price_basket_mc_with_factor_model(
            &basket,
            &[100.0, 101.0, 99.0, 97.0, 103.0],
            &[0.2, 0.22, 0.18, 0.21, 0.19],
            &model,
            0.02,
            &[0.0; 5],
            120_000,
            BasketCopula::Gaussian,
        );

        assert!(out.price.is_finite() && out.price >= 0.0);
    }

    #[test]
    fn ten_asset_100k_paths_under_five_seconds() {
        let basket = BasketOption {
            weights: vec![0.1; 10],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        };
        let spots = vec![100.0; 10];
        let vols = vec![0.2; 10];
        let dividends = vec![0.0; 10];
        let mut corr = vec![vec![0.0; 10]; 10];
        let mut i = 0usize;
        while i < 10 {
            corr[i][i] = 1.0;
            let mut j = i + 1;
            while j < 10 {
                corr[i][j] = 0.35;
                corr[j][i] = 0.35;
                j += 1;
            }
            i += 1;
        }

        let start = Instant::now();
        let out = price_basket_mc(&basket, &spots, &vols, &corr, 0.02, &dividends, 100_000);
        let elapsed = start.elapsed().as_secs_f64();

        assert!(out.price.is_finite());
        assert!(
            elapsed < 5.0,
            "10-asset 100K basket MC runtime {}s exceeds 5s",
            elapsed
        );
    }

    #[test]
    fn quanto_and_outperformance_are_priced() {
        let outperf = OutperformanceBasketOption {
            leader_index: 0,
            lagger_weights: vec![0.0, 0.5, 0.5],
            strike: 1.0,
            maturity: 1.0,
            option_type: OptionType::Call,
        };
        let quanto = QuantoBasketOption {
            basket: BasketOption {
                weights: vec![0.6, 0.4],
                strike: 100.0,
                maturity: 1.0,
                is_call: true,
                basket_type: BasketType::Average,
            },
            fx_rate: 1.2,
            fx_vol: 0.12,
            asset_fx_corr: vec![0.3, 0.25],
            domestic_rate: 0.03,
            foreign_rate: 0.02,
        };

        let corr3 = vec![
            vec![1.0, 0.4, 0.35],
            vec![0.4, 1.0, 0.3],
            vec![0.35, 0.3, 1.0],
        ];
        let corr2 = vec![vec![1.0, 0.4], vec![0.4, 1.0]];

        let out_outperf = price_outperformance_basket_mc(
            &outperf,
            &[100.0, 95.0, 105.0],
            &[0.2, 0.22, 0.21],
            &corr3,
            0.02,
            &[0.0, 0.0, 0.0],
            120_000,
        );
        let out_quanto = price_quanto_basket_mc(
            &quanto,
            &[100.0, 98.0],
            &[0.2, 0.22],
            &corr2,
            &[0.01, 0.01],
            120_000,
        );

        assert!(out_outperf.price.is_finite() && out_outperf.price >= 0.0);
        assert!(out_quanto.price.is_finite() && out_quanto.price >= 0.0);
    }
}
