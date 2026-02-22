//! Rough Bergomi Monte Carlo engine with fBm path generation and implied-vol surface extraction.
//!
//! The implementation supports two fractional Brownian motion schemes:
//! full Cholesky and a hybrid coarse-grid/interpolation method selected automatically by step count.
//! `rbergomi_european_mc` prices calls with antithetic sampling and reports stderr/diagnostics,
//! while `rbergomi_implied_vol_surface` reuses terminal samples across strikes per expiry.
//! References: Bayer, Friz, and Gatheral (2016); Bennedsen, Lunde, and Pakkanen (2017).
//! Numerical controls include input-domain validation (`H in (0,1)`, `rho in (-1,1)`),
//! positivity floors on variance/spot, and covariance-PSD checks in Cholesky setup.
//! Use this module for rough-volatility smile dynamics; use Heston/SLV modules for smoother alternatives.

use crate::core::{Diagnostics, PricingResult};
use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};
use crate::pricing::OptionType;
use crate::vol::implied::implied_vol;

const DEFAULT_SEED: u64 = 12_345;
const FBM_HYBRID_SWITCH_STEPS: usize = 128;
const SURFACE_PATHS: usize = 12_000;
const SURFACE_STEPS_PER_YEAR: f64 = 128.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbmScheme {
    Cholesky,
    Hybrid,
}

#[derive(Debug, Clone)]
enum FbmGenerator {
    Cholesky(CholeskyFbmGenerator),
    Hybrid(HybridFbmGenerator),
}

impl FbmGenerator {
    fn auto(hurst: f64, maturity: f64, n_steps: usize) -> Result<Self, String> {
        if n_steps <= FBM_HYBRID_SWITCH_STEPS {
            return Ok(Self::Cholesky(CholeskyFbmGenerator::new(
                hurst, maturity, n_steps,
            )?));
        }

        Ok(Self::Hybrid(HybridFbmGenerator::new(
            hurst, maturity, n_steps,
        )?))
    }

    #[cfg(test)]
    fn with_scheme(
        hurst: f64,
        maturity: f64,
        n_steps: usize,
        scheme: FbmScheme,
    ) -> Result<Self, String> {
        match scheme {
            FbmScheme::Cholesky => Ok(Self::Cholesky(CholeskyFbmGenerator::new(
                hurst, maturity, n_steps,
            )?)),
            FbmScheme::Hybrid => Ok(Self::Hybrid(HybridFbmGenerator::new(
                hurst, maturity, n_steps,
            )?)),
        }
    }

    fn n_steps(&self) -> usize {
        match self {
            Self::Cholesky(g) => g.n_steps,
            Self::Hybrid(g) => g.n_steps,
        }
    }

    fn sample_path(&self, backbone_normals: &[f64], correction_normals: &[f64], out: &mut [f64]) {
        match self {
            Self::Cholesky(g) => g.sample_path(backbone_normals, out),
            Self::Hybrid(g) => g.sample_path(backbone_normals, correction_normals, out),
        }
    }
}

#[derive(Debug, Clone)]
struct CholeskyFbmGenerator {
    n_steps: usize,
    lower: Vec<Vec<f64>>,
}

impl CholeskyFbmGenerator {
    fn new(hurst: f64, maturity: f64, n_steps: usize) -> Result<Self, String> {
        validate_fbm_inputs(hurst, maturity, n_steps)?;

        let dt = maturity / n_steps as f64;
        let times = (1..=n_steps).map(|i| i as f64 * dt).collect::<Vec<_>>();
        let cov = covariance_matrix(&times, hurst);
        let lower = cholesky_lower(&cov)?;

        Ok(Self { n_steps, lower })
    }

    fn sample_path(&self, normals: &[f64], out: &mut [f64]) {
        assert_eq!(normals.len(), self.n_steps);
        assert_eq!(out.len(), self.n_steps + 1);

        out[0] = 0.0;
        for i in 0..self.n_steps {
            let mut v = 0.0;
            for (j, lij) in self.lower[i].iter().enumerate().take(i + 1) {
                v += *lij * normals[j];
            }
            out[i + 1] = v;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HybridNode {
    left_idx: usize,
    right_idx: usize,
    alpha: f64,
    residual_std: f64,
}

#[derive(Debug, Clone)]
struct HybridFbmGenerator {
    n_steps: usize,
    coarse_steps: usize,
    coarse_lower: Vec<Vec<f64>>,
    nodes: Vec<HybridNode>,
}

impl HybridFbmGenerator {
    fn new(hurst: f64, maturity: f64, n_steps: usize) -> Result<Self, String> {
        validate_fbm_inputs(hurst, maturity, n_steps)?;

        let coarse_steps = choose_hybrid_coarse_steps(n_steps);
        let coarse_dt = maturity / coarse_steps as f64;
        let coarse_times = (1..=coarse_steps)
            .map(|i| i as f64 * coarse_dt)
            .collect::<Vec<_>>();
        let coarse_cov = covariance_matrix(&coarse_times, hurst);
        let coarse_lower = cholesky_lower(&coarse_cov)?;

        let fine_dt = maturity / n_steps as f64;
        let mut nodes = Vec::with_capacity(n_steps + 1);
        nodes.push(HybridNode {
            left_idx: 0,
            right_idx: 0,
            alpha: 0.0,
            residual_std: 0.0,
        });

        for i in 1..=n_steps {
            let t = i as f64 * fine_dt;
            let u = (t / coarse_dt).clamp(0.0, coarse_steps as f64);
            let left = u.floor() as usize;

            if left >= coarse_steps {
                nodes.push(HybridNode {
                    left_idx: coarse_steps,
                    right_idx: coarse_steps,
                    alpha: 0.0,
                    residual_std: 0.0,
                });
                continue;
            }

            let alpha = (u - left as f64).clamp(0.0, 1.0);
            if alpha <= 1.0e-12 {
                nodes.push(HybridNode {
                    left_idx: left,
                    right_idx: left,
                    alpha: 0.0,
                    residual_std: 0.0,
                });
                continue;
            }

            let right = left + 1;
            let a = 1.0 - alpha;
            let b = alpha;

            let tl = left as f64 * coarse_dt;
            let tr = right as f64 * coarse_dt;
            let var_interp = a * a * tl.powf(2.0 * hurst)
                + b * b * tr.powf(2.0 * hurst)
                + 2.0 * a * b * fbm_covariance(tl, tr, hurst);

            let target_var = t.powf(2.0 * hurst);
            let residual_std = (target_var - var_interp).max(0.0).sqrt();

            nodes.push(HybridNode {
                left_idx: left,
                right_idx: right,
                alpha,
                residual_std,
            });
        }

        Ok(Self {
            n_steps,
            coarse_steps,
            coarse_lower,
            nodes,
        })
    }

    fn sample_path(&self, backbone_normals: &[f64], correction_normals: &[f64], out: &mut [f64]) {
        assert_eq!(backbone_normals.len(), self.n_steps);
        assert_eq!(correction_normals.len(), self.n_steps);
        assert_eq!(out.len(), self.n_steps + 1);

        let mut coarse_path = vec![0.0_f64; self.coarse_steps + 1];
        for i in 0..self.coarse_steps {
            let mut v = 0.0;
            for (j, lij) in self.coarse_lower[i].iter().enumerate().take(i + 1) {
                v += *lij * backbone_normals[j];
            }
            coarse_path[i + 1] = v;
        }

        out[0] = 0.0;
        for i in 1..=self.n_steps {
            let node = self.nodes[i];
            if node.left_idx == node.right_idx {
                out[i] = coarse_path[node.left_idx];
                continue;
            }

            let coarse_interp = (1.0 - node.alpha) * coarse_path[node.left_idx]
                + node.alpha * coarse_path[node.right_idx];
            out[i] = coarse_interp + node.residual_std * correction_normals[i - 1];
        }
    }
}

fn validate_fbm_inputs(hurst: f64, maturity: f64, n_steps: usize) -> Result<(), String> {
    if !hurst.is_finite() || hurst <= 0.0 || hurst >= 1.0 {
        return Err("hurst must be finite and in (0, 1)".to_string());
    }
    if !maturity.is_finite() || maturity <= 0.0 {
        return Err("maturity must be finite and > 0".to_string());
    }
    if n_steps == 0 {
        return Err("n_steps must be > 0".to_string());
    }
    Ok(())
}

fn choose_hybrid_coarse_steps(n_steps: usize) -> usize {
    let coarse = (n_steps as f64).sqrt().round() as usize;
    coarse.clamp(8, 64).min(n_steps)
}

fn covariance_matrix(times: &[f64], hurst: f64) -> Vec<Vec<f64>> {
    let n = times.len();
    let mut cov = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let c = fbm_covariance(times[i], times[j], hurst);
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }

    cov
}

fn cholesky_lower(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err("matrix must be square and non-empty".to_string());
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
                    return Err("covariance matrix is not positive semidefinite".to_string());
                }
                l[i][j] = sum.max(tol).sqrt();
            } else if l[j][j] > tol {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

pub fn fbm_covariance(s: f64, t: f64, hurst: f64) -> f64 {
    let h2 = 2.0 * hurst;
    0.5 * (s.powf(h2) + t.powf(h2) - (t - s).abs().powf(h2))
}

pub fn fbm_path_cholesky(
    hurst: f64,
    maturity: f64,
    n_steps: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let generator = CholeskyFbmGenerator::new(hurst, maturity, n_steps)?;
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut z_backbone = vec![0.0_f64; n_steps];
    let mut path = vec![0.0_f64; n_steps + 1];

    for zi in &mut z_backbone {
        *zi = sample_standard_normal(&mut rng);
    }

    generator.sample_path(&z_backbone, &mut path);
    Ok(path)
}

pub fn fbm_path_hybrid(
    hurst: f64,
    maturity: f64,
    n_steps: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let generator = HybridFbmGenerator::new(hurst, maturity, n_steps)?;
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut z_backbone = vec![0.0_f64; n_steps];
    let mut z_correction = vec![0.0_f64; n_steps];
    let mut path = vec![0.0_f64; n_steps + 1];

    for zi in &mut z_backbone {
        *zi = sample_standard_normal(&mut rng);
    }
    for zi in &mut z_correction {
        *zi = sample_standard_normal(&mut rng);
    }

    generator.sample_path(&z_backbone, &z_correction, &mut path);
    Ok(path)
}

#[cfg(test)]
fn sample_fbm_path(
    hurst: f64,
    maturity: f64,
    n_steps: usize,
    seed: u64,
    scheme: FbmScheme,
) -> Result<Vec<f64>, String> {
    let generator = FbmGenerator::with_scheme(hurst, maturity, n_steps, scheme)?;
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut z_backbone = vec![0.0_f64; n_steps];
    let mut z_correction = vec![0.0_f64; n_steps];
    let mut path = vec![0.0_f64; n_steps + 1];

    for zi in &mut z_backbone {
        *zi = sample_standard_normal(&mut rng);
    }
    for zi in &mut z_correction {
        *zi = sample_standard_normal(&mut rng);
    }

    generator.sample_path(&z_backbone, &z_correction, &mut path);
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
fn simulate_terminal_from_normals(
    spot: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    fbm_generator: &FbmGenerator,
    backbone_normals: &[f64],
    correction_normals: &[f64],
    asset_perp_normals: &[f64],
    fbm_path: &mut [f64],
) -> f64 {
    let n_steps = fbm_generator.n_steps();
    let dt = maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_perp = (1.0 - rho * rho).max(0.0).sqrt();

    fbm_generator.sample_path(backbone_normals, correction_normals, fbm_path);

    let mut s = spot;
    for i in 0..n_steps {
        let t = i as f64 * dt;
        let variance = xi0 * (eta * fbm_path[i] - 0.5 * eta * eta * t.powf(2.0 * hurst)).exp();
        let zs = rho * backbone_normals[i] + rho_perp * asset_perp_normals[i];

        s *= ((r - q - 0.5 * variance) * dt + variance.max(0.0).sqrt() * sqrt_dt * zs).exp();
        s = s.max(1.0e-12);
    }

    s
}

#[allow(clippy::too_many_arguments)]
fn simulate_terminal_spots(
    spot: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let fbm_generator = FbmGenerator::auto(hurst, maturity, n_steps)?;

    let mut terminals = Vec::with_capacity(n_paths);
    let mut z_backbone = vec![0.0_f64; n_steps];
    let mut z_backbone_anti = vec![0.0_f64; n_steps];
    let mut z_correction = vec![0.0_f64; n_steps];
    let mut z_correction_anti = vec![0.0_f64; n_steps];
    let mut z_perp = vec![0.0_f64; n_steps];
    let mut z_perp_anti = vec![0.0_f64; n_steps];
    let mut fbm_path = vec![0.0_f64; n_steps + 1];
    let mut fbm_path_anti = vec![0.0_f64; n_steps + 1];

    for pair in 0..(n_paths / 2) {
        let mut rng = FastRng::from_seed(
            FastRngKind::Xoshiro256PlusPlus,
            resolve_stream_seed(seed, pair, true),
        );

        for i in 0..n_steps {
            z_backbone[i] = sample_standard_normal(&mut rng);
            z_correction[i] = sample_standard_normal(&mut rng);
            z_perp[i] = sample_standard_normal(&mut rng);

            z_backbone_anti[i] = -z_backbone[i];
            z_correction_anti[i] = -z_correction[i];
            z_perp_anti[i] = -z_perp[i];
        }

        let s_t = simulate_terminal_from_normals(
            spot,
            r,
            q,
            maturity,
            hurst,
            eta,
            rho,
            xi0,
            &fbm_generator,
            &z_backbone,
            &z_correction,
            &z_perp,
            &mut fbm_path,
        );
        let s_t_anti = simulate_terminal_from_normals(
            spot,
            r,
            q,
            maturity,
            hurst,
            eta,
            rho,
            xi0,
            &fbm_generator,
            &z_backbone_anti,
            &z_correction_anti,
            &z_perp_anti,
            &mut fbm_path_anti,
        );

        terminals.push(s_t);
        terminals.push(s_t_anti);
    }

    if n_paths % 2 == 1 {
        let mut rng = FastRng::from_seed(
            FastRngKind::Xoshiro256PlusPlus,
            resolve_stream_seed(seed, n_paths / 2, true),
        );
        for i in 0..n_steps {
            z_backbone[i] = sample_standard_normal(&mut rng);
            z_correction[i] = sample_standard_normal(&mut rng);
            z_perp[i] = sample_standard_normal(&mut rng);
        }

        terminals.push(simulate_terminal_from_normals(
            spot,
            r,
            q,
            maturity,
            hurst,
            eta,
            rho,
            xi0,
            &fbm_generator,
            &z_backbone,
            &z_correction,
            &z_perp,
            &mut fbm_path,
        ));
    }

    Ok(terminals)
}

#[allow(clippy::too_many_arguments)]
fn validate_rbergomi_inputs(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    n_paths: usize,
    n_steps: usize,
) -> Result<(), String> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err("spot must be finite and > 0".to_string());
    }
    if !strike.is_finite() || strike <= 0.0 {
        return Err("strike must be finite and > 0".to_string());
    }
    if !r.is_finite() || !q.is_finite() {
        return Err("rates must be finite".to_string());
    }
    if !maturity.is_finite() || maturity < 0.0 {
        return Err("maturity must be finite and >= 0".to_string());
    }
    if !hurst.is_finite() || hurst <= 0.0 || hurst >= 1.0 {
        return Err("hurst must be finite and in (0, 1)".to_string());
    }
    if !eta.is_finite() || eta < 0.0 {
        return Err("eta must be finite and >= 0".to_string());
    }
    if !rho.is_finite() || rho <= -1.0 || rho >= 1.0 {
        return Err("rho must be finite and in (-1, 1)".to_string());
    }
    if !xi0.is_finite() || xi0 <= 0.0 {
        return Err("xi0 must be finite and > 0".to_string());
    }
    if n_paths == 0 || n_steps == 0 {
        return Err("n_paths and n_steps must be > 0".to_string());
    }

    Ok(())
}

fn discounted_call_payoff_mean_and_stderr(
    terminals: &[f64],
    strike: f64,
    r: f64,
    maturity: f64,
) -> (f64, f64) {
    let discount = (-r * maturity).exp();
    let n = terminals.len() as f64;

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for s_t in terminals {
        let payoff = (*s_t - strike).max(0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let mean = sum / n;
    let variance = if terminals.len() > 1 {
        (sum_sq - n * mean * mean).max(0.0) / (n - 1.0)
    } else {
        0.0
    };

    (discount * mean, discount * (variance / n).sqrt())
}

fn implied_vol_with_dividend(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    maturity: f64,
    price: f64,
) -> Option<f64> {
    if maturity <= 0.0 {
        return Some(0.0);
    }

    let dividend_adjusted_spot = spot * (-q * maturity).exp();
    implied_vol(
        OptionType::Call,
        dividend_adjusted_spot,
        strike,
        r,
        maturity,
        price,
        1.0e-10,
        128,
    )
    .ok()
    .filter(|v| v.is_finite() && *v >= 0.0)
}

#[allow(clippy::too_many_arguments)]
pub fn rbergomi_european_mc(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    if validate_rbergomi_inputs(
        spot, strike, r, q, maturity, hurst, eta, rho, xi0, n_paths, n_steps,
    )
    .is_err()
    {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: Diagnostics::new(),
        };
    }

    if maturity == 0.0 {
        let intrinsic = (spot - strike).max(0.0);
        let mut diagnostics = Diagnostics::new();
        diagnostics.insert("num_paths", n_paths as f64);
        diagnostics.insert("num_steps", n_steps as f64);
        diagnostics.insert("vol", xi0.sqrt());
        diagnostics.insert("effective_vol", 0.0);

        return PricingResult {
            price: intrinsic,
            stderr: Some(0.0),
            greeks: None,
            diagnostics,
        };
    }

    let terminals = match simulate_terminal_spots(
        spot,
        r,
        q,
        maturity,
        hurst,
        eta,
        rho,
        xi0,
        n_paths,
        n_steps,
        DEFAULT_SEED,
    ) {
        Ok(v) => v,
        Err(_) => {
            return PricingResult {
                price: f64::NAN,
                stderr: None,
                greeks: None,
                diagnostics: Diagnostics::new(),
            };
        }
    };

    let (price, stderr) = discounted_call_payoff_mean_and_stderr(&terminals, strike, r, maturity);
    let implied = implied_vol_with_dividend(spot, strike, r, q, maturity, price);

    let mut diagnostics = Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", n_steps as f64);
    diagnostics.insert("vol", xi0.sqrt());
    diagnostics.insert("var_of_var", eta);
    if let Some(iv) = implied {
        diagnostics.insert("effective_vol", iv);
    }

    PricingResult {
        price,
        stderr: Some(stderr),
        greeks: None,
        diagnostics,
    }
}

pub fn rbergomi_implied_vol_surface(
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    expiries: &[f64],
    strikes: &[f64],
) -> Vec<Vec<f64>> {
    let spot = 1.0;
    let r = 0.0;
    let q = 0.0;

    let mut surface = Vec::with_capacity(expiries.len());

    for (expiry_idx, &expiry) in expiries.iter().enumerate() {
        if expiry <= 0.0 || strikes.is_empty() {
            surface.push(vec![f64::NAN; strikes.len()]);
            continue;
        }

        let n_steps = ((SURFACE_STEPS_PER_YEAR * expiry).ceil() as usize).clamp(16, 512);
        let terminals = match simulate_terminal_spots(
            spot,
            r,
            q,
            expiry,
            hurst,
            eta,
            rho,
            xi0,
            SURFACE_PATHS,
            n_steps,
            resolve_stream_seed(DEFAULT_SEED, expiry_idx, true),
        ) {
            Ok(v) => v,
            Err(_) => {
                surface.push(vec![f64::NAN; strikes.len()]);
                continue;
            }
        };

        let mut row = Vec::with_capacity(strikes.len());
        for &strike in strikes {
            if strike <= 0.0 || !strike.is_finite() {
                row.push(f64::NAN);
                continue;
            }

            let (price, _stderr) =
                discounted_call_payoff_mean_and_stderr(&terminals, strike, r, expiry);
            let iv =
                implied_vol_with_dividend(spot, strike, r, q, expiry, price).unwrap_or(f64::NAN);
            row.push(iv);
        }
        surface.push(row);
    }

    surface
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::european::black_scholes_price;

    fn sample_fbm_moments(hurst: f64, maturity: f64, n_steps: usize, n_paths: usize) -> (f64, f64) {
        let generator = CholeskyFbmGenerator::new(hurst, maturity, n_steps).unwrap();
        let mut path = vec![0.0_f64; n_steps + 1];
        let mut z = vec![0.0_f64; n_steps];

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for i in 0..n_paths {
            let mut rng = FastRng::from_seed(
                FastRngKind::Xoshiro256PlusPlus,
                resolve_stream_seed(77, i, true),
            );
            for zi in &mut z {
                *zi = sample_standard_normal(&mut rng);
            }
            generator.sample_path(&z, &mut path);
            let x = path[n_steps];
            sum += x;
            sum_sq += x * x;
        }

        let n = n_paths as f64;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        (mean, var)
    }

    #[test]
    fn fbm_paths_have_correct_variance() {
        let hurst = 0.1;
        let maturity = 1.0;
        let (_mean, sample_var) = sample_fbm_moments(hurst, maturity, 24, 3_000);
        let expected_var = maturity.powf(2.0 * hurst);

        assert!((sample_var - expected_var).abs() < 8.0e-2);
    }

    #[test]
    fn fbm_increments_have_correct_covariance_structure() {
        let hurst = 0.2;
        let maturity = 1.0;
        let n_steps = 64;
        let n_paths = 5_000;

        let generator = CholeskyFbmGenerator::new(hurst, maturity, n_steps).unwrap();
        let mut path = vec![0.0_f64; n_steps + 1];
        let mut z = vec![0.0_f64; n_steps];

        let mut inc1 = Vec::with_capacity(n_paths);
        let mut inc2 = Vec::with_capacity(n_paths);

        for i in 0..n_paths {
            let mut rng = FastRng::from_seed(
                FastRngKind::Xoshiro256PlusPlus,
                resolve_stream_seed(81, i, true),
            );
            for zi in &mut z {
                *zi = sample_standard_normal(&mut rng);
            }
            generator.sample_path(&z, &mut path);
            inc1.push(path[1] - path[0]);
            inc2.push(path[2] - path[1]);
        }

        let n = n_paths as f64;
        let m1 = inc1.iter().sum::<f64>() / n;
        let m2 = inc2.iter().sum::<f64>() / n;
        let cov = inc1
            .iter()
            .zip(inc2.iter())
            .map(|(a, b)| (a - m1) * (b - m2))
            .sum::<f64>()
            / (n - 1.0);

        let dt = maturity / n_steps as f64;
        let expected = 0.5 * dt.powf(2.0 * hurst) * (2.0_f64.powf(2.0 * hurst) - 2.0);

        assert!((cov - expected).abs() <= expected.abs() * 0.35 + 2.0e-4);
    }

    #[test]
    fn rbergomi_h_half_and_zero_eta_matches_lognormal() {
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.01;
        let q = 0.0;
        let maturity = 1.0;
        let hurst = 0.5;
        let eta = 0.0;
        let rho = -0.8;
        let xi0 = 0.04;

        let mc = rbergomi_european_mc(
            spot, strike, r, q, maturity, hurst, eta, rho, xi0, 25_000, 64,
        );
        let bs = black_scholes_price(
            OptionType::Call,
            spot * (-q * maturity).exp(),
            strike,
            r,
            xi0.sqrt(),
            maturity,
        );

        let stderr = mc.stderr.unwrap_or(0.0);
        assert!((mc.price - bs).abs() <= 4.0 * stderr + 2.5e-1);
    }

    #[test]
    fn rbergomi_call_price_is_positive_and_bounded_by_spot() {
        let spot = 100.0;
        let strike = 110.0;
        let maturity = 0.75;

        let res = rbergomi_european_mc(
            spot, strike, 0.01, 0.0, maturity, 0.1, 1.9, -0.85, 0.04, 15_000, 64,
        );

        let stderr = res.stderr.unwrap_or(0.0);
        assert!(res.price >= 0.0);
        assert!(res.price <= spot + 4.0 * stderr + 1.0e-8);
    }

    #[test]
    fn lower_hurst_produces_steeper_short_term_skew() {
        let spot = 100.0;
        let r = 0.0;
        let q = 0.0;
        let maturity = 0.20;
        let eta = 2.0;
        let rho = -0.9;
        let xi0 = 0.04;

        let iv_low_h_85 =
            rbergomi_european_mc(spot, 85.0, r, q, maturity, 0.08, eta, rho, xi0, 18_000, 64)
                .diagnostics
                .get("effective_vol")
                .copied()
                .unwrap_or(f64::NAN);
        let iv_low_h_115 =
            rbergomi_european_mc(spot, 115.0, r, q, maturity, 0.08, eta, rho, xi0, 18_000, 64)
                .diagnostics
                .get("effective_vol")
                .copied()
                .unwrap_or(f64::NAN);

        let iv_high_h_85 =
            rbergomi_european_mc(spot, 85.0, r, q, maturity, 0.45, eta, rho, xi0, 18_000, 64)
                .diagnostics
                .get("effective_vol")
                .copied()
                .unwrap_or(f64::NAN);
        let iv_high_h_115 =
            rbergomi_european_mc(spot, 115.0, r, q, maturity, 0.45, eta, rho, xi0, 18_000, 64)
                .diagnostics
                .get("effective_vol")
                .copied()
                .unwrap_or(f64::NAN);

        let skew_low = iv_low_h_85 - iv_low_h_115;
        let skew_high = iv_high_h_85 - iv_high_h_115;

        assert!(
            skew_low > skew_high - 2.0e-2,
            "skew_low={} skew_high={}",
            skew_low,
            skew_high
        );
    }

    #[test]
    fn short_dated_atm_iv_matches_forward_variance_level() {
        let xi0 = 0.04;
        let res = rbergomi_european_mc(
            100.0,
            100.0,
            0.0,
            0.0,
            1.0 / 52.0,
            0.10,
            0.0,
            -0.8,
            xi0,
            30_000,
            48,
        );

        let iv = res
            .diagnostics
            .get("effective_vol")
            .copied()
            .unwrap_or(f64::NAN);

        assert!(
            (iv - xi0.sqrt()).abs() <= 3.0e-2,
            "iv={} target={}",
            iv,
            xi0.sqrt()
        );
    }

    #[test]
    fn explicit_scheme_sampling_works() {
        let a = sample_fbm_path(0.1, 1.0, 48, 42, FbmScheme::Cholesky).unwrap();
        let b = sample_fbm_path(0.1, 1.0, 256, 42, FbmScheme::Hybrid).unwrap();

        assert_eq!(a.len(), 49);
        assert_eq!(b.len(), 257);
    }
}
