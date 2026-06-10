//! Module `models::rough_bergomi`.
//!
//! Implements rough bergomi workflows with concrete routines such as `fbm_covariance`, `fbm_path_cholesky`, `fbm_path_hybrid`, `rbergomi_european_mc`.
//!
//! References: Bayer, Friz, Gatheral (2016), McCrickerd and Pakkanen (2018), rough-vol covariance formulas around Eq. (2.5).
//!
//! The variance process is driven by the Riemann-Liouville Volterra process
//! `W~_t = sqrt(2H) \int_0^t (t-s)^{H-1/2} dW_s` with exact covariance
//! `E[W~_s W~_t] = 2H \int_0^{min(s,t)} (s-u)^{H-1/2} (t-u)^{H-1/2} du`
//! (so `E[W~_t^2] = t^{2H}` on the diagonal), evaluated by Gauss-Legendre
//! quadrature with an endpoint-singularity substitution. Pricing simulates the
//! joint Gaussian vector of Brownian increments `dW` and `W~` at the grid times
//! with the exact cross-covariance
//! `E[W~_t (W_u - W_v)] = sqrt(2H)/(H+1/2) * [(t-v)^{H+1/2} - (t-u)^{H+1/2}]`
//! for `v < u <= t`, so the asset shock correlates with the Brownian driver of
//! the Volterra integral (not with Cholesky innovations of the covariance).
//!
//! Key types and purpose: `FbmScheme` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
use std::sync::OnceLock;

use crate::core::{Diagnostics, PricingResult};
use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};
use crate::pricing::OptionType;
use crate::vol::implied::implied_vol;

const DEFAULT_SEED: u64 = 12_345;
const SURFACE_PATHS: usize = 12_000;
const SURFACE_STEPS_PER_YEAR: f64 = 128.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbmScheme {
    Cholesky,
    Hybrid,
}

/// Exact Cholesky sampler of the Volterra process at the grid times.
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

/// Approximate standalone Volterra path sampler: exact Cholesky on a coarse
/// grid, interpolated to the fine grid with a variance-matching residual.
///
/// Only used by [`fbm_path_hybrid`]; pricing uses the exact joint generator so
/// that the spot-vol correlation is not diluted by the residual noise.
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
            let var_interp = a * a * fbm_covariance(tl, tl, hurst)
                + b * b * fbm_covariance(tr, tr, hurst)
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

/// Exact joint Gaussian sampler of `[dW_1, ..., dW_n, W~_{t_1}, ..., W~_{t_n}]`
/// where `dW_i = W_{t_i} - W_{t_{i-1}}` are the increments of the Brownian
/// motion driving the Volterra integral. Built from the exact auto- and
/// cross-covariances, so the asset diffusion can be correlated with the true
/// Brownian driver of the variance process.
#[derive(Debug, Clone)]
struct JointVolterraGenerator {
    n_steps: usize,
    /// Cholesky factor with the (well-conditioned) `dW` block leading.
    lower: Vec<Vec<f64>>,
}

impl JointVolterraGenerator {
    fn new(hurst: f64, maturity: f64, n_steps: usize) -> Result<Self, String> {
        validate_fbm_inputs(hurst, maturity, n_steps)?;

        let n = n_steps;
        let dt = maturity / n as f64;
        let dim = 2 * n;
        let mut cov = vec![vec![0.0_f64; dim]; dim];

        // dW block: independent increments with variance dt.
        for (i, row) in cov.iter_mut().enumerate().take(n) {
            row[i] = dt;
        }

        // Cross block: E[W~_{t_i} dW_j] for the aligned grid (zero for j > i).
        for i in 1..=n {
            let t_i = i as f64 * dt;
            for j in 1..=i {
                let c = volterra_bm_cross_cov(t_i, (j - 1) as f64 * dt, j as f64 * dt, hurst);
                cov[n + i - 1][j - 1] = c;
                cov[j - 1][n + i - 1] = c;
            }
        }

        // W~ block: exact Volterra covariance.
        for i in 1..=n {
            let t_i = i as f64 * dt;
            for j in 1..=i {
                let t_j = j as f64 * dt;
                let c = fbm_covariance(t_j, t_i, hurst);
                cov[n + i - 1][n + j - 1] = c;
                cov[n + j - 1][n + i - 1] = c;
            }
        }

        let lower = cholesky_lower(&cov)?;
        Ok(Self { n_steps: n, lower })
    }

    /// Maps `2 * n_steps` iid standard normals to `(dw, wtilde)`, with
    /// `wtilde[0] = 0` and `wtilde[i]` the Volterra process at `t_i`.
    fn sample(&self, normals: &[f64], dw: &mut [f64], wtilde: &mut [f64]) {
        let n = self.n_steps;
        assert_eq!(normals.len(), 2 * n);
        assert_eq!(dw.len(), n);
        assert_eq!(wtilde.len(), n + 1);

        for (i, dwi) in dw.iter_mut().enumerate() {
            let mut v = 0.0;
            for (j, lij) in self.lower[i].iter().enumerate().take(i + 1) {
                v += *lij * normals[j];
            }
            *dwi = v;
        }
        wtilde[0] = 0.0;
        for i in 0..n {
            let mut v = 0.0;
            for (j, lij) in self.lower[n + i].iter().enumerate().take(n + i + 1) {
                v += *lij * normals[j];
            }
            wtilde[i + 1] = v;
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
    // Scale the PSD rejection threshold with the running diagonal magnitude:
    // an absolute cutoff is dimension- and scale-blind (slightly negative
    // pivots of order eps * max diag are routine round-off for large joint
    // covariance matrices, e.g. 1024x1024 hybrid-scheme grids). Tiny negative
    // pivots within tolerance are clamped to zero; only materially negative
    // pivots are treated as a hard PSD failure.
    let mut diag_max = 0.0_f64;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for (&lik, &ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
            }

            let tol = 1.0e-12 * diag_max.max(1.0);
            if i == j {
                if sum < -tol {
                    return Err("covariance matrix is not positive semidefinite".to_string());
                }
                let pivot = sum.max(0.0);
                diag_max = diag_max.max(pivot);
                l[i][j] = pivot.sqrt();
            } else if l[j][j] * l[j][j] > tol {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

static GL64: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();

fn gl64() -> &'static (Vec<f64>, Vec<f64>) {
    GL64.get_or_init(|| gauss_legendre_nodes(64))
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` via Newton iteration on the
/// Legendre recurrence (Numerical Recipes `gauleg`).
fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];
    let m = n.div_ceil(2);

    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..100 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z -= p1 / pp;
            if (z - z1).abs() <= 1.0e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let weight = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i] = weight;
        w[n - 1 - i] = weight;
    }

    (x, w)
}

/// Gauss-Legendre quadrature of `f` over `[lo, hi]` with the cached 64-node rule.
fn gl64_integrate(lo: f64, hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let (nodes, weights) = gl64();
    let half = 0.5 * (hi - lo);
    let mid = 0.5 * (hi + lo);
    let mut acc = 0.0;
    for (&x, &w) in nodes.iter().zip(weights.iter()) {
        acc += w * f(mid + half * x);
    }
    half * acc
}

/// Exact covariance of the Riemann-Liouville Volterra process driving rough
/// Bergomi, `W~_t = sqrt(2H) \int_0^t (t-s)^{H-1/2} dW_s`:
///
/// `E[W~_s W~_t] = 2H \int_0^{min(s,t)} (s-u)^{H-1/2} (t-u)^{H-1/2} du`
///
/// The diagonal is exact: `E[W~_t^2] = t^{2H}`. Off-diagonal values are
/// computed by Gauss-Legendre quadrature after substitutions that remove the
/// endpoint singularity: with `w = min - u`, `delta = |t - s|`, the integral
/// `\int_0^{min} w^{H-1/2} (delta + w)^{H-1/2} dw` is split at `w = delta`;
/// on `[0, min(delta, min)]` the substitution `w = x^{1/(H+1/2)}` flattens the
/// `w^{H-1/2}` factor exactly, and on `[delta, min]` the integrand is smooth in
/// `log w`.
pub fn fbm_covariance(s: f64, t: f64, hurst: f64) -> f64 {
    let (a, b) = if s <= t { (s, t) } else { (t, s) };
    if a <= 0.0 {
        return 0.0;
    }

    let two_h = 2.0 * hurst;
    let delta = b - a;
    if delta <= 0.0 {
        return a.powf(two_h);
    }

    let g = hurst - 0.5;
    let m = 1.0 / (hurst + 0.5);

    // I1: w in [0, min(delta, a)] with w = x^m, so w^g dw = m dx exactly.
    let c = delta.min(a);
    let x_hi = c.powf(hurst + 0.5);
    let i1 = m * gl64_integrate(0.0, x_hi, |x| (delta + x.powf(m)).powf(g));

    // I2: w in [delta, a] (only when a > delta), smooth on a log scale.
    let i2 = if a > delta {
        gl64_integrate(delta.ln(), a.ln(), |u| {
            let w = u.exp();
            ((g + 1.0) * u).exp() * (delta + w).powf(g)
        })
    } else {
        0.0
    };

    two_h * (i1 + i2)
}

/// Exact cross-covariance between the Volterra process and an increment of its
/// driving Brownian motion: for `v < u` and `t > v`,
///
/// `E[W~_t (W_u - W_v)] = sqrt(2H)/(H + 1/2) * [(t - v)^{H+1/2} - (t - max(0, t - u))...]`
///
/// i.e. `sqrt(2H)/(H+1/2) * [(t-v)^{H+1/2} - (t-u)^{H+1/2}]` for `u <= t`, with
/// the integration truncated at `t` when the increment straddles it, and zero
/// when `v >= t`.
fn volterra_bm_cross_cov(t: f64, v: f64, u: f64, hurst: f64) -> f64 {
    if t <= v || u <= v {
        return 0.0;
    }
    let hp = hurst + 0.5;
    let upper = u.min(t);
    (2.0 * hurst).sqrt() / hp * ((t - v).powf(hp) - (t - upper).powf(hp))
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
    match scheme {
        FbmScheme::Cholesky => fbm_path_cholesky(hurst, maturity, n_steps, seed),
        FbmScheme::Hybrid => fbm_path_hybrid(hurst, maturity, n_steps, seed),
    }
}

/// Evolves the asset along one path given the Volterra path `wtilde`, the
/// Brownian increments `dw` driving it, and independent normals for the
/// orthogonal component. The asset shock is `rho * dw + sqrt(1-rho^2) * dW_perp`
/// so the spot correlates with the true Brownian driver of the variance.
#[allow(clippy::too_many_arguments)]
fn evolve_asset_path(
    spot: f64,
    r: f64,
    q: f64,
    maturity: f64,
    hurst: f64,
    eta: f64,
    rho: f64,
    xi0: f64,
    dw: &[f64],
    wtilde: &[f64],
    perp_normals: &[f64],
    sign: f64,
) -> f64 {
    let n_steps = dw.len();
    let dt = maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_perp = (1.0 - rho * rho).max(0.0).sqrt();

    let mut s = spot;
    for i in 0..n_steps {
        let t = i as f64 * dt;
        let variance = xi0 * (eta * sign * wtilde[i] - 0.5 * eta * eta * t.powf(2.0 * hurst)).exp();
        let shock = rho * sign * dw[i] + rho_perp * sqrt_dt * sign * perp_normals[i];

        s *= ((r - q - 0.5 * variance) * dt + variance.max(0.0).sqrt() * shock).exp();
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
    let generator = JointVolterraGenerator::new(hurst, maturity, n_steps)?;

    let mut terminals = Vec::with_capacity(n_paths);
    let mut z_joint = vec![0.0_f64; 2 * n_steps];
    let mut z_perp = vec![0.0_f64; n_steps];
    let mut dw = vec![0.0_f64; n_steps];
    let mut wtilde = vec![0.0_f64; n_steps + 1];

    let draw_path = |rng: &mut FastRng, z_joint: &mut [f64], z_perp: &mut [f64]| {
        for zi in z_joint.iter_mut() {
            *zi = sample_standard_normal(rng);
        }
        for zi in z_perp.iter_mut() {
            *zi = sample_standard_normal(rng);
        }
    };

    for pair in 0..(n_paths / 2) {
        let mut rng = FastRng::from_seed(
            FastRngKind::Xoshiro256PlusPlus,
            resolve_stream_seed(seed, pair, true),
        );
        draw_path(&mut rng, &mut z_joint, &mut z_perp);
        generator.sample(&z_joint, &mut dw, &mut wtilde);

        // Antithetic pair: the map normals -> (dw, wtilde) is linear, so the
        // antithetic path is obtained by flipping the sign of every shock.
        for &sign in &[1.0, -1.0] {
            terminals.push(evolve_asset_path(
                spot, r, q, maturity, hurst, eta, rho, xi0, &dw, &wtilde, &z_perp, sign,
            ));
        }
    }

    if n_paths % 2 == 1 {
        let mut rng = FastRng::from_seed(
            FastRngKind::Xoshiro256PlusPlus,
            resolve_stream_seed(seed, n_paths / 2, true),
        );
        draw_path(&mut rng, &mut z_joint, &mut z_perp);
        generator.sample(&z_joint, &mut dw, &mut wtilde);
        terminals.push(evolve_asset_path(
            spot, r, q, maturity, hurst, eta, rho, xi0, &dw, &wtilde, &z_perp, 1.0,
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

    /// Brute-force reference for the Volterra covariance: midpoint rule on the
    /// singularity-flattening substitution `u = min - w`, `w = x^{1/(H+1/2)}`
    /// over the full range, with a large number of nodes.
    fn volterra_cov_reference(s: f64, t: f64, hurst: f64) -> f64 {
        let (a, b) = if s <= t { (s, t) } else { (t, s) };
        if a <= 0.0 {
            return 0.0;
        }
        let delta = b - a;
        let g = hurst - 0.5;
        let m = 1.0 / (hurst + 0.5);
        let x_hi = a.powf(hurst + 0.5);
        let n = 2_000_000;
        let dx = x_hi / n as f64;
        let mut acc = 0.0;
        for k in 0..n {
            let x = (k as f64 + 0.5) * dx;
            acc += (delta + x.powf(m)).powf(g);
        }
        2.0 * hurst * m * acc * dx
    }

    #[test]
    fn volterra_covariance_diagonal_is_exact() {
        for &hurst in &[0.05, 0.1, 0.3, 0.5, 0.7, 0.9] {
            for &t in &[0.1, 0.25, 1.0, 2.0, 5.0] {
                let c = fbm_covariance(t, t, hurst);
                let expected = t.powf(2.0 * hurst);
                assert!(
                    (c - expected).abs() <= 1.0e-14 * expected.max(1.0),
                    "H={hurst} t={t} got={c} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn volterra_covariance_matches_brute_force_quadrature() {
        for &hurst in &[0.1, 0.3, 0.7] {
            for &(s, t) in &[(0.25, 0.5), (0.5, 0.52), (0.1, 2.0), (1.0, 1.015_625)] {
                let got = fbm_covariance(s, t, hurst);
                let reference = volterra_cov_reference(s, t, hurst);
                assert!(
                    (got - reference).abs() <= 1.0e-7 * reference.abs().max(1.0e-8),
                    "H={hurst} s={s} t={t} got={got} reference={reference}"
                );
            }
        }
    }

    #[test]
    fn volterra_covariance_h_half_is_brownian() {
        for &(s, t) in &[(0.25, 0.5), (0.5, 0.5), (0.75, 2.0), (1.5, 1.0)] {
            let c = fbm_covariance(s, t, 0.5);
            assert!(
                (c - s.min(t)).abs() <= 1.0e-12,
                "s={s} t={t} got={c} expected={}",
                s.min(t)
            );
        }
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

        // Volterra (non-stationary) increments: Cov(W~_{t1}, W~_{t2} - W~_{t1})
        // = C(t1, t2) - C(t1, t1).
        let dt = maturity / n_steps as f64;
        let expected = fbm_covariance(dt, 2.0 * dt, hurst) - fbm_covariance(dt, dt, hurst);

        assert!(
            (cov - expected).abs() <= expected.abs() * 0.35 + 2.0e-4,
            "cov={cov} expected={expected}"
        );
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

    /// At `H = 1/2` the Volterra kernel is constant, so `W~ = W` and the model
    /// degenerates to a standard Bergomi model driven by an ordinary Brownian
    /// motion. Compare against a direct standard-BM implementation with the
    /// same Euler scheme, including a non-zero spot-vol correlation.
    #[test]
    fn rbergomi_h_half_degenerates_to_standard_bergomi() {
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.0;
        let q = 0.0;
        let maturity = 1.0;
        let hurst = 0.5;
        let eta = 1.5;
        let rho = -0.7;
        let xi0 = 0.04;
        let n_steps = 64_usize;
        let n_paths = 40_000_usize;

        let mc = rbergomi_european_mc(
            spot, strike, r, q, maturity, hurst, eta, rho, xi0, n_paths, n_steps,
        );

        // Direct standard-BM Bergomi: v_t = xi0 * exp(eta * W_t - eta^2 t / 2),
        // dS/S = sqrt(v) (rho dW + sqrt(1-rho^2) dW_perp), antithetic pairs.
        let dt = maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho_perp = (1.0 - rho * rho).max(0.0).sqrt();
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut z_w = vec![0.0_f64; n_steps];
        let mut z_p = vec![0.0_f64; n_steps];
        for pair in 0..(n_paths / 2) {
            let mut rng = FastRng::from_seed(
                FastRngKind::Xoshiro256PlusPlus,
                resolve_stream_seed(999, pair, true),
            );
            for i in 0..n_steps {
                z_w[i] = sample_standard_normal(&mut rng);
                z_p[i] = sample_standard_normal(&mut rng);
            }
            for sign in [1.0_f64, -1.0] {
                let mut s = spot;
                let mut w = 0.0;
                for i in 0..n_steps {
                    let t = i as f64 * dt;
                    let v = xi0 * (eta * w - 0.5 * eta * eta * t).exp();
                    let shock = rho * sign * z_w[i] * sqrt_dt + rho_perp * sqrt_dt * sign * z_p[i];
                    s *= ((r - q - 0.5 * v) * dt + v.max(0.0).sqrt() * shock).exp();
                    s = s.max(1.0e-12);
                    w += sign * z_w[i] * sqrt_dt;
                }
                let payoff = (s - strike).max(0.0_f64);
                sum += payoff;
                sum_sq += payoff * payoff;
            }
        }
        let n = (2 * (n_paths / 2)) as f64;
        let ref_mean = sum / n;
        let ref_var = (sum_sq - n * ref_mean * ref_mean).max(0.0) / (n - 1.0);
        let ref_price = ref_mean; // r = 0
        let ref_stderr = (ref_var / n).sqrt();

        let stderr = mc.stderr.unwrap_or(0.0);
        let combined = (stderr * stderr + ref_stderr * ref_stderr).sqrt();
        assert!(
            (mc.price - ref_price).abs() <= 3.0 * combined + 1.0e-6,
            "rbergomi={} standard={} combined_stderr={}",
            mc.price,
            ref_price,
            combined
        );
    }

    /// Under the risk-neutral simulation the terminal spot must be a
    /// martingale after deflating by the money-market account:
    /// `E[S_T] = S_0 * exp((r - q) T)`.
    #[test]
    fn rbergomi_terminal_spot_is_martingale_at_h_03() {
        let spot = 100.0;
        let r = 0.02;
        let q = 0.01;
        let maturity = 1.0;
        let hurst = 0.3;
        let eta = 1.5;
        let rho = -0.6;
        let xi0 = 0.04;
        let n_paths = 60_000;
        let n_steps = 64;

        let terminals = simulate_terminal_spots(
            spot, r, q, maturity, hurst, eta, rho, xi0, n_paths, n_steps, 4_242,
        )
        .unwrap();

        let n = terminals.len() as f64;
        let mean = terminals.iter().sum::<f64>() / n;
        let var = terminals
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / (n - 1.0);
        let stderr = (var / n).sqrt();
        let forward = spot * ((r - q) * maturity).exp();

        assert!(
            (mean - forward).abs() <= 3.0 * stderr,
            "mean={mean} forward={forward} stderr={stderr}"
        );
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

    /// The joint generator must reproduce the exact cross-covariance between
    /// the Volterra process and the Brownian increments that drive it.
    #[test]
    fn joint_generator_reproduces_cross_covariance() {
        let hurst = 0.3;
        let maturity = 1.0;
        let n_steps = 8;
        let n_paths = 60_000;
        let dt = maturity / n_steps as f64;

        let generator = JointVolterraGenerator::new(hurst, maturity, n_steps).unwrap();
        let mut z = vec![0.0_f64; 2 * n_steps];
        let mut dw = vec![0.0_f64; n_steps];
        let mut wtilde = vec![0.0_f64; n_steps + 1];

        // Accumulate E[W~_{t_n} dW_j] and E[W~_{t_n}^2].
        let mut cross = vec![0.0_f64; n_steps];
        let mut var_wt = 0.0;
        for i in 0..n_paths {
            let mut rng = FastRng::from_seed(
                FastRngKind::Xoshiro256PlusPlus,
                resolve_stream_seed(31, i, true),
            );
            for zi in &mut z {
                *zi = sample_standard_normal(&mut rng);
            }
            generator.sample(&z, &mut dw, &mut wtilde);
            let wt = wtilde[n_steps];
            var_wt += wt * wt;
            for (j, &dwj) in dw.iter().enumerate() {
                cross[j] += wt * dwj;
            }
        }

        let n = n_paths as f64;
        var_wt /= n;
        let expected_var = maturity.powf(2.0 * hurst);
        assert!(
            (var_wt - expected_var).abs() <= 0.05 * expected_var,
            "var={var_wt} expected={expected_var}"
        );

        for (j, c) in cross.iter().enumerate() {
            let sample = c / n;
            let expected =
                volterra_bm_cross_cov(maturity, j as f64 * dt, (j + 1) as f64 * dt, hurst);
            // MC noise on each cross moment is ~ sqrt(Var(W~) * dt / n).
            let noise = 4.0 * (expected_var * dt / n).sqrt();
            assert!(
                (sample - expected).abs() <= noise + 0.02 * expected.abs(),
                "j={j} sample={sample} expected={expected}"
            );
        }
    }
}
