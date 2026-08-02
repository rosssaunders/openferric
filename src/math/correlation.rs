//! Correlation-matrix and copula utilities for multi-asset Monte Carlo.
//!
//! References:
//! - Higham, N. (2002), *Computing the nearest correlation matrix*.
//! - Glasserman, P. (2004), *Monte Carlo Methods in Financial Engineering*.
//!
//! This module centralizes correlation handling used across multi-asset engines:
//! validation/repair, Cholesky factorization, factor-model generation, copula
//! sampling, and stress transformations.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::math::fast_rng::{FastRng, sample_standard_normal};
use crate::math::{normal_cdf, normal_inv_cdf};

/// Configuration for nearest-PSD / nearest-correlation projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsdProjectionConfig {
    /// Convergence tolerance in Frobenius norm.
    pub tol: f64,
    /// Maximum number of Higham alternating-projection iterations.
    pub max_iterations: usize,
}

impl Default for PsdProjectionConfig {
    fn default() -> Self {
        Self {
            tol: 1.0e-10,
            max_iterations: 100,
        }
    }
}

/// Copula family for dependence simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CopulaFamily {
    /// Gaussian copula.
    Gaussian,
    /// Student-t copula with integer degrees of freedom.
    StudentT {
        /// Degrees of freedom (must be >= 2).
        degrees_of_freedom: u32,
    },
}

/// Stress scenarios for correlation matrices.
#[derive(Debug, Clone, PartialEq)]
pub enum CorrelationStressScenario {
    /// Multiply all off-diagonal elements by `factor`.
    ScaleOffDiagonal { factor: f64 },
    /// Add `shift` to all off-diagonal elements.
    AdditiveShift { shift: f64 },
    /// Floor off-diagonal entries to at least `floor`.
    FloorOffDiagonal { floor: f64 },
    /// Cap off-diagonal entries to at most `cap`.
    CapOffDiagonal { cap: f64 },
    /// Override one pair `(i, j)`.
    OverridePair { i: usize, j: usize, value: f64 },
}

/// One- or multi-factor correlation model.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorCorrelationModel {
    /// One-factor loadings `beta_i`, with `corr(i,j)=beta_i*beta_j`.
    OneFactor { loadings: Vec<f64> },
    /// Multi-factor loadings per asset row.
    ///
    /// `loadings[i][k]` is loading of asset `i` to factor `k`.
    /// Row norm must be <= 1 to keep idiosyncratic variance non-negative.
    MultiFactor { loadings: Vec<Vec<f64>> },
}

impl FactorCorrelationModel {
    /// Number of assets represented by this model.
    pub fn n_assets(&self) -> usize {
        match self {
            Self::OneFactor { loadings } => loadings.len(),
            Self::MultiFactor { loadings } => loadings.len(),
        }
    }

    /// Number of systemic factors.
    pub fn n_factors(&self) -> usize {
        match self {
            Self::OneFactor { .. } => 1,
            Self::MultiFactor { loadings } => loadings.first().map_or(0, Vec::len),
        }
    }

    /// Validates model shape and parameter bounds.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::OneFactor { loadings } => {
                if loadings.is_empty() {
                    return Err("one-factor model requires at least one loading".to_string());
                }
                if loadings
                    .iter()
                    .any(|b| !b.is_finite() || b.abs() > 1.0 + 1.0e-12)
                {
                    return Err("one-factor loadings must be finite and in [-1, 1]".to_string());
                }
            }
            Self::MultiFactor { loadings } => {
                if loadings.is_empty() {
                    return Err("multi-factor model requires at least one asset row".to_string());
                }
                let n_factors = loadings[0].len();
                if n_factors == 0 {
                    return Err("multi-factor model requires at least one factor".to_string());
                }
                if loadings.iter().any(|row| row.len() != n_factors) {
                    return Err("all multi-factor rows must have the same length".to_string());
                }
                for row in loadings {
                    if row.iter().any(|x| !x.is_finite()) {
                        return Err("multi-factor loadings must be finite".to_string());
                    }
                    let norm2 = row.iter().map(|x| x * x).sum::<f64>();
                    if norm2 > 1.0 + 1.0e-10 {
                        return Err(
                            "multi-factor row norm must be <= 1 for unit-variance assets"
                                .to_string(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Builds the implied full correlation matrix.
    pub fn correlation_matrix(&self) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;

        match self {
            Self::OneFactor { loadings } => {
                let n = loadings.len();
                let mut corr = vec![vec![0.0; n]; n];
                for i in 0..n {
                    corr[i][i] = 1.0;
                    for j in (i + 1)..n {
                        let rho = (loadings[i] * loadings[j]).clamp(-1.0, 1.0);
                        corr[i][j] = rho;
                        corr[j][i] = rho;
                    }
                }
                Ok(corr)
            }
            Self::MultiFactor { loadings } => {
                let n = loadings.len();
                let mut corr = vec![vec![0.0; n]; n];
                for i in 0..n {
                    corr[i][i] = 1.0;
                    for j in (i + 1)..n {
                        let rho = loadings[i]
                            .iter()
                            .zip(loadings[j].iter())
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            .clamp(-1.0, 1.0);
                        corr[i][j] = rho;
                        corr[j][i] = rho;
                    }
                }
                Ok(corr)
            }
        }
    }

    /// Samples one vector of correlated standard normals from the factor model.
    pub fn sample_correlated_normals(
        &self,
        rng: &mut FastRng,
        out: &mut [f64],
    ) -> Result<(), String> {
        self.validate()?;
        if out.len() != self.n_assets() {
            return Err("output length does not match factor model asset count".to_string());
        }

        match self {
            Self::OneFactor { loadings } => {
                let m = sample_standard_normal(rng);
                for (i, out_i) in out.iter_mut().enumerate() {
                    let beta = loadings[i].clamp(-1.0, 1.0);
                    let idio = (1.0 - beta * beta).max(0.0).sqrt();
                    let eps = sample_standard_normal(rng);
                    *out_i = beta.mul_add(m, idio * eps);
                }
            }
            Self::MultiFactor { loadings } => {
                let n_factors = loadings[0].len();
                let mut factors = vec![0.0; n_factors];
                for f in &mut factors {
                    *f = sample_standard_normal(rng);
                }

                for (i, out_i) in out.iter_mut().enumerate() {
                    let systematic = loadings[i]
                        .iter()
                        .zip(factors.iter())
                        .map(|(l, f)| l * f)
                        .sum::<f64>();
                    let norm2 = loadings[i].iter().map(|x| x * x).sum::<f64>();
                    let idio = (1.0 - norm2).max(0.0).sqrt();
                    let eps = sample_standard_normal(rng);
                    *out_i = systematic + idio * eps;
                }
            }
        }

        Ok(())
    }
}

/// Validates that `corr_matrix` is a finite, symmetric `n_assets x n_assets`
/// correlation matrix with unit diagonal and entries in `[-1, 1]`.
pub fn validate_correlation_matrix(
    corr_matrix: &[Vec<f64>],
    n_assets: usize,
) -> Result<(), String> {
    if corr_matrix.len() != n_assets || corr_matrix.iter().any(|row| row.len() != n_assets) {
        return Err("correlation matrix dimensions must match asset count".to_string());
    }

    for (i, row_i) in corr_matrix.iter().enumerate().take(n_assets) {
        let di = row_i[i];
        if !di.is_finite() || (di - 1.0).abs() > 1.0e-10 {
            return Err("correlation matrix diagonal must be 1".to_string());
        }
        for (j, rho) in row_i.iter().copied().enumerate().take(n_assets) {
            if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
                return Err("correlation entries must be finite and in [-1, 1]".to_string());
            }
            if (rho - corr_matrix[j][i]).abs() > 1.0e-10 {
                return Err("correlation matrix must be symmetric".to_string());
            }
        }
    }

    Ok(())
}

/// Returns the minimum eigenvalue of a symmetric matrix.
pub fn min_eigenvalue_symmetric(matrix: &[Vec<f64>]) -> Option<f64> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let m = to_dmatrix(matrix);
    let eig = SymmetricEigen::new(m);
    eig.eigenvalues.iter().copied().reduce(f64::min)
}

/// Returns `true` if matrix is positive semidefinite within tolerance `tol`.
pub fn is_positive_semidefinite(matrix: &[Vec<f64>], tol: f64) -> bool {
    min_eigenvalue_symmetric(matrix).is_some_and(|lmin| lmin >= -tol)
}

/// Computes a nearest correlation matrix using Higham (2002) alternating projections.
///
/// The algorithm alternates between:
/// - Projection onto symmetric PSD matrices (`S`), and
/// - Projection onto unit-diagonal affine space (`U`).
///
/// The output is additionally symmetrized and re-normalized to keep diagonal entries
/// at exactly one.
pub fn nearest_correlation_matrix_higham(
    matrix: &[Vec<f64>],
    cfg: PsdProjectionConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err("matrix must be square and non-empty".to_string());
    }
    if matrix
        .iter()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1.0e6)
    {
        return Err("matrix entries must be finite and reasonably bounded".to_string());
    }

    let mut y = to_dmatrix(matrix);
    y = symmetrize(&y);
    for i in 0..n {
        y[(i, i)] = 1.0;
    }

    let mut delta_s = DMatrix::<f64>::zeros(n, n);

    for _ in 0..cfg.max_iterations {
        let r = symmetrize(&(y.clone() - delta_s.clone()));
        let x = project_psd(&r);
        delta_s = x.clone() - r;

        let mut y_next = x;
        for i in 0..n {
            y_next[(i, i)] = 1.0;
        }
        y_next = symmetrize(&y_next);

        let diff = frobenius_norm(&(y_next.clone() - y.clone()));
        y = y_next;
        if diff < cfg.tol {
            break;
        }
    }

    // Final cleanup pass: PSD projection and exact unit diagonal.
    y = project_psd(&symmetrize(&y));
    for i in 0..n {
        y[(i, i)] = 1.0;
    }
    y = symmetrize(&y);

    let mut out = from_dmatrix(&y);
    let mut i = 0usize;
    while i < n {
        out[i][i] = 1.0;
        let mut j = i + 1;
        while j < n {
            let clipped = out[i][j].clamp(-1.0, 1.0);
            out[i][j] = clipped;
            out[j][i] = clipped;
            j += 1;
        }
        i += 1;
    }

    Ok(out)
}

/// Validates a correlation matrix and repairs it with Higham projection if needed.
///
/// Returns `(matrix, was_repaired)`.
pub fn validate_or_repair_correlation_matrix(
    corr_matrix: &[Vec<f64>],
    n_assets: usize,
    cfg: PsdProjectionConfig,
) -> Result<(Vec<Vec<f64>>, bool), String> {
    validate_correlation_matrix(corr_matrix, n_assets)?;
    if is_positive_semidefinite(corr_matrix, cfg.tol) {
        return Ok((corr_matrix.to_vec(), false));
    }

    let repaired = nearest_correlation_matrix_higham(corr_matrix, cfg)?;
    validate_correlation_matrix(&repaired, n_assets)?;
    if !is_positive_semidefinite(&repaired, 1.0e-8) {
        return Err("nearest-correlation projection did not produce PSD output".to_string());
    }

    Ok((repaired, true))
}

/// Cholesky decomposition for symmetric positive semidefinite matrices.
///
/// Returns lower-triangular `L` such that `L L^T ~= matrix`.
pub fn cholesky_lower_psd(matrix: &[Vec<f64>], tol: f64) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut l = vec![vec![0.0_f64; n]; n];

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
                // Standard PSD Cholesky: a (numerically) zero pivot means the
                // matrix is rank-deficient at this column; set the pivot to
                // zero rather than flooring it at sqrt(tol), which would
                // amplify round-off noise in the dependent column by ~1/sqrt(tol).
                l[i][j] = if sum <= tol { 0.0 } else { sum.sqrt() };
            } else if l[j][j] > 0.0 {
                l[i][j] = sum / l[j][j];
            }
            // For a zero pivot the dependent column entries stay 0.
        }
    }

    Some(l)
}

/// Applies a Cholesky factor to independent normals.
pub fn correlate_normals(chol: &[Vec<f64>], indep: &[f64], out: &mut [f64]) {
    for i in 0..chol.len() {
        let mut sum = 0.0;
        for (j, lij) in chol[i].iter().enumerate().take(i + 1) {
            sum += *lij * indep[j];
        }
        out[i] = sum;
    }
}

/// Samples copula-uniform variates from a Cholesky factor.
pub fn sample_copula_uniforms_from_cholesky(
    chol: &[Vec<f64>],
    copula: CopulaFamily,
    rng: &mut FastRng,
    out_uniforms: &mut [f64],
) -> Result<(), String> {
    let n = chol.len();
    if n == 0 || out_uniforms.len() != n {
        return Err("copula output length must match cholesky dimension".to_string());
    }

    let mut indep = vec![0.0; n];
    let mut corr = vec![0.0; n];
    for z in &mut indep {
        *z = sample_standard_normal(rng);
    }
    correlate_normals(chol, &indep, &mut corr);

    map_copula_normals_to_uniforms(&corr, copula, rng, out_uniforms)
}

/// Samples copula-uniform variates from a factor model.
pub fn sample_copula_uniforms_from_factor_model(
    model: &FactorCorrelationModel,
    copula: CopulaFamily,
    rng: &mut FastRng,
    out_uniforms: &mut [f64],
) -> Result<(), String> {
    let n = model.n_assets();
    if out_uniforms.len() != n {
        return Err("copula output length must match factor model asset count".to_string());
    }

    let mut corr_normals = vec![0.0; n];
    model.sample_correlated_normals(rng, &mut corr_normals)?;
    map_copula_normals_to_uniforms(&corr_normals, copula, rng, out_uniforms)
}

/// Applies correlation stress scenarios and optionally repairs to the nearest PSD matrix.
pub fn apply_correlation_stress(
    base: &[Vec<f64>],
    scenarios: &[CorrelationStressScenario],
    repair_to_psd: bool,
    cfg: PsdProjectionConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let n = base.len();
    validate_correlation_matrix(base, n)?;

    let mut out = base.to_vec();

    for scenario in scenarios {
        match scenario {
            CorrelationStressScenario::ScaleOffDiagonal { factor } => {
                if !factor.is_finite() {
                    return Err("scale factor must be finite".to_string());
                }
                let mut i = 0usize;
                while i < n {
                    let mut j = i + 1;
                    while j < n {
                        let rho = (out[i][j] * factor).clamp(-0.999_999, 0.999_999);
                        out[i][j] = rho;
                        out[j][i] = rho;
                        j += 1;
                    }
                    i += 1;
                }
            }
            CorrelationStressScenario::AdditiveShift { shift } => {
                if !shift.is_finite() {
                    return Err("additive shift must be finite".to_string());
                }
                let mut i = 0usize;
                while i < n {
                    let mut j = i + 1;
                    while j < n {
                        let rho = (out[i][j] + shift).clamp(-0.999_999, 0.999_999);
                        out[i][j] = rho;
                        out[j][i] = rho;
                        j += 1;
                    }
                    i += 1;
                }
            }
            CorrelationStressScenario::FloorOffDiagonal { floor } => {
                if !floor.is_finite() || *floor < -1.0 || *floor > 1.0 {
                    return Err("floor must be finite and in [-1, 1]".to_string());
                }
                let mut i = 0usize;
                while i < n {
                    let mut j = i + 1;
                    while j < n {
                        let rho = out[i][j].max(*floor).clamp(-0.999_999, 0.999_999);
                        out[i][j] = rho;
                        out[j][i] = rho;
                        j += 1;
                    }
                    i += 1;
                }
            }
            CorrelationStressScenario::CapOffDiagonal { cap } => {
                if !cap.is_finite() || *cap < -1.0 || *cap > 1.0 {
                    return Err("cap must be finite and in [-1, 1]".to_string());
                }
                let mut i = 0usize;
                while i < n {
                    let mut j = i + 1;
                    while j < n {
                        let rho = out[i][j].min(*cap).clamp(-0.999_999, 0.999_999);
                        out[i][j] = rho;
                        out[j][i] = rho;
                        j += 1;
                    }
                    i += 1;
                }
            }
            CorrelationStressScenario::OverridePair { i, j, value } => {
                if *i >= n || *j >= n || *i == *j {
                    return Err("override pair indices must be distinct and in-range".to_string());
                }
                if !value.is_finite() || !(-1.0..=1.0).contains(value) {
                    return Err("override pair value must be finite and in [-1, 1]".to_string());
                }
                let rho = (*value).clamp(-0.999_999, 0.999_999);
                out[*i][*j] = rho;
                out[*j][*i] = rho;
            }
        }

        for (i, row) in out.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }
    }

    if repair_to_psd && !is_positive_semidefinite(&out, cfg.tol) {
        out = nearest_correlation_matrix_higham(&out, cfg)?;
    }

    validate_correlation_matrix(&out, n)?;
    Ok(out)
}

fn map_copula_normals_to_uniforms(
    corr_normals: &[f64],
    copula: CopulaFamily,
    rng: &mut FastRng,
    out_uniforms: &mut [f64],
) -> Result<(), String> {
    match copula {
        CopulaFamily::Gaussian => {
            for (u, z) in out_uniforms.iter_mut().zip(corr_normals.iter()) {
                // `normal_cdf` is relative-accurate in the lower tail, so the
                // lower clamp only needs to guard the hard zero returned for
                // z < -37; the inverse CDF handles 1e-300 fine. Near one the
                // attainable resolution is limited by f64 spacing, hence the
                // 1 - 1e-16 upper clamp (largest f64 strictly below 1).
                *u = normal_cdf(*z).clamp(COPULA_UNIFORM_MIN, COPULA_UNIFORM_MAX);
            }
        }
        CopulaFamily::StudentT { degrees_of_freedom } => {
            if degrees_of_freedom < 2 {
                return Err("student-t copula requires degrees_of_freedom >= 2".to_string());
            }
            let dof_f = degrees_of_freedom as f64;
            let chi2 = sample_chi_square(dof_f, rng);
            let scale = (chi2 / dof_f).max(1.0e-16).sqrt();
            let student = StudentsT::new(0.0, 1.0, dof_f).map_err(|e| e.to_string())?;

            for (u, z) in out_uniforms.iter_mut().zip(corr_normals.iter()) {
                let t = *z / scale;
                *u = student.cdf(t).clamp(COPULA_UNIFORM_MIN, COPULA_UNIFORM_MAX);
            }
        }
    }
    Ok(())
}

/// Lower clamp for copula uniforms; far below any value an accurate normal
/// CDF produces for realistic normals, but keeps `u > 0` for inverse CDFs.
const COPULA_UNIFORM_MIN: f64 = 1.0e-300;
/// Upper clamp for copula uniforms: largest representable f64 strictly below 1.
const COPULA_UNIFORM_MAX: f64 = 1.0 - 1.0e-16;

/// Samples a chi-square variate with (possibly non-integer) `dof > 0` degrees
/// of freedom as `2 * Gamma(dof / 2, scale = 1)` using the Marsaglia-Tsang
/// (2000) squeeze method. O(1) per draw instead of summing `dof` squared
/// normals.
fn sample_chi_square(dof: f64, rng: &mut FastRng) -> f64 {
    2.0 * sample_gamma_marsaglia_tsang(0.5 * dof, rng)
}

/// Marsaglia-Tsang (2000) Gamma(shape, scale = 1) sampler.
fn sample_gamma_marsaglia_tsang(shape: f64, rng: &mut FastRng) -> f64 {
    debug_assert!(shape > 0.0, "gamma shape must be positive");

    if shape < 1.0 {
        // Boosting: Gamma(a) = Gamma(a + 1) * U^(1/a).
        let u = rng.random_f64().max(f64::MIN_POSITIVE);
        return sample_gamma_marsaglia_tsang(shape + 1.0, rng) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (3.0 * d.sqrt());
    loop {
        let x = sample_standard_normal(rng);
        let v = {
            let t = 1.0 + c * x;
            t * t * t
        };
        if v <= 0.0 {
            continue;
        }
        let u = rng.random_f64();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u > 0.0 && u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn to_dmatrix(matrix: &[Vec<f64>]) -> DMatrix<f64> {
    let n = matrix.len();
    let data = matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    DMatrix::from_row_slice(n, n, &data)
}

fn from_dmatrix(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let n = matrix.nrows();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = matrix[(i, j)];
        }
    }
    out
}

fn symmetrize(m: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (m + m.transpose())
}

fn project_psd(m: &DMatrix<f64>) -> DMatrix<f64> {
    let eig = SymmetricEigen::new(symmetrize(m));
    let vals = eig
        .eigenvalues
        .iter()
        .map(|v| (*v).max(0.0))
        .collect::<Vec<_>>();
    let d = DMatrix::from_diagonal(&DVector::from_vec(vals));
    symmetrize(&(eig.eigenvectors.clone() * d * eig.eigenvectors.transpose()))
}

fn frobenius_norm(m: &DMatrix<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Applies inverse-normal transform to copula uniforms to recover standard-normal marginals.
pub fn copula_uniforms_to_normals(uniforms: &[f64], out_normals: &mut [f64]) -> Result<(), String> {
    if uniforms.len() != out_normals.len() {
        return Err("uniform and output lengths must match".to_string());
    }
    for (u, z) in uniforms.iter().zip(out_normals.iter_mut()) {
        *z = normal_inv_cdf((*u).clamp(COPULA_UNIFORM_MIN, COPULA_UNIFORM_MAX));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fast_rng::{FastRng, FastRngKind};
    use approx::assert_relative_eq;

    #[test]
    fn higham_projection_repairs_non_psd_matrix() {
        let bad = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, -0.95],
            vec![0.95, -0.95, 1.0],
        ];

        assert!(validate_correlation_matrix(&bad, 3).is_ok());
        assert!(!is_positive_semidefinite(&bad, 1.0e-12));

        let repaired = nearest_correlation_matrix_higham(&bad, PsdProjectionConfig::default())
            .expect("repair should succeed");

        validate_correlation_matrix(&repaired, 3).expect("repaired matrix should remain valid");
        assert!(is_positive_semidefinite(&repaired, 1.0e-8));
    }

    #[test]
    fn factor_model_implies_psd_correlation() {
        let model = FactorCorrelationModel::MultiFactor {
            loadings: vec![
                vec![0.6, 0.1],
                vec![0.4, -0.2],
                vec![0.2, 0.3],
                vec![0.1, -0.4],
            ],
        };
        let corr = model.correlation_matrix().expect("valid factor model");
        validate_correlation_matrix(&corr, 4).expect("valid corr matrix");
        assert!(is_positive_semidefinite(&corr, 1.0e-10));
    }

    #[test]
    fn psd_cholesky_handles_rank_deficient_matrix_without_noise_amplification() {
        // Rank-1 correlation matrix: all entries 1.
        let n = 4;
        let ones = vec![vec![1.0; n]; n];
        let l = cholesky_lower_psd(&ones, 1.0e-12).expect("rank-1 matrix is PSD");

        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for (&left, &right) in l[i].iter().zip(&l[j]) {
                    acc += left * right;
                }
                assert!(
                    (acc - 1.0).abs() < 1.0e-12,
                    "L L^T ({i},{j}) = {acc}, expected 1"
                );
            }
        }

        // Zero pivots must be exactly zero, not floored at sqrt(tol).
        for (i, row) in l.iter().enumerate().skip(1) {
            assert_eq!(row[i], 0.0, "pivot {i} should collapse to zero");
        }
    }

    #[test]
    fn marsaglia_tsang_chi_square_matches_moments() {
        let n = 200_000usize;
        for &nu in &[5.0_f64, 8.5] {
            let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 12345);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for _ in 0..n {
                let x = sample_chi_square(nu, &mut rng);
                assert!(x.is_finite() && x > 0.0);
                sum += x;
                sum_sq += x * x;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;

            // Mean nu, variance 2*nu. Std errors: sqrt(2nu/n) and ~sqrt(8nu^2... )
            let mean_tol = 5.0 * (2.0 * nu / n as f64).sqrt();
            let var_tol = 0.05 * 2.0 * nu + 0.5;
            assert!(
                (mean - nu).abs() < mean_tol,
                "nu={nu} mean={mean} tol={mean_tol}"
            );
            assert!(
                (var - 2.0 * nu).abs() < var_tol,
                "nu={nu} var={var} expected={}",
                2.0 * nu
            );
        }
    }

    #[test]
    fn t_copula_uniforms_are_bounded() {
        let corr = vec![
            vec![1.0, 0.4, 0.2],
            vec![0.4, 1.0, -0.3],
            vec![0.2, -0.3, 1.0],
        ];
        let chol = cholesky_lower_psd(&corr, 1.0e-12).expect("cholesky");
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 7);
        let mut u = vec![0.0; 3];

        sample_copula_uniforms_from_cholesky(
            &chol,
            CopulaFamily::StudentT {
                degrees_of_freedom: 6,
            },
            &mut rng,
            &mut u,
        )
        .expect("t-copula sample");

        assert!(u.iter().all(|x| x.is_finite() && *x > 0.0 && *x < 1.0));
    }

    #[test]
    fn one_factor_matrix_and_sampling_match_theoretical_moments() {
        let loadings = vec![0.75, -0.4, 1.0];
        let model = FactorCorrelationModel::OneFactor {
            loadings: loadings.clone(),
        };
        assert_eq!(model.n_assets(), 3);
        assert_eq!(model.n_factors(), 1);
        model.validate().unwrap();

        let expected = vec![
            vec![1.0, -0.3, 0.75],
            vec![-0.3, 1.0, -0.4],
            vec![0.75, -0.4, 1.0],
        ];
        let actual = model.correlation_matrix().unwrap();
        for (actual_row, expected_row) in actual.iter().zip(&expected) {
            for (actual_entry, expected_entry) in actual_row.iter().zip(expected_row) {
                assert_relative_eq!(*actual_entry, *expected_entry, epsilon = 2.0 * f64::EPSILON);
            }
        }

        let n = 120_000usize;
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 0xC0FF_EE11);
        let mut sums = [0.0; 3];
        let mut cross = [[0.0; 3]; 3];
        let mut sample = [0.0; 3];
        for _ in 0..n {
            model
                .sample_correlated_normals(&mut rng, &mut sample)
                .unwrap();
            for i in 0..3 {
                sums[i] += sample[i];
                for j in 0..3 {
                    cross[i][j] += sample[i] * sample[j];
                }
            }
        }

        let n_f64 = n as f64;
        let means = sums.map(|sum| sum / n_f64);
        let mean_band = 6.0 / n_f64.sqrt();
        for (i, mean) in means.iter().enumerate() {
            assert!(mean.abs() < mean_band, "asset {i}: mean={mean}");
        }
        for i in 0..3 {
            for j in 0..3 {
                let covariance = cross[i][j] / n_f64 - means[i] * means[j];
                let rho = expected[i][j];
                // For jointly normal unit-variance variables,
                // Var(XY) = 1 + rho^2.  Six standard errors gives a
                // distribution-derived sampling budget, not a flat band.
                let covariance_band = 6.0 * ((1.0 + rho * rho) / n_f64).sqrt();
                assert!(
                    (covariance - rho).abs() < covariance_band,
                    "({i},{j}): covariance={covariance}, rho={rho}, band={covariance_band}"
                );
            }
        }
    }

    #[test]
    fn factor_models_reject_every_shape_and_parameter_error() {
        for model in [
            FactorCorrelationModel::OneFactor { loadings: vec![] },
            FactorCorrelationModel::OneFactor {
                loadings: vec![f64::NAN],
            },
            FactorCorrelationModel::OneFactor {
                loadings: vec![1.001],
            },
            FactorCorrelationModel::MultiFactor { loadings: vec![] },
            FactorCorrelationModel::MultiFactor {
                loadings: vec![vec![]],
            },
            FactorCorrelationModel::MultiFactor {
                loadings: vec![vec![0.2], vec![0.1, 0.2]],
            },
            FactorCorrelationModel::MultiFactor {
                loadings: vec![vec![f64::INFINITY]],
            },
            FactorCorrelationModel::MultiFactor {
                loadings: vec![vec![0.8, 0.8]],
            },
        ] {
            assert!(model.validate().is_err(), "model should fail: {model:?}");
            assert!(model.correlation_matrix().is_err());
        }

        let one_factor = FactorCorrelationModel::OneFactor {
            loadings: vec![0.4, 0.6],
        };
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 9);
        assert!(
            one_factor
                .sample_correlated_normals(&mut rng, &mut [0.0])
                .is_err()
        );

        let multi_factor = FactorCorrelationModel::MultiFactor {
            loadings: vec![vec![0.5, 0.1], vec![-0.2, 0.7]],
        };
        assert_eq!(multi_factor.n_assets(), 2);
        assert_eq!(multi_factor.n_factors(), 2);
        let corr = multi_factor.correlation_matrix().unwrap();
        assert_relative_eq!(corr[0][0], 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(corr[0][1], -0.03, epsilon = f64::EPSILON);
        assert_relative_eq!(corr[1][0], -0.03, epsilon = f64::EPSILON);
        assert_relative_eq!(corr[1][1], 1.0, epsilon = f64::EPSILON);
        let mut out = [0.0; 2];
        multi_factor
            .sample_correlated_normals(&mut rng, &mut out)
            .unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn correlation_stress_scenarios_match_sequential_arithmetic() {
        let base = vec![
            vec![1.0, 0.2, -0.4],
            vec![0.2, 1.0, 0.7],
            vec![-0.4, 0.7, 1.0],
        ];
        let stressed = apply_correlation_stress(
            &base,
            &[
                CorrelationStressScenario::ScaleOffDiagonal { factor: 2.0 },
                CorrelationStressScenario::AdditiveShift { shift: 0.1 },
                CorrelationStressScenario::FloorOffDiagonal { floor: -0.6 },
                CorrelationStressScenario::CapOffDiagonal { cap: 0.8 },
                CorrelationStressScenario::OverridePair {
                    i: 0,
                    j: 2,
                    value: -0.25,
                },
            ],
            false,
            PsdProjectionConfig::default(),
        )
        .unwrap();
        let expected = [[1.0, 0.5, -0.25], [0.5, 1.0, 0.8], [-0.25, 0.8, 1.0]];
        for (actual_row, expected_row) in stressed.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert_relative_eq!(*actual, expected, epsilon = 2.0 * f64::EPSILON);
            }
        }

        let bad = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, -0.95],
            vec![0.95, -0.95, 1.0],
        ];
        let repaired =
            apply_correlation_stress(&bad, &[], true, PsdProjectionConfig::default()).unwrap();
        assert!(is_positive_semidefinite(&repaired, 1.0e-8));
        validate_correlation_matrix(&repaired, 3).unwrap();
    }

    #[test]
    fn correlation_stress_rejects_invalid_scenario_parameters() {
        let base = vec![vec![1.0, 0.2], vec![0.2, 1.0]];
        let invalid = vec![
            CorrelationStressScenario::ScaleOffDiagonal { factor: f64::NAN },
            CorrelationStressScenario::AdditiveShift {
                shift: f64::INFINITY,
            },
            CorrelationStressScenario::FloorOffDiagonal { floor: -1.01 },
            CorrelationStressScenario::CapOffDiagonal { cap: 1.01 },
            CorrelationStressScenario::OverridePair {
                i: 0,
                j: 0,
                value: 0.2,
            },
            CorrelationStressScenario::OverridePair {
                i: 0,
                j: 2,
                value: 0.2,
            },
            CorrelationStressScenario::OverridePair {
                i: 0,
                j: 1,
                value: f64::NAN,
            },
        ];
        for scenario in invalid {
            assert!(
                apply_correlation_stress(
                    &base,
                    &[scenario],
                    false,
                    PsdProjectionConfig::default(),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn matrix_validation_repair_cholesky_and_multiplication_cover_boundaries() {
        for (matrix, n_assets) in [
            (vec![], 1),
            (vec![vec![1.0], vec![0.0]], 2),
            (vec![vec![0.9, 0.0], vec![0.0, 1.0]], 2),
            (vec![vec![1.0, 1.1], vec![1.1, 1.0]], 2),
            (vec![vec![1.0, 0.2], vec![0.3, 1.0]], 2),
        ] {
            assert!(validate_correlation_matrix(&matrix, n_assets).is_err());
        }
        assert_eq!(min_eigenvalue_symmetric(&[]), None);
        assert_eq!(min_eigenvalue_symmetric(&[vec![1.0, 0.0]]), None);
        assert!(!is_positive_semidefinite(&[], 1.0e-12));
        assert!(nearest_correlation_matrix_higham(&[], PsdProjectionConfig::default()).is_err());
        assert!(
            nearest_correlation_matrix_higham(
                &[vec![1.0, 1.0e7], vec![1.0e7, 1.0]],
                PsdProjectionConfig::default(),
            )
            .is_err()
        );

        let valid = vec![vec![1.0, 0.25], vec![0.25, 1.0]];
        let (unchanged, was_repaired) =
            validate_or_repair_correlation_matrix(&valid, 2, PsdProjectionConfig::default())
                .unwrap();
        assert_eq!(unchanged, valid);
        assert!(!was_repaired);

        let bad = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, -0.95],
            vec![0.95, -0.95, 1.0],
        ];
        let (_, was_repaired) =
            validate_or_repair_correlation_matrix(&bad, 3, PsdProjectionConfig::default()).unwrap();
        assert!(was_repaired);

        assert!(cholesky_lower_psd(&[], 1.0e-12).is_none());
        assert!(cholesky_lower_psd(&[vec![1.0, 0.0]], 1.0e-12).is_none());
        assert!(cholesky_lower_psd(&[vec![1.0, 2.0], vec![2.0, 1.0]], 1.0e-12).is_none());

        let lower = vec![
            vec![2.0, 0.0, 0.0],
            vec![3.0, 4.0, 0.0],
            vec![-1.0, 2.0, 5.0],
        ];
        let mut product = [0.0; 3];
        correlate_normals(&lower, &[0.5, -1.0, 2.0], &mut product);
        assert_eq!(product, [1.0, -2.5, 7.5]);
    }

    #[test]
    fn gaussian_and_student_t_copula_transforms_cover_success_and_errors() {
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 73);
        let normals = [-2.0, 0.0, 1.5];
        let mut uniforms = [0.0; 3];
        map_copula_normals_to_uniforms(&normals, CopulaFamily::Gaussian, &mut rng, &mut uniforms)
            .unwrap();
        let mut round_trip = [0.0; 3];
        copula_uniforms_to_normals(&uniforms, &mut round_trip).unwrap();
        for (actual, expected) in round_trip.iter().zip(normals) {
            // The inverse uses the repository's documented BSM rational
            // approximation, whose measured absolute error is below 5e-9.
            assert_relative_eq!(*actual, expected, epsilon = 5.0e-9);
        }

        let mut extremes = [0.0; 2];
        map_copula_normals_to_uniforms(
            &[f64::NEG_INFINITY, f64::INFINITY],
            CopulaFamily::Gaussian,
            &mut rng,
            &mut extremes,
        )
        .unwrap();
        assert_eq!(extremes, [COPULA_UNIFORM_MIN, COPULA_UNIFORM_MAX]);
        copula_uniforms_to_normals(&extremes, &mut [0.0; 2]).unwrap();

        let mut t_uniforms = [0.0; 2];
        map_copula_normals_to_uniforms(
            &[0.0, 0.0],
            CopulaFamily::StudentT {
                degrees_of_freedom: 2,
            },
            &mut rng,
            &mut t_uniforms,
        )
        .unwrap();
        for uniform in t_uniforms {
            assert_relative_eq!(uniform, 0.5, epsilon = 2.0 * f64::EPSILON);
        }
        assert!(
            map_copula_normals_to_uniforms(
                &[0.0],
                CopulaFamily::StudentT {
                    degrees_of_freedom: 1,
                },
                &mut rng,
                &mut [0.0],
            )
            .is_err()
        );
        assert!(copula_uniforms_to_normals(&[0.5], &mut [0.0; 2]).is_err());

        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(
            sample_copula_uniforms_from_cholesky(
                &identity,
                CopulaFamily::Gaussian,
                &mut rng,
                &mut [0.0],
            )
            .is_err()
        );
        let mut sampled = [0.0; 2];
        sample_copula_uniforms_from_cholesky(
            &identity,
            CopulaFamily::Gaussian,
            &mut rng,
            &mut sampled,
        )
        .unwrap();
        assert!(sampled.iter().all(|u| *u > 0.0 && *u < 1.0));

        let factor_model = FactorCorrelationModel::OneFactor {
            loadings: vec![0.3, -0.6],
        };
        assert!(
            sample_copula_uniforms_from_factor_model(
                &factor_model,
                CopulaFamily::Gaussian,
                &mut rng,
                &mut [0.0],
            )
            .is_err()
        );
        sample_copula_uniforms_from_factor_model(
            &factor_model,
            CopulaFamily::Gaussian,
            &mut rng,
            &mut sampled,
        )
        .unwrap();
        assert!(sampled.iter().all(|u| *u > 0.0 && *u < 1.0));
    }

    #[test]
    fn fractional_shape_gamma_boosting_matches_chi_square_moments() {
        // nu < 2 exercises the Gamma(shape < 1) boosting identity used by the
        // general chi-square sampler (the copula API itself restricts nu>=2).
        let nu = 0.75_f64;
        let n = 160_000usize;
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 0x51A9_E123);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let sample = sample_chi_square(nu, &mut rng);
            sum += sample;
            sum_sq += sample * sample;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        let mean_band = 6.0 * (2.0 * nu / n as f64).sqrt();
        let variance_band = 6.0 * ((8.0 * nu * nu + 48.0 * nu) / n as f64).sqrt();
        assert!((mean - nu).abs() < mean_band);
        assert!((variance - 2.0 * nu).abs() < variance_band);
    }
}
