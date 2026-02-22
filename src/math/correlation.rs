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
                l[i][j] = sum.max(tol).sqrt();
            } else if l[j][j] > tol {
                l[i][j] = sum / l[j][j];
            }
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
                *u = normal_cdf(*z).clamp(1.0e-12, 1.0 - 1.0e-12);
            }
        }
        CopulaFamily::StudentT { degrees_of_freedom } => {
            if degrees_of_freedom < 2 {
                return Err("student-t copula requires degrees_of_freedom >= 2".to_string());
            }
            let dof_f = degrees_of_freedom as f64;
            let chi2 = sample_chi_square_integer_dof(degrees_of_freedom, rng);
            let scale = (chi2 / dof_f).max(1.0e-16).sqrt();
            let student = StudentsT::new(0.0, 1.0, dof_f).map_err(|e| e.to_string())?;

            for (u, z) in out_uniforms.iter_mut().zip(corr_normals.iter()) {
                let t = *z / scale;
                *u = student.cdf(t).clamp(1.0e-12, 1.0 - 1.0e-12);
            }
        }
    }
    Ok(())
}

fn sample_chi_square_integer_dof(degrees_of_freedom: u32, rng: &mut FastRng) -> f64 {
    let mut sum = 0.0;
    for _ in 0..degrees_of_freedom {
        let z = sample_standard_normal(rng);
        sum += z * z;
    }
    sum
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
        *z = normal_inv_cdf((*u).clamp(1.0e-12, 1.0 - 1.0e-12));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fast_rng::{FastRng, FastRngKind};

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
}
