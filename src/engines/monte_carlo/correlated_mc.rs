//! Correlated Monte Carlo helpers for multi-asset path generation.
//!
//! References:
//! - Glasserman (2004) for correlated-path simulation.
//! - Higham (2002) for nearest-correlation PSD repair.

use crate::core::PricingError;
use crate::math::fast_rng::{FastRng, sample_standard_normal};
use crate::math::{
    FactorCorrelationModel, PsdProjectionConfig, cholesky_lower_psd, correlate_normals,
    validate_or_repair_correlation_matrix,
};

/// Builds a Cholesky factor from a correlation matrix, repairing to nearest PSD if needed.
///
/// Returns `(lower_cholesky, was_repaired)`.
pub fn cholesky_for_correlation(
    corr_matrix: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, bool), PricingError> {
    let (corr, was_repaired) = validate_or_repair_correlation_matrix(
        corr_matrix,
        corr_matrix.len(),
        PsdProjectionConfig::default(),
    )
    .map_err(PricingError::InvalidInput)?;

    let chol = cholesky_lower_psd(&corr, 1.0e-12).ok_or_else(|| {
        PricingError::InvalidInput("correlation matrix could not be factorized".to_string())
    })?;

    Ok((chol, was_repaired))
}

/// Samples one vector of correlated standard normals from a Cholesky factor.
pub fn sample_correlated_normals_cholesky(
    chol: &[Vec<f64>],
    rng: &mut FastRng,
    out: &mut [f64],
) -> Result<(), PricingError> {
    if chol.is_empty() || out.len() != chol.len() {
        return Err(PricingError::InvalidInput(
            "output length must match cholesky dimension".to_string(),
        ));
    }

    let mut indep = vec![0.0; chol.len()];
    for z in &mut indep {
        *z = sample_standard_normal(rng);
    }

    correlate_normals(chol, &indep, out);
    Ok(())
}

/// Samples one vector of correlated standard normals from a factor model.
pub fn sample_correlated_normals_factor(
    model: &FactorCorrelationModel,
    rng: &mut FastRng,
    out: &mut [f64],
) -> Result<(), PricingError> {
    model
        .sample_correlated_normals(rng, out)
        .map_err(PricingError::InvalidInput)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fast_rng::{FastRng, FastRngKind};

    #[test]
    fn cholesky_builder_repairs_non_psd_input() {
        let bad = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, -0.95],
            vec![0.95, -0.95, 1.0],
        ];

        let (chol, was_repaired) = cholesky_for_correlation(&bad).unwrap();
        assert!(was_repaired);
        assert_eq!(chol.len(), 3);
    }

    #[test]
    fn factor_sampling_returns_finite_normals() {
        let model = FactorCorrelationModel::OneFactor {
            loadings: vec![0.6, 0.5, 0.4],
        };
        let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, 42);
        let mut z = vec![0.0; 3];

        sample_correlated_normals_factor(&model, &mut rng, &mut z).unwrap();
        assert!(z.iter().all(|x| x.is_finite()));
    }
}
