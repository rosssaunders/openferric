//! Calibration diagnostics and warning synthesis.

use crate::calibration::core::{
    BoxConstraints, CalibrationDiagnostics, CalibrationWarningFlag, ConvergenceInfo, FitQuality,
    InstrumentError, ParameterStability,
};

pub fn fit_quality(errors: &[InstrumentError]) -> FitQuality {
    if errors.is_empty() {
        return FitQuality {
            rmse: 0.0,
            mae: 0.0,
            max_abs_error: 0.0,
            liquid_rmse: 0.0,
        };
    }

    let n = errors.len() as f64;
    let rmse = (errors
        .iter()
        .map(|e| e.signed_error * e.signed_error)
        .sum::<f64>()
        / n)
        .sqrt();
    let mae = errors.iter().map(|e| e.abs_error).sum::<f64>() / n;
    let max_abs_error = errors.iter().map(|e| e.abs_error).fold(0.0_f64, f64::max);

    let liquid: Vec<&InstrumentError> = errors.iter().filter(|e| e.liquid).collect();
    let liquid_rmse = if liquid.is_empty() {
        rmse
    } else {
        (liquid
            .iter()
            .map(|e| e.signed_error * e.signed_error)
            .sum::<f64>()
            / liquid.len() as f64)
            .sqrt()
    };

    FitQuality {
        rmse,
        mae,
        max_abs_error,
        liquid_rmse,
    }
}

pub fn parameter_stability(
    names: Vec<String>,
    previous: &[f64],
    current: &[f64],
    threshold: f64,
) -> ParameterStability {
    let n = previous.len().min(current.len());
    let mut relative_changes = Vec::with_capacity(n);
    for i in 0..n {
        let base = previous[i].abs().max(1e-12);
        relative_changes.push((current[i] - previous[i]).abs() / base);
    }

    let max_relative_change = relative_changes.iter().copied().fold(0.0_f64, f64::max);

    ParameterStability {
        parameter_names: names,
        relative_changes,
        max_relative_change,
        stable: max_relative_change <= threshold.max(1e-6),
    }
}

pub fn warning_flags(
    convergence: &ConvergenceInfo,
    condition_number: f64,
    fit: &FitQuality,
    bounds: Option<&BoxConstraints>,
    params: Option<&[f64]>,
    stability: Option<&ParameterStability>,
) -> Vec<CalibrationWarningFlag> {
    let mut out = Vec::new();

    if !convergence.converged {
        out.push(CalibrationWarningFlag::NonConvergent);
    }

    if !condition_number.is_finite() || condition_number > 1e8 {
        out.push(CalibrationWarningFlag::IllConditioned);
    }

    if fit.liquid_rmse > 0.005 {
        out.push(CalibrationWarningFlag::PoorFit);
    }

    if let (Some(b), Some(x)) = (bounds, params)
        && b.hits_boundary(x, 1e-6)
    {
        out.push(CalibrationWarningFlag::HitBoundary);
    }

    if let Some(stability) = stability
        && !stability.stable
    {
        out.push(CalibrationWarningFlag::UnstableParameters);
    }

    out
}

pub fn diagnostics(
    errors: &[InstrumentError],
    convergence: &ConvergenceInfo,
    condition_number: f64,
    bounds: Option<&BoxConstraints>,
    params: Option<&[f64]>,
    stability: Option<ParameterStability>,
) -> CalibrationDiagnostics {
    let fit = fit_quality(errors);
    let flags = warning_flags(
        convergence,
        condition_number,
        &fit,
        bounds,
        params,
        stability.as_ref(),
    );

    CalibrationDiagnostics {
        fit_quality: fit,
        parameter_stability: stability,
        warning_flags: flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::core::TerminationReason;

    fn error(id: &str, signed_error: f64, liquid: bool) -> InstrumentError {
        InstrumentError {
            id: id.to_string(),
            market_mid: 0.2,
            market_bid: None,
            market_ask: None,
            model: 0.2 + signed_error,
            signed_error,
            effective_error: signed_error,
            abs_error: signed_error.abs(),
            weight: 1.0,
            within_bid_ask: false,
            liquid,
        }
    }

    fn convergence(converged: bool) -> ConvergenceInfo {
        ConvergenceInfo {
            iterations: 3,
            objective_evaluations: 8,
            gradient_norm: 1.0e-9,
            step_norm: 2.0e-9,
            converged,
            reason: if converged {
                TerminationReason::GradientTolerance
            } else {
                TerminationReason::MaxIterations
            },
        }
    }

    #[test]
    fn fit_quality_matches_unweighted_population_metrics_and_liquid_subset() {
        assert_eq!(
            fit_quality(&[]),
            FitQuality {
                rmse: 0.0,
                mae: 0.0,
                max_abs_error: 0.0,
                liquid_rmse: 0.0,
            }
        );

        let errors = [error("liquid", 3.0, true), error("illiquid", -4.0, false)];
        let fit = fit_quality(&errors);
        assert_eq!(fit.rmse, 12.5_f64.sqrt());
        assert_eq!(fit.mae, 3.5);
        assert_eq!(fit.max_abs_error, 4.0);
        assert_eq!(fit.liquid_rmse, 3.0);

        let no_liquid = [error("a", 3.0, false), error("b", -4.0, false)];
        assert_eq!(fit_quality(&no_liquid).liquid_rmse, 12.5_f64.sqrt());
    }

    #[test]
    fn parameter_stability_uses_zero_base_floor_and_numeric_dimensions() {
        let stable = parameter_stability(
            vec!["zero".into(), "level".into()],
            &[0.0, 2.0],
            &[5.0e-19, 2.4],
            0.25,
        );
        assert_eq!(stable.parameter_names, ["zero", "level"]);
        for (actual, expected) in stable.relative_changes.iter().zip([5.0e-7_f64, 0.2_f64]) {
            let roundoff = 4.0 * f64::EPSILON * expected.max(1.0);
            assert!((actual - expected).abs() <= roundoff);
        }
        assert!((stable.max_relative_change - 0.2).abs() <= 4.0 * f64::EPSILON);
        assert!(stable.stable);

        let unstable = parameter_stability(vec!["x".into()], &[1.0], &[1.1], 0.05);
        assert!(!unstable.stable);
        assert!((unstable.max_relative_change - 0.1).abs() <= f64::EPSILON);

        // Labels are diagnostic metadata: omitting one must not hide a
        // material numeric change from the stability decision.
        let missing_labels = parameter_stability(Vec::new(), &[1.0], &[2.0], 0.5);
        assert!(missing_labels.parameter_names.is_empty());
        assert_eq!(missing_labels.relative_changes, [1.0]);
        assert_eq!(missing_labels.max_relative_change, 1.0);
        assert!(!missing_labels.stable);

        let empty = parameter_stability(Vec::new(), &[], &[], 0.0);
        assert!(empty.parameter_names.is_empty());
        assert!(empty.relative_changes.is_empty());
        assert_eq!(empty.max_relative_change, 0.0);
        assert!(empty.stable);
    }

    #[test]
    fn warning_synthesis_covers_every_flag_and_threshold_boundary() {
        let fit = FitQuality {
            rmse: 0.006,
            mae: 0.006,
            max_abs_error: 0.006,
            liquid_rmse: 0.006,
        };
        let bounds = BoxConstraints::new(vec![0.0], vec![1.0]).unwrap();
        let unstable = ParameterStability {
            parameter_names: vec!["x".into()],
            relative_changes: vec![0.2],
            max_relative_change: 0.2,
            stable: false,
        };
        assert_eq!(
            warning_flags(
                &convergence(false),
                f64::INFINITY,
                &fit,
                Some(&bounds),
                Some(&[0.0]),
                Some(&unstable),
            ),
            [
                CalibrationWarningFlag::NonConvergent,
                CalibrationWarningFlag::IllConditioned,
                CalibrationWarningFlag::PoorFit,
                CalibrationWarningFlag::HitBoundary,
                CalibrationWarningFlag::UnstableParameters,
            ]
        );

        let threshold_fit = FitQuality {
            rmse: 0.005,
            mae: 0.005,
            max_abs_error: 0.005,
            liquid_rmse: 0.005,
        };
        let stable = ParameterStability {
            parameter_names: vec!["x".into()],
            relative_changes: vec![0.0],
            max_relative_change: 0.0,
            stable: true,
        };
        assert!(
            warning_flags(
                &convergence(true),
                1.0e8,
                &threshold_fit,
                Some(&bounds),
                Some(&[0.5]),
                Some(&stable),
            )
            .is_empty()
        );
        assert!(
            warning_flags(
                &convergence(true),
                1.0,
                &threshold_fit,
                Some(&bounds),
                None,
                None,
            )
            .is_empty()
        );
    }

    #[test]
    fn diagnostics_aggregates_fit_stability_and_warning_flags() {
        let bounds = BoxConstraints::new(vec![-1.0], vec![1.0]).unwrap();
        let stability = ParameterStability {
            parameter_names: vec!["rho".into()],
            relative_changes: vec![0.3],
            max_relative_change: 0.3,
            stable: false,
        };
        let result = diagnostics(
            &[error("q", 0.01, true)],
            &convergence(false),
            2.0,
            Some(&bounds),
            Some(&[0.0]),
            Some(stability.clone()),
        );
        assert_eq!(result.fit_quality.rmse, 0.01);
        assert_eq!(result.parameter_stability, Some(stability));
        assert_eq!(
            result.warning_flags,
            [
                CalibrationWarningFlag::NonConvergent,
                CalibrationWarningFlag::PoorFit,
                CalibrationWarningFlag::UnstableParameters,
            ]
        );
    }
}
