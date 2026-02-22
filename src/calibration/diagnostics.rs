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
