//! Unified calibration core abstractions.
//!
//! References:
//! - Nocedal and Wright, *Numerical Optimization* (2nd ed.), Ch. 10.
//! - More (1978), Levenberg-Marquardt implementation and convergence behavior.
//! - Gatheral, *The Volatility Surface* (2006), static-arbitrage diagnostics.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Box constraints `lower <= x <= upper` used by all optimizers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxConstraints {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl BoxConstraints {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, String> {
        if lower.is_empty() || lower.len() != upper.len() {
            return Err("constraints require same non-zero lower/upper dimensions".to_string());
        }
        for i in 0..lower.len() {
            if !lower[i].is_finite() || !upper[i].is_finite() || lower[i] > upper[i] {
                return Err(format!(
                    "invalid bound at index {i}: [{}, {}]",
                    lower[i], upper[i]
                ));
            }
        }
        Ok(Self { lower, upper })
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }

    pub fn clamp(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, v)| v.clamp(self.lower[i], self.upper[i]))
            .collect()
    }

    pub fn hits_boundary(&self, x: &[f64], eps: f64) -> bool {
        x.iter().enumerate().any(|(i, &v)| {
            (v - self.lower[i]).abs() <= eps.max(1e-12)
                || (self.upper[i] - v).abs() <= eps.max(1e-12)
        })
    }
}

/// Single instrument-level calibration error record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentError {
    pub id: String,
    pub market_mid: f64,
    pub market_bid: Option<f64>,
    pub market_ask: Option<f64>,
    pub model: f64,
    pub signed_error: f64,
    pub effective_error: f64,
    pub abs_error: f64,
    pub weight: f64,
    pub within_bid_ask: bool,
    pub liquid: bool,
}

/// Optimizer termination reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    GradientTolerance,
    StepTolerance,
    ObjectiveTolerance,
    Stagnation,
    MaxIterations,
    NumericalFailure,
}

/// Convergence metadata for optimization runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub iterations: usize,
    pub objective_evaluations: usize,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub converged: bool,
    pub reason: TerminationReason,
}

/// Aggregate fit-quality metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitQuality {
    pub rmse: f64,
    pub mae: f64,
    pub max_abs_error: f64,
    pub liquid_rmse: f64,
}

/// Day-over-day style parameter stability summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterStability {
    pub parameter_names: Vec<String>,
    pub relative_changes: Vec<f64>,
    pub max_relative_change: f64,
    pub stable: bool,
}

/// High-level warning flags derived from fit/convergence diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationWarningFlag {
    IllConditioned,
    HitBoundary,
    PoorFit,
    NonConvergent,
    UnstableParameters,
}

/// Derived diagnostics attached to every calibration result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDiagnostics {
    pub fit_quality: FitQuality,
    pub parameter_stability: Option<ParameterStability>,
    pub warning_flags: Vec<CalibrationWarningFlag>,
}

/// Standardized calibration output payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResult<P> {
    pub params: P,
    pub objective: f64,
    pub per_instrument_error: Vec<InstrumentError>,
    /// Jacobian matrix as row-major vectors: one row per instrument residual.
    pub jacobian: Vec<Vec<f64>>,
    pub condition_number: f64,
    pub convergence: ConvergenceInfo,
    pub diagnostics: CalibrationDiagnostics,
}

/// Model-specific calibrator interface.
pub trait Calibrator<P> {
    type Instrument;

    fn name(&self) -> &'static str;

    fn calibrate(&self, instruments: &[Self::Instrument]) -> Result<CalibrationResult<P>, String>;
}

/// Converts a dense matrix into nested row vectors.
pub fn matrix_to_rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Condition number `kappa = sigma_max / sigma_min` from SVD singular values.
pub fn matrix_condition_number(jacobian: &DMatrix<f64>) -> f64 {
    if jacobian.nrows() == 0 || jacobian.ncols() == 0 {
        return 1.0;
    }

    let svd = jacobian.clone().svd(false, false);
    let mut sigma_max: f64 = 0.0;
    let mut sigma_min = f64::INFINITY;

    for s in svd.singular_values.iter() {
        sigma_max = sigma_max.max(*s);
        if *s > 1e-14 {
            sigma_min = sigma_min.min(*s);
        }
    }

    if sigma_min.is_finite() && sigma_min > 0.0 {
        sigma_max / sigma_min
    } else {
        f64::INFINITY
    }
}

#[inline]
pub fn finite_metric(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::MAX }
}

pub fn sanitize_convergence(mut c: ConvergenceInfo) -> ConvergenceInfo {
    c.gradient_norm = finite_metric(c.gradient_norm);
    c.step_norm = finite_metric(c.step_norm);
    c
}
