//! Unified model calibration framework.
//!
//! This module standardizes model calibration across volatility and rates models:
//! - common `Calibrator` trait and `CalibrationResult` payload,
//! - constrained optimizers (LM / DE / Nelder-Mead),
//! - model-specific calibrators (Heston, SABR, SVI, Hull-White),
//! - fit diagnostics and warning flags,
//! - serde-compatible calibration outputs for persistence/audit.

pub mod core;
pub mod diagnostics;
pub mod heston;
pub mod hull_white;
pub mod instruments;
pub mod optimizers;
pub mod sabr;
pub mod svi;

pub use core::{
    BoxConstraints, CalibrationDiagnostics, CalibrationResult, CalibrationWarningFlag, Calibrator,
    ConvergenceInfo, FitQuality, InstrumentError, ParameterStability, TerminationReason,
};
pub use diagnostics::{diagnostics, fit_quality, parameter_stability, warning_flags};
pub use heston::{HestonCalibrationParams, HestonCalibrator};
pub use hull_white::{HullWhiteCalibrationParams, HullWhiteCalibrator};
pub use instruments::{OptionVolQuote, SwaptionVolQuote, VolQuote, bid_ask_aware_error};
pub use optimizers::{
    DifferentialEvolutionOptions, LmOptions, NelderMeadOptions, OptimisationResult,
    differential_evolution, levenberg_marquardt, nelder_mead,
};
pub use sabr::{SabrCalibrationParams, SabrCalibrator};
pub use svi::{
    SviCalibrationParams, SviCalibrator, SviJumpWingsCalibrationParams, SviParameterization,
    SviRawCalibrationParams,
};
