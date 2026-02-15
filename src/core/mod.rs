//! Core traits, common domain types, and library-wide result/error structures.

use std::collections::HashMap;

use crate::market::Market;

pub mod types;

pub use types::*;

/// Standardized Greeks container used by engine results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// First derivative to spot.
    pub delta: f64,
    /// Second derivative to spot.
    pub gamma: f64,
    /// First derivative to volatility.
    pub vega: f64,
    /// First derivative to time.
    pub theta: f64,
    /// First derivative to rate.
    pub rho: f64,
}

/// Common trait implemented by every priceable instrument.
pub trait Instrument: std::fmt::Debug {
    /// Returns a short type identifier for diagnostics and bindings.
    fn instrument_type(&self) -> &str;
}

/// Pricing engine abstraction over an instrument type.
pub trait PricingEngine<I: Instrument> {
    /// Prices an instrument under the provided market state.
    fn price(&self, instrument: &I, market: &Market) -> Result<PricingResult, PricingError>;
}

/// Unified engine result payload.
#[derive(Debug, Clone)]
pub struct PricingResult {
    /// Present value.
    pub price: f64,
    /// Standard error (typically Monte Carlo only).
    pub stderr: Option<f64>,
    /// Greeks when available from the engine.
    pub greeks: Option<Greeks>,
    /// Engine-specific scalar diagnostics.
    pub diagnostics: HashMap<String, f64>,
}

/// Engine and model errors surfaced by the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingError {
    /// Input validation error.
    InvalidInput(String),
    /// Non-convergence in an iterative algorithm.
    ConvergenceFailure(String),
    /// Required market datum is unavailable.
    MarketDataMissing(String),
    /// Numerical issue (overflow, invalid state, etc.).
    NumericalError(String),
}

impl std::fmt::Display for PricingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::ConvergenceFailure(msg) => write!(f, "convergence failure: {msg}"),
            Self::MarketDataMissing(msg) => write!(f, "market data missing: {msg}"),
            Self::NumericalError(msg) => write!(f, "numerical error: {msg}"),
        }
    }
}

impl std::error::Error for PricingError {}
