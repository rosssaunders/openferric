//! Module `core::engine`.
//!
//! Implements engine abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `Greeks`, `Instrument`, `PricingEngine`, `DiagKey`, `Diagnostics` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.

use crate::market::Market;

/// Standardized Greeks container used by engine results.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Prices an instrument and, when implemented, computes Greeks via AAD.
    ///
    /// Default behavior falls back to [`PricingEngine::price`].
    #[inline]
    fn price_with_greeks_aad(
        &self,
        instrument: &I,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        self.price(instrument, market)
    }
}

/// Requested execution strategy for hardware-aware pricing engines.
///
/// `Auto` lets an engine select a backend from the available CPU features,
/// thread count, and workload size. The explicit variants request a
/// particular backend; engines return an error when a requested backend is
/// unavailable or cannot preserve the instrument's pricing semantics.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ExecutionPolicy {
    /// Select a backend from runtime capabilities and workload size.
    #[default]
    Auto,
    /// Use the portable scalar implementation.
    Scalar,
    /// Use a single-threaded SIMD implementation.
    Simd,
    /// Use the CPU thread pool, combining it with SIMD when supported.
    Parallel,
    /// Request a GPU implementation.
    Gpu,
    /// Request a just-in-time compiled implementation.
    Jit,
}

/// Backend actually used for a pricing operation.
///
/// Pricing diagnostics encode this enum through
/// [`ExecutionBackend::diagnostic_code`]. A parallel backend may also use
/// SIMD; inspect the `vector_width` diagnostic to distinguish that case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ExecutionBackend {
    Scalar = 0,
    Simd = 1,
    Parallel = 2,
    Gpu = 3,
    Jit = 4,
}

impl ExecutionBackend {
    /// Stable numeric representation stored in scalar pricing diagnostics.
    #[inline]
    pub const fn diagnostic_code(self) -> f64 {
        self as u8 as f64
    }

    /// Decodes a backend diagnostic produced by this library.
    #[inline]
    pub const fn from_diagnostic_code(code: f64) -> Option<Self> {
        match code as u8 {
            0 if code == 0.0 => Some(Self::Scalar),
            1 if code == 1.0 => Some(Self::Simd),
            2 if code == 2.0 => Some(Self::Parallel),
            3 if code == 3.0 => Some(Self::Gpu),
            4 if code == 4.0 => Some(Self::Jit),
            _ => None,
        }
    }
}

/// Compact key set for engine diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DiagKey {
    BarrierLevel,
    ConversionValue,
    CreditSpread,
    D,
    D1,
    D2,
    Delta,
    DiscountFactor,
    DoubleKnockoutBase,
    EffectiveVol,
    ExecutionBackend,
    ExerciseDates,
    FairVariance,
    FairVolatility,
    InsideBarriers,
    Integral,
    MaxExercises,
    MinExercises,
    Npv,
    NumPaths,
    NumThreads,
    ObservationCount,
    NumSpaceSteps,
    NumSteps,
    NumTimeSteps,
    Pd,
    Pm,
    Pu,
    PvForward,
    Rho,
    RhoDomestic,
    RhoForeign,
    SMax,
    SeriesTerms,
    SurvivalDigital,
    U,
    VarOfVar,
    VectorWidth,
    Vol,
    VolAdj,
}

impl DiagKey {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BarrierLevel => "barrier_level",
            Self::ConversionValue => "conversion_value",
            Self::CreditSpread => "credit_spread",
            Self::D => "d",
            Self::D1 => "d1",
            Self::D2 => "d2",
            Self::Delta => "delta",
            Self::DiscountFactor => "discount_factor",
            Self::DoubleKnockoutBase => "double_knockout_base",
            Self::EffectiveVol => "effective_vol",
            Self::ExecutionBackend => "execution_backend",
            Self::ExerciseDates => "exercise_dates",
            Self::FairVariance => "fair_variance",
            Self::FairVolatility => "fair_volatility",
            Self::InsideBarriers => "inside_barriers",
            Self::Integral => "integral",
            Self::MaxExercises => "max_exercises",
            Self::MinExercises => "min_exercises",
            Self::Npv => "npv",
            Self::NumPaths => "num_paths",
            Self::NumThreads => "num_threads",
            Self::ObservationCount => "observation_count",
            Self::NumSpaceSteps => "num_space_steps",
            Self::NumSteps => "num_steps",
            Self::NumTimeSteps => "num_time_steps",
            Self::Pd => "pd",
            Self::Pm => "pm",
            Self::Pu => "pu",
            Self::PvForward => "pv_forward",
            Self::Rho => "rho",
            Self::RhoDomestic => "rho_domestic",
            Self::RhoForeign => "rho_foreign",
            Self::SMax => "s_max",
            Self::SeriesTerms => "series_terms",
            Self::SurvivalDigital => "survival_digital",
            Self::U => "u",
            Self::VarOfVar => "var_of_var",
            Self::VectorWidth => "vector_width",
            Self::Vol => "vol",
            Self::VolAdj => "vol_adj",
        }
    }
}

impl std::str::FromStr for DiagKey {
    type Err = ();

    fn from_str(key: &str) -> Result<Self, Self::Err> {
        match key {
            "barrier_level" => Ok(Self::BarrierLevel),
            "conversion_value" => Ok(Self::ConversionValue),
            "credit_spread" => Ok(Self::CreditSpread),
            "d" => Ok(Self::D),
            "d1" => Ok(Self::D1),
            "d2" => Ok(Self::D2),
            "delta" => Ok(Self::Delta),
            "discount_factor" => Ok(Self::DiscountFactor),
            "double_knockout_base" => Ok(Self::DoubleKnockoutBase),
            "effective_vol" => Ok(Self::EffectiveVol),
            "execution_backend" => Ok(Self::ExecutionBackend),
            "exercise_dates" => Ok(Self::ExerciseDates),
            "fair_variance" => Ok(Self::FairVariance),
            "fair_volatility" => Ok(Self::FairVolatility),
            "inside_barriers" => Ok(Self::InsideBarriers),
            "integral" => Ok(Self::Integral),
            "max_exercises" => Ok(Self::MaxExercises),
            "min_exercises" => Ok(Self::MinExercises),
            "npv" => Ok(Self::Npv),
            "num_paths" => Ok(Self::NumPaths),
            "num_threads" => Ok(Self::NumThreads),
            "observation_count" => Ok(Self::ObservationCount),
            "num_space_steps" => Ok(Self::NumSpaceSteps),
            "num_steps" => Ok(Self::NumSteps),
            "num_time_steps" => Ok(Self::NumTimeSteps),
            "pd" => Ok(Self::Pd),
            "pm" => Ok(Self::Pm),
            "pu" => Ok(Self::Pu),
            "pv_forward" => Ok(Self::PvForward),
            "rho" => Ok(Self::Rho),
            "rho_domestic" => Ok(Self::RhoDomestic),
            "rho_foreign" => Ok(Self::RhoForeign),
            "s_max" => Ok(Self::SMax),
            "series_terms" => Ok(Self::SeriesTerms),
            "survival_digital" => Ok(Self::SurvivalDigital),
            "u" => Ok(Self::U),
            "var_of_var" => Ok(Self::VarOfVar),
            "vector_width" => Ok(Self::VectorWidth),
            "vol" => Ok(Self::Vol),
            "vol_adj" => Ok(Self::VolAdj),
            _ => Err(()),
        }
    }
}

/// Inline diagnostics storage used in [`PricingResult`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostics {
    entries: [Option<(DiagKey, f64)>; 8],
}

impl Diagnostics {
    pub const CAPACITY: usize = 8;

    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries[0].is_none()
    }

    #[inline]
    pub fn insert(&mut self, key: &'static str, value: f64) -> Option<f64> {
        let key: DiagKey = key.parse().unwrap_or_else(|()| {
            panic!("unsupported diagnostics key `{key}`; add it to core::DiagKey")
        });
        self.insert_key(key, value)
    }

    /// Insert a diagnostic value using a pre-resolved `DiagKey`, avoiding the
    /// string-to-enum match on the hot path.
    #[inline]
    pub fn insert_key(&mut self, key: DiagKey, value: f64) -> Option<f64> {
        for (entry_key, existing) in self.entries.iter_mut().flatten() {
            if *entry_key == key {
                let prev = *existing;
                *existing = value;
                return Some(prev);
            }
        }

        for entry in &mut self.entries {
            if entry.is_none() {
                *entry = Some((key, value));
                return None;
            }
        }

        panic!("diagnostics capacity exceeded ({})", Self::CAPACITY);
    }

    #[inline]
    fn iter_entries(&self) -> impl Iterator<Item = &(DiagKey, f64)> {
        self.entries.iter().filter_map(Option::as_ref)
    }

    #[inline]
    fn find_entry(&self, key: DiagKey) -> Option<&f64> {
        self.iter_entries()
            .find_map(|(entry_key, value)| (*entry_key == key).then_some(value))
    }

    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        key.parse::<DiagKey>()
            .ok()
            .and_then(|diag_key| self.find_entry(diag_key))
            .is_some()
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&f64> {
        let key: DiagKey = key.parse().ok()?;
        self.find_entry(key)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &f64)> {
        self.iter_entries().map(|(k, v)| (k.as_str(), v))
    }
}

/// Unified engine result payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PricingResult {
    /// Present value.
    pub price: f64,
    /// Standard error (typically Monte Carlo only).
    pub stderr: Option<f64>,
    /// Greeks when available from the engine.
    pub greeks: Option<Greeks>,
    /// Engine-specific scalar diagnostics.
    pub diagnostics: Diagnostics,
}

const _: [(); 1] = [(); (std::mem::size_of::<PricingResult>() <= 384) as usize];

/// Engine and model errors surfaced by the API.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_backend_diagnostic_codes_round_trip() {
        for backend in [
            ExecutionBackend::Scalar,
            ExecutionBackend::Simd,
            ExecutionBackend::Parallel,
            ExecutionBackend::Gpu,
            ExecutionBackend::Jit,
        ] {
            assert_eq!(
                ExecutionBackend::from_diagnostic_code(backend.diagnostic_code()),
                Some(backend)
            );
        }
        assert_eq!(ExecutionBackend::from_diagnostic_code(-1.0), None);
        assert_eq!(ExecutionBackend::from_diagnostic_code(1.5), None);
        assert_eq!(ExecutionBackend::from_diagnostic_code(f64::NAN), None);
    }

    #[test]
    fn execution_diagnostic_keys_use_stable_names() {
        assert_eq!(DiagKey::ExecutionBackend.as_str(), "execution_backend");
        assert_eq!(DiagKey::VectorWidth.as_str(), "vector_width");
        assert_eq!(
            "execution_backend".parse::<DiagKey>(),
            Ok(DiagKey::ExecutionBackend)
        );
        assert_eq!("vector_width".parse::<DiagKey>(), Ok(DiagKey::VectorWidth));
    }
}
