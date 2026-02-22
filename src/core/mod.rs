//! Core traits, common domain types, and library-wide result/error structures.

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

/// Named first-order sensitivity from AAD.
#[derive(Debug, Clone, PartialEq)]
pub struct AadSensitivity {
    /// Risk factor identifier.
    pub factor: String,
    /// Partial derivative with respect to `factor`.
    pub value: f64,
}

/// AAD pricing payload: scalar price and gradient vector.
#[derive(Debug, Clone, PartialEq)]
pub struct AadPricingResult {
    /// Present value.
    pub price: f64,
    /// Gradient entries keyed by risk factor name.
    pub gradient: Vec<AadSensitivity>,
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

    /// Prices and returns first-order sensitivities via AAD when available.
    ///
    /// Default implementation returns a valid price with an empty gradient.
    fn price_with_greeks_aad(
        &self,
        instrument: &I,
        market: &Market,
    ) -> Result<AadPricingResult, PricingError> {
        let result = self.price(instrument, market)?;
        Ok(AadPricingResult {
            price: result.price,
            gradient: Vec::new(),
        })
    }
}

/// Compact key set for engine diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            Self::Vol => "vol",
            Self::VolAdj => "vol_adj",
        }
    }

    #[inline]
    pub fn from_str(key: &str) -> Option<Self> {
        match key {
            "barrier_level" => Some(Self::BarrierLevel),
            "conversion_value" => Some(Self::ConversionValue),
            "credit_spread" => Some(Self::CreditSpread),
            "d" => Some(Self::D),
            "d1" => Some(Self::D1),
            "d2" => Some(Self::D2),
            "delta" => Some(Self::Delta),
            "discount_factor" => Some(Self::DiscountFactor),
            "double_knockout_base" => Some(Self::DoubleKnockoutBase),
            "effective_vol" => Some(Self::EffectiveVol),
            "exercise_dates" => Some(Self::ExerciseDates),
            "fair_variance" => Some(Self::FairVariance),
            "fair_volatility" => Some(Self::FairVolatility),
            "inside_barriers" => Some(Self::InsideBarriers),
            "integral" => Some(Self::Integral),
            "max_exercises" => Some(Self::MaxExercises),
            "min_exercises" => Some(Self::MinExercises),
            "npv" => Some(Self::Npv),
            "num_paths" => Some(Self::NumPaths),
            "num_threads" => Some(Self::NumThreads),
            "observation_count" => Some(Self::ObservationCount),
            "num_space_steps" => Some(Self::NumSpaceSteps),
            "num_steps" => Some(Self::NumSteps),
            "num_time_steps" => Some(Self::NumTimeSteps),
            "pd" => Some(Self::Pd),
            "pm" => Some(Self::Pm),
            "pu" => Some(Self::Pu),
            "pv_forward" => Some(Self::PvForward),
            "rho" => Some(Self::Rho),
            "rho_domestic" => Some(Self::RhoDomestic),
            "rho_foreign" => Some(Self::RhoForeign),
            "s_max" => Some(Self::SMax),
            "series_terms" => Some(Self::SeriesTerms),
            "survival_digital" => Some(Self::SurvivalDigital),
            "u" => Some(Self::U),
            "var_of_var" => Some(Self::VarOfVar),
            "vol" => Some(Self::Vol),
            "vol_adj" => Some(Self::VolAdj),
            _ => None,
        }
    }
}

/// Inline diagnostics storage used in [`PricingResult`].
#[derive(Debug, Clone, Default, PartialEq)]
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
        self.len() == 0
    }

    #[inline]
    pub fn insert(&mut self, key: &'static str, value: f64) -> Option<f64> {
        let key = DiagKey::from_str(key).unwrap_or_else(|| {
            panic!("unsupported diagnostics key `{key}`; add it to core::DiagKey")
        });

        for entry in &mut self.entries {
            if let Some((entry_key, existing)) = entry {
                if *entry_key == key {
                    let prev = *existing;
                    *existing = value;
                    return Some(prev);
                }
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
        DiagKey::from_str(key)
            .and_then(|diag_key| self.find_entry(diag_key))
            .is_some()
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&f64> {
        let key = DiagKey::from_str(key)?;
        self.find_entry(key)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &f64)> {
        self.iter_entries().map(|(k, v)| (k.as_str(), v))
    }
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
    pub diagnostics: Diagnostics,
}

const _: [(); 1] = [(); (std::mem::size_of::<PricingResult>() <= 384) as usize];

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
