//! Module `core::types`.
//!
//! Implements types abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `OptionType`, `ExerciseStyle`, `BarrierDirection`, `BarrierStyle`, `BarrierSpec` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.
/// Plain-vanilla option side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptionType {
    /// Call option payoff profile.
    Call,
    /// Put option payoff profile.
    Put,
}

impl OptionType {
    /// Returns +1.0 for calls and -1.0 for puts.
    ///
    /// # Examples
    /// ```rust
    /// use openferric::core::OptionType;
    ///
    /// assert_eq!(OptionType::Call.sign(), 1.0);
    /// assert_eq!(OptionType::Put.sign(), -1.0);
    /// ```
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}

/// Exercise rights for an option contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExerciseStyle {
    /// Exercise only at expiry.
    European,
    /// Exercise at any time up to expiry.
    American,
    /// Exercise at specific times (in year fractions).
    Bermudan { dates: Vec<f64> },
}

/// Barrier crossing direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BarrierDirection {
    /// Barrier is breached when spot moves upward through the level.
    Up,
    /// Barrier is breached when spot moves downward through the level.
    Down,
}

/// Barrier knock behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BarrierStyle {
    /// Option activates once the barrier is hit.
    In,
    /// Option deactivates once the barrier is hit.
    Out,
}

/// Barrier contract parameters.
///
/// # Examples
/// ```rust
/// use openferric::core::{BarrierDirection, BarrierSpec, BarrierStyle};
///
/// let b = BarrierSpec {
///     direction: BarrierDirection::Down,
///     style: BarrierStyle::Out,
///     level: 80.0,
///     rebate: 2.0,
/// };
/// assert_eq!(b.level, 80.0);
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BarrierSpec {
    /// Barrier direction.
    pub direction: BarrierDirection,
    /// Knock-in or knock-out.
    pub style: BarrierStyle,
    /// Barrier level in spot units.
    pub level: f64,
    /// Cash rebate paid upon knock event or at maturity, depending on model.
    pub rebate: f64,
}

/// Averaging method for Asian options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Averaging {
    /// Arithmetic averaging.
    Arithmetic,
    /// Geometric averaging.
    Geometric,
}

/// Strike convention for Asian options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StrikeType {
    /// Fixed strike contract.
    Fixed,
    /// Floating strike contract.
    Floating,
}

/// Asian option contract parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AsianSpec {
    /// Averaging method.
    pub averaging: Averaging,
    /// Strike convention.
    pub strike_type: StrikeType,
    /// Observation times in year fractions.
    pub observation_times: Vec<f64>,
}
