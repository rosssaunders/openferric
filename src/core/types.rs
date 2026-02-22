//! Foundational contract and payoff enums shared across pricing modules.
//!
//! This module defines the typed vocabulary used by engines: option side and exercise style,
//! barrier direction/style tuples, and Asian averaging/strike conventions.
//! Key types are [`OptionType`], [`ExerciseStyle`], [`BarrierSpec`], and [`AsianSpec`].
//! References: Hull (2018), Ch. 10 (plain options) and Haug (2007), barrier and Asian chapters.
//! Numerical note: these are data containers, but helper methods (for example `OptionType::sign`)
//! support branch-light implementations in closed-form and Monte Carlo code.
//! Use this module when building instrument definitions; pricing formulas live in engines/models.

/// Plain-vanilla option side.
///
/// This selects whether payoff is call-like (`max(S-K, 0)`) or put-like (`max(K-S, 0)`),
/// following the sign conventions used in Hull, Ch. 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Call option payoff profile.
    Call,
    /// Put option payoff profile.
    Put,
}

impl OptionType {
    /// Returns `+1.0` for calls and `-1.0` for puts.
    ///
    /// This is useful for branch-free formulas where payoff direction is represented
    /// by a scalar multiplier.
    ///
    /// # Examples
    /// ```
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
///
/// `Bermudan { dates }` dates are year fractions from valuation time and should
/// lie in `(0, expiry]` at instrument-validation time.
#[derive(Debug, Clone, PartialEq)]
pub enum ExerciseStyle {
    /// Exercise only at expiry.
    European,
    /// Exercise at any time up to expiry.
    American,
    /// Exercise at specific times (in year fractions).
    Bermudan { dates: Vec<f64> },
}

/// Barrier crossing direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierDirection {
    /// Barrier is breached when spot moves upward through the level.
    Up,
    /// Barrier is breached when spot moves downward through the level.
    Down,
}

/// Barrier knock behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierStyle {
    /// Option activates once the barrier is hit.
    In,
    /// Option deactivates once the barrier is hit.
    Out,
}

/// Barrier contract parameters.
///
/// Used by barrier engines implementing one-touch knock-in/out logic
/// (see Haug, *The Complete Guide to Option Pricing Formulas*, barrier chapter).
///
/// # Examples
/// ```
/// use openferric::core::{BarrierDirection, BarrierSpec, BarrierStyle};
///
/// let spec = BarrierSpec {
///     direction: BarrierDirection::Up,
///     style: BarrierStyle::Out,
///     level: 120.0,
///     rebate: 0.0,
/// };
///
/// assert_eq!(spec.level, 120.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Averaging {
    /// Arithmetic averaging.
    Arithmetic,
    /// Geometric averaging.
    Geometric,
}

/// Strike convention for Asian options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrikeType {
    /// Fixed strike contract.
    Fixed,
    /// Floating strike contract.
    Floating,
}

/// Asian option contract parameters.
///
/// Observation times are year fractions from valuation time. Engines typically
/// assume strictly increasing times and may reject empty schedules.
///
/// # Examples
/// ```
/// use openferric::core::{AsianSpec, Averaging, StrikeType};
///
/// let asian = AsianSpec {
///     averaging: Averaging::Arithmetic,
///     strike_type: StrikeType::Fixed,
///     observation_times: vec![0.25, 0.5, 0.75, 1.0],
/// };
///
/// assert_eq!(asian.observation_times.len(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AsianSpec {
    /// Averaging method.
    pub averaging: Averaging,
    /// Strike convention.
    pub strike_type: StrikeType,
    /// Observation times in year fractions.
    pub observation_times: Vec<f64>,
}
