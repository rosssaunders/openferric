/// Plain-vanilla option side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Call option payoff profile.
    Call,
    /// Put option payoff profile.
    Put,
}

impl OptionType {
    /// Returns +1.0 for calls and -1.0 for puts.
    pub fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}

/// Exercise rights for an option contract.
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
#[derive(Debug, Clone, PartialEq)]
pub struct AsianSpec {
    /// Averaging method.
    pub averaging: Averaging,
    /// Strike convention.
    pub strike_type: StrikeType,
    /// Observation times in year fractions.
    pub observation_times: Vec<f64>,
}
