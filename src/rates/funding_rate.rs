//! Module `rates::funding_rate`.
//!
//! Minimal funding-rate term structure support for discrete funding products.

use chrono::{DateTime, Utc};

const HOURS_PER_YEAR: f64 = 8_760.0;
const SECONDS_PER_YEAR: f64 = HOURS_PER_YEAR * 3_600.0;

/// Flat funding-rate curve with an optional volatility parameter.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FundingRateCurve {
    /// Flat annualized funding rate used for forward intervals.
    pub flat_rate: f64,
    /// Annualized funding-rate volatility used for convexity-style bumping.
    pub volatility: f64,
}

impl FundingRateCurve {
    /// Builds a flat funding-rate curve.
    pub fn new(flat_rate: f64) -> Self {
        Self {
            flat_rate,
            volatility: 0.0,
        }
    }

    /// Sets the annualized funding-rate volatility.
    pub fn with_volatility(mut self, volatility: f64) -> Self {
        self.volatility = volatility.max(0.0);
        self
    }

    /// Flat forward funding rate for any interval.
    pub fn forward_rate(&self, _start: DateTime<Utc>, _end: DateTime<Utc>) -> f64 {
        self.flat_rate
    }

    /// Expected funding rate including a simple convexity-style volatility adjustment.
    pub fn expected_rate(
        &self,
        as_of: DateTime<Utc>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> f64 {
        let remaining_years = ((end - as_of).num_seconds().max(0) as f64) / SECONDS_PER_YEAR;
        self.forward_rate(start, end) + 0.5 * self.volatility * self.volatility * remaining_years
    }

    /// Returns a copy with a parallel rate shift applied.
    pub fn parallel_shifted(&self, bump: f64) -> Self {
        Self {
            flat_rate: self.flat_rate + bump,
            volatility: self.volatility,
        }
    }

    /// Returns a copy with a volatility shift applied.
    pub fn volatility_shifted(&self, bump: f64) -> Self {
        Self {
            flat_rate: self.flat_rate,
            volatility: (self.volatility + bump).max(0.0),
        }
    }
}
