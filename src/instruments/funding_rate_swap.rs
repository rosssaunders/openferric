//! Module `instruments::funding_rate_swap`.
//!
//! Discrete funding-rate swaps with periodic UTC settlement boundaries.

use chrono::{DateTime, Duration, Utc};

use crate::core::{Instrument, PricingError};
use crate::rates::{FundingRateCurve, YieldCurve};

const HOURS_PER_YEAR: f64 = 8_760.0;
const SECONDS_PER_YEAR: f64 = HOURS_PER_YEAR * 3_600.0;
const DEFAULT_SETTLEMENT_INTERVAL_HOURS: u32 = 8;

/// Discrete funding-rate swap settled on periodic UTC boundaries.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FundingRateSwap {
    /// Position size.
    pub notional: f64,
    /// Locked market-implied APR at entry.
    pub fixed_rate: f64,
    /// Time the position was opened.
    pub entry_time: DateTime<Utc>,
    /// Final expiry time.
    pub maturity: DateTime<Utc>,
    /// Settlement interval in hours, typically 8.
    pub settlement_interval_hours: u32,
    /// Venue name.
    pub venue: String,
    /// Underlying asset symbol.
    pub asset: String,
}

impl FundingRateSwap {
    /// Builds a funding-rate swap with the standard 8h settlement convention.
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        entry_time: DateTime<Utc>,
        maturity: DateTime<Utc>,
        venue: impl Into<String>,
        asset: impl Into<String>,
    ) -> Self {
        Self {
            notional,
            fixed_rate,
            entry_time,
            maturity,
            settlement_interval_hours: DEFAULT_SETTLEMENT_INTERVAL_HOURS,
            venue: venue.into(),
            asset: asset.into(),
        }
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.notional.is_finite() || self.notional == 0.0 {
            return Err(PricingError::InvalidInput(
                "funding-rate swap notional must be finite and non-zero".to_string(),
            ));
        }
        if !self.fixed_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "funding-rate swap fixed_rate must be finite".to_string(),
            ));
        }
        if self.maturity <= self.entry_time {
            return Err(PricingError::InvalidInput(
                "funding-rate swap maturity must be after entry_time".to_string(),
            ));
        }
        if self.settlement_interval_hours == 0 {
            return Err(PricingError::InvalidInput(
                "funding-rate swap settlement_interval_hours must be > 0".to_string(),
            ));
        }
        if self.venue.trim().is_empty() {
            return Err(PricingError::InvalidInput(
                "funding-rate swap venue cannot be empty".to_string(),
            ));
        }
        if self.asset.trim().is_empty() {
            return Err(PricingError::InvalidInput(
                "funding-rate swap asset cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// All settlement times strictly after entry and up to maturity.
    pub fn settlement_schedule(&self) -> Vec<DateTime<Utc>> {
        if self.settlement_interval_hours == 0 || self.maturity <= self.entry_time {
            return Vec::new();
        }

        let interval_seconds = i64::from(self.settlement_interval_hours) * 3_600;
        let first_settlement =
            (self.entry_time.timestamp().div_euclid(interval_seconds) + 1) * interval_seconds;
        let maturity_timestamp = self.maturity.timestamp();

        let mut schedule = Vec::new();
        let mut settlement_timestamp = first_settlement;
        while settlement_timestamp <= maturity_timestamp {
            if let Some(settlement) = DateTime::from_timestamp(settlement_timestamp, 0) {
                schedule.push(settlement);
            }
            settlement_timestamp += interval_seconds;
        }
        schedule
    }

    /// PnL for one standard 8h settlement interval.
    pub fn interval_pnl(fixed_rate: f64, floating_rate: f64, notional: f64) -> f64 {
        Self::interval_pnl_with_interval_hours(
            fixed_rate,
            floating_rate,
            notional,
            DEFAULT_SETTLEMENT_INTERVAL_HOURS,
        )
    }

    /// Total realized PnL from historical settlement fixings.
    pub fn realized_pnl(&self, fixings: &[(DateTime<Utc>, f64)]) -> f64 {
        let scheduled_timestamps = self
            .settlement_schedule()
            .into_iter()
            .map(|settlement| settlement.timestamp())
            .collect::<std::collections::HashSet<_>>();

        fixings
            .iter()
            .filter(|(timestamp, _)| scheduled_timestamps.contains(&timestamp.timestamp()))
            .map(|(_, floating_rate)| self.interval_pnl_for_rate(*floating_rate))
            .sum()
    }

    /// Mark-to-market of remaining unsettled intervals using the funding curve.
    pub fn mark_to_market(
        &self,
        curve: &FundingRateCurve,
        discount_curve: Option<&YieldCurve>,
        as_of: DateTime<Utc>,
    ) -> f64 {
        self.settlement_schedule()
            .into_iter()
            .filter(|settlement| *settlement > as_of)
            .map(|settlement| {
                let interval_start =
                    settlement - Duration::hours(i64::from(self.settlement_interval_hours));
                let expected_rate = curve.expected_rate(as_of, interval_start, settlement);
                let t = (settlement.signed_duration_since(as_of).num_seconds() as f64)
                    / SECONDS_PER_YEAR;
                let discount_factor = discount_curve.map_or(1.0, |dc| dc.discount_factor(t));
                self.interval_pnl_for_rate(expected_rate) * discount_factor
            })
            .sum()
    }

    /// Fixed year fraction for one funding interval.
    pub fn interval_year_fraction(&self) -> f64 {
        f64::from(self.settlement_interval_hours) / HOURS_PER_YEAR
    }

    fn interval_pnl_for_rate(&self, floating_rate: f64) -> f64 {
        Self::interval_pnl_with_interval_hours(
            self.fixed_rate,
            floating_rate,
            self.notional,
            self.settlement_interval_hours,
        )
    }

    fn interval_pnl_with_interval_hours(
        fixed_rate: f64,
        floating_rate: f64,
        notional: f64,
        settlement_interval_hours: u32,
    ) -> f64 {
        (floating_rate - fixed_rate)
            * notional
            * (f64::from(settlement_interval_hours) / HOURS_PER_YEAR)
    }
}

impl Instrument for FundingRateSwap {
    fn instrument_type(&self) -> &str {
        "FundingRateSwap"
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use chrono::{DateTime, NaiveDate, Utc};

    use super::FundingRateSwap;
    use crate::rates::{FundingRateCurve, YieldCurve};

    fn dt(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
        DateTime::from_naive_utc_and_offset(
            NaiveDate::from_ymd_opt(year, month, day)
                .expect("valid date")
                .and_hms_opt(hour, 0, 0)
                .expect("valid hour"),
            Utc,
        )
    }

    #[test]
    fn settlement_schedule_uses_eight_hour_boundaries() {
        let swap = FundingRateSwap {
            notional: 1.0,
            fixed_rate: 0.10,
            entry_time: dt(2026, 1, 1, 1),
            maturity: dt(2026, 1, 2, 2),
            settlement_interval_hours: 8,
            venue: "Hyperliquid".to_string(),
            asset: "BTCUSDT".to_string(),
        };

        let schedule = swap.settlement_schedule();
        assert_eq!(
            schedule,
            vec![dt(2026, 1, 1, 8), dt(2026, 1, 1, 16), dt(2026, 1, 2, 0)]
        );
        assert!(schedule.windows(2).all(|w| (w[1] - w[0]).num_hours() == 8));
    }

    #[test]
    fn realized_pnl_matches_manual_sum() {
        let swap = FundingRateSwap {
            notional: 1_000.0,
            fixed_rate: 0.10,
            entry_time: dt(2026, 1, 1, 0),
            maturity: dt(2026, 1, 2, 0),
            settlement_interval_hours: 8,
            venue: "Bybit".to_string(),
            asset: "BTCUSDT".to_string(),
        };

        let fixings = vec![
            (dt(2026, 1, 1, 8), 0.12),
            (dt(2026, 1, 1, 16), 0.08),
            (dt(2026, 1, 2, 0), 0.11),
        ];

        let expected = ((0.12 - 0.10) + (0.08 - 0.10) + (0.11 - 0.10))
            * swap.notional
            * swap.interval_year_fraction();
        assert_relative_eq!(swap.realized_pnl(&fixings), expected, epsilon = 1.0e-12);
    }

    #[test]
    fn flat_curve_mtm_matches_remaining_interval_formula() {
        let swap = FundingRateSwap {
            notional: 1_000.0,
            fixed_rate: 0.10,
            entry_time: dt(2026, 1, 1, 0),
            maturity: dt(2026, 1, 2, 0),
            settlement_interval_hours: 8,
            venue: "Binance".to_string(),
            asset: "BTCUSDT".to_string(),
        };
        let curve = FundingRateCurve::flat(0.13);

        let mtm = swap.mark_to_market(&curve, None, dt(2026, 1, 1, 8));
        let expected = 2.0 * (0.13 - 0.10) * swap.notional * swap.interval_year_fraction();

        assert_relative_eq!(mtm, expected, epsilon = 1.0e-12);
    }

    #[test]
    fn mtm_applies_discount_curve_to_future_cashflows() {
        let swap = FundingRateSwap {
            notional: 1_000.0,
            fixed_rate: 0.10,
            entry_time: dt(2026, 1, 1, 0),
            maturity: dt(2026, 1, 2, 0),
            settlement_interval_hours: 8,
            venue: "Binance".to_string(),
            asset: "BTCUSDT".to_string(),
        };
        let curve = FundingRateCurve::flat(0.13);
        let interval = swap.interval_year_fraction();
        let discount_curve = YieldCurve::new(vec![(interval, 0.99), (2.0 * interval, 0.97)]);

        let mtm = swap.mark_to_market(&curve, Some(&discount_curve), dt(2026, 1, 1, 8));
        let expected_interval_pnl = (0.13 - 0.10) * swap.notional * interval;
        let expected = expected_interval_pnl * 0.99 + expected_interval_pnl * 0.97;

        assert_relative_eq!(mtm, expected, epsilon = 1.0e-12);
    }
}
