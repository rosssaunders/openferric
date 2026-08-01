//! Funding-rate term structures for crypto perpetuals.
//!
//! This module models Binance-style 8-hour funding settlements as a forward
//! curve in year-fraction time, with linear or piecewise-constant
//! interpolation between observed settlement points and exact integration of
//! the corresponding rate path (trapezoidal for linear, stepwise for
//! piecewise-constant).

use crate::math::interpolation::{
    ExtrapolationMode, Interpolator, LinearInterpolator, PiecewiseConstantInterpolator,
};
use chrono::{DateTime, Utc};

const FUNDING_PERIODS_PER_YEAR: f64 = 1095.0;
const FUNDING_INTERVAL_YEARS: f64 = 1.0 / FUNDING_PERIODS_PER_YEAR;
const MILLISECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 60.0 * 60.0 * 1000.0;

/// A single observed perpetual funding-rate fixing.
///
/// The `rate` is expressed as a per-8-hour funding rate. Curves are typically
/// built from a homogeneous series for a single `(venue, asset)` pair.
#[derive(Debug, Clone, PartialEq)]
pub struct FundingRateSnapshot {
    /// Exchange or execution venue identifier.
    pub venue: String,
    /// Asset or perpetual symbol identifier.
    pub asset: String,
    /// Funding rate for one 8-hour settlement period.
    pub rate: f64,
    /// Observation timestamp in UTC.
    pub timestamp: DateTime<Utc>,
}

/// Rolling distribution statistics for funding rates.
///
/// `vol` is the population standard deviation of per-8-hour rates, and
/// `kurtosis` is excess kurtosis.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FundingRateStats {
    /// Number of observations used in the window.
    pub window_size: usize,
    /// Arithmetic mean of the funding rates.
    pub mean: f64,
    /// Population standard deviation of the funding rates.
    pub vol: f64,
    /// Standardized third central moment.
    pub skew: f64,
    /// Excess kurtosis.
    pub kurtosis: f64,
}

impl FundingRateStats {
    /// Computes statistics for a slice of per-8-hour funding rates.
    pub fn from_rates(rates: &[f64]) -> Self {
        if rates.is_empty() {
            return Self::default();
        }

        let window_size = rates.len();
        let mean = rates.iter().sum::<f64>() / window_size as f64;

        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;

        for rate in rates {
            let centered = *rate - mean;
            let centered2 = centered * centered;
            m2 += centered2;
            m3 += centered2 * centered;
            m4 += centered2 * centered2;
        }

        m2 /= window_size as f64;
        m3 /= window_size as f64;
        m4 /= window_size as f64;

        let vol = m2.sqrt();
        let (skew, kurtosis) = if m2 <= f64::EPSILON {
            (0.0, 0.0)
        } else {
            (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
        };

        Self {
            window_size,
            mean,
            vol,
            skew,
            kurtosis,
        }
    }
}

/// Interpolation strategy for funding rate curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FundingRateInterpolation {
    /// Piecewise-linear interpolation between rate nodes.
    #[default]
    Linear,
    /// Piecewise-constant (step function): each rate holds flat until the next node.
    /// Appropriate for Boros-style markets where each swap locks a fixed APR.
    PiecewiseConstant,
}

/// Forward funding-rate term structure built from observed snapshots.
///
/// Times `t` are expressed in years from the earliest snapshot in the input
/// history. `forward_rate(t)` returns a per-8-hour rate, while
/// `cumulative_index(t)` returns total accrued funding over `[0, t]`.
pub struct FundingRateCurve {
    snapshots: Vec<FundingRateSnapshot>,
    anchor_timestamp: Option<DateTime<Utc>>,
    nodes: Vec<(f64, f64)>,
    interpolation_mode: FundingRateInterpolation,
    interpolator: Option<std::sync::Arc<dyn Interpolator + Send + Sync>>,
}

impl std::fmt::Debug for FundingRateCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FundingRateCurve")
            .field("snapshots", &self.snapshots)
            .field("anchor_timestamp", &self.anchor_timestamp)
            .field("nodes", &self.nodes)
            .field("interpolation_mode", &self.interpolation_mode)
            .field("interpolator", &self.interpolator.is_some())
            .finish()
    }
}

impl Clone for FundingRateCurve {
    fn clone(&self) -> Self {
        // All fields are immutable after construction; the interpolator is
        // shared via Arc rather than rebuilt (clones sit on bump-and-reprice
        // paths such as parallel_shifted).
        Self {
            snapshots: self.snapshots.clone(),
            anchor_timestamp: self.anchor_timestamp,
            nodes: self.nodes.clone(),
            interpolation_mode: self.interpolation_mode,
            interpolator: self.interpolator.clone(),
        }
    }
}

impl FundingRateCurve {
    /// Builds a funding curve from a snapshot history.
    ///
    /// Snapshots are sorted by timestamp, non-finite rates are removed, and
    /// duplicate timestamps keep the last supplied observation.
    pub fn new(snapshots: Vec<FundingRateSnapshot>) -> Self {
        Self::new_with_interpolation(snapshots, FundingRateInterpolation::Linear)
    }

    /// Builds a funding curve with an explicit interpolation strategy.
    pub fn new_with_interpolation(
        snapshots: Vec<FundingRateSnapshot>,
        mode: FundingRateInterpolation,
    ) -> Self {
        let snapshots = sanitize_snapshots(snapshots);
        let anchor_timestamp = snapshots
            .first()
            .map(|snapshot| snapshot.timestamp.to_owned());
        let nodes = build_nodes(&snapshots, anchor_timestamp.as_ref());
        let interpolator = build_interpolator(&nodes, mode);

        Self {
            snapshots,
            anchor_timestamp,
            nodes,
            interpolation_mode: mode,
            interpolator,
        }
    }

    /// Returns the sorted snapshot history used to build the curve.
    pub fn snapshots(&self) -> &[FundingRateSnapshot] {
        &self.snapshots
    }

    /// Returns the curve anchor timestamp.
    pub fn anchor_timestamp(&self) -> Option<DateTime<Utc>> {
        self.anchor_timestamp.as_ref().cloned()
    }

    /// Returns the internal curve nodes as `(time_in_years, per_8h_rate)`.
    pub fn nodes(&self) -> &[(f64, f64)] {
        &self.nodes
    }

    /// Returns the interpolation mode used by this curve.
    pub fn interpolation_mode(&self) -> FundingRateInterpolation {
        self.interpolation_mode
    }

    /// Converts a per-8-hour funding rate to an annualized APR.
    #[inline]
    pub fn per_period_rate_to_apr(rate: f64) -> f64 {
        rate * FUNDING_PERIODS_PER_YEAR
    }

    /// Converts an annualized APR to a per-8-hour funding rate.
    #[inline]
    pub fn apr_to_per_period_rate(apr: f64) -> f64 {
        apr / FUNDING_PERIODS_PER_YEAR
    }

    /// Returns the standard 8-hour funding interval in years.
    #[inline]
    pub fn settlement_interval_years() -> f64 {
        FUNDING_INTERVAL_YEARS
    }

    /// Returns the interpolated per-8-hour funding rate at time `t`.
    pub fn forward_rate(&self, t: f64) -> f64 {
        if !t.is_finite() {
            return f64::NAN;
        }
        if self.nodes.is_empty() {
            return 0.0;
        }
        if let Some(interpolator) = &self.interpolator {
            return interpolator.value(t.max(0.0)).unwrap_or(f64::NAN);
        }
        self.nodes[0].1
    }

    /// Returns the cumulative funding index accrued from the anchor to time `t`.
    pub fn cumulative_index(&self, t: f64) -> f64 {
        if !t.is_finite() || t <= 0.0 || self.nodes.is_empty() {
            return 0.0;
        }

        FUNDING_PERIODS_PER_YEAR * self.integrated_forward_rate(t)
    }

    /// Returns the funding discount factor `exp(-cumulative_index(t))`.
    pub fn discount_factor(&self, t: f64) -> f64 {
        (-self.cumulative_index(t)).exp()
    }

    /// Builds a flat funding-rate curve from a constant annualised APR.
    ///
    /// This is a convenience constructor for pricing and risk calculations
    /// that don't require a full historical snapshot series.
    pub fn flat(apr: f64) -> Self {
        let per_period = Self::apr_to_per_period_rate(apr);
        let snapshot = FundingRateSnapshot {
            venue: "synthetic".to_string(),
            asset: "flat".to_string(),
            rate: per_period,
            timestamp: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        };
        Self::new(vec![snapshot])
    }

    /// Returns a copy with a parallel shift applied to all forward rates (in APR terms).
    pub fn parallel_shifted(&self, bump_apr: f64) -> Self {
        let shifted: Vec<FundingRateSnapshot> = self
            .snapshots
            .iter()
            .map(|s| FundingRateSnapshot {
                venue: s.venue.clone(),
                asset: s.asset.clone(),
                rate: s.rate + Self::apr_to_per_period_rate(bump_apr),
                timestamp: s.timestamp,
            })
            .collect();
        if shifted.is_empty() {
            return self.clone();
        }
        Self::new_with_interpolation(shifted, self.interpolation_mode)
    }

    /// Identity transform retained for API compatibility.
    ///
    /// A funding forward curve stores conditional expected rates, not a
    /// volatility state. Consequently a volatility bump is undefined at the
    /// curve level and this method deliberately leaves the curve unchanged.
    /// Linear funding swaps have exact zero vega; nonlinear funding options
    /// require a separate stochastic model rather than this method.
    pub fn volatility_shifted(&self, _bump: f64) -> Self {
        self.clone()
    }

    /// Expected funding rate for a future interval, in APR terms.
    ///
    /// For the snapshot-based curve this returns the interpolated forward rate
    /// converted to APR. The `as_of` parameter is accepted for API compatibility
    /// but does not affect the result.
    pub fn expected_rate(
        &self,
        _as_of: DateTime<Utc>,
        start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> f64 {
        let t = self
            .anchor_timestamp
            .map(|anchor| (start - anchor).num_milliseconds() as f64 / MILLISECONDS_PER_YEAR)
            .unwrap_or(0.0);
        Self::per_period_rate_to_apr(self.forward_rate(t.max(0.0)))
    }

    /// Returns rolling stats keyed by the last timestamp in each full window.
    pub fn rolling_stats(&self, window_size: usize) -> Vec<(DateTime<Utc>, FundingRateStats)> {
        if window_size == 0 || self.snapshots.len() < window_size {
            return Vec::new();
        }

        self.snapshots
            .windows(window_size)
            .map(|window| {
                let rates = window
                    .iter()
                    .map(|snapshot| snapshot.rate)
                    .collect::<Vec<_>>();
                (
                    window[window.len() - 1].timestamp,
                    FundingRateStats::from_rates(&rates),
                )
            })
            .collect()
    }

    fn integrated_forward_rate(&self, t: f64) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        if self.nodes.len() == 1 {
            return self.nodes[0].1 * t.max(0.0);
        }

        let horizon = t.max(0.0);
        let mut area = 0.0;

        for window in self.nodes.windows(2) {
            let (start_t, start_rate) = window[0];
            let (end_t, end_rate) = window[1];

            if horizon <= start_t {
                return area;
            }

            let upper = horizon.min(end_t);
            if upper > start_t {
                area += match self.interpolation_mode {
                    // Trapezoid under the piecewise-linear rate path.
                    FundingRateInterpolation::Linear => {
                        let upper_rate = linear_rate(start_t, end_t, start_rate, end_rate, upper);
                        0.5 * (start_rate + upper_rate) * (upper - start_t)
                    }
                    // Step function: the node rate holds flat until the next
                    // node, matching `forward_rate` for piecewise-constant
                    // curves.
                    FundingRateInterpolation::PiecewiseConstant => start_rate * (upper - start_t),
                };
            }

            if horizon <= end_t {
                return area;
            }
        }

        let (last_t, last_rate) = self.nodes[self.nodes.len() - 1];
        area + last_rate * (horizon - last_t)
    }
}

/// Weighted composite funding curve across multiple venues.
///
/// Weights are normalized by the sum of positive finite weights when
/// aggregating forward rates or cumulative indices.
#[derive(Debug, Clone, Default)]
pub struct MultiVenueFundingCurve {
    curves: Vec<(FundingRateCurve, f64)>,
}

impl MultiVenueFundingCurve {
    /// Builds a weighted multi-venue funding curve.
    pub fn new(curves: Vec<(FundingRateCurve, f64)>) -> Self {
        let curves = curves
            .into_iter()
            .filter(|(_, weight)| weight.is_finite() && *weight > 0.0)
            .collect();
        Self { curves }
    }

    /// Returns the configured `(curve, weight)` pairs.
    pub fn curves(&self) -> &[(FundingRateCurve, f64)] {
        &self.curves
    }

    /// Returns the weighted-average forward funding rate at time `t`.
    pub fn forward_rate(&self, t: f64) -> f64 {
        weighted_average(
            self.curves
                .iter()
                .map(|(curve, weight)| (*weight, curve.forward_rate(t))),
        )
    }

    /// Returns the weighted-average cumulative funding index at time `t`.
    pub fn cumulative_index(&self, t: f64) -> f64 {
        weighted_average(
            self.curves
                .iter()
                .map(|(curve, weight)| (*weight, curve.cumulative_index(t))),
        )
    }

    /// Returns the discount factor implied by the weighted cumulative index.
    pub fn discount_factor(&self, t: f64) -> f64 {
        (-self.cumulative_index(t)).exp()
    }
}

fn sanitize_snapshots(mut snapshots: Vec<FundingRateSnapshot>) -> Vec<FundingRateSnapshot> {
    snapshots.retain(|snapshot| snapshot.rate.is_finite());
    snapshots.sort_by_key(|snapshot| snapshot.timestamp);

    let mut out: Vec<FundingRateSnapshot> = Vec::with_capacity(snapshots.len());
    for snapshot in snapshots {
        if let Some(last) = out.last_mut()
            && last.timestamp == snapshot.timestamp
        {
            *last = snapshot;
            continue;
        }
        out.push(snapshot);
    }
    out
}

fn build_nodes(
    snapshots: &[FundingRateSnapshot],
    anchor_timestamp: Option<&DateTime<Utc>>,
) -> Vec<(f64, f64)> {
    let Some(anchor_timestamp) = anchor_timestamp else {
        return Vec::new();
    };

    snapshots
        .iter()
        .map(|snapshot| {
            (
                year_fraction_between(anchor_timestamp, &snapshot.timestamp),
                snapshot.rate,
            )
        })
        .collect()
}

fn build_interpolator(
    nodes: &[(f64, f64)],
    mode: FundingRateInterpolation,
) -> Option<std::sync::Arc<dyn Interpolator + Send + Sync>> {
    if nodes.len() < 2 {
        return None;
    }

    let x = nodes.iter().map(|(time, _)| *time).collect::<Vec<_>>();
    let y = nodes.iter().map(|(_, rate)| *rate).collect::<Vec<_>>();
    match mode {
        FundingRateInterpolation::Linear => LinearInterpolator::new(x, y, ExtrapolationMode::Flat)
            .ok()
            .map(|i| std::sync::Arc::new(i) as std::sync::Arc<dyn Interpolator + Send + Sync>),
        FundingRateInterpolation::PiecewiseConstant => {
            PiecewiseConstantInterpolator::new(x, y, ExtrapolationMode::Flat)
                .ok()
                .map(|i| std::sync::Arc::new(i) as std::sync::Arc<dyn Interpolator + Send + Sync>)
        }
    }
}

fn year_fraction_between(start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    (end.signed_duration_since(*start).num_milliseconds() as f64) / MILLISECONDS_PER_YEAR
}

fn linear_rate(start_t: f64, end_t: f64, start_rate: f64, end_rate: f64, t: f64) -> f64 {
    if (end_t - start_t).abs() <= f64::EPSILON {
        return end_rate;
    }
    let weight = (t - start_t) / (end_t - start_t);
    start_rate + weight * (end_rate - start_rate)
}

fn weighted_average<I>(values: I) -> f64
where
    I: Iterator<Item = (f64, f64)>,
{
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (weight, value) in values {
        weighted_sum += weight * value;
        total_weight += weight;
    }

    if total_weight <= 0.0 {
        0.0
    } else {
        weighted_sum / total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FUNDING_INTERVAL_YEARS, FUNDING_PERIODS_PER_YEAR, FundingRateCurve,
        FundingRateInterpolation, FundingRateSnapshot, FundingRateStats, MultiVenueFundingCurve,
    };
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    #[test]
    fn round_trip_forward_rate_matches_input_nodes() {
        let curve = FundingRateCurve::new(vec![
            snapshot("binance", "BTCUSDT", 0.0001, 2026, 3, 30, 0),
            snapshot("binance", "BTCUSDT", 0.0002, 2026, 3, 30, 8),
            snapshot("binance", "BTCUSDT", -0.0001, 2026, 3, 30, 16),
        ]);

        assert_relative_eq!(curve.forward_rate(0.0), 0.0001, epsilon = 1.0e-12);
        assert_relative_eq!(
            curve.forward_rate(FUNDING_INTERVAL_YEARS),
            0.0002,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            curve.forward_rate(2.0 * FUNDING_INTERVAL_YEARS),
            -0.0001,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            curve.forward_rate(0.5 * FUNDING_INTERVAL_YEARS),
            0.00015,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn cumulative_index_matches_manual_sum() {
        let curve = FundingRateCurve::new(vec![
            snapshot("binance", "ETHUSDT", 0.0001, 2026, 3, 30, 0),
            snapshot("binance", "ETHUSDT", 0.0003, 2026, 3, 30, 8),
            snapshot("binance", "ETHUSDT", 0.0005, 2026, 3, 30, 16),
        ]);

        let manual = 0.5 * (0.0001 + 0.0003) + 0.5 * (0.0003 + 0.0005);
        assert_relative_eq!(
            curve.cumulative_index(2.0 * FUNDING_INTERVAL_YEARS),
            manual,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn multi_venue_curve_uses_weighted_average() {
        let venue_a = FundingRateCurve::new(vec![
            snapshot("binance", "BTCUSDT", 0.0001, 2026, 3, 30, 0),
            snapshot("binance", "BTCUSDT", 0.0001, 2026, 3, 30, 8),
        ]);
        let venue_b = FundingRateCurve::new(vec![
            snapshot("bybit", "BTCUSDT", 0.0003, 2026, 3, 30, 0),
            snapshot("bybit", "BTCUSDT", 0.0003, 2026, 3, 30, 8),
        ]);
        let curve = MultiVenueFundingCurve::new(vec![(venue_a, 1.0), (venue_b, 3.0)]);

        assert_relative_eq!(curve.forward_rate(0.0), 0.00025, epsilon = 1.0e-12);
        assert_relative_eq!(
            curve.cumulative_index(FUNDING_INTERVAL_YEARS),
            0.00025,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn apr_conversion_is_consistent() {
        let rate = 0.0001;
        let apr = FundingRateCurve::per_period_rate_to_apr(rate);

        assert_relative_eq!(apr, 0.1095, epsilon = 1.0e-12);
        assert_relative_eq!(
            FundingRateCurve::apr_to_per_period_rate(apr),
            rate,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn handles_empty_single_point_and_duplicate_timestamps() {
        let empty = FundingRateCurve::new(Vec::new());
        assert_relative_eq!(empty.forward_rate(0.5), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(empty.cumulative_index(0.5), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(empty.discount_factor(0.5), 1.0, epsilon = 1.0e-12);

        let single =
            FundingRateCurve::new(vec![snapshot("binance", "SOLUSDT", 0.0002, 2026, 3, 30, 0)]);
        assert_relative_eq!(single.forward_rate(5.0), 0.0002, epsilon = 1.0e-12);
        assert_relative_eq!(
            single.cumulative_index(2.0 * FUNDING_INTERVAL_YEARS),
            0.0004,
            epsilon = 1.0e-12
        );

        let deduped = FundingRateCurve::new(vec![
            snapshot("binance", "SOLUSDT", 0.0001, 2026, 3, 30, 0),
            snapshot("binance", "SOLUSDT", 0.0004, 2026, 3, 30, 0),
            snapshot("binance", "SOLUSDT", 0.0002, 2026, 3, 30, 8),
        ]);
        assert_eq!(deduped.nodes().len(), 2);
        assert_relative_eq!(deduped.forward_rate(0.0), 0.0004, epsilon = 1.0e-12);
    }

    #[test]
    fn rolling_stats_cover_mean_vol_and_shape() {
        let curve = FundingRateCurve::new(vec![
            snapshot("binance", "BTCUSDT", 0.0, 2026, 3, 30, 0),
            snapshot("binance", "BTCUSDT", 1.0, 2026, 3, 30, 8),
            snapshot("binance", "BTCUSDT", 2.0, 2026, 3, 30, 16),
        ]);

        let stats = curve.rolling_stats(3);
        assert_eq!(stats.len(), 1);
        assert_eq!(
            stats[0].1,
            FundingRateStats {
                window_size: 3,
                mean: 1.0,
                vol: (2.0_f64 / 3.0).sqrt(),
                skew: 0.0,
                kurtosis: -1.5,
            }
        );
    }

    #[test]
    fn piecewise_constant_holds_rate_flat_between_nodes() {
        let snaps = vec![
            FundingRateSnapshot {
                venue: "binance".into(),
                asset: "BTCUSDT".into(),
                rate: 0.001,
                timestamp: Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            },
            FundingRateSnapshot {
                venue: "binance".into(),
                asset: "BTCUSDT".into(),
                rate: 0.003,
                timestamp: Utc.with_ymd_and_hms(2025, 4, 1, 0, 0, 0).unwrap(),
            },
        ];
        let curve = FundingRateCurve::new_with_interpolation(
            snaps,
            FundingRateInterpolation::PiecewiseConstant,
        );
        // Midpoint between nodes should return first node's rate (step function)
        let midpoint = 0.125; // ~halfway between 0 and 0.25 years
        let rate = curve.forward_rate(midpoint);
        assert!(
            (rate - 0.001).abs() < 1e-10,
            "piecewise constant should hold first rate flat, got {rate}"
        );
        // At the second node, should return second rate
        let at_second = 0.25; // approximately where the second node is
        let rate2 = curve.forward_rate(at_second);
        // The exact time depends on node computation, but after the jump it should be 0.003
        assert!(
            rate2 > 0.001,
            "should have jumped to higher rate at second node"
        );
    }

    #[test]
    fn parallel_shift_preserves_piecewise_constant_interpolation_exactly() {
        let curve = FundingRateCurve::new_with_interpolation(
            vec![
                snapshot("binance", "BTCUSDT", 0.001, 2025, 1, 1, 0),
                snapshot("binance", "BTCUSDT", 0.005, 2025, 7, 1, 0),
            ],
            FundingRateInterpolation::PiecewiseConstant,
        );
        let bump_apr = 0.1095; // exactly 0.0001 per eight-hour period
        let shifted = curve.parallel_shifted(bump_apr);

        assert_eq!(
            shifted.interpolation_mode(),
            FundingRateInterpolation::PiecewiseConstant
        );
        for time in [0.0, 0.1, 0.25, 0.49, 0.75] {
            assert_relative_eq!(
                shifted.forward_rate(time),
                curve.forward_rate(time) + 0.0001,
                epsilon = 2.0e-16
            );
        }
    }

    #[test]
    fn step_curve_discount_factor_consistent_with_forward_rate() {
        // For a piecewise-constant curve, cumulative_index/discount_factor must
        // integrate the same step function that forward_rate returns:
        // DF(t) = exp(-PERIODS_PER_YEAR * sum_i r_i * dt_i), so discount
        // factors telescope with the flat forward rate over each segment.
        let curve = FundingRateCurve::new_with_interpolation(
            vec![
                snapshot("binance", "BTCUSDT", 0.001, 2025, 1, 1, 0),
                snapshot("binance", "BTCUSDT", 0.003, 2025, 4, 1, 0),
                snapshot("binance", "BTCUSDT", 0.002, 2025, 7, 1, 0),
            ],
            FundingRateInterpolation::PiecewiseConstant,
        );
        let nodes = curve.nodes().to_vec();
        let (t1, r0) = (nodes[1].0, nodes[0].1);
        let (t2, r1) = (nodes[2].0, nodes[1].1);
        let r2 = nodes[2].1;

        // Within one segment the forward rate is flat, so
        // DF(tb)/DF(ta) = exp(-PERIODS_PER_YEAR * forward_rate(ta) * (tb - ta)).
        for (ta, tb) in [
            (0.2 * t1, 0.7 * t1),
            (t1 + 0.1 * (t2 - t1), t1 + 0.9 * (t2 - t1)),
            (t2 + 0.05, t2 + 0.20),
        ] {
            let lhs = curve.discount_factor(tb) / curve.discount_factor(ta);
            let rhs = (-FUNDING_PERIODS_PER_YEAR * curve.forward_rate(ta) * (tb - ta)).exp();
            assert_relative_eq!(lhs, rhs, epsilon = 1.0e-12);
        }

        // Across segments the integral telescopes: exp(-integral) over [0, T].
        let horizon = t2 + 0.1;
        let integral = r0 * t1 + r1 * (t2 - t1) + r2 * (horizon - t2);
        assert_relative_eq!(
            curve.cumulative_index(horizon),
            FUNDING_PERIODS_PER_YEAR * integral,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            curve.discount_factor(horizon),
            (-FUNDING_PERIODS_PER_YEAR * integral).exp(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn linear_interpolation_differs_from_piecewise_constant() {
        let snaps = vec![
            FundingRateSnapshot {
                venue: "binance".into(),
                asset: "BTCUSDT".into(),
                rate: 0.001,
                timestamp: Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            },
            FundingRateSnapshot {
                venue: "binance".into(),
                asset: "BTCUSDT".into(),
                rate: 0.005,
                timestamp: Utc.with_ymd_and_hms(2025, 7, 1, 0, 0, 0).unwrap(),
            },
        ];
        let linear = FundingRateCurve::new(snaps.clone());
        let step = FundingRateCurve::new_with_interpolation(
            snaps,
            FundingRateInterpolation::PiecewiseConstant,
        );
        let mid = 0.25; // quarter way between nodes
        let linear_rate = linear.forward_rate(mid);
        let step_rate = step.forward_rate(mid);
        // Linear should interpolate between 0.001 and 0.005
        // Step should hold at 0.001
        assert!(
            (linear_rate - step_rate).abs() > 1e-6,
            "linear and step should differ at midpoint: linear={linear_rate}, step={step_rate}"
        );
        assert!(
            (step_rate - 0.001).abs() < 1e-10,
            "step should hold first rate: got {step_rate}"
        );
    }

    fn snapshot(
        venue: &str,
        asset: &str,
        rate: f64,
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
    ) -> FundingRateSnapshot {
        FundingRateSnapshot {
            venue: venue.to_string(),
            asset: asset.to_string(),
            rate,
            timestamp: Utc
                .with_ymd_and_hms(year, month, day, hour, 0, 0)
                .single()
                .expect("valid UTC timestamp"),
        }
    }
}
