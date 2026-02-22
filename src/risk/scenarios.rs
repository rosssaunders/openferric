//! Scenario and stress-testing engine for portfolio-level PnL attribution.
//!
//! This module provides a serializable scenario-definition layer plus an execution engine for
//! running `N scenarios x M trades` attribution batches.
//!
//! Core capabilities:
//! - Scenario definitions via [`ScenarioDefinition`] with serde JSON support.
//! - Scenario families: historical replay, hypothetical shocks, 2D parametric stress grids,
//!   and reverse stress calibrated to a target loss.
//! - PnL explain decomposition per trade and scenario:
//!   `theta + delta + gamma + vega + cross_gamma_vanna + unexplained`.
//! - Market snapshot diffing and day-over-day attribution using [`MarketSnapshotDiff`].
//! - Result table output at scenario x trade x PnL component granularity.
//! - Heatmap extraction for 2D stress grids.
//!
//! Numerical notes:
//! - Delta/Gamma/Vega/Theta are interpreted as local sensitivities around the base state.
//! - `cross_gamma` is applied to `dS * (dr + dspread)` and `vanna` to `dS * dvol`.
//! - Reverse stress solves for a scalar multiplier on a seed shock via bounded bisection.
//! - For large shocks or highly non-linear payoffs, use [`run_scenario_batch_with_pricer`]
//!   to provide full revaluation PnL and inspect the unexplained residual.

use std::collections::BTreeMap;

use crate::core::{Greeks, PricingError};
use crate::market::{CreditCurveSnapshot, Market, MarketSnapshot, SampledVolSurface, VolSource};

use super::portfolio::Position;

const EPS: f64 = 1.0e-12;
const DOD_HORIZON_YEARS: f64 = 1.0 / 252.0;

/// Scenario category used by resolved scenarios and result rows.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioKind {
    HistoricalReplay,
    Hypothetical,
    ParametricStress2d,
    ReverseStress,
}

/// Shock factor axis used by parametric stress definitions.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ShockFactor {
    Spot,
    Vol,
    Rate,
    CreditSpread,
}

/// Serializable market shock vector used by all scenario definitions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct MarketShock {
    /// Relative spot move, e.g. `-0.10` for -10%.
    pub spot_shock_pct: f64,
    /// Relative implied-vol move, e.g. `0.25` for +25%.
    pub vol_shock_pct: f64,
    /// Absolute rate move in decimal (100 bp = `0.01`).
    pub rate_shock_abs: f64,
    /// Absolute credit-spread/hazard move in decimal.
    pub credit_spread_shock_abs: f64,
    /// Time horizon applied to theta in years.
    pub horizon_years: f64,
}

impl MarketShock {
    /// Returns this shock scaled by `multiplier`.
    #[inline]
    pub fn scaled(self, multiplier: f64) -> Self {
        Self {
            spot_shock_pct: self.spot_shock_pct * multiplier,
            vol_shock_pct: self.vol_shock_pct * multiplier,
            rate_shock_abs: self.rate_shock_abs * multiplier,
            credit_spread_shock_abs: self.credit_spread_shock_abs * multiplier,
            horizon_years: self.horizon_years * multiplier,
        }
    }

    /// Adds a factor-specific increment to the shock vector.
    #[inline]
    pub fn add_factor(&mut self, factor: ShockFactor, amount: f64) {
        match factor {
            ShockFactor::Spot => self.spot_shock_pct += amount,
            ShockFactor::Vol => self.vol_shock_pct += amount,
            ShockFactor::Rate => self.rate_shock_abs += amount,
            ShockFactor::CreditSpread => self.credit_spread_shock_abs += amount,
        }
    }

    /// Returns the current value for the requested factor.
    #[inline]
    pub fn factor_value(self, factor: ShockFactor) -> f64 {
        match factor {
            ShockFactor::Spot => self.spot_shock_pct,
            ShockFactor::Vol => self.vol_shock_pct,
            ShockFactor::Rate => self.rate_shock_abs,
            ShockFactor::CreditSpread => self.credit_spread_shock_abs,
        }
    }
}

/// Historical replay scenario definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistoricalReplayDefinition {
    pub scenario_id: String,
    pub replay_date: Option<String>,
    pub shock: MarketShock,
}

/// Hypothetical scenario definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HypotheticalScenarioDefinition {
    pub scenario_id: String,
    pub description: Option<String>,
    pub shock: MarketShock,
}

/// One axis of a 2D stress grid.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StressAxis {
    pub factor: ShockFactor,
    pub shocks: Vec<f64>,
}

/// 2D parametric stress-grid definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParametricStress2dDefinition {
    pub scenario_id: String,
    pub x_axis: StressAxis,
    pub y_axis: StressAxis,
    #[serde(default)]
    pub base_shock: MarketShock,
}

/// Reverse-stress definition solved against a target portfolio loss.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReverseStressDefinition {
    pub scenario_id: String,
    /// Target loss in PnL currency units, provided as a positive number.
    pub target_loss: f64,
    /// Seed move direction and relative factor sizing.
    pub seed_shock: MarketShock,
    /// Upper bound for the scalar multiplier on `seed_shock`.
    #[serde(default = "default_reverse_max_scale")]
    pub max_scale: f64,
    /// Absolute PnL tolerance for bisection solve.
    #[serde(default = "default_reverse_tolerance")]
    pub tolerance: f64,
    /// Maximum bisection iterations.
    #[serde(default = "default_reverse_max_iterations")]
    pub max_iterations: u32,
}

const fn default_reverse_max_scale() -> f64 {
    10.0
}

const fn default_reverse_tolerance() -> f64 {
    1.0e-4
}

const fn default_reverse_max_iterations() -> u32 {
    64
}

/// Serializable top-level scenario definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "scenario_type", rename_all = "snake_case")]
pub enum ScenarioDefinition {
    HistoricalReplay(HistoricalReplayDefinition),
    Hypothetical(HypotheticalScenarioDefinition),
    ParametricStress2d(ParametricStress2dDefinition),
    ReverseStress(ReverseStressDefinition),
}

/// Parametric-grid coordinate for a resolved stress scenario.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StressGridPoint {
    pub x_factor: ShockFactor,
    pub y_factor: ShockFactor,
    pub x_value: f64,
    pub y_value: f64,
    pub x_index: usize,
    pub y_index: usize,
}

/// Scenario row after expanding any grids and solving reverse stress.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedScenario {
    pub scenario_id: String,
    pub parent_scenario_id: String,
    pub kind: ScenarioKind,
    pub shock: MarketShock,
    pub grid_point: Option<StressGridPoint>,
    pub reverse_stress_scale: Option<f64>,
}

/// Trade payload used by the scenario engine.
#[derive(Debug, Clone)]
pub struct ScenarioTrade<I> {
    pub trade_id: String,
    pub instrument: I,
    pub quantity: f64,
    pub greeks: Greeks,
    pub spot: f64,
    pub implied_vol: f64,
    /// Mixed derivative `d^2V / dS dVol`.
    pub vanna: f64,
    /// Generic cross-gamma loaded to `dS * (dr + dspread)`.
    pub cross_gamma: f64,
    /// First-order credit spread sensitivity.
    pub credit_delta: f64,
}

impl<I> ScenarioTrade<I> {
    /// Constructs a trade from standard Greeks and base market levels.
    pub fn new(
        trade_id: impl Into<String>,
        instrument: I,
        quantity: f64,
        greeks: Greeks,
        spot: f64,
        implied_vol: f64,
    ) -> Self {
        assert!(
            spot.is_finite() && spot > 0.0,
            "spot must be finite and > 0"
        );
        assert!(
            implied_vol.is_finite() && implied_vol >= 0.0,
            "implied_vol must be finite and >= 0"
        );

        Self {
            trade_id: trade_id.into(),
            instrument,
            quantity,
            greeks,
            spot,
            implied_vol,
            vanna: 0.0,
            cross_gamma: 0.0,
            credit_delta: 0.0,
        }
    }

    /// Sets `vanna` and `cross_gamma` terms.
    #[inline]
    pub fn with_cross_terms(mut self, vanna: f64, cross_gamma: f64) -> Self {
        self.vanna = vanna;
        self.cross_gamma = cross_gamma;
        self
    }

    /// Sets first-order credit spread sensitivity.
    #[inline]
    pub fn with_credit_delta(mut self, credit_delta: f64) -> Self {
        self.credit_delta = credit_delta;
        self
    }

    /// Converts from an existing [`Position`] with defaulted cross terms.
    pub fn from_position(trade_id: impl Into<String>, position: Position<I>) -> Self {
        Self::new(
            trade_id,
            position.instrument,
            position.quantity,
            position.greeks,
            position.spot,
            position.implied_vol,
        )
    }
}

/// Explained PnL component set.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ExplainedPnlComponents {
    pub theta: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub cross_gamma_vanna: f64,
}

impl ExplainedPnlComponents {
    /// Sum of explained terms.
    #[inline]
    pub fn explained(self) -> f64 {
        self.theta + self.delta + self.gamma + self.vega + self.cross_gamma_vanna
    }
}

/// Scenario x trade output row with explained and unexplained PnL.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioTradePnlRow {
    pub scenario_id: String,
    pub parent_scenario_id: String,
    pub scenario_kind: ScenarioKind,
    pub trade_id: String,
    pub theta: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub cross_gamma_vanna: f64,
    pub explained_pnl: f64,
    pub observed_pnl: f64,
    pub unexplained_pnl: f64,
    pub unexplained_ratio: f64,
}

/// Aggregated portfolio attribution row per scenario.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioPortfolioPnlRow {
    pub scenario_id: String,
    pub parent_scenario_id: String,
    pub scenario_kind: ScenarioKind,
    pub theta: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub cross_gamma_vanna: f64,
    pub explained_pnl: f64,
    pub observed_pnl: f64,
    pub unexplained_pnl: f64,
    pub unexplained_ratio: f64,
}

/// Scenario result table at scenario x trade granularity.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ScenarioResultTable {
    pub rows: Vec<ScenarioTradePnlRow>,
}

impl ScenarioResultTable {
    /// Aggregates to one row per scenario.
    pub fn portfolio_rows(&self) -> Vec<ScenarioPortfolioPnlRow> {
        let mut grouped: BTreeMap<(String, String, ScenarioKind), ScenarioPortfolioPnlRow> =
            BTreeMap::new();

        for row in &self.rows {
            let key = (
                row.scenario_id.clone(),
                row.parent_scenario_id.clone(),
                row.scenario_kind,
            );
            let entry = grouped
                .entry(key)
                .or_insert_with(|| ScenarioPortfolioPnlRow {
                    scenario_id: row.scenario_id.clone(),
                    parent_scenario_id: row.parent_scenario_id.clone(),
                    scenario_kind: row.scenario_kind,
                    theta: 0.0,
                    delta: 0.0,
                    gamma: 0.0,
                    vega: 0.0,
                    cross_gamma_vanna: 0.0,
                    explained_pnl: 0.0,
                    observed_pnl: 0.0,
                    unexplained_pnl: 0.0,
                    unexplained_ratio: 0.0,
                });

            entry.theta += row.theta;
            entry.delta += row.delta;
            entry.gamma += row.gamma;
            entry.vega += row.vega;
            entry.cross_gamma_vanna += row.cross_gamma_vanna;
            entry.explained_pnl += row.explained_pnl;
            entry.observed_pnl += row.observed_pnl;
            entry.unexplained_pnl += row.unexplained_pnl;
        }

        let mut out: Vec<_> = grouped.into_values().collect();
        for row in &mut out {
            row.unexplained_ratio = if row.observed_pnl.abs() > EPS {
                row.unexplained_pnl.abs() / row.observed_pnl.abs()
            } else {
                0.0
            };
        }
        out
    }
}

/// Full scenario run output with resolved scenarios and result table.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioRunResult {
    pub resolved_scenarios: Vec<ResolvedScenario>,
    pub table: ScenarioResultTable,
}

impl ScenarioRunResult {
    /// Extracts a 2D heatmap for a parametric stress parent scenario id.
    pub fn stress_heatmap(&self, parent_scenario_id: &str) -> Option<StressHeatmap2d> {
        let grid_scenarios: Vec<&ResolvedScenario> = self
            .resolved_scenarios
            .iter()
            .filter(|s| s.parent_scenario_id == parent_scenario_id)
            .filter(|s| s.kind == ScenarioKind::ParametricStress2d)
            .filter(|s| s.grid_point.is_some())
            .collect();

        if grid_scenarios.is_empty() {
            return None;
        }

        let first = grid_scenarios[0].grid_point.as_ref()?;
        let x_factor = first.x_factor;
        let y_factor = first.y_factor;

        let x_len = grid_scenarios
            .iter()
            .filter_map(|s| s.grid_point.as_ref().map(|p| p.x_index))
            .max()?
            + 1;
        let y_len = grid_scenarios
            .iter()
            .filter_map(|s| s.grid_point.as_ref().map(|p| p.y_index))
            .max()?
            + 1;

        let mut x_values = vec![0.0; x_len];
        let mut y_values = vec![0.0; y_len];
        let mut pnl = vec![vec![0.0; x_len]; y_len];

        let totals: BTreeMap<String, f64> = self
            .table
            .portfolio_rows()
            .into_iter()
            .map(|row| (row.scenario_id, row.observed_pnl))
            .collect();

        for scenario in &grid_scenarios {
            let point = scenario
                .grid_point
                .as_ref()
                .expect("grid scenarios should carry coordinates");
            if point.x_factor != x_factor || point.y_factor != y_factor {
                return None;
            }

            x_values[point.x_index] = point.x_value;
            y_values[point.y_index] = point.y_value;
            pnl[point.y_index][point.x_index] =
                totals.get(&scenario.scenario_id).copied().unwrap_or(0.0);
        }

        Some(StressHeatmap2d {
            parent_scenario_id: parent_scenario_id.to_string(),
            x_factor,
            y_factor,
            x_values,
            y_values,
            pnl,
        })
    }
}

/// 2D stress-grid heatmap data container.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StressHeatmap2d {
    pub parent_scenario_id: String,
    pub x_factor: ShockFactor,
    pub y_factor: ShockFactor,
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    /// Matrix shape: `[y_index][x_index]`.
    pub pnl: Vec<Vec<f64>>,
}

/// Per-market diff between two snapshots.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarketLevelDiff {
    pub market_id: String,
    pub spot_pct_change: f64,
    pub rate_abs_change: f64,
    pub dividend_yield_abs_change: f64,
    pub atm_vol_pct_change: f64,
}

/// Per-asset spot price diff.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpotPriceDiff {
    pub asset_id: String,
    pub spot_pct_change: f64,
}

/// Approximate parallel shift summary for a yield curve.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YieldCurveDiff {
    pub curve_id: String,
    pub parallel_shift_abs: f64,
    pub max_abs_shift: f64,
}

/// Approximate hazard/spread shift summary for a credit curve.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CreditCurveDiff {
    pub curve_id: String,
    pub avg_hazard_shift_abs: f64,
    pub max_hazard_shift_abs: f64,
}

/// Snapshot-level market diff used for day-over-day scenario attribution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarketSnapshotDiff {
    pub from_snapshot_id: String,
    pub to_snapshot_id: String,
    pub market_diffs: Vec<MarketLevelDiff>,
    pub spot_price_diffs: Vec<SpotPriceDiff>,
    pub yield_curve_diffs: Vec<YieldCurveDiff>,
    pub credit_curve_diffs: Vec<CreditCurveDiff>,
}

impl MarketSnapshotDiff {
    /// Converts detailed diffs to an aggregate shock vector.
    pub fn to_market_shock(&self, horizon_years: f64) -> MarketShock {
        let mut spot_moves = Vec::new();
        let mut vol_moves = Vec::new();
        let mut rate_moves = Vec::new();
        let mut credit_moves = Vec::new();

        for diff in &self.market_diffs {
            spot_moves.push(diff.spot_pct_change);
            vol_moves.push(diff.atm_vol_pct_change);
            rate_moves.push(diff.rate_abs_change);
        }
        for diff in &self.spot_price_diffs {
            spot_moves.push(diff.spot_pct_change);
        }
        for diff in &self.yield_curve_diffs {
            rate_moves.push(diff.parallel_shift_abs);
        }
        for diff in &self.credit_curve_diffs {
            credit_moves.push(diff.avg_hazard_shift_abs);
        }

        MarketShock {
            spot_shock_pct: mean(&spot_moves),
            vol_shock_pct: mean(&vol_moves),
            rate_shock_abs: mean(&rate_moves),
            credit_spread_shock_abs: mean(&credit_moves),
            horizon_years,
        }
    }
}

/// Day-over-day attribution output bundle.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DayOverDayAttribution {
    pub market_diff: MarketSnapshotDiff,
    pub scenario: ResolvedScenario,
    pub table: ScenarioResultTable,
    pub portfolio: ScenarioPortfolioPnlRow,
}

/// Computes explained PnL components for one trade under one shock.
pub fn explained_pnl_components<I>(
    trade: &ScenarioTrade<I>,
    shock: &MarketShock,
) -> ExplainedPnlComponents {
    let ds = trade.spot * shock.spot_shock_pct;
    let dvol = trade.implied_vol * shock.vol_shock_pct;
    let dr_plus_spread = shock.rate_shock_abs + shock.credit_spread_shock_abs;

    let theta = trade.quantity * trade.greeks.theta * shock.horizon_years;
    let delta = trade.quantity
        * (trade.greeks.delta * ds
            + trade.greeks.rho * shock.rate_shock_abs
            + trade.credit_delta * shock.credit_spread_shock_abs);
    let gamma = 0.5 * trade.quantity * trade.greeks.gamma * ds * ds;
    let vega = trade.quantity * trade.greeks.vega * dvol;
    let cross_gamma_vanna =
        trade.quantity * (trade.vanna * ds * dvol + trade.cross_gamma * ds * dr_plus_spread);

    ExplainedPnlComponents {
        theta,
        delta,
        gamma,
        vega,
        cross_gamma_vanna,
    }
}

/// Applies a [`MarketShock`] to a [`Market`] snapshot.
pub fn apply_market_shock(market: &Market, shock: &MarketShock) -> Market {
    let spot = (market.spot * (1.0 + shock.spot_shock_pct)).max(1.0e-8);
    let rate = market.rate + shock.rate_shock_abs;
    let vol_scale = (1.0 + shock.vol_shock_pct).max(1.0e-8);
    let vol = scale_vol_source(&market.vol, vol_scale, spot);

    Market {
        spot,
        rate,
        dividend_yield: market.dividend_yield,
        dividend_schedule: market.dividend_schedule.clone(),
        vol,
        reference_date: market.reference_date.clone(),
    }
}

/// Computes detailed diff between two [`MarketSnapshot`] values.
pub fn diff_market_snapshots(
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
) -> MarketSnapshotDiff {
    let mut market_diffs = Vec::new();
    let previous_markets: BTreeMap<&str, &Market> = previous
        .markets
        .iter()
        .map(|(id, market)| (id.as_str(), market))
        .collect();

    for (market_id, current_market) in &current.markets {
        if let Some(previous_market) = previous_markets.get(market_id.as_str()) {
            let previous_atm_vol = atm_vol(previous_market);
            let current_atm_vol = atm_vol(current_market);
            market_diffs.push(MarketLevelDiff {
                market_id: market_id.clone(),
                spot_pct_change: pct_change(previous_market.spot, current_market.spot),
                rate_abs_change: current_market.rate - previous_market.rate,
                dividend_yield_abs_change: current_market.dividend_yield
                    - previous_market.dividend_yield,
                atm_vol_pct_change: pct_change(previous_atm_vol, current_atm_vol),
            });
        }
    }
    market_diffs.sort_by(|a, b| a.market_id.cmp(&b.market_id));

    let mut spot_price_diffs = Vec::new();
    let previous_spots: BTreeMap<&str, f64> = previous
        .spot_prices
        .iter()
        .map(|(id, spot)| (id.as_str(), *spot))
        .collect();
    for (asset_id, current_spot) in &current.spot_prices {
        if let Some(previous_spot) = previous_spots.get(asset_id.as_str()) {
            spot_price_diffs.push(SpotPriceDiff {
                asset_id: asset_id.clone(),
                spot_pct_change: pct_change(*previous_spot, *current_spot),
            });
        }
    }
    spot_price_diffs.sort_by(|a, b| a.asset_id.cmp(&b.asset_id));

    let mut yield_curve_diffs = Vec::new();
    let previous_curves: BTreeMap<&str, &crate::rates::YieldCurve> = previous
        .yield_curves
        .iter()
        .map(|(id, curve)| (id.as_str(), curve))
        .collect();
    for (curve_id, current_curve) in &current.yield_curves {
        if let Some(previous_curve) = previous_curves.get(curve_id.as_str()) {
            let shifts: Vec<f64> = current_curve
                .tenors
                .iter()
                .filter_map(|(tenor, _)| {
                    let current_rate = current_curve.zero_rate(*tenor);
                    let previous_rate = previous_curve.zero_rate(*tenor);
                    let shift = current_rate - previous_rate;
                    shift.is_finite().then_some(shift)
                })
                .collect();

            yield_curve_diffs.push(YieldCurveDiff {
                curve_id: curve_id.clone(),
                parallel_shift_abs: mean(&shifts),
                max_abs_shift: max_abs(&shifts),
            });
        }
    }
    yield_curve_diffs.sort_by(|a, b| a.curve_id.cmp(&b.curve_id));

    let mut credit_curve_diffs = Vec::new();
    let previous_credit: BTreeMap<&str, &CreditCurveSnapshot> = previous
        .credit_curves
        .iter()
        .map(|curve| (curve.curve_id.as_str(), curve))
        .collect();
    for current_curve in &current.credit_curves {
        if let Some(previous_curve) = previous_credit.get(current_curve.curve_id.as_str()) {
            let shifts = credit_curve_hazard_shifts(previous_curve, current_curve);
            credit_curve_diffs.push(CreditCurveDiff {
                curve_id: current_curve.curve_id.clone(),
                avg_hazard_shift_abs: mean(&shifts),
                max_hazard_shift_abs: max_abs(&shifts),
            });
        }
    }
    credit_curve_diffs.sort_by(|a, b| a.curve_id.cmp(&b.curve_id));

    MarketSnapshotDiff {
        from_snapshot_id: previous.snapshot_id.clone(),
        to_snapshot_id: current.snapshot_id.clone(),
        market_diffs,
        spot_price_diffs,
        yield_curve_diffs,
        credit_curve_diffs,
    }
}

/// Builds a historical-replay scenario from a market snapshot diff.
pub fn historical_replay_from_diff(
    scenario_id: impl Into<String>,
    replay_date: Option<String>,
    market_diff: &MarketSnapshotDiff,
    horizon_years: f64,
) -> ScenarioDefinition {
    ScenarioDefinition::HistoricalReplay(HistoricalReplayDefinition {
        scenario_id: scenario_id.into(),
        replay_date,
        shock: market_diff.to_market_shock(horizon_years),
    })
}

/// Runs a scenario batch using Greek-based explained PnL as observed PnL.
pub fn run_scenario_batch<I>(
    trades: &[ScenarioTrade<I>],
    definitions: &[ScenarioDefinition],
) -> Result<ScenarioRunResult, PricingError>
where
    I: Sync,
{
    run_scenario_batch_with_pricer(trades, definitions, |trade, shock| {
        Ok(explained_pnl_components(trade, shock).explained())
    })
}

/// Runs a scenario batch with user-provided observed PnL (full revaluation) callback.
pub fn run_scenario_batch_with_pricer<I, F>(
    trades: &[ScenarioTrade<I>],
    definitions: &[ScenarioDefinition],
    pricer: F,
) -> Result<ScenarioRunResult, PricingError>
where
    I: Sync,
    F: Fn(&ScenarioTrade<I>, &MarketShock) -> Result<f64, PricingError> + Sync,
{
    let resolved_scenarios = resolve_scenarios(definitions, trades)?;
    let mut rows = Vec::with_capacity(trades.len().saturating_mul(resolved_scenarios.len()));

    for scenario in &resolved_scenarios {
        for trade in trades {
            let components = explained_pnl_components(trade, &scenario.shock);
            let explained_pnl = components.explained();
            let observed_pnl = pricer(trade, &scenario.shock)?;
            if !observed_pnl.is_finite() {
                return Err(PricingError::NumericalError(format!(
                    "observed pnl is not finite for scenario `{}` trade `{}`",
                    scenario.scenario_id, trade.trade_id
                )));
            }
            let unexplained_pnl = observed_pnl - explained_pnl;
            let unexplained_ratio = if observed_pnl.abs() > EPS {
                unexplained_pnl.abs() / observed_pnl.abs()
            } else {
                0.0
            };

            rows.push(ScenarioTradePnlRow {
                scenario_id: scenario.scenario_id.clone(),
                parent_scenario_id: scenario.parent_scenario_id.clone(),
                scenario_kind: scenario.kind,
                trade_id: trade.trade_id.clone(),
                theta: components.theta,
                delta: components.delta,
                gamma: components.gamma,
                vega: components.vega,
                cross_gamma_vanna: components.cross_gamma_vanna,
                explained_pnl,
                observed_pnl,
                unexplained_pnl,
                unexplained_ratio,
            });
        }
    }

    Ok(ScenarioRunResult {
        resolved_scenarios,
        table: ScenarioResultTable { rows },
    })
}

/// Computes day-over-day attribution from snapshot diffs using explained PnL.
pub fn day_over_day_attribution<I>(
    trades: &[ScenarioTrade<I>],
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
) -> Result<DayOverDayAttribution, PricingError>
where
    I: Sync,
{
    day_over_day_attribution_with_pricer(trades, previous, current, |trade, shock| {
        Ok(explained_pnl_components(trade, shock).explained())
    })
}

/// Computes day-over-day attribution from snapshot diffs with full revaluation callback.
pub fn day_over_day_attribution_with_pricer<I, F>(
    trades: &[ScenarioTrade<I>],
    previous: &MarketSnapshot,
    current: &MarketSnapshot,
    pricer: F,
) -> Result<DayOverDayAttribution, PricingError>
where
    I: Sync,
    F: Fn(&ScenarioTrade<I>, &MarketShock) -> Result<f64, PricingError> + Sync,
{
    let market_diff = diff_market_snapshots(previous, current);
    let scenario_id = format!(
        "dod:{}->{}",
        market_diff.from_snapshot_id, market_diff.to_snapshot_id
    );
    let replay = historical_replay_from_diff(scenario_id, None, &market_diff, DOD_HORIZON_YEARS);

    let run = run_scenario_batch_with_pricer(trades, &[replay], pricer)?;
    let ScenarioRunResult {
        resolved_scenarios,
        table,
    } = run;

    let scenario = resolved_scenarios.first().cloned().ok_or_else(|| {
        PricingError::InvalidInput("day-over-day scenario not resolved".to_string())
    })?;

    let portfolio = table.portfolio_rows().into_iter().next().ok_or_else(|| {
        PricingError::InvalidInput("day-over-day attribution table is empty".to_string())
    })?;

    Ok(DayOverDayAttribution {
        market_diff,
        scenario,
        table,
        portfolio,
    })
}

fn resolve_scenarios<I>(
    definitions: &[ScenarioDefinition],
    trades: &[ScenarioTrade<I>],
) -> Result<Vec<ResolvedScenario>, PricingError> {
    let mut out = Vec::new();

    for definition in definitions {
        match definition {
            ScenarioDefinition::HistoricalReplay(def) => {
                validate_scenario_id(&def.scenario_id)?;
                out.push(ResolvedScenario {
                    scenario_id: def.scenario_id.clone(),
                    parent_scenario_id: def.scenario_id.clone(),
                    kind: ScenarioKind::HistoricalReplay,
                    shock: def.shock,
                    grid_point: None,
                    reverse_stress_scale: None,
                });
            }
            ScenarioDefinition::Hypothetical(def) => {
                validate_scenario_id(&def.scenario_id)?;
                out.push(ResolvedScenario {
                    scenario_id: def.scenario_id.clone(),
                    parent_scenario_id: def.scenario_id.clone(),
                    kind: ScenarioKind::Hypothetical,
                    shock: def.shock,
                    grid_point: None,
                    reverse_stress_scale: None,
                });
            }
            ScenarioDefinition::ParametricStress2d(def) => {
                validate_scenario_id(&def.scenario_id)?;
                if def.x_axis.shocks.is_empty() || def.y_axis.shocks.is_empty() {
                    return Err(PricingError::InvalidInput(format!(
                        "parametric stress `{}` requires non-empty x/y axes",
                        def.scenario_id
                    )));
                }

                for (y_index, y) in def.y_axis.shocks.iter().copied().enumerate() {
                    for (x_index, x) in def.x_axis.shocks.iter().copied().enumerate() {
                        let mut shock = def.base_shock;
                        shock.add_factor(def.x_axis.factor, x);
                        shock.add_factor(def.y_axis.factor, y);

                        out.push(ResolvedScenario {
                            scenario_id: format!("{}::x{}::y{}", def.scenario_id, x_index, y_index),
                            parent_scenario_id: def.scenario_id.clone(),
                            kind: ScenarioKind::ParametricStress2d,
                            shock,
                            grid_point: Some(StressGridPoint {
                                x_factor: def.x_axis.factor,
                                y_factor: def.y_axis.factor,
                                x_value: x,
                                y_value: y,
                                x_index,
                                y_index,
                            }),
                            reverse_stress_scale: None,
                        });
                    }
                }
            }
            ScenarioDefinition::ReverseStress(def) => {
                validate_scenario_id(&def.scenario_id)?;
                if def.target_loss.abs() <= EPS {
                    return Err(PricingError::InvalidInput(format!(
                        "reverse stress `{}` target_loss must be non-zero",
                        def.scenario_id
                    )));
                }
                let scale = solve_reverse_stress_scale(def, trades);
                out.push(ResolvedScenario {
                    scenario_id: def.scenario_id.clone(),
                    parent_scenario_id: def.scenario_id.clone(),
                    kind: ScenarioKind::ReverseStress,
                    shock: def.seed_shock.scaled(scale),
                    grid_point: None,
                    reverse_stress_scale: Some(scale),
                });
            }
        }
    }

    Ok(out)
}

fn solve_reverse_stress_scale<I>(
    definition: &ReverseStressDefinition,
    trades: &[ScenarioTrade<I>],
) -> f64 {
    let target_pnl = -definition.target_loss.abs();
    let tolerance = definition.tolerance.abs().max(EPS);
    let mut lo = 0.0;
    let mut hi = definition.max_scale.abs().max(1.0e-6);

    let eval = |scale: f64| {
        let shock = definition.seed_shock.scaled(scale);
        portfolio_explained_pnl(trades, &shock)
    };

    let mut f_lo = eval(lo) - target_pnl;
    let mut f_hi = eval(hi) - target_pnl;

    if f_lo.abs() <= tolerance {
        return lo;
    }
    if f_hi.abs() <= tolerance {
        return hi;
    }

    if f_lo.signum() == f_hi.signum() {
        return if f_lo.abs() <= f_hi.abs() { lo } else { hi };
    }

    for _ in 0..definition.max_iterations.max(1) {
        let mid = 0.5 * (lo + hi);
        let f_mid = eval(mid) - target_pnl;

        if f_mid.abs() <= tolerance {
            return mid;
        }

        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
            f_hi = f_mid;
        }
    }

    if f_lo.abs() <= f_hi.abs() { lo } else { hi }
}

fn portfolio_explained_pnl<I>(trades: &[ScenarioTrade<I>], shock: &MarketShock) -> f64 {
    trades
        .iter()
        .map(|trade| explained_pnl_components(trade, shock).explained())
        .sum()
}

fn validate_scenario_id(id: &str) -> Result<(), PricingError> {
    if id.trim().is_empty() {
        Err(PricingError::InvalidInput(
            "scenario_id must not be empty".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn atm_vol(market: &Market) -> f64 {
    market.vol_for(market.spot.max(1.0e-8), 1.0).max(1.0e-8)
}

fn scale_vol_source(source: &VolSource, scale: f64, spot: f64) -> VolSource {
    match source {
        VolSource::Flat(vol) => VolSource::Flat((vol * scale).max(1.0e-8)),
        VolSource::Sampled(surface) => {
            let mut out = surface.clone();
            for row in &mut out.vols {
                for value in row {
                    *value = (*value * scale).max(1.0e-8);
                }
            }
            VolSource::Sampled(out)
        }
        VolSource::Parametric(surface) => {
            let mut sampled = SampledVolSurface::from_surface(surface, spot.max(1.0e-8));
            for row in &mut sampled.vols {
                for value in row {
                    *value = (*value * scale).max(1.0e-8);
                }
            }
            VolSource::Sampled(sampled)
        }
    }
}

fn credit_curve_hazard_shifts(
    previous: &CreditCurveSnapshot,
    current: &CreditCurveSnapshot,
) -> Vec<f64> {
    let tenors: Vec<f64> = if current.survival_curve.tenors.is_empty() {
        vec![1.0, 3.0, 5.0]
    } else {
        current
            .survival_curve
            .tenors
            .iter()
            .map(|(tenor, _)| *tenor)
            .collect()
    };

    tenors
        .into_iter()
        .map(|tenor| {
            current.survival_curve.hazard_rate(tenor) - previous.survival_curve.hazard_rate(tenor)
        })
        .filter(|shift| shift.is_finite())
        .collect()
}

fn pct_change(previous: f64, current: f64) -> f64 {
    if previous.abs() > EPS {
        (current - previous) / previous
    } else {
        0.0
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use approx::assert_relative_eq;

    use crate::core::OptionType;
    use crate::market::MarketBuilder;
    use crate::pricing::european::{black_scholes_greeks, black_scholes_price};

    use super::*;

    #[derive(Debug, Clone)]
    struct VanillaSpec {
        option_type: OptionType,
        strike: f64,
        maturity: f64,
        rate: f64,
    }

    fn to_core_greeks(g: crate::pricing::european::Greeks) -> Greeks {
        Greeks {
            delta: g.delta,
            gamma: g.gamma,
            vega: g.vega,
            theta: g.theta,
            rho: g.rho,
        }
    }

    fn vanilla_trade(
        trade_id: &str,
        option_type: OptionType,
        strike: f64,
        maturity: f64,
        rate: f64,
        spot: f64,
        vol: f64,
        quantity: f64,
    ) -> ScenarioTrade<VanillaSpec> {
        let greeks = to_core_greeks(black_scholes_greeks(
            option_type,
            spot,
            strike.max(1.0),
            rate,
            vol,
            maturity,
        ));

        ScenarioTrade::new(
            trade_id,
            VanillaSpec {
                option_type,
                strike,
                maturity,
                rate,
            },
            quantity,
            greeks,
            spot,
            vol,
        )
    }

    fn vanilla_reprice_pnl(
        trade: &ScenarioTrade<VanillaSpec>,
        shock: &MarketShock,
    ) -> Result<f64, PricingError> {
        let base = black_scholes_price(
            trade.instrument.option_type,
            trade.spot,
            trade.instrument.strike,
            trade.instrument.rate,
            trade.implied_vol.max(1.0e-8),
            trade.instrument.maturity.max(1.0e-8),
        );

        let stressed_spot = (trade.spot * (1.0 + shock.spot_shock_pct)).max(1.0e-8);
        let stressed_vol = (trade.implied_vol * (1.0 + shock.vol_shock_pct)).max(1.0e-8);
        let stressed_rate = trade.instrument.rate + shock.rate_shock_abs;
        let stressed_t = (trade.instrument.maturity - shock.horizon_years).max(1.0e-8);

        let stressed = black_scholes_price(
            trade.instrument.option_type,
            stressed_spot,
            trade.instrument.strike,
            stressed_rate,
            stressed_vol,
            stressed_t,
        );

        Ok(trade.quantity * (stressed - base))
    }

    fn sample_market_snapshot(id: &str, spot: f64, rate: f64, vol: f64) -> MarketSnapshot {
        let market = MarketBuilder::default()
            .spot(spot)
            .rate(rate)
            .flat_vol(vol)
            .build()
            .unwrap();

        let mut snapshot = MarketSnapshot::new(id, 0);
        snapshot.markets.push(("SPX".to_string(), market));
        snapshot.spot_prices.push(("SPX".to_string(), spot));
        snapshot.yield_curves.push((
            "USD".to_string(),
            crate::rates::YieldCurve::new(vec![
                (1.0, (-(rate + 0.0000) * 1.0).exp()),
                (5.0, (-(rate + 0.0025) * 5.0).exp()),
            ]),
        ));
        snapshot.credit_curves.push(CreditCurveSnapshot {
            curve_id: "IG".to_string(),
            survival_curve: crate::credit::SurvivalCurve::from_piecewise_hazard(
                &[1.0, 3.0, 5.0],
                &[0.01, 0.012, 0.014],
            ),
            recovery_rate: 0.4,
        });
        snapshot
    }

    #[test]
    fn scenario_definitions_are_json_roundtrip_serializable() {
        let defs = vec![
            ScenarioDefinition::HistoricalReplay(HistoricalReplayDefinition {
                scenario_id: "hist-covid".to_string(),
                replay_date: Some("2020-03-16".to_string()),
                shock: MarketShock {
                    spot_shock_pct: -0.12,
                    vol_shock_pct: 0.45,
                    rate_shock_abs: -0.0025,
                    credit_spread_shock_abs: 0.01,
                    horizon_years: DOD_HORIZON_YEARS,
                },
            }),
            ScenarioDefinition::Hypothetical(HypotheticalScenarioDefinition {
                scenario_id: "hypo".to_string(),
                description: Some("desk what-if".to_string()),
                shock: MarketShock {
                    spot_shock_pct: 0.05,
                    vol_shock_pct: -0.10,
                    rate_shock_abs: 0.001,
                    credit_spread_shock_abs: -0.002,
                    horizon_years: 5.0 / 252.0,
                },
            }),
            ScenarioDefinition::ParametricStress2d(ParametricStress2dDefinition {
                scenario_id: "grid".to_string(),
                x_axis: StressAxis {
                    factor: ShockFactor::Spot,
                    shocks: vec![-0.1, 0.0, 0.1],
                },
                y_axis: StressAxis {
                    factor: ShockFactor::Vol,
                    shocks: vec![-0.2, 0.2],
                },
                base_shock: MarketShock {
                    horizon_years: DOD_HORIZON_YEARS,
                    ..MarketShock::default()
                },
            }),
            ScenarioDefinition::ReverseStress(ReverseStressDefinition {
                scenario_id: "reverse".to_string(),
                target_loss: 1_000_000.0,
                seed_shock: MarketShock {
                    spot_shock_pct: -0.01,
                    vol_shock_pct: 0.00,
                    rate_shock_abs: 0.0,
                    credit_spread_shock_abs: 0.0,
                    horizon_years: 0.0,
                },
                max_scale: 20.0,
                tolerance: 1.0,
                max_iterations: 32,
            }),
        ];

        let json = serde_json::to_string_pretty(&defs).unwrap();
        let back: Vec<ScenarioDefinition> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, defs);
    }

    #[test]
    fn market_snapshot_diff_and_day_over_day_attribution_work() {
        let previous = sample_market_snapshot("2026-02-20", 100.0, 0.02, 0.20);
        let mut current = sample_market_snapshot("2026-02-21", 102.0, 0.021, 0.23);
        current.credit_curves[0].survival_curve =
            crate::credit::SurvivalCurve::from_piecewise_hazard(
                &[1.0, 3.0, 5.0],
                &[0.011, 0.013, 0.015],
            );

        let diff = diff_market_snapshots(&previous, &current);
        assert_eq!(diff.from_snapshot_id, "2026-02-20");
        assert_eq!(diff.to_snapshot_id, "2026-02-21");
        assert_eq!(diff.market_diffs.len(), 1);

        let dod_shock = diff.to_market_shock(DOD_HORIZON_YEARS);
        assert!(dod_shock.spot_shock_pct > 0.0);
        assert!(dod_shock.vol_shock_pct > 0.0);
        assert!(dod_shock.rate_shock_abs > 0.0);
        assert!(dod_shock.credit_spread_shock_abs > 0.0);

        let trade = ScenarioTrade::new(
            "T1",
            (),
            10.0,
            Greeks {
                delta: 0.5,
                gamma: 0.1,
                vega: 1.0,
                theta: -0.5,
                rho: 0.2,
            },
            100.0,
            0.2,
        );

        let attribution = day_over_day_attribution(&[trade], &previous, &current).unwrap();
        assert_eq!(attribution.scenario.kind, ScenarioKind::HistoricalReplay);
        assert_eq!(attribution.table.rows.len(), 1);
        assert!(attribution.portfolio.observed_pnl.is_finite());
    }

    #[test]
    fn pnl_explain_residual_is_small_for_vanilla_portfolio() {
        let mut trades = Vec::new();
        for i in 0..16 {
            let is_call = i % 2 == 0;
            let strike = 85.0 + i as f64 * 2.0;
            let maturity = 0.35 + 0.04 * i as f64;
            let spot = 100.0;
            let vol = 0.18 + 0.002 * i as f64;
            let rate = 0.02;
            let qty = 100.0;

            trades.push(vanilla_trade(
                &format!("V{i}"),
                if is_call {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
                strike,
                maturity,
                rate,
                spot,
                vol,
                qty,
            ));
        }

        let definitions = vec![ScenarioDefinition::Hypothetical(
            HypotheticalScenarioDefinition {
                scenario_id: "vanilla-hypo".to_string(),
                description: None,
                shock: MarketShock {
                    spot_shock_pct: 0.01,
                    vol_shock_pct: 0.04,
                    rate_shock_abs: 0.0005,
                    credit_spread_shock_abs: 0.0,
                    horizon_years: 1.0 / 252.0,
                },
            },
        )];

        let run =
            run_scenario_batch_with_pricer(&trades, &definitions, vanilla_reprice_pnl).unwrap();
        let portfolio_rows = run.table.portfolio_rows();
        assert_eq!(portfolio_rows.len(), 1);

        let unexplained_ratio = portfolio_rows[0].unexplained_ratio;
        assert!(
            unexplained_ratio < 0.05,
            "vanilla portfolio residual too large: {unexplained_ratio}"
        );
    }

    #[test]
    fn scenario_batch_scales_to_1000_trades_x_20_scenarios_under_30s() {
        let mut trades = Vec::new();
        for i in 0..1000 {
            trades.push(ScenarioTrade::new(
                format!("T{i}"),
                (),
                10.0,
                Greeks {
                    delta: 0.5,
                    gamma: 0.03,
                    vega: 2.0,
                    theta: -0.4,
                    rho: 0.2,
                },
                100.0 + (i % 10) as f64,
                0.2,
            ));
        }

        let definitions = vec![ScenarioDefinition::ParametricStress2d(
            ParametricStress2dDefinition {
                scenario_id: "grid-20".to_string(),
                x_axis: StressAxis {
                    factor: ShockFactor::Spot,
                    shocks: vec![-0.15, -0.075, 0.0, 0.075, 0.15],
                },
                y_axis: StressAxis {
                    factor: ShockFactor::Vol,
                    shocks: vec![-0.30, -0.10, 0.10, 0.30],
                },
                base_shock: MarketShock {
                    horizon_years: DOD_HORIZON_YEARS,
                    ..MarketShock::default()
                },
            },
        )];

        let start = Instant::now();
        let run = run_scenario_batch(&trades, &definitions).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(run.resolved_scenarios.len(), 20);
        assert_eq!(run.table.rows.len(), 20_000);
        assert!(
            elapsed.as_secs_f64() < 30.0,
            "scenario batch exceeded 30s: {:.3}s",
            elapsed.as_secs_f64()
        );
    }

    #[test]
    fn stress_grid_produces_heatmap_data() {
        let trades = vec![ScenarioTrade::new(
            "T1",
            (),
            25.0,
            Greeks {
                delta: 0.6,
                gamma: 0.08,
                vega: 1.8,
                theta: -0.2,
                rho: 0.0,
            },
            100.0,
            0.2,
        )];

        let definitions = vec![ScenarioDefinition::ParametricStress2d(
            ParametricStress2dDefinition {
                scenario_id: "grid".to_string(),
                x_axis: StressAxis {
                    factor: ShockFactor::Spot,
                    shocks: vec![-0.10, 0.0, 0.10],
                },
                y_axis: StressAxis {
                    factor: ShockFactor::Vol,
                    shocks: vec![-0.20, 0.20],
                },
                base_shock: MarketShock::default(),
            },
        )];

        let run = run_scenario_batch(&trades, &definitions).unwrap();
        let heatmap = run.stress_heatmap("grid").expect("expected heatmap data");

        assert_eq!(heatmap.x_values.len(), 3);
        assert_eq!(heatmap.y_values.len(), 2);
        assert_eq!(heatmap.pnl.len(), 2);
        assert_eq!(heatmap.pnl[0].len(), 3);
        assert!(
            heatmap
                .pnl
                .iter()
                .flat_map(|row| row.iter())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn reverse_stress_finds_target_loss_scale() {
        let trades = vec![ScenarioTrade::new(
            "T1",
            (),
            1.0,
            Greeks {
                delta: 50.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            },
            100.0,
            0.2,
        )];

        let definitions = vec![ScenarioDefinition::ReverseStress(ReverseStressDefinition {
            scenario_id: "rev".to_string(),
            target_loss: 100.0,
            seed_shock: MarketShock {
                spot_shock_pct: -0.01,
                vol_shock_pct: 0.0,
                rate_shock_abs: 0.0,
                credit_spread_shock_abs: 0.0,
                horizon_years: 0.0,
            },
            max_scale: 10.0,
            tolerance: 1.0e-9,
            max_iterations: 64,
        })];

        let run = run_scenario_batch(&trades, &definitions).unwrap();
        assert_eq!(run.resolved_scenarios.len(), 1);
        let resolved = &run.resolved_scenarios[0];
        assert_eq!(resolved.kind, ScenarioKind::ReverseStress);

        let scale = resolved.reverse_stress_scale.unwrap();
        assert_relative_eq!(scale, 2.0, epsilon = 1.0e-6);

        let portfolio = run.table.portfolio_rows();
        assert_eq!(portfolio.len(), 1);
        assert_relative_eq!(portfolio[0].observed_pnl, -100.0, epsilon = 1.0e-5);
    }
}
