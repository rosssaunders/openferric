use std::panic::{self, UnwindSafe};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use openferric::instruments::FundingRateSwap as CoreFundingRateSwap;
use openferric::models::{CIR, Vasicek};
use openferric::pricing::funding_rate_swap::{
    FundingRateSwapRisks as CoreFundingRateSwapRisks, funding_rate_swap_dv01 as core_dv01,
    funding_rate_swap_mtm as core_mtm, funding_rate_swap_risks as core_risks,
    funding_rate_swap_theta as core_theta, funding_rate_swap_vega as core_vega,
};
use openferric::rates::{
    FundingRateCurve as CoreFundingRateCurve, FundingRateSnapshot as CoreFundingRateSnapshot,
    MultiVenueFundingCurve as CoreMultiVenueFundingCurve,
};
use openferric::risk::{
    FundingRateModel as CoreFundingRateModel, InherentLeverage as CoreInherentLeverage,
    LiquidationPosition as CoreLiquidationPosition, LiquidationRisk as CoreLiquidationRisk,
    LiquidationSimulator as CoreLiquidationSimulator, MarginCalculator as CoreMarginCalculator,
    MarginParams as CoreMarginParams, StressScenario as CoreStressScenario,
};

const MS_PER_YEAR: f64 = 365.0 * 24.0 * 60.0 * 60.0 * 1000.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FundingRateSnapshotWire {
    venue: String,
    asset: String,
    rate: f64,
    timestamp_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightedCurveWire {
    weight: f64,
    snapshots: Vec<FundingRateSnapshotWire>,
}

#[inline]
fn js_error(message: impl Into<String>) -> JsValue {
    JsValue::from_str(&message.into())
}

fn catch_unwind_js<T, F>(f: F) -> Result<T, JsValue>
where
    F: FnOnce() -> T + UnwindSafe,
{
    panic::catch_unwind(f).map_err(|payload| {
        if let Some(message) = payload.downcast_ref::<&str>() {
            return js_error(*message);
        }
        if let Some(message) = payload.downcast_ref::<String>() {
            return js_error(message.clone());
        }
        js_error("operation failed")
    })
}

fn parse_timestamp_ms(timestamp_ms: f64) -> Result<DateTime<Utc>, JsValue> {
    if !timestamp_ms.is_finite() {
        return Err(js_error("timestamp_ms must be finite"));
    }
    let millis = timestamp_ms.round();
    if !(i64::MIN as f64..=i64::MAX as f64).contains(&millis) {
        return Err(js_error("timestamp_ms is out of range"));
    }
    DateTime::from_timestamp_millis(millis as i64)
        .ok_or_else(|| js_error("timestamp_ms is invalid"))
}

#[inline]
fn timestamp_ms(value: DateTime<Utc>) -> f64 {
    value.timestamp_millis() as f64
}

fn snapshot_wire_to_core(
    snapshot: FundingRateSnapshotWire,
) -> Result<CoreFundingRateSnapshot, JsValue> {
    Ok(CoreFundingRateSnapshot {
        venue: snapshot.venue,
        asset: snapshot.asset,
        rate: snapshot.rate,
        timestamp: parse_timestamp_ms(snapshot.timestamp_ms)?,
    })
}

fn snapshot_core_to_wire(snapshot: CoreFundingRateSnapshot) -> FundingRateSnapshotWire {
    FundingRateSnapshotWire {
        venue: snapshot.venue,
        asset: snapshot.asset,
        rate: snapshot.rate,
        timestamp_ms: timestamp_ms(snapshot.timestamp),
    }
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone)]
pub struct FundingRateSnapshot {
    pub venue: String,
    pub asset: String,
    pub rate: f64,
    pub timestamp_ms: f64,
}

impl FundingRateSnapshot {
    fn to_core(&self) -> Result<CoreFundingRateSnapshot, JsValue> {
        snapshot_wire_to_core(FundingRateSnapshotWire {
            venue: self.venue.clone(),
            asset: self.asset.clone(),
            rate: self.rate,
            timestamp_ms: self.timestamp_ms,
        })
    }
}

#[wasm_bindgen]
impl FundingRateSnapshot {
    #[wasm_bindgen(constructor)]
    pub fn new(
        venue: String,
        asset: String,
        rate: f64,
        timestamp_ms: f64,
    ) -> Result<Self, JsValue> {
        let snapshot = Self {
            venue,
            asset,
            rate,
            timestamp_ms,
        };
        let _ = snapshot.to_core()?;
        Ok(snapshot)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&FundingRateSnapshotWire {
            venue: self.venue.clone(),
            asset: self.asset.clone(),
            rate: self.rate,
            timestamp_ms: self.timestamp_ms,
        })
        .map_err(|err| js_error(err.to_string()))
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct FundingRateCurve {
    inner: CoreFundingRateCurve,
}

#[wasm_bindgen]
impl FundingRateCurve {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(snapshots_json: &str) -> Result<Self, JsValue> {
        let wires = serde_json::from_str::<Vec<FundingRateSnapshotWire>>(snapshots_json)
            .map_err(|err| js_error(err.to_string()))?;
        let snapshots = wires
            .into_iter()
            .map(snapshot_wire_to_core)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            inner: CoreFundingRateCurve::new(snapshots),
        })
    }

    #[wasm_bindgen(js_name = flat)]
    pub fn flat(apr: f64) -> Self {
        Self {
            inner: CoreFundingRateCurve::flat(apr),
        }
    }

    pub fn forward_rate(&self, t: f64) -> f64 {
        self.inner.forward_rate(t)
    }

    #[wasm_bindgen(js_name = forwardApr)]
    pub fn forward_apr(&self, t: f64) -> f64 {
        CoreFundingRateCurve::per_period_rate_to_apr(self.inner.forward_rate(t))
    }

    pub fn cumulative_index(&self, t: f64) -> f64 {
        self.inner.cumulative_index(t)
    }

    pub fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }

    #[wasm_bindgen(js_name = expectedRate)]
    pub fn expected_rate(
        &self,
        as_of_timestamp_ms: f64,
        start_timestamp_ms: f64,
        end_timestamp_ms: f64,
    ) -> Result<f64, JsValue> {
        Ok(self.inner.expected_rate(
            parse_timestamp_ms(as_of_timestamp_ms)?,
            parse_timestamp_ms(start_timestamp_ms)?,
            parse_timestamp_ms(end_timestamp_ms)?,
        ))
    }

    #[wasm_bindgen(js_name = snapshotsJson)]
    pub fn snapshots_json(&self) -> Result<String, JsValue> {
        let snapshots = self
            .inner
            .snapshots()
            .iter()
            .cloned()
            .map(snapshot_core_to_wire)
            .collect::<Vec<_>>();
        serde_json::to_string(&snapshots).map_err(|err| js_error(err.to_string()))
    }

    #[wasm_bindgen(js_name = nodesFlat)]
    pub fn nodes_flat(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.inner.nodes().len() * 2);
        for (time, rate) in self.inner.nodes() {
            flat.push(*time);
            flat.push(*rate);
        }
        flat
    }

    #[wasm_bindgen(js_name = anchorTimestampMs)]
    pub fn anchor_timestamp_ms(&self) -> f64 {
        self.inner
            .anchor_timestamp()
            .map(timestamp_ms)
            .unwrap_or(f64::NAN)
    }

    #[wasm_bindgen(js_name = parallelShifted)]
    pub fn parallel_shifted(&self, bump_apr: f64) -> Self {
        Self {
            inner: self.inner.parallel_shifted(bump_apr),
        }
    }

    #[wasm_bindgen(js_name = perPeriodRateToApr)]
    pub fn per_period_rate_to_apr(rate: f64) -> f64 {
        CoreFundingRateCurve::per_period_rate_to_apr(rate)
    }

    #[wasm_bindgen(js_name = aprToPerPeriodRate)]
    pub fn apr_to_per_period_rate(apr: f64) -> f64 {
        CoreFundingRateCurve::apr_to_per_period_rate(apr)
    }

    #[wasm_bindgen(js_name = settlementIntervalYears)]
    pub fn settlement_interval_years() -> f64 {
        CoreFundingRateCurve::settlement_interval_years()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct MultiVenueFundingCurve {
    inner: CoreMultiVenueFundingCurve,
}

#[wasm_bindgen]
impl MultiVenueFundingCurve {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(curves_json: &str) -> Result<Self, JsValue> {
        let wires = serde_json::from_str::<Vec<WeightedCurveWire>>(curves_json)
            .map_err(|err| js_error(err.to_string()))?;
        let curves = wires
            .into_iter()
            .map(|wire| {
                let curve = wire
                    .snapshots
                    .into_iter()
                    .map(snapshot_wire_to_core)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((CoreFundingRateCurve::new(curve), wire.weight))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        Ok(Self {
            inner: CoreMultiVenueFundingCurve::new(curves),
        })
    }

    pub fn forward_rate(&self, t: f64) -> f64 {
        self.inner.forward_rate(t)
    }

    #[wasm_bindgen(js_name = forwardApr)]
    pub fn forward_apr(&self, t: f64) -> f64 {
        CoreFundingRateCurve::per_period_rate_to_apr(self.inner.forward_rate(t))
    }

    pub fn cumulative_index(&self, t: f64) -> f64 {
        self.inner.cumulative_index(t)
    }

    pub fn discount_factor(&self, t: f64) -> f64 {
        self.inner.discount_factor(t)
    }
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone)]
pub struct FundingRateSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub entry_time_ms: f64,
    pub maturity_time_ms: f64,
    pub settlement_interval_hours: f64,
    pub venue: String,
    pub asset: String,
}

impl FundingRateSwap {
    fn to_core(&self) -> Result<CoreFundingRateSwap, JsValue> {
        let mut swap = CoreFundingRateSwap::new(
            self.notional,
            self.fixed_rate,
            parse_timestamp_ms(self.entry_time_ms)?,
            parse_timestamp_ms(self.maturity_time_ms)?,
            self.venue.clone(),
            self.asset.clone(),
        );
        swap.settlement_interval_hours = self.settlement_interval_hours as u32;
        Ok(swap)
    }
}

#[wasm_bindgen]
impl FundingRateSwap {
    #[wasm_bindgen(constructor)]
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        entry_time_ms: f64,
        maturity_time_ms: f64,
        venue: String,
        asset: String,
    ) -> Result<Self, JsValue> {
        let swap = Self {
            notional,
            fixed_rate,
            entry_time_ms,
            maturity_time_ms,
            settlement_interval_hours: 8.0,
            venue,
            asset,
        };
        swap.validate()?;
        Ok(swap)
    }

    pub fn validate(&self) -> Result<(), JsValue> {
        self.to_core()?
            .validate()
            .map_err(|err| js_error(err.to_string()))
    }

    #[wasm_bindgen(js_name = settlementSchedule)]
    pub fn settlement_schedule(&self) -> Result<Vec<f64>, JsValue> {
        Ok(self
            .to_core()?
            .settlement_schedule()
            .into_iter()
            .map(timestamp_ms)
            .collect())
    }

    #[wasm_bindgen(js_name = intervalYearFraction)]
    pub fn interval_year_fraction(&self) -> Result<f64, JsValue> {
        Ok(self.to_core()?.interval_year_fraction())
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct FundingRateSwapRisks {
    pub mtm: f64,
    pub dv01: f64,
    pub vega: f64,
    pub theta: f64,
}

impl FundingRateSwapRisks {
    fn from_core(risks: CoreFundingRateSwapRisks) -> Self {
        Self {
            mtm: risks.mtm,
            dv01: risks.dv01,
            vega: risks.vega,
            theta: risks.theta,
        }
    }
}

#[wasm_bindgen]
impl FundingRateSwapRisks {
    #[wasm_bindgen(constructor)]
    pub fn new(mtm: f64, dv01: f64, vega: f64, theta: f64) -> Self {
        Self {
            mtm,
            dv01,
            vega,
            theta,
        }
    }
}

#[wasm_bindgen]
pub fn funding_rate_swap_mtm(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of_timestamp_ms: f64,
) -> Result<f64, JsValue> {
    Ok(core_mtm(
        &swap.to_core()?,
        &curve.inner,
        parse_timestamp_ms(as_of_timestamp_ms)?,
    ))
}

#[wasm_bindgen]
pub fn funding_rate_swap_dv01(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of_timestamp_ms: f64,
) -> Result<f64, JsValue> {
    Ok(core_dv01(
        &swap.to_core()?,
        &curve.inner,
        parse_timestamp_ms(as_of_timestamp_ms)?,
    ))
}

#[wasm_bindgen]
pub fn funding_rate_swap_theta(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of_timestamp_ms: f64,
) -> Result<f64, JsValue> {
    Ok(core_theta(
        &swap.to_core()?,
        &curve.inner,
        parse_timestamp_ms(as_of_timestamp_ms)?,
    ))
}

#[wasm_bindgen]
pub fn funding_rate_swap_vega(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of_timestamp_ms: f64,
) -> Result<f64, JsValue> {
    Ok(core_vega(
        &swap.to_core()?,
        &curve.inner,
        parse_timestamp_ms(as_of_timestamp_ms)?,
    ))
}

#[wasm_bindgen]
pub fn funding_rate_swap_risks(
    swap: &FundingRateSwap,
    curve: &FundingRateCurve,
    as_of_timestamp_ms: f64,
) -> Result<FundingRateSwapRisks, JsValue> {
    Ok(FundingRateSwapRisks::from_core(core_risks(
        &swap.to_core()?,
        &curve.inner,
        parse_timestamp_ms(as_of_timestamp_ms)?,
    )))
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct MarginParams {
    pub initial_margin_ratio: f64,
    pub maintenance_margin_ratio: f64,
    pub funding_rate_vol: f64,
    pub time_to_maturity: f64,
    pub tick_size: f64,
}

impl MarginParams {
    fn to_core(self) -> CoreMarginParams {
        CoreMarginParams {
            initial_margin_ratio: self.initial_margin_ratio,
            maintenance_margin_ratio: self.maintenance_margin_ratio,
            funding_rate_vol: self.funding_rate_vol,
            time_to_maturity: self.time_to_maturity,
            tick_size: self.tick_size,
        }
    }
}

#[wasm_bindgen]
impl MarginParams {
    #[wasm_bindgen(constructor)]
    pub fn new(
        initial_margin_ratio: f64,
        maintenance_margin_ratio: f64,
        funding_rate_vol: f64,
        time_to_maturity: f64,
        tick_size: f64,
    ) -> Self {
        Self {
            initial_margin_ratio,
            maintenance_margin_ratio,
            funding_rate_vol,
            time_to_maturity,
            tick_size,
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, Default)]
pub struct MarginCalculator;

#[wasm_bindgen]
impl MarginCalculator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    #[wasm_bindgen(js_name = initialMargin)]
    pub fn initial_margin(notional: f64, params: &MarginParams) -> Result<f64, JsValue> {
        catch_unwind_js(|| CoreMarginCalculator::initial_margin(notional, &params.to_core()))
    }

    #[wasm_bindgen(js_name = maintenanceMargin)]
    pub fn maintenance_margin(notional: f64, params: &MarginParams) -> Result<f64, JsValue> {
        catch_unwind_js(|| CoreMarginCalculator::maintenance_margin(notional, &params.to_core()))
    }

    #[wasm_bindgen(js_name = healthRatio)]
    pub fn health_ratio(
        collateral: f64,
        notional: f64,
        unrealized_pnl: f64,
        params: &MarginParams,
    ) -> Result<f64, JsValue> {
        catch_unwind_js(|| {
            CoreMarginCalculator::health_ratio(
                collateral,
                notional,
                unrealized_pnl,
                &params.to_core(),
            )
        })
    }

    #[wasm_bindgen(js_name = liquidationRate)]
    pub fn liquidation_rate(
        entry_rate: f64,
        collateral: f64,
        notional: f64,
        params: &MarginParams,
    ) -> Result<f64, JsValue> {
        catch_unwind_js(|| {
            CoreMarginCalculator::liquidation_rate(
                entry_rate,
                collateral,
                notional,
                &params.to_core(),
            )
        })
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, Default)]
pub struct InherentLeverage;

#[wasm_bindgen]
impl InherentLeverage {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    pub fn leverage(notional: f64, yu_cost: f64) -> Result<f64, JsValue> {
        catch_unwind_js(|| CoreInherentLeverage::leverage(notional, yu_cost))
    }

    #[wasm_bindgen(js_name = leveragedReturn)]
    pub fn leveraged_return(rate_move: f64, leverage: f64) -> Result<f64, JsValue> {
        catch_unwind_js(|| CoreInherentLeverage::leveraged_return(rate_move, leverage))
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct FundingRateModel {
    inner: CoreFundingRateModel,
}

#[wasm_bindgen]
impl FundingRateModel {
    pub fn vasicek(a: f64, b: f64, sigma: f64) -> Self {
        Self {
            inner: CoreFundingRateModel::Vasicek(Vasicek { a, b, sigma }),
        }
    }

    pub fn cir(a: f64, b: f64, sigma: f64) -> Self {
        Self {
            inner: CoreFundingRateModel::CIR(CIR { a, b, sigma }),
        }
    }

    pub fn kind(&self) -> String {
        match self.inner {
            CoreFundingRateModel::Vasicek(_) => "vasicek".to_string(),
            CoreFundingRateModel::CIR(_) => "cir".to_string(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct LiquidationPosition {
    inner: CoreLiquidationPosition,
}

#[wasm_bindgen]
impl LiquidationPosition {
    #[wasm_bindgen(constructor)]
    pub fn new(size: f64, entry_rate: f64, collateral: f64, margin_params: &MarginParams) -> Self {
        Self {
            inner: CoreLiquidationPosition {
                size,
                entry_rate,
                collateral,
                margin_params: margin_params.to_core(),
            },
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct StressScenario {
    inner: CoreStressScenario,
}

#[wasm_bindgen]
impl StressScenario {
    pub fn baseline() -> Self {
        Self {
            inner: CoreStressScenario::Baseline,
        }
    }

    #[wasm_bindgen(js_name = liquidationCascade)]
    pub fn liquidation_cascade(vol_multiplier: f64) -> Self {
        Self {
            inner: CoreStressScenario::LiquidationCascade { vol_multiplier },
        }
    }

    #[wasm_bindgen(js_name = meanShift)]
    pub fn mean_shift(shift: f64) -> Self {
        Self {
            inner: CoreStressScenario::MeanShift { shift },
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct LiquidationRisk {
    pub prob_liquidation: f64,
    pub expected_time_to_liquidation: f64,
    pub worst_case_funding_rate: f64,
}

impl LiquidationRisk {
    fn from_core(risk: CoreLiquidationRisk) -> Self {
        Self {
            prob_liquidation: risk.prob_liquidation,
            expected_time_to_liquidation: risk.expected_time_to_liquidation.unwrap_or(f64::NAN),
            worst_case_funding_rate: risk.worst_case_funding_rate,
        }
    }
}

#[wasm_bindgen]
impl LiquidationRisk {
    #[wasm_bindgen(js_name = hasExpectedTime)]
    pub fn has_expected_time(&self) -> bool {
        self.expected_time_to_liquidation.is_finite()
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct LiquidationSimulator {
    inner: CoreLiquidationSimulator,
}

#[wasm_bindgen]
impl LiquidationSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        position: &LiquidationPosition,
        model: &FundingRateModel,
        initial_funding_rate: f64,
        num_paths: f64,
        steps: f64,
        seed: f64,
    ) -> Result<Self, JsValue> {
        catch_unwind_js(|| Self {
            inner: CoreLiquidationSimulator::new(
                position.inner,
                model.inner,
                initial_funding_rate,
                num_paths as usize,
                steps as usize,
                seed as u64,
            ),
        })
    }

    pub fn simulate(&self) -> Result<LiquidationRisk, JsValue> {
        catch_unwind_js(|| LiquidationRisk::from_core(self.inner.simulate()))
    }

    #[wasm_bindgen(js_name = simulateStress)]
    pub fn simulate_stress(&self, scenario: &StressScenario) -> Result<LiquidationRisk, JsValue> {
        catch_unwind_js(|| LiquidationRisk::from_core(self.inner.simulate_stress(scenario.inner)))
    }
}

#[wasm_bindgen]
pub fn years_to_timestamp_ms(anchor_timestamp_ms: f64, year_fraction: f64) -> f64 {
    anchor_timestamp_ms + year_fraction * MS_PER_YEAR
}
