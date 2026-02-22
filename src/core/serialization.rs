//! Canonical trade, market data, portfolio, and pricing-audit serialization.
//!
//! These types define stable serde payloads used to persist and transport
//! trades, market snapshots, and pricing outputs.
//!
//! # Examples
//! ```rust
//! use openferric::core::{
//!     from_json, from_msgpack, to_json_pretty, to_msgpack, PortfolioSnapshot, Trade,
//!     TradeMetadata, TradeProduct,
//! };
//! use openferric::instruments::VanillaOption;
//! use openferric::core::{ExerciseStyle, OptionType};
//!
//! let trade = Trade {
//!     metadata: TradeMetadata {
//!         trade_id: "TRD-001".to_string(),
//!         version: 1,
//!         timestamp: "2026-02-22T11:00:00Z".to_string(),
//!     },
//!     market_data_ref: Some("SNAP-2026-02-22".to_string()),
//!     product: TradeProduct::VanillaOption(VanillaOption {
//!         option_type: OptionType::Call,
//!         strike: 100.0,
//!         expiry: 1.0,
//!         exercise: ExerciseStyle::European,
//!     }),
//! };
//!
//! let portfolio = PortfolioSnapshot {
//!     portfolio_id: "PF-001".to_string(),
//!     market_snapshot_id: "SNAP-2026-02-22".to_string(),
//!     as_of: "2026-02-22T11:00:00Z".to_string(),
//!     trades: vec![trade],
//! };
//!
//! let json = to_json_pretty(&portfolio).expect("json serialization");
//! let decoded: PortfolioSnapshot = from_json(&json).expect("json deserialization");
//! assert_eq!(decoded, portfolio);
//!
//! let bytes = to_msgpack(&portfolio).expect("msgpack serialization");
//! let decoded_msgpack: PortfolioSnapshot = from_msgpack(&bytes).expect("msgpack deserialization");
//! assert_eq!(decoded_msgpack, portfolio);
//! ```

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;

use crate::core::PricingResult;
use crate::credit::cds_option::CdsOption;
use crate::credit::{CdoTranche, Cds, CdsIndex, DatedCds, NthToDefaultBasket, SyntheticCdo};
use crate::instruments::mbs::MbsPassThrough;
use crate::instruments::{
    AbandonmentOption, AsianOption, AssetOrNothingOption, Autocallable, BarrierOption,
    BasketOption, BestOfTwoCallOption, CashOrNothingOption, CatastropheBond, CommodityForward,
    CommodityFutures, CommodityOption, CommoditySpreadOption, ConvertibleBond,
    DeferInvestmentOption, DoubleBarrierOption, DualRangeAccrual, EmployeeStockOption,
    ExoticOption, ExpandOption, ForwardStartOption, FuturesOption, FxOption, GapOption,
    PhoenixAutocallable, PowerOption, RangeAccrual, RealOptionInstrument, SpreadOption,
    SwingOption, Tarf, TwoAssetCorrelationOption, VanillaOption, VarianceSwap, VolatilitySwap,
    WeatherOption, WeatherSwap, WorstOfTwoCallOption,
};
use crate::rates::cms::CmsSpreadOption;
use crate::rates::{
    BasisSwap, CapFloor, FixedRateBond, ForwardRateAgreement, Future, InflationIndexedBond,
    InterestRateSwap, OvernightIndexSwap, Swaption, XccySwap, YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
};
use crate::vol::sabr::SabrParams;
use crate::vol::surface::SviParams;

/// Canonical product payload tagged by `product_type` in JSON.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "product_type", content = "payload", rename_all = "snake_case")]
pub enum TradeProduct {
    VanillaOption(VanillaOption),
    BarrierOption(BarrierOption),
    AsianOption(AsianOption),
    BasketOption(BasketOption),
    CashOrNothingOption(CashOrNothingOption),
    AssetOrNothingOption(AssetOrNothingOption),
    GapOption(GapOption),
    DoubleBarrierOption(DoubleBarrierOption),
    SpreadOption(SpreadOption),
    FxOption(FxOption),
    FuturesOption(FuturesOption),
    ForwardStartOption(ForwardStartOption),
    CommodityForward(CommodityForward),
    CommodityFutures(CommodityFutures),
    CommodityOption(CommodityOption),
    CommoditySpreadOption(CommoditySpreadOption),
    ConvertibleBond(ConvertibleBond),
    EmployeeStockOption(EmployeeStockOption),
    ExoticOption(ExoticOption),
    PowerOption(PowerOption),
    BestOfTwoCallOption(BestOfTwoCallOption),
    WorstOfTwoCallOption(WorstOfTwoCallOption),
    TwoAssetCorrelationOption(TwoAssetCorrelationOption),
    RangeAccrual(RangeAccrual),
    DualRangeAccrual(DualRangeAccrual),
    DeferInvestmentOption(DeferInvestmentOption),
    ExpandOption(ExpandOption),
    AbandonmentOption(AbandonmentOption),
    RealOptionInstrument(RealOptionInstrument),
    SwingOption(SwingOption),
    Tarf(Tarf),
    VarianceSwap(VarianceSwap),
    VolatilitySwap(VolatilitySwap),
    WeatherSwap(WeatherSwap),
    WeatherOption(WeatherOption),
    CatastropheBond(CatastropheBond),
    Autocallable(Autocallable),
    PhoenixAutocallable(PhoenixAutocallable),
    MbsPassThrough(MbsPassThrough),
    FixedRateBond(FixedRateBond),
    CapFloor(CapFloor),
    ForwardRateAgreement(ForwardRateAgreement),
    Future(Future),
    InterestRateSwap(InterestRateSwap),
    Swaption(Swaption),
    OvernightIndexSwap(OvernightIndexSwap),
    BasisSwap(BasisSwap),
    XccySwap(XccySwap),
    ZeroCouponInflationSwap(ZeroCouponInflationSwap),
    YearOnYearInflationSwap(YearOnYearInflationSwap),
    InflationIndexedBond(InflationIndexedBond),
    CmsSpreadOption(CmsSpreadOption),
    Cds(Cds),
    CdsOption(CdsOption),
    CdoTranche(CdoTranche),
    SyntheticCdo(SyntheticCdo),
    CdsIndex(CdsIndex),
    NthToDefaultBasket(NthToDefaultBasket),
    DatedCds(DatedCds),
}

/// Trade metadata required for persistence and audit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TradeMetadata {
    pub trade_id: String,
    pub version: u64,
    /// RFC3339 timestamp string (UTC recommended).
    pub timestamp: String,
}

/// Canonical trade payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    pub metadata: TradeMetadata,
    /// Optional pointer to a market snapshot identifier.
    pub market_data_ref: Option<String>,
    pub product: TradeProduct,
}

/// Curve interpolation method for serialized market data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    Linear,
    LogLinear,
    CubicSpline,
    FlatForward,
}

/// Semantic meaning of serialized curve values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveValueType {
    ZeroRate,
    DiscountFactor,
    SurvivalProbability,
    ForwardPrice,
}

/// Yield curve market data with explicit interpolation settings.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct YieldCurveSnapshot {
    pub pillars: Vec<f64>,
    pub rates: Vec<f64>,
    pub interpolation: InterpolationMethod,
    pub value_type: CurveValueType,
}

impl YieldCurveSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.pillars.is_empty() {
            return Err("yield curve pillars must be non-empty".to_string());
        }
        if self.pillars.len() != self.rates.len() {
            return Err("yield curve pillars/rates length mismatch".to_string());
        }
        if self.pillars.windows(2).any(|w| w[1] <= w[0]) {
            return Err("yield curve pillars must be strictly increasing".to_string());
        }
        Ok(())
    }
}

/// Credit curve snapshot with survival probabilities and recovery.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CreditCurveSnapshot {
    pub pillars: Vec<f64>,
    pub survival_probabilities: Vec<f64>,
    pub recovery_rate: f64,
    pub interpolation: InterpolationMethod,
}

/// Parametric model metadata for vol surfaces.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum VolSurfaceModel {
    Svi {
        parameters_by_expiry: Vec<(f64, SviParams)>,
    },
    Sabr {
        parameters_by_expiry: Vec<(f64, SabrParams)>,
    },
    Custom {
        name: String,
        parameters: BTreeMap<String, f64>,
    },
}

/// Vol surface snapshot as a strikes x expiries matrix.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VolSurfaceSnapshot {
    pub expiries: Vec<f64>,
    pub strikes: Vec<f64>,
    /// Row-major volatility matrix `[expiry_index][strike_index]`.
    pub vols: Vec<Vec<f64>>,
    pub model: Option<VolSurfaceModel>,
}

/// Forward curve market data for one asset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForwardCurveSnapshot {
    pub points: Vec<(f64, f64)>,
    pub interpolation: InterpolationMethod,
}

/// Point-in-time market snapshot containing all curves/surfaces/quotes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarketSnapshot {
    pub snapshot_id: String,
    /// RFC3339 timestamp string (UTC recommended).
    pub as_of: String,
    pub yield_curves: BTreeMap<String, YieldCurveSnapshot>,
    pub credit_curves: BTreeMap<String, CreditCurveSnapshot>,
    pub vol_surfaces: BTreeMap<String, VolSurfaceSnapshot>,
    pub spots: BTreeMap<String, f64>,
    pub forwards: BTreeMap<String, ForwardCurveSnapshot>,
}

/// Portfolio container with trade list and shared market reference.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PortfolioSnapshot {
    pub portfolio_id: String,
    pub market_snapshot_id: String,
    /// RFC3339 timestamp string (UTC recommended).
    pub as_of: String,
    pub trades: Vec<Trade>,
}

/// Scenario output linked to a bumped market setup.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioPricingResult {
    pub scenario_name: String,
    pub bump_description: Option<String>,
    pub result: PricingResult,
}

/// Serializable pricing result with full input/model/output audit trail.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PricingAuditTrail {
    pub trade: Trade,
    pub market_snapshot_id: String,
    pub engine_name: String,
    pub model_name: String,
    pub model_parameters: BTreeMap<String, f64>,
    pub base_result: PricingResult,
    pub scenario_results: Vec<ScenarioPricingResult>,
    /// RFC3339 timestamp string (UTC recommended).
    pub generated_at: String,
    pub notes: Vec<String>,
}

/// Serialize a value to pretty JSON.
pub fn to_json_pretty<T: serde::Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(value)
}

/// Deserialize a value from JSON.
pub fn from_json<T: DeserializeOwned>(payload: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(payload)
}

/// Serialize a value to MessagePack bytes.
pub fn to_msgpack<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(value)
}

/// Deserialize a value from MessagePack bytes.
pub fn from_msgpack<T: DeserializeOwned>(payload: &[u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(payload)
}
