//! Module `instruments::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these contract types as immutable pricing inputs; pair with engine modules for valuation and risk, rather than embedding valuation logic in instruments.

pub mod asian;
pub mod autocallable;
pub mod barrier;
pub mod basket;
pub mod bermudan;
pub mod black76;
pub mod cliquet;
pub mod commodity;
pub mod convertible;
pub mod digital;
pub mod double_barrier;
pub mod employee_stock_option;
pub mod exotic;
pub mod fx;
pub mod mbs;
pub mod power;
pub mod rainbow;
pub mod range_accrual;
pub mod real_option;
pub mod spread;
pub mod swing;
pub mod tarf;
pub mod vanilla;
pub mod variance_swap;
pub mod weather;

pub use asian::AsianOption;
pub use autocallable::{Autocallable, PhoenixAutocallable};
pub use barrier::{BarrierOption, BarrierOptionBuilder};
pub use basket::{BasketOption, BasketType};
pub use bermudan::BermudanOption;
pub use black76::FuturesOption;
pub use cliquet::{CliquetOption, ForwardStartOption};
pub use commodity::{CommodityForward, CommodityFutures, CommodityOption, CommoditySpreadOption};
pub use convertible::ConvertibleBond;
pub use digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
pub use double_barrier::{DoubleBarrierOption, DoubleBarrierType};
pub use employee_stock_option::EmployeeStockOption;
pub use exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFixedOption, LookbackFloatingOption,
    QuantoOption,
};
pub use fx::FxOption;
pub use mbs::{ConstantCpr, MbsCashflow, MbsPassThrough, PrepaymentModel, PsaModel};
pub use power::PowerOption;
pub use rainbow::{BestOfTwoCallOption, TwoAssetCorrelationOption, WorstOfTwoCallOption};
pub use range_accrual::{DualRangeAccrual, RangeAccrual};
pub use real_option::{
    AbandonmentOption, DeferInvestmentOption, DiscreteCashFlow, ExpandOption,
    RealOptionBinomialSpec, RealOptionInstrument,
};
pub use spread::SpreadOption;
pub use swing::SwingOption;
pub use tarf::{Tarf, TarfType};
pub use vanilla::VanillaOption;
pub use variance_swap::{VarianceOptionQuote, VarianceSwap, VolatilitySwap};
pub use weather::{
    CatastropheBond, DegreeDayType, WeatherOption, WeatherSwap, cdd_day, cumulative_cdd,
    cumulative_degree_days, cumulative_hdd, hdd_day,
};

/// Trade-level metadata for serialization and audit trails.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TradeMetadata {
    pub trade_id: String,
    pub version: u64,
    pub timestamp_unix_ms: i64,
}

/// Tagged instrument payload for trade serialization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum TradeInstrument {
    AsianOption(AsianOption),
    Autocallable(Autocallable),
    PhoenixAutocallable(PhoenixAutocallable),
    BarrierOption(BarrierOption),
    BasketOption(BasketOption),
    BermudanOption(BermudanOption),
    FuturesOption(FuturesOption),
    CliquetOption(CliquetOption),
    ForwardStartOption(ForwardStartOption),
    CommodityForward(CommodityForward),
    CommodityFutures(CommodityFutures),
    CommodityOption(CommodityOption),
    CommoditySpreadOption(CommoditySpreadOption),
    ConvertibleBond(ConvertibleBond),
    CashOrNothingOption(CashOrNothingOption),
    AssetOrNothingOption(AssetOrNothingOption),
    GapOption(GapOption),
    DoubleBarrierOption(DoubleBarrierOption),
    EmployeeStockOption(EmployeeStockOption),
    LookbackFloatingOption(LookbackFloatingOption),
    LookbackFixedOption(LookbackFixedOption),
    ChooserOption(ChooserOption),
    QuantoOption(QuantoOption),
    CompoundOption(CompoundOption),
    ExoticOption(ExoticOption),
    FxOption(FxOption),
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
    SpreadOption(SpreadOption),
    SwingOption(SwingOption),
    Tarf(Tarf),
    VanillaOption(VanillaOption),
    VarianceSwap(VarianceSwap),
    VolatilitySwap(VolatilitySwap),
    WeatherSwap(WeatherSwap),
    WeatherOption(WeatherOption),
    CatastropheBond(CatastropheBond),
    MbsPassThrough(MbsPassThrough),
}

/// Trade record container.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    pub metadata: TradeMetadata,
    pub instrument: TradeInstrument,
}

/// Portfolio container for batch valuation with shared market data references.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Portfolio {
    pub portfolio_id: String,
    pub market_snapshot_id: Option<String>,
    pub trades: Vec<Trade>,
}
