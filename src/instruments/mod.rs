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
pub mod mbs;
pub mod autocallable;
pub mod barrier;
pub mod basket;
pub mod black76;
pub mod cliquet;
pub mod commodity;
pub mod convertible;
pub mod digital;
pub mod double_barrier;
pub mod employee_stock_option;
pub mod exotic;
pub mod fx;
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
pub use power::PowerOption;
pub use range_accrual::{DualRangeAccrual, RangeAccrual};
pub use rainbow::{BestOfTwoCallOption, TwoAssetCorrelationOption, WorstOfTwoCallOption};
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
