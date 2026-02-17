//! Instrument definitions.

pub mod asian;
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
pub mod real_option;
pub mod spread;
pub mod swing;
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
pub use rainbow::{BestOfTwoCallOption, TwoAssetCorrelationOption, WorstOfTwoCallOption};
pub use real_option::{
    AbandonmentOption, DeferInvestmentOption, DiscreteCashFlow, ExpandOption,
    RealOptionBinomialSpec, RealOptionInstrument,
};
pub use spread::SpreadOption;
pub use swing::SwingOption;
pub use vanilla::VanillaOption;
pub use variance_swap::{VarianceOptionQuote, VarianceSwap, VolatilitySwap};
pub use weather::{
    CatastropheBond, DegreeDayType, WeatherOption, WeatherSwap, cdd_day, cumulative_cdd,
    cumulative_degree_days, cumulative_hdd, hdd_day,
};
