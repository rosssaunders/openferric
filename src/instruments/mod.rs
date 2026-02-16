//! Instrument definitions.

pub mod asian;
pub mod barrier;
pub mod cliquet;
pub mod convertible;
pub mod digital;
pub mod double_barrier;
pub mod exotic;
pub mod fx;
pub mod power;
pub mod rainbow;
pub mod spread;
pub mod vanilla;
pub mod variance_swap;

pub use asian::AsianOption;
pub use barrier::{BarrierOption, BarrierOptionBuilder};
pub use cliquet::{CliquetOption, ForwardStartOption};
pub use convertible::ConvertibleBond;
pub use digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
pub use double_barrier::{DoubleBarrierOption, DoubleBarrierType};
pub use exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFixedOption, LookbackFloatingOption,
    QuantoOption,
};
pub use fx::FxOption;
pub use power::PowerOption;
pub use rainbow::{BestOfTwoCallOption, TwoAssetCorrelationOption, WorstOfTwoCallOption};
pub use spread::SpreadOption;
pub use vanilla::VanillaOption;
pub use variance_swap::{VarianceOptionQuote, VarianceSwap, VolatilitySwap};
