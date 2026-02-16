//! Instrument definitions.

pub mod asian;
pub mod barrier;
pub mod cliquet;
pub mod digital;
pub mod exotic;
pub mod fx;
pub mod spread;
pub mod vanilla;

pub use asian::AsianOption;
pub use barrier::{BarrierOption, BarrierOptionBuilder};
pub use cliquet::{CliquetOption, ForwardStartOption};
pub use digital::{AssetOrNothingOption, CashOrNothingOption, GapOption};
pub use exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFixedOption, LookbackFloatingOption,
    QuantoOption,
};
pub use fx::FxOption;
pub use spread::SpreadOption;
pub use vanilla::VanillaOption;
