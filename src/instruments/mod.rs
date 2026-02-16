//! Instrument definitions.

pub mod asian;
pub mod barrier;
pub mod exotic;
pub mod vanilla;

pub use asian::AsianOption;
pub use barrier::{BarrierOption, BarrierOptionBuilder};
pub use exotic::{
    ChooserOption, CompoundOption, ExoticOption, LookbackFloatingOption, QuantoOption,
};
pub use vanilla::VanillaOption;
