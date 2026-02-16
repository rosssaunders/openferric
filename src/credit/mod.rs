//! Credit-risk primitives including survival curves and CDS analytics.

pub mod cds;
pub mod survival_curve;

pub use cds::Cds;
pub use survival_curve::SurvivalCurve;
