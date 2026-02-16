//! Credit-risk primitives including survival curves and CDS analytics.

pub mod bootstrap;
pub mod cdo;
pub mod cds;
pub mod copula;
pub mod survival_curve;

pub use bootstrap::bootstrap_survival_curve_from_cds_spreads;
pub use cdo::{CdoTranche, SyntheticCdo, vasicek_portfolio_loss_cdf};
pub use cds::Cds;
pub use copula::{BasketDefaultSimulation, GaussianCopula};
pub use survival_curve::SurvivalCurve;
