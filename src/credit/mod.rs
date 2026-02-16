//! Credit-risk primitives including survival curves and CDS analytics.

pub mod bootstrap;
pub mod cdo;
pub mod cds;
pub mod cds_index;
pub mod copula;
pub mod isda;
pub mod survival_curve;

pub use bootstrap::bootstrap_survival_curve_from_cds_spreads;
pub use cdo::{CdoTranche, SyntheticCdo, vasicek_portfolio_loss_cdf};
pub use cds::Cds;
pub use cds_index::{CdsIndex, NthToDefaultBasket, first_to_default_spread_copula};
pub use copula::{BasketDefaultSimulation, GaussianCopula};
pub use isda::{
    CdsDateRule, CdsPriceResult, DatedCds, IsdaConventions, ProtectionSide, cash_settle_date,
    generate_imm_schedule, hazard_from_par_spread, next_imm_twentieth, previous_imm_twentieth,
    price_isda_flat, price_midpoint_flat, step_in_date,
};
pub use survival_curve::SurvivalCurve;
