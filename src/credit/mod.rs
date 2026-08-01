//! Module `credit::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 24-25, O'Kane (2008) Ch. 3, representative cashflow identities as in Eq. (24.7) and Eq. (25.5).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: interpolation/extrapolation and day-count conventions materially affect PVs; handle near-zero rates/hazards to avoid cancellation.
//!
//! When to use: use these routines for CDS/tranche and survival-curve workflows; consider structural credit models when capital-structure dynamics are required explicitly.

pub mod bootstrap;
pub mod cdo;
pub mod cds;
pub mod cds_index;
pub mod cds_option;
pub mod copula;
pub mod heterogeneous_cdo;
pub mod isda;
pub mod survival_curve;

pub use bootstrap::bootstrap_survival_curve_from_cds_spreads;
pub use cdo::{CdoTranche, SyntheticCdo, vasicek_portfolio_loss_cdf};
pub use cds::Cds;
pub use cds_index::{CdsIndex, NthToDefaultBasket, first_to_default_spread_copula};
pub use copula::{BasketDefaultSimulation, GaussianCopula};
pub use heterogeneous_cdo::{BaseCorrelationPair, CdoReferenceName, HeterogeneousSyntheticCdo};
pub use isda::{
    CdsDateRule, CdsPriceResult, DatedCds, IsdaConventions, ProtectionSide, cash_settle_date,
    cash_settle_date_with_calendar, generate_imm_schedule, hazard_from_par_spread,
    next_imm_twentieth, previous_imm_twentieth, price_isda_flat, price_isda_flat_legacy_analytic,
    price_isda_flat_with_calendar, price_midpoint_flat, price_midpoint_flat_with_calendar,
    step_in_date, step_in_date_with_calendar,
};
pub use survival_curve::SurvivalCurve;
