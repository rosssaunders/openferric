//! Risk analytics: VaR/ES, XVA, and portfolio-level aggregation.

pub mod fva;
pub mod kva;
pub mod mva;
pub mod portfolio;
pub mod var;
pub mod xva;

pub use fva::{CsaTerms, fva_from_profile, funding_exposure_profile};
pub use kva::{
    SaCcrAssetClass, kva_from_profile, netting_set_exposure, regulatory_capital, sa_ccr_ead,
};
pub use mva::{SimmMargin, SimmRiskClass, mva_from_profile};
pub use portfolio::{AggregatedGreeks, Portfolio, Position};
pub use var::{
    cornish_fisher_var, cornish_fisher_var_from_pnl, delta_gamma_normal_var, delta_normal_var,
    historical_expected_shortfall, historical_var, normal_expected_shortfall,
};
pub use xva::XvaCalculator;
