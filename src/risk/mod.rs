//! Top-level risk namespace for market, XVA, and sensitivity analytics.
//!
//! This module wires and re-exports:
//! - `var`: historical/parametric VaR and ES,
//! - `portfolio` + `sensitivities`: position aggregation, finite-difference Greeks, scenario
//!   explain, and lightweight SIMM/FRTB-style charge aggregation,
//! - `scenarios`: scenario-definition, market-diff, and stress testing engine with
//!   attribution tables,
//! - `xva`, `fva`, `mva`, `kva`, `wrong_way_risk`: valuation adjustments and WWR models.
//!
//! It is intentionally a facade: domain logic lives in submodules, while this file defines the
//! public import surface (`openferric::risk::*`) for downstream code.

pub mod fva;
pub mod kva;
pub mod mva;
pub mod portfolio;
pub mod scenarios;
pub mod sensitivities;
pub mod var;
pub mod wrong_way_risk;
pub mod xva;

pub use fva::{CsaTerms, funding_exposure_profile, fva_from_profile};
pub use kva::{
    SaCcrAssetClass, kva_from_profile, netting_set_exposure, regulatory_capital, sa_ccr_ead,
};
pub use mva::{SimmMargin, SimmRiskClass, mva_from_profile};
pub use portfolio::{AggregatedGreeks, Portfolio, Position};
pub use scenarios::*;
pub use sensitivities::*;
pub use var::{
    backtest_historical_var_from_prices, cornish_fisher_var, cornish_fisher_var_from_pnl,
    delta_gamma_normal_var, delta_normal_var, historical_expected_shortfall,
    historical_expected_shortfall_from_prices, historical_var, historical_var_from_prices,
    normal_expected_shortfall, rolling_historical_var_from_prices,
};
pub use xva::XvaCalculator;
