//! Module `models::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
pub mod cgmy;
pub mod commodity;
pub mod hjm;
pub mod hw_calibration;
pub mod lmm;
pub mod nig;
pub mod rough_bergomi;
pub mod short_rate;
pub mod slv;
pub mod stochastic;
pub mod variance_gamma;

pub use cgmy::Cgmy;
pub use commodity::{
    CommodityForwardCurve, CommoditySeasonalityModel, CommodityStorageContract, CurveStructure,
    ForwardInterpolation, FuturesQuote, SchwartzOneFactor, SchwartzSmithTwoFactor, SeasonalityMode,
    StorageLsmConfig, StorageValuation, TwoFactorCommodityProcess, TwoFactorSpreadModel,
    VolumeConstrainedSwing, convenience_yield_from_term_structure, implied_convenience_yield,
    intrinsic_storage_value, value_storage_intrinsic_extrinsic,
};
pub use hjm::{HjmFactor, HjmFactorShape, HjmModel};
pub use hw_calibration::{
    AtmSwaptionVolQuote, calibrate_hull_white_params, hw_atm_swaption_vol_approx,
};
pub use lmm::{LmmModel, LmmParams, black_swaption_price, initial_swap_rate_annuity};
pub use nig::Nig;
pub use rough_bergomi::{
    FbmScheme, fbm_covariance, fbm_path_cholesky, fbm_path_hybrid, rbergomi_european_mc,
    rbergomi_implied_vol_surface,
};
pub use short_rate::{CIR, HullWhite, Vasicek};
pub use slv::{
    LeverageSlice, LeverageSurface, SlvParams, calibrate_leverage_surface,
    nadaraya_watson_conditional_mean, slv_mc_price, slv_mc_price_checked,
};
pub use stochastic::*;
pub use variance_gamma::VarianceGamma;
