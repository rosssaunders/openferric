//! Module `engines::monte_carlo::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.

pub mod correlated_mc;
pub mod mc_aad;
pub mod mc_engine;
pub mod mc_greeks;
#[cfg(feature = "parallel")]
pub mod mc_parallel;
pub mod mc_qmc;
pub mod mc_simd;
pub mod spread_mc;

pub use correlated_mc::{
    cholesky_for_correlation, sample_correlated_normals_cholesky, sample_correlated_normals_factor,
};
pub use mc_aad::{HestonAadConfig, heston_price_delta_aad, mc_european_pathwise_aad};
pub use mc_engine::{
    ArithmeticAsianMC, MonteCarloInstrument, MonteCarloPricingEngine, VarianceReduction,
    mc_european_with_arena,
};
pub use mc_greeks::MonteCarloGreeksEngine;
#[cfg(feature = "parallel")]
pub use mc_parallel::{
    GreeksGridPoint, mc_european_parallel, mc_european_sequential, mc_greeks_grid_parallel,
    mc_greeks_grid_sequential,
};
pub use mc_qmc::{mc_european_qmc, mc_european_qmc_with_seed};
pub use mc_simd::{
    SoaPaths, mc_european_call_soa, mc_european_call_soa_scalar, simulate_gbm_paths_soa,
    simulate_gbm_paths_soa_scalar,
};
pub use spread_mc::SpreadMonteCarloEngine;
