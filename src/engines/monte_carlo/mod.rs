//! Monte Carlo pricing module map for path simulation, variance reduction, and acceleration.
//!
//! Core generic flow lives in `mc_engine` with the [`VarianceReduction`] enum:
//! - Antithetic variates: paired shocks reduce estimator variance for many smooth payoffs.
//! - Control variates: built-in analytic controls (for example Black-Scholes vanilla and
//!   geometric-Asian controls) reduce Monte Carlo noise without increasing path count.
//! - Plain Monte Carlo: baseline estimator for products without convenient controls.
//!
//! Additional submodules:
//! - `mc_qmc`: Sobol-based quasi-Monte Carlo for lower-discrepancy convergence.
//! - `mc_simd`: structure-of-arrays GBM kernels with optional AVX2/FMA acceleration.
//! - `mc_parallel` (feature `parallel`): rayon-parallel path chunking and grid calculations.
//! - `mc_greeks`: finite-difference/pathwise Greek estimation helpers.
//! - `spread_mc`: dedicated spread-option Monte Carlo engine.
//!
//! Convergence notes: standard error scales roughly as `O(N^-1/2)` for pseudo-random MC;
//! QMC and variance-reduction methods can materially improve constants in practice.

pub mod mc_engine;
pub mod mc_greeks;
#[cfg(feature = "parallel")]
pub mod mc_parallel;
pub mod mc_qmc;
pub mod mc_simd;
pub mod spread_mc;

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
