//! Monte Carlo pricing engines.

pub mod mc_engine;
pub mod mc_greeks;
#[cfg(feature = "parallel")]
pub mod mc_parallel;
pub mod mc_qmc;
pub mod mc_simd;
pub mod spread_mc;

pub use mc_engine::{
    ArithmeticAsianMC, MonteCarloInstrument, MonteCarloPricingEngine, VarianceReduction,
};
pub use mc_greeks::MonteCarloGreeksEngine;
#[cfg(feature = "parallel")]
pub use mc_parallel::{
    mc_european_parallel, mc_european_sequential, mc_greeks_grid_parallel,
    mc_greeks_grid_sequential, GreeksGridPoint,
};
pub use mc_qmc::{mc_european_qmc, mc_european_qmc_with_seed};
pub use mc_simd::{
    mc_european_call_soa, mc_european_call_soa_scalar, simulate_gbm_paths_soa,
    simulate_gbm_paths_soa_scalar, SoaPaths,
};
pub use spread_mc::SpreadMonteCarloEngine;
