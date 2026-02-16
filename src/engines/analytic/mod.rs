//! Closed-form analytic pricing engines.

pub mod asian_geometric;
pub mod bachelier;
pub mod barrier_analytic;
pub mod binary_barrier;
pub mod black76;
pub mod black_scholes;
pub mod bs_simd;
pub mod digital;
pub mod double_barrier;
pub mod exotic;
pub mod fx;
pub mod heston;
pub mod power;
pub mod rainbow;
pub mod spread;
pub mod variance_swap;

pub use asian_geometric::GeometricAsianEngine;
pub use bachelier::{bachelier_greeks, bachelier_price};
pub use barrier_analytic::BarrierAnalyticEngine;
pub use binary_barrier::{BinaryBarrierType, cash_or_nothing_barrier_price};
pub use black_scholes::{BlackScholesEngine, black_scholes};
pub use black76::{Black76Engine, black76_greeks, black76_price};
pub use bs_simd::{bs_greeks_batch, bs_price_batch, normal_cdf_approx, normal_cdf_batch_approx};
pub use digital::DigitalAnalyticEngine;
pub use double_barrier::DoubleBarrierAnalyticEngine;
pub use exotic::ExoticAnalyticEngine;
pub use fx::{FxGreeks, GarmanKohlhagenEngine};
pub use heston::HestonEngine;
pub use power::{PowerOptionEngine, power_option_price};
pub use rainbow::{
    RainbowAnalyticEngine, best_of_two_call_price, two_asset_correlation_price,
    worst_of_two_call_price,
};
pub use spread::{
    SpreadAnalyticEngine, SpreadAnalyticMethod, kirk_spread_price, margrabe_exchange_price,
};
pub use variance_swap::{
    VarianceSwapEngine, fair_variance_strike_from_quotes, fair_volatility_strike_from_variance,
    variance_swap_mtm, volatility_swap_mtm,
};
