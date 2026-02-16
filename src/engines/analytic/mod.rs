//! Closed-form analytic pricing engines.

pub mod asian_geometric;
pub mod barrier_analytic;
pub mod black_scholes;
pub mod digital;
pub mod exotic;
pub mod fx;
pub mod heston;
pub mod spread;

pub use asian_geometric::GeometricAsianEngine;
pub use barrier_analytic::BarrierAnalyticEngine;
pub use black_scholes::{BlackScholesEngine, black_scholes};
pub use digital::DigitalAnalyticEngine;
pub use exotic::ExoticAnalyticEngine;
pub use fx::{FxGreeks, GarmanKohlhagenEngine};
pub use heston::HestonEngine;
pub use spread::{
    SpreadAnalyticEngine, SpreadAnalyticMethod, kirk_spread_price, margrabe_exchange_price,
};
