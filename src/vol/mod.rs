pub mod andreasen_huge;
pub mod builder;
pub mod fengler;
pub mod implied;
pub mod jaeckel;
pub mod local_vol;
pub mod mixture;
pub mod sabr;
pub mod smile;
pub mod surface;

/// Arbitrage violation types for vol surface validation.
#[derive(Debug, Clone)]
pub enum ArbitrageViolation {
    /// Butterfly arbitrage: negative risk-neutral density.
    Butterfly {
        strike: f64,
        expiry: f64,
        density: f64,
    },
    /// Calendar arbitrage: total variance decreasing in time.
    Calendar {
        strike: f64,
        t1: f64,
        t2: f64,
        dw_dt: f64,
    },
}
