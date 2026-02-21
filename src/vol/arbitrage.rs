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
