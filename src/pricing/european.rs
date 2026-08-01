//! Module `pricing::european`.
//!
//! Implements european workflows with concrete routines such as `black_scholes_price`, `black_76_price`, `black_scholes_greeks`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `Greeks` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::engines::analytic::black_scholes::{
    bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega,
};
use crate::pricing::OptionType;

#[derive(Debug, Clone, Copy)]
/// First-order and second-order sensitivities for a European option under BSM assumptions.
///
/// The fields correspond to:
/// - `delta = dV/dS`
/// - `gamma = d²V/dS²`
/// - `vega = dV/dσ`
/// - `theta = dV/dt`
/// - `rho = dV/dr`
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

/// Black-Scholes-Merton spot-option price with zero dividend yield.
///
/// Parameters:
/// - `option_type`: call or put payoff direction.
/// - `s`: current spot price.
/// - `k`: strike price.
/// - `r`: continuously compounded risk-free rate.
/// - `sigma`: annualized volatility.
/// - `t`: time to expiry in years.
///
/// Edge cases:
/// - Delegates to kernel logic that handles `t <= 0` or `sigma <= 0` by intrinsic value.
///
/// # Examples
/// ```rust
/// use openferric::core::OptionType;
/// use openferric::pricing::european::black_scholes_price;
///
/// let call = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
/// let put = black_scholes_price(OptionType::Put, 100.0, 100.0, 0.05, 0.20, 1.0);
/// assert!(call > put);
/// ```
pub fn black_scholes_price(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    // Compatibility path delegates to the optimized engine kernel (q = 0.0).
    bs_price(option_type, s, k, r, 0.0, sigma, t)
}

/// Black-76 price for options on forwards/futures.
///
/// Parameters:
/// - `f`: forward/futures level.
/// - other parameters follow the same units as [`black_scholes_price`].
///
/// Edge cases:
/// - Returns discounted intrinsic value when `t <= 0` or `sigma <= 0`.
///
/// # Examples
/// ```rust
/// use openferric::core::OptionType;
/// use openferric::pricing::european::black_76_price;
///
/// let call = black_76_price(OptionType::Call, 103.0, 100.0, 0.03, 0.18, 1.0);
/// let put = black_76_price(OptionType::Put, 103.0, 100.0, 0.03, 0.18, 1.0);
/// assert!(call > 0.0 && put > 0.0);
/// ```
pub fn black_76_price(option_type: OptionType, f: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    crate::engines::analytic::black76_price(option_type, f, k, r, sigma, t).unwrap_or(f64::NAN)
}

/// Computes Black-Scholes Greeks with zero dividend yield.
///
/// Parameters match [`black_scholes_price`].
///
/// # Examples
/// ```rust
/// use openferric::core::OptionType;
/// use openferric::pricing::european::black_scholes_greeks;
///
/// let g = black_scholes_greeks(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
/// assert!(g.delta > 0.0 && g.gamma > 0.0 && g.vega > 0.0);
/// ```
pub fn black_scholes_greeks(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> Greeks {
    // Compatibility path delegates to optimized engine kernels (q = 0.0).
    let delta = bs_delta(option_type, s, k, r, 0.0, sigma, t);
    let gamma = bs_gamma(s, k, r, 0.0, sigma, t);
    let vega = bs_vega(s, k, r, 0.0, sigma, t);
    let theta = bs_theta(option_type, s, k, r, 0.0, sigma, t);
    let rho = bs_rho(option_type, s, k, r, 0.0, sigma, t);
    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_to_scipy(actual: f64, expected: f64, label: &str) {
        let tolerance = 256.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.15}, SciPy={expected:.15}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn black_scholes_known_value() {
        let call = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        // scipy.stats.norm 1.17.1 evaluated from the closed-form BSM equation.
        assert_close_to_scipy(call, 10.450583572185565, "ATM call");

        let put = black_scholes_price(OptionType::Put, 100.0, 100.0, 0.05, 0.2, 1.0);
        assert_close_to_scipy(put, 5.573526022256971, "ATM put");
    }

    #[test]
    fn put_call_parity_black_scholes() {
        let s = 100.0;
        let k = 95.0;
        let r = 0.03;
        let sigma = 0.22;
        let t = 1.4;

        let c = black_scholes_price(OptionType::Call, s, k, r, sigma, t);
        let p = black_scholes_price(OptionType::Put, s, k, r, sigma, t);
        let rhs = s - k * (-r * t).exp();

        let roundoff = 64.0 * f64::EPSILON * c.abs().max(p.abs()).max(rhs.abs()).max(1.0);
        assert!(((c - p) - rhs).abs() <= roundoff);
    }

    #[test]
    fn black_76_put_call_parity() {
        let f = 103.0;
        let k = 100.0;
        let r = 0.04;
        let sigma = 0.18;
        let t = 0.75;

        let c = black_76_price(OptionType::Call, f, k, r, sigma, t);
        let p = black_76_price(OptionType::Put, f, k, r, sigma, t);

        let rhs = (-r * t).exp() * (f - k);
        let roundoff = 64.0 * f64::EPSILON * c.abs().max(p.abs()).max(rhs.abs()).max(1.0);
        assert!(((c - p) - rhs).abs() <= roundoff);
    }

    #[test]
    fn greeks_match_scipy_closed_form_reference() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;
        let g = black_scholes_greeks(OptionType::Call, s, k, r, sigma, t);

        // Independent scipy.stats.norm CDF/PDF references for d1=0.35, d2=0.15.
        assert_close_to_scipy(g.delta, 0.6368306511756191, "delta");
        assert_close_to_scipy(g.gamma, 0.018762017345846895, "gamma");
        assert_close_to_scipy(g.vega, 37.52403469169379, "vega");
        assert_close_to_scipy(g.theta, -6.414027546438197, "theta");
        assert_close_to_scipy(g.rho, 53.232481545376345, "rho");
    }
}
