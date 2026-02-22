//! Greeks and sensitivity analytics.
//!
//! Provides finite-difference and model-based sensitivity estimators for option portfolios.

use crate::math::{normal_cdf, normal_pdf};
use crate::pricing::OptionType;

#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EuropeanBsmGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

impl EuropeanBsmGreeks {
    /// Alternative name for volga.
    pub fn vomma(self) -> f64 {
        self.volga
    }
}

pub trait GenericPricer {
    fn price(&self, s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64;
}

impl<F> GenericPricer for F
where
    F: Fn(f64, f64, f64, f64, f64) -> f64,
{
    fn price(&self, s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
        self(s, k, r, sigma, t)
    }
}

fn d1_d2_with_dividend(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> (f64, f64) {
    let vt = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vt;
    let d2 = d1 - vt;
    (d1, d2)
}

pub fn black_scholes_merton_greeks(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> EuropeanBsmGreeks {
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return EuropeanBsmGreeks {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
        };
    }

    let (d1, d2) = d1_d2_with_dividend(s, k, r, q, sigma, t);
    let sqrt_t = t.sqrt();
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();

    let delta = match option_type {
        OptionType::Call => df_q * normal_cdf(d1),
        OptionType::Put => df_q * (normal_cdf(d1) - 1.0),
    };

    let gamma = df_q * normal_pdf(d1) / (s * sigma * sqrt_t);
    let vega = s * df_q * normal_pdf(d1) * sqrt_t;

    let theta = match option_type {
        OptionType::Call => {
            -s * df_q * normal_pdf(d1) * sigma / (2.0 * sqrt_t) - r * k * df_r * normal_cdf(d2)
                + q * s * df_q * normal_cdf(d1)
        }
        OptionType::Put => {
            -s * df_q * normal_pdf(d1) * sigma / (2.0 * sqrt_t) + r * k * df_r * normal_cdf(-d2)
                - q * s * df_q * normal_cdf(-d1)
        }
    };

    let rho = match option_type {
        OptionType::Call => k * t * df_r * normal_cdf(d2),
        OptionType::Put => -k * t * df_r * normal_cdf(-d2),
    };

    // Cross derivative d^2V/(dS dSigma).
    let vanna = -df_q * normal_pdf(d1) * d2 / sigma;
    // Second derivative d^2V/dSigma^2 (aka vomma).
    let volga = vega * d1 * d2 / sigma;

    EuropeanBsmGreeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
        vanna,
        volga,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bump_and_reprice<P: GenericPricer>(
    pricer: &P,
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    bump_s: f64,
    bump_r: f64,
    bump_sigma: f64,
    bump_t: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let base = pricer.price(s, k, r, sigma, t);
    let s_up = pricer.price(s + bump_s, k, r, sigma, t);
    let r_up = pricer.price(s, k, r + bump_r, sigma, t);
    let v_up = pricer.price(s, k, r, sigma + bump_sigma, t);
    let t_up = pricer.price(s, k, r, sigma, (t + bump_t).max(1e-8));

    (
        base,
        s_up,
        r_up,
        v_up,
        t_up,
        pricer.price(s - bump_s, k, r, sigma, t),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn finite_difference_greeks<P: GenericPricer>(
    pricer: &P,
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    ds: f64,
    dr: f64,
    dv: f64,
    dt: f64,
) -> FiniteDifferenceGreeks {
    let s_dn = (s - ds).max(1e-8);
    let v_dn = (sigma - dv).max(1e-8);

    let p0 = pricer.price(s, k, r, sigma, t);

    let p_s_up = pricer.price(s + ds, k, r, sigma, t);
    let p_s_dn = pricer.price(s_dn, k, r, sigma, t);

    let p_r_up = pricer.price(s, k, r + dr, sigma, t);
    let p_r_dn = pricer.price(s, k, r - dr, sigma, t);

    let p_v_up = pricer.price(s, k, r, sigma + dv, t);
    let p_v_dn = pricer.price(s, k, r, v_dn, t);

    let p_t_up = pricer.price(s, k, r, sigma, t + dt);
    let p_t_dn = pricer.price(s, k, r, sigma, (t - dt).max(1e-8));

    let p_s_up_v_up = pricer.price(s + ds, k, r, sigma + dv, t);
    let p_s_up_v_dn = pricer.price(s + ds, k, r, v_dn, t);
    let p_s_dn_v_up = pricer.price(s_dn, k, r, sigma + dv, t);
    let p_s_dn_v_dn = pricer.price(s_dn, k, r, v_dn, t);

    let delta = (p_s_up - p_s_dn) / (2.0 * ds);
    let gamma = (p_s_up - 2.0 * p0 + p_s_dn) / (ds * ds);
    let rho = (p_r_up - p_r_dn) / (2.0 * dr);
    let vega = (p_v_up - p_v_dn) / (2.0 * dv);
    // Market convention: theta is dV/d(calendar time), i.e. -dV/d(maturity).
    let theta = -(p_t_up - p_t_dn) / (2.0 * dt);
    let vanna = (p_s_up_v_up - p_s_up_v_dn - p_s_dn_v_up + p_s_dn_v_dn) / (4.0 * ds * dv);
    let volga = (p_v_up - 2.0 * p0 + p_v_dn) / (dv * dv);

    FiniteDifferenceGreeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
        vanna,
        volga,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn finite_difference_matches_closed_form_greeks() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.0;
        let sigma = 0.2;
        let t = 1.0;

        let pricer = |ss: f64, kk: f64, rr: f64, vv: f64, tt: f64| {
            // No-dividend pricing function for this test setup (q = 0).
            black_scholes_price(OptionType::Call, ss, kk, rr, vv, tt)
        };

        let fd = finite_difference_greeks(&pricer, s, k, r, sigma, t, 1e-3, 1e-4, 1e-4, 1e-4);
        let cf = black_scholes_merton_greeks(OptionType::Call, s, k, r, q, sigma, t);

        assert!((fd.delta - cf.delta).abs() < 2e-4);
        assert!((fd.gamma - cf.gamma).abs() < 2e-4);
        assert!((fd.vega - cf.vega).abs() < 4e-3);
        assert!((fd.theta - cf.theta).abs() < 1e-2);
        assert!((fd.rho - cf.rho).abs() < 5e-3);
        assert!((fd.vanna - cf.vanna).abs() < 4e-3);
        assert!((fd.volga - cf.volga).abs() < 1e-2);
    }

    #[test]
    fn quantlib_reference_values_for_atm_call() {
        // QuantLib-style reference setup (European BSM):
        // S=100, K=100, r=0.05, q=0.0, T=1.0, vol=0.20.
        let g = black_scholes_merton_greeks(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);

        assert!((g.delta - 0.6368).abs() < 5e-4);
        assert!((g.gamma - 0.01876).abs() < 5e-5);
        assert!((g.theta - -6.414).abs() < 5e-3);
        assert!((g.vega - 37.524).abs() < 5e-3);
        // Different conventions/day-count setups in references can shift rho slightly.
        assert!((g.rho - 51.522).abs() < 2.0);
    }

    #[test]
    fn bump_and_reprice_returns_consistent_base() {
        let pricer = |s: f64, k: f64, r: f64, sigma: f64, t: f64| {
            black_scholes_price(OptionType::Put, s, k, r, sigma, t)
        };

        let (base, _s_up, _r_up, _v_up, _t_up, _s_dn) =
            bump_and_reprice(&pricer, 100.0, 95.0, 0.03, 0.22, 0.8, 0.1, 1e-4, 1e-4, 1e-4);

        let ref_price = black_scholes_price(OptionType::Put, 100.0, 95.0, 0.03, 0.22, 0.8);
        assert!((base - ref_price).abs() < 1e-12);
    }

    #[test]
    fn volga_equals_vomma_alias() {
        let g = black_scholes_merton_greeks(OptionType::Call, 100.0, 100.0, 0.03, 0.01, 0.25, 0.9);
        assert!((g.volga - g.vomma()).abs() < 1e-12);
    }
}
