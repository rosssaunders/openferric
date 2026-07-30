//! Module `greeks::bsm`.
//!
//! Implements bsm workflows with concrete routines such as `black_scholes_merton_greeks`, `bump_and_reprice`, `finite_difference_greeks`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `FiniteDifferenceGreeks`, `EuropeanBsmGreeks`, `GenericPricer` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: choose this module when its API directly matches your instrument/model assumptions; otherwise use a more specialized engine module.
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
    crate::engines::analytic::bs_inline::stable_d1_d2(s, k, r, q, sigma, t)
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
    if !s.is_finite()
        || !k.is_finite()
        || !r.is_finite()
        || !q.is_finite()
        || !sigma.is_finite()
        || !t.is_finite()
        || s < 0.0
        || k < 0.0
        || sigma < 0.0
        || (s == 0.0 && k == 0.0)
    {
        return EuropeanBsmGreeks {
            delta: f64::NAN,
            gamma: f64::NAN,
            vega: f64::NAN,
            theta: f64::NAN,
            rho: f64::NAN,
            vanna: f64::NAN,
            volga: f64::NAN,
        };
    }
    if t <= 0.0 {
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
    if sigma == 0.0 || s == 0.0 || k == 0.0 {
        use crate::engines::analytic::black_scholes::{
            bs_delta, bs_gamma, bs_rho, bs_theta, bs_vega,
        };

        let delta = bs_delta(option_type, s, k, r, q, sigma, t);
        let gamma = bs_gamma(s, k, r, q, sigma, t);
        let vega = bs_vega(s, k, r, q, sigma, t);
        let theta = bs_theta(option_type, s, k, r, q, sigma, t);
        let rho = bs_rho(option_type, s, k, r, q, sigma, t);
        let undefined = [delta, gamma, vega, theta, rho]
            .iter()
            .any(|value| value.is_nan());
        return EuropeanBsmGreeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
            vanna: if undefined { f64::NAN } else { 0.0 },
            volga: if undefined { f64::NAN } else { 0.0 },
        };
    }

    let (d1, d2) = d1_d2_with_dividend(s, k, r, q, sigma, t);
    let sqrt_t = t.sqrt();
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();

    let delta = match option_type {
        OptionType::Call => df_q * normal_cdf(d1),
        OptionType::Put => -df_q * normal_cdf(-d1),
    };

    let gamma = crate::engines::analytic::bs_inline::stable_gamma(
        df_q * normal_pdf(d1),
        s,
        sigma,
        sqrt_t,
        d1,
        -q * t,
    );
    let vega = s * df_q * normal_pdf(d1) * sqrt_t;
    let theta_common = crate::engines::analytic::bs_inline::stable_theta_diffusion(
        s * df_q * normal_pdf(d1),
        sigma,
        sqrt_t,
    );

    let theta = match option_type {
        OptionType::Call => {
            theta_common - r * k * df_r * normal_cdf(d2) + q * s * df_q * normal_cdf(d1)
        }
        OptionType::Put => {
            theta_common + r * k * df_r * normal_cdf(-d2) - q * s * df_q * normal_cdf(-d1)
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

/// Second derivative on a possibly non-uniform 3-point stencil
/// `(x - h_dn, x, x + h_up)`. Exact for quadratics for any `h_up, h_dn > 0`.
/// Returns 0 when the down-spacing collapses (degenerate base point at the
/// domain boundary), where no second-order information is available.
#[inline]
fn second_derivative_nonuniform(p_up: f64, p0: f64, p_dn: f64, h_up: f64, h_dn: f64) -> f64 {
    if h_dn <= 0.0 || h_up <= 0.0 {
        return 0.0;
    }
    2.0 * (h_dn * p_up - (h_up + h_dn) * p0 + h_up * p_dn) / (h_up * h_dn * (h_up + h_dn))
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
    // Down bumps are floored to keep the pricer in its valid domain, but never
    // above the base point. All difference quotients below use the ACTUAL
    // spacings, so a clamped (asymmetric) stencil is still consistent instead
    // of being scaled as if it were symmetric.
    let s_up = s + ds;
    let s_dn = (s - ds).max(1e-8).min(s);
    let v_up = sigma + dv;
    let v_dn = (sigma - dv).max(1e-8).min(sigma);
    let t_up = t + dt;
    let t_dn = (t - dt).max(1e-8).min(t);

    let hs_up = s_up - s;
    let hs_dn = s - s_dn;
    let hv_up = v_up - sigma;
    let hv_dn = sigma - v_dn;

    let p0 = pricer.price(s, k, r, sigma, t);

    let p_s_up = pricer.price(s_up, k, r, sigma, t);
    let p_s_dn = pricer.price(s_dn, k, r, sigma, t);

    let p_r_up = pricer.price(s, k, r + dr, sigma, t);
    let p_r_dn = pricer.price(s, k, r - dr, sigma, t);

    let p_v_up = pricer.price(s, k, r, v_up, t);
    let p_v_dn = pricer.price(s, k, r, v_dn, t);

    let p_t_up = pricer.price(s, k, r, sigma, t_up);
    let p_t_dn = pricer.price(s, k, r, sigma, t_dn);

    let p_s_up_v_up = pricer.price(s_up, k, r, v_up, t);
    let p_s_up_v_dn = pricer.price(s_up, k, r, v_dn, t);
    let p_s_dn_v_up = pricer.price(s_dn, k, r, v_up, t);
    let p_s_dn_v_dn = pricer.price(s_dn, k, r, v_dn, t);

    // First derivatives over the actual (possibly one-sided) spans. The up
    // bumps are never clamped, so these denominators are strictly positive.
    let delta = (p_s_up - p_s_dn) / (s_up - s_dn);
    let gamma = second_derivative_nonuniform(p_s_up, p0, p_s_dn, hs_up, hs_dn);
    let rho = (p_r_up - p_r_dn) / (2.0 * dr);
    let vega = (p_v_up - p_v_dn) / (v_up - v_dn);
    // Market convention: theta is dV/d(calendar time), i.e. -dV/d(maturity).
    let theta = -(p_t_up - p_t_dn) / (t_up - t_dn);
    let vanna =
        (p_s_up_v_up - p_s_up_v_dn - p_s_dn_v_up + p_s_dn_v_dn) / ((s_up - s_dn) * (v_up - v_dn));
    let volga = second_derivative_nonuniform(p_v_up, p0, p_v_dn, hv_up, hv_dn);

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
    fn finite_difference_greeks_are_finite_near_degenerate_inputs() {
        let pricer = |s: f64, k: f64, r: f64, v: f64, t: f64| {
            black_scholes_price(OptionType::Call, s, k, r, v, t)
        };

        // sigma -> 0 and t -> 0: clamped down-bumps used to be divided by the
        // nominal symmetric spacing, corrupting the Greeks.
        let (s, k, r, sigma, t) = (100.0, 90.0, 0.05, 1.0e-9, 1.0e-9);
        let fd = finite_difference_greeks(&pricer, s, k, r, sigma, t, 1e-3, 1e-4, 1e-4, 1e-4);

        for (name, v) in [
            ("delta", fd.delta),
            ("gamma", fd.gamma),
            ("vega", fd.vega),
            ("theta", fd.theta),
            ("rho", fd.rho),
            ("vanna", fd.vanna),
            ("volga", fd.volga),
        ] {
            assert!(v.is_finite(), "{name}={v} not finite");
        }

        // Analytic deterministic limits for a deep-ITM call as sigma, t -> 0:
        // delta -> 1, gamma -> 0, vega -> 0, theta -> -r*K (continuous comp).
        assert!((fd.delta - 1.0).abs() < 1e-6, "delta={}", fd.delta);
        assert!(fd.gamma.abs() < 1e-6, "gamma={}", fd.gamma);
        assert!(fd.vega.abs() < 1e-3, "vega={}", fd.vega);
        assert!((fd.theta + r * k).abs() < 1e-2, "theta={}", fd.theta);
        assert!(fd.volga.abs() < 1.0, "volga={}", fd.volga);
        assert!(fd.vanna.abs() < 1.0, "vanna={}", fd.vanna);
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

    #[test]
    fn zero_volatility_greeks_use_deterministic_limits() {
        let (r, q, t, k) = (0.03_f64, 0.01_f64, 1.5_f64, 100.0_f64);
        let df_r = (-r * t).exp();
        let df_q = (-q * t).exp();

        let put = black_scholes_merton_greeks(OptionType::Put, 80.0, k, r, q, 0.0, t);
        assert_eq!(put.delta, -df_q);
        assert_eq!(put.gamma, 0.0);
        assert_eq!(put.vega, 0.0);
        assert_eq!(put.theta, -q * 80.0 * df_q + r * k * df_r);
        assert_eq!(put.rho, -k * t * df_r);
        assert_eq!(put.vanna, 0.0);
        assert_eq!(put.volga, 0.0);

        let kink = black_scholes_merton_greeks(OptionType::Call, k, k, 0.0, 0.0, 0.0, t);
        for value in [
            kink.delta, kink.gamma, kink.vega, kink.theta, kink.rho, kink.vanna, kink.volga,
        ] {
            assert!(value.is_nan());
        }

        let undefined_origin =
            black_scholes_merton_greeks(OptionType::Call, 0.0, 0.0, r, q, 0.2, t);
        for value in [
            undefined_origin.delta,
            undefined_origin.gamma,
            undefined_origin.vega,
            undefined_origin.theta,
            undefined_origin.rho,
            undefined_origin.vanna,
            undefined_origin.volga,
        ] {
            assert!(value.is_nan());
        }
    }
}
