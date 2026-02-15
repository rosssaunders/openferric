#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
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

    (base, s_up, r_up, v_up, t_up, pricer.price(s - bump_s, k, r, sigma, t))
}

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
    let p0 = pricer.price(s, k, r, sigma, t);

    let p_s_up = pricer.price(s + ds, k, r, sigma, t);
    let p_s_dn = pricer.price((s - ds).max(1e-8), k, r, sigma, t);

    let p_r_up = pricer.price(s, k, r + dr, sigma, t);
    let p_r_dn = pricer.price(s, k, r - dr, sigma, t);

    let p_v_up = pricer.price(s, k, r, sigma + dv, t);
    let p_v_dn = pricer.price(s, k, r, (sigma - dv).max(1e-8), t);

    let p_t_up = pricer.price(s, k, r, sigma, t + dt);
    let p_t_dn = pricer.price(s, k, r, sigma, (t - dt).max(1e-8));

    let delta = (p_s_up - p_s_dn) / (2.0 * ds);
    let gamma = (p_s_up - 2.0 * p0 + p_s_dn) / (ds * ds);
    let rho = (p_r_up - p_r_dn) / (2.0 * dr);
    let vega = (p_v_up - p_v_dn) / (2.0 * dv);
    let theta = (p_t_up - p_t_dn) / (2.0 * dt);

    FiniteDifferenceGreeks {
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
    use crate::pricing::OptionType;
    use crate::pricing::european::{black_scholes_greeks, black_scholes_price};

    #[test]
    fn finite_difference_matches_closed_form_greeks() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let pricer = |ss: f64, kk: f64, rr: f64, vv: f64, tt: f64| {
            black_scholes_price(OptionType::Call, ss, kk, rr, vv, tt)
        };

        let fd = finite_difference_greeks(&pricer, s, k, r, sigma, t, 1e-3, 1e-4, 1e-4, 1e-4);
        let cf = black_scholes_greeks(OptionType::Call, s, k, r, sigma, t);

        assert!((fd.delta - cf.delta).abs() < 2e-4);
        assert!((fd.gamma - cf.gamma).abs() < 2e-4);
        assert!((fd.vega - cf.vega).abs() < 4e-3);
        assert!((fd.rho - cf.rho).abs() < 5e-3);
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
}
