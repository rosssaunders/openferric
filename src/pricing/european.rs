use crate::math::{normal_cdf, normal_pdf};
use crate::pricing::OptionType;

#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

fn d1_d2(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> (f64, f64) {
    let vt = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / vt;
    let d2 = d1 - vt;
    (d1, d2)
}

pub fn black_scholes_price(option_type: OptionType, s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return match option_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };
    }

    let (d1, d2) = d1_d2(s, k, r, sigma, t);
    let df = (-r * t).exp();
    match option_type {
        OptionType::Call => s * normal_cdf(d1) - k * df * normal_cdf(d2),
        OptionType::Put => k * df * normal_cdf(-d2) - s * normal_cdf(-d1),
    }
}

pub fn black_76_price(option_type: OptionType, f: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (-r * t).exp()
            * match option_type {
                OptionType::Call => (f - k).max(0.0),
                OptionType::Put => (k - f).max(0.0),
            };
    }

    let vt = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / vt;
    let d2 = d1 - vt;
    let df = (-r * t).exp();

    match option_type {
        OptionType::Call => df * (f * normal_cdf(d1) - k * normal_cdf(d2)),
        OptionType::Put => df * (k * normal_cdf(-d2) - f * normal_cdf(-d1)),
    }
}

pub fn black_scholes_greeks(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
) -> Greeks {
    if t <= 0.0 || sigma <= 0.0 {
        return Greeks {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }

    let (d1, d2) = d1_d2(s, k, r, sigma, t);
    let sqrt_t = t.sqrt();
    let df = (-r * t).exp();

    let delta = match option_type {
        OptionType::Call => normal_cdf(d1),
        OptionType::Put => normal_cdf(d1) - 1.0,
    };

    let gamma = normal_pdf(d1) / (s * sigma * sqrt_t);
    let vega = s * normal_pdf(d1) * sqrt_t;

    let theta = match option_type {
        OptionType::Call => {
            -s * normal_pdf(d1) * sigma / (2.0 * sqrt_t) - r * k * df * normal_cdf(d2)
        }
        OptionType::Put => {
            -s * normal_pdf(d1) * sigma / (2.0 * sqrt_t) + r * k * df * normal_cdf(-d2)
        }
    };

    let rho = match option_type {
        OptionType::Call => k * t * df * normal_cdf(d2),
        OptionType::Put => -k * t * df * normal_cdf(-d2),
    };

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
    use approx::assert_relative_eq;

    #[test]
    fn black_scholes_known_value() {
        let call = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        assert_relative_eq!(call, 10.4506, epsilon = 2e-4);

        let put = black_scholes_price(OptionType::Put, 100.0, 100.0, 0.05, 0.2, 1.0);
        assert_relative_eq!(put, 5.5735, epsilon = 2e-4);
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

        assert_relative_eq!(c - p, rhs, epsilon = 2e-6);
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

        assert_relative_eq!(c - p, (-r * t).exp() * (f - k), epsilon = 2e-6);
    }

    #[test]
    fn greeks_are_consistent_with_finite_differences() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;
        let ds = 1e-3;

        let g = black_scholes_greeks(OptionType::Call, s, k, r, sigma, t);

        let p_up = black_scholes_price(OptionType::Call, s + ds, k, r, sigma, t);
        let p_dn = black_scholes_price(OptionType::Call, s - ds, k, r, sigma, t);
        let p_0 = black_scholes_price(OptionType::Call, s, k, r, sigma, t);

        let delta_fd = (p_up - p_dn) / (2.0 * ds);
        let gamma_fd = (p_up - 2.0 * p_0 + p_dn) / (ds * ds);

        assert_relative_eq!(g.delta, delta_fd, epsilon = 1e-4);
        assert_relative_eq!(g.gamma, gamma_fd, epsilon = 1e-4);
    }
}
