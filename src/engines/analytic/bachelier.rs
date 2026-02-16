use crate::core::{Greeks, OptionType, PricingError};
use crate::math::{normal_cdf, normal_pdf};

fn intrinsic(option_type: OptionType, forward: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (forward - strike).max(0.0),
        OptionType::Put => (strike - forward).max(0.0),
    }
}

/// Bachelier (normal) model price for European options on forwards.
#[allow(clippy::too_many_arguments)]
pub fn bachelier_price(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    r: f64,
    sigma_n: f64,
    t: f64,
) -> Result<f64, PricingError> {
    if !forward.is_finite() || !strike.is_finite() {
        return Err(PricingError::InvalidInput(
            "bachelier forward and strike must be finite".to_string(),
        ));
    }
    if !r.is_finite() {
        return Err(PricingError::InvalidInput(
            "bachelier r must be finite".to_string(),
        ));
    }
    if !sigma_n.is_finite() || sigma_n < 0.0 {
        return Err(PricingError::InvalidInput(
            "bachelier sigma_n must be finite and >= 0".to_string(),
        ));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(PricingError::InvalidInput(
            "bachelier t must be finite and >= 0".to_string(),
        ));
    }

    if t <= 0.0 {
        return Ok(intrinsic(option_type, forward, strike));
    }

    let df = (-r * t).exp();
    if sigma_n <= 0.0 {
        return Ok(df * intrinsic(option_type, forward, strike));
    }

    let sqrt_t = t.sqrt();
    let denom = sigma_n * sqrt_t;
    let d = (forward - strike) / denom;

    let price = match option_type {
        OptionType::Call => df * ((forward - strike) * normal_cdf(d) + denom * normal_pdf(d)),
        OptionType::Put => df * ((strike - forward) * normal_cdf(-d) + denom * normal_pdf(d)),
    };

    Ok(price)
}

/// Bachelier (normal) model Greeks for European options on forwards.
#[allow(clippy::too_many_arguments)]
pub fn bachelier_greeks(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    r: f64,
    sigma_n: f64,
    t: f64,
) -> Result<Greeks, PricingError> {
    let price = bachelier_price(option_type, forward, strike, r, sigma_n, t)?;
    if t <= 0.0 || sigma_n <= 0.0 {
        return Ok(Greeks {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        });
    }

    let df = (-r * t).exp();
    let sqrt_t = t.sqrt();
    let denom = sigma_n * sqrt_t;
    let d = (forward - strike) / denom;

    let delta = match option_type {
        OptionType::Call => df * normal_cdf(d),
        OptionType::Put => df * (normal_cdf(d) - 1.0),
    };
    let gamma = df * normal_pdf(d) / denom;
    let vega = df * sqrt_t * normal_pdf(d);
    let theta = r * price - df * sigma_n * normal_pdf(d) / (2.0 * sqrt_t);

    // Sensitivity to r for fixed forward input.
    let rho = -t * price;

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn atm_reference_value() {
        let call = bachelier_price(OptionType::Call, 100.0, 100.0, 0.05, 20.0, 1.0).unwrap();
        assert_relative_eq!(call, 7.589_712_748_1, epsilon = 1e-6);
    }

    #[test]
    fn put_call_parity_holds() {
        let f = 100.0;
        let k = 95.0;
        let r = 0.03;
        let sigma_n = 12.0;
        let t = 1.4;

        let c = bachelier_price(OptionType::Call, f, k, r, sigma_n, t).unwrap();
        let p = bachelier_price(OptionType::Put, f, k, r, sigma_n, t).unwrap();

        assert_relative_eq!(c - p, (-r * t).exp() * (f - k), epsilon = 2e-10);
    }

    #[test]
    fn greeks_match_finite_differences() {
        let f = 100.0;
        let k = 98.0;
        let r = 0.01;
        let sigma_n = 15.0;
        let t = 1.25;

        let g = bachelier_greeks(OptionType::Call, f, k, r, sigma_n, t).unwrap();

        let eps_f = 1.0e-4;
        let p_up = bachelier_price(OptionType::Call, f + eps_f, k, r, sigma_n, t).unwrap();
        let p_dn = bachelier_price(OptionType::Call, f - eps_f, k, r, sigma_n, t).unwrap();
        let p_0 = bachelier_price(OptionType::Call, f, k, r, sigma_n, t).unwrap();

        let delta_fd = (p_up - p_dn) / (2.0 * eps_f);
        let gamma_fd = (p_up - 2.0 * p_0 + p_dn) / (eps_f * eps_f);
        assert_relative_eq!(g.delta, delta_fd, epsilon = 2e-6);
        assert_relative_eq!(g.gamma, gamma_fd, epsilon = 2e-6);

        let eps_vol = 1.0e-5;
        let v_up = bachelier_price(OptionType::Call, f, k, r, sigma_n + eps_vol, t).unwrap();
        let v_dn = bachelier_price(OptionType::Call, f, k, r, sigma_n - eps_vol, t).unwrap();
        let vega_fd = (v_up - v_dn) / (2.0 * eps_vol);
        assert_relative_eq!(g.vega, vega_fd, epsilon = 2e-6);

        let eps_t = 1.0e-5;
        let t_up = bachelier_price(OptionType::Call, f, k, r, sigma_n, t + eps_t).unwrap();
        let t_dn = bachelier_price(OptionType::Call, f, k, r, sigma_n, t - eps_t).unwrap();
        let theta_fd = -(t_up - t_dn) / (2.0 * eps_t);
        assert_relative_eq!(g.theta, theta_fd, epsilon = 2e-6);
    }
}
