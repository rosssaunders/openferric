//! Module `engines::analytic::bachelier`.
//!
//! Implements bachelier workflows with concrete routines such as `bachelier_price`, `bachelier_greeks`.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Primary API surface: free functions `bachelier_price`, `bachelier_greeks`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use crate::core::{Greeks, OptionType, PricingError};
use crate::math::{normal_cdf, normal_pdf};

#[inline]
fn intrinsic(option_type: OptionType, forward: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (forward - strike).max(0.0),
        OptionType::Put => (strike - forward).max(0.0),
    }
}

#[inline]
fn normalised_moneyness(forward: f64, strike: f64, standard_deviation: f64) -> f64 {
    let difference = forward - strike;
    if difference.is_finite() {
        difference / standard_deviation
    } else {
        forward / standard_deviation - strike / standard_deviation
    }
}

/// Dimensionless value of an out-of-the-money Bachelier option:
/// `phi(x) - x Phi(-x)`, for `x >= 0`.
#[inline]
fn standard_normal_otm_value(x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    if x < 8.0 {
        return normal_pdf(x).mul_add(1.0, -(x * normal_cdf(-x)));
    }

    // From the asymptotic Mills-ratio expansion:
    // 1 - x R(x) ~ x^-2 - 3x^-4 + 15x^-6 - 105x^-8 + ...
    // Stop at the least term because this is an asymptotic, not convergent,
    // series.
    let inverse_sq = 1.0 / (x * x);
    let mut term = inverse_sq;
    let mut sum = term;
    let mut previous_abs = term.abs();
    let mut odd = 3.0;
    for _ in 0..128 {
        let next = term * (-odd * inverse_sq);
        if next.abs() >= previous_abs {
            break;
        }
        let updated = sum + next;
        if updated == sum {
            break;
        }
        sum = updated;
        term = next;
        previous_abs = next.abs();
        odd += 2.0;
    }
    normal_pdf(x) * sum
}

/// Bachelier (normal) model price for European options on forwards.
#[allow(clippy::too_many_arguments)]
#[inline]
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
    if denom == 0.0 {
        return Ok(df * intrinsic(option_type, forward, strike));
    }
    let difference = forward - strike;
    let d = normalised_moneyness(forward, strike, denom);
    let otm = df * denom * standard_normal_otm_value(d.abs());
    let price = match option_type {
        OptionType::Call if difference > 0.0 => df * difference + otm,
        OptionType::Put if difference < 0.0 => -df * difference + otm,
        _ => otm,
    };

    if !price.is_finite() || price < 0.0 {
        return Err(PricingError::NumericalError(format!(
            "bachelier price invariant failed: price={price}"
        )));
    }
    Ok(price)
}

/// Bachelier (normal) model Greeks for European options on forwards.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn bachelier_greeks(
    option_type: OptionType,
    forward: f64,
    strike: f64,
    r: f64,
    sigma_n: f64,
    t: f64,
) -> Result<Greeks, PricingError> {
    let price = bachelier_price(option_type, forward, strike, r, sigma_n, t)?;
    if t <= 0.0 {
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
    if sigma_n == 0.0 || denom == 0.0 {
        let payoff_moneyness = match option_type {
            OptionType::Call => forward - strike,
            OptionType::Put => strike - forward,
        };
        let delta = match payoff_moneyness.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => match option_type {
                OptionType::Call => df,
                OptionType::Put => -df,
            },
            Some(std::cmp::Ordering::Less) => 0.0,
            Some(std::cmp::Ordering::Equal) | None => f64::NAN,
        };
        let at_boundary = forward == strike;
        return Ok(Greeks {
            delta,
            gamma: if at_boundary { f64::NAN } else { 0.0 },
            // At F=K this is the right derivative as sigma_n approaches zero.
            vega: if at_boundary {
                df * sqrt_t * normal_pdf(0.0)
            } else {
                0.0
            },
            theta: r * price,
            rho: -t * price,
        });
    }
    let d = normalised_moneyness(forward, strike, denom);

    let nd = normal_cdf(d);
    let pdf_d = normal_pdf(d);
    let delta = match option_type {
        OptionType::Call => df * nd,
        OptionType::Put => -df * normal_cdf(-d),
    };
    let gamma = df * pdf_d / denom;
    let vega = df * sqrt_t * pdf_d;
    let theta = r.mul_add(price, -(df * sigma_n * pdf_d / (2.0 * sqrt_t)));

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

    #[test]
    fn deep_otm_put_matches_high_precision_reference_and_homogeneity() {
        // 100-decimal erfc reference for d=9.
        const EXPECTED: f64 = 1.224_779_180_843_489_7;
        let scaled = bachelier_price(OptionType::Put, 1.0e21, 1.0e20, 0.0, 1.0e20, 1.0).unwrap();
        let unit = bachelier_price(OptionType::Put, 10.0, 1.0, 0.0, 1.0, 1.0).unwrap();
        assert!(scaled > 0.0);
        assert!(unit > 0.0);
        assert!(
            ((scaled - EXPECTED) / EXPECTED).abs() <= 2.0e-14,
            "scaled={scaled:.17e}"
        );
        assert!(
            ((scaled / 1.0e20 - unit) / unit).abs() <= 2.0e-14,
            "scaled={scaled:.17e} unit={unit:.17e}"
        );
    }

    #[test]
    fn zero_and_underflowed_width_return_deterministic_limit_greeks() {
        let r = 0.05_f64;
        let t = 2.0_f64;
        let df = (-r * t).exp();

        for (option_type, forward, expected_delta) in
            [(OptionType::Call, 2.0, df), (OptionType::Put, 0.0, -df)]
        {
            let greeks = bachelier_greeks(option_type, forward, 1.0, r, 0.0, t).unwrap();
            let price = bachelier_price(option_type, forward, 1.0, r, 0.0, t).unwrap();
            assert_eq!(greeks.delta, expected_delta);
            assert_eq!(greeks.gamma, 0.0);
            assert_eq!(greeks.vega, 0.0);
            assert_eq!(greeks.theta, r * price);
            assert_eq!(greeks.rho, -t * price);
        }

        // sigma_n and sqrt(T) are individually nonzero, but their product
        // underflows. The option remains an ITM discounted-forward payoff.
        let underflow =
            bachelier_greeks(OptionType::Call, 2.0, 1.0, 0.03, 1.0e-300, 1.0e-100).unwrap();
        assert_eq!(underflow.delta, 1.0);
        assert_eq!(underflow.gamma, 0.0);
        assert_eq!(underflow.vega, 0.0);
        assert_eq!(underflow.theta, 0.03);
        assert_eq!(underflow.rho, -1.0e-100);
    }

    #[test]
    fn zero_width_at_the_strike_exposes_the_one_sided_vega_and_undefined_spot_greeks() {
        let r = 0.05_f64;
        let t = 2.0_f64;
        let df = (-r * t).exp();
        let greeks = bachelier_greeks(OptionType::Call, 1.0, 1.0, r, 0.0, t).unwrap();
        assert!(greeks.delta.is_nan());
        assert!(greeks.gamma.is_nan());
        assert_eq!(greeks.vega, df * t.sqrt() * 0.398_942_280_401_432_7);
        assert_eq!(greeks.theta, 0.0);
        assert_eq!(greeks.rho, 0.0);
    }
}
