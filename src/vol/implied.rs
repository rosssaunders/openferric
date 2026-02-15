use crate::math::normal_pdf;
use crate::pricing::OptionType;
use crate::pricing::european::black_scholes_price;
use std::f64::consts::PI;

pub fn lets_be_rational_initial_guess(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    t: f64,
    market_price: f64,
) -> f64 {
    if t <= 0.0 {
        return 1e-4;
    }

    let df = (-r * t).exp();
    let intrinsic = match option_type {
        OptionType::Call => (s - k * df).max(0.0),
        OptionType::Put => (k * df - s).max(0.0),
    };

    let time_value = (market_price - intrinsic).max(1e-10);
    let atm_guess = ((2.0 * PI) / t).sqrt() * (time_value / s.max(1e-10));
    let m = (s / k).ln().abs();

    // Jaeckel-inspired rational-style scaling on moneyness/time value.
    let scaled = atm_guess * (1.0 + 0.5 * m + 0.125 * m * m);
    scaled.clamp(1e-4, 5.0)
}

#[allow(clippy::too_many_arguments)]
pub fn implied_vol_newton(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    t: f64,
    market_price: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    if t <= 0.0 {
        return Err("t must be > 0".to_string());
    }
    if market_price <= 0.0 {
        return Err("market_price must be > 0".to_string());
    }

    let mut sigma = lets_be_rational_initial_guess(option_type, s, k, r, t, market_price);

    for _ in 0..max_iter {
        let price = black_scholes_price(option_type, s, k, r, sigma, t);
        let diff = price - market_price;
        if diff.abs() < tol {
            return Ok(sigma);
        }

        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let vega = s * normal_pdf(d1) * t.sqrt();

        if vega.abs() < 1e-10 {
            break;
        }

        sigma = (sigma - diff / vega).clamp(1e-6, 5.0);
    }

    // Robust fallback: bisection on volatility interval.
    let mut lo = 1e-6;
    let mut hi = 5.0;
    let mut flo = black_scholes_price(option_type, s, k, r, lo, t) - market_price;

    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let fm = black_scholes_price(option_type, s, k, r, mid, t) - market_price;

        if fm.abs() < tol {
            return Ok(mid);
        }

        if flo * fm <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }

    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn implied_vol_recovers_true_sigma_call() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let t = 1.0;
        let sigma = 0.2;

        let price = black_scholes_price(OptionType::Call, s, k, r, sigma, t);
        let iv = implied_vol_newton(OptionType::Call, s, k, r, t, price, 1e-12, 50).unwrap();

        assert_relative_eq!(iv, sigma, epsilon = 1e-8);
    }

    #[test]
    fn implied_vol_recovers_true_sigma_put() {
        let s = 100.0;
        let k = 110.0;
        let r = 0.02;
        let t = 0.75;
        let sigma = 0.35;

        let price = black_scholes_price(OptionType::Put, s, k, r, sigma, t);
        let iv = implied_vol_newton(OptionType::Put, s, k, r, t, price, 1e-11, 100).unwrap();

        assert_relative_eq!(iv, sigma, epsilon = 1e-7);
    }

    #[test]
    fn lets_be_rational_guess_is_positive_and_reasonable() {
        let guess = lets_be_rational_initial_guess(OptionType::Call, 100.0, 100.0, 0.01, 1.0, 8.0);
        assert!(guess > 0.0);
        assert!(guess < 2.0);
    }
}
