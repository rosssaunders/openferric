//! Black implied-volatility inversion routines.
//!
//! Primary path uses a Jaeckel-style normalized implied-vol solver with fallback to
//! Newton + bisection. See Peter Jaeckel, "Let's Be Rational" (2015).

use crate::math::normal_pdf;
use crate::pricing::OptionType;
use crate::pricing::european::black_scholes_price;
use crate::vol::jaeckel::implied_vol_jaeckel;
#[cfg(test)]
use rand::RngExt;
use std::f64::consts::PI;

/// Heuristic initial guess for implied volatility.
///
/// Inputs use Black-Scholes notation:
/// `s` spot, `k` strike, `r` continuously-compounded rate, `t` expiry in years.
/// `market_price` is option premium in currency units.
///
/// The guess is shaped by time value and log-moneyness to reduce Newton iterations.
///
/// # Numerical notes
/// The output is clamped to `[1e-4, 5.0]`.
///
/// # Examples
/// ```
/// use openferric::pricing::OptionType;
/// use openferric::vol::implied::lets_be_rational_initial_guess;
///
/// let guess = lets_be_rational_initial_guess(OptionType::Call, 100.0, 100.0, 0.02, 1.0, 8.0);
/// assert!(guess > 0.0);
/// ```
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
/// Computes Black implied volatility from market option price.
///
/// Parameters:
/// - `option_type`: call/put flag
/// - `s`, `k`, `r`, `t`: Black-Scholes state and maturity
/// - `market_price`: observed premium
/// - `tol`: target absolute pricing error
/// - `max_iter`: Newton iteration cap before fallback
///
/// # Returns
/// Implied volatility as a non-negative scalar.
///
/// # Errors
/// Returns `Err` if inputs are non-finite, violate positivity constraints,
/// or if price is outside no-arbitrage bounds.
///
/// # Numerical stability
/// - Returns `0.0` when price is effectively intrinsic.
/// - Attempts a Jaeckel solver path first; falls back to Newton + bisection.
///
/// # Examples
/// ```
/// use openferric::pricing::OptionType;
/// use openferric::pricing::european::black_scholes_price;
/// use openferric::vol::implied::implied_vol;
///
/// let sigma = 0.25;
/// let price = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.03, sigma, 1.0);
/// let iv = implied_vol(OptionType::Call, 100.0, 100.0, 0.03, 1.0, price, 1e-12, 64).unwrap();
/// assert!((iv - sigma).abs() < 1e-8);
/// ```
pub fn implied_vol(
    option_type: OptionType,
    s: f64,
    k: f64,
    r: f64,
    t: f64,
    market_price: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    if !s.is_finite()
        || !k.is_finite()
        || !r.is_finite()
        || !t.is_finite()
        || !market_price.is_finite()
    {
        return Err("inputs must be finite".to_string());
    }
    if s <= 0.0 || k <= 0.0 {
        return Err("s and k must be > 0".to_string());
    }
    if t <= 0.0 {
        return Err("t must be > 0".to_string());
    }
    if market_price < 0.0 {
        return Err("market_price must be >= 0".to_string());
    }

    let df = (-r * t).exp();
    let intrinsic = match option_type {
        OptionType::Call => (s - k * df).max(0.0),
        OptionType::Put => (k * df - s).max(0.0),
    };
    let upper = match option_type {
        OptionType::Call => s,
        OptionType::Put => k * df,
    };
    let price_tol = 32.0 * f64::EPSILON * (1.0 + upper.abs());
    if market_price < intrinsic - price_tol || market_price > upper + price_tol {
        return Err(format!(
            "market_price out of no-arbitrage bounds: price={market_price}, intrinsic={intrinsic}, upper={upper}"
        ));
    }
    if market_price <= intrinsic + price_tol {
        return Ok(0.0);
    }

    let growth = (r * t).exp();
    if growth.is_finite() {
        let forward = s * growth;
        let undiscounted_price = market_price * growth;
        let is_call = matches!(option_type, OptionType::Call);
        if let Ok(iv) = implied_vol_jaeckel(undiscounted_price, forward, k, t, is_call)
            && iv.is_finite()
            && iv >= 0.0
        {
            return Ok(iv);
        }
    }

    implied_vol_newton(option_type, s, k, r, t, market_price, tol, max_iter)
}

#[allow(clippy::too_many_arguments)]
/// Newton-Raphson implied-vol solver with robust bisection fallback.
///
/// This is a direct inversion of Black-Scholes price using vega for Newton steps.
///
/// # Limitations
/// Newton can stagnate when vega is tiny (deep ITM/OTM short-dated options);
/// in that case the implementation falls back to bisection.
///
/// # Examples
/// ```
/// use openferric::pricing::OptionType;
/// use openferric::pricing::european::black_scholes_price;
/// use openferric::vol::implied::implied_vol_newton;
///
/// let sigma = 0.35;
/// let price = black_scholes_price(OptionType::Put, 100.0, 110.0, 0.01, sigma, 0.75);
/// let iv = implied_vol_newton(OptionType::Put, 100.0, 110.0, 0.01, 0.75, price, 1e-12, 100)
///     .unwrap();
/// assert!((iv - sigma).abs() < 1e-7);
/// ```
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
    if market_price < 0.0 {
        return Err("market_price must be >= 0".to_string());
    }
    if market_price == 0.0 {
        return Ok(0.0);
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
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

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

    #[test]
    fn implied_vol_round_trip_reprices_market_price() {
        let option_type = OptionType::Call;
        let s = 100.0;
        let k = 105.0;
        let r = 0.03;
        let t = 1.4;
        let sigma = 0.28;

        let market_price = black_scholes_price(option_type, s, k, r, sigma, t);
        let iv = implied_vol_newton(option_type, s, k, r, t, market_price, 1e-12, 100).unwrap();
        let repriced = black_scholes_price(option_type, s, k, r, iv, t);

        assert_relative_eq!(repriced, market_price, epsilon = 1e-9);
    }

    #[test]
    fn jaeckel_main_path_round_trip_to_machine_precision() {
        let cases = [
            (OptionType::Call, 100.0, 80.0, 0.03, 0.2, 0.5),
            (OptionType::Call, 100.0, 100.0, 0.01, 0.3, 1.0),
            (OptionType::Call, 100.0, 120.0, 0.02, 0.25, 2.0),
            (OptionType::Put, 100.0, 80.0, 0.03, 0.2, 0.5),
            (OptionType::Put, 100.0, 100.0, 0.01, 0.3, 1.0),
            (OptionType::Put, 100.0, 120.0, 0.02, 0.25, 2.0),
        ];

        for (option_type, s, k, r, sigma, t) in cases {
            let price = black_scholes_price(option_type, s, k, r, sigma, t);
            let iv = implied_vol(option_type, s, k, r, t, price, 1e-14, 32).unwrap();
            let repriced = black_scholes_price(option_type, s, k, r, iv, t);
            let err = (repriced - price).abs();
            assert!(
                err <= 1e-12 * price.max(1.0),
                "option={option_type:?} s={s} k={k} r={r} t={t} sigma={sigma} iv={iv} err={err}"
            );
        }
    }

    #[test]
    fn jaeckel_matches_newton_for_normal_cases() {
        let strikes = [85.0, 95.0, 100.0, 105.0, 115.0];
        let expiries = [0.25, 1.0, 3.0];
        let sigmas = [0.1, 0.2, 0.4];

        for &option_type in &[OptionType::Call, OptionType::Put] {
            for &k in &strikes {
                for &t in &expiries {
                    for &sigma in &sigmas {
                        let s = 100.0;
                        let r = 0.015;
                        let price = black_scholes_price(option_type, s, k, r, sigma, t);
                        let iv_j = implied_vol(option_type, s, k, r, t, price, 1e-12, 64).unwrap();
                        let iv_n =
                            implied_vol_newton(option_type, s, k, r, t, price, 1e-12, 64).unwrap();
                        assert!(
                            (iv_j - iv_n).abs() < 1e-10,
                            "option={option_type:?} k={k} t={t} sigma={sigma} iv_j={iv_j} iv_n={iv_n}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn jaeckel_handles_extreme_moneyness_ratios() {
        let s = 100.0;
        let r = 0.0;
        let t = 30.0;
        let sigma = 0.8;

        let extreme_ratios = [0.01, 100.0];
        for &ratio in &extreme_ratios {
            let k = s * ratio;
            for &option_type in &[OptionType::Call, OptionType::Put] {
                let price = black_scholes_price(option_type, s, k, r, sigma, t);
                let iv = implied_vol(option_type, s, k, r, t, price, 1e-12, 64).unwrap();
                assert!(
                    (iv - sigma).abs() < 1e-8,
                    "ratio={ratio} option={option_type:?} iv={iv} sigma={sigma}"
                );
            }
        }
    }

    #[test]
    fn jaeckel_handles_short_and_long_expiry() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.01;
        let sigma = 0.25;
        for &t in &[0.001, 30.0] {
            let price = black_scholes_price(OptionType::Call, s, k, r, sigma, t);
            let iv = implied_vol(OptionType::Call, s, k, r, t, price, 1e-12, 64).unwrap();
            assert!((iv - sigma).abs() < 1e-10, "t={t} iv={iv} sigma={sigma}");
        }
    }

    #[test]
    fn jaeckel_handles_near_zero_vol() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.0;
        let t = 1.0;
        let sigma = 0.001;
        let price = black_scholes_price(OptionType::Call, s, k, r, sigma, t);
        let iv = implied_vol(OptionType::Call, s, k, r, t, price, 1e-14, 64).unwrap();
        assert!((iv - sigma).abs() < 1e-9, "iv={iv} sigma={sigma}");
    }

    #[test]
    #[ignore = "performance benchmark; run with --ignored"]
    fn benchmark_jaeckel_vs_newton_for_10k_random_options() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut inputs = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let s = 100.0;
            let k = s * rng.random_range(0.5..1.5);
            let r = rng.random_range(-0.01..0.05);
            let t = rng.random_range(0.05..5.0);
            let sigma = rng.random_range(0.05..1.2);
            let option_type = if rng.random_bool(0.5) {
                OptionType::Call
            } else {
                OptionType::Put
            };
            let price = black_scholes_price(option_type, s, k, r, sigma, t);
            inputs.push((option_type, s, k, r, t, price));
        }

        let start_j = Instant::now();
        let mut sum_j = 0.0;
        for &(option_type, s, k, r, t, price) in &inputs {
            sum_j += implied_vol(option_type, s, k, r, t, price, 1e-12, 64).unwrap();
        }
        let dur_j = start_j.elapsed();

        let start_n = Instant::now();
        let mut sum_n = 0.0;
        for &(option_type, s, k, r, t, price) in &inputs {
            sum_n += implied_vol_newton(option_type, s, k, r, t, price, 1e-12, 64).unwrap();
        }
        let dur_n = start_n.elapsed();

        assert!(sum_j.is_finite() && sum_n.is_finite());
        assert!(
            dur_j < dur_n,
            "expected jaeckel path to be faster: jaeckel={dur_j:?}, newton={dur_n:?}"
        );
    }
}
