//! Volatility analytics for Jaeckel.
//!
//! Module openferric::vol::jaeckel provides smile/surface construction or volatility inversion utilities.

use crate::core::PricingError;
use crate::math::{normal_cdf, normal_inv_cdf, normal_pdf};

const MAX_SIGMA_SQRT_T: f64 = 64.0;
const MAX_HOUSEHOLDER_ITERS: usize = 8;

#[derive(Debug, Clone, Copy)]
struct BlackEval {
    price: f64,
    first: f64,
    second: f64,
    third: f64,
}

#[inline]
fn normalized_intrinsic(x: f64, is_call: bool) -> f64 {
    let parity = (0.5 * x).exp() - (-0.5 * x).exp();
    if is_call {
        parity.max(0.0)
    } else {
        (-parity).max(0.0)
    }
}

#[inline]
fn normalized_upper_bound(x: f64, is_call: bool) -> f64 {
    if is_call {
        (0.5 * x).exp()
    } else {
        (-0.5 * x).exp()
    }
}

#[inline]
fn normalized_black_call(x: f64, s: f64) -> f64 {
    if s <= 0.0 {
        return normalized_intrinsic(x, true);
    }
    let exph = (0.5 * x).exp();
    let expmh = (-0.5 * x).exp();
    let inv_s = 1.0 / s;
    let d1 = x * inv_s + 0.5 * s;
    let d2 = d1 - s;
    exph * normal_cdf(d1) - expmh * normal_cdf(d2)
}

#[inline]
pub fn normalized_black(x: f64, s: f64, is_call: bool) -> f64 {
    let call = normalized_black_call(x, s);
    if is_call {
        call
    } else {
        call - ((0.5 * x).exp() - (-0.5 * x).exp())
    }
}

#[inline]
fn normalized_black_call_with_derivatives(x: f64, s: f64) -> BlackEval {
    let exph = (0.5 * x).exp();
    let expmh = (-0.5 * x).exp();
    let inv_s = 1.0 / s;
    let d1 = x * inv_s + 0.5 * s;
    let d2 = d1 - s;

    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let pdf1 = normal_pdf(d1);

    let price = exph * nd1 - expmh * nd2;
    let first = exph * pdf1;

    let a = 0.5 - x * inv_s * inv_s;
    let q = -d1 * a;
    let second = q * first;

    let a_prime = 2.0 * x * inv_s * inv_s * inv_s;
    let q_prime = -(a * a + d1 * a_prime);
    let third = (q * q + q_prime) * first;

    BlackEval {
        price,
        first,
        second,
        third,
    }
}

#[inline]
fn map_to_otm_call(beta: f64, x: f64, is_call: bool) -> (f64, f64) {
    let parity = (0.5 * x).exp() - (-0.5 * x).exp();
    if is_call {
        if x > 0.0 {
            ((beta - parity).max(0.0), -x)
        } else {
            (beta.max(0.0), x)
        }
    } else if x < 0.0 {
        ((beta + parity).max(0.0), x)
    } else {
        (beta.max(0.0), -x)
    }
}

#[inline]
fn initial_guess_rational(beta: f64, x: f64) -> f64 {
    // x is mapped to out-of-the-money call space, so x <= 0 and 0 < beta < exp(x/2).
    let exph = (0.5 * x).exp();
    let y = (beta / exph).clamp(1e-16, 1.0 - 1e-12);

    // Exact ATM inversion at x = 0 and a good base elsewhere.
    let mut s = 2.0 * normal_inv_cdf(0.5 * (1.0 + y));
    if !s.is_finite() || s <= 0.0 {
        s = (2.0 * std::f64::consts::PI).sqrt() * y;
    }

    // Rational moneyness correction (Table-1-style functional form).
    let m = -x;
    let corr = (1.0 + 0.23 * m + 0.006 * m * m) / (1.0 + 0.12 * m + 0.001 * m * m);
    s *= corr;

    // Deep OTM asymptotic floor.
    if y < 0.15 {
        let denom = (-2.0 * y.ln()).sqrt();
        if denom.is_finite() && denom > 0.0 {
            s = s.max(m / denom);
        }
    }

    s.clamp(1e-12, MAX_SIGMA_SQRT_T)
}

#[inline]
fn householder4_step(f: f64, f1: f64, f2: f64, f3: f64) -> Option<f64> {
    if !f.is_finite() || !f1.is_finite() || !f2.is_finite() || !f3.is_finite() || f1 <= 0.0 {
        return None;
    }

    let r = f / f1;
    let a = f2 / f1;
    let b = f3 / f1;
    let a1 = r * a;
    let b1 = r * r * b;
    let denom = 1.0 - a1 + b1 / 6.0;
    if !denom.is_finite() || denom.abs() < 1e-14 {
        return Some(-r);
    }

    Some(-r * (1.0 - 0.5 * a1) / denom)
}

pub fn implied_vol_jaeckel_normalized(
    beta: f64,
    x: f64,
    is_call: bool,
) -> Result<f64, PricingError> {
    if !beta.is_finite() || !x.is_finite() {
        return Err(PricingError::InvalidInput(
            "beta and x must be finite".to_string(),
        ));
    }
    if beta < 0.0 {
        return Err(PricingError::InvalidInput("beta must be >= 0".to_string()));
    }

    let intrinsic = normalized_intrinsic(x, is_call);
    let upper = normalized_upper_bound(x, is_call);
    let price_tol = 32.0 * f64::EPSILON * (1.0 + upper.abs());
    if beta < intrinsic - price_tol || beta > upper + price_tol {
        return Err(PricingError::InvalidInput(format!(
            "normalized option price out of bounds: beta={beta}, intrinsic={intrinsic}, upper={upper}"
        )));
    }
    if beta <= intrinsic + price_tol {
        return Ok(0.0);
    }

    // Transform to an out-of-the-money call: x_work <= 0, 0 < beta_work < exp(x_work/2).
    let (beta_work, x_work) = map_to_otm_call(beta, x, is_call);
    if beta_work <= price_tol {
        return Ok(0.0);
    }

    let guess = initial_guess_rational(beta_work, x_work);
    let mut lo = 0.0;
    let mut hi = guess.max(1e-8);

    let mut eval_hi = normalized_black_call_with_derivatives(x_work, hi);
    while eval_hi.price < beta_work && hi < MAX_SIGMA_SQRT_T {
        lo = hi;
        hi = (hi * 2.0).min(MAX_SIGMA_SQRT_T);
        eval_hi = normalized_black_call_with_derivatives(x_work, hi);
    }

    if eval_hi.price < beta_work {
        return Err(PricingError::ConvergenceFailure(format!(
            "failed to bracket normalized implied vol: beta={beta_work}, x={x_work}"
        )));
    }

    let mut s = guess.clamp(lo, hi);
    for _ in 0..MAX_HOUSEHOLDER_ITERS {
        let eval = normalized_black_call_with_derivatives(x_work, s);
        let f = eval.price - beta_work;
        if f.abs() <= price_tol.max(8.0 * f64::EPSILON * (1.0 + beta_work)) {
            return Ok(s.max(0.0));
        }

        if f > 0.0 {
            hi = s;
        } else {
            lo = s;
        }

        if (hi - lo) <= 16.0 * f64::EPSILON * (1.0 + s.abs()) {
            return Ok(0.5 * (lo + hi));
        }

        let mut s_next = 0.5 * (lo + hi);
        if let Some(step) = householder4_step(f, eval.first, eval.second, eval.third) {
            let candidate = s + step;
            if candidate.is_finite() && candidate > lo && candidate < hi {
                s_next = candidate;
            }
        }
        s = s_next;
    }

    Ok(0.5 * (lo + hi))
}

pub fn implied_vol_jaeckel(
    price: f64,
    forward: f64,
    strike: f64,
    t: f64,
    is_call: bool,
) -> Result<f64, PricingError> {
    if !price.is_finite() || !forward.is_finite() || !strike.is_finite() || !t.is_finite() {
        return Err(PricingError::InvalidInput(
            "price, forward, strike, and t must be finite".to_string(),
        ));
    }
    if forward <= 0.0 || strike <= 0.0 {
        return Err(PricingError::InvalidInput(
            "forward and strike must be > 0".to_string(),
        ));
    }
    if t <= 0.0 {
        return Err(PricingError::InvalidInput("t must be > 0".to_string()));
    }
    if price < 0.0 {
        return Err(PricingError::InvalidInput("price must be >= 0".to_string()));
    }

    let intrinsic = if is_call {
        (forward - strike).max(0.0)
    } else {
        (strike - forward).max(0.0)
    };
    let upper = if is_call { forward } else { strike };
    let price_tol = 32.0 * f64::EPSILON * (1.0 + upper.abs());
    if price < intrinsic - price_tol || price > upper + price_tol {
        return Err(PricingError::InvalidInput(format!(
            "option price out of bounds: price={price}, intrinsic={intrinsic}, upper={upper}"
        )));
    }
    if price <= intrinsic + price_tol {
        return Ok(0.0);
    }

    let beta = price / (forward * strike).sqrt();
    let x = (forward / strike).ln();
    let s = implied_vol_jaeckel_normalized(beta, x, is_call)?;
    Ok((s / t.sqrt()).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normalized_black_put_call_parity() {
        let x = 0.3;
        let s = 0.7;
        let call = normalized_black(x, s, true);
        let put = normalized_black(x, s, false);
        let parity = (0.5 * x).exp() - (-0.5 * x).exp();
        assert_relative_eq!(call - put, parity, epsilon = 1e-12);
    }

    #[test]
    fn normalized_solver_recovers_sigma_sqrt_t() {
        let x = -0.7;
        let s_true = 0.42;
        let beta = normalized_black(x, s_true, true);
        let s = implied_vol_jaeckel_normalized(beta, x, true).unwrap();
        assert_relative_eq!(s, s_true, epsilon = 1e-10);
    }
}
