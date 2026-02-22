//! Module `pricing::range_accrual`.
//!
//! Implements range accrual workflows with concrete routines such as `range_accrual_mc_price`, `dual_range_accrual_mc_price`, `range_accrual_rate_delta`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `RangeAccrualResult` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::instruments::range_accrual::{DualRangeAccrual, RangeAccrual};

/// Result of range accrual MC pricing.
#[derive(Debug, Clone)]
pub struct RangeAccrualResult {
    /// Present value of the range accrual note.
    pub price: f64,
    /// Standard error of the MC estimate.
    pub std_error: f64,
    /// Expected accrual fraction (proportion of days in range).
    pub expected_accrual_fraction: f64,
}

/// Price a single-rate range accrual via Monte Carlo.
///
/// Simulates a short rate (or reference rate) as mean-reverting OU process:
///   dr = kappa * (theta - r) * dt + sigma * dW
///
/// # Arguments
/// * `ra` - Range accrual instrument
/// * `r0` - Initial reference rate
/// * `kappa` - Mean reversion speed
/// * `theta` - Long-run mean rate
/// * `sigma` - Rate volatility
/// * `discount_rate` - Risk-free rate for discounting
/// * `num_paths` - Number of MC paths
/// * `seed` - RNG seed
pub fn range_accrual_mc_price(
    ra: &RangeAccrual,
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> Result<RangeAccrualResult, String> {
    ra.validate()?;
    if num_paths == 0 {
        return Err("num_paths must be > 0".to_string());
    }

    let n_fix = ra.fixing_times.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut sum_pv = 0.0;
    let mut sum_pv2 = 0.0;
    let mut total_accrual = 0.0;

    // Pre-compute time intervals
    let mut dts = Vec::with_capacity(n_fix);
    let mut prev_t = 0.0;
    for &t in &ra.fixing_times {
        dts.push(t - prev_t);
        prev_t = t;
    }

    let df = (-discount_rate * ra.payment_time).exp();

    for _ in 0..num_paths {
        let mut r = r0;
        let mut days_in_range = 0usize;

        for &dt in &dts {
            let z: f64 = StandardNormal.sample(&mut rng);
            r += kappa * (theta - r) * dt + sigma * dt.sqrt() * z;

            if r >= ra.lower_bound && r <= ra.upper_bound {
                days_in_range += 1;
            }
        }

        let accrual_fraction = days_in_range as f64 / n_fix as f64;
        let coupon = ra.notional * ra.coupon_rate * accrual_fraction;
        let pv = coupon * df;

        sum_pv += pv;
        sum_pv2 += pv * pv;
        total_accrual += accrual_fraction;
    }

    let n = num_paths as f64;
    let mean = sum_pv / n;
    let variance = (sum_pv2 / n - mean * mean).max(0.0);

    Ok(RangeAccrualResult {
        price: mean,
        std_error: (variance / n).sqrt(),
        expected_accrual_fraction: total_accrual / n,
    })
}

/// Price a dual-rate range accrual via Monte Carlo with correlated OU processes.
///
/// Two rates: r1 and r2, with correlation rho.
/// Accrual condition: (r1 - r2) ∈ [lower, upper].
pub fn dual_range_accrual_mc_price(
    dra: &DualRangeAccrual,
    r1_0: f64,
    r2_0: f64,
    kappa1: f64,
    theta1: f64,
    sigma1: f64,
    kappa2: f64,
    theta2: f64,
    sigma2: f64,
    rho: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
) -> Result<RangeAccrualResult, String> {
    dra.validate()?;
    if num_paths == 0 {
        return Err("num_paths must be > 0".to_string());
    }
    if rho.abs() > 1.0 {
        return Err("rho must be in [-1, 1]".to_string());
    }

    let n_fix = dra.fixing_times.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut sum_pv = 0.0;
    let mut sum_pv2 = 0.0;
    let mut total_accrual = 0.0;

    let mut dts = Vec::with_capacity(n_fix);
    let mut prev_t = 0.0;
    for &t in &dra.fixing_times {
        dts.push(t - prev_t);
        prev_t = t;
    }

    let df = (-discount_rate * dra.payment_time).exp();
    let rho_comp = (1.0 - rho * rho).sqrt();

    for _ in 0..num_paths {
        let mut r1 = r1_0;
        let mut r2 = r2_0;
        let mut days_in_range = 0usize;

        for &dt in &dts {
            let z1: f64 = StandardNormal.sample(&mut rng);
            let z2: f64 = StandardNormal.sample(&mut rng);
            let w2 = rho * z1 + rho_comp * z2;

            r1 += kappa1 * (theta1 - r1) * dt + sigma1 * dt.sqrt() * z1;
            r2 += kappa2 * (theta2 - r2) * dt + sigma2 * dt.sqrt() * w2;

            let spread = r1 - r2;
            if spread >= dra.lower_bound && spread <= dra.upper_bound {
                days_in_range += 1;
            }
        }

        let accrual_fraction = days_in_range as f64 / n_fix as f64;
        let coupon = dra.notional * dra.coupon_rate * accrual_fraction;
        let pv = coupon * df;

        sum_pv += pv;
        sum_pv2 += pv * pv;
        total_accrual += accrual_fraction;
    }

    let n = num_paths as f64;
    let mean = sum_pv / n;
    let variance = (sum_pv2 / n - mean * mean).max(0.0);

    Ok(RangeAccrualResult {
        price: mean,
        std_error: (variance / n).sqrt(),
        expected_accrual_fraction: total_accrual / n,
    })
}

/// Rate sensitivity via bump-and-reprice.
pub fn range_accrual_rate_delta(
    ra: &RangeAccrual,
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    discount_rate: f64,
    num_paths: usize,
    seed: u64,
    bump: f64,
) -> Result<f64, String> {
    let p_up = range_accrual_mc_price(
        ra,
        r0 + bump,
        kappa,
        theta,
        sigma,
        discount_rate,
        num_paths,
        seed,
    )?;
    let p_down = range_accrual_mc_price(
        ra,
        r0 - bump,
        kappa,
        theta,
        sigma,
        discount_rate,
        num_paths,
        seed,
    )?;
    Ok((p_up.price - p_down.price) / (2.0 * bump))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_range_accrual() -> RangeAccrual {
        // Daily fixings for 1 year (252 business days)
        let fixing_times: Vec<f64> = (1..=252).map(|d| d as f64 / 252.0).collect();
        RangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.02,
            upper_bound: 0.06,
            fixing_times,
            payment_time: 1.0,
        }
    }

    #[test]
    fn range_accrual_price_is_positive() {
        let ra = make_range_accrual();
        let result = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 5000, 42).unwrap();
        assert!(result.price > 0.0);
        assert!(result.price.is_finite());
    }

    #[test]
    fn range_accrual_fraction_bounded() {
        let ra = make_range_accrual();
        let result = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 5000, 42).unwrap();
        assert!(result.expected_accrual_fraction >= 0.0);
        assert!(result.expected_accrual_fraction <= 1.0);
    }

    #[test]
    fn range_accrual_narrow_range_less_valuable() {
        let mut ra = make_range_accrual();
        let wide = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 5000, 42).unwrap();

        ra.lower_bound = 0.039;
        ra.upper_bound = 0.041;
        let narrow = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 5000, 42).unwrap();

        assert!(narrow.price < wide.price);
    }

    #[test]
    fn dual_range_accrual_price_is_positive() {
        let fixing_times: Vec<f64> = (1..=252).map(|d| d as f64 / 252.0).collect();
        let dra = DualRangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.0,
            upper_bound: 0.02,
            fixing_times,
            payment_time: 1.0,
        };
        let result = dual_range_accrual_mc_price(
            &dra, 0.05, 0.03, // r1_0, r2_0 (spread = 0.02)
            1.0, 0.05, 0.01, // kappa1, theta1, sigma1
            1.0, 0.03, 0.01, // kappa2, theta2, sigma2
            0.5,  // rho
            0.03, // discount rate
            5000, 42,
        )
        .unwrap();
        assert!(result.price > 0.0);
        assert!(result.price.is_finite());
    }

    #[test]
    fn dual_range_high_correlation_differs_from_low() {
        let fixing_times: Vec<f64> = (1..=252).map(|d| d as f64 / 252.0).collect();
        let dra = DualRangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.0,
            upper_bound: 0.03,
            fixing_times,
            payment_time: 1.0,
        };
        let high_rho = dual_range_accrual_mc_price(
            &dra, 0.05, 0.03, 1.0, 0.05, 0.01, 1.0, 0.03, 0.01, 0.9, 0.03, 5000, 42,
        )
        .unwrap();
        let low_rho = dual_range_accrual_mc_price(
            &dra, 0.05, 0.03, 1.0, 0.05, 0.01, 1.0, 0.03, 0.01, 0.1, 0.03, 5000, 42,
        )
        .unwrap();
        // Correlation affects spread volatility and hence accrual
        assert!((high_rho.price - low_rho.price).abs() > 0.01);
    }
}
