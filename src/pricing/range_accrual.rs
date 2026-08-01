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
    use crate::math::normal_cdf;

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

    fn interval_probability(mean: f64, variance: f64, lower: f64, upper: f64) -> f64 {
        if variance <= 0.0 {
            return if mean >= lower && mean <= upper {
                1.0
            } else {
                0.0
            };
        }
        let std_dev = variance.sqrt();
        normal_cdf((upper - mean) / std_dev) - normal_cdf((lower - mean) / std_dev)
    }

    /// Exact expectation of the Euler-discretised OU contract implemented by
    /// `range_accrual_mc_price`.  Each fixing is Gaussian; correlations
    /// between fixing indicators affect variance, but not the expected coupon.
    fn exact_single_euler_price(
        ra: &RangeAccrual,
        r0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        discount_rate: f64,
    ) -> f64 {
        let mut mean = r0;
        let mut variance = 0.0;
        let mut previous = 0.0;
        let mut expected_fixings = 0.0;
        for &time in &ra.fixing_times {
            let dt = time - previous;
            let persistence = 1.0 - kappa * dt;
            mean = persistence * mean + kappa * theta * dt;
            variance = persistence * persistence * variance + sigma * sigma * dt;
            expected_fixings +=
                interval_probability(mean, variance, ra.lower_bound, ra.upper_bound);
            previous = time;
        }
        let expected_fraction = expected_fixings / ra.fixing_times.len() as f64;
        ra.notional * ra.coupon_rate * expected_fraction * (-discount_rate * ra.payment_time).exp()
    }

    #[allow(clippy::too_many_arguments)]
    fn exact_dual_euler_price(
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
    ) -> f64 {
        let (mut mean1, mut mean2) = (r1_0, r2_0);
        let (mut variance1, mut variance2, mut covariance) = (0.0, 0.0, 0.0);
        let mut previous = 0.0;
        let mut expected_fixings = 0.0;

        for &time in &dra.fixing_times {
            let dt = time - previous;
            let persistence1 = 1.0 - kappa1 * dt;
            let persistence2 = 1.0 - kappa2 * dt;
            mean1 = persistence1 * mean1 + kappa1 * theta1 * dt;
            mean2 = persistence2 * mean2 + kappa2 * theta2 * dt;
            variance1 = persistence1 * persistence1 * variance1 + sigma1 * sigma1 * dt;
            variance2 = persistence2 * persistence2 * variance2 + sigma2 * sigma2 * dt;
            covariance = persistence1 * persistence2 * covariance + rho * sigma1 * sigma2 * dt;

            let spread_variance = (variance1 + variance2 - 2.0 * covariance).max(0.0);
            expected_fixings += interval_probability(
                mean1 - mean2,
                spread_variance,
                dra.lower_bound,
                dra.upper_bound,
            );
            previous = time;
        }

        let expected_fraction = expected_fixings / dra.fixing_times.len() as f64;
        dra.notional
            * dra.coupon_rate
            * expected_fraction
            * (-discount_rate * dra.payment_time).exp()
    }

    #[test]
    fn range_accrual_mc_matches_exact_euler_gaussian_expectation() {
        let ra = make_range_accrual();
        let result = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 40_000, 42).unwrap();
        let analytic = exact_single_euler_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03);
        // Independently evaluated from the Euler Gaussian marginals with
        // SciPy 1.17.1's `norm.cdf`.
        let exact = 48_487.410_315_461_3;
        assert!((analytic - exact).abs() <= 64.0 * f64::EPSILON * exact);
        let roundoff = 32.0 * f64::EPSILON * exact.abs().max(1.0);
        assert!(
            (result.price - exact).abs() <= 4.0 * result.std_error + roundoff,
            "single range accrual mismatch: mc={} exact={exact} stderr={}",
            result.price,
            result.std_error
        );
    }

    #[test]
    fn range_accrual_fraction_bounded() {
        let ra = make_range_accrual();
        let result = range_accrual_mc_price(&ra, 0.04, 1.0, 0.04, 0.01, 0.03, 5000, 42).unwrap();
        assert!(result.expected_accrual_fraction >= 0.0);
        assert!(result.expected_accrual_fraction <= 1.0);
    }

    #[test]
    fn range_accrual_rate_delta_matches_deterministic_central_bump() {
        // One deterministic fixing is placed exactly at the lower boundary.
        // The upward initial-rate bump accrues the coupon and the downward bump
        // does not, giving an exact nonzero reference for the public delta API.
        let ra = RangeAccrual {
            notional: 1_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.04,
            upper_bound: 0.06,
            fixing_times: vec![1.0],
            payment_time: 1.0,
        };
        let discount_rate = 0.03;
        let bump = 1.0e-3;
        let actual =
            range_accrual_rate_delta(&ra, 0.04, 0.0, 0.04, 0.0, discount_rate, 256, 42, bump)
                .unwrap();
        let coupon_pv = ra.notional * ra.coupon_rate * (-discount_rate * ra.payment_time).exp();
        let exact = coupon_pv / (2.0 * bump);
        let roundoff = 64.0 * 256.0 * f64::EPSILON * exact.abs().max(1.0);
        assert!(
            (actual - exact).abs() <= roundoff,
            "range-accrual delta={actual} deterministic reference={exact}"
        );
    }

    #[test]
    fn stochastic_range_accrual_delta_matches_exact_gaussian_marginals() {
        const N_REPLICATES: usize = 12;
        const PATHS_PER_REPLICATE: usize = 40_000;
        let ra = RangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.035,
            upper_bound: 0.055,
            fixing_times: (1..=24).map(|month| month as f64 / 12.0).collect(),
            payment_time: 2.0,
        };
        let r0 = 0.04;
        let kappa = 0.8;
        let theta = 0.045;
        let sigma = 0.012;
        let discount_rate = 0.03;
        let bump = 0.002;

        let exact = (exact_single_euler_price(&ra, r0 + bump, kappa, theta, sigma, discount_rate)
            - exact_single_euler_price(&ra, r0 - bump, kappa, theta, sigma, discount_rate))
            / (2.0 * bump);
        // Independently evaluated with SciPy 1.17.1 normal CDFs at the exact
        // Euler marginal means and variances.
        let scipy_reference = 489_499.986_626_609_8;
        assert!((exact - scipy_reference).abs() <= 2.0e-7);

        let mut estimates = [0.0_f64; N_REPLICATES];
        for (replicate, estimate) in estimates.iter_mut().enumerate() {
            *estimate = range_accrual_rate_delta(
                &ra,
                r0,
                kappa,
                theta,
                sigma,
                discount_rate,
                PATHS_PER_REPLICATE,
                30_011 + 65_537 * replicate as u64,
                bump,
            )
            .unwrap();
        }
        let mean = estimates.iter().sum::<f64>() / N_REPLICATES as f64;
        let variance = estimates
            .iter()
            .map(|estimate| (estimate - mean).powi(2))
            .sum::<f64>()
            / (N_REPLICATES - 1) as f64;
        let implementation_se = (variance / N_REPLICATES as f64).sqrt();
        assert!(
            (mean - scipy_reference).abs() <= 4.0 * implementation_se,
            "stochastic range-accrual delta={mean} reference={scipy_reference} implementation_se={implementation_se}"
        );
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
    fn dual_range_accrual_mc_matches_exact_euler_gaussian_expectation() {
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
            40_000, 42,
        )
        .unwrap();
        let analytic = exact_dual_euler_price(
            &dra, 0.05, 0.03, 1.0, 0.05, 0.01, 1.0, 0.03, 0.01, 0.5, 0.03,
        );
        // Independent SciPy 1.17.1 reference for the correlated Euler spread
        // marginals.  Symmetry puts exactly half the mass in [0, 0.02].
        let exact = 24_243.705_157_730_634;
        assert!((analytic - exact).abs() <= 64.0 * f64::EPSILON * exact);
        let roundoff = 32.0 * f64::EPSILON * exact.abs().max(1.0);
        assert!(
            (result.price - exact).abs() <= 4.0 * result.std_error + roundoff,
            "dual range accrual mismatch: mc={} exact={exact} stderr={}",
            result.price,
            result.std_error
        );
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
        let high_exact = exact_dual_euler_price(
            &dra, 0.05, 0.03, 1.0, 0.05, 0.01, 1.0, 0.03, 0.01, 0.9, 0.03,
        );
        let low_exact = exact_dual_euler_price(
            &dra, 0.05, 0.03, 1.0, 0.05, 0.01, 1.0, 0.03, 0.01, 0.1, 0.03,
        );
        for (label, result, exact) in [
            ("rho=0.9", high_rho, high_exact),
            ("rho=0.1", low_rho, low_exact),
        ] {
            let roundoff = 32.0 * f64::EPSILON * exact.abs().max(1.0);
            assert!(
                (result.price - exact).abs() <= 4.0 * result.std_error + roundoff,
                "dual range accrual {label} mismatch: mc={} exact={exact} stderr={}",
                result.price,
                result.std_error
            );
        }
        // Higher rate correlation lowers spread variance; with the interval
        // centred around the expected spread, that raises the exact accrual.
        assert!(high_exact > low_exact);
    }
}
