//! Module `pricing::american`.
//!
//! Implements american workflows with concrete routines such as `crr_binomial_american`, `longstaff_schwartz_american_put`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: free functions `crr_binomial_american`, `longstaff_schwartz_american_put`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::pricing::OptionType;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Prices an American option on a CRR binomial tree.
///
/// Returns `f64::NAN` (instead of panicking) when `steps == 0`, so invalid
/// user-controlled input from bindings cannot abort the host process.
pub fn crr_binomial_american(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
) -> f64 {
    if steps == 0 {
        return f64::NAN;
    }

    let intrinsic = |spot: f64| match option_type {
        OptionType::Call => (spot - k).max(0.0),
        OptionType::Put => (k - spot).max(0.0),
    };
    if t <= 0.0 {
        return intrinsic(s0);
    }

    if sigma <= 0.0 {
        // The zero-width CRR limit is a deterministic path.  Pin the exact
        // finite exercise grid instead of evaluating the undefined 0/0 CRR
        // probability.  Discount every exercise cashflow to time zero and
        // choose the holder-optimal date, including immediate exercise.
        return (0..=steps)
            .map(|i| {
                let exercise_time = t * i as f64 / steps as f64;
                let spot = s0 * (r * exercise_time).exp();
                (-r * exercise_time).exp() * intrinsic(spot)
            })
            .fold(0.0, f64::max);
    }

    let dt = t / steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let disc = (-r * dt).exp();
    let p = ((r * dt).exp() - d) / (u - d);

    let mut values = vec![0.0_f64; steps + 1];
    for (j, v) in values.iter_mut().enumerate().take(steps + 1) {
        let s_t = s0 * u.powf(j as f64) * d.powf((steps - j) as f64);
        *v = match option_type {
            OptionType::Call => (s_t - k).max(0.0),
            OptionType::Put => (k - s_t).max(0.0),
        };
    }

    for i in (0..steps).rev() {
        for j in 0..=i {
            let s_t = s0 * u.powf(j as f64) * d.powf((i - j) as f64);
            let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            let exercise = match option_type {
                OptionType::Call => (s_t - k).max(0.0),
                OptionType::Put => (k - s_t).max(0.0),
            };
            values[j] = continuation.max(exercise);
        }
    }

    values[0]
}

/// Prices an American put with the Longstaff-Schwartz Monte Carlo method.
///
/// Returns `f64::NAN` (instead of panicking) when `steps <= 1` or
/// `paths <= 2`, so invalid user-controlled input from bindings cannot abort
/// the host process.
#[allow(clippy::too_many_arguments)]
pub fn longstaff_schwartz_american_put(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    paths: usize,
    seed: u64,
) -> f64 {
    if steps <= 1 || paths <= 2 {
        return f64::NAN;
    }

    if t <= 0.0 {
        return (k - s0).max(0.0);
    }

    if sigma <= 0.0 {
        return (0..=steps)
            .map(|i| {
                let exercise_time = t * i as f64 / steps as f64;
                let spot = s0 * (r * exercise_time).exp();
                (-r * exercise_time).exp() * (k - spot).max(0.0)
            })
            .fold(0.0, f64::max);
    }

    let dt = t / steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let disc = (-r * dt).exp();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sim_paths = vec![vec![0.0_f64; steps + 1]; paths];

    for path in &mut sim_paths {
        path[0] = s0;
        for ti in 1..=steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            path[ti] = path[ti - 1] * (drift + vol * z).exp();
        }
    }

    let mut values: Vec<f64> = sim_paths.iter().map(|p| (k - p[steps]).max(0.0)).collect();

    for ti in (1..steps).rev() {
        for v in &mut values {
            *v *= disc;
        }

        let itm: Vec<usize> = sim_paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if (k - p[ti]).max(0.0) > 0.0 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if itm.len() >= 3 {
            let mut x = DMatrix::<f64>::zeros(itm.len(), 3);
            let mut y = DVector::<f64>::zeros(itm.len());

            for (row, &idx) in itm.iter().enumerate() {
                let s = sim_paths[idx][ti];
                x[(row, 0)] = 1.0;
                x[(row, 1)] = s;
                x[(row, 2)] = s * s;
                y[row] = values[idx];
            }

            let xtx = x.transpose() * &x;
            let xty = x.transpose() * &y;
            let beta = xtx
                .lu()
                .solve(&xty)
                .unwrap_or_else(|| DVector::<f64>::zeros(3));

            for &idx in &itm {
                let s = sim_paths[idx][ti];
                let continuation = beta[0] + beta[1] * s + beta[2] * s * s;
                let exercise = (k - s).max(0.0);
                if exercise > continuation {
                    values[idx] = exercise;
                }
            }
        }
    }

    let continuation = values.iter().map(|v| v * disc).sum::<f64>() / paths as f64;
    // American exercise includes t=0.  The regression rollback starts at the
    // first positive time index, so enforce the root exercise decision here.
    continuation.max((k - s0).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn zero_vol_american_methods_match_exact_exercise_cashflows() {
        let s0: f64 = 100.0;
        let k: f64 = 95.0;
        let r: f64 = 0.05;
        let t: f64 = 1.0;

        // With positive rates a deterministic non-dividend call is optimally
        // held to maturity: PV(ST-K) = S0-K*exp(-rT).
        let call = crr_binomial_american(OptionType::Call, s0, k, r, 0.0, t, 64);
        let call_reference = (s0 - k * (-r * t).exp()).max(0.0);
        assert!((call - call_reference).abs() <= 8.0 * f64::EPSILON * call_reference.max(1.0));

        // The deterministic ITM put loses value as spot grows, so immediate
        // exercise is exact for both the CRR and LSM entry points.
        let put_spot: f64 = 90.0;
        let put_strike: f64 = 100.0;
        let put_reference = put_strike - put_spot;
        let crr_put = crr_binomial_american(OptionType::Put, put_spot, put_strike, r, 0.0, t, 64);
        let lsm_put =
            longstaff_schwartz_american_put(put_spot, put_strike, r, 0.0, t, 64, 1_000, 42);
        assert_eq!(crr_put, put_reference);
        assert_eq!(lsm_put, put_reference);
    }

    #[test]
    fn crr_american_put_matches_independent_decimal_finite_grid() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let eur_put = black_scholes_price(OptionType::Put, s0, k, r, sigma, t);
        let am_put = crr_binomial_american(OptionType::Put, s0, k, r, sigma, t, 500);

        // Independently generated with an 80-digit Python Decimal CRR
        // recurrence using the same stated 500-step methodology.  The
        // tolerance only covers accumulated binary64/libm roundoff; it is not
        // an economic range around an American-option price.
        const DECIMAL_500_STEP_REFERENCE: f64 = 6.088_810_110_703_653;
        const BINARY64_ACCUMULATION_BUDGET: f64 = 2.0e-12;
        assert!(
            (am_put - DECIMAL_500_STEP_REFERENCE).abs() <= BINARY64_ACCUMULATION_BUDGET,
            "500-step CRR={am_put:.17e}, Decimal={DECIMAL_500_STEP_REFERENCE:.17e}, roundoff budget={BINARY64_ACCUMULATION_BUDGET:.3e}"
        );

        // Supplemental no-arbitrage property.
        assert!(am_put >= eur_put - 1e-8);
    }

    #[test]
    fn longstaff_schwartz_matches_quantlib_crr_with_batch_stderr() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let batch_prices = (0..16_u64)
            .map(|batch| {
                longstaff_schwartz_american_put(s0, k, r, sigma, t, 50, 10_000, 0xA4E1_0000 + batch)
            })
            .collect::<Vec<_>>();
        let n = batch_prices.len() as f64;
        let mean = batch_prices.iter().sum::<f64>() / n;
        let batch_stderr = (batch_prices
            .iter()
            .map(|price| (price - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
            / n)
            .sqrt();
        // QuantLib 1.43 CRR, 10,000 steps.  The extra 4e-5 covers the observed
        // 10k-to-20k lattice refinement (3.65e-5).
        let reference = 6.090_298_054_322_291;
        let tolerance = 4.0 * batch_stderr + 4.0e-5;
        assert!(
            (mean - reference).abs() <= tolerance,
            "mean={mean} reference={reference} batch_stderr={batch_stderr} tolerance={tolerance}"
        );
    }

    #[test]
    fn crr_binomial_zero_steps_returns_nan_without_panicking() {
        let px = crr_binomial_american(OptionType::Put, 100.0, 100.0, 0.05, 0.2, 1.0, 0);
        assert!(px.is_nan());
    }

    #[test]
    fn longstaff_schwartz_invalid_inputs_return_nan_without_panicking() {
        // steps <= 1
        let px = longstaff_schwartz_american_put(100.0, 100.0, 0.05, 0.2, 1.0, 0, 1_000, 7);
        assert!(px.is_nan());
        let px = longstaff_schwartz_american_put(100.0, 100.0, 0.05, 0.2, 1.0, 1, 1_000, 7);
        assert!(px.is_nan());
        // paths <= 2
        let px = longstaff_schwartz_american_put(100.0, 100.0, 0.05, 0.2, 1.0, 50, 2, 7);
        assert!(px.is_nan());
    }
}
