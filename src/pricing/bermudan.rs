//! Module `pricing::bermudan`.
//!
//! Implements bermudan workflows with concrete routines such as `longstaff_schwartz_bermudan`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Primary API surface: free functions `longstaff_schwartz_bermudan`.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::pricing::OptionType;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

fn intrinsic(option_type: OptionType, s: f64, k: f64) -> f64 {
    match option_type {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

/// Prices a Bermudan option with the Longstaff-Schwartz Monte Carlo method.
///
/// Returns `f64::NAN` (instead of panicking) when `steps <= 1` or
/// `num_paths <= 2`, so invalid user-controlled input from bindings cannot
/// abort the host process.
#[allow(clippy::too_many_arguments)]
pub fn longstaff_schwartz_bermudan(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    exercise_steps: &[usize],
    num_paths: usize,
    seed: u64,
) -> f64 {
    if steps <= 1 || num_paths <= 2 {
        return f64::NAN;
    }

    let dt = t / steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let disc = (-r * dt).exp();

    let mut can_exercise = vec![false; steps + 1];
    for &e in exercise_steps {
        if e <= steps {
            can_exercise[e] = true;
        }
    }
    can_exercise[steps] = true;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut paths = vec![vec![0.0_f64; steps + 1]; num_paths];

    for path in &mut paths {
        path[0] = s0;
        for ti in 1..=steps {
            let z: f64 = StandardNormal.sample(&mut rng);
            path[ti] = path[ti - 1] * (drift + vol * z).exp();
        }
    }

    let mut values: Vec<f64> = paths
        .iter()
        .map(|p| intrinsic(option_type, p[steps], k))
        .collect();

    for ti in (1..steps).rev() {
        for v in &mut values {
            *v *= disc;
        }

        if !can_exercise[ti] {
            continue;
        }

        let itm: Vec<usize> = paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if intrinsic(option_type, p[ti], k) > 0.0 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if itm.len() < 3 {
            continue;
        }

        let mut x = DMatrix::<f64>::zeros(itm.len(), 3);
        let mut y = DVector::<f64>::zeros(itm.len());

        for (row, &idx) in itm.iter().enumerate() {
            let s = paths[idx][ti];
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
            let s = paths[idx][ti];
            let continuation = beta[0] + beta[1] * s + beta[2] * s * s;
            let exercise = intrinsic(option_type, s, k);
            if exercise > continuation {
                values[idx] = exercise;
            }
        }
    }

    values.iter().map(|v| v * disc).sum::<f64>() / num_paths as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ExerciseStyle, PricingEngine};
    use crate::engines::tree::binomial::BinomialTreeEngine;
    use crate::instruments::vanilla::VanillaOption;
    use crate::market::Market;

    #[test]
    fn bermudan_lsm_matches_independent_binomial_with_batch_stderr() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let exercise_steps = vec![13, 26, 39, 52];
        let estimates = (0..16)
            .map(|batch| {
                longstaff_schwartz_bermudan(
                    OptionType::Put,
                    s0,
                    k,
                    r,
                    sigma,
                    t,
                    52,
                    &exercise_steps,
                    30_000,
                    99 + batch,
                )
            })
            .collect::<Vec<_>>();
        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let variance = estimates
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (estimates.len() - 1) as f64;
        let batch_stderr = (variance / estimates.len() as f64).sqrt();

        let dates = exercise_steps
            .iter()
            .map(|step| t * *step as f64 / 52.0)
            .collect::<Vec<_>>();
        let option = VanillaOption {
            option_type: OptionType::Put,
            strike: k,
            expiry: t,
            exercise: ExerciseStyle::Bermudan { dates },
        };
        let market = Market::builder()
            .spot(s0)
            .rate(r)
            .dividend_yield(0.0)
            .flat_vol(sigma)
            .build()
            .unwrap();
        let coarse = BinomialTreeEngine::new(2_500)
            .price(&option, &market)
            .unwrap()
            .price;
        let reference = BinomialTreeEngine::new(5_000)
            .price(&option, &market)
            .unwrap()
            .price;
        let reference_refinement = (reference - coarse).abs();
        let err = (mean - reference).abs();

        assert!(
            err <= 4.0 * batch_stderr + reference_refinement,
            "LSM mean={mean} tree={reference} err={err} batch_se={batch_stderr} tree_refinement={reference_refinement}"
        );
    }

    #[test]
    fn bermudan_invalid_inputs_return_nan_without_panicking() {
        let exercise_steps = vec![1];
        // steps <= 1
        let px = longstaff_schwartz_bermudan(
            OptionType::Put,
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            1,
            &exercise_steps,
            1_000,
            99,
        );
        assert!(px.is_nan());
        // num_paths <= 2
        let px = longstaff_schwartz_bermudan(
            OptionType::Put,
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            52,
            &exercise_steps,
            2,
            99,
        );
        assert!(px.is_nan());
    }
}
