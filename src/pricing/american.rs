//! Legacy pricing routines for American products.
//!
//! Kept for backward compatibility with historical product-focused APIs.

use crate::pricing::OptionType;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

pub fn crr_binomial_american(
    option_type: OptionType,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
) -> f64 {
    assert!(steps > 0);

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
    assert!(steps > 1);
    assert!(paths > 2);

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

    values.iter().map(|v| v * disc).sum::<f64>() / paths as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn american_put_is_worth_at_least_european_put() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let eur_put = black_scholes_price(OptionType::Put, s0, k, r, sigma, t);
        let am_put = crr_binomial_american(OptionType::Put, s0, k, r, sigma, t, 500);

        assert!(am_put >= eur_put - 1e-8);
    }

    #[test]
    fn longstaff_schwartz_reasonable_against_binomial() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let lsm = longstaff_schwartz_american_put(s0, k, r, sigma, t, 50, 40_000, 7);
        let tree = crr_binomial_american(OptionType::Put, s0, k, r, sigma, t, 800);

        assert!((lsm - tree).abs() < 0.5);
    }
}
