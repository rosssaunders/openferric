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
    assert!(steps > 1);
    assert!(num_paths > 2);

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
    use crate::pricing::american::crr_binomial_american;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn bermudan_put_between_european_and_american() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let exercise_steps = vec![13, 26, 39, 52];
        let berm = longstaff_schwartz_bermudan(
            OptionType::Put,
            s0,
            k,
            r,
            sigma,
            t,
            52,
            &exercise_steps,
            60_000,
            99,
        );

        let eur = black_scholes_price(OptionType::Put, s0, k, r, sigma, t);
        let am = crr_binomial_american(OptionType::Put, s0, k, r, sigma, t, 500);

        assert!(berm >= eur - 0.2);
        assert!(berm <= am + 0.3);
    }
}
