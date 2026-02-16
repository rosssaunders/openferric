//! SIMD-friendly Monte Carlo routines using structure-of-arrays (SoA) path layout.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::math::simd_math::{exp_f64x4, load_f64x4, splat_f64x4, store_f64x4};

/// Structure-of-arrays path storage:
/// `levels[step][path] = S(step, path)`.
#[derive(Debug, Clone)]
pub struct SoaPaths {
    pub num_steps: usize,
    pub num_paths: usize,
    pub levels: Vec<Vec<f64>>,
}

impl SoaPaths {
    #[inline]
    pub fn terminal(&self) -> &[f64] {
        &self.levels[self.num_steps]
    }
}

/// Scalar SoA GBM simulation.
pub fn simulate_gbm_paths_soa_scalar(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> SoaPaths {
    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut levels = vec![vec![0.0_f64; num_paths]; num_steps + 1];
    levels[0].fill(s0);

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut z = vec![0.0_f64; num_paths];

    for step in 0..num_steps {
        for zi in &mut z {
            *zi = StandardNormal.sample(&mut rng);
        }

        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];

        for i in 0..num_paths {
            let growth = (drift + diffusion * z[i]).exp();
            next[i] = prev[i] * growth;
        }
    }

    SoaPaths {
        num_steps,
        num_paths,
        levels,
    }
}

/// Runtime-dispatched SoA GBM simulation (AVX2+FMA when available).
pub fn simulate_gbm_paths_soa(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> SoaPaths {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                simulate_gbm_paths_soa_avx2(s0, r, q, vol, t, num_paths, num_steps, seed)
            };
        }
    }

    simulate_gbm_paths_soa_scalar(s0, r, q, vol, t, num_paths, num_steps, seed)
}

/// Scalar European call Monte Carlo over SoA GBM paths.
pub fn mc_european_call_soa_scalar(
    s0: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> f64 {
    let paths = simulate_gbm_paths_soa_scalar(s0, r, q, vol, t, num_paths, num_steps, seed);
    let terminal = paths.terminal();
    let mean_payoff = terminal
        .iter()
        .map(|&st| (st - strike).max(0.0))
        .sum::<f64>()
        / num_paths as f64;
    (-r * t).exp() * mean_payoff
}

/// Runtime-dispatched European call Monte Carlo over SoA GBM paths.
pub fn mc_european_call_soa(
    s0: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> f64 {
    let paths = simulate_gbm_paths_soa(s0, r, q, vol, t, num_paths, num_steps, seed);
    let terminal = paths.terminal();
    let mean_payoff = terminal
        .iter()
        .map(|&st| (st - strike).max(0.0))
        .sum::<f64>()
        / num_paths as f64;
    (-r * t).exp() * mean_payoff
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn simulate_gbm_paths_soa_avx2(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> SoaPaths {
    use std::arch::x86_64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut levels = vec![vec![0.0_f64; num_paths]; num_steps + 1];
    levels[0].fill(s0);

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();

    let drift_v = unsafe { splat_f64x4(drift) };
    let diffusion_v = unsafe { splat_f64x4(diffusion) };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut z = vec![0.0_f64; num_paths];

    for step in 0..num_steps {
        for zi in &mut z {
            *zi = StandardNormal.sample(&mut rng);
        }

        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];

        let mut i = 0usize;
        while i + 4 <= num_paths {
            // SAFETY: bounds checked by loop condition.
            let s = unsafe { load_f64x4(prev, i) };
            // SAFETY: bounds checked by loop condition.
            let z_vec = unsafe { load_f64x4(&z, i) };
            let x = _mm256_fmadd_pd(diffusion_v, z_vec, drift_v);
            let growth = unsafe { exp_f64x4(x) };
            let s_next = _mm256_mul_pd(s, growth);
            // SAFETY: bounds checked by loop condition.
            unsafe { store_f64x4(next, i, s_next) };
            i += 4;
        }

        while i < num_paths {
            let growth = (drift + diffusion * z[i]).exp();
            next[i] = prev[i] * growth;
            i += 1;
        }
    }

    SoaPaths {
        num_steps,
        num_paths,
        levels,
    }
}
