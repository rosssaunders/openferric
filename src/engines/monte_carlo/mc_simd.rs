//! SIMD-friendly Monte Carlo routines using structure-of-arrays (SoA) path layout.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

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

    let drift_v = _mm256_set1_pd(drift);
    let diffusion_v = _mm256_set1_pd(diffusion);

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
            let s = unsafe { _mm256_loadu_pd(prev.as_ptr().add(i)) };
            // SAFETY: bounds checked by loop condition.
            let z_vec = unsafe { _mm256_loadu_pd(z.as_ptr().add(i)) };
            let x = _mm256_fmadd_pd(diffusion_v, z_vec, drift_v);
            let growth = unsafe { exp256_pd(x) };
            let s_next = _mm256_mul_pd(s, growth);
            // SAFETY: bounds checked by loop condition.
            unsafe { _mm256_storeu_pd(next.as_mut_ptr().add(i), s_next) };
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

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256_pd(x: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    let max_x = _mm256_set1_pd(709.782_712_893_384);
    let min_x = _mm256_set1_pd(-708.396_418_532_264_1);
    let x = _mm256_max_pd(min_x, _mm256_min_pd(x, max_x));

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    let half = _mm256_set1_pd(0.5);

    let n = _mm256_floor_pd(_mm256_fmadd_pd(x, log2e, half));
    let r = _mm256_fnmadd_pd(n, ln2, x);

    let c11 = _mm256_set1_pd(1.0 / 39_916_800.0);
    let c10 = _mm256_set1_pd(1.0 / 3_628_800.0);
    let c9 = _mm256_set1_pd(1.0 / 362_880.0);
    let c8 = _mm256_set1_pd(1.0 / 40_320.0);
    let c7 = _mm256_set1_pd(1.0 / 5_040.0);
    let c6 = _mm256_set1_pd(1.0 / 720.0);
    let c5 = _mm256_set1_pd(1.0 / 120.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c2 = _mm256_set1_pd(0.5);
    let c1 = _mm256_set1_pd(1.0);
    let c0 = _mm256_set1_pd(1.0);

    let mut poly = c11;
    poly = _mm256_fmadd_pd(poly, r, c10);
    poly = _mm256_fmadd_pd(poly, r, c9);
    poly = _mm256_fmadd_pd(poly, r, c8);
    poly = _mm256_fmadd_pd(poly, r, c7);
    poly = _mm256_fmadd_pd(poly, r, c6);
    poly = _mm256_fmadd_pd(poly, r, c5);
    poly = _mm256_fmadd_pd(poly, r, c4);
    poly = _mm256_fmadd_pd(poly, r, c3);
    poly = _mm256_fmadd_pd(poly, r, c2);
    poly = _mm256_fmadd_pd(poly, r, c1);
    poly = _mm256_fmadd_pd(poly, r, c0);

    let n_i32 = _mm256_cvtpd_epi32(n);
    let n_i64 = _mm256_cvtepi32_epi64(n_i32);
    let exp_bits = _mm256_slli_epi64(_mm256_add_epi64(n_i64, _mm256_set1_epi64x(1023)), 52);
    let two_pow_n = _mm256_castsi256_pd(exp_bits);

    _mm256_mul_pd(poly, two_pow_n)
}
