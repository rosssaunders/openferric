//! Module `engines::monte_carlo::mc_simd`.
//!
//! Implements mc simd workflows with concrete routines such as `simulate_gbm_paths_soa_scalar`, `simulate_gbm_paths_soa`, `mc_european_call_soa_scalar`, `mc_european_call_soa`.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `SoaPaths` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.

use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::math::simd_math::{fast_exp_f64x4, load_f64x4, splat_f64x4, store_f64x4};

/// Structure-of-arrays path storage:
/// `levels[step][path] = S(step, path)`.
///
/// This materializes the full `(num_steps + 1) x num_paths` matrix (e.g. ~2GB
/// for 1M paths x 252 steps). Only use it when the consumer genuinely needs
/// the whole path; terminal-only payoffs should use the
/// `simulate_gbm_terminal_soa*` variants, which ping-pong two step buffers.
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

/// Scalar terminal-only SoA GBM simulation.
///
/// Identical dynamics and RNG consumption order to
/// `simulate_gbm_paths_soa_scalar`, but stores only two step buffers
/// (ping-pong) instead of the full path matrix.
pub fn simulate_gbm_terminal_soa_scalar(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut cur = vec![s0; num_paths];
    let mut next = vec![0.0_f64; num_paths];

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);

    for _ in 0..num_steps {
        for i in 0..num_paths {
            let z = sample_standard_normal(&mut rng);
            let growth = diffusion.mul_add(z, drift).exp();
            next[i] = cur[i] * growth;
        }
        std::mem::swap(&mut cur, &mut next);
    }

    cur
}

/// Runtime-dispatched terminal-only SoA GBM simulation
/// (AVX-512 > AVX2+FMA on x86-64, NEON on AArch64, then scalar) using two
/// ping-pong step buffers.
pub fn simulate_gbm_terminal_soa(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                simulate_gbm_terminal_soa_avx512(s0, r, q, vol, t, num_paths, num_steps, seed)
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                simulate_gbm_terminal_soa_avx2(s0, r, q, vol, t, num_paths, num_steps, seed)
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // AArch64 guarantees Advanced SIMD (NEON).
        unsafe { simulate_gbm_terminal_soa_neon(s0, r, q, vol, t, num_paths, num_steps, seed) }
    }

    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    {
        simulate_gbm_terminal_soa_scalar(s0, r, q, vol, t, num_paths, num_steps, seed)
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
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);

    for step in 0..num_steps {
        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];

        for i in 0..num_paths {
            let z = sample_standard_normal(&mut rng);
            let growth = diffusion.mul_add(z, drift).exp();
            next[i] = prev[i] * growth;
        }
    }

    SoaPaths {
        num_steps,
        num_paths,
        levels,
    }
}

/// Runtime-dispatched SoA GBM simulation
/// (AVX-512 > AVX2+FMA on x86-64, NEON on AArch64, then scalar).
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
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                simulate_gbm_paths_soa_avx512(s0, r, q, vol, t, num_paths, num_steps, seed)
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            return unsafe {
                simulate_gbm_paths_soa_avx2(s0, r, q, vol, t, num_paths, num_steps, seed)
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // AArch64 guarantees Advanced SIMD (NEON). Keeping this as a separate
        // multiversioned function also lets portable builds omit all NEON
        // intrinsics when the `simd` feature is disabled.
        unsafe { simulate_gbm_paths_soa_neon(s0, r, q, vol, t, num_paths, num_steps, seed) }
    }

    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    {
        simulate_gbm_paths_soa_scalar(s0, r, q, vol, t, num_paths, num_steps, seed)
    }
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
    // Terminal-only payoff: ping-pong step buffers instead of the full
    // (num_steps + 1) x num_paths path matrix.
    let terminal = simulate_gbm_terminal_soa_scalar(s0, r, q, vol, t, num_paths, num_steps, seed);
    let mut sum = 0.0_f64;
    let mut i = 0;
    while i + 4 <= num_paths {
        sum += (terminal[i] - strike).max(0.0);
        sum += (terminal[i + 1] - strike).max(0.0);
        sum += (terminal[i + 2] - strike).max(0.0);
        sum += (terminal[i + 3] - strike).max(0.0);
        i += 4;
    }
    while i < num_paths {
        sum += (terminal[i] - strike).max(0.0);
        i += 1;
    }
    let mean_payoff = sum / num_paths as f64;
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
    // Terminal-only payoff: ping-pong step buffers instead of the full
    // (num_steps + 1) x num_paths path matrix.
    let terminal = simulate_gbm_terminal_soa(s0, r, q, vol, t, num_paths, num_steps, seed);
    let mean_payoff = terminal
        .iter()
        .map(|&st| (st - strike).max(0.0))
        .sum::<f64>()
        / num_paths as f64;
    (-r * t).exp() * mean_payoff
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn simulate_gbm_terminal_soa_avx2(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    use crate::math::fast_rng::Xoshiro256PlusPlus;
    use std::arch::x86_64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut cur = vec![s0; num_paths];
    let mut next = vec![0.0_f64; num_paths];

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();

    let drift_v = unsafe { splat_f64x4(drift) };
    let diffusion_v = unsafe { splat_f64x4(diffusion) };

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let buf_size = (num_paths + 3) & !3;
    let mut normal_buf = vec![0.0_f64; buf_size];

    for _ in 0..num_steps {
        // Batch-generate all normals for this step via SIMD inverse CDF.
        unsafe {
            crate::math::simd_math::fill_normals_simd(&mut rng, &mut normal_buf[..num_paths])
        };

        let mut i = 0usize;
        while i + 4 <= num_paths {
            unsafe {
                let s = load_f64x4(&cur, i);
                let z_vec = _mm256_loadu_pd(normal_buf.as_ptr().add(i));
                let x = _mm256_fmadd_pd(diffusion_v, z_vec, drift_v);
                let growth = fast_exp_f64x4(x);
                let s_next = _mm256_mul_pd(s, growth);
                store_f64x4(&mut next, i, s_next);
            }
            i += 4;
        }

        while i < num_paths {
            let z = normal_buf[i];
            let growth = diffusion.mul_add(z, drift).exp();
            next[i] = cur[i] * growth;
            i += 1;
        }

        std::mem::swap(&mut cur, &mut next);
    }

    cur
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simulate_gbm_terminal_soa_avx512(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    use crate::math::fast_rng::Xoshiro256PlusPlus;
    use crate::math::simd_avx512::{fast_exp_f64x8, load_f64x8, splat_f64x8, store_f64x8};
    use std::arch::x86_64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut cur = vec![s0; num_paths];
    let mut next = vec![0.0_f64; num_paths];

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();

    let drift_v = splat_f64x8(drift);
    let diffusion_v = splat_f64x8(diffusion);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let buf_size = (num_paths + 7) & !7;
    let mut normal_buf = vec![0.0_f64; buf_size];

    for _ in 0..num_steps {
        // Batch-generate all normals for this step via AVX-512 inverse CDF.
        crate::math::simd_avx512::fill_normals_simd_avx512(&mut rng, &mut normal_buf[..num_paths]);

        let mut i = 0usize;
        while i + 8 <= num_paths {
            let s = load_f64x8(&cur, i);
            let z_vec = _mm512_loadu_pd(normal_buf.as_ptr().add(i));
            let x = _mm512_fmadd_pd(diffusion_v, z_vec, drift_v);
            let growth = fast_exp_f64x8(x);
            let s_next = _mm512_mul_pd(s, growth);
            store_f64x8(&mut next, i, s_next);
            i += 8;
        }

        while i < num_paths {
            let z = normal_buf[i];
            let growth = diffusion.mul_add(z, drift).exp();
            next[i] = cur[i] * growth;
            i += 1;
        }

        std::mem::swap(&mut cur, &mut next);
    }

    cur
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
    use crate::math::fast_rng::Xoshiro256PlusPlus;
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Pre-allocate a normal buffer for batch SIMD inverse CDF.
    // Sized to handle one full step's worth of paths (rounded up to multiple of 4).
    let buf_size = (num_paths + 3) & !3;
    let mut normal_buf = vec![0.0_f64; buf_size];

    for step in 0..num_steps {
        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];

        // Batch-generate all normals for this step via SIMD inverse CDF.
        unsafe {
            crate::math::simd_math::fill_normals_simd(&mut rng, &mut normal_buf[..num_paths])
        };

        let mut i = 0usize;
        while i + 4 <= num_paths {
            unsafe {
                let s = load_f64x4(prev, i);
                let z_vec = _mm256_loadu_pd(normal_buf.as_ptr().add(i));
                let x = _mm256_fmadd_pd(diffusion_v, z_vec, drift_v);
                let growth = fast_exp_f64x4(x);
                let s_next = _mm256_mul_pd(s, growth);
                store_f64x4(next, i, s_next);
            }
            i += 4;
        }

        while i < num_paths {
            let z = normal_buf[i];
            let growth = diffusion.mul_add(z, drift).exp();
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
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simulate_gbm_paths_soa_avx512(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> SoaPaths {
    use crate::math::fast_rng::Xoshiro256PlusPlus;
    use crate::math::simd_avx512::{fast_exp_f64x8, load_f64x8, splat_f64x8, store_f64x8};
    use std::arch::x86_64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut levels = vec![vec![0.0_f64; num_paths]; num_steps + 1];
    levels[0].fill(s0);

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();

    let drift_v = splat_f64x8(drift);
    let diffusion_v = splat_f64x8(diffusion);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Pre-allocate a normal buffer for batch SIMD inverse CDF.
    // Sized to handle one full step's worth of paths (rounded up to multiple of 8).
    let buf_size = (num_paths + 7) & !7;
    let mut normal_buf = vec![0.0_f64; buf_size];

    for step in 0..num_steps {
        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];

        // Batch-generate all normals for this step via AVX-512 inverse CDF.
        crate::math::simd_avx512::fill_normals_simd_avx512(&mut rng, &mut normal_buf[..num_paths]);

        let mut i = 0usize;
        while i + 8 <= num_paths {
            let s = load_f64x8(prev, i);
            let z_vec = _mm512_loadu_pd(normal_buf.as_ptr().add(i));
            let x = _mm512_fmadd_pd(diffusion_v, z_vec, drift_v);
            let growth = fast_exp_f64x8(x);
            let s_next = _mm512_mul_pd(s, growth);
            store_f64x8(next, i, s_next);
            i += 8;
        }

        while i < num_paths {
            let z = normal_buf[i];
            let growth = diffusion.mul_add(z, drift).exp();
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

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simulate_gbm_terminal_soa_neon(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    use crate::math::simd_neon::{load_f64x2, simd_exp_f64x2, splat_f64x2, store_f64x2};
    use std::arch::aarch64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut cur = vec![s0; num_paths];
    let mut next = vec![0.0_f64; num_paths];
    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();
    let drift_v = splat_f64x2(drift);
    let diffusion_v = splat_f64x2(diffusion);
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut normal_buf = vec![0.0_f64; num_paths];

    for _ in 0..num_steps {
        for normal in &mut normal_buf {
            *normal = sample_standard_normal(&mut rng);
        }

        let mut i = 0usize;
        while i + 2 <= num_paths {
            let previous = load_f64x2(&cur, i);
            let z = load_f64x2(&normal_buf, i);
            let exponent = vfmaq_f64(drift_v, diffusion_v, z);
            let next_value = vmulq_f64(previous, simd_exp_f64x2(exponent));
            store_f64x2(&mut next, i, next_value);
            i += 2;
        }
        while i < num_paths {
            next[i] = cur[i] * diffusion.mul_add(normal_buf[i], drift).exp();
            i += 1;
        }
        std::mem::swap(&mut cur, &mut next);
    }

    cur
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn simulate_gbm_paths_soa_neon(
    s0: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> SoaPaths {
    use crate::math::simd_neon::{load_f64x2, simd_exp_f64x2, splat_f64x2, store_f64x2};
    use std::arch::aarch64::*;

    assert!(num_paths > 0, "num_paths must be > 0");
    assert!(num_steps > 0, "num_steps must be > 0");

    let mut levels = vec![vec![0.0_f64; num_paths]; num_steps + 1];
    levels[0].fill(s0);

    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let diffusion = vol * dt.sqrt();
    let drift_v = splat_f64x2(drift);
    let diffusion_v = splat_f64x2(diffusion);
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, seed);
    let mut normal_buf = vec![0.0_f64; num_paths];

    for step in 0..num_steps {
        for normal in &mut normal_buf {
            *normal = sample_standard_normal(&mut rng);
        }

        let (prev_head, prev_tail) = levels.split_at_mut(step + 1);
        let prev = &prev_head[step];
        let next = &mut prev_tail[0];
        let mut i = 0usize;
        while i + 2 <= num_paths {
            let previous = load_f64x2(prev, i);
            let z = load_f64x2(&normal_buf, i);
            let exponent = vfmaq_f64(drift_v, diffusion_v, z);
            let next_value = vmulq_f64(previous, simd_exp_f64x2(exponent));
            store_f64x2(next, i, next_value);
            i += 2;
        }
        while i < num_paths {
            next[i] = prev[i] * diffusion.mul_add(normal_buf[i], drift).exp();
            i += 1;
        }
    }

    SoaPaths {
        num_steps,
        num_paths,
        levels,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_only_scalar_matches_full_path_terminal() {
        let paths = simulate_gbm_paths_soa_scalar(100.0, 0.05, 0.01, 0.2, 1.0, 1_003, 16, 42);
        let term = simulate_gbm_terminal_soa_scalar(100.0, 0.05, 0.01, 0.2, 1.0, 1_003, 16, 42);
        assert_eq!(paths.terminal(), &term[..]);
    }

    #[test]
    fn terminal_only_dispatched_matches_full_path_terminal() {
        // Path count not a multiple of 4/8 to exercise the scalar tail.
        let paths = simulate_gbm_paths_soa(100.0, 0.05, 0.01, 0.2, 1.0, 1_003, 16, 7);
        let term = simulate_gbm_terminal_soa(100.0, 0.05, 0.01, 0.2, 1.0, 1_003, 16, 7);
        assert_eq!(paths.terminal(), &term[..]);
    }

    #[test]
    fn mc_european_call_soa_agrees_between_scalar_and_dispatched() {
        let scalar = mc_european_call_soa_scalar(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 20_000, 8, 11);
        let dispatched = mc_european_call_soa(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 20_000, 8, 11);
        // Different RNG/exp code paths (SIMD vs scalar), so only require
        // statistical agreement.
        assert!(
            (scalar - dispatched).abs() / scalar < 0.05,
            "scalar={scalar} dispatched={dispatched}"
        );
    }

    #[test]
    fn runtime_dispatch_handles_odd_path_tail() {
        let paths = simulate_gbm_paths_soa(100.0, 0.04, 0.01, 0.25, 1.5, 17, 9, 42);
        assert_eq!(paths.levels.len(), 10);
        assert!(paths.levels.iter().all(|level| level.len() == 17));
        assert!(paths.levels[0].iter().all(|&spot| spot == 100.0));
        assert!(
            paths
                .levels
                .iter()
                .flatten()
                .all(|spot| spot.is_finite() && *spot > 0.0)
        );
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_soa_matches_scalar_stream_and_tail() {
        let scalar = simulate_gbm_paths_soa_scalar(97.0, 0.03, 0.005, 0.31, 2.0, 19, 11, 7);
        let neon = unsafe { simulate_gbm_paths_soa_neon(97.0, 0.03, 0.005, 0.31, 2.0, 19, 11, 7) };

        for (scalar_level, neon_level) in scalar.levels.iter().zip(&neon.levels) {
            for (&expected, &actual) in scalar_level.iter().zip(neon_level) {
                let scale = expected.abs().max(1.0);
                assert!((actual - expected).abs() <= 2.0e-12 * scale);
            }
        }
    }
}
