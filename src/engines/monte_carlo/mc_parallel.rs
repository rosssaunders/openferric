//! Module `engines::monte_carlo::mc_parallel`.
//!
//! Implements mc parallel workflows with concrete routines such as `mc_european_parallel`, `mc_european_sequential`, `mc_greeks_grid_sequential`, `mc_greeks_grid_parallel`.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `GreeksGridPoint` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
use rayon::prelude::*;

use crate::core::{ExerciseStyle, OptionType, PricingResult};
use crate::engines::analytic::black_scholes::{bs_delta, bs_gamma, bs_vega};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind, sample_standard_normal};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreeksGridPoint {
    pub spot: f64,
    pub vol: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

#[inline]
fn uniform_open01(u: f64) -> f64 {
    u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
}

#[inline(always)]
fn payoff(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
fn split_paths(n_paths: usize, n_chunks: usize) -> Vec<usize> {
    let chunks = n_chunks.max(1);
    let base = n_paths / chunks;
    let rem = n_paths % chunks;
    (0..chunks)
        .map(|i| if i < rem { base + 1 } else { base })
        .filter(|&n| n > 0)
        .collect()
}

/// Exact single-step GBM chunk for European vanilla options.
///
/// Instead of simulating per-step, directly samples terminal spot:
///   S_T = S_0 * exp(total_drift + total_diffusion * Z)
/// One exp() per path instead of one per step.
///
/// Uses batch SIMD inverse CDF when AVX2+FMA are available: pre-generates
/// a block of uniform randoms, batch-converts to normals, then processes
/// the block with vectorized exp + payoff. This amortizes the inv-CDF cost
/// across 4-wide SIMD lanes and improves cache locality.
#[allow(clippy::too_many_arguments)]
#[inline]
fn simulate_chunk_exact(
    option_type: OptionType,
    strike: f64,
    spot0: f64,
    total_drift: f64,
    total_diffusion: f64,
    n_paths: usize,
    chunk_seed: u64,
) -> (f64, f64, usize) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                simulate_chunk_exact_avx2(
                    option_type,
                    strike,
                    spot0,
                    total_drift,
                    total_diffusion,
                    n_paths,
                    chunk_seed,
                )
            };
        }
    }
    simulate_chunk_exact_scalar(
        option_type,
        strike,
        spot0,
        total_drift,
        total_diffusion,
        n_paths,
        chunk_seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_chunk_exact_scalar(
    option_type: OptionType,
    strike: f64,
    spot0: f64,
    total_drift: f64,
    total_diffusion: f64,
    n_paths: usize,
    chunk_seed: u64,
) -> (f64, f64, usize) {
    let mut rng = FastRng::from_seed(FastRngKind::Xoshiro256PlusPlus, chunk_seed);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    let mut remaining = n_paths;
    while remaining >= 8 {
        let z0 = sample_standard_normal(&mut rng);
        let z1 = sample_standard_normal(&mut rng);
        let z2 = sample_standard_normal(&mut rng);
        let z3 = sample_standard_normal(&mut rng);
        let z4 = sample_standard_normal(&mut rng);
        let z5 = sample_standard_normal(&mut rng);
        let z6 = sample_standard_normal(&mut rng);
        let z7 = sample_standard_normal(&mut rng);

        let s0 = spot0 * total_diffusion.mul_add(z0, total_drift).exp();
        let s1 = spot0 * total_diffusion.mul_add(z1, total_drift).exp();
        let s2 = spot0 * total_diffusion.mul_add(z2, total_drift).exp();
        let s3 = spot0 * total_diffusion.mul_add(z3, total_drift).exp();
        let s4 = spot0 * total_diffusion.mul_add(z4, total_drift).exp();
        let s5 = spot0 * total_diffusion.mul_add(z5, total_drift).exp();
        let s6 = spot0 * total_diffusion.mul_add(z6, total_drift).exp();
        let s7 = spot0 * total_diffusion.mul_add(z7, total_drift).exp();

        let p0 = payoff(option_type, s0, strike);
        let p1 = payoff(option_type, s1, strike);
        let p2 = payoff(option_type, s2, strike);
        let p3 = payoff(option_type, s3, strike);
        let p4 = payoff(option_type, s4, strike);
        let p5 = payoff(option_type, s5, strike);
        let p6 = payoff(option_type, s6, strike);
        let p7 = payoff(option_type, s7, strike);

        sum += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
        sum_sq += p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3 + p4 * p4 + p5 * p5 + p6 * p6 + p7 * p7;
        remaining -= 8;
    }
    while remaining > 0 {
        let z = sample_standard_normal(&mut rng);
        let s = spot0 * total_diffusion.mul_add(z, total_drift).exp();
        let px = payoff(option_type, s, strike);
        sum += px;
        sum_sq += px * px;
        remaining -= 1;
    }

    (sum, sum_sq, n_paths)
}

/// AVX2+FMA accelerated chunk simulation with batch inverse CDF.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn simulate_chunk_exact_avx2(
    option_type: OptionType,
    strike: f64,
    spot0: f64,
    total_drift: f64,
    total_diffusion: f64,
    n_paths: usize,
    chunk_seed: u64,
) -> (f64, f64, usize) {
    use crate::math::fast_rng::Xoshiro256PlusPlus;
    use crate::math::simd_math::{fast_exp_f64x4, splat_f64x4};
    use std::arch::x86_64::*;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    let spot_v = unsafe { splat_f64x4(spot0) };
    let drift_v = unsafe { splat_f64x4(total_drift) };
    let diff_v = unsafe { splat_f64x4(total_diffusion) };
    let strike_v = unsafe { splat_f64x4(strike) };
    let zero_v = unsafe { _mm256_setzero_pd() };

    // Block size for batch normal generation (fits in L1 cache).
    const BLOCK: usize = 512;
    let mut normals = [0.0_f64; BLOCK];

    let mut remaining = n_paths;

    // Process full blocks of BLOCK paths.
    while remaining >= BLOCK {
        unsafe { crate::math::simd_math::fill_normals_simd(&mut rng, &mut normals) };

        let mut j = 0usize;
        while j + 4 <= BLOCK {
            unsafe {
                let z_vec = _mm256_loadu_pd(normals.as_ptr().add(j));
                let exponent = _mm256_fmadd_pd(diff_v, z_vec, drift_v);
                let growth = fast_exp_f64x4(exponent);
                let s_terminal = _mm256_mul_pd(spot_v, growth);

                let payoff_v = match option_type {
                    OptionType::Call => _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v),
                    OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v),
                };

                // Extract and accumulate (horizontal reduction every 4 payoffs).
                let mut pay = [0.0_f64; 4];
                _mm256_storeu_pd(pay.as_mut_ptr(), payoff_v);
                sum += pay[0] + pay[1] + pay[2] + pay[3];
                sum_sq += pay[0] * pay[0] + pay[1] * pay[1] + pay[2] * pay[2] + pay[3] * pay[3];
            }
            j += 4;
        }
        remaining -= BLOCK;
    }

    // Handle remaining paths with smaller batch.
    if remaining >= 4 {
        let batch = remaining & !3;
        unsafe { crate::math::simd_math::fill_normals_simd(&mut rng, &mut normals[..batch]) };

        let mut j = 0usize;
        while j + 4 <= batch {
            unsafe {
                let z_vec = _mm256_loadu_pd(normals.as_ptr().add(j));
                let exponent = _mm256_fmadd_pd(diff_v, z_vec, drift_v);
                let growth = fast_exp_f64x4(exponent);
                let s_terminal = _mm256_mul_pd(spot_v, growth);

                let payoff_v = match option_type {
                    OptionType::Call => _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v),
                    OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v),
                };

                let mut pay = [0.0_f64; 4];
                _mm256_storeu_pd(pay.as_mut_ptr(), payoff_v);
                sum += pay[0] + pay[1] + pay[2] + pay[3];
                sum_sq += pay[0] * pay[0] + pay[1] * pay[1] + pay[2] * pay[2] + pay[3] * pay[3];
            }
            j += 4;
        }
        remaining -= batch;
    }

    // Scalar tail
    let mut fast_rng = FastRng::from_seed(
        FastRngKind::Xoshiro256PlusPlus,
        chunk_seed.wrapping_add(0x1234_5678),
    );
    for _ in 0..remaining {
        let z = sample_standard_normal(&mut fast_rng);
        let s = spot0 * total_diffusion.mul_add(z, total_drift).exp();
        let px = payoff(option_type, s, strike);
        sum += px;
        sum_sq += px * px;
    }

    (sum, sum_sq, n_paths)
}

/// Parallel Monte Carlo pricer for European vanilla options.
///
/// Uses exact single-step GBM simulation — one exp() per path, not per step.
/// Work is explicitly split into thread-sized chunks and reduced in parallel.
pub fn mc_european_parallel(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    _n_steps: usize,
) -> PricingResult {
    if n_paths == 0 {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let vol = market.vol_for(instrument.strike, instrument.expiry);
    if vol <= 0.0 || !vol.is_finite() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let t = instrument.expiry;
    // Exact terminal-value drift and diffusion.
    let total_drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * t;
    let total_diffusion = vol * t.sqrt();
    let discount = (-market.rate * t).exp();

    let chunks = split_paths(n_paths, rayon::current_num_threads());
    let base_seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let (sum, sum_sq, total_paths) = chunks
        .par_iter()
        .enumerate()
        .map(|(i, &chunk)| {
            let chunk_seed =
                base_seed.wrapping_add((i as u64).wrapping_mul(6_364_136_223_846_793_005));
            simulate_chunk_exact(
                instrument.option_type,
                instrument.strike,
                market.spot,
                total_drift,
                total_diffusion,
                chunk,
                chunk_seed,
            )
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0_usize),
            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
        );

    let n = total_paths as f64;
    let mean = sum / n;
    let variance = if total_paths > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", 1.0);
    diagnostics.insert("num_threads", rayon::current_num_threads() as f64);
    diagnostics.insert("vol", vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
        greeks: None,
        diagnostics,
    }
}

/// Sequential baseline using exact single-step GBM simulation.
pub fn mc_european_sequential(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    _n_steps: usize,
) -> PricingResult {
    if n_paths == 0 {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let vol = market.vol_for(instrument.strike, instrument.expiry);
    if vol <= 0.0 || !vol.is_finite() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }
    let t = instrument.expiry;
    let total_drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * t;
    let total_diffusion = vol * t.sqrt();
    let discount = (-market.rate * t).exp();

    let (sum, sum_sq, total_paths) = simulate_chunk_exact(
        instrument.option_type,
        instrument.strike,
        market.spot,
        total_drift,
        total_diffusion,
        n_paths,
        0xDEAD_BEEF_CAFE_BABE,
    );
    let n = total_paths as f64;
    let mean = sum / n;
    let variance = if total_paths > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert("num_paths", n_paths as f64);
    diagnostics.insert("num_steps", 1.0);
    diagnostics.insert("vol", vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
        greeks: None,
        diagnostics,
    }
}

#[inline]
fn greeks_grid_point(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> GreeksGridPoint {
    GreeksGridPoint {
        spot,
        vol,
        delta: bs_delta(option_type, spot, strike, rate, dividend_yield, vol, expiry),
        gamma: bs_gamma(spot, strike, rate, dividend_yield, vol, expiry),
        vega: bs_vega(spot, strike, rate, dividend_yield, vol, expiry),
    }
}

#[inline]
fn build_grid(spots: &[f64], vols: &[f64]) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(spots.len() * vols.len());
    for &spot in spots {
        for &vol in vols {
            points.push((spot, vol));
        }
    }
    points
}

/// Sequential delta/gamma/vega grid for spot-vol pairs.
#[allow(clippy::too_many_arguments)]
pub fn mc_greeks_grid_sequential(
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    spots: &[f64],
    vols: &[f64],
) -> Vec<GreeksGridPoint> {
    let points = build_grid(spots, vols);
    points
        .iter()
        .map(|&(spot, vol)| {
            greeks_grid_point(option_type, spot, strike, rate, dividend_yield, vol, expiry)
        })
        .collect()
}

/// Parallel delta/gamma/vega grid for spot-vol pairs using Rayon `par_iter`.
#[allow(clippy::too_many_arguments)]
pub fn mc_greeks_grid_parallel(
    option_type: OptionType,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    spots: &[f64],
    vols: &[f64],
) -> Vec<GreeksGridPoint> {
    let points = build_grid(spots, vols);
    points
        .par_iter()
        .map(|&(spot, vol)| {
            greeks_grid_point(option_type, spot, strike, rate, dividend_yield, vol, expiry)
        })
        .collect()
}
