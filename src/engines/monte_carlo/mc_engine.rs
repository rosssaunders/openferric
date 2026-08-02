//! Module `engines::monte_carlo::mc_engine`.
//!
//! Implements mc engine workflows with concrete routines such as `mc_european_with_arena`.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `VarianceReduction`, `MonteCarloInstrument`, `MonteCarloPricingEngine`, `ArithmeticAsianMC` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
use std::any::Any;
use std::sync::Arc;

use crate::core::{
    Averaging, BarrierStyle, DiagKey, ExecutionBackend, ExecutionPolicy, ExerciseStyle, Instrument,
    OptionType, PricingEngine, PricingError, PricingResult, StrikeType,
};
use crate::instruments::{AsianOption, BarrierOption, VanillaOption};
use crate::market::{DividendSchedule, Market};
use crate::math::approx_tier::AccuracyTier;
use crate::math::arena::PricingArena;
use crate::math::fast_norm::beasley_springer_moro_inv_cdf;
use crate::math::fast_rng::{
    FastRng, FastRngKind, Xoshiro256PlusPlus, resolve_stream_seed, sample_standard_normal,
    uniform_open01,
};
#[cfg(feature = "parallel")]
use crate::mc::simulation::{estimated_parallel_work_savings, should_auto_parallelize_path};
use crate::mc::{
    ControlVariate, CpuExecutionPolicy, GbmPathGenerator, MonteCarloEngine, PathGenerator,
};
use crate::models::Gbm;
use crate::pricing::asian::geometric_asian_discrete_fixed_closed_form;
use crate::pricing::european::black_scholes_price;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Variance reduction scheme.
#[derive(Debug, Clone)]
pub enum VarianceReduction {
    /// No variance reduction.
    None,
    /// Antithetic variates.
    Antithetic,
    /// Built-in control variates:
    /// - European vanilla uses Black-Scholes.
    /// - Arithmetic fixed-strike Asian uses geometric Asian.
    ControlVariate,
}

/// Instrument interface required by the generic Monte Carlo engine.
pub trait MonteCarloInstrument: Instrument {
    /// Validates instrument fields for Monte Carlo pricing.
    fn validate_for_mc(&self) -> Result<(), PricingError>;
    /// Returns maturity in years.
    fn maturity(&self) -> f64;
    /// Returns a strike-like value for volatility lookup.
    fn reference_strike(&self, spot: f64) -> f64;
    /// Computes path payoff.
    fn payoff_from_path(&self, path: &[f64]) -> f64;
    /// Computes the path payoff expressed as a value at maturity, with access
    /// to the continuously compounded short rate for cashflows paid before
    /// maturity (e.g. knock-out rebates paid at hit time): such cashflows are
    /// compounded forward to maturity so that the engine's single
    /// discount-from-maturity factor prices them correctly.
    ///
    /// Defaults to `payoff_from_path` (all cashflows at maturity).
    fn payoff_from_path_with_rate(&self, path: &[f64], _rate: f64) -> f64 {
        self.payoff_from_path(path)
    }
    /// Optional control variate for this instrument.
    fn control_variate(&self, _market: &Market, _vol: f64) -> Option<ControlVariate> {
        None
    }
}

fn vanilla_payoff(option_type: crate::core::OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        crate::core::OptionType::Call => (spot - strike).max(0.0),
        crate::core::OptionType::Put => (strike - spot).max(0.0),
    }
}

/// Mergeable centered moments using the Welford/Chan recurrences.
///
/// Keeping `M2 = Σ(x-mean)²` avoids the catastrophic cancellation in
/// `Σx² - (Σx)²/n` for nearly deterministic Monte Carlo payoffs.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RunningMoments {
    count: usize,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    #[inline(always)]
    pub(crate) fn record(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline]
    pub(crate) fn merge(&mut self, other: Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other;
            return;
        }
        let lhs_count = self.count as f64;
        let rhs_count = other.count as f64;
        let total_count = lhs_count + rhs_count;
        let delta = other.mean - self.mean;
        self.mean += delta * (rhs_count / total_count);
        self.m2 += other.m2 + delta * delta * (lhs_count * rhs_count / total_count);
        self.count += other.count;
    }

    #[inline]
    pub(crate) fn count(self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn mean(self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.mean }
    }

    #[inline]
    pub(crate) fn sample_variance(self) -> f64 {
        if self.count <= 1 {
            return 0.0;
        }
        let variance = self.m2 / (self.count as f64 - 1.0);
        if variance < 0.0 { 0.0 } else { variance }
    }
}

fn black_scholes_price_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    let adjusted_spot = spot * (-dividend_yield * expiry).exp();
    black_scholes_price(option_type, adjusted_spot, strike, rate, vol, expiry)
}

fn path_hits_barrier(path: &[f64], barrier: f64, direction: crate::core::BarrierDirection) -> bool {
    match direction {
        crate::core::BarrierDirection::Up => path.iter().any(|&s| s >= barrier),
        crate::core::BarrierDirection::Down => path.iter().any(|&s| s <= barrier),
    }
}

/// Index of the first path point that breaches the barrier, if any.
fn first_barrier_hit_index(
    path: &[f64],
    barrier: f64,
    direction: crate::core::BarrierDirection,
) -> Option<usize> {
    match direction {
        crate::core::BarrierDirection::Up => path.iter().position(|&s| s >= barrier),
        crate::core::BarrierDirection::Down => path.iter().position(|&s| s <= barrier),
    }
}

fn average_for_observations(
    path: &[f64],
    maturity: f64,
    observation_times: &[f64],
    averaging: Averaging,
) -> f64 {
    if observation_times.is_empty() || maturity <= 0.0 {
        return path[path.len() - 1];
    }

    let last_idx = path.len().saturating_sub(1) as f64;

    match averaging {
        Averaging::Arithmetic => {
            let sum = observation_times
                .iter()
                .map(|&t| {
                    let idx = ((t / maturity) * last_idx).round() as usize;
                    path[idx.min(path.len() - 1)]
                })
                .sum::<f64>();
            sum / observation_times.len() as f64
        }
        Averaging::Geometric => {
            let mean_log = observation_times
                .iter()
                .map(|&t| {
                    let idx = ((t / maturity) * last_idx).round() as usize;
                    path[idx.min(path.len() - 1)].max(1e-12).ln()
                })
                .sum::<f64>()
                / observation_times.len() as f64;
            mean_log.exp()
        }
    }
}

#[derive(Debug, Clone)]
struct DiscreteDividendGbmPathGenerator {
    base: GbmPathGenerator,
    schedule: DividendSchedule,
}

impl DiscreteDividendGbmPathGenerator {
    fn new(base: GbmPathGenerator, schedule: DividendSchedule) -> Self {
        Self { base, schedule }
    }
}

impl PathGenerator for DiscreteDividendGbmPathGenerator {
    fn steps(&self) -> usize {
        self.base.steps
    }

    fn generate_from_normals(&self, normals_1: &[f64], normals_2: &[f64]) -> Vec<f64> {
        let mut path = vec![0.0_f64; self.base.steps + 1];
        self.generate_into(normals_1, normals_2, &mut path);
        path
    }

    fn generate_into(&self, normals_1: &[f64], _normals_2: &[f64], out: &mut [f64]) {
        let dt = self.base.maturity / self.base.steps as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.base.model.mu - 0.5 * self.base.model.sigma * self.base.model.sigma) * dt;
        let diffusion = self.base.model.sigma * sqrt_dt;

        let mut s = self.base.s0;
        out[0] = s;

        let events = self.schedule.events();
        let mut ev_idx = 0usize;
        let maturity = self.base.maturity;

        for (j, &z) in normals_1.iter().enumerate().take(self.base.steps) {
            s *= diffusion.mul_add(z, drift).exp();
            let t = (j + 1) as f64 * dt;
            while ev_idx < events.len()
                && events[ev_idx].time <= maturity + 1.0e-12
                && events[ev_idx].time <= t + 1.0e-12
            {
                s = events[ev_idx].apply_jump(s);
                ev_idx += 1;
            }
            out[j + 1] = s;
        }
    }

    fn num_normal_streams(&self) -> usize {
        1
    }
}

const ARENA_MC_SEED: u64 = 42;

/// Sequential European vanilla Monte Carlo pricer reusing a pre-allocated arena.
///
/// Uses exact single-step GBM simulation: S_T = S_0 * exp((μ - σ²/2)*T + σ*√T*Z).
/// For European options the payoff depends only on the terminal spot, so the
/// per-step path simulation is unnecessary. This reduces exp() calls from
/// O(paths * steps) to O(paths) — typically a 100–250× reduction.
pub fn mc_european_with_arena(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
    arena: &mut PricingArena,
) -> PricingResult {
    if n_paths == 0 {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if instrument.validate().is_err()
        || market.validate().is_err()
        || !matches!(instrument.exercise, ExerciseStyle::European)
    {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: vanilla_payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if market.has_discrete_dividends() {
        let engine = MonteCarloPricingEngine::new(n_paths, n_steps.max(1), ARENA_MC_SEED);
        return engine.price(instrument, market).unwrap_or(PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        });
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
    let mu = market.rate - market.dividend_yield;
    let discount = (-market.rate * t).exp();
    // Exact GBM terminal value: S_T = S_0 * exp((μ - σ²/2)*T + σ*√T*Z)
    // One exp() per path instead of one per step — the key optimization.
    let total_drift = (mu - 0.5 * vol * vol) * t;
    let total_diffusion = vol * t.sqrt();

    let _ = arena.payoff_slice(n_paths);
    let payoff_buffer = &mut arena.payoff_buffer;

    let option_type = instrument.option_type;
    let strike = instrument.strike;
    let spot = market.spot;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(ARENA_MC_SEED);
    let mut i = 0;

    // SIMD fast path: process 4 paths at a time with AVX2 fast_exp_f64x4.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Guarded by runtime CPU feature detection.
            unsafe {
                mc_exact_avx2_inner(
                    &mut rng,
                    payoff_buffer,
                    n_paths,
                    spot,
                    total_drift,
                    total_diffusion,
                    option_type,
                    strike,
                    &mut i,
                );
            }
        }
    }

    // Scalar remainder (or full scalar path on non-SIMD builds).
    // Process 4 at a time for ILP.
    while i + 4 <= n_paths {
        let z0 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
        let z1 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
        let z2 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
        let z3 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));

        let s0 = spot * total_diffusion.mul_add(z0, total_drift).exp();
        let s1 = spot * total_diffusion.mul_add(z1, total_drift).exp();
        let s2 = spot * total_diffusion.mul_add(z2, total_drift).exp();
        let s3 = spot * total_diffusion.mul_add(z3, total_drift).exp();

        payoff_buffer[i] = vanilla_payoff(option_type, s0, strike);
        payoff_buffer[i + 1] = vanilla_payoff(option_type, s1, strike);
        payoff_buffer[i + 2] = vanilla_payoff(option_type, s2, strike);
        payoff_buffer[i + 3] = vanilla_payoff(option_type, s3, strike);
        i += 4;
    }
    while i < n_paths {
        let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
        let s_t = spot * total_diffusion.mul_add(z, total_drift).exp();
        payoff_buffer[i] = vanilla_payoff(option_type, s_t, strike);
        i += 1;
    }

    let payoffs = &payoff_buffer[..n_paths];
    let mut stats = RunningMoments::default();
    for &v in payoffs {
        stats.record(v);
    }
    let n = stats.count() as f64;
    let mean = stats.mean();
    let variance = stats.sample_variance();

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert_key(crate::core::DiagKey::NumPaths, n_paths as f64);
    diagnostics.insert_key(crate::core::DiagKey::NumSteps, 1.0);
    diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
        greeks: None,
        diagnostics,
    }
}

/// AVX2+FMA inner loop for exact European MC with batch SIMD inverse CDF.
///
/// Pre-generates a block of uniform random numbers, batch-converts them to
/// normal variates via vectorized Acklam approximation, then consumes the
/// block with vectorized exp + payoff computation. This eliminates the
/// scalar inverse-CDF bottleneck that dominated the previous implementation.
///
/// Uses the adaptive approximation tier to select between degree-11 (High)
/// and degree-7 (Fast) exp() approximants.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn mc_exact_avx2_inner(
    rng: &mut Xoshiro256PlusPlus,
    payoff_buffer: &mut [f64],
    n_paths: usize,
    spot: f64,
    total_drift: f64,
    total_diffusion: f64,
    option_type: crate::core::OptionType,
    strike: f64,
    i: &mut usize,
) {
    use crate::math::approx_tier::tiered_exp_f64x4;
    use crate::math::simd_math::{fill_normals_simd, splat_f64x4, store_f64x4};
    use std::arch::x86_64::*;

    // MC always uses Fast tier — stochastic error dwarfs approximation error.
    let tier = AccuracyTier::for_mc(n_paths, 1);

    let spot_v = unsafe { splat_f64x4(spot) };
    let drift_v = unsafe { splat_f64x4(total_drift) };
    let diff_v = unsafe { splat_f64x4(total_diffusion) };
    let strike_v = unsafe { splat_f64x4(strike) };
    let zero_v = _mm256_setzero_pd();

    // Block size for batch normal generation. 256 is ~2KB which fits in L1 cache.
    const BLOCK: usize = 256;
    let mut normals = [0.0_f64; BLOCK];

    while *i + BLOCK <= n_paths {
        // Batch generate BLOCK normal variates using SIMD inverse CDF.
        unsafe { fill_normals_simd(rng, &mut normals) };

        // Consume the block 4 at a time with vectorized exp + payoff.
        let mut j = 0usize;
        while j + 4 <= BLOCK {
            let z_vec = unsafe { _mm256_loadu_pd(normals.as_ptr().add(j)) };
            let exponent = _mm256_fmadd_pd(diff_v, z_vec, drift_v);
            let growth = unsafe { tiered_exp_f64x4(exponent, tier) };
            let s_terminal = _mm256_mul_pd(spot_v, growth);

            let payoff_v = match option_type {
                crate::core::OptionType::Call => {
                    _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v)
                }
                crate::core::OptionType::Put => {
                    _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v)
                }
            };

            unsafe { store_f64x4(payoff_buffer, *i + j, payoff_v) };
            j += 4;
        }
        *i += BLOCK;
    }

    // Handle remaining paths (< BLOCK) with the same batch approach.
    let remaining = n_paths - *i;
    if remaining >= 4 {
        let batch = remaining & !3; // round down to multiple of 4
        unsafe { fill_normals_simd(rng, &mut normals[..batch]) };

        let mut j = 0usize;
        while j + 4 <= batch {
            let z_vec = unsafe { _mm256_loadu_pd(normals.as_ptr().add(j)) };
            let exponent = _mm256_fmadd_pd(diff_v, z_vec, drift_v);
            let growth = unsafe { tiered_exp_f64x4(exponent, tier) };
            let s_terminal = _mm256_mul_pd(spot_v, growth);

            let payoff_v = match option_type {
                crate::core::OptionType::Call => {
                    _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v)
                }
                crate::core::OptionType::Put => {
                    _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v)
                }
            };

            unsafe { store_f64x4(payoff_buffer, *i + j, payoff_v) };
            j += 4;
        }
        *i += batch;
    }
}

#[cfg(all(feature = "gpu", not(target_family = "wasm")))]
const AUTO_GPU_MIN_PATHS: usize = 1_000_000;
const EXACT_PARALLEL_CHUNK_SAMPLES: usize = 4_096;
#[cfg(feature = "parallel")]
const EXACT_PARALLEL_MIN_SAVED_SAMPLES: usize = 1_024;

/// Exact-terminal SIMD is cheap enough that Rayon must remove at least a
/// quarter of one 4,096-sample chunk from the serial critical path. The
/// `mc_exact_terminal_rayon_crossover` benchmark brackets this boundary at
/// 5,119/5,120 effective samples for 2-thread and host-sized pools.
#[cfg(feature = "parallel")]
fn should_auto_parallelize_exact_terminal(samples: usize, threads: usize) -> bool {
    estimated_parallel_work_savings(samples, 1, EXACT_PARALLEL_CHUNK_SAMPLES, threads)
        >= EXACT_PARALLEL_MIN_SAVED_SAMPLES
}

type ExactTerminalStats = RunningMoments;

#[derive(Debug, Clone, Copy)]
struct ExactTerminalParameters {
    option_type: OptionType,
    strike: f64,
    spot: f64,
    total_drift: f64,
    total_diffusion: f64,
    seed: u64,
    rng_kind: FastRngKind,
    reproducible: bool,
    antithetic: bool,
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    accuracy_tier: AccuracyTier,
}

#[inline(always)]
fn exact_terminal_normal(rng: &mut FastRng) -> f64 {
    sample_standard_normal(rng)
}

#[inline(always)]
fn exact_terminal_payoff(parameters: &ExactTerminalParameters, z: f64) -> f64 {
    let terminal = parameters.spot
        * parameters
            .total_diffusion
            .mul_add(z, parameters.total_drift)
            .exp();
    let payoff = vanilla_payoff(parameters.option_type, terminal, parameters.strike);
    if parameters.antithetic {
        let antithetic_terminal = parameters.spot
            * parameters
                .total_diffusion
                .mul_add(-z, parameters.total_drift)
                .exp();
        0.5 * (payoff
            + vanilla_payoff(
                parameters.option_type,
                antithetic_terminal,
                parameters.strike,
            ))
    } else {
        payoff
    }
}

fn exact_terminal_stats_scalar(
    parameters: &ExactTerminalParameters,
    rng: &mut FastRng,
    start: usize,
    end: usize,
) -> ExactTerminalStats {
    let mut stats = ExactTerminalStats::default();
    for _ in start..end {
        let z = exact_terminal_normal(rng);
        stats.record(exact_terminal_payoff(parameters, z));
    }
    stats
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn exact_terminal_stats_avx512(
    parameters: &ExactTerminalParameters,
    rng: &mut FastRng,
    start: usize,
    end: usize,
) -> ExactTerminalStats {
    use crate::math::simd_avx512::{exp_f64x8, fast_exp_f64x8};
    use std::arch::x86_64::*;

    let spot = _mm512_set1_pd(parameters.spot);
    let drift = _mm512_set1_pd(parameters.total_drift);
    let diffusion = _mm512_set1_pd(parameters.total_diffusion);
    let strike = _mm512_set1_pd(parameters.strike);
    let zero = _mm512_setzero_pd();
    let mut stats = ExactTerminalStats::default();
    let mut i = start;

    while i + 8 <= end {
        let mut normals = [0.0_f64; 8];
        for normal in &mut normals {
            *normal = exact_terminal_normal(rng);
        }

        let z = unsafe { _mm512_loadu_pd(normals.as_ptr()) };
        let exponent = _mm512_fmadd_pd(diffusion, z, drift);
        let growth = match parameters.accuracy_tier {
            AccuracyTier::High => unsafe { exp_f64x8(exponent) },
            AccuracyTier::Fast => unsafe { fast_exp_f64x8(exponent) },
        };
        let terminal = _mm512_mul_pd(spot, growth);
        let mut payoff = match parameters.option_type {
            OptionType::Call => _mm512_max_pd(_mm512_sub_pd(terminal, strike), zero),
            OptionType::Put => _mm512_max_pd(_mm512_sub_pd(strike, terminal), zero),
        };

        if parameters.antithetic {
            let anti_exponent = _mm512_fnmadd_pd(diffusion, z, drift);
            let anti_growth = match parameters.accuracy_tier {
                AccuracyTier::High => unsafe { exp_f64x8(anti_exponent) },
                AccuracyTier::Fast => unsafe { fast_exp_f64x8(anti_exponent) },
            };
            let anti_terminal = _mm512_mul_pd(spot, anti_growth);
            let anti_payoff = match parameters.option_type {
                OptionType::Call => _mm512_max_pd(_mm512_sub_pd(anti_terminal, strike), zero),
                OptionType::Put => _mm512_max_pd(_mm512_sub_pd(strike, anti_terminal), zero),
            };
            payoff = _mm512_mul_pd(_mm512_add_pd(payoff, anti_payoff), _mm512_set1_pd(0.5));
        }

        let mut values = [0.0_f64; 8];
        unsafe { _mm512_storeu_pd(values.as_mut_ptr(), payoff) };
        for value in values {
            stats.record(value);
        }
        i += 8;
    }

    stats.merge(exact_terminal_stats_scalar(parameters, rng, i, end));
    stats
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exact_terminal_stats_avx2(
    parameters: &ExactTerminalParameters,
    rng: &mut FastRng,
    start: usize,
    end: usize,
) -> ExactTerminalStats {
    use crate::math::approx_tier::tiered_exp_f64x4;
    use std::arch::x86_64::*;

    let spot = _mm256_set1_pd(parameters.spot);
    let drift = _mm256_set1_pd(parameters.total_drift);
    let diffusion = _mm256_set1_pd(parameters.total_diffusion);
    let strike = _mm256_set1_pd(parameters.strike);
    let zero = _mm256_setzero_pd();
    let mut stats = ExactTerminalStats::default();
    let mut i = start;

    while i + 4 <= end {
        let mut normals = [0.0_f64; 4];
        for normal in &mut normals {
            *normal = exact_terminal_normal(rng);
        }

        let z = unsafe { _mm256_loadu_pd(normals.as_ptr()) };
        let exponent = _mm256_fmadd_pd(diffusion, z, drift);
        let growth = unsafe { tiered_exp_f64x4(exponent, parameters.accuracy_tier) };
        let terminal = _mm256_mul_pd(spot, growth);
        let mut payoff = match parameters.option_type {
            OptionType::Call => _mm256_max_pd(_mm256_sub_pd(terminal, strike), zero),
            OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike, terminal), zero),
        };

        if parameters.antithetic {
            let anti_exponent = _mm256_fnmadd_pd(diffusion, z, drift);
            let anti_growth = unsafe { tiered_exp_f64x4(anti_exponent, parameters.accuracy_tier) };
            let anti_terminal = _mm256_mul_pd(spot, anti_growth);
            let anti_payoff = match parameters.option_type {
                OptionType::Call => _mm256_max_pd(_mm256_sub_pd(anti_terminal, strike), zero),
                OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike, anti_terminal), zero),
            };
            payoff = _mm256_mul_pd(_mm256_add_pd(payoff, anti_payoff), _mm256_set1_pd(0.5));
        }

        let mut values = [0.0_f64; 4];
        unsafe { _mm256_storeu_pd(values.as_mut_ptr(), payoff) };
        for value in values {
            stats.record(value);
        }
        i += 4;
    }

    stats.merge(exact_terminal_stats_scalar(parameters, rng, i, end));
    stats
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn exact_terminal_stats_neon(
    parameters: &ExactTerminalParameters,
    rng: &mut FastRng,
    start: usize,
    end: usize,
) -> ExactTerminalStats {
    use crate::math::approx_tier::tiered_exp_f64x2;
    use std::arch::aarch64::*;

    let spot = vdupq_n_f64(parameters.spot);
    let drift = vdupq_n_f64(parameters.total_drift);
    let diffusion = vdupq_n_f64(parameters.total_diffusion);
    let strike = vdupq_n_f64(parameters.strike);
    let zero = vdupq_n_f64(0.0);
    let mut stats = ExactTerminalStats::default();
    let mut i = start;

    while i + 2 <= end {
        let normals = [exact_terminal_normal(rng), exact_terminal_normal(rng)];
        let z = unsafe { vld1q_f64(normals.as_ptr()) };
        let exponent = vfmaq_f64(drift, diffusion, z);
        let growth = unsafe { tiered_exp_f64x2(exponent, parameters.accuracy_tier) };
        let terminal = vmulq_f64(spot, growth);
        let mut payoff = match parameters.option_type {
            OptionType::Call => vmaxq_f64(vsubq_f64(terminal, strike), zero),
            OptionType::Put => vmaxq_f64(vsubq_f64(strike, terminal), zero),
        };

        if parameters.antithetic {
            let anti_exponent = vfmsq_f64(drift, diffusion, z);
            let anti_growth = unsafe { tiered_exp_f64x2(anti_exponent, parameters.accuracy_tier) };
            let anti_terminal = vmulq_f64(spot, anti_growth);
            let anti_payoff = match parameters.option_type {
                OptionType::Call => vmaxq_f64(vsubq_f64(anti_terminal, strike), zero),
                OptionType::Put => vmaxq_f64(vsubq_f64(strike, anti_terminal), zero),
            };
            payoff = vmulq_n_f64(vaddq_f64(payoff, anti_payoff), 0.5);
        }

        let mut values = [0.0_f64; 2];
        unsafe { vst1q_f64(values.as_mut_ptr(), payoff) };
        for value in values {
            stats.record(value);
        }
        i += 2;
    }

    stats.merge(exact_terminal_stats_scalar(parameters, rng, i, end));
    stats
}

#[inline]
fn available_simd_width_f64() -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return 8;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return 4;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        2
    }

    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    {
        1
    }
}

fn exact_terminal_stats_range(
    parameters: &ExactTerminalParameters,
    start: usize,
    end: usize,
    use_simd: bool,
    stream_index: usize,
) -> ExactTerminalStats {
    let seed = resolve_stream_seed(parameters.seed, stream_index, parameters.reproducible);
    let mut rng = FastRng::from_seed(parameters.rng_kind, seed);

    if use_simd {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: Runtime feature detection guards the AVX-512 kernel.
                return unsafe { exact_terminal_stats_avx512(parameters, &mut rng, start, end) };
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: Runtime feature detection guards the AVX2+FMA kernel.
                return unsafe { exact_terminal_stats_avx2(parameters, &mut rng, start, end) };
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            // AArch64 guarantees NEON support.
            return unsafe { exact_terminal_stats_neon(parameters, &mut rng, start, end) };
        }
    }

    exact_terminal_stats_scalar(parameters, &mut rng, start, end)
}

fn exact_terminal_stats(
    parameters: &ExactTerminalParameters,
    samples: usize,
    backend: ExecutionBackend,
) -> ExactTerminalStats {
    let use_simd = !matches!(backend, ExecutionBackend::Scalar);
    let chunk_count = samples.div_ceil(EXACT_PARALLEL_CHUNK_SAMPLES);
    let simulate_chunk = |chunk_index: usize| {
        let start = chunk_index * EXACT_PARALLEL_CHUNK_SAMPLES;
        let end = (start + EXACT_PARALLEL_CHUNK_SAMPLES).min(samples);
        exact_terminal_stats_range(parameters, start, end, use_simd, chunk_index)
    };

    #[cfg(feature = "parallel")]
    let partials = if matches!(backend, ExecutionBackend::Parallel) {
        (0..chunk_count)
            .into_par_iter()
            .map(simulate_chunk)
            .collect::<Vec<_>>()
    } else {
        (0..chunk_count).map(simulate_chunk).collect::<Vec<_>>()
    };

    #[cfg(not(feature = "parallel"))]
    let partials = (0..chunk_count).map(simulate_chunk).collect::<Vec<_>>();

    // Fixed-size streams and an ordered reduction make seeded results
    // independent of scheduling and thread-pool size without reseeding every
    // individual path.
    let mut stats = ExactTerminalStats::default();
    for partial in partials {
        stats.merge(partial);
    }
    stats
}

impl MonteCarloInstrument for VanillaOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, _spot: f64) -> f64 {
        self.strike
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        vanilla_payoff(self.option_type, path[path.len() - 1], self.strike)
    }

    fn control_variate(&self, market: &Market, vol: f64) -> Option<ControlVariate> {
        if !matches!(self.exercise, ExerciseStyle::European) {
            return None;
        }
        if market.has_discrete_dividends() {
            return None;
        }

        let expected_discounted = black_scholes_price_with_dividend(
            self.option_type,
            market.spot,
            self.strike,
            market.rate,
            market.dividend_yield,
            vol,
            self.expiry,
        );
        let discount_factor = (-market.rate * self.expiry).exp();
        let option_type = self.option_type;
        let strike = self.strike;

        Some(ControlVariate {
            expected: expected_discounted / discount_factor,
            evaluator: Arc::new(move |path: &[f64]| {
                vanilla_payoff(option_type, path[path.len() - 1], strike)
            }),
        })
    }
}

impl MonteCarloInstrument for BarrierOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, _spot: f64) -> f64 {
        self.strike
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        let hit = path_hits_barrier(path, self.barrier.level, self.barrier.direction);
        let active = match self.barrier.style {
            BarrierStyle::In => hit,
            BarrierStyle::Out => !hit,
        };

        if active {
            vanilla_payoff(self.option_type, path[path.len() - 1], self.strike)
        } else {
            self.barrier.rebate
        }
    }

    fn payoff_from_path_with_rate(&self, path: &[f64], rate: f64) -> f64 {
        let hit_idx = first_barrier_hit_index(path, self.barrier.level, self.barrier.direction);
        let active = match self.barrier.style {
            BarrierStyle::In => hit_idx.is_some(),
            BarrierStyle::Out => hit_idx.is_none(),
        };

        if active {
            vanilla_payoff(self.option_type, path[path.len() - 1], self.strike)
        } else {
            match self.barrier.style {
                // Knock-out breached: the rebate is paid at the hit time, not
                // at expiry. The engine discounts the returned value from
                // maturity, so compound the rebate forward from t_hit to T.
                BarrierStyle::Out => {
                    let idx = hit_idx.expect("inactive knock-out implies a barrier hit");
                    let n_steps = path.len().saturating_sub(1);
                    let t_hit = if n_steps > 0 {
                        self.expiry * idx as f64 / n_steps as f64
                    } else {
                        0.0
                    };
                    self.barrier.rebate * (rate * (self.expiry - t_hit)).exp()
                }
                // Knock-in never triggered: rebate is paid at expiry.
                BarrierStyle::In => self.barrier.rebate,
            }
        }
    }
}

impl MonteCarloInstrument for AsianOption {
    fn validate_for_mc(&self) -> Result<(), PricingError> {
        self.validate()
    }

    fn maturity(&self) -> f64 {
        self.expiry
    }

    fn reference_strike(&self, spot: f64) -> f64 {
        match self.asian.strike_type {
            StrikeType::Fixed => self.strike,
            StrikeType::Floating => spot,
        }
    }

    fn payoff_from_path(&self, path: &[f64]) -> f64 {
        let avg = average_for_observations(
            path,
            self.expiry,
            &self.asian.observation_times,
            self.asian.averaging,
        );
        let st = path[path.len() - 1];
        match self.asian.strike_type {
            StrikeType::Fixed => vanilla_payoff(self.option_type, avg, self.strike),
            StrikeType::Floating => vanilla_payoff(self.option_type, st, avg),
        }
    }

    fn control_variate(&self, market: &Market, vol: f64) -> Option<ControlVariate> {
        if self.asian.averaging != Averaging::Arithmetic
            || self.asian.strike_type != StrikeType::Fixed
        {
            return None;
        }
        if market.has_discrete_dividends() {
            return None;
        }

        let expected_discounted = geometric_asian_discrete_fixed_closed_form(
            self.option_type,
            market.spot,
            self.strike,
            market.rate,
            market.dividend_yield,
            vol,
            &self.asian.observation_times,
        );
        let discount_factor = (-market.rate * self.expiry).exp();
        let option_type = self.option_type;
        let strike = self.strike;
        let expiry = self.expiry;
        let observation_times = self.asian.observation_times.clone();

        Some(ControlVariate {
            expected: expected_discounted / discount_factor,
            evaluator: Arc::new(move |path: &[f64]| {
                let geometric_avg = average_for_observations(
                    path,
                    expiry,
                    &observation_times,
                    Averaging::Geometric,
                );
                vanilla_payoff(option_type, geometric_avg, strike)
            }),
        })
    }
}

/// Generic Monte Carlo pricing engine.
#[derive(Debug, Clone)]
pub struct MonteCarloPricingEngine {
    /// Number of simulated paths.
    pub num_paths: usize,
    /// Number of time steps per path.
    pub num_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Pseudo-random number generator backend.
    pub(crate) rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub(crate) reproducible: bool,
    /// Variance reduction configuration.
    pub variance_reduction: VarianceReduction,
    /// Approximation accuracy tier for SIMD math (exp, inverse CDF).
    /// Defaults to automatic selection based on path count.
    pub accuracy_tier: Option<AccuracyTier>,
    /// Hardware execution strategy.
    pub execution_policy: ExecutionPolicy,
}

impl MonteCarloPricingEngine {
    /// Creates an engine with explicit path and time-step counts.
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
            variance_reduction: VarianceReduction::None,
            accuracy_tier: None,
            execution_policy: ExecutionPolicy::Auto,
        }
    }

    /// Sets the variance reduction scheme.
    pub fn with_variance_reduction(mut self, variance_reduction: VarianceReduction) -> Self {
        self.variance_reduction = variance_reduction;
        self
    }

    /// Returns the configured random-number generator.
    pub fn rng_kind(&self) -> FastRngKind {
        self.rng_kind
    }

    /// Returns whether seeded stream splitting is reproducible.
    pub fn is_reproducible(&self) -> bool {
        self.reproducible
    }

    /// Chooses RNG backend for path simulation.
    pub fn with_rng_kind(mut self, rng_kind: FastRngKind) -> Self {
        self.rng_kind = rng_kind;
        if matches!(rng_kind, FastRngKind::ThreadRng) {
            self.reproducible = false;
        }
        self
    }

    /// Uses a reproducible seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        // ThreadRng deliberately ignores deterministic seeds. Preserve that
        // backend choice, but never advertise the resulting stream as
        // reproducible.
        self.reproducible = !matches!(self.rng_kind, FastRngKind::ThreadRng);
        self
    }

    /// Uses non-reproducible stream seeds.
    pub fn with_randomized_streams(mut self) -> Self {
        self.reproducible = false;
        self
    }

    /// Uses thread-local RNG (non-reproducible).
    pub fn with_thread_rng(mut self) -> Self {
        self.rng_kind = FastRngKind::ThreadRng;
        self.reproducible = false;
        self
    }

    /// Override the approximation accuracy tier for SIMD math functions.
    ///
    /// By default the tier is selected automatically via
    /// `AccuracyTier::for_mc(num_paths, num_steps)`. Use this to force
    /// a specific tier (e.g., `AccuracyTier::High` for risk reports).
    pub fn with_accuracy_tier(mut self, tier: AccuracyTier) -> Self {
        self.accuracy_tier = Some(tier);
        self
    }

    /// Selects a hardware execution strategy.
    ///
    /// `Auto` uses workload thresholds and runtime capabilities. `Gpu` is
    /// available for eligible native exact-terminal European requests when
    /// the feature is compiled; `Jit` is not implemented by this engine.
    pub fn with_execution_policy(mut self, execution_policy: ExecutionPolicy) -> Self {
        self.execution_policy = execution_policy;
        self
    }

    /// Returns the effective accuracy tier for this engine configuration.
    pub fn effective_accuracy_tier(&self) -> AccuracyTier {
        self.accuracy_tier
            .unwrap_or_else(|| AccuracyTier::for_mc(self.num_paths, self.num_steps))
    }

    #[inline]
    fn exact_terminal_eligible(&self, instrument: &VanillaOption, market: &Market) -> bool {
        matches!(instrument.exercise, ExerciseStyle::European)
            && !market.has_discrete_dividends()
            && matches!(
                self.variance_reduction,
                VarianceReduction::None | VarianceReduction::Antithetic
            )
    }

    #[inline]
    fn effective_samples(&self) -> usize {
        if matches!(self.variance_reduction, VarianceReduction::Antithetic) {
            self.num_paths.div_ceil(2)
        } else {
            self.num_paths
        }
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    fn gpu_request_eligibility(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<(), &'static str> {
        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err("GPU execution only supports European vanilla options");
        }
        if market.has_discrete_dividends() {
            return Err("GPU execution does not support discrete dividends");
        }
        if !matches!(self.variance_reduction, VarianceReduction::None) {
            return Err("GPU execution does not support variance reduction");
        }
        if !self.reproducible {
            return Err("GPU execution requires reproducible seeded streams");
        }
        if !matches!(self.rng_kind, FastRngKind::Xoshiro256PlusPlus) {
            return Err("GPU execution cannot preserve the requested CPU RNG backend");
        }
        if self.num_paths > u32::MAX as usize {
            return Err("GPU execution requires num_paths to fit in u32");
        }
        Ok(())
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    fn should_auto_select_gpu(
        &self,
        vanilla: Option<&VanillaOption>,
        market: &Market,
        gpu_ready: bool,
    ) -> bool {
        gpu_ready
            && self.num_paths >= AUTO_GPU_MIN_PATHS
            && vanilla.is_some_and(|vanilla| self.gpu_request_eligibility(vanilla, market).is_ok())
    }

    fn auto_cpu_backend(&self, exact_terminal: bool) -> ExecutionBackend {
        let samples = self.effective_samples();
        #[cfg(feature = "parallel")]
        {
            let threads = rayon::current_num_threads();
            let use_parallel = if exact_terminal {
                should_auto_parallelize_exact_terminal(samples, threads)
            } else {
                // The higher-level engine currently constructs a one-stream
                // GBM generator for non-exact pricing.
                should_auto_parallelize_path(samples, self.num_steps, 1, threads)
            };
            if use_parallel {
                return ExecutionBackend::Parallel;
            }
        }

        // Exact-terminal SIMD has no separate allocation/setup pass. Keep the
        // portable native-width policy until stable benchmarks exist for each
        // supported CPU generation; one host's crossover must not silently
        // change dispatch on other machines.
        let simd_width = available_simd_width_f64();
        if exact_terminal && simd_width > 1 && samples >= simd_width {
            ExecutionBackend::Simd
        } else {
            ExecutionBackend::Scalar
        }
    }

    /// Resolves the requested execution policy for a concrete pricing call.
    ///
    /// The returned value is the backend that will be reported in
    /// [`PricingResult::diagnostics`].
    pub fn resolve_execution_backend<T>(
        &self,
        instrument: &T,
        market: &Market,
    ) -> Result<ExecutionBackend, PricingError>
    where
        T: MonteCarloInstrument + Sync + Any,
    {
        let vanilla = (instrument as &dyn Any).downcast_ref::<VanillaOption>();
        let exact_terminal =
            vanilla.is_some_and(|vanilla| self.exact_terminal_eligible(vanilla, market));
        let simd_available = exact_terminal && available_simd_width_f64() > 1;

        match self.execution_policy {
            ExecutionPolicy::Scalar => Ok(ExecutionBackend::Scalar),
            ExecutionPolicy::Simd if simd_available => Ok(ExecutionBackend::Simd),
            ExecutionPolicy::Simd => Err(PricingError::InvalidInput(
                "SIMD execution is unavailable for this build, CPU, or instrument".to_string(),
            )),
            ExecutionPolicy::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    Ok(ExecutionBackend::Parallel)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    Err(PricingError::InvalidInput(
                        "parallel execution requires the `parallel` feature".to_string(),
                    ))
                }
            }
            ExecutionPolicy::Gpu => {
                #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
                {
                    let vanilla = vanilla.ok_or_else(|| {
                        PricingError::InvalidInput(
                            "GPU execution only supports European vanilla options".to_string(),
                        )
                    })?;
                    self.gpu_request_eligibility(vanilla, market)
                        .map_err(|message| PricingError::InvalidInput(message.to_string()))?;
                    Ok(ExecutionBackend::Gpu)
                }
                #[cfg(not(all(feature = "gpu", not(target_family = "wasm"))))]
                {
                    Err(PricingError::InvalidInput(
                        "GPU execution requires a native build with the `gpu` feature".to_string(),
                    ))
                }
            }
            ExecutionPolicy::Jit => Err(PricingError::InvalidInput(
                "JIT execution is not implemented by MonteCarloPricingEngine".to_string(),
            )),
            ExecutionPolicy::Auto => {
                #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
                if self.should_auto_select_gpu(vanilla, market, crate::engines::gpu::gpu_is_ready())
                {
                    return Ok(ExecutionBackend::Gpu);
                }

                Ok(self.auto_cpu_backend(exact_terminal))
            }
        }
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    fn price_gpu_terminal(
        &self,
        instrument: &VanillaOption,
        market: &Market,
        vol: f64,
    ) -> Result<PricingResult, String> {
        // S0*exp(-qT) simulated with drift r is distributionally identical to
        // S0 simulated with drift r-q, allowing the q-free GPU API to preserve
        // continuous-dividend semantics.
        let adjusted_spot = market.spot * (-market.dividend_yield * instrument.expiry).exp();
        let result = crate::engines::gpu::mc_european_gpu(
            adjusted_spot,
            instrument.strike,
            market.rate,
            vol,
            instrument.expiry,
            self.num_paths,
            self.num_steps,
            self.seed,
            matches!(instrument.option_type, OptionType::Call),
        )?;

        Ok(self.gpu_pricing_result(vol, result))
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    fn gpu_pricing_result(
        &self,
        vol: f64,
        result: crate::engines::gpu::GpuMcResult,
    ) -> PricingResult {
        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumPaths, self.num_paths as f64);
        diagnostics.insert_key(DiagKey::NumSteps, 1.0);
        diagnostics.insert_key(DiagKey::NumThreads, 0.0);
        diagnostics.insert_key(DiagKey::VectorWidth, 0.0);
        diagnostics.insert_key(
            DiagKey::ExecutionBackend,
            ExecutionBackend::Gpu.diagnostic_code(),
        );
        diagnostics.insert_key(DiagKey::Vol, vol);

        PricingResult {
            price: result.price,
            stderr: Some(result.stderr),
            greeks: None,
            diagnostics,
        }
    }

    fn price_exact_terminal(
        &self,
        instrument: &VanillaOption,
        market: &Market,
        vol: f64,
        backend: ExecutionBackend,
    ) -> PricingResult {
        let antithetic = matches!(self.variance_reduction, VarianceReduction::Antithetic);
        let samples = self.effective_samples();
        let maturity = instrument.expiry;
        let parameters = ExactTerminalParameters {
            option_type: instrument.option_type,
            strike: instrument.strike,
            spot: market.spot,
            total_drift: (market.rate - market.dividend_yield - 0.5 * vol * vol) * maturity,
            total_diffusion: vol * maturity.sqrt(),
            seed: self.seed,
            rng_kind: self.rng_kind,
            reproducible: self.reproducible,
            antithetic,
            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
            accuracy_tier: self.effective_accuracy_tier(),
        };
        let stats = exact_terminal_stats(&parameters, samples, backend);
        let n = stats.count() as f64;
        let mean = stats.mean();
        let variance = stats.sample_variance();
        let discount = (-market.rate * maturity).exp();
        let available_vector_width = available_simd_width_f64();
        let vector_width = if matches!(backend, ExecutionBackend::Scalar)
            || stats.count() < available_vector_width
        {
            1
        } else {
            available_vector_width
        };
        let num_threads = execution_threads(backend);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumPaths, self.num_paths as f64);
        diagnostics.insert_key(DiagKey::NumSteps, 1.0);
        diagnostics.insert_key(DiagKey::NumThreads, num_threads as f64);
        diagnostics.insert_key(DiagKey::VectorWidth, vector_width as f64);
        diagnostics.insert_key(DiagKey::ExecutionBackend, backend.diagnostic_code());
        diagnostics.insert_key(DiagKey::Vol, vol);

        PricingResult {
            price: discount * mean,
            stderr: Some(discount * (variance / n).sqrt()),
            greeks: None,
            diagnostics,
        }
    }
}

#[inline]
fn execution_threads(backend: ExecutionBackend) -> usize {
    if !matches!(backend, ExecutionBackend::Parallel) {
        return 1;
    }

    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Dedicated Monte Carlo engine for arithmetic-average fixed-strike Asian options.
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticAsianMC {
    /// Number of simulated paths.
    pub paths: usize,
    /// Number of time steps per path.
    pub steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Pseudo-random number generator backend.
    pub(crate) rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub(crate) reproducible: bool,
    /// Enables geometric-Asian control variate.
    pub control_variate: bool,
}

impl ArithmeticAsianMC {
    /// Creates an arithmetic Asian Monte Carlo engine.
    pub fn new(paths: usize, steps: usize, seed: u64) -> Self {
        Self {
            paths,
            steps,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
            control_variate: true,
        }
    }

    /// Enables/disables control variate.
    pub fn with_control_variate(mut self, control_variate: bool) -> Self {
        self.control_variate = control_variate;
        self
    }

    /// Returns the configured random-number generator.
    pub fn rng_kind(&self) -> FastRngKind {
        self.rng_kind
    }

    /// Returns whether seeded stream splitting is reproducible.
    pub fn is_reproducible(&self) -> bool {
        self.reproducible
    }

    /// Chooses RNG backend for path simulation.
    pub fn with_rng_kind(mut self, rng_kind: FastRngKind) -> Self {
        self.rng_kind = rng_kind;
        if matches!(rng_kind, FastRngKind::ThreadRng) {
            self.reproducible = false;
        }
        self
    }

    /// Uses a reproducible seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.reproducible = !matches!(self.rng_kind, FastRngKind::ThreadRng);
        self
    }

    /// Uses non-reproducible stream seeds.
    pub fn with_randomized_streams(mut self) -> Self {
        self.reproducible = false;
        self
    }

    /// Uses thread-local RNG (non-reproducible).
    pub fn with_thread_rng(mut self) -> Self {
        self.rng_kind = FastRngKind::ThreadRng;
        self.reproducible = false;
        self
    }
}

impl PricingEngine<AsianOption> for ArithmeticAsianMC {
    fn price(
        &self,
        instrument: &AsianOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;
        market.validate()?;

        if instrument.asian.averaging != Averaging::Arithmetic {
            return Err(PricingError::InvalidInput(
                "ArithmeticAsianMC requires Averaging::Arithmetic".to_string(),
            ));
        }
        if instrument.asian.strike_type != StrikeType::Fixed {
            return Err(PricingError::InvalidInput(
                "ArithmeticAsianMC currently supports StrikeType::Fixed only".to_string(),
            ));
        }
        if self.paths == 0 {
            return Err(PricingError::InvalidInput(
                "arithmetic asian mc paths must be > 0".to_string(),
            ));
        }
        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "arithmetic asian mc steps must be > 0".to_string(),
            ));
        }

        let maturity = instrument.expiry;
        let vol = market.checked_vol_for(instrument.strike, maturity)?;

        let generator = GbmPathGenerator {
            model: Gbm {
                mu: market.rate - market.dividend_yield,
                sigma: vol,
            },
            s0: market.spot,
            maturity,
            steps: self.steps,
        };

        let mut engine = MonteCarloEngine::new(self.paths, self.seed)
            .with_rng_kind(self.rng_kind)
            .with_antithetic(true);
        if !self.reproducible {
            engine = engine.with_randomized_streams();
        }
        if self.control_variate && !market.has_discrete_dividends() {
            let expected_discounted = geometric_asian_discrete_fixed_closed_form(
                instrument.option_type,
                market.spot,
                instrument.strike,
                market.rate,
                market.dividend_yield,
                vol,
                &instrument.asian.observation_times,
            );
            let discount_factor = (-market.rate * maturity).exp();
            let option_type = instrument.option_type;
            let strike = instrument.strike;
            let expiry = instrument.expiry;
            let observation_times = instrument.asian.observation_times.clone();

            let cv = ControlVariate {
                expected: expected_discounted / discount_factor,
                evaluator: Arc::new(move |path: &[f64]| {
                    let geometric_avg = average_for_observations(
                        path,
                        expiry,
                        &observation_times,
                        Averaging::Geometric,
                    );
                    vanilla_payoff(option_type, geometric_avg, strike)
                }),
            };
            engine = engine.with_control_variate(cv);
        }

        let discount = (-market.rate * maturity).exp();
        let (price, stderr) = if market.has_discrete_dividends() {
            let generator = DiscreteDividendGbmPathGenerator::new(
                generator.clone(),
                market.dividend_schedule.clone(),
            );
            engine.run(
                &generator,
                |path| {
                    let arithmetic_avg = average_for_observations(
                        path,
                        maturity,
                        &instrument.asian.observation_times,
                        Averaging::Arithmetic,
                    );
                    vanilla_payoff(instrument.option_type, arithmetic_avg, instrument.strike)
                },
                discount,
            )
        } else {
            engine.run(
                &generator,
                |path| {
                    let arithmetic_avg = average_for_observations(
                        path,
                        maturity,
                        &instrument.asian.observation_times,
                        Averaging::Arithmetic,
                    );
                    vanilla_payoff(instrument.option_type, arithmetic_avg, instrument.strike)
                },
                discount,
            )
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::NumPaths, self.paths as f64);
        diagnostics.insert_key(crate::core::DiagKey::NumSteps, self.steps as f64);
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }
}

impl<T> PricingEngine<T> for MonteCarloPricingEngine
where
    T: MonteCarloInstrument + Sync + Any,
{
    fn price(&self, instrument: &T, market: &Market) -> Result<PricingResult, PricingError> {
        instrument.validate_for_mc()?;
        market.validate()?;

        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }
        if self.num_steps == 0 {
            return Err(PricingError::InvalidInput(
                "num_steps must be > 0".to_string(),
            ));
        }

        let maturity = instrument.maturity();
        if maturity < 0.0 {
            return Err(PricingError::InvalidInput(
                "instrument maturity must be >= 0".to_string(),
            ));
        }

        if maturity == 0.0 {
            let payoff = instrument.payoff_from_path(&[market.spot]);
            let mut diagnostics = crate::core::Diagnostics::new();
            diagnostics.insert_key(DiagKey::NumPaths, self.num_paths as f64);
            diagnostics.insert_key(DiagKey::NumSteps, 0.0);
            diagnostics.insert_key(DiagKey::NumThreads, 1.0);
            diagnostics.insert_key(DiagKey::VectorWidth, 1.0);
            diagnostics.insert_key(
                DiagKey::ExecutionBackend,
                ExecutionBackend::Scalar.diagnostic_code(),
            );
            return Ok(PricingResult {
                price: payoff,
                stderr: Some(0.0),
                greeks: None,
                diagnostics,
            });
        }

        let ref_strike = instrument.reference_strike(market.spot).max(1e-12);
        let vol = market.checked_vol_for(ref_strike, maturity)?;

        let resolved_backend = self.resolve_execution_backend(instrument, market)?;
        #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
        let backend = {
            let mut backend = resolved_backend;
            if matches!(backend, ExecutionBackend::Gpu) {
                let vanilla = (instrument as &dyn Any)
                    .downcast_ref::<VanillaOption>()
                    .expect("GPU backend resolution guarantees a vanilla option");
                match self.price_gpu_terminal(vanilla, market, vol) {
                    Ok(result) => return Ok(result),
                    Err(error) if matches!(self.execution_policy, ExecutionPolicy::Gpu) => {
                        return Err(PricingError::NumericalError(format!(
                            "GPU execution failed: {error}"
                        )));
                    }
                    Err(_) => {
                        // Auto policy treats adapter/device/limit errors as a
                        // capability miss and continues on the best CPU backend.
                        backend =
                            self.auto_cpu_backend(self.exact_terminal_eligible(vanilla, market));
                    }
                }
            }
            backend
        };
        #[cfg(not(all(feature = "gpu", not(target_family = "wasm"))))]
        let backend = resolved_backend;

        if let Some(vanilla) = (instrument as &dyn Any).downcast_ref::<VanillaOption>()
            && self.exact_terminal_eligible(vanilla, market)
        {
            return Ok(self.price_exact_terminal(vanilla, market, vol, backend));
        }

        let generator = GbmPathGenerator {
            model: Gbm {
                mu: market.rate - market.dividend_yield,
                sigma: vol,
            },
            s0: market.spot,
            maturity,
            steps: self.num_steps,
        };

        let generic_policy = match backend {
            ExecutionBackend::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    CpuExecutionPolicy::Parallel
                }
                #[cfg(not(feature = "parallel"))]
                {
                    unreachable!("parallel backend requires the `parallel` feature")
                }
            }
            ExecutionBackend::Scalar | ExecutionBackend::Simd => CpuExecutionPolicy::Scalar,
            ExecutionBackend::Gpu | ExecutionBackend::Jit => {
                unreachable!("unsupported backends are rejected by resolve_execution_backend")
            }
        };
        let mut base = MonteCarloEngine::new(self.num_paths, self.seed)
            .with_rng_kind(self.rng_kind)
            .with_execution_policy(generic_policy);
        if !self.reproducible {
            base = base.with_randomized_streams();
        }
        let base = match &self.variance_reduction {
            VarianceReduction::Antithetic => base.with_antithetic(true),
            _ => base.with_antithetic(false),
        };

        let engine = match &self.variance_reduction {
            VarianceReduction::ControlVariate => {
                if let Some(cv) = instrument.control_variate(market, vol) {
                    base.with_control_variate(cv)
                } else {
                    base
                }
            }
            _ => base,
        };

        let discount_factor = (-market.rate * maturity).exp();
        let (price, stderr) = if market.has_discrete_dividends() {
            let generator = DiscreteDividendGbmPathGenerator::new(
                generator.clone(),
                market.dividend_schedule.clone(),
            );
            engine.run(
                &generator,
                |path| instrument.payoff_from_path_with_rate(path, market.rate),
                discount_factor,
            )
        } else {
            engine.run(
                &generator,
                |path| instrument.payoff_from_path_with_rate(path, market.rate),
                discount_factor,
            )
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumPaths, self.num_paths as f64);
        diagnostics.insert_key(DiagKey::NumSteps, self.num_steps as f64);
        diagnostics.insert_key(DiagKey::NumThreads, execution_threads(backend) as f64);
        diagnostics.insert_key(DiagKey::VectorWidth, 1.0);
        diagnostics.insert_key(DiagKey::ExecutionBackend, backend.diagnostic_code());
        diagnostics.insert_key(DiagKey::Vol, vol);

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }

    fn price_with_greeks_aad(
        &self,
        instrument: &T,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        if let Some(vanilla) = (instrument as &dyn Any).downcast_ref::<VanillaOption>() {
            return super::mc_aad::mc_european_pathwise_aad(self, vanilla, market);
        }

        self.price(instrument, market)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AsianSpec, PricingEngine};
    use crate::instruments::{AsianOption, VanillaOption};
    use crate::market::{DividendEvent, DividendSchedule};
    use crate::math::arena::PricingArena;

    #[test]
    fn centered_moments_preserve_constant_and_nan_semantics() {
        let value = 3.162_277_660_168_379_6e-25;
        let mut stats = RunningMoments::default();
        for sample in [value; 3] {
            stats.record(sample);
        }
        assert_eq!(stats.sample_variance(), 0.0);

        stats.record(f64::NAN);
        assert!(stats.mean().is_nan());
        assert!(stats.sample_variance().is_nan());
    }

    #[test]
    fn centered_moments_preserve_small_nonzero_variance() {
        let mut stats = RunningMoments::default();
        for sample in [100.0 - 2.0e-10, 100.0 - 1.0e-10, 100.0, 100.0 + 1.0e-10] {
            stats.record(sample);
        }
        assert!(stats.sample_variance().is_finite());
        assert!(stats.sample_variance() > 0.0);
        assert!(stats.sample_variance() < 1.0e-18);
    }

    #[test]
    fn mc_european_call_matches_black_scholes_within_four_stderr() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let result = MonteCarloPricingEngine::new(100_000, 252, 42)
            .price(&option, &market)
            .expect("mc pricing succeeds");

        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        let stderr = result.stderr.expect("MC reports standard error");
        let tolerance = 4.0 * stderr + 1.0e-12;
        assert!(
            (result.price - bs).abs() <= tolerance,
            "MC/BS error exceeds sampling budget: mc={} bs={} stderr={} tolerance={}",
            result.price,
            bs,
            stderr,
            tolerance
        );
    }

    #[test]
    fn near_deterministic_exact_terminal_has_finite_nonzero_stderr() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.0)
            .dividend_yield(0.0)
            .flat_vol(1.0e-10)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(1.0, 1.0);
        let result = MonteCarloPricingEngine::new(100_000, 1, 123)
            .with_execution_policy(ExecutionPolicy::Scalar)
            .price(&option, &market)
            .unwrap();
        let stderr = result.stderr.unwrap();
        assert!(stderr.is_finite());
        assert!(stderr > 0.0, "stderr={stderr}");
        assert!(stderr < 1.0e-9, "stderr={stderr}");
    }

    #[test]
    fn thread_rng_never_claims_seeded_reproducibility() {
        for engine in [
            MonteCarloPricingEngine::new(32, 1, 1)
                .with_thread_rng()
                .with_seed(42),
            MonteCarloPricingEngine::new(32, 1, 1)
                .with_seed(42)
                .with_thread_rng(),
        ] {
            assert_eq!(engine.rng_kind, FastRngKind::ThreadRng);
            assert!(!engine.reproducible);
        }

        for engine in [
            ArithmeticAsianMC::new(32, 4, 1)
                .with_thread_rng()
                .with_seed(42),
            ArithmeticAsianMC::new(32, 4, 1)
                .with_seed(42)
                .with_thread_rng(),
        ] {
            assert_eq!(engine.rng_kind, FastRngKind::ThreadRng);
            assert!(!engine.reproducible);
        }
    }

    #[test]
    fn pricing_boundary_revalidates_directly_constructed_market() {
        let valid = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(100.0, 1.0);
        let engine = MonteCarloPricingEngine::new(256, 1, 42);
        for invalid in [
            Market {
                spot: f64::NAN,
                ..valid.clone()
            },
            Market {
                rate: f64::INFINITY,
                ..valid.clone()
            },
            Market {
                dividend_yield: f64::NEG_INFINITY,
                ..valid.clone()
            },
            Market {
                vol: crate::market::VolSource::Flat(f64::NAN),
                ..valid.clone()
            },
        ] {
            assert!(engine.price(&option, &invalid).is_err());
        }
    }

    #[test]
    fn mc_antithetic_has_lower_stderr_than_plain_mc() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let plain = MonteCarloPricingEngine::new(100_000, 252, 42)
            .price(&option, &market)
            .expect("plain MC succeeds");
        let antithetic = MonteCarloPricingEngine::new(100_000, 252, 42)
            .with_variance_reduction(VarianceReduction::Antithetic)
            .price(&option, &market)
            .expect("antithetic MC succeeds");

        assert!(
            antithetic.stderr.expect("stderr present") < plain.stderr.expect("stderr present"),
            "expected antithetic stderr < plain stderr"
        );
    }

    #[test]
    fn mc_european_control_variate_matches_bs_within_four_stderr() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let result = MonteCarloPricingEngine::new(100_000, 252, 42)
            .with_variance_reduction(VarianceReduction::ControlVariate)
            .price(&option, &market)
            .expect("control-variate MC succeeds");

        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        let stderr = result.stderr.expect("MC reports standard error");
        let tolerance = 4.0 * stderr + 1.0e-12;
        assert!(
            (result.price - bs).abs() <= tolerance,
            "control-variate MC/BS error exceeds sampling budget: mc={} bs={} stderr={} tolerance={}",
            result.price,
            bs,
            stderr,
            tolerance
        );
    }

    #[test]
    fn geometric_asian_matches_exact_scipy_discrete_reference() {
        // 12 equally-spaced averaging dates on [0, 1].
        let observation_times: Vec<f64> = (0..12).map(|m| m as f64 / 11.0).collect();
        let price = geometric_asian_discrete_fixed_closed_form(
            OptionType::Call,
            100.0,
            100.0,
            0.05,
            0.0,
            0.20,
            &observation_times,
        );

        // Independently evaluated from the exact discrete lognormal moments
        // with SciPy 1.17.1's normal CDF.
        assert!(
            (price - 5.439_231_718_855_941).abs() <= 3.0e-12,
            "geometric Asian reference mismatch: got={price}"
        );
    }

    #[test]
    fn arithmetic_asian_mc_matches_scipy_qmc_reference_within_sampling_error() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .expect("valid market");

        let observation_times: Vec<f64> = (0..12).map(|m| m as f64 / 11.0).collect();
        let option = AsianOption::new(
            OptionType::Call,
            100.0,
            1.0,
            AsianSpec {
                averaging: Averaging::Arithmetic,
                strike_type: StrikeType::Fixed,
                observation_times,
            },
        );

        let result = ArithmeticAsianMC::new(100_000, 252, 42)
            .with_control_variate(true)
            .price(&option, &market)
            .expect("arithmetic Asian MC succeeds");

        // Independent SciPy scrambled-Sobol reference for the same twelve
        // discrete observations (32 independent scrambles x 2^20 paths).
        // Its replicate standard error is combined with the implementation's
        // reported MC standard error; there is no fixed price cushion.
        let expected: f64 = 5.675_787_099_986_969;
        let reference_stderr: f64 = 5.739_423_093_943_816_6e-5;
        let implementation_stderr = result.stderr.expect("MC stderr");
        let combined_stderr = implementation_stderr.hypot(reference_stderr);
        let roundoff = 32.0 * f64::EPSILON * expected.abs().max(1.0);
        let tolerance = 4.0 * combined_stderr + roundoff;
        assert!(
            (result.price - expected).abs() <= tolerance,
            "arithmetic Asian MC mismatch: mc={} expected={} implementation_stderr={implementation_stderr} reference_stderr={reference_stderr} tolerance={}",
            result.price,
            expected,
            tolerance
        );
    }

    #[test]
    fn mc_knock_out_rebate_is_discounted_from_hit_time() {
        use crate::instruments::BarrierOption;

        // High rate + long maturity: a rebate paid at the (early) hit time is
        // worth substantially more than the same rebate discounted from expiry.
        let rate = 0.10;
        let expiry = 5.0;
        let rebate = 10.0;
        let market = Market::builder()
            .spot(100.0)
            .rate(rate)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .expect("valid market");

        // Barrier just above spot: virtually every path knocks out early, and
        // surviving paths (S_t < 105 for all t) can never reach strike 200, so
        // the price is purely the rebate value.
        let option = BarrierOption::builder()
            .call()
            .strike(200.0)
            .expiry(expiry)
            .up_and_out(105.0)
            .rebate(rebate)
            .build()
            .expect("valid barrier option");

        let result = MonteCarloPricingEngine::new(100_000, 250, 42)
            .price(&option, &market)
            .expect("mc pricing succeeds");

        // Pay-at-expiry would cap the value at rebate * exp(-r*T) ≈ 6.065.
        let pay_at_expiry_bound = rebate * (-rate * expiry).exp();
        assert!(
            result.price > pay_at_expiry_bound * 1.3,
            "rebate-at-hit value {} should clearly exceed pay-at-expiry bound {}",
            result.price,
            pay_at_expiry_bound
        );
        assert!(
            result.price < rebate,
            "discounted rebate {} cannot exceed undiscounted rebate {}",
            result.price,
            rebate
        );

        // Independent SciPy Brownian-bridge Sobol reference for the *same*
        // 250-date discrete-monitoring contract, including the surviving
        // K=200 terminal call (64 scrambles x 2^15 paths).  The former
        // continuous first-passage transform valued a different contract.
        let reference = 9.227_409_760_796_569;
        let reference_stderr = 6.234_678_252_332_711e-4;
        let implementation_stderr = result.stderr.expect("MC stderr");
        let combined_stderr = implementation_stderr.hypot(reference_stderr);
        let roundoff = 32.0 * f64::EPSILON * reference;
        assert!(
            (result.price - reference).abs() <= 4.0 * combined_stderr + roundoff,
            "discrete rebate-at-hit mismatch: mc={} reference={} implementation_stderr={implementation_stderr} reference_stderr={reference_stderr}",
            result.price,
            reference,
        );
    }

    #[test]
    fn mc_seeded_xoshiro_is_reproducible() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let first = MonteCarloPricingEngine::new(20_000, 64, 123)
            .with_rng_kind(FastRngKind::Xoshiro256PlusPlus)
            .price(&option, &market)
            .expect("first run succeeds");
        let second = MonteCarloPricingEngine::new(20_000, 64, 123)
            .with_rng_kind(FastRngKind::Xoshiro256PlusPlus)
            .price(&option, &market)
            .expect("second run succeeds");

        assert_eq!(first.price, second.price);
        assert_eq!(first.stderr, second.stderr);
    }

    #[test]
    fn scalar_exact_terminal_dispatch_matches_generic_one_step_distribution() {
        let market = Market::builder()
            .spot(103.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.27)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_put(100.0, 1.25);
        let engine = MonteCarloPricingEngine::new(8_193, 252, 91)
            .with_execution_policy(ExecutionPolicy::Scalar);

        let optimized = engine
            .price(&option, &market)
            .expect("optimized pricing succeeds");

        let generator = GbmPathGenerator {
            model: Gbm {
                mu: market.rate - market.dividend_yield,
                sigma: 0.27,
            },
            s0: market.spot,
            maturity: option.expiry,
            steps: 1,
        };
        let (generic_price, generic_stderr) = MonteCarloEngine::new(8_193, 91)
            .with_execution_policy(CpuExecutionPolicy::Scalar)
            .run(
                &generator,
                |path| vanilla_payoff(option.option_type, path[1], option.strike),
                (-market.rate * option.expiry).exp(),
            );

        let optimized_stderr = optimized.stderr.expect("optimized stderr");
        let combined_stderr = optimized_stderr.hypot(generic_stderr);
        assert!(
            (optimized.price - generic_price).abs() <= 4.0 * combined_stderr,
            "optimized={} generic={} combined_stderr={combined_stderr}",
            optimized.price,
            generic_price
        );
        assert_eq!(
            optimized.diagnostics.get("num_steps"),
            Some(&1.0),
            "terminal-only pricing should report one simulated step"
        );
        assert_eq!(
            optimized.diagnostics.get("execution_backend"),
            Some(&ExecutionBackend::Scalar.diagnostic_code())
        );
        assert_eq!(optimized.diagnostics.get("vector_width"), Some(&1.0));
    }

    #[test]
    fn exact_terminal_dispatch_preserves_rng_and_antithetic_configuration() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.02)
            .dividend_yield(0.005)
            .flat_vol(0.23)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(105.0, 0.75);

        for rng_kind in [
            FastRngKind::Xoshiro256PlusPlus,
            FastRngKind::Pcg64,
            FastRngKind::StdRng,
        ] {
            let engine = MonteCarloPricingEngine::new(10_001, 128, 777)
                .with_rng_kind(rng_kind)
                .with_variance_reduction(VarianceReduction::Antithetic)
                .with_execution_policy(ExecutionPolicy::Scalar);
            let first = engine.price(&option, &market).expect("first price");
            let second = engine.price(&option, &market).expect("second price");
            assert_eq!(first.price, second.price, "rng={rng_kind:?}");
            assert_eq!(first.stderr, second.stderr, "rng={rng_kind:?}");
        }
    }

    #[test]
    fn path_dependent_semantics_force_safe_generic_fallback() {
        let schedule =
            DividendSchedule::new(vec![DividendEvent::cash(0.5, 1.5).expect("valid dividend")])
                .expect("valid schedule");
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.0)
            .dividend_schedule(schedule)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let result = MonteCarloPricingEngine::new(2_000, 17, 42)
            .with_execution_policy(ExecutionPolicy::Scalar)
            .price(&option, &market)
            .expect("pricing succeeds");

        assert_eq!(result.diagnostics.get("num_steps"), Some(&17.0));
        assert_eq!(
            result.diagnostics.get("execution_backend"),
            Some(&ExecutionBackend::Scalar.diagnostic_code())
        );
    }

    #[test]
    fn control_variate_preserves_generic_path_semantics() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let result = MonteCarloPricingEngine::new(2_000, 19, 42)
            .with_variance_reduction(VarianceReduction::ControlVariate)
            .with_execution_policy(ExecutionPolicy::Scalar)
            .price(&option, &market)
            .expect("pricing succeeds");

        assert_eq!(result.diagnostics.get("num_steps"), Some(&19.0));
    }

    #[test]
    fn unsupported_jit_backend_returns_clear_error() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let error = MonteCarloPricingEngine::new(1_000, 16, 42)
            .with_execution_policy(ExecutionPolicy::Jit)
            .price(&option, &market)
            .expect_err("unsupported backend should fail");
        assert!(error.to_string().contains("not implemented"));
    }

    #[cfg(not(all(feature = "gpu", not(target_family = "wasm"))))]
    #[test]
    fn explicit_gpu_requires_native_gpu_feature() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let error = MonteCarloPricingEngine::new(1_000, 16, 42)
            .with_execution_policy(ExecutionPolicy::Gpu)
            .price(&option, &market)
            .expect_err("GPU backend should require its feature");
        assert!(error.to_string().contains("`gpu` feature"));
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    #[test]
    fn gpu_policy_resolution_validates_semantics_without_requesting_adapter() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let explicit = MonteCarloPricingEngine::new(10_000, 16, 42)
            .with_execution_policy(ExecutionPolicy::Gpu);
        assert_eq!(
            explicit
                .resolve_execution_backend(&option, &market)
                .expect("continuous yield is transformed into adjusted spot"),
            ExecutionBackend::Gpu
        );

        let auto = MonteCarloPricingEngine::new(AUTO_GPU_MIN_PATHS, 16, 42);
        assert!(!auto.should_auto_select_gpu(Some(&option), &market, false));
        assert!(auto.should_auto_select_gpu(Some(&option), &market, true));
        assert!(
            !MonteCarloPricingEngine::new(AUTO_GPU_MIN_PATHS - 1, 16, 42).should_auto_select_gpu(
                Some(&option),
                &market,
                true
            )
        );

        for invalid in [
            explicit.clone().with_rng_kind(FastRngKind::Pcg64),
            explicit.clone().with_randomized_streams(),
            explicit
                .clone()
                .with_variance_reduction(VarianceReduction::Antithetic),
        ] {
            assert!(invalid.resolve_execution_backend(&option, &market).is_err());
        }
    }

    #[cfg(all(feature = "gpu", not(target_family = "wasm")))]
    #[test]
    fn gpu_result_reports_backend_and_f32_execution_metadata() {
        let result = MonteCarloPricingEngine::new(2_000_000, 252, 42).gpu_pricing_result(
            0.2,
            crate::engines::gpu::GpuMcResult {
                price: 10.25,
                stderr: 0.0125,
            },
        );

        assert_eq!(result.price, 10.25);
        assert_eq!(result.stderr, Some(0.0125));
        assert_eq!(
            result.diagnostics.get("execution_backend"),
            Some(&ExecutionBackend::Gpu.diagnostic_code())
        );
        assert_eq!(result.diagnostics.get("num_steps"), Some(&1.0));
        assert_eq!(result.diagnostics.get("num_threads"), Some(&0.0));
        assert_eq!(result.diagnostics.get("vector_width"), Some(&0.0));
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_exact_terminal_matches_scalar_for_identical_streams() {
        if available_simd_width_f64() == 1 {
            return;
        }

        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.01)
            .flat_vol(0.3)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_put(110.0, 1.5);
        let scalar = MonteCarloPricingEngine::new(16_387, 252, 123)
            .with_execution_policy(ExecutionPolicy::Scalar)
            .price(&option, &market)
            .expect("scalar price");
        let simd = MonteCarloPricingEngine::new(16_387, 252, 123)
            .with_execution_policy(ExecutionPolicy::Simd)
            .with_accuracy_tier(AccuracyTier::High)
            .price(&option, &market)
            .expect("SIMD price");

        assert!((scalar.price - simd.price).abs() <= 1.0e-10);
        assert!(
            (scalar.stderr.expect("scalar stderr") - simd.stderr.expect("SIMD stderr")).abs()
                <= 1.0e-10
        );
        assert_eq!(
            simd.diagnostics.get("execution_backend"),
            Some(&ExecutionBackend::Simd.diagnostic_code())
        );
        assert_eq!(
            simd.diagnostics.get("vector_width"),
            Some(&(available_simd_width_f64() as f64))
        );
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_exact_terminal_applies_accuracy_tier() {
        let half_ln_2 = std::f64::consts::LN_2 * 0.5;
        let market = Market::builder()
            .spot(100.0)
            .rate(-half_ln_2)
            .dividend_yield(0.0)
            // Keep the stochastic term below one ULP so every lane exercises
            // the reduced-range endpoint where the two exp tiers differ.
            .flat_vol(1e-300)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(1.0, 1.0);
        let price = |tier| {
            MonteCarloPricingEngine::new(2, 1, 42)
                .with_execution_policy(ExecutionPolicy::Simd)
                .with_accuracy_tier(tier)
                .price(&option, &market)
                .expect("NEON exact-terminal price")
                .price
        };

        let fast = price(AccuracyTier::Fast);
        let high = price(AccuracyTier::High);
        assert_ne!(
            fast.to_bits(),
            high.to_bits(),
            "Fast and High must dispatch to distinct NEON exp kernels"
        );
        assert!(((fast - high) / high).abs() <= 1e-7);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn exact_terminal_scalar_tail_reports_unit_vector_width() {
        if available_simd_width_f64() == 1 {
            return;
        }

        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);

        let simd = MonteCarloPricingEngine::new(1, 8, 42)
            .with_execution_policy(ExecutionPolicy::Simd)
            .price(&option, &market)
            .expect("SIMD policy accepts an exact-terminal scalar tail");
        assert_eq!(simd.diagnostics.get("vector_width"), Some(&1.0));

        #[cfg(feature = "parallel")]
        {
            let parallel = MonteCarloPricingEngine::new(1, 8, 42)
                .with_execution_policy(ExecutionPolicy::Parallel)
                .price(&option, &market)
                .expect("parallel policy accepts an exact-terminal scalar tail");
            assert_eq!(parallel.diagnostics.get("vector_width"), Some(&1.0));
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn auto_simd_dispatch_uses_effective_samples_at_native_vector_boundary() {
        let width = available_simd_width_f64();
        if width == 1 {
            return;
        }
        let minimum_samples = width;
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(100.0, 1.0);

        #[cfg(feature = "parallel")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        for (variance_reduction, requested_below, requested_at) in [
            (
                VarianceReduction::None,
                minimum_samples - 1,
                minimum_samples,
            ),
            (
                VarianceReduction::Antithetic,
                2 * (minimum_samples - 1),
                2 * minimum_samples - 1,
            ),
        ] {
            let resolve = |paths, variance_reduction| {
                MonteCarloPricingEngine::new(paths, 1, 42)
                    .with_variance_reduction(variance_reduction)
                    .resolve_execution_backend(&option, &market)
                    .unwrap()
            };
            #[cfg(feature = "parallel")]
            let (below, at) = pool.install(|| {
                (
                    resolve(requested_below, variance_reduction.clone()),
                    resolve(requested_at, variance_reduction),
                )
            });
            #[cfg(not(feature = "parallel"))]
            let (below, at) = (
                resolve(requested_below, variance_reduction.clone()),
                resolve(requested_at, variance_reduction),
            );
            assert_eq!(below, ExecutionBackend::Scalar);
            assert_eq!(at, ExecutionBackend::Simd);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn auto_exact_terminal_rayon_boundary_uses_effective_samples_and_pool_size() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(100.0, 1.0);
        let host_threads = rayon::current_num_threads();
        let mut pool_sizes = vec![1_usize, 2, host_threads];
        pool_sizes.sort_unstable();
        pool_sizes.dedup();

        for threads in pool_sizes {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for (variance_reduction, requested_below, requested_at) in [
                (VarianceReduction::None, 5_119, 5_120),
                (VarianceReduction::Antithetic, 2 * 5_119, 2 * 5_120 - 1),
            ] {
                let resolve = |paths| {
                    MonteCarloPricingEngine::new(paths, 1, 42)
                        .with_variance_reduction(variance_reduction.clone())
                        .resolve_execution_backend(&option, &market)
                        .unwrap()
                };
                let (below, at) =
                    pool.install(|| (resolve(requested_below), resolve(requested_at)));
                assert_ne!(below, ExecutionBackend::Parallel);
                if threads == 1 {
                    assert_ne!(at, ExecutionBackend::Parallel);
                } else {
                    assert_eq!(at, ExecutionBackend::Parallel);
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn auto_path_rayon_boundary_uses_effective_samples_and_pool_size() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = VanillaOption::american_put(100.0, 1.0);
        let host_threads = rayon::current_num_threads();
        let mut pool_sizes = vec![1_usize, 2, host_threads];
        pool_sizes.sort_unstable();
        pool_sizes.dedup();

        for threads in pool_sizes {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (effective_below, effective_at) =
                if threads <= 2 { (511, 512) } else { (767, 768) };
            for (variance_reduction, requested_below, requested_at) in [
                (VarianceReduction::None, effective_below, effective_at),
                (
                    VarianceReduction::Antithetic,
                    2 * effective_below,
                    2 * effective_at - 1,
                ),
            ] {
                let resolve = |paths| {
                    MonteCarloPricingEngine::new(paths, 1, 42)
                        .with_variance_reduction(variance_reduction.clone())
                        .resolve_execution_backend(&option, &market)
                        .unwrap()
                };
                let (below, at) =
                    pool.install(|| (resolve(requested_below), resolve(requested_at)));
                assert_eq!(below, ExecutionBackend::Scalar);
                if threads == 1 {
                    assert_eq!(at, ExecutionBackend::Scalar);
                } else {
                    assert_eq!(at, ExecutionBackend::Parallel);
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn auto_accuracy_avoids_fast_tier_bias_in_low_variance_case() {
        if available_simd_width_f64() == 1 {
            return;
        }
        let market = Market::builder()
            .spot(100.0)
            .rate(-0.5 * std::f64::consts::LN_2)
            .dividend_yield(0.0)
            .flat_vol(1.0e-4)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(1.0, 1.0);
        let base = MonteCarloPricingEngine::new(16_384, 1, 42)
            .with_variance_reduction(VarianceReduction::Antithetic);
        let scalar = base
            .clone()
            .with_execution_policy(ExecutionPolicy::Scalar)
            .price(&option, &market)
            .unwrap();
        let fast = base
            .clone()
            .with_execution_policy(ExecutionPolicy::Simd)
            .with_accuracy_tier(AccuracyTier::Fast)
            .price(&option, &market)
            .unwrap();
        let automatic = base.price(&option, &market).unwrap();

        let fast_bias = (fast.price - scalar.price).abs();
        let auto_bias = (automatic.price - scalar.price).abs();
        assert_eq!(AccuracyTier::for_mc(16_384, 1), AccuracyTier::High);
        assert!(
            fast_bias > 5.0 * fast.stderr.unwrap(),
            "fast_bias={fast_bias:e}, stderr={:?}",
            fast.stderr
        );
        assert!(auto_bias < 1.0e-10, "auto_bias={auto_bias:e}");
        assert!(
            auto_bias < fast_bias * 1.0e-3,
            "auto_bias={auto_bias:e}, fast_bias={fast_bias:e}"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_exact_terminal_is_independent_of_pool_size() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let engine = MonteCarloPricingEngine::new(50_003, 252, 314)
            .with_execution_policy(ExecutionPolicy::Parallel);

        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| engine.price(&option, &market).expect("parallel price"))
        };
        let two_threads = run(2);
        let four_threads = run(4);

        assert_eq!(two_threads.price, four_threads.price);
        assert_eq!(two_threads.stderr, four_threads.stderr);
        assert_eq!(
            two_threads.diagnostics.get("execution_backend"),
            Some(&ExecutionBackend::Parallel.diagnostic_code())
        );
        assert_eq!(two_threads.diagnostics.get("num_threads"), Some(&2.0));
        assert_eq!(four_threads.diagnostics.get("num_threads"), Some(&4.0));
    }

    #[test]
    fn mc_european_with_arena_converges_to_black_scholes() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let n_paths = 200_000;
        let n_steps = 1; // n_steps is unused by exact simulation but kept for API compat

        let mut arena = PricingArena::with_capacity(n_paths, n_steps);
        let arena_result = mc_european_with_arena(&option, &market, n_paths, n_steps, &mut arena);

        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.2, 1.0);
        let stderr = arena_result.stderr.expect("arena MC reports stderr");
        let roundoff = 16.0 * f64::EPSILON * bs.abs().max(1.0);
        assert!(
            (arena_result.price - bs).abs() <= 4.0 * stderr + roundoff,
            "arena MC/BS error exceeds sampling budget: mc={} bs={} stderr={stderr}",
            arena_result.price,
            bs,
        );
    }

    #[test]
    fn mc_european_with_arena_is_reusable_across_calls() {
        let market = Market::builder()
            .spot(95.0)
            .rate(0.02)
            .dividend_yield(0.01)
            .flat_vol(0.3)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_put(100.0, 1.5);
        let mut arena = PricingArena::with_capacity(128, 16);

        let first = mc_european_with_arena(&option, &market, 4_000, 32, &mut arena);
        let second = mc_european_with_arena(&option, &market, 4_000, 96, &mut arena);
        let third = mc_european_with_arena(&option, &market, 4_000, 32, &mut arena);

        assert!(first.price.is_finite());
        assert!(second.price.is_finite());
        assert!(third.price.is_finite());
        // Payoff buffer grows to accommodate the larger path count.
        assert!(arena.payoff_buffer.len() >= 4_000);
        // Same seed + same parameters ⇒ identical results.
        assert_eq!(first.price, third.price);
        assert_eq!(first.stderr, third.stderr);
    }

    #[test]
    fn arena_pricer_covers_guards_expiry_and_discrete_dividend_fallback() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let mut arena = PricingArena::with_capacity(32, 8);

        let option = VanillaOption::european_call(100.0, 1.0);
        assert!(
            mc_european_with_arena(&option, &market, 0, 8, &mut arena)
                .price
                .is_nan()
        );

        let american = VanillaOption::american_call(100.0, 1.0);
        assert!(
            mc_european_with_arena(&american, &market, 32, 8, &mut arena)
                .price
                .is_nan()
        );

        let expired = VanillaOption::european_put(105.0, 0.0);
        let expired_result = mc_european_with_arena(&expired, &market, 32, 8, &mut arena);
        assert_eq!(expired_result.price, 5.0);
        assert_eq!(expired_result.stderr, Some(0.0));

        let schedule =
            DividendSchedule::new(vec![DividendEvent::cash(0.5, 1.25).unwrap()]).unwrap();
        let dividend_market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.0)
            .dividend_schedule(schedule)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let arena_result = mc_european_with_arena(&option, &dividend_market, 257, 7, &mut arena);
        let generic_result = MonteCarloPricingEngine::new(257, 7, ARENA_MC_SEED)
            .price(&option, &dividend_market)
            .unwrap();
        assert_eq!(arena_result.price, generic_result.price);
        assert_eq!(arena_result.stderr, generic_result.stderr);
        assert_eq!(arena_result.diagnostics, generic_result.diagnostics);
    }

    #[test]
    fn asian_path_semantics_cover_floating_strike_and_geometric_averages() {
        let path = [100.0, 80.0, 120.0];
        let observation_times = vec![0.0, 0.5, 1.0];

        let floating_call = AsianOption::new(
            OptionType::Call,
            100.0,
            1.0,
            AsianSpec {
                averaging: Averaging::Arithmetic,
                strike_type: StrikeType::Floating,
                observation_times: observation_times.clone(),
            },
        );
        assert_eq!(floating_call.reference_strike(97.0), 97.0);
        assert_eq!(floating_call.payoff_from_path(&path), 20.0);

        let floating_put = AsianOption::new(
            OptionType::Put,
            100.0,
            1.0,
            AsianSpec {
                averaging: Averaging::Geometric,
                strike_type: StrikeType::Floating,
                observation_times,
            },
        );
        let geometric_average = ((100.0_f64.ln() + 80.0_f64.ln() + 120.0_f64.ln()) / 3.0).exp();
        assert_eq!(floating_put.reference_strike(103.0), 103.0);
        assert_eq!(floating_put.payoff_from_path(&path), 0.0);

        let floating_geometric_call = AsianOption {
            option_type: OptionType::Call,
            ..floating_put.clone()
        };
        assert!(
            (floating_geometric_call.payoff_from_path(&path) - (120.0 - geometric_average)).abs()
                <= 4.0 * f64::EPSILON * 120.0
        );
        assert!(
            floating_geometric_call
                .control_variate(
                    &Market::builder()
                        .spot(100.0)
                        .rate(0.03)
                        .dividend_yield(0.0)
                        .flat_vol(0.2)
                        .build()
                        .unwrap(),
                    0.2,
                )
                .is_none()
        );
    }

    fn arithmetic_asian_for_guard_tests(
        averaging: Averaging,
        strike_type: StrikeType,
    ) -> AsianOption {
        AsianOption::new(
            OptionType::Call,
            100.0,
            1.0,
            AsianSpec {
                averaging,
                strike_type,
                observation_times: vec![0.5, 1.0],
            },
        )
    }

    #[test]
    fn arithmetic_asian_engine_rejects_unsupported_contracts_and_empty_work() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.02)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let arithmetic_fixed =
            arithmetic_asian_for_guard_tests(Averaging::Arithmetic, StrikeType::Fixed);
        let geometric_fixed =
            arithmetic_asian_for_guard_tests(Averaging::Geometric, StrikeType::Fixed);
        let arithmetic_floating =
            arithmetic_asian_for_guard_tests(Averaging::Arithmetic, StrikeType::Floating);

        assert!(
            ArithmeticAsianMC::new(32, 4, 1)
                .price(&geometric_fixed, &market)
                .unwrap_err()
                .to_string()
                .contains("Averaging::Arithmetic")
        );
        assert!(
            ArithmeticAsianMC::new(32, 4, 1)
                .price(&arithmetic_floating, &market)
                .unwrap_err()
                .to_string()
                .contains("StrikeType::Fixed")
        );
        assert!(
            ArithmeticAsianMC::new(0, 4, 1)
                .price(&arithmetic_fixed, &market)
                .is_err()
        );
        assert!(
            ArithmeticAsianMC::new(32, 0, 1)
                .price(&arithmetic_fixed, &market)
                .is_err()
        );

        assert!(
            MonteCarloPricingEngine::new(0, 4, 1)
                .price(&VanillaOption::european_call(100.0, 1.0), &market)
                .is_err()
        );
        assert!(
            MonteCarloPricingEngine::new(32, 0, 1)
                .price(&VanillaOption::european_call(100.0, 1.0), &market)
                .is_err()
        );
    }

    #[test]
    fn arithmetic_asian_discrete_dividend_disables_control_variate_exactly() {
        let schedule = DividendSchedule::new(vec![DividendEvent::cash(0.5, 2.0).unwrap()]).unwrap();
        let market = Market::builder()
            .spot(100.0)
            .rate(0.02)
            .dividend_yield(0.0)
            .dividend_schedule(schedule)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let option = arithmetic_asian_for_guard_tests(Averaging::Arithmetic, StrikeType::Fixed);
        let with_requested_cv = ArithmeticAsianMC::new(512, 8, 77)
            .with_control_variate(true)
            .price(&option, &market)
            .unwrap();
        let without_cv = ArithmeticAsianMC::new(512, 8, 77)
            .with_control_variate(false)
            .price(&option, &market)
            .unwrap();
        assert_eq!(with_requested_cv.price, without_cv.price);
        assert_eq!(with_requested_cv.stderr, without_cv.stderr);
    }

    #[test]
    fn generic_engine_zero_maturity_and_configuration_accessors_are_exact() {
        let market = Market::builder()
            .spot(90.0)
            .rate(0.02)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let expired = VanillaOption::european_put(100.0, 0.0);
        let result = MonteCarloPricingEngine::new(16, 4, 1)
            .price(&expired, &market)
            .unwrap();
        assert_eq!(result.price, 10.0);
        assert_eq!(result.stderr, Some(0.0));
        assert_eq!(result.diagnostics.get("num_steps"), Some(&0.0));

        let engine = MonteCarloPricingEngine::new(64, 8, 1)
            .with_rng_kind(FastRngKind::Pcg64)
            .with_randomized_streams()
            .with_accuracy_tier(AccuracyTier::Fast)
            .with_execution_policy(ExecutionPolicy::Scalar);
        assert_eq!(engine.rng_kind(), FastRngKind::Pcg64);
        assert!(!engine.is_reproducible());
        assert_eq!(engine.effective_accuracy_tier(), AccuracyTier::Fast);
        assert_eq!(
            engine.resolve_execution_backend(&expired, &market).unwrap(),
            ExecutionBackend::Scalar
        );
        assert!(engine.with_seed(99).is_reproducible());
    }
}
