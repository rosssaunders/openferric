use std::sync::Arc;

use crate::core::{
    Averaging, BarrierStyle, ExerciseStyle, Instrument, OptionType, PricingEngine, PricingError,
    PricingResult, StrikeType,
};
use crate::instruments::{AsianOption, BarrierOption, VanillaOption};
use crate::market::Market;
use crate::math::arena::PricingArena;
use crate::math::fast_norm::beasley_springer_moro_inv_cdf;
use crate::math::fast_rng::{FastRngKind, Xoshiro256PlusPlus, uniform_open01};
use crate::mc::{ControlVariate, GbmPathGenerator, MonteCarloEngine};
use crate::models::Gbm;
use crate::pricing::asian::geometric_asian_discrete_fixed_closed_form;
use crate::pricing::european::black_scholes_price;

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
    _n_steps: usize,
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

    if instrument.validate().is_err() || !matches!(instrument.exercise, ExerciseStyle::European) {
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

    let n = n_paths as f64;
    let payoffs = &payoff_buffer[..n_paths];
    // Kahan summation for improved numerical accuracy (the FMA is free).
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut comp = 0.0_f64;
    let mut comp_sq = 0.0_f64;
    for &v in payoffs {
        // Kahan summation for sum
        let y = v - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        // Kahan summation for sum_sq
        let v2 = v * v;
        let y2 = v2 - comp_sq;
        let t2 = sum_sq + y2;
        comp_sq = (t2 - sum_sq) - y2;
        sum_sq = t2;
    }
    let mean = sum / n;
    let variance = if n_paths > 1 {
        (sum_sq - sum * sum / n) / (n - 1.0)
    } else {
        0.0
    };

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
    use std::arch::x86_64::*;
    use crate::math::simd_math::{fast_exp_f64x4, fill_normals_simd, splat_f64x4, store_f64x4};

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
            let growth = unsafe { fast_exp_f64x4(exponent) };
            let s_terminal = _mm256_mul_pd(spot_v, growth);

            let payoff_v = match option_type {
                crate::core::OptionType::Call => _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v),
                crate::core::OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v),
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
            let growth = unsafe { fast_exp_f64x4(exponent) };
            let s_terminal = _mm256_mul_pd(spot_v, growth);

            let payoff_v = match option_type {
                crate::core::OptionType::Call => _mm256_max_pd(_mm256_sub_pd(s_terminal, strike_v), zero_v),
                crate::core::OptionType::Put => _mm256_max_pd(_mm256_sub_pd(strike_v, s_terminal), zero_v),
            };

            unsafe { store_f64x4(payoff_buffer, *i + j, payoff_v) };
            j += 4;
        }
        *i += batch;
    }
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
    pub rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub reproducible: bool,
    /// Variance reduction configuration.
    pub variance_reduction: VarianceReduction,
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
        }
    }

    /// Sets the variance reduction scheme.
    pub fn with_variance_reduction(mut self, variance_reduction: VarianceReduction) -> Self {
        self.variance_reduction = variance_reduction;
        self
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
        self.reproducible = true;
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
    pub rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub reproducible: bool,
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
        self.reproducible = true;
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
        let vol = market.vol_for(instrument.strike, maturity);
        if vol <= 0.0 || !vol.is_finite() {
            return Err(PricingError::InvalidInput(
                "market volatility must be finite and > 0".to_string(),
            ));
        }

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
        if self.control_variate {
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
        let (price, stderr) = engine.run(
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
        );

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
    T: MonteCarloInstrument + Sync,
{
    fn price(&self, instrument: &T, market: &Market) -> Result<PricingResult, PricingError> {
        instrument.validate_for_mc()?;

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
            return Ok(PricingResult {
                price: payoff,
                stderr: Some(0.0),
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let ref_strike = instrument.reference_strike(market.spot).max(1e-12);
        let vol = market.vol_for(ref_strike, maturity);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
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

        let mut base =
            MonteCarloEngine::new(self.num_paths, self.seed).with_rng_kind(self.rng_kind);
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
        let (price, stderr) = engine.run(
            &generator,
            |path| instrument.payoff_from_path(path),
            discount_factor,
        );

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::NumPaths, self.num_paths as f64);
        diagnostics.insert_key(crate::core::DiagKey::NumSteps, self.num_steps as f64);
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AsianSpec, PricingEngine};
    use crate::instruments::{AsianOption, VanillaOption};
    use crate::math::arena::PricingArena;

    #[test]
    fn mc_european_call_matches_black_scholes_within_one_percent() {
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
        let rel_err = ((result.price - bs) / bs).abs();
        assert!(
            rel_err <= 0.01,
            "MC/BS relative error too high: mc={} bs={} rel_err={}",
            result.price,
            bs,
            rel_err
        );
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
    fn mc_european_control_variate_is_within_half_percent_of_bs() {
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
        let rel_err = ((result.price - bs) / bs).abs();
        assert!(
            rel_err <= 0.005,
            "MC/BS relative error too high with control variate: mc={} bs={} rel_err={}",
            result.price,
            bs,
            rel_err
        );
    }

    #[test]
    fn geometric_asian_kemna_vorst_reference_is_approximately_5_31() {
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

        assert!(
            (price - 5.31).abs() <= 0.15,
            "geometric Asian reference mismatch: got={} expected≈5.31",
            price
        );
    }

    #[test]
    fn arithmetic_asian_mc_kemna_turnbull_reference_within_two_percent() {
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

        let expected = 5.73;
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err <= 0.02,
            "arithmetic Asian MC mismatch: mc={} expected={} rel_err={}",
            result.price,
            expected,
            rel_err
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
        let rel_err = ((arena_result.price - bs) / bs).abs();
        assert!(
            rel_err <= 0.01,
            "arena MC/BS relative error too high: mc={} bs={} rel_err={}",
            arena_result.price,
            bs,
            rel_err
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
}
