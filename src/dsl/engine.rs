//! Multi-asset Monte Carlo engine for DSL products.
//!
//! Generates correlated GBM paths and evaluates compiled products.

use crate::core::{
    DiagKey, Diagnostics, ExecutionBackend, ExecutionPolicy, Greeks, Instrument, PricingEngine,
    PricingError, PricingResult,
};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::dsl::eval::evaluate_product_with_plan_batch_neon;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::dsl::eval::evaluate_product_with_plan_batch_x86;
use crate::dsl::eval::{
    ProductExecutionPlan, build_execution_plan_for_validated_product,
    evaluate_product_with_plan_in_place,
};
use crate::dsl::ir::{CompiledProduct, UnderlyingType};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use crate::dsl::jit::{JitEvaluationScratch, JitProductEvaluator, jit_platform_supported};
use crate::dsl::market::{AssetMarketData, MultiAssetMarket};
use crate::engines::monte_carlo::correlated_mc::{
    cholesky_for_correlation, sample_correlated_normals_cholesky_with_scratch,
};
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::math::simd_math::{fast_exp_f64x4, ln_f64x4, load_f64x4, store_f64x4};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::math::simd_neon::{load_f64x2, simd_exp_f64x2, simd_ln_f64x2, store_f64x2};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use std::collections::VecDeque;
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use std::sync::{Arc, Mutex, OnceLock};

/// Per-asset stepping strategy for path generation.
///
/// Built once from `AssetMarketData` + time step `dt` and reused for every path.
#[derive(Debug, Clone, Copy)]
enum AssetStepper {
    /// Geometric Brownian Motion for equities (with dividend yield) and FX (Garman-Kohlhagen).
    Gbm { drift: f64, diffusion: f64 },
    /// Schwartz one-factor mean-reverting log-price model for commodities.
    SchwartzOneF {
        long_run_log_mean: f64,
        exp_neg_kappa_dt: f64,
        vol_step: f64,
    },
    /// Vasicek / Hull-White short-rate model for interest rates.
    Vasicek {
        long_run_mean: f64,
        exp_neg_a_dt: f64,
        vol_step: f64,
    },
}

fn market_asset_type(asset: &AssetMarketData) -> UnderlyingType {
    match asset {
        AssetMarketData::Equity { .. } => UnderlyingType::Equity,
        AssetMarketData::Fx { .. } => UnderlyingType::Fx,
        AssetMarketData::Commodity { .. } => UnderlyingType::Commodity,
        AssetMarketData::Rate { .. } => UnderlyingType::Rate,
    }
}

fn underlying_type_name(underlying_type: UnderlyingType) -> &'static str {
    match underlying_type {
        UnderlyingType::Equity => "equity",
        UnderlyingType::Fx => "FX",
        UnderlyingType::Commodity => "commodity",
        UnderlyingType::Rate => "rate",
    }
}

impl AssetStepper {
    /// Build a stepper from asset market data and simulation parameters.
    fn from_asset(asset: &AssetMarketData, rate: f64, dt: f64) -> Self {
        let sqrt_dt = dt.sqrt();
        match asset {
            AssetMarketData::Equity {
                vol,
                dividend_yield,
                ..
            } => Self::Gbm {
                drift: (rate - dividend_yield - 0.5 * vol * vol) * dt,
                diffusion: vol * sqrt_dt,
            },
            AssetMarketData::Fx {
                vol,
                domestic_rate,
                foreign_rate,
                ..
            } => {
                // Garman-Kohlhagen: drift = (r_d - r_f - 0.5 σ²) dt
                let _ = rate; // use explicit domestic/foreign rates
                Self::Gbm {
                    drift: (domestic_rate - foreign_rate - 0.5 * vol * vol) * dt,
                    diffusion: vol * sqrt_dt,
                }
            }
            AssetMarketData::Commodity { vol, kappa, mu, .. } => Self::SchwartzOneF {
                long_run_log_mean: *mu,
                exp_neg_kappa_dt: (-kappa * dt).exp(),
                vol_step: vol * sqrt_dt,
            },
            AssetMarketData::Rate {
                vol,
                mean_reversion,
                long_run_mean,
                ..
            } => Self::Vasicek {
                long_run_mean: *long_run_mean,
                exp_neg_a_dt: (-mean_reversion * dt).exp(),
                vol_step: vol * sqrt_dt,
            },
        }
    }

    /// Advance one time step given previous value and a standard normal draw.
    #[inline]
    fn step(self, prev: f64, z: f64) -> f64 {
        match self {
            Self::Gbm { drift, diffusion } => prev * (drift + diffusion * z).exp(),
            Self::SchwartzOneF {
                long_run_log_mean,
                exp_neg_kappa_dt,
                vol_step,
            } => {
                // Schwartz 1-factor: d(ln S) = κ(μ - ln S) dt + σ dW
                let log_prev = prev.ln();
                let log_next = long_run_log_mean
                    + exp_neg_kappa_dt * (log_prev - long_run_log_mean)
                    + vol_step * z;
                log_next.exp()
            }
            Self::Vasicek {
                long_run_mean,
                exp_neg_a_dt,
                vol_step,
            } => {
                // Vasicek: dr = a(θ - r) dt + σ dW
                long_run_mean + exp_neg_a_dt * (prev - long_run_mean) + vol_step * z
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn step_batch_x86(self, prev: &[f64; 4], z: &[f64; 4], next: &mut [f64; 4]) {
        use std::arch::x86_64::*;

        let prev_v = unsafe { load_f64x4(prev, 0) };
        let z_v = unsafe { _mm256_loadu_pd(z.as_ptr()) };

        let next_v = unsafe {
            match self {
                Self::Gbm { drift, diffusion } => {
                    let drift_v = _mm256_set1_pd(drift);
                    let diffusion_v = _mm256_set1_pd(diffusion);
                    let exponent = _mm256_fmadd_pd(diffusion_v, z_v, drift_v);
                    let growth = fast_exp_f64x4(exponent);
                    _mm256_mul_pd(prev_v, growth)
                }
                Self::SchwartzOneF {
                    long_run_log_mean,
                    exp_neg_kappa_dt,
                    vol_step,
                } => {
                    let mean_v = _mm256_set1_pd(long_run_log_mean);
                    let revert_v = _mm256_set1_pd(exp_neg_kappa_dt);
                    let vol_v = _mm256_set1_pd(vol_step);
                    let log_prev = ln_f64x4(prev_v);
                    let reverted =
                        _mm256_fmadd_pd(revert_v, _mm256_sub_pd(log_prev, mean_v), mean_v);
                    let log_next = _mm256_fmadd_pd(vol_v, z_v, reverted);
                    fast_exp_f64x4(log_next)
                }
                Self::Vasicek {
                    long_run_mean,
                    exp_neg_a_dt,
                    vol_step,
                } => {
                    let mean_v = _mm256_set1_pd(long_run_mean);
                    let revert_v = _mm256_set1_pd(exp_neg_a_dt);
                    let vol_v = _mm256_set1_pd(vol_step);
                    let reverted = _mm256_fmadd_pd(revert_v, _mm256_sub_pd(prev_v, mean_v), mean_v);
                    _mm256_fmadd_pd(vol_v, z_v, reverted)
                }
            }
        };

        unsafe { store_f64x4(next, 0, next_v) };
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn step_batch_neon(self, prev: &[f64; 2], z: &[f64; 2], next: &mut [f64; 2]) {
        use std::arch::aarch64::*;

        let prev_v = unsafe { load_f64x2(prev, 0) };
        let z_v = unsafe { vld1q_f64(z.as_ptr()) };

        let next_v = unsafe {
            match self {
                Self::Gbm { drift, diffusion } => {
                    let drift_v = vdupq_n_f64(drift);
                    let diffusion_v = vdupq_n_f64(diffusion);
                    let exponent = vfmaq_f64(drift_v, diffusion_v, z_v);
                    let growth = simd_exp_f64x2(exponent);
                    vmulq_f64(prev_v, growth)
                }
                Self::SchwartzOneF {
                    long_run_log_mean,
                    exp_neg_kappa_dt,
                    vol_step,
                } => {
                    let mean_v = vdupq_n_f64(long_run_log_mean);
                    let revert_v = vdupq_n_f64(exp_neg_kappa_dt);
                    let vol_v = vdupq_n_f64(vol_step);
                    let log_prev = simd_ln_f64x2(prev_v);
                    let reverted = vfmaq_f64(mean_v, revert_v, vsubq_f64(log_prev, mean_v));
                    let log_next = vfmaq_f64(reverted, vol_v, z_v);
                    simd_exp_f64x2(log_next)
                }
                Self::Vasicek {
                    long_run_mean,
                    exp_neg_a_dt,
                    vol_step,
                } => {
                    let mean_v = vdupq_n_f64(long_run_mean);
                    let revert_v = vdupq_n_f64(exp_neg_a_dt);
                    let vol_v = vdupq_n_f64(vol_step);
                    let reverted = vfmaq_f64(mean_v, revert_v, vsubq_f64(prev_v, mean_v));
                    vfmaq_f64(reverted, vol_v, z_v)
                }
            }
        };

        unsafe { store_f64x2(next, 0, next_v) };
    }
}

/// A DSL product wrapped as an `Instrument` for compatibility.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DslProduct {
    pub product: CompiledProduct,
}

impl Instrument for DslProduct {
    fn instrument_type(&self) -> &str {
        "DslProduct"
    }
}

impl DslProduct {
    pub fn new(product: CompiledProduct) -> Self {
        Self { product }
    }
}

/// Multi-asset Monte Carlo engine for DSL products.
#[derive(Debug, Clone)]
pub struct DslMonteCarloEngine {
    pub num_paths: usize,
    pub num_steps: usize,
    pub seed: u64,
    pub rng_kind: FastRngKind,
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
const MIN_AUTO_JIT_PATHS: usize = 4_096;

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
const JIT_EVALUATOR_CACHE_CAPACITY: usize = 16;

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
struct CachedJitEvaluator {
    product: CompiledProduct,
    num_steps: usize,
    rate_bits: u64,
    evaluator: Arc<JitProductEvaluator>,
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn cached_jit_evaluator(
    product: &CompiledProduct,
    num_steps: usize,
    rate: f64,
) -> Result<Arc<JitProductEvaluator>, PricingError> {
    static CACHE: OnceLock<Mutex<VecDeque<CachedJitEvaluator>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    let rate_bits = rate.to_bits();

    {
        let mut guard = cache.lock().map_err(|_| {
            PricingError::NumericalError("DSL JIT evaluator cache lock is poisoned".to_string())
        })?;
        if let Some(position) = guard.iter().position(|entry| {
            entry.num_steps == num_steps
                && entry.rate_bits == rate_bits
                && &entry.product == product
        }) {
            let entry = guard
                .remove(position)
                .expect("position came from the same cache");
            let evaluator = Arc::clone(&entry.evaluator);
            guard.push_back(entry);
            return Ok(evaluator);
        }
    }

    // Compile outside the lock: executable-code generation can be relatively
    // expensive, and unrelated pricing calls should not be blocked by it.
    let compiled = Arc::new(
        JitProductEvaluator::compile(product, num_steps, rate)
            .map_err(|error| PricingError::NumericalError(error.to_string()))?,
    );

    let mut guard = cache.lock().map_err(|_| {
        PricingError::NumericalError("DSL JIT evaluator cache lock is poisoned".to_string())
    })?;
    if let Some(position) = guard.iter().position(|entry| {
        entry.num_steps == num_steps && entry.rate_bits == rate_bits && &entry.product == product
    }) {
        let entry = guard
            .remove(position)
            .expect("position came from the same cache");
        let evaluator = Arc::clone(&entry.evaluator);
        guard.push_back(entry);
        return Ok(evaluator);
    }
    if guard.len() == JIT_EVALUATOR_CACHE_CAPACITY {
        guard.pop_front();
    }
    guard.push_back(CachedJitEvaluator {
        product: product.clone(),
        num_steps,
        rate_bits,
        evaluator: Arc::clone(&compiled),
    });
    Ok(compiled)
}

impl DslMonteCarloEngine {
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
        }
    }

    /// Resolve a requested policy to the backend that will execute.
    pub fn resolve_execution_backend(
        &self,
        execution_policy: ExecutionPolicy,
    ) -> Result<ExecutionBackend, PricingError> {
        let simd_available =
            simd_runtime_available() && should_simd_path_batch(self.num_paths, simd_path_lanes());

        match execution_policy {
            ExecutionPolicy::Scalar => Ok(ExecutionBackend::Scalar),
            ExecutionPolicy::Simd if simd_available => Ok(ExecutionBackend::Simd),
            ExecutionPolicy::Simd => Err(PricingError::InvalidInput(
                "SIMD execution is unavailable for this build, CPU, or path count".to_string(),
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
            ExecutionPolicy::Gpu => Err(PricingError::InvalidInput(
                "GPU execution is not implemented by DslMonteCarloEngine".to_string(),
            )),
            ExecutionPolicy::Jit => {
                #[cfg(all(feature = "jit", not(target_family = "wasm")))]
                {
                    if jit_platform_supported() {
                        Ok(ExecutionBackend::Jit)
                    } else {
                        Err(PricingError::InvalidInput(format!(
                            "JIT execution is unsupported on {}; the current native backend requires x86_64",
                            std::env::consts::ARCH
                        )))
                    }
                }
                #[cfg(not(all(feature = "jit", not(target_family = "wasm"))))]
                {
                    Err(PricingError::InvalidInput(
                        "JIT execution requires the native `jit` feature".to_string(),
                    ))
                }
            }
            ExecutionPolicy::Auto => {
                // The current SIMD evaluator handles four/two paths together
                // and is preferable to scalar JIT code when available. On a
                // non-SIMD build, JIT compilation is amortized only for a
                // sufficiently large path count.
                #[cfg(all(feature = "jit", not(target_family = "wasm")))]
                if jit_platform_supported()
                    && !simd_available
                    && self.num_paths >= MIN_AUTO_JIT_PATHS
                {
                    return Ok(ExecutionBackend::Jit);
                }

                #[cfg(feature = "parallel")]
                if should_parallelize_paths(self.num_paths) {
                    return Ok(ExecutionBackend::Parallel);
                }

                if simd_available {
                    Ok(ExecutionBackend::Simd)
                } else {
                    Ok(ExecutionBackend::Scalar)
                }
            }
        }
    }

    fn execution_thread_count(&self, backend: ExecutionBackend) -> usize {
        #[cfg(feature = "parallel")]
        if matches!(backend, ExecutionBackend::Parallel)
            || (matches!(backend, ExecutionBackend::Jit) && should_parallelize_jit(self.num_paths))
        {
            return rayon::current_num_threads();
        }
        #[cfg(not(feature = "parallel"))]
        let _ = backend;
        1
    }

    fn execution_vector_width(&self, backend: ExecutionBackend) -> usize {
        match backend {
            ExecutionBackend::Simd => simd_path_lanes(),
            ExecutionBackend::Parallel => {
                let lanes = simd_path_lanes();
                if simd_runtime_available() && should_simd_path_batch(self.num_paths, lanes) {
                    lanes
                } else {
                    1
                }
            }
            _ => 1,
        }
    }

    /// Price a DSL product under a multi-asset market.
    pub fn price_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
    ) -> Result<PricingResult, PricingError> {
        self.price_multi_asset_with_policy(product, market, ExecutionPolicy::Auto)
    }

    /// Price a DSL product with an explicit hardware execution policy.
    ///
    /// `Auto` preserves the normal adaptive behavior. A requested backend
    /// returns an error when its feature or runtime capability is unavailable.
    pub fn price_multi_asset_with_policy(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        execution_policy: ExecutionPolicy,
    ) -> Result<PricingResult, PricingError> {
        product.validate()?;
        market.validate()?;

        if market.assets.len() != product.num_underlyings {
            return Err(PricingError::InvalidInput(format!(
                "market has {} assets, but product declares {} underlyings",
                market.assets.len(),
                product.num_underlyings
            )));
        }
        for underlying in &product.underlyings {
            let market_type = market_asset_type(&market.assets[underlying.asset_index]);
            if underlying.underlying_type != market_type {
                return Err(PricingError::InvalidInput(format!(
                    "product underlying '{}' at asset index {} declares {}, but the market provides {} data",
                    underlying.name,
                    underlying.asset_index,
                    underlying_type_name(underlying.underlying_type),
                    underlying_type_name(market_type)
                )));
            }
        }

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
        if product.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "product maturity must be > 0".to_string(),
            ));
        }

        let execution_backend = self.resolve_execution_backend(execution_policy)?;

        let n_steps = self.num_steps;
        let dt = product.maturity / n_steps as f64;

        // Build Cholesky factor for correlation.
        let (chol, _) = cholesky_for_correlation(&market.correlation)?;

        let initial_spots = market.initial_spots();
        let interpreter_state = if matches!(
            execution_backend,
            ExecutionBackend::Scalar | ExecutionBackend::Simd | ExecutionBackend::Parallel
        ) {
            Some((
                build_execution_plan_for_validated_product(product, n_steps, market.rate)
                    .map_err(|e| PricingError::NumericalError(e.to_string()))?,
                product.max_local_slots(),
            ))
        } else {
            None
        };

        // Build per-asset steppers.
        let steppers: Vec<AssetStepper> = market
            .assets
            .iter()
            .map(|a| AssetStepper::from_asset(a, market.rate, dt))
            .collect();

        let chunk_stats = match execution_backend {
            ExecutionBackend::Scalar => {
                let (execution_plan, num_locals) = interpreter_state
                    .as_ref()
                    .expect("scalar backend prepares interpreter state");
                self.price_path_chunk_scalar(
                    self.num_paths,
                    self.seed,
                    product,
                    &initial_spots,
                    execution_plan,
                    &chol,
                    &steppers,
                    *num_locals,
                )?
            }
            ExecutionBackend::Simd => {
                let (execution_plan, num_locals) = interpreter_state
                    .as_ref()
                    .expect("SIMD backend prepares interpreter state");
                self.price_path_chunk(
                    self.num_paths,
                    self.seed,
                    product,
                    &initial_spots,
                    execution_plan,
                    &chol,
                    &steppers,
                    *num_locals,
                )?
            }
            ExecutionBackend::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    let (execution_plan, num_locals) = interpreter_state
                        .as_ref()
                        .expect("parallel backend prepares interpreter state");
                    self.price_path_chunks_parallel(
                        product,
                        &initial_spots,
                        execution_plan,
                        &chol,
                        &steppers,
                        *num_locals,
                    )?
                }
                #[cfg(not(feature = "parallel"))]
                unreachable!("parallel backend is rejected during policy resolution")
            }
            ExecutionBackend::Jit => {
                #[cfg(all(feature = "jit", not(target_family = "wasm")))]
                {
                    let evaluator = cached_jit_evaluator(product, self.num_steps, market.rate)?;
                    #[cfg(feature = "parallel")]
                    if should_parallelize_jit(self.num_paths) {
                        self.price_path_chunks_jit(&initial_spots, &evaluator, &chol, &steppers)?
                    } else {
                        self.price_path_chunk_jit(
                            self.num_paths,
                            self.seed,
                            &initial_spots,
                            &evaluator,
                            &chol,
                            &steppers,
                        )?
                    }
                    #[cfg(not(feature = "parallel"))]
                    self.price_path_chunk_jit(
                        self.num_paths,
                        self.seed,
                        &initial_spots,
                        &evaluator,
                        &chol,
                        &steppers,
                    )?
                }
                #[cfg(not(all(feature = "jit", not(target_family = "wasm"))))]
                unreachable!("JIT backend is rejected during policy resolution")
            }
            ExecutionBackend::Gpu => {
                unreachable!("GPU backend is rejected during policy resolution")
            }
        };

        let n = chunk_stats.num_paths as f64;
        let mean = chunk_stats.mean_pv;
        let variance = chunk_stats.sample_variance();
        let stderr = (variance / n).sqrt();
        if !mean.is_finite() || !stderr.is_finite() {
            return Err(PricingError::NumericalError(
                "DSL evaluation produced a non-finite price or standard error".to_string(),
            ));
        }

        let mut diagnostics = Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumPaths, n);
        diagnostics.insert_key(DiagKey::NumSteps, n_steps as f64);
        diagnostics.insert_key(
            DiagKey::ExecutionBackend,
            execution_backend.diagnostic_code(),
        );
        diagnostics.insert_key(
            DiagKey::NumThreads,
            self.execution_thread_count(execution_backend) as f64,
        );
        diagnostics.insert_key(
            DiagKey::VectorWidth,
            self.execution_vector_width(execution_backend) as f64,
        );

        Ok(PricingResult {
            price: mean,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }

    fn price_path_chunk(
        &self,
        num_paths: usize,
        seed: u64,
        product: &CompiledProduct,
        initial_spots: &[f64],
        plan: &ProductExecutionPlan,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
        num_locals: usize,
    ) -> Result<ChunkStats, PricingError> {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if should_simd_path_batch(num_paths, SIMD_X86_PATH_LANES)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: Guarded by runtime AVX2+FMA detection.
            return unsafe {
                self.price_path_chunk_simd_x86(
                    num_paths,
                    seed,
                    product,
                    initial_spots,
                    plan,
                    chol,
                    steppers,
                    num_locals,
                )
            };
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if should_simd_path_batch(num_paths, SIMD_NEON_PATH_LANES) {
            return unsafe {
                self.price_path_chunk_simd_neon(
                    num_paths,
                    seed,
                    product,
                    initial_spots,
                    plan,
                    chol,
                    steppers,
                    num_locals,
                )
            };
        }

        self.price_path_chunk_scalar(
            num_paths,
            seed,
            product,
            initial_spots,
            plan,
            chol,
            steppers,
            num_locals,
        )
    }

    fn price_path_chunk_scalar(
        &self,
        num_paths: usize,
        seed: u64,
        product: &CompiledProduct,
        initial_spots: &[f64],
        plan: &ProductExecutionPlan,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
        num_locals: usize,
    ) -> Result<ChunkStats, PricingError> {
        if plan.snapshot_count() == 0 {
            return Ok(ChunkStats::zeros(num_paths));
        }

        let n_assets = initial_spots.len();
        let n_steps = self.num_steps;
        let mut rng = FastRng::from_seed(self.rng_kind, seed);
        let mut stats = ChunkStats::default();
        let mut corr_normals = vec![0.0; n_assets];
        let mut indep_normals = vec![0.0; n_assets];
        let mut current_spots = initial_spots.to_vec();
        let mut next_spots = vec![0.0; n_assets];
        let mut observation_spots = vec![vec![0.0; n_assets]; plan.snapshot_count()];
        let mut locals = vec![0.0_f64; num_locals];
        let mut state = vec![0.0_f64; product.state_vars.len()];
        let mut stack = vec![0.0_f64; plan.max_stack()];

        for _ in 0..num_paths {
            current_spots.copy_from_slice(initial_spots);
            if let Some(snapshot_index) = plan.snapshot_index_for_step(0) {
                observation_spots[snapshot_index].copy_from_slice(initial_spots);
            }

            for step in 0..n_steps {
                sample_correlated_normals_cholesky_with_scratch(
                    chol,
                    &mut rng,
                    &mut indep_normals,
                    &mut corr_normals,
                )?;

                for asset in 0..n_assets {
                    let prev = current_spots[asset];
                    let z = corr_normals[asset];
                    next_spots[asset] = steppers[asset].step(prev, z);
                }

                if let Some(snapshot_index) = plan.snapshot_index_for_step(step + 1) {
                    observation_spots[snapshot_index].copy_from_slice(&next_spots);
                }

                std::mem::swap(&mut current_spots, &mut next_spots);
            }

            let pv = evaluate_product_with_plan_in_place(
                product,
                plan,
                &observation_spots,
                initial_spots,
                &mut locals,
                &mut state,
                &mut stack,
            )
            .map_err(|e| PricingError::NumericalError(e.to_string()))?;

            stats.record(pv);
        }

        debug_assert_eq!(stats.num_paths, num_paths);
        Ok(stats)
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    fn price_path_chunk_jit(
        &self,
        num_paths: usize,
        seed: u64,
        initial_spots: &[f64],
        evaluator: &JitProductEvaluator,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
    ) -> Result<ChunkStats, PricingError> {
        if evaluator.snapshot_count() == 0 {
            return Ok(ChunkStats::zeros(num_paths));
        }

        let n_assets = initial_spots.len();
        let mut rng = FastRng::from_seed(self.rng_kind, seed);
        let mut stats = ChunkStats::default();
        let mut corr_normals = vec![0.0; n_assets];
        let mut indep_normals = vec![0.0; n_assets];
        let mut current_spots = initial_spots.to_vec();
        let mut next_spots = vec![0.0; n_assets];
        let mut observation_spots = vec![vec![0.0; n_assets]; evaluator.snapshot_count()];
        let mut scratch: JitEvaluationScratch = evaluator.new_scratch();

        for _ in 0..num_paths {
            current_spots.copy_from_slice(initial_spots);
            if let Some(snapshot) = evaluator.snapshot_index_for_step(0) {
                observation_spots[snapshot].copy_from_slice(initial_spots);
            }

            for step in 0..self.num_steps {
                sample_correlated_normals_cholesky_with_scratch(
                    chol,
                    &mut rng,
                    &mut indep_normals,
                    &mut corr_normals,
                )?;
                for asset in 0..n_assets {
                    next_spots[asset] =
                        steppers[asset].step(current_spots[asset], corr_normals[asset]);
                }
                if let Some(snapshot) = evaluator.snapshot_index_for_step(step + 1) {
                    observation_spots[snapshot].copy_from_slice(&next_spots);
                }
                std::mem::swap(&mut current_spots, &mut next_spots);
            }

            let pv = evaluator
                .evaluate_captured_observations(&observation_spots, initial_spots, &mut scratch)
                .map_err(|error| PricingError::NumericalError(error.to_string()))?;
            stats.record(pv);
        }

        debug_assert_eq!(stats.num_paths, num_paths);
        Ok(stats)
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn price_path_chunk_simd_x86(
        &self,
        num_paths: usize,
        seed: u64,
        product: &CompiledProduct,
        initial_spots: &[f64],
        plan: &ProductExecutionPlan,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
        num_locals: usize,
    ) -> Result<ChunkStats, PricingError> {
        const LANES: usize = SIMD_X86_PATH_LANES;

        if plan.snapshot_count() == 0 {
            return Ok(ChunkStats::zeros(num_paths));
        }

        let n_assets = initial_spots.len();
        let n_steps = self.num_steps;
        let n_state = product.state_vars.len();
        let n_snapshots = plan.snapshot_count();
        let mut rng = FastRng::from_seed(self.rng_kind, seed);
        let mut stats = ChunkStats::default();
        let mut corr_normals = vec![vec![0.0; n_assets]; LANES];
        let mut indep_normals = vec![vec![0.0; n_assets]; LANES];
        let mut current_spots = vec![[0.0; LANES]; n_assets];
        let mut next_spots = vec![[0.0; LANES]; n_assets];
        let mut observation_spots = vec![vec![[0.0; LANES]; n_assets]; n_snapshots];
        let mut locals = vec![[0.0_f64; LANES]; num_locals];
        let mut state = vec![[0.0_f64; LANES]; n_state];
        let mut stack = vec![[0.0_f64; LANES]; plan.max_stack()];

        let simd_paths = num_paths / LANES * LANES;
        let mut processed = 0usize;
        while processed < simd_paths {
            for (asset, &spot0) in initial_spots.iter().enumerate() {
                current_spots[asset].fill(spot0);
            }
            if let Some(snapshot_index) = plan.snapshot_index_for_step(0) {
                for (asset, &spot0) in initial_spots.iter().enumerate() {
                    observation_spots[snapshot_index][asset].fill(spot0);
                }
            }

            for step in 0..n_steps {
                for lane in 0..LANES {
                    sample_correlated_normals_cholesky_with_scratch(
                        chol,
                        &mut rng,
                        &mut indep_normals[lane],
                        &mut corr_normals[lane],
                    )?;
                }

                for asset in 0..n_assets {
                    let z = [
                        corr_normals[0][asset],
                        corr_normals[1][asset],
                        corr_normals[2][asset],
                        corr_normals[3][asset],
                    ];
                    unsafe {
                        steppers[asset].step_batch_x86(
                            &current_spots[asset],
                            &z,
                            &mut next_spots[asset],
                        )
                    };
                }

                if let Some(snapshot_index) = plan.snapshot_index_for_step(step + 1) {
                    for (asset, asset_next_spots) in next_spots.iter().enumerate().take(n_assets) {
                        observation_spots[snapshot_index][asset] = *asset_next_spots;
                    }
                }

                std::mem::swap(&mut current_spots, &mut next_spots);
            }

            let pv_batch = evaluate_product_with_plan_batch_x86(
                product,
                plan,
                &observation_spots,
                initial_spots,
                &mut locals,
                &mut state,
                &mut stack,
            )
            .map_err(|e| PricingError::NumericalError(e.to_string()))?;

            for pv in pv_batch {
                stats.record(pv);
            }

            processed += LANES;
        }

        if simd_paths < num_paths {
            let tail = self.price_path_chunk_scalar(
                num_paths - simd_paths,
                seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
                product,
                initial_spots,
                plan,
                chol,
                steppers,
                num_locals,
            )?;
            stats.merge(tail);
        }

        debug_assert_eq!(stats.num_paths, num_paths);
        Ok(stats)
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn price_path_chunk_simd_neon(
        &self,
        num_paths: usize,
        seed: u64,
        product: &CompiledProduct,
        initial_spots: &[f64],
        plan: &ProductExecutionPlan,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
        num_locals: usize,
    ) -> Result<ChunkStats, PricingError> {
        const LANES: usize = SIMD_NEON_PATH_LANES;

        if plan.snapshot_count() == 0 {
            return Ok(ChunkStats::zeros(num_paths));
        }

        let n_assets = initial_spots.len();
        let n_steps = self.num_steps;
        let n_state = product.state_vars.len();
        let n_snapshots = plan.snapshot_count();
        let mut rng = FastRng::from_seed(self.rng_kind, seed);
        let mut stats = ChunkStats::default();
        let mut corr_normals = vec![vec![0.0; n_assets]; LANES];
        let mut indep_normals = vec![vec![0.0; n_assets]; LANES];
        let mut current_spots = vec![[0.0; LANES]; n_assets];
        let mut next_spots = vec![[0.0; LANES]; n_assets];
        let mut observation_spots = vec![vec![[0.0; LANES]; n_assets]; n_snapshots];
        let mut locals = vec![[0.0_f64; LANES]; num_locals];
        let mut state = vec![[0.0_f64; LANES]; n_state];
        let mut stack = vec![[0.0_f64; LANES]; plan.max_stack()];

        let simd_paths = num_paths / LANES * LANES;
        let mut processed = 0usize;
        while processed < simd_paths {
            for (asset, &spot0) in initial_spots.iter().enumerate() {
                current_spots[asset].fill(spot0);
            }
            if let Some(snapshot_index) = plan.snapshot_index_for_step(0) {
                for (asset, &spot0) in initial_spots.iter().enumerate() {
                    observation_spots[snapshot_index][asset].fill(spot0);
                }
            }

            for step in 0..n_steps {
                for lane in 0..LANES {
                    sample_correlated_normals_cholesky_with_scratch(
                        chol,
                        &mut rng,
                        &mut indep_normals[lane],
                        &mut corr_normals[lane],
                    )?;
                }

                for asset in 0..n_assets {
                    let z = [corr_normals[0][asset], corr_normals[1][asset]];
                    unsafe {
                        steppers[asset].step_batch_neon(
                            &current_spots[asset],
                            &z,
                            &mut next_spots[asset],
                        )
                    };
                }

                if let Some(snapshot_index) = plan.snapshot_index_for_step(step + 1) {
                    for (asset, asset_next_spots) in next_spots.iter().enumerate().take(n_assets) {
                        observation_spots[snapshot_index][asset] = *asset_next_spots;
                    }
                }

                std::mem::swap(&mut current_spots, &mut next_spots);
            }

            let pv_batch = evaluate_product_with_plan_batch_neon(
                product,
                plan,
                &observation_spots,
                initial_spots,
                &mut locals,
                &mut state,
                &mut stack,
            )
            .map_err(|e| PricingError::NumericalError(e.to_string()))?;

            for pv in pv_batch {
                stats.record(pv);
            }

            processed += LANES;
        }

        if simd_paths < num_paths {
            let tail = self.price_path_chunk_scalar(
                num_paths - simd_paths,
                seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
                product,
                initial_spots,
                plan,
                chol,
                steppers,
                num_locals,
            )?;
            stats.merge(tail);
        }

        debug_assert_eq!(stats.num_paths, num_paths);
        Ok(stats)
    }

    #[cfg(feature = "parallel")]
    fn price_path_chunks_parallel(
        &self,
        product: &CompiledProduct,
        initial_spots: &[f64],
        plan: &ProductExecutionPlan,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
        num_locals: usize,
    ) -> Result<ChunkStats, PricingError> {
        let chunk_sizes = split_path_chunks(self.num_paths);
        let chunk_results: Vec<Result<ChunkStats, PricingError>> = chunk_sizes
            .par_iter()
            .enumerate()
            .map(|(i, &chunk_paths)| {
                let chunk_seed = self
                    .seed
                    .wrapping_add((i as u64).wrapping_mul(6_364_136_223_846_793_005));
                self.price_path_chunk(
                    chunk_paths,
                    chunk_seed,
                    product,
                    initial_spots,
                    plan,
                    chol,
                    steppers,
                    num_locals,
                )
            })
            .collect();

        let mut total = ChunkStats::default();
        for result in chunk_results {
            let chunk = result?;
            total.merge(chunk);
        }
        Ok(total)
    }

    #[cfg(all(feature = "jit", feature = "parallel", not(target_family = "wasm")))]
    fn price_path_chunks_jit(
        &self,
        initial_spots: &[f64],
        evaluator: &JitProductEvaluator,
        chol: &[Vec<f64>],
        steppers: &[AssetStepper],
    ) -> Result<ChunkStats, PricingError> {
        let chunk_sizes = split_path_chunks(self.num_paths);
        let chunk_results: Vec<Result<ChunkStats, PricingError>> = chunk_sizes
            .par_iter()
            .enumerate()
            .map(|(i, &chunk_paths)| {
                let chunk_seed = self
                    .seed
                    .wrapping_add((i as u64).wrapping_mul(6_364_136_223_846_793_005));
                self.price_path_chunk_jit(
                    chunk_paths,
                    chunk_seed,
                    initial_spots,
                    evaluator,
                    chol,
                    steppers,
                )
            })
            .collect();

        let mut total = ChunkStats::default();
        for result in chunk_results {
            let chunk = result?;
            total.merge(chunk);
        }
        Ok(total)
    }

    /// Compute Greeks via bump-and-reprice.
    pub fn greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
    ) -> Result<Greeks, PricingError> {
        if asset_index >= market.assets.len() {
            return Err(PricingError::InvalidInput(format!(
                "asset_index {asset_index} out of range (have {} assets)",
                market.assets.len()
            )));
        }

        let base = self.price_multi_asset(product, market)?;

        // Delta: bump spot by 1%
        let spot_bump = 0.01 * market.assets[asset_index].initial_value();
        let mut market_up = market.clone();
        market_up.assets[asset_index] = market.assets[asset_index].with_spot_bump(spot_bump);
        let price_up = self.price_multi_asset(product, &market_up)?.price;

        let mut market_down = market.clone();
        market_down.assets[asset_index] = market.assets[asset_index].with_spot_bump(-spot_bump);
        let price_down = self.price_multi_asset(product, &market_down)?.price;

        let delta = (price_up - price_down) / (2.0 * spot_bump);
        let gamma = (price_up - 2.0 * base.price + price_down) / (spot_bump * spot_bump);

        // Vega: bump vol by 1%
        let vol_bump = 0.01;
        let mut market_vega = market.clone();
        market_vega.assets[asset_index] = market.assets[asset_index].with_vol_bump(vol_bump);
        let price_vega = self.price_multi_asset(product, &market_vega)?.price;
        let vega = (price_vega - base.price) / vol_bump;

        // Rho: bump rate by 1bp
        let rate_bump = 0.0001;
        let mut market_rho = market.clone();
        market_rho.rate += rate_bump;
        let price_rho = self.price_multi_asset(product, &market_rho)?.price;
        let rho = (price_rho - base.price) / rate_bump;

        let theta = 0.0;

        Ok(Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        })
    }

    /// Compute per-asset greeks including higher-order sensitivities (vanna, volga).
    pub fn extended_greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
        base_price: f64,
    ) -> Result<ExtendedGreeks, PricingError> {
        if asset_index >= market.assets.len() {
            return Err(PricingError::InvalidInput(format!(
                "asset_index {asset_index} out of range (have {} assets)",
                market.assets.len()
            )));
        }

        let spot_bump = 0.01 * market.assets[asset_index].initial_value();
        let vol_bump = 0.01;
        let asset_data = &market.assets[asset_index];

        // Spot up / down
        let mut market_spot_up = market.clone();
        market_spot_up.assets[asset_index] = asset_data.with_spot_bump(spot_bump);
        let price_spot_up = self.price_multi_asset(product, &market_spot_up)?.price;

        let mut market_spot_down = market.clone();
        market_spot_down.assets[asset_index] = asset_data.with_spot_bump(-spot_bump);
        let price_spot_down = self.price_multi_asset(product, &market_spot_down)?.price;

        let delta = (price_spot_up - price_spot_down) / (2.0 * spot_bump);
        let gamma = (price_spot_up - 2.0 * base_price + price_spot_down) / (spot_bump * spot_bump);

        // Vol up / down
        let mut market_vol_up = market.clone();
        market_vol_up.assets[asset_index] = asset_data.with_vol_bump(vol_bump);
        let price_vol_up = self.price_multi_asset(product, &market_vol_up)?.price;

        let mut market_vol_down = market.clone();
        market_vol_down.assets[asset_index] = asset_data.with_vol_bump(-vol_bump);
        let price_vol_down = self.price_multi_asset(product, &market_vol_down)?.price;

        let vega = (price_vol_up - price_vol_down) / (2.0 * vol_bump);
        let volga = (price_vol_up - 2.0 * base_price + price_vol_down) / (vol_bump * vol_bump);

        // Vanna: cross derivative d²V/(dS dσ)
        // Bump spot+vol jointly for the four corners
        let mut m_up_up = market.clone();
        m_up_up.assets[asset_index] = asset_data.with_spot_bump(spot_bump).with_vol_bump(vol_bump);
        let p_up_up = self.price_multi_asset(product, &m_up_up)?.price;

        let mut m_up_down = market.clone();
        m_up_down.assets[asset_index] = asset_data
            .with_spot_bump(spot_bump)
            .with_vol_bump(-vol_bump);
        let p_up_down = self.price_multi_asset(product, &m_up_down)?.price;

        let mut m_down_up = market.clone();
        m_down_up.assets[asset_index] = asset_data
            .with_spot_bump(-spot_bump)
            .with_vol_bump(vol_bump);
        let p_down_up = self.price_multi_asset(product, &m_down_up)?.price;

        let mut m_down_down = market.clone();
        m_down_down.assets[asset_index] = asset_data
            .with_spot_bump(-spot_bump)
            .with_vol_bump(-vol_bump);
        let p_down_down = self.price_multi_asset(product, &m_down_down)?.price;

        let vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4.0 * spot_bump * vol_bump);

        // Rho
        let rate_bump = 0.0001;
        let mut market_rho = market.clone();
        market_rho.rate += rate_bump;
        let price_rho = self.price_multi_asset(product, &market_rho)?.price;
        let rho = (price_rho - base_price) / rate_bump;

        Ok(ExtendedGreeks {
            delta,
            gamma,
            vega,
            theta: 0.0,
            rho,
            vanna,
            volga,
        })
    }

    /// Compute cross-asset sensitivities between a pair of underlyings.
    pub fn cross_greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_i: usize,
        asset_j: usize,
        _base_price: f64,
    ) -> Result<CrossGreeks, PricingError> {
        let n = market.assets.len();
        if asset_i >= n || asset_j >= n {
            return Err(PricingError::InvalidInput(format!(
                "asset indices ({asset_i}, {asset_j}) out of range (have {n} assets)"
            )));
        }

        // Cross-gamma: d²V/(dSi dSj)
        let bump_i = 0.01 * market.assets[asset_i].initial_value();
        let bump_j = 0.01 * market.assets[asset_j].initial_value();

        let mut m_pp = market.clone();
        m_pp.assets[asset_i] = market.assets[asset_i].with_spot_bump(bump_i);
        m_pp.assets[asset_j] = market.assets[asset_j].with_spot_bump(bump_j);
        let p_pp = self.price_multi_asset(product, &m_pp)?.price;

        let mut m_pm = market.clone();
        m_pm.assets[asset_i] = market.assets[asset_i].with_spot_bump(bump_i);
        m_pm.assets[asset_j] = market.assets[asset_j].with_spot_bump(-bump_j);
        let p_pm = self.price_multi_asset(product, &m_pm)?.price;

        let mut m_mp = market.clone();
        m_mp.assets[asset_i] = market.assets[asset_i].with_spot_bump(-bump_i);
        m_mp.assets[asset_j] = market.assets[asset_j].with_spot_bump(bump_j);
        let p_mp = self.price_multi_asset(product, &m_mp)?.price;

        let mut m_mm = market.clone();
        m_mm.assets[asset_i] = market.assets[asset_i].with_spot_bump(-bump_i);
        m_mm.assets[asset_j] = market.assets[asset_j].with_spot_bump(-bump_j);
        let p_mm = self.price_multi_asset(product, &m_mm)?.price;

        let cross_gamma = (p_pp - p_pm - p_mp + p_mm) / (4.0 * bump_i * bump_j);

        // Correlation sensitivity: dV/dρij
        let corr_bump = 0.01;
        let mut m_corr_up = market.clone();
        let rho_val = m_corr_up.correlation[asset_i][asset_j];
        let rho_up = (rho_val + corr_bump).min(1.0);
        m_corr_up.correlation[asset_i][asset_j] = rho_up;
        m_corr_up.correlation[asset_j][asset_i] = rho_up;
        let p_corr_up = self.price_multi_asset(product, &m_corr_up)?.price;

        let mut m_corr_down = market.clone();
        let rho_down = (rho_val - corr_bump).max(-1.0);
        m_corr_down.correlation[asset_i][asset_j] = rho_down;
        m_corr_down.correlation[asset_j][asset_i] = rho_down;
        let p_corr_down = self.price_multi_asset(product, &m_corr_down)?.price;

        let corr_sens = (p_corr_up - p_corr_down) / (rho_up - rho_down);

        Ok(CrossGreeks {
            cross_gamma,
            corr_sens,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ChunkStats {
    mean_pv: f64,
    m2_pv: f64,
    num_paths: usize,
}

impl ChunkStats {
    #[inline]
    fn zeros(num_paths: usize) -> Self {
        Self {
            mean_pv: 0.0,
            m2_pv: 0.0,
            num_paths,
        }
    }

    #[inline(always)]
    fn record(&mut self, pv: f64) {
        self.num_paths += 1;
        let delta = pv - self.mean_pv;
        self.mean_pv += delta / self.num_paths as f64;
        let delta2 = pv - self.mean_pv;
        self.m2_pv += delta * delta2;
    }

    #[inline]
    #[cfg(any(
        test,
        feature = "parallel",
        all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    fn merge(&mut self, other: Self) {
        if other.num_paths == 0 {
            return;
        }
        if self.num_paths == 0 {
            *self = other;
            return;
        }

        let lhs_count = self.num_paths as f64;
        let rhs_count = other.num_paths as f64;
        let total_count = lhs_count + rhs_count;
        let delta = other.mean_pv - self.mean_pv;
        self.mean_pv += delta * rhs_count / total_count;
        self.m2_pv += other.m2_pv + delta * delta * lhs_count * rhs_count / total_count;
        self.num_paths += other.num_paths;
    }

    #[inline]
    fn sample_variance(self) -> f64 {
        if self.num_paths <= 1 {
            return 0.0;
        }
        let variance = self.m2_pv / (self.num_paths as f64 - 1.0);
        if variance < 0.0 { 0.0 } else { variance }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
const SIMD_X86_PATH_LANES: usize = 4;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
const SIMD_NEON_PATH_LANES: usize = 2;

#[inline]
fn should_simd_path_batch(num_paths: usize, lanes: usize) -> bool {
    num_paths >= lanes * 4
}

#[inline]
fn simd_path_lanes() -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        return SIMD_X86_PATH_LANES;
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return SIMD_NEON_PATH_LANES;
    }
    #[allow(unreachable_code)]
    1
}

#[inline]
fn simd_runtime_available() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return true;
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(feature = "parallel")]
const MIN_PARALLEL_PATHS: usize = 8_192;
#[cfg(feature = "parallel")]
const PATH_CHUNK_SIZE: usize = 4_096;

#[cfg(feature = "parallel")]
#[inline]
fn should_parallelize_paths(num_paths: usize) -> bool {
    num_paths >= MIN_PARALLEL_PATHS && rayon::current_num_threads() > 1
}

#[cfg(feature = "parallel")]
#[inline]
fn should_parallelize_jit(num_paths: usize) -> bool {
    num_paths >= MIN_PARALLEL_PATHS
}

#[cfg(feature = "parallel")]
fn split_path_chunks(num_paths: usize) -> Vec<usize> {
    let mut remaining = num_paths;
    let mut chunks = Vec::new();
    while remaining > 0 {
        let chunk = remaining.min(PATH_CHUNK_SIZE);
        chunks.push(chunk);
        remaining -= chunk;
    }
    chunks
}

/// Extended per-asset greeks including higher-order sensitivities.
#[derive(Debug, Clone, Copy)]
pub struct ExtendedGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

/// Cross-asset sensitivities between a pair of underlyings.
#[derive(Debug, Clone, Copy)]
pub struct CrossGreeks {
    /// d²V/(dSi dSj)
    pub cross_gamma: f64,
    /// dV/dρij
    pub corr_sens: f64,
}

/// Implement `PricingEngine<DslProduct>` for single-asset convenience.
/// Uses a standard `Market` and wraps it into `MultiAssetMarket`.
impl PricingEngine<DslProduct> for DslMonteCarloEngine {
    fn price(
        &self,
        instrument: &DslProduct,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        let vol = market.checked_vol_for(market.spot, instrument.product.maturity)?;
        let multi_market =
            MultiAssetMarket::single(market.spot, vol, market.rate, market.dividend_yield);
        self.price_multi_asset(&instrument.product, &multi_market)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::ir::*;

    /// Build a simple forward product: pays S(T) at maturity.
    /// PV should be approximately S(0) * exp(-q*T) for zero-coupon.
    fn make_forward_product() -> CompiledProduct {
        CompiledProduct {
            name: "Forward".to_string(),
            notional: 1.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "SPX".to_string(),
                asset_index: 0,
                underlying_type: Default::default(),
            }],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::Call {
                        func: BuiltinFn::Price,
                        args: vec![Expr::Literal(Value::F64(0.0))],
                    },
                }],
            }],
        }
    }

    fn make_constant_product(amount: f64) -> CompiledProduct {
        CompiledProduct {
            name: "Constant".to_string(),
            notional: 1.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "SPX".to_string(),
                asset_index: 0,
                underlying_type: Default::default(),
            }],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::Literal(Value::F64(amount)),
                }],
            }],
        }
    }

    #[test]
    fn price_rejects_product_market_dimension_mismatch() {
        let mut product = make_forward_product();
        product.num_underlyings = 2;
        product.underlyings.push(UnderlyingDef {
            name: "SX5E".to_string(),
            asset_index: 1,
            underlying_type: Default::default(),
        });
        let one_asset_market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(16, 4, 42);

        let error = engine
            .price_multi_asset_with_policy(&product, &one_asset_market, ExecutionPolicy::Scalar)
            .unwrap_err();
        assert!(
            error.to_string().contains("product declares 2"),
            "unexpected error: {error}"
        );
    }

    fn single_market_for_type(underlying_type: UnderlyingType) -> MultiAssetMarket {
        let asset = match underlying_type {
            UnderlyingType::Equity => AssetMarketData::Equity {
                spot: 100.0,
                vol: 0.2,
                dividend_yield: 0.01,
            },
            UnderlyingType::Fx => AssetMarketData::Fx {
                spot: 1.1,
                vol: 0.15,
                domestic_rate: 0.03,
                foreign_rate: 0.02,
            },
            UnderlyingType::Commodity => AssetMarketData::Commodity {
                spot: 75.0,
                vol: 0.3,
                convenience_yield: 0.01,
                kappa: 1.0,
                mu: 75.0_f64.ln(),
            },
            UnderlyingType::Rate => AssetMarketData::Rate {
                initial_rate: 0.03,
                vol: 0.01,
                mean_reversion: 0.1,
                long_run_mean: 0.04,
            },
        };
        MultiAssetMarket {
            assets: vec![asset],
            correlation: vec![vec![1.0]],
            rate: 0.03,
        }
    }

    #[test]
    fn price_rejects_underlying_market_type_mismatches_for_all_variants() {
        let cases = [
            (UnderlyingType::Equity, UnderlyingType::Fx),
            (UnderlyingType::Fx, UnderlyingType::Commodity),
            (UnderlyingType::Commodity, UnderlyingType::Rate),
            (UnderlyingType::Rate, UnderlyingType::Equity),
        ];
        let engine = DslMonteCarloEngine::new(16, 4, 42);

        for (declared_type, market_type) in cases {
            let mut product = make_forward_product();
            product.underlyings[0].name = "TEST".to_string();
            product.underlyings[0].underlying_type = declared_type;
            let market = single_market_for_type(market_type);

            let error = engine
                .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Scalar)
                .unwrap_err();
            let message = error.to_string();
            assert!(message.contains("'TEST'"), "unexpected error: {message}");
            assert!(
                message.contains("asset index 0"),
                "unexpected error: {message}"
            );
            assert!(
                message.contains(underlying_type_name(declared_type)),
                "unexpected error: {message}"
            );
            assert!(
                message.contains(underlying_type_name(market_type)),
                "unexpected error: {message}"
            );
        }
    }

    #[test]
    fn fixed_cashflow_without_underlying_allows_any_market_variant() {
        let mut product = make_constant_product(10.0);
        product.underlyings.clear();
        let market = single_market_for_type(UnderlyingType::Rate);
        let engine = DslMonteCarloEngine::new(16, 4, 42);

        let result = engine
            .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Scalar)
            .unwrap();
        let expected = 10.0 * (-market.rate).exp();
        assert!((result.price - expected).abs() < 1.0e-12);
    }

    #[test]
    fn pricing_backends_reject_non_finite_results() {
        let mut product = make_constant_product(1.0);
        product.schedules[0].body = vec![Statement::Redeem {
            amount: Expr::Call {
                func: BuiltinFn::Exp,
                args: vec![Expr::Literal(Value::F64(1_000.0))],
            },
        }];
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(16, 4, 42);

        let mut policies = vec![ExecutionPolicy::Scalar];
        if engine
            .resolve_execution_backend(ExecutionPolicy::Simd)
            .is_ok()
        {
            policies.push(ExecutionPolicy::Simd);
        }
        #[cfg(feature = "parallel")]
        policies.push(ExecutionPolicy::Parallel);
        #[cfg(all(feature = "jit", not(target_family = "wasm"), target_arch = "x86_64"))]
        policies.push(ExecutionPolicy::Jit);

        for policy in policies {
            let error = engine
                .price_multi_asset_with_policy(&product, &market, policy)
                .unwrap_err();
            assert!(
                error.to_string().contains("non-finite"),
                "{policy:?} returned unexpected error: {error}"
            );
        }
    }

    #[test]
    fn forward_product_prices_near_forward() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(100_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Forward price = S(0) * exp((r-q)*T) = 100 * exp(0.03) ≈ 103.045
        // Discounted = 100 * exp((r-q)*T) * exp(-r*T) = 100 * exp(-q*T) ≈ 98.02
        let expected = 100.0 * (-0.02f64).exp();
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err < 0.02,
            "forward price error: got {}, expected {expected}, rel_err {rel_err}",
            result.price
        );
    }

    #[test]
    fn multi_asset_correlation_produces_correlated_paths() {
        // Simple 2-asset product: pays max(S1(T)/S1(0), S2(T)/S2(0)) * notional
        let product = CompiledProduct {
            name: "BestOf".to_string(),
            notional: 100.0,
            maturity: 1.0,
            num_underlyings: 2,
            underlyings: vec![
                UnderlyingDef {
                    name: "A".to_string(),
                    asset_index: 0,
                    underlying_type: Default::default(),
                },
                UnderlyingDef {
                    name: "B".to_string(),
                    asset_index: 1,
                    underlying_type: Default::default(),
                },
            ],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::BinOp {
                        op: BinOp::Mul,
                        lhs: Box::new(Expr::Notional),
                        rhs: Box::new(Expr::Call {
                            func: BuiltinFn::BestOf,
                            args: vec![Expr::Call {
                                func: BuiltinFn::Performances,
                                args: vec![],
                            }],
                        }),
                    },
                }],
            }],
        };

        let market = MultiAssetMarket {
            assets: vec![
                crate::dsl::market::AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.20,
                    dividend_yield: 0.0,
                },
                crate::dsl::market::AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.20,
                    dividend_yield: 0.0,
                },
            ],
            correlation: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            rate: 0.05,
        };

        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Best-of should be > 100 (it's a convex payoff on performance)
        assert!(
            result.price > 90.0,
            "best-of price {} should be > 90",
            result.price
        );
        assert!(result.stderr.unwrap() < 2.0, "stderr should be reasonable");
    }

    #[test]
    fn single_asset_via_market_trait() {
        let product = make_forward_product();
        let dsl_product = DslProduct::new(product);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price(&dsl_product, &market).unwrap();

        let expected = 100.0 * (-0.02f64).exp();
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err < 0.02,
            "single-asset forward price error: got {}, expected {expected}",
            result.price
        );
    }

    #[test]
    fn greeks_produce_sensible_delta() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
        let engine = DslMonteCarloEngine::new(50_000, 252, 42);

        let greeks = engine.greeks_multi_asset(&product, &market, 0).unwrap();

        // Forward delta ≈ exp(-r*T) ≈ 0.951
        let expected_delta = (-0.05f64).exp();
        assert!(
            (greeks.delta - expected_delta).abs() < 0.05,
            "delta {} should be near {expected_delta}",
            greeks.delta
        );
    }

    #[test]
    fn schwartz_stepper_mean_reverts() {
        // Schwartz one-factor: commodity price should mean-revert toward exp(mu).
        let stepper = AssetStepper::SchwartzOneF {
            long_run_log_mean: (100.0_f64).ln(),       // mu = ln(100)
            exp_neg_kappa_dt: (-2.0 * 0.01_f64).exp(), // kappa=2, dt=0.01
            vol_step: 0.30 * (0.01_f64).sqrt(),
        };
        // Starting far below mean: 50. With z=0 the step should pull toward 100.
        let next = stepper.step(50.0, 0.0);
        assert!(
            next > 50.0 && next < 100.0,
            "Schwartz step from 50 with z=0 should move toward 100, got {next}"
        );
        // Starting far above mean: 200. Should pull down.
        let next_high = stepper.step(200.0, 0.0);
        assert!(
            next_high < 200.0 && next_high > 100.0,
            "Schwartz step from 200 with z=0 should move toward 100, got {next_high}"
        );
    }

    #[test]
    fn vasicek_stepper_mean_reverts() {
        // Vasicek: rate should mean-revert toward long_run_mean.
        let stepper = AssetStepper::Vasicek {
            long_run_mean: 0.05,
            exp_neg_a_dt: (-0.01_f64).exp(), // a=1, dt=0.01
            vol_step: 0.01 * (0.01_f64).sqrt(),
        };
        // Starting above mean: r=0.10. With z=0 should pull down.
        let next = stepper.step(0.10, 0.0);
        assert!(
            next < 0.10 && next > 0.05,
            "Vasicek step from 0.10 with z=0 should move toward 0.05, got {next}"
        );
        // Starting below mean: r=0.01.
        let next_low = stepper.step(0.01, 0.0);
        assert!(
            next_low > 0.01 && next_low < 0.05,
            "Vasicek step from 0.01 with z=0 should move toward 0.05, got {next_low}"
        );
    }

    #[test]
    fn fx_stepper_uses_garman_kohlhagen() {
        // FX stepper should use domestic/foreign rates, not the market rate.
        let asset = AssetMarketData::Fx {
            spot: 1.10,
            vol: 0.10,
            domestic_rate: 0.05,
            foreign_rate: 0.02,
        };
        let dt = 1.0 / 252.0;
        let stepper = AssetStepper::from_asset(&asset, 0.99, dt); // market.rate=0.99 should be ignored
        match stepper {
            AssetStepper::Gbm { drift, .. } => {
                // Expected drift = (r_d - r_f - 0.5 σ²) dt = (0.05 - 0.02 - 0.005) * dt
                let expected_drift = (0.05 - 0.02 - 0.5 * 0.10 * 0.10) * dt;
                assert!(
                    (drift - expected_drift).abs() < 1e-12,
                    "FX drift {drift} != expected {expected_drift}"
                );
            }
            _ => panic!("FX asset should produce Gbm stepper"),
        }
    }

    #[test]
    fn commodity_forward_product_mean_reverts() {
        // Build a forward product on a commodity, price it, and verify
        // the price reflects mean-reversion toward exp(mu).
        let product = CompiledProduct {
            name: "CommodityForward".to_string(),
            notional: 1.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "WTI".to_string(),
                asset_index: 0,
                underlying_type: crate::dsl::ir::UnderlyingType::Commodity,
            }],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::Call {
                        func: BuiltinFn::Price,
                        args: vec![Expr::Literal(Value::F64(0.0))],
                    },
                }],
            }],
        };

        let market = MultiAssetMarket {
            assets: vec![AssetMarketData::Commodity {
                spot: 80.0,
                vol: 0.30,
                convenience_yield: 0.0,
                kappa: 1.0,
                mu: (100.0_f64).ln(), // mean-reverts toward 100
            }],
            correlation: vec![vec![1.0]],
            rate: 0.05,
        };

        let engine = DslMonteCarloEngine::new(100_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Starting at 80, mean-reverting toward 100 over 1 year with kappa=1.
        // Expected E[S(T)] ≈ exp(mu + (ln(S0) - mu)*exp(-kappa*T))
        // = exp(ln(100) + (ln(80) - ln(100))*exp(-1)) ≈ exp(4.605 - 0.223*0.368) ≈ 92.1
        // Discounted by exp(-r*T) ≈ 0.951
        assert!(
            result.price > 70.0 && result.price < 110.0,
            "commodity forward price {} should be between 70 and 110",
            result.price
        );
    }

    #[test]
    fn constant_payoff_roundoff_does_not_make_stderr_nan() {
        // This value made the former raw-moment variance numerator a few ulps
        // below zero when accumulated three times.
        let product = make_constant_product(3.162_277_660_168_379_6e-25);
        let market = MultiAssetMarket::single(100.0, 0.20, 0.0, 0.0);
        let result = DslMonteCarloEngine::new(3, 1, 42)
            .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Scalar)
            .unwrap();

        assert!(result.price.is_finite());
        assert_eq!(result.stderr, Some(0.0));
    }

    #[test]
    fn centered_chunk_stats_preserve_small_variance_at_large_offset() {
        let mut whole = ChunkStats::default();
        for value in [
            1.0e12,
            1.0e12 + 0.25,
            1.0e12 + 0.5,
            1.0e12 + 0.75,
            1.0e12 + 1.0,
        ] {
            whole.record(value);
        }
        assert_eq!(whole.mean_pv, 1.0e12 + 0.5);
        assert!((whole.sample_variance() - 0.15625).abs() <= f64::EPSILON);

        let mut left = ChunkStats::default();
        left.record(1.0e12);
        left.record(1.0e12 + 0.25);
        let mut right = ChunkStats::default();
        right.record(1.0e12 + 0.5);
        right.record(1.0e12 + 0.75);
        right.record(1.0e12 + 1.0);
        left.merge(right);
        assert!((left.sample_variance() - 0.15625).abs() <= f64::EPSILON);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn explicit_parallel_small_batch_reports_scalar_vector_width() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let result = DslMonteCarloEngine::new(1, 2, 42)
            .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Parallel)
            .unwrap();

        assert_eq!(
            result.diagnostics.get(DiagKey::ExecutionBackend.as_str()),
            Some(&ExecutionBackend::Parallel.diagnostic_code())
        );
        assert_eq!(
            result.diagnostics.get(DiagKey::VectorWidth.as_str()),
            Some(&1.0)
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn explicit_parallel_is_reproducible_across_pool_sizes() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(10_003, 8, 314);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    engine
                        .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Parallel)
                        .unwrap()
                })
        };

        let one_thread = run(1);
        let four_threads = run(4);
        assert_eq!(one_thread.price.to_bits(), four_threads.price.to_bits());
        assert_eq!(
            one_thread.stderr.unwrap().to_bits(),
            four_threads.stderr.unwrap().to_bits()
        );
    }

    #[cfg(all(
        feature = "jit",
        feature = "parallel",
        not(target_family = "wasm"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn explicit_jit_is_reproducible_across_pool_sizes() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(MIN_PARALLEL_PATHS + 1, 8, 2718);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    engine
                        .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Jit)
                        .unwrap()
                })
        };

        let one_thread = run(1);
        let four_threads = run(4);
        assert_eq!(one_thread.price.to_bits(), four_threads.price.to_bits());
        assert_eq!(
            one_thread.stderr.unwrap().to_bits(),
            four_threads.stderr.unwrap().to_bits()
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm"), target_arch = "x86_64"))]
    #[test]
    fn explicit_jit_policy_matches_scalar_and_reports_backend() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(1_024, 32, 42);

        let scalar = engine
            .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Scalar)
            .unwrap();
        let jitted = engine
            .price_multi_asset_with_policy(&product, &market, ExecutionPolicy::Jit)
            .unwrap();

        assert_eq!(jitted.price, scalar.price);
        assert_eq!(jitted.stderr, scalar.stderr);
        assert_eq!(
            jitted.diagnostics.get(DiagKey::ExecutionBackend.as_str()),
            Some(&ExecutionBackend::Jit.diagnostic_code())
        );
        assert_eq!(
            jitted.diagnostics.get(DiagKey::VectorWidth.as_str()),
            Some(&1.0)
        );
    }

    #[cfg(all(
        feature = "jit",
        not(feature = "simd"),
        not(feature = "parallel"),
        not(target_family = "wasm"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn auto_jit_policy_uses_conservative_path_threshold() {
        let below = DslMonteCarloEngine::new(MIN_AUTO_JIT_PATHS - 1, 32, 42);
        assert_eq!(
            below
                .resolve_execution_backend(ExecutionPolicy::Auto)
                .unwrap(),
            ExecutionBackend::Scalar
        );

        let at_threshold = DslMonteCarloEngine::new(MIN_AUTO_JIT_PATHS, 32, 42);
        assert_eq!(
            at_threshold
                .resolve_execution_backend(ExecutionPolicy::Auto)
                .unwrap(),
            ExecutionBackend::Jit
        );
    }

    #[cfg(all(
        feature = "jit",
        not(target_family = "wasm"),
        not(target_arch = "x86_64")
    ))]
    #[test]
    fn unsupported_jit_policy_returns_error_and_auto_avoids_jit() {
        let engine = DslMonteCarloEngine::new(MIN_AUTO_JIT_PATHS, 32, 42);

        let error = engine
            .resolve_execution_backend(ExecutionPolicy::Jit)
            .unwrap_err();
        assert!(
            error.to_string().contains("unsupported"),
            "unexpected error: {error}"
        );
        assert_ne!(
            engine
                .resolve_execution_backend(ExecutionPolicy::Auto)
                .unwrap(),
            ExecutionBackend::Jit
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm"), target_arch = "x86_64"))]
    #[test]
    fn prepared_jit_evaluator_cache_reuses_identical_plan() {
        let product = make_forward_product();
        let first = cached_jit_evaluator(&product, 32, 0.05).unwrap();
        let second = cached_jit_evaluator(&product, 32, 0.05).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm"), target_arch = "x86_64"))]
    #[test]
    fn prepared_jit_evaluator_cache_retains_alternating_plans() {
        let first_product = make_forward_product();
        let second_product = make_constant_product(17.0);
        let first = cached_jit_evaluator(&first_product, 31, 0.041).unwrap();
        let second = cached_jit_evaluator(&second_product, 33, 0.042).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));

        let first_again = cached_jit_evaluator(&first_product, 31, 0.041).unwrap();
        let second_again = cached_jit_evaluator(&second_product, 33, 0.042).unwrap();
        assert!(Arc::ptr_eq(&first, &first_again));
        assert!(Arc::ptr_eq(&second, &second_again));
    }
}
