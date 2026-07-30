//! Module `mc::simulation`.
//!
//! Implements simulation abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `PathEvaluator`, `PathGenerator`, `GbmPathGenerator`, `HestonPathGenerator`, `ControlVariate` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};
use crate::models::{Gbm, Heston};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) const PATH_SIMULATION_CHUNK_SAMPLES: usize = 256;

/// Estimates the amount of serial work removed by distributing fixed-size
/// chunks over `threads` workers. Work is expressed in caller-defined units
/// (`work_per_sample`), which lets exact-terminal and time-stepped simulations
/// use separate cost models without sharing a global path-count cutoff.
#[cfg(feature = "parallel")]
pub(crate) fn estimated_parallel_work_savings(
    samples: usize,
    work_per_sample: usize,
    chunk_samples: usize,
    threads: usize,
) -> usize {
    if samples == 0 || work_per_sample == 0 || chunk_samples == 0 || threads <= 1 {
        return 0;
    }

    let chunk_count = samples.div_ceil(chunk_samples);
    let workers = threads.min(chunk_count);
    if workers <= 1 {
        return 0;
    }

    // All chunks except the final tail are equal. Assign full chunks as evenly
    // as possible and place the tail on a least-loaded worker; this models the
    // critical-path load of Rayon's work-stealing schedule without depending on
    // scheduling order.
    let full_chunks = samples / chunk_samples;
    let tail_samples = samples % chunk_samples;
    let base_full_chunks = full_chunks / workers;
    let extra_full_chunks = full_chunks % workers;
    let max_full_load =
        (base_full_chunks + usize::from(extra_full_chunks > 0)).saturating_mul(chunk_samples);
    let critical_samples = if tail_samples == 0 {
        max_full_load
    } else if full_chunks < workers {
        max_full_load.max(tail_samples)
    } else if extra_full_chunks == 0 {
        base_full_chunks
            .saturating_mul(chunk_samples)
            .saturating_add(tail_samples)
    } else {
        max_full_load.max(
            base_full_chunks
                .saturating_mul(chunk_samples)
                .saturating_add(tail_samples),
        )
    };

    samples
        .saturating_sub(critical_samples)
        .saturating_mul(work_per_sample)
}

/// Selects Rayon for the generic path engine when enough step work can be
/// removed from the serial critical path to cover fixed scheduling plus a
/// bounded allowance for waking a larger pool.
///
/// `mc_path_rayon_crossover` benchmarks show that two 256-path, one-step GBM
/// chunks are faster on a 2-thread pool, while the 24-thread host needs three
/// such chunks to amortize pool wake-up. Longer paths and two-stream models
/// cross over with smaller tail chunks because each sample contains more work.
#[cfg(feature = "parallel")]
pub(crate) fn should_auto_parallelize_path(
    samples: usize,
    steps: usize,
    normal_streams: usize,
    threads: usize,
) -> bool {
    if threads <= 1 {
        return false;
    }
    let step_work = steps.max(1).saturating_mul(normal_streams.max(1));
    let saved_work =
        estimated_parallel_work_savings(samples, step_work, PATH_SIMULATION_CHUNK_SAMPLES, threads);
    // Waking a large pool costs more than scheduling into a two-thread pool,
    // but the cost plateaus once a handful of tasks can be stolen. Capping the
    // allowance at four workers lets expensive two-chunk paths parallelize
    // without making the threshold grow with unrelated idle workers.
    let wakeup_allowance = PATH_SIMULATION_CHUNK_SAMPLES.saturating_mul(threads.min(4).div_ceil(2));
    saved_work >= wakeup_allowance
}

pub type PathEvaluator = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// CPU execution strategy for the generic path simulation engine.
///
/// SIMD, GPU, and JIT execution require workload-specific implementations and
/// are exposed by the higher-level pricing engines. Keeping this policy
/// limited to backends the generic callback engine can actually execute avoids
/// silently accepting an unsupported hardware request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CpuExecutionPolicy {
    /// Select scalar or parallel execution from the available CPU resources
    /// and workload size.
    #[default]
    Auto,
    /// Run deterministically on the calling thread.
    Scalar,
    /// Run fixed simulation chunks on the Rayon thread pool.
    #[cfg(feature = "parallel")]
    Parallel,
}

pub trait PathGenerator: Send + Sync {
    fn steps(&self) -> usize;
    fn generate_from_normals(&self, normals_1: &[f64], normals_2: &[f64]) -> Vec<f64>;

    /// Write path directly into a pre-allocated buffer, avoiding per-path heap allocation.
    /// Default implementation delegates to `generate_from_normals` and copies.
    fn generate_into(&self, normals_1: &[f64], normals_2: &[f64], out: &mut [f64]) {
        let path = self.generate_from_normals(normals_1, normals_2);
        out[..path.len()].copy_from_slice(&path);
    }

    /// Number of independent normal streams required per time step.
    /// GBM needs 1 (asset diffusion only), Heston needs 2 (asset + variance).
    /// The MC engine skips generating unused streams for a ~2× speedup on RNG.
    fn num_normal_streams(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub struct GbmPathGenerator {
    pub model: Gbm,
    pub s0: f64,
    pub maturity: f64,
    pub steps: usize,
}

impl PathGenerator for GbmPathGenerator {
    fn steps(&self) -> usize {
        self.steps
    }

    fn generate_from_normals(&self, normals_1: &[f64], _normals_2: &[f64]) -> Vec<f64> {
        let mut path = vec![0.0_f64; self.steps + 1];
        self.generate_into(normals_1, _normals_2, &mut path);
        path
    }

    fn generate_into(&self, normals_1: &[f64], _normals_2: &[f64], out: &mut [f64]) {
        let dt = self.maturity / self.steps as f64;
        let sqrt_dt = dt.sqrt();
        let drift = (self.model.mu - 0.5 * self.model.sigma * self.model.sigma) * dt;
        let diffusion = self.model.sigma * sqrt_dt;

        let mut s = self.s0;
        out[0] = s;

        for (j, &z) in normals_1.iter().enumerate().take(self.steps) {
            s *= diffusion.mul_add(z, drift).exp();
            out[j + 1] = s;
        }
    }

    /// GBM only uses one normal stream (no variance process).
    fn num_normal_streams(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone)]
pub struct HestonPathGenerator {
    pub model: Heston,
    pub s0: f64,
    pub maturity: f64,
    pub steps: usize,
}

impl PathGenerator for HestonPathGenerator {
    fn steps(&self) -> usize {
        self.steps
    }

    fn generate_from_normals(&self, normals_1: &[f64], normals_2: &[f64]) -> Vec<f64> {
        let mut path = vec![0.0_f64; self.steps + 1];
        self.generate_into(normals_1, normals_2, &mut path);
        path
    }

    fn generate_into(&self, normals_1: &[f64], normals_2: &[f64], out: &mut [f64]) {
        let dt = self.maturity / self.steps as f64;

        let mut s = self.s0;
        let mut v = self.model.v0;
        out[0] = s;

        for (j, (&z1, &z2)) in normals_1
            .iter()
            .zip(normals_2.iter())
            .enumerate()
            .take(self.steps)
        {
            let (s_next, v_next) = self.model.step_euler(s, v, dt, z1, z2);
            s = s_next.max(1e-12);
            v = v_next.max(0.0);
            out[j + 1] = s;
        }
    }
}

#[derive(Clone)]
pub struct ControlVariate {
    pub expected: f64,
    pub evaluator: PathEvaluator,
}

#[derive(Debug, Clone, Copy, Default)]
struct BivariateMoments {
    count: usize,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    co_moment: f64,
}

impl BivariateMoments {
    #[inline(always)]
    fn record(&mut self, x: f64, y: f64) {
        self.count += 1;
        let n = self.count as f64;
        let delta_x = x - self.mean_x;
        let delta_y = y - self.mean_y;
        self.mean_x += delta_x / n;
        self.mean_y += delta_y / n;
        self.m2_x += delta_x * (x - self.mean_x);
        self.m2_y += delta_y * (y - self.mean_y);
        self.co_moment += delta_x * (y - self.mean_y);
    }

    #[inline]
    fn merge(&mut self, other: Self) {
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
        let weight = lhs_count * rhs_count / total_count;
        let delta_x = other.mean_x - self.mean_x;
        let delta_y = other.mean_y - self.mean_y;
        self.mean_x += delta_x * rhs_count / total_count;
        self.mean_y += delta_y * rhs_count / total_count;
        self.m2_x += other.m2_x + delta_x * delta_x * weight;
        self.m2_y += other.m2_y + delta_y * delta_y * weight;
        self.co_moment += other.co_moment + delta_x * delta_y * weight;
        self.count += other.count;
    }

    #[inline]
    fn sample_variance_x(self) -> f64 {
        if self.count > 1 {
            let variance = self.m2_x / (self.count as f64 - 1.0);
            if variance < 0.0 { 0.0 } else { variance }
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub struct MonteCarloEngine {
    pub num_paths: usize,
    pub antithetic: bool,
    pub control_variate: Option<ControlVariate>,
    pub seed: u64,
    pub(crate) rng_kind: FastRngKind,
    pub(crate) reproducible: bool,
    /// CPU execution strategy for the generic callback engine.
    pub execution_policy: CpuExecutionPolicy,
}

impl MonteCarloEngine {
    pub fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            num_paths,
            antithetic: false,
            control_variate: None,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
            execution_policy: CpuExecutionPolicy::Auto,
        }
    }

    pub fn with_antithetic(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
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

    pub fn with_control_variate(mut self, control_variate: ControlVariate) -> Self {
        self.control_variate = Some(control_variate);
        self
    }

    pub fn with_rng_kind(mut self, rng_kind: FastRngKind) -> Self {
        self.rng_kind = rng_kind;
        if matches!(rng_kind, FastRngKind::ThreadRng) {
            self.reproducible = false;
        }
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.reproducible = !matches!(self.rng_kind, FastRngKind::ThreadRng);
        self
    }

    pub fn with_randomized_streams(mut self) -> Self {
        self.reproducible = false;
        self
    }

    pub fn with_thread_rng(mut self) -> Self {
        self.rng_kind = FastRngKind::ThreadRng;
        self.reproducible = false;
        self
    }

    /// Sets the CPU execution strategy for generic path simulation.
    ///
    /// The generic engine supports scalar and, when enabled, Rayon-parallel
    /// execution.
    pub fn with_execution_policy(mut self, execution_policy: CpuExecutionPolicy) -> Self {
        self.execution_policy = execution_policy;
        self
    }

    pub fn run<G, P>(&self, generator: &G, payoff: P, discount_factor: f64) -> (f64, f64)
    where
        G: PathGenerator,
        P: Fn(&[f64]) -> f64 + Send + Sync,
    {
        assert!(self.num_paths > 0, "num_paths must be > 0");

        let samples = if self.antithetic {
            self.num_paths.div_ceil(2)
        } else {
            self.num_paths
        };

        let steps = generator.steps();
        let num_streams = generator.num_normal_streams();
        let control = self.control_variate.clone();
        let rng_kind = self.rng_kind;
        let reproducible = self.reproducible;
        let base_seed = self.seed;
        let path_len = steps + 1;
        let antithetic = self.antithetic;
        let has_cv = control.is_some();

        type Scratch = (Vec<f64>, Vec<f64>, Vec<f64>);
        let chunk_count = samples.div_ceil(PATH_SIMULATION_CHUNK_SAMPLES);
        let make_scratch = || {
            (
                vec![0.0_f64; steps],
                vec![0.0_f64; steps],
                vec![0.0_f64; path_len],
            )
        };

        // A fixed chunk owns one RNG stream. Chunk boundaries and seeds do not
        // depend on the Rayon pool, while map_init reuses path/normal buffers
        // for every chunk handled by a worker.
        let simulate_chunk = |scratch: &mut Scratch, chunk_index: usize| -> BivariateMoments {
            let (z1, z2, path) = scratch;
            let seed = resolve_stream_seed(base_seed, chunk_index, reproducible);
            let mut rng = FastRng::from_seed(rng_kind, seed);
            let chunk_start = chunk_index * PATH_SIMULATION_CHUNK_SAMPLES;
            let chunk_end = (chunk_start + PATH_SIMULATION_CHUNK_SAMPLES).min(samples);
            let mut stats = BivariateMoments::default();

            for _ in chunk_start..chunk_end {
                // Only generate as many normal streams as the model needs.
                // GBM needs 1 stream (skipping z2 halves RNG + inverse-CDF work).
                for j in 0..steps {
                    z1[j] = sample_standard_normal(&mut rng);
                    if num_streams >= 2 {
                        z2[j] = sample_standard_normal(&mut rng);
                    }
                }

                generator.generate_into(z1, z2, path);
                let x = payoff(path);
                let y = if has_cv {
                    (control.as_ref().unwrap().evaluator)(path)
                } else {
                    0.0
                };

                let (x, y) = if antithetic {
                    for value in z1.iter_mut() {
                        *value = -*value;
                    }
                    for value in z2.iter_mut() {
                        *value = -*value;
                    }
                    generator.generate_into(z1, z2, path);
                    let xa = payoff(path);
                    let ya = if has_cv {
                        (control.as_ref().unwrap().evaluator)(path)
                    } else {
                        0.0
                    };
                    (0.5 * (x + xa), 0.5 * (y + ya))
                } else {
                    (x, y)
                };

                stats.record(x, y);
            }

            stats
        };

        let reduce_fn = |mut a: BivariateMoments, b: BivariateMoments| {
            a.merge(b);
            a
        };

        #[cfg(feature = "parallel")]
        let stats = {
            let use_parallel = match self.execution_policy {
                CpuExecutionPolicy::Parallel => true,
                CpuExecutionPolicy::Auto => should_auto_parallelize_path(
                    samples,
                    steps,
                    num_streams,
                    rayon::current_num_threads(),
                ),
                CpuExecutionPolicy::Scalar => false,
            };

            if use_parallel {
                let partials = (0..chunk_count)
                    .into_par_iter()
                    .map_init(&make_scratch, &simulate_chunk)
                    .collect::<Vec<_>>();
                partials
                    .into_iter()
                    .fold(BivariateMoments::default(), reduce_fn)
            } else {
                let mut scratch = make_scratch();
                (0..chunk_count)
                    .map(|chunk_index| simulate_chunk(&mut scratch, chunk_index))
                    .fold(BivariateMoments::default(), reduce_fn)
            }
        };

        #[cfg(not(feature = "parallel"))]
        let stats = {
            let mut scratch = make_scratch();
            (0..chunk_count)
                .map(|chunk_index| simulate_chunk(&mut scratch, chunk_index))
                .fold(BivariateMoments::default(), reduce_fn)
        };

        let n = stats.count as f64;

        if let Some(cv) = &control {
            // The common sample-variance denominator cancels in beta.
            let beta = if stats.m2_y.is_finite() && stats.m2_y > 0.0 {
                stats.co_moment / stats.m2_y
            } else {
                0.0
            };
            let cv_expected = cv.expected;

            // adj = X + beta(E[Y] - Y). Centered moments are translation
            // invariant, so the expected-value term changes only the mean.
            let mean = stats.mean_x + beta * (cv_expected - stats.mean_y);
            let adjusted_m2 = stats.m2_x + beta * beta * stats.m2_y - 2.0 * beta * stats.co_moment;
            let var = if stats.count > 1 {
                let variance = adjusted_m2 / (n - 1.0);
                if variance < 0.0 { 0.0 } else { variance }
            } else {
                0.0
            };
            let price = discount_factor * mean;
            let stderr = discount_factor * (var / n).sqrt();
            (price, stderr)
        } else {
            let price = discount_factor * stats.mean_x;
            let stderr = discount_factor * (stats.sample_variance_x() / n).sqrt();
            (price, stderr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::OptionType;
    use crate::pricing::european::black_scholes_price;

    #[test]
    fn gbm_path_generator_returns_expected_length() {
        let generator = GbmPathGenerator {
            model: Gbm {
                mu: 0.05,
                sigma: 0.2,
            },
            s0: 100.0,
            maturity: 1.0,
            steps: 50,
        };
        let normals = vec![0.0; 50];
        let path = generator.generate_from_normals(&normals, &normals);
        assert_eq!(path.len(), 51);
        assert!(path.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn heston_path_generator_returns_expected_length() {
        let generator = HestonPathGenerator {
            model: Heston {
                mu: 0.03,
                kappa: 1.5,
                theta: 0.04,
                xi: 0.5,
                rho: -0.7,
                v0: 0.04,
            },
            s0: 100.0,
            maturity: 1.0,
            steps: 40,
        };
        let z1 = vec![0.1; 40];
        let z2 = vec![-0.2; 40];
        let path = generator.generate_from_normals(&z1, &z2);
        assert_eq!(path.len(), 41);
        assert!(path.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn mc_call_converges_to_black_scholes_within_two_stderr() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let generator = GbmPathGenerator {
            model: Gbm { mu: r, sigma },
            s0,
            maturity: t,
            steps: 252,
        };
        let engine = MonteCarloEngine::new(60_000, 42).with_antithetic(true);

        let discount = (-r * t).exp();
        let (price, stderr) = engine.run(
            &generator,
            |path| (path[path.len() - 1] - k).max(0.0),
            discount,
        );

        let bs = black_scholes_price(OptionType::Call, s0, k, r, sigma, t);
        assert!((price - bs).abs() <= 2.0 * stderr + 2e-2);
    }

    #[test]
    fn control_variate_improves_or_matches_error() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let generator = GbmPathGenerator {
            model: Gbm { mu: r, sigma },
            s0,
            maturity: t,
            steps: 126,
        };

        let discount = (-r * t).exp();
        let bs = black_scholes_price(OptionType::Call, s0, k, r, sigma, t);

        let base = MonteCarloEngine::new(20_000, 123).with_antithetic(false);
        let (p0, _e0) = base.run(
            &generator,
            |path| (path[path.len() - 1] - k).max(0.0),
            discount,
        );

        let cv = ControlVariate {
            expected: s0 * (r * t).exp(),
            evaluator: Arc::new(|path: &[f64]| path[path.len() - 1]),
        };
        let with_cv = MonteCarloEngine::new(20_000, 123)
            .with_antithetic(false)
            .with_control_variate(cv);
        let (p1, _e1) = with_cv.run(
            &generator,
            |path| (path[path.len() - 1] - k).max(0.0),
            discount,
        );

        assert!((p1 - bs).abs() <= (p0 - bs).abs() + 0.15);
    }

    #[test]
    fn constant_payoff_has_zero_stderr_with_and_without_control_variate() {
        let generator = GbmPathGenerator {
            model: Gbm {
                mu: 0.03,
                sigma: 0.2,
            },
            s0: 100.0,
            maturity: 1.0,
            steps: 1,
        };
        // Three copies of 0.1 make the sum-of-squares variance formula
        // slightly negative in binary floating point without cancellation
        // protection.
        let base = MonteCarloEngine::new(3, 7).with_execution_policy(CpuExecutionPolicy::Scalar);
        let (price, stderr) = base.run(&generator, |_| 0.1, 0.95);
        assert!((price - 0.095).abs() <= f64::EPSILON);
        assert_eq!(stderr, 0.0);

        let control = ControlVariate {
            expected: 0.2,
            evaluator: Arc::new(|_| 0.2),
        };
        let (price, stderr) = base
            .with_control_variate(control)
            .run(&generator, |_| 0.1, 0.95);
        assert!((price - 0.095).abs() <= f64::EPSILON);
        assert_eq!(stderr, 0.0);
    }

    #[test]
    fn near_constant_payoff_retains_nonzero_stderr() {
        let generator = GbmPathGenerator {
            model: Gbm {
                mu: 0.0,
                sigma: 0.2,
            },
            s0: 100.0,
            maturity: 1.0,
            steps: 1,
        };
        let (_, stderr) = MonteCarloEngine::new(50_000, 17)
            .with_execution_policy(CpuExecutionPolicy::Scalar)
            .run(
                &generator,
                |path| 100.0 + 1.0e-9 * (path[path.len() - 1] / 100.0),
                1.0,
            );
        assert!(stderr.is_finite());
        assert!(stderr > 0.0, "stderr={stderr}");
        assert!(stderr < 1.0e-9, "stderr={stderr}");
    }

    #[test]
    fn thread_rng_never_claims_seeded_reproducibility() {
        for engine in [
            MonteCarloEngine::new(32, 1).with_thread_rng().with_seed(42),
            MonteCarloEngine::new(32, 1).with_seed(42).with_thread_rng(),
        ] {
            assert_eq!(engine.rng_kind, FastRngKind::ThreadRng);
            assert!(!engine.reproducible);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn path_auto_parallel_cost_model_respects_work_shape_and_pool_size() {
        assert!(!should_auto_parallelize_path(512, 1, 1, 1));

        assert!(!should_auto_parallelize_path(511, 1, 1, 2));
        assert!(should_auto_parallelize_path(512, 1, 1, 2));

        assert!(!should_auto_parallelize_path(767, 1, 1, 24));
        assert!(should_auto_parallelize_path(768, 1, 1, 24));

        // Longer samples repay the larger-pool wake-up allowance with a
        // smaller tail: 64 paths × 8 steps provide 512 saved work units.
        assert!(!should_auto_parallelize_path(319, 8, 1, 24));
        assert!(should_auto_parallelize_path(320, 8, 1, 24));

        // A single chunk cannot benefit regardless of per-path cost.
        assert!(!should_auto_parallelize_path(256, 1_024, 2, 24));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_work_savings_accounts_for_pool_size_and_tail_balance() {
        assert_eq!(estimated_parallel_work_savings(8_193, 1, 4_096, 1), 0);
        assert_eq!(estimated_parallel_work_savings(8_193, 1, 4_096, 2), 4_096);
        assert_eq!(estimated_parallel_work_savings(8_193, 1, 4_096, 24), 4_097);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn fixed_chunk_streams_match_across_scalar_and_parallel_pools() {
        let generator = GbmPathGenerator {
            model: Gbm {
                mu: 0.04,
                sigma: 0.3,
            },
            s0: 100.0,
            maturity: 1.5,
            steps: 37,
        };
        let control = ControlVariate {
            expected: 100.0 * (0.04_f64 * 1.5).exp(),
            evaluator: Arc::new(|path: &[f64]| path[path.len() - 1]),
        };
        let base = MonteCarloEngine::new(10_003, 918)
            .with_antithetic(true)
            .with_control_variate(control);
        let payoff = |path: &[f64]| (path[path.len() - 1] - 105.0).max(0.0);
        let discount = (-0.04_f64 * 1.5).exp();

        let scalar = base
            .clone()
            .with_execution_policy(CpuExecutionPolicy::Scalar)
            .run(&generator, payoff, discount);
        let run_parallel = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| {
                    base.clone()
                        .with_execution_policy(CpuExecutionPolicy::Parallel)
                        .run(&generator, payoff, discount)
                })
        };

        assert_eq!(scalar, run_parallel(2));
        assert_eq!(scalar, run_parallel(4));
    }
}
