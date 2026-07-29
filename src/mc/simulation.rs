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

#[derive(Clone)]
pub struct MonteCarloEngine {
    pub num_paths: usize,
    pub antithetic: bool,
    pub control_variate: Option<ControlVariate>,
    pub seed: u64,
    pub rng_kind: FastRngKind,
    pub reproducible: bool,
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
        self.reproducible = true;
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

        // Accumulator: (sum_x, sum_x2, sum_y, sum_xy, sum_y2, count)
        // Using a tuple-struct alias for clarity.
        type Acc = (f64, f64, f64, f64, f64, u64);
        let identity: Acc = (0.0, 0.0, 0.0, 0.0, 0.0, 0);
        type Scratch = (Vec<f64>, Vec<f64>, Vec<f64>);
        const CHUNK_SAMPLES: usize = 256;
        let chunk_count = samples.div_ceil(CHUNK_SAMPLES);
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
        let simulate_chunk = |scratch: &mut Scratch, chunk_index: usize| -> Acc {
            let (z1, z2, path) = scratch;
            let seed = resolve_stream_seed(base_seed, chunk_index, reproducible);
            let mut rng = FastRng::from_seed(rng_kind, seed);
            let chunk_start = chunk_index * CHUNK_SAMPLES;
            let chunk_end = (chunk_start + CHUNK_SAMPLES).min(samples);
            let mut stats = identity;

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

                stats.0 += x;
                stats.1 += x * x;
                stats.2 += y;
                stats.3 += x * y;
                stats.4 += y * y;
                stats.5 += 1;
            }

            stats
        };

        let reduce_fn = |a: Acc, b: Acc| -> Acc {
            (
                a.0 + b.0,
                a.1 + b.1,
                a.2 + b.2,
                a.3 + b.3,
                a.4 + b.4,
                a.5 + b.5,
            )
        };

        #[cfg(feature = "parallel")]
        let (sum_x, sum_x2, sum_y, sum_xy, sum_y2, count) = {
            const AUTO_PARALLEL_MIN_NORMALS: usize = 32_768;
            let work_items = samples.saturating_mul(steps.max(1));
            let use_parallel = match self.execution_policy {
                CpuExecutionPolicy::Parallel => true,
                CpuExecutionPolicy::Auto => {
                    rayon::current_num_threads() > 1 && work_items >= AUTO_PARALLEL_MIN_NORMALS
                }
                CpuExecutionPolicy::Scalar => false,
            };

            if use_parallel {
                let partials = (0..chunk_count)
                    .into_par_iter()
                    .map_init(&make_scratch, &simulate_chunk)
                    .collect::<Vec<_>>();
                partials.into_iter().fold(identity, reduce_fn)
            } else {
                let mut scratch = make_scratch();
                (0..chunk_count)
                    .map(|chunk_index| simulate_chunk(&mut scratch, chunk_index))
                    .fold(identity, reduce_fn)
            }
        };

        #[cfg(not(feature = "parallel"))]
        let (sum_x, sum_x2, sum_y, sum_xy, sum_y2, count) = {
            let mut scratch = make_scratch();
            (0..chunk_count)
                .map(|chunk_index| simulate_chunk(&mut scratch, chunk_index))
                .fold(identity, reduce_fn)
        };

        let n = count as f64;

        if let Some(cv) = &control {
            // Derive control-variate adjusted statistics from accumulated sums.
            // cov(X,Y) = (sum_xy - sum_x * sum_y / n) / (n - 1)
            // var(Y)    = (sum_y2 - sum_y^2 / n)       / (n - 1)
            let denom = (n - 1.0).max(1.0);
            let cov_xy = (sum_xy - sum_x * sum_y / n) / denom;
            let var_y = (sum_y2 - sum_y * sum_y / n) / denom;

            let beta = if var_y > 1e-16 { cov_xy / var_y } else { 0.0 };
            let cv_expected = cv.expected;

            // Adjusted value: adj_i = x_i + beta * (cv_expected - y_i)
            // sum_adj   = sum_x + beta * (n * cv_expected - sum_y)
            // sum_adj^2 = sum_x2 + 2*beta*cv_expected*sum_x - 2*beta*sum_xy
            //           + beta^2 * (n*cv_expected^2 - 2*cv_expected*sum_y + sum_y2)
            let sum_adj = sum_x + beta * (n * cv_expected - sum_y);
            let sum_adj_sq = sum_x2 + 2.0 * beta * cv_expected * sum_x - 2.0 * beta * sum_xy
                + beta
                    * beta
                    * (n * cv_expected * cv_expected - 2.0 * cv_expected * sum_y + sum_y2);

            let mean = sum_adj / n;
            let var = if n > 1.0 {
                (sum_adj_sq - sum_adj * sum_adj / n) / (n - 1.0)
            } else {
                0.0
            };
            let var = if var < 0.0 { 0.0 } else { var };
            let price = discount_factor * mean;
            let stderr = discount_factor * (var / n).sqrt();
            (price, stderr)
        } else {
            let mean = sum_x / n;
            let var = if n > 1.0 {
                (sum_x2 - sum_x * sum_x / n) / (n - 1.0)
            } else {
                0.0
            };
            let var = if var < 0.0 { 0.0 } else { var };
            let price = discount_factor * mean;
            let stderr = discount_factor * (var / n).sqrt();
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
