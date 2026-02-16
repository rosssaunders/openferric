use crate::math::normal_inv_cdf;
use crate::models::{Gbm, Heston};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

pub type PathEvaluator = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

#[inline]
fn uniform_open01(u: f64) -> f64 {
    u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
}

#[inline]
fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    normal_inv_cdf(uniform_open01(rng.random::<f64>()))
}

pub trait PathGenerator: Send + Sync {
    fn steps(&self) -> usize;
    fn generate_from_normals(&self, normals_1: &[f64], normals_2: &[f64]) -> Vec<f64>;
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
        let dt = self.maturity / self.steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut path = Vec::with_capacity(self.steps + 1);
        let mut s = self.s0;
        path.push(s);

        for &z in normals_1.iter().take(self.steps) {
            s += self.model.mu * s * dt + self.model.sigma * s * sqrt_dt * z;
            s = s.max(1e-12);
            path.push(s);
        }

        path
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
        let dt = self.maturity / self.steps as f64;

        let mut path = Vec::with_capacity(self.steps + 1);
        let mut s = self.s0;
        let mut v = self.model.v0;
        path.push(s);

        for (&z1, &z2) in normals_1.iter().zip(normals_2.iter()).take(self.steps) {
            let (s_next, v_next) = self.model.step_euler(s, v, dt, z1, z2);
            s = s_next.max(1e-12);
            v = v_next.max(0.0);
            path.push(s);
        }

        path
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
}

impl MonteCarloEngine {
    pub fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            num_paths,
            antithetic: false,
            control_variate: None,
            seed,
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
        let control = self.control_variate.clone();

        let simulate_sample = |i: usize| {
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64 * 7_919));
            let mut z1 = vec![0.0_f64; steps];
            let mut z2 = vec![0.0_f64; steps];

            for j in 0..steps {
                z1[j] = sample_standard_normal(&mut rng);
                z2[j] = sample_standard_normal(&mut rng);
            }

            let path = generator.generate_from_normals(&z1, &z2);
            let x = payoff(&path);
            let y = control.as_ref().map_or(0.0, |c| (c.evaluator)(&path));

            if self.antithetic {
                let z1a: Vec<f64> = z1.iter().map(|v| -v).collect();
                let z2a: Vec<f64> = z2.iter().map(|v| -v).collect();
                let path_a = generator.generate_from_normals(&z1a, &z2a);
                let xa = payoff(&path_a);
                let ya = control.as_ref().map_or(0.0, |c| (c.evaluator)(&path_a));
                (0.5 * (x + xa), 0.5 * (y + ya))
            } else {
                (x, y)
            }
        };

        #[cfg(feature = "parallel")]
        let values = (0..samples)
            .into_par_iter()
            .map(simulate_sample)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let values = (0..samples).map(simulate_sample).collect::<Vec<_>>();
        let n = values.len() as f64;

        let adjusted: Vec<f64> = if let Some(cv) = &control {
            let mean_x = values.iter().map(|(x, _)| *x).sum::<f64>() / n;
            let mean_y = values.iter().map(|(_, y)| *y).sum::<f64>() / n;

            let cov_xy = values
                .iter()
                .map(|(x, y)| (x - mean_x) * (y - mean_y))
                .sum::<f64>()
                / (n - 1.0).max(1.0);

            let var_y = values
                .iter()
                .map(|(_, y)| (y - mean_y).powi(2))
                .sum::<f64>()
                / (n - 1.0).max(1.0);

            let beta = if var_y > 1e-16 { cov_xy / var_y } else { 0.0 };
            values
                .iter()
                .map(|(x, y)| x + beta * (cv.expected - y))
                .collect()
        } else {
            values.iter().map(|(x, _)| *x).collect()
        };

        let mean = adjusted.iter().sum::<f64>() / n;
        let var = if adjusted.len() > 1 {
            adjusted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        let price = discount_factor * mean;
        let stderr = discount_factor * (var / n).sqrt();
        (price, stderr)
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
}
