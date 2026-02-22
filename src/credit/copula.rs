//! Credit analytics for Copula.
//!
//! Module openferric::credit::copula provides pricing helpers and model utilities for credit products.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::math::normal_cdf;

use super::survival_curve::SurvivalCurve;

/// One-factor Gaussian copula model for correlated defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCopula {
    /// Common-factor loading in `[-1, 1]`.
    pub rho: f64,
}

/// Result of one basket default simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct BasketDefaultSimulation {
    /// Simulated default times in years.
    pub default_times: Vec<f64>,
    /// Realized common market factor.
    pub market_factor: f64,
    /// Realized latent variables `Z_i`.
    pub latent_variables: Vec<f64>,
}

impl BasketDefaultSimulation {
    /// Number of defaults observed by `horizon`.
    pub fn defaults_by(&self, horizon: f64) -> usize {
        self.default_times
            .iter()
            .filter(|&&tau| tau <= horizon)
            .count()
    }
}

impl GaussianCopula {
    pub fn new(rho: f64) -> Self {
        Self {
            rho: rho.clamp(-0.999_999, 0.999_999),
        }
    }

    /// Simulates a homogeneous basket using one shared survival curve.
    pub fn simulate_homogeneous<R: Rng + ?Sized>(
        &self,
        num_names: usize,
        survival_curve: &SurvivalCurve,
        rng: &mut R,
    ) -> BasketDefaultSimulation {
        let market_factor: f64 = StandardNormal.sample(rng);
        let sqrt_term = (1.0 - self.rho * self.rho).sqrt();

        let mut latent_variables = Vec::with_capacity(num_names);
        let mut default_times = Vec::with_capacity(num_names);
        for _ in 0..num_names {
            let epsilon: f64 = StandardNormal.sample(rng);
            let z = self.rho * market_factor + sqrt_term * epsilon;
            let u = normal_cdf(z).clamp(1.0e-15, 1.0 - 1.0e-15);
            latent_variables.push(z);
            default_times.push(survival_curve.inverse_survival_prob(u));
        }

        BasketDefaultSimulation {
            default_times,
            market_factor,
            latent_variables,
        }
    }

    /// Simulates a heterogeneous basket using one survival curve per name.
    pub fn simulate<R: Rng + ?Sized>(
        &self,
        survival_curves: &[SurvivalCurve],
        rng: &mut R,
    ) -> BasketDefaultSimulation {
        let market_factor: f64 = StandardNormal.sample(rng);
        let sqrt_term = (1.0 - self.rho * self.rho).sqrt();

        let mut latent_variables = Vec::with_capacity(survival_curves.len());
        let mut default_times = Vec::with_capacity(survival_curves.len());
        for curve in survival_curves {
            let epsilon: f64 = StandardNormal.sample(rng);
            let z = self.rho * market_factor + sqrt_term * epsilon;
            let u = normal_cdf(z).clamp(1.0e-15, 1.0 - 1.0e-15);
            latent_variables.push(z);
            default_times.push(curve.inverse_survival_prob(u));
        }

        BasketDefaultSimulation {
            default_times,
            market_factor,
            latent_variables,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn simulation_returns_one_default_time_per_name() {
        let curve = SurvivalCurve::new(vec![(1.0, 0.98), (3.0, 0.92), (5.0, 0.85), (10.0, 0.70)]);
        let model = GaussianCopula::new(0.3);
        let mut rng = StdRng::seed_from_u64(42);

        let sim = model.simulate_homogeneous(125, &curve, &mut rng);
        assert_eq!(sim.default_times.len(), 125);
        assert_eq!(sim.latent_variables.len(), 125);
        assert!(sim.default_times.iter().all(|t| t.is_finite() && *t >= 0.0));
    }

    #[test]
    fn higher_rho_increases_joint_defaults() {
        let curve = SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.02]);
        let horizon = 5.0;
        let n_paths = 20_000;

        let mut rng_low = StdRng::seed_from_u64(7);
        let mut rng_high = StdRng::seed_from_u64(7);
        let model_low = GaussianCopula::new(0.0);
        let model_high = GaussianCopula::new(0.7);

        let mut both_default_low = 0usize;
        let mut both_default_high = 0usize;
        for _ in 0..n_paths {
            let sim_low = model_low.simulate_homogeneous(2, &curve, &mut rng_low);
            let sim_high = model_high.simulate_homogeneous(2, &curve, &mut rng_high);

            if sim_low.defaults_by(horizon) == 2 {
                both_default_low += 1;
            }
            if sim_high.defaults_by(horizon) == 2 {
                both_default_high += 1;
            }
        }

        let p_both_low = both_default_low as f64 / n_paths as f64;
        let p_both_high = both_default_high as f64 / n_paths as f64;
        assert!(p_both_high > p_both_low + 0.01);
    }
}
