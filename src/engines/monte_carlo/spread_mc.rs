#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::spread::SpreadOption;
use crate::market::Market;
use crate::math::fast_rng::{resolve_stream_seed, sample_standard_normal, FastRng, FastRngKind};

/// Monte Carlo spread-option engine under correlated GBM.
#[derive(Debug, Clone)]
pub struct SpreadMonteCarloEngine {
    /// Number of simulated paths.
    pub num_paths: usize,
    /// RNG seed.
    pub seed: u64,
    /// Enables antithetic variates.
    pub antithetic: bool,
    /// Pseudo-random number generator backend.
    pub rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub reproducible: bool,
}

impl SpreadMonteCarloEngine {
    /// Creates a spread Monte Carlo engine.
    pub fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            num_paths,
            seed,
            antithetic: true,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
        }
    }

    /// Enables/disables antithetic variates.
    pub fn with_antithetic(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
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

impl PricingEngine<SpreadOption> for SpreadMonteCarloEngine {
    fn price(
        &self,
        instrument: &SpreadOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "spread MC num_paths must be > 0".to_string(),
            ));
        }
        if instrument.t <= 0.0 {
            return Ok(PricingResult {
                price: (instrument.s1 - instrument.s2 - instrument.k).max(0.0),
                stderr: Some(0.0),
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let sqrt_t = instrument.t.sqrt();
        let corr_tail = (1.0 - instrument.rho * instrument.rho).max(0.0).sqrt();
        let drift1 =
            (instrument.r - instrument.q1 - 0.5 * instrument.vol1 * instrument.vol1) * instrument.t;
        let drift2 =
            (instrument.r - instrument.q2 - 0.5 * instrument.vol2 * instrument.vol2) * instrument.t;
        let vol_term1 = instrument.vol1 * sqrt_t;
        let vol_term2 = instrument.vol2 * sqrt_t;
        let discount = (-instrument.r * instrument.t).exp();

        let samples = if self.antithetic {
            self.num_paths.div_ceil(2)
        } else {
            self.num_paths
        };
        let rng_kind = self.rng_kind;
        let reproducible = self.reproducible;
        let base_seed = self.seed;

        let simulate_sample = |i: usize| {
            let seed = resolve_stream_seed(base_seed, i, reproducible);
            let mut rng = FastRng::from_seed(rng_kind, seed);
            let z1 = sample_standard_normal(&mut rng);
            let z2 = sample_standard_normal(&mut rng);

            let payoff = terminal_spread_payoff(
                instrument, z1, z2, corr_tail, drift1, drift2, vol_term1, vol_term2,
            );

            if self.antithetic {
                let anti = terminal_spread_payoff(
                    instrument, -z1, -z2, corr_tail, drift1, drift2, vol_term1, vol_term2,
                );
                0.5 * (payoff + anti)
            } else {
                payoff
            }
        };

        #[cfg(feature = "parallel")]
        let payoffs = (0..samples)
            .into_par_iter()
            .map(simulate_sample)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let payoffs = (0..samples).map(simulate_sample).collect::<Vec<_>>();

        let n = payoffs.len() as f64;
        let mean = payoffs.iter().sum::<f64>() / n;
        let variance = if payoffs.len() > 1 {
            payoffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_paths", self.num_paths as f64);
        diagnostics.insert("rho", instrument.rho);

        Ok(PricingResult {
            price: discount * mean,
            stderr: Some(discount * (variance / n).sqrt()),
            greeks: None,
            diagnostics,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn terminal_spread_payoff(
    option: &SpreadOption,
    z1: f64,
    z2: f64,
    corr_tail: f64,
    drift1: f64,
    drift2: f64,
    vol_term1: f64,
    vol_term2: f64,
) -> f64 {
    let w1 = z1;
    let w2 = option.rho * z1 + corr_tail * z2;

    let s1_t = option.s1 * (drift1 + vol_term1 * w1).exp();
    let s2_t = option.s2 * (drift2 + vol_term2 * w2).exp();

    (s1_t - s2_t - option.k).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PricingEngine;
    use crate::engines::analytic::kirk_spread_price;

    #[test]
    fn spread_mc_matches_kirk_within_one_percent_for_near_zero_strike_case() {
        let option = SpreadOption {
            s1: 100.0,
            s2: 96.0,
            k: 3.0,
            vol1: 0.20,
            vol2: 0.15,
            rho: 0.5,
            q1: 0.0,
            q2: 0.0,
            r: 0.05,
            t: 0.5,
        };
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();

        let mc = SpreadMonteCarloEngine::new(300_000, 13)
            .price(&option, &market)
            .unwrap()
            .price;
        let kirk = kirk_spread_price(&option).unwrap();
        let rel_err = ((mc - kirk) / kirk).abs();

        assert!(
            rel_err <= 0.01,
            "spread MC/Kirk mismatch exceeds 1%: mc={} kirk={} rel_err={}",
            mc,
            kirk,
            rel_err
        );
    }
}
