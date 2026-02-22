//! Least-squares Monte Carlo components for Longstaff Schwartz.
//!
//! These routines support regression-based early exercise valuation.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::core::{
    BarrierDirection, BarrierStyle, ExerciseStyle, OptionType, PricingEngine, PricingError,
    PricingResult,
};
use crate::instruments::{BarrierOption, VanillaOption};
use crate::market::Market;

/// Longstaff-Schwartz least-squares Monte Carlo engine.
#[derive(Debug, Clone)]
pub struct LongstaffSchwartzEngine {
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// RNG seed.
    pub seed: u64,
}

impl LongstaffSchwartzEngine {
    /// Creates a Longstaff-Schwartz engine.
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
        }
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

fn path_hits_barrier(path: &[f64], level: f64, direction: BarrierDirection) -> bool {
    match direction {
        BarrierDirection::Up => path.iter().any(|&s| s >= level),
        BarrierDirection::Down => path.iter().any(|&s| s <= level),
    }
}

fn mean_and_stderr(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = if values.len() > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, (var / n).sqrt())
}

impl PricingEngine<VanillaOption> for LongstaffSchwartzEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.num_steps < 2 {
            return Err(PricingError::InvalidInput(
                "num_steps must be >= 2 for Longstaff-Schwartz".to_string(),
            ));
        }
        if self.num_paths < 3 {
            return Err(PricingError::InvalidInput(
                "num_paths must be >= 3 for Longstaff-Schwartz".to_string(),
            ));
        }

        if instrument.expiry == 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                stderr: Some(0.0),
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let dt = instrument.expiry / self.num_steps as f64;
        let drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * dt;
        let step_vol = vol * dt.sqrt();
        let disc = (-market.rate * dt).exp();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut paths = vec![vec![0.0_f64; self.num_steps + 1]; self.num_paths];
        for path in &mut paths {
            path[0] = market.spot;
            for ti in 1..=self.num_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                path[ti] = path[ti - 1] * (drift + step_vol * z).exp();
            }
        }

        let mut values: Vec<f64> = paths
            .iter()
            .map(|p| intrinsic(instrument.option_type, p[self.num_steps], instrument.strike))
            .collect();

        let mut can_exercise = vec![false; self.num_steps + 1];
        match &instrument.exercise {
            ExerciseStyle::European => {
                can_exercise[self.num_steps] = true;
            }
            ExerciseStyle::American => {
                for flag in can_exercise.iter_mut().take(self.num_steps).skip(1) {
                    *flag = true;
                }
                can_exercise[self.num_steps] = true;
            }
            ExerciseStyle::Bermudan { dates } => {
                for &date in dates {
                    let idx = ((date / instrument.expiry) * self.num_steps as f64).round() as usize;
                    can_exercise[idx.min(self.num_steps)] = true;
                }
                can_exercise[self.num_steps] = true;
            }
        }

        for ti in (1..self.num_steps).rev() {
            for value in &mut values {
                *value *= disc;
            }

            if !can_exercise[ti] {
                continue;
            }

            let itm: Vec<usize> = paths
                .iter()
                .enumerate()
                .filter_map(|(idx, path)| {
                    (intrinsic(instrument.option_type, path[ti], instrument.strike) > 0.0)
                        .then_some(idx)
                })
                .collect();

            if itm.len() < 3 {
                continue;
            }

            let mut x = DMatrix::<f64>::zeros(itm.len(), 3);
            let mut y = DVector::<f64>::zeros(itm.len());
            for (row, idx) in itm.iter().copied().enumerate() {
                let s = paths[idx][ti];
                x[(row, 0)] = 1.0;
                x[(row, 1)] = s;
                x[(row, 2)] = s * s;
                y[row] = values[idx];
            }

            let xtx = x.transpose() * &x;
            let xty = x.transpose() * &y;
            let beta = xtx
                .lu()
                .solve(&xty)
                .unwrap_or_else(|| DVector::<f64>::zeros(3));

            for idx in itm {
                let s = paths[idx][ti];
                let continuation = beta[0] + beta[1] * s + beta[2] * s * s;
                let exercise = intrinsic(instrument.option_type, s, instrument.strike);
                if exercise > continuation {
                    values[idx] = exercise;
                }
            }
        }

        let discounted: Vec<f64> = values.into_iter().map(|v| v * disc).collect();
        let (price, stderr) = mean_and_stderr(&discounted);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_paths", self.num_paths as f64);
        diagnostics.insert("num_steps", self.num_steps as f64);
        diagnostics.insert("vol", vol);

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<BarrierOption> for LongstaffSchwartzEngine {
    fn price(
        &self,
        instrument: &BarrierOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.num_steps == 0 {
            return Err(PricingError::InvalidInput(
                "num_steps must be > 0".to_string(),
            ));
        }
        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let dt = instrument.expiry / self.num_steps as f64;
        let drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * dt;
        let step_vol = vol * dt.sqrt();
        let discount = (-market.rate * instrument.expiry).exp();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut pv = Vec::with_capacity(self.num_paths);
        for _ in 0..self.num_paths {
            let mut path = Vec::with_capacity(self.num_steps + 1);
            let mut s = market.spot;
            path.push(s);
            for _ in 0..self.num_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                s *= (drift + step_vol * z).exp();
                path.push(s);
            }

            let hit = path_hits_barrier(
                &path,
                instrument.barrier.level,
                instrument.barrier.direction,
            );
            let active = match instrument.barrier.style {
                BarrierStyle::In => hit,
                BarrierStyle::Out => !hit,
            };
            let payoff = if active {
                intrinsic(
                    instrument.option_type,
                    path[path.len() - 1],
                    instrument.strike,
                )
            } else {
                instrument.barrier.rebate
            };
            pv.push(discount * payoff);
        }

        let (price, stderr) = mean_and_stderr(&pv);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_paths", self.num_paths as f64);
        diagnostics.insert("num_steps", self.num_steps as f64);
        diagnostics.insert("vol", vol);

        Ok(PricingResult {
            price,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }
}
