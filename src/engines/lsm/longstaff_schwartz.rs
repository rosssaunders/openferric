use nalgebra::{Matrix3, Vector3};

use crate::core::{
    BarrierDirection, BarrierStyle, ExerciseStyle, OptionType, PricingEngine, PricingError,
    PricingResult,
};
use crate::instruments::{BarrierOption, VanillaOption};
use crate::market::Market;
use crate::math::fast_norm::beasley_springer_moro_inv_cdf;
use crate::math::fast_rng::{uniform_open01, Xoshiro256PlusPlus};

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
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &v in values {
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let var = if values.len() > 1 {
        (sum_sq - sum * sum / n) / (n - 1.0)
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

        // Flat 2D array for paths: paths[path_idx * stride + step]
        let stride = self.num_steps + 1;
        let mut paths = vec![0.0_f64; self.num_paths * stride];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        for pi in 0..self.num_paths {
            let base = pi * stride;
            paths[base] = market.spot;
            for ti in 1..=self.num_steps {
                let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                paths[base + ti] = paths[base + ti - 1] * step_vol.mul_add(z, drift).exp();
            }
        }

        let mut values: Vec<f64> = (0..self.num_paths)
            .map(|pi| intrinsic(instrument.option_type, paths[pi * stride + self.num_steps], instrument.strike))
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

            let itm: Vec<usize> = (0..self.num_paths)
                .filter(|&idx| {
                    intrinsic(instrument.option_type, paths[idx * stride + ti], instrument.strike) > 0.0
                })
                .collect();

            if itm.len() < 3 {
                continue;
            }

            // Fixed-size 3x3 regression using nalgebra Matrix3/Vector3 (no heap allocation)
            let mut xtx = Matrix3::<f64>::zeros();
            let mut xty = Vector3::<f64>::zeros();
            for &idx in &itm {
                let s = paths[idx * stride + ti];
                let s2 = s * s;
                let y = values[idx];
                let row = Vector3::new(1.0, s, s2);
                for r in 0..3 {
                    for c in 0..3 {
                        xtx[(r, c)] += row[r] * row[c];
                    }
                    xty[r] += row[r] * y;
                }
            }
            let beta = xtx.lu().solve(&xty).unwrap_or(Vector3::zeros());

            for idx in itm {
                let s = paths[idx * stride + ti];
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

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut pv = Vec::with_capacity(self.num_paths);
        for _ in 0..self.num_paths {
            let mut path = Vec::with_capacity(self.num_steps + 1);
            let mut s = market.spot;
            path.push(s);
            for _ in 0..self.num_steps {
                let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                s *= step_vol.mul_add(z, drift).exp();
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
