//! Module `engines::lsm::longstaff_schwartz`.
//!
//! Implements longstaff schwartz abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `LongstaffSchwartzEngine` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
use nalgebra::{Matrix3, Vector3};

use crate::core::{
    BarrierDirection, BarrierStyle, ExerciseStyle, OptionType, PricingEngine, PricingError,
    PricingResult,
};
use crate::instruments::{BarrierOption, BermudanOption, VanillaOption};
use crate::market::Market;
use crate::math::fast_norm::beasley_springer_moro_inv_cdf;
use crate::math::fast_rng::{Xoshiro256PlusPlus, uniform_open01};
use crate::models::Heston;

/// Dynamics used by the Bermudan LSM path simulation.
#[derive(Debug, Clone, Copy)]
pub enum LsmDynamics {
    /// Geometric Brownian motion with a single implied volatility.
    Gbm,
    /// Log-Euler simulation with state/time-dependent volatility from `market.vol_for(S, t)`.
    LocalVolEuler,
    /// Heston stochastic-volatility Euler scheme (full truncation on variance).
    HestonEuler {
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        v0: f64,
    },
}

/// Exercise-boundary point at one Bermudan decision time.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExerciseBoundaryPoint {
    /// Exercise date in year fractions.
    pub time: f64,
    /// Strike used at this exercise date.
    pub strike: f64,
    /// Estimated optimal boundary (`S*`); `None` when no path exercised.
    pub boundary_spot: Option<f64>,
    /// Number of in-the-money paths used for regression.
    pub itm_paths: usize,
    /// Number of paths that exercised under the policy.
    pub exercised_paths: usize,
}

/// Bermudan LSM output including price and exercise-boundary diagnostics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BermudanLsmOutput {
    /// Standard engine result payload.
    pub result: PricingResult,
    /// Exercise boundary across all decision dates in chronological order.
    pub exercise_boundary: Vec<ExerciseBoundaryPoint>,
}

/// Longstaff-Schwartz least-squares Monte Carlo engine.
#[derive(Debug, Clone)]
pub struct LongstaffSchwartzEngine {
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Dynamics used for Bermudan path simulation.
    pub dynamics: LsmDynamics,
}

impl LongstaffSchwartzEngine {
    /// Creates a Longstaff-Schwartz engine.
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            dynamics: LsmDynamics::Gbm,
        }
    }

    /// Uses local-vol Euler dynamics for Bermudan pricing.
    pub fn with_local_vol_dynamics(mut self) -> Self {
        self.dynamics = LsmDynamics::LocalVolEuler;
        self
    }

    /// Uses Heston Euler dynamics for Bermudan pricing.
    ///
    /// The spot drift is set to risk-neutral drift `r-q` from `Market`; the
    /// `mu` field of `model` is ignored.
    pub fn with_heston_dynamics(mut self, model: Heston) -> Self {
        self.dynamics = LsmDynamics::HestonEuler {
            kappa: model.kappa,
            theta: model.theta,
            xi: model.xi,
            rho: model.rho,
            v0: model.v0,
        };
        self
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

#[inline]
fn regression_beta(
    itm: &[usize],
    paths: &[f64],
    stride: usize,
    step: usize,
    values: &[f64],
) -> Vector3<f64> {
    let mut s1 = 0.0_f64;
    let mut s_s = 0.0_f64;
    let mut s_s2 = 0.0_f64;
    let mut s_s3 = 0.0_f64;
    let mut s_s4 = 0.0_f64;
    let mut s_y = 0.0_f64;
    let mut s_sy = 0.0_f64;
    let mut s_s2y = 0.0_f64;

    for &idx in itm {
        let s = paths[idx * stride + step];
        let s2 = s * s;
        let y = values[idx];
        s1 += 1.0;
        s_s += s;
        s_s2 += s2;
        s_s3 += s2 * s;
        s_s4 += s2 * s2;
        s_y += y;
        s_sy += s * y;
        s_s2y += s2 * y;
    }

    let xtx = Matrix3::new(s1, s_s, s_s2, s_s, s_s2, s_s3, s_s2, s_s3, s_s4);
    let xty = Vector3::new(s_y, s_sy, s_s2y);
    xtx.lu().solve(&xty).unwrap_or(Vector3::zeros())
}

#[inline]
fn boundary_from_exercised(option_type: OptionType, exercised_spots: &[f64]) -> Option<f64> {
    if exercised_spots.is_empty() {
        return None;
    }
    match option_type {
        OptionType::Put => exercised_spots.iter().copied().reduce(f64::max),
        OptionType::Call => exercised_spots.iter().copied().reduce(f64::min),
    }
}

impl LongstaffSchwartzEngine {
    fn simulate_bermudan_paths(
        &self,
        instrument: &BermudanOption,
        market: &Market,
        reference_strike: f64,
    ) -> Result<(Vec<f64>, usize), PricingError> {
        let dt = instrument.expiry / self.num_steps as f64;
        let sqrt_dt = dt.sqrt();
        let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);
        let drift_rn = market.rate - effective_dividend_yield;
        let raw_stride = self.num_steps + 1;
        let stride = (raw_stride + 7) & !7;
        let mut paths = vec![0.0_f64; self.num_paths * stride];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);

        match self.dynamics {
            LsmDynamics::Gbm => {
                let vol = market.vol_for(reference_strike, instrument.expiry);
                if !vol.is_finite() || vol <= 0.0 {
                    return Err(PricingError::InvalidInput(
                        "market volatility must be finite and > 0".to_string(),
                    ));
                }
                let drift = (drift_rn - 0.5 * vol * vol) * dt;
                let step_vol = vol * sqrt_dt;
                for pi in 0..self.num_paths {
                    let base = pi * stride;
                    paths[base] = market.spot;
                    for ti in 1..=self.num_steps {
                        let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                        paths[base + ti] = paths[base + ti - 1] * step_vol.mul_add(z, drift).exp();
                    }
                }
            }
            LsmDynamics::LocalVolEuler => {
                for pi in 0..self.num_paths {
                    let base = pi * stride;
                    let mut s = market.spot;
                    paths[base] = s;
                    for ti in 1..=self.num_steps {
                        let t = (ti as f64 * dt).max(1.0e-8);
                        let sigma = market.vol_for(s.max(1.0e-8), t);
                        if !sigma.is_finite() || sigma <= 0.0 {
                            return Err(PricingError::InvalidInput(
                                "local volatility surface returned non-positive value".to_string(),
                            ));
                        }
                        let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                        let drift = (drift_rn - 0.5 * sigma * sigma) * dt;
                        s *= (drift + sigma * sqrt_dt * z).exp();
                        paths[base + ti] = s.max(1.0e-12);
                        s = paths[base + ti];
                    }
                }
            }
            LsmDynamics::HestonEuler {
                kappa,
                theta,
                xi,
                rho,
                v0,
            } => {
                let heston = Heston {
                    mu: drift_rn,
                    kappa,
                    theta,
                    xi,
                    rho,
                    v0,
                };
                if !heston.validate() {
                    return Err(PricingError::InvalidInput(
                        "invalid Heston parameters for Bermudan LSM dynamics".to_string(),
                    ));
                }
                for pi in 0..self.num_paths {
                    let base = pi * stride;
                    let mut s = market.spot;
                    let mut v = v0;
                    paths[base] = s;
                    for ti in 1..=self.num_steps {
                        let z1 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                        let z2 = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                        let (s_next, v_next) = heston.step_euler(s, v, dt, z1, z2);
                        s = s_next.max(1.0e-12);
                        v = v_next.max(0.0);
                        paths[base + ti] = s;
                    }
                }
            }
        }

        Ok((paths, stride))
    }

    /// Prices a Bermudan option and returns the estimated optimal exercise boundary.
    ///
    /// Boundary extraction is pathwise:
    /// - put: largest exercised spot at each decision date,
    /// - call: smallest exercised spot at each decision date.
    ///
    /// References:
    /// - Longstaff and Schwartz (2001), least-squares continuation regression.
    /// - Glasserman (2004), Monte Carlo implementation details.
    pub fn price_bermudan_with_boundary(
        &self,
        instrument: &BermudanOption,
        market: &Market,
    ) -> Result<BermudanLsmOutput, PricingError> {
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

        let schedule = instrument.effective_schedule()?;
        let terminal_strike = schedule.last().map(|(_, k)| *k).ok_or_else(|| {
            PricingError::InvalidInput("bermudan schedule cannot be empty".to_string())
        })?;
        let (paths, stride) = self.simulate_bermudan_paths(instrument, market, terminal_strike)?;

        let dt = instrument.expiry / self.num_steps as f64;
        let disc = (-market.rate * dt).exp();
        let mut step_schedule = vec![None::<(f64, f64)>; self.num_steps + 1];
        for &(t, k) in &schedule {
            let idx = (((t / instrument.expiry) * self.num_steps as f64).round() as usize)
                .clamp(1, self.num_steps);
            step_schedule[idx] = Some((t, k));
        }

        let mut values: Vec<f64> = (0..self.num_paths)
            .map(|pi| {
                intrinsic(
                    instrument.option_type,
                    paths[pi * stride + self.num_steps],
                    terminal_strike,
                )
            })
            .collect();

        let terminal_itm = values.iter().filter(|v| **v > 0.0).count();
        let mut boundary_rev = vec![ExerciseBoundaryPoint {
            time: instrument.expiry,
            strike: terminal_strike,
            boundary_spot: Some(terminal_strike),
            itm_paths: terminal_itm,
            exercised_paths: terminal_itm,
        }];

        for ti in (1..self.num_steps).rev() {
            for value in &mut values {
                *value *= disc;
            }

            let Some((time, strike)) = step_schedule[ti] else {
                continue;
            };

            let itm: Vec<usize> = (0..self.num_paths)
                .filter(|&idx| {
                    intrinsic(instrument.option_type, paths[idx * stride + ti], strike) > 0.0
                })
                .collect();

            if itm.len() < 3 {
                boundary_rev.push(ExerciseBoundaryPoint {
                    time,
                    strike,
                    boundary_spot: None,
                    itm_paths: itm.len(),
                    exercised_paths: 0,
                });
                continue;
            }

            let beta = regression_beta(&itm, &paths, stride, ti, &values);
            let mut exercised_spots = Vec::with_capacity(itm.len());
            for idx in itm.iter().copied() {
                let s = paths[idx * stride + ti];
                let continuation = beta[0] + beta[1] * s + beta[2] * s * s;
                let exercise = intrinsic(instrument.option_type, s, strike);
                if exercise > continuation {
                    values[idx] = exercise;
                    exercised_spots.push(s);
                }
            }

            boundary_rev.push(ExerciseBoundaryPoint {
                time,
                strike,
                boundary_spot: boundary_from_exercised(instrument.option_type, &exercised_spots),
                itm_paths: itm.len(),
                exercised_paths: exercised_spots.len(),
            });
        }

        let discounted: Vec<f64> = values.into_iter().map(|v| v * disc).collect();
        let (price, stderr) = mean_and_stderr(&discounted);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::NumPaths, self.num_paths as f64);
        diagnostics.insert_key(crate::core::DiagKey::NumSteps, self.num_steps as f64);
        diagnostics.insert_key(crate::core::DiagKey::ExerciseDates, schedule.len() as f64);

        if let LsmDynamics::Gbm = self.dynamics {
            diagnostics.insert_key(
                crate::core::DiagKey::Vol,
                market.vol_for(terminal_strike, instrument.expiry),
            );
        }

        boundary_rev.reverse();
        Ok(BermudanLsmOutput {
            result: PricingResult {
                price,
                stderr: Some(stderr),
                greeks: None,
                diagnostics,
            },
            exercise_boundary: boundary_rev,
        })
    }
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
        let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);
        let drift = (market.rate - effective_dividend_yield - 0.5 * vol * vol) * dt;
        let step_vol = vol * dt.sqrt();
        let disc = (-market.rate * dt).exp();

        // Flat 2D array for paths: paths[path_idx * stride + step]
        // Pad stride to multiple of 8 (64 bytes / cache line) for better memory access patterns.
        let raw_stride = self.num_steps + 1;
        let stride = (raw_stride + 7) & !7;
        let mut paths = vec![0.0_f64; self.num_paths * stride];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);

        // Use batch SIMD inverse CDF when available.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let buf_size = (self.num_steps + 3) & !3;
                let mut normal_buf = vec![0.0_f64; buf_size];
                for pi in 0..self.num_paths {
                    let base = pi * stride;
                    paths[base] = market.spot;
                    unsafe {
                        crate::math::simd_math::fill_normals_simd(
                            &mut rng,
                            &mut normal_buf[..self.num_steps],
                        );
                    }
                    for ti in 1..=self.num_steps {
                        let z = normal_buf[ti - 1];
                        paths[base + ti] = paths[base + ti - 1] * step_vol.mul_add(z, drift).exp();
                    }
                }
            } else {
                for pi in 0..self.num_paths {
                    let base = pi * stride;
                    paths[base] = market.spot;
                    for ti in 1..=self.num_steps {
                        let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                        paths[base + ti] = paths[base + ti - 1] * step_vol.mul_add(z, drift).exp();
                    }
                }
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            for pi in 0..self.num_paths {
                let base = pi * stride;
                paths[base] = market.spot;
                for ti in 1..=self.num_steps {
                    let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                    paths[base + ti] = paths[base + ti - 1] * step_vol.mul_add(z, drift).exp();
                }
            }
        }

        let mut values: Vec<f64> = (0..self.num_paths)
            .map(|pi| {
                intrinsic(
                    instrument.option_type,
                    paths[pi * stride + self.num_steps],
                    instrument.strike,
                )
            })
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
                    intrinsic(
                        instrument.option_type,
                        paths[idx * stride + ti],
                        instrument.strike,
                    ) > 0.0
                })
                .collect();

            if itm.len() < 3 {
                continue;
            }

            // Inlined 3x3 symmetric normal equations (avoids nalgebra iterator overhead).
            // XtX is symmetric so we only compute the upper triangle + diagonal.
            let mut s1 = 0.0_f64;
            let mut s_s = 0.0_f64;
            let mut s_s2 = 0.0_f64;
            let mut s_s3 = 0.0_f64;
            let mut s_s4 = 0.0_f64;
            let mut s_y = 0.0_f64;
            let mut s_sy = 0.0_f64;
            let mut s_s2y = 0.0_f64;
            let n_itm = itm.len() as f64;
            for &idx in &itm {
                let s = paths[idx * stride + ti];
                let s2 = s * s;
                let y = values[idx];
                s1 += 1.0;
                s_s += s;
                s_s2 += s2;
                s_s3 += s2 * s;
                s_s4 += s2 * s2;
                s_y += y;
                s_sy += s * y;
                s_s2y += s2 * y;
            }
            let _ = n_itm;
            let xtx = Matrix3::new(s1, s_s, s_s2, s_s, s_s2, s_s3, s_s2, s_s3, s_s4);
            let xty = Vector3::new(s_y, s_sy, s_s2y);
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

impl PricingEngine<BermudanOption> for LongstaffSchwartzEngine {
    fn price(
        &self,
        instrument: &BermudanOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        self.price_bermudan_with_boundary(instrument, market)
            .map(|out| out.result)
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
        let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);
        let drift = (market.rate - effective_dividend_yield - 0.5 * vol * vol) * dt;
        let step_vol = vol * dt.sqrt();
        let discount = (-market.rate * instrument.expiry).exp();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut pv = Vec::with_capacity(self.num_paths);

        // Reuse a single path buffer instead of allocating per-path.
        let mut path = vec![0.0_f64; self.num_steps + 1];

        // Pre-allocate normal buffer for batch SIMD inverse CDF.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        let use_simd = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        let use_simd = false;

        let buf_size = (self.num_steps + 3) & !3;
        #[allow(unused_mut)]
        let mut normal_buf = vec![0.0_f64; buf_size];

        for _ in 0..self.num_paths {
            path[0] = market.spot;

            if use_simd {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                unsafe {
                    crate::math::simd_math::fill_normals_simd(
                        &mut rng,
                        &mut normal_buf[..self.num_steps],
                    );
                }
                for ti in 0..self.num_steps {
                    path[ti + 1] = path[ti] * step_vol.mul_add(normal_buf[ti], drift).exp();
                }
            } else {
                for ti in 0..self.num_steps {
                    let z = beasley_springer_moro_inv_cdf(uniform_open01(rng.next_f64()));
                    path[ti + 1] = path[ti] * step_vol.mul_add(z, drift).exp();
                }
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
