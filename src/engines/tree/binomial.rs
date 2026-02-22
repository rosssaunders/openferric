//! Module `engines::tree::binomial`.
//!
//! Implements binomial abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13, Cox-Ross-Rubinstein (1979), and backward-induction recursions around Eq. (13.10).
//!
//! Key types and purpose: `BinomialTreeEngine` define the core data contracts for this module.
//!
//! Numerical considerations: convergence is first- to second-order in time-step count depending on tree parameterization; deep ITM/OTM nodes may need larger depth.
//!
//! When to use: use trees for early-exercise intuition and lattice diagnostics; use analytic formulas for plain vanillas and Monte Carlo/PDE for richer dynamics.
use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Cox-Ross-Rubinstein binomial tree engine.
#[derive(Debug, Clone)]
pub struct BinomialTreeEngine {
    /// Number of tree steps.
    pub steps: usize,
}

impl BinomialTreeEngine {
    /// Creates a tree engine with the given number of steps.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

#[inline(always)]
fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

fn bermudan_exercise_steps(dates: &[f64], expiry: f64, steps: usize) -> Vec<bool> {
    let mut flags = vec![false; steps + 1];
    for &t in dates {
        if expiry <= 0.0 {
            continue;
        }
        let idx = ((t / expiry) * steps as f64).round() as usize;
        flags[idx.min(steps)] = true;
    }
    flags[steps] = true;
    flags
}

impl PricingEngine<VanillaOption> for BinomialTreeEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "binomial steps must be > 0".to_string(),
            ));
        }

        if instrument.expiry == 0.0 {
            return Ok(PricingResult {
                price: intrinsic(instrument.option_type, market.spot, instrument.strike),
                stderr: None,
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

        let dt = instrument.expiry / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);
        let growth = ((market.rate - effective_dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-market.rate * dt).exp();

        let bermudan_flags = match &instrument.exercise {
            ExerciseStyle::Bermudan { dates } => Some(bermudan_exercise_steps(
                dates,
                instrument.expiry,
                self.steps,
            )),
            _ => None,
        };

        // Fast path: for European / American we avoid the Vec lookup entirely.
        let is_american = matches!(instrument.exercise, ExerciseStyle::American);
        let is_bermudan = bermudan_flags.is_some();

        // Multiplicative recurrence replaces O(steps^2) powf() calls with multiplications.
        // spot * u^j * d^(steps-j) = spot * d^steps * (u/d)^j
        let ratio = u / d;
        let one_minus_p = 1.0 - p;

        // Pre-compute discounted probabilities to avoid repeated multiplications.
        let disc_p = disc * p;
        let disc_1mp = disc * one_minus_p;

        let mut values = vec![0.0_f64; self.steps + 1];
        {
            let mut st = market.spot * d.powi(self.steps as i32);
            for value in values.iter_mut().take(self.steps + 1) {
                *value = intrinsic(instrument.option_type, st, instrument.strike);
                st *= ratio;
            }
        }

        let mut base = market.spot * d.powi((self.steps - 1) as i32);
        for i in (0..self.steps).rev() {
            let can_exercise = if is_bermudan {
                // SAFETY: bermudan_flags is Some when is_bermudan is true.
                unsafe { bermudan_flags.as_ref().unwrap_unchecked()[i] }
            } else {
                is_american
            };

            if can_exercise {
                let mut st = base;
                for j in 0..=i {
                    let continuation = disc_p.mul_add(values[j + 1], disc_1mp * values[j]);
                    let exercise = intrinsic(instrument.option_type, st, instrument.strike);
                    values[j] = continuation.max(exercise);
                    st *= ratio;
                }
            } else {
                // SIMD backward induction when available.
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if i >= 3 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
                    {
                        // SAFETY: Guarded by runtime CPU feature detection.
                        unsafe { binomial_backward_avx2(&mut values, i, disc_p, disc_1mp) };
                    } else {
                        let mut j = 0;
                        while j <= i {
                            values[j] = disc_p.mul_add(values[j + 1], disc_1mp * values[j]);
                            j += 1;
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    let mut j = 0;
                    while j + 4 <= i + 1 {
                        values[j] = disc_p.mul_add(values[j + 1], disc_1mp * values[j]);
                        values[j + 1] = disc_p.mul_add(values[j + 2], disc_1mp * values[j + 1]);
                        values[j + 2] = disc_p.mul_add(values[j + 3], disc_1mp * values[j + 2]);
                        values[j + 3] = disc_p.mul_add(values[j + 4], disc_1mp * values[j + 3]);
                        j += 4;
                    }
                    while j <= i {
                        values[j] = disc_p.mul_add(values[j + 1], disc_1mp * values[j]);
                        j += 1;
                    }
                }
            }
            base *= u;
        }

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::NumSteps, self.steps as f64);
        diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

        Ok(PricingResult {
            price: values[0],
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

/// AVX2+FMA backward induction: `values[j] = disc_p * values[j+1] + disc_1mp * values[j]`
/// for j in 0..=step_index. Processes 4 nodes at a time with FMA.
///
/// Because each value[j] depends on value[j+1] (which was also just updated),
/// we process from the END backwards to avoid write-after-read hazards.
/// Actually — the dependency is: values[j] reads values[j+1] which has NOT yet
/// been overwritten (we iterate j=0..=i, so j+1 is ahead). With forward iteration,
/// values[j] reads the old values[j+1]. With SIMD we load 4 consecutive "up" values
/// (shifted by 1) and 4 "down" values, both from the OLD array, so there's no hazard
/// as long as we don't write ahead of our read window. We process sequentially in blocks
/// of 4 which is safe.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn binomial_backward_avx2(
    values: &mut [f64],
    step_index: usize,
    disc_p: f64,
    disc_1mp: f64,
) {
    use std::arch::x86_64::*;

    let dp = _mm256_set1_pd(disc_p);
    let d1mp = _mm256_set1_pd(disc_1mp);
    let n = step_index + 1; // number of nodes to update
    let mut j = 0usize;

    while j + 4 <= n {
        // values[j..j+4] = disc_1mp * values[j..j+4] + disc_p * values[j+1..j+5]
        unsafe {
            let v_down = _mm256_loadu_pd(values.as_ptr().add(j));
            let v_up = _mm256_loadu_pd(values.as_ptr().add(j + 1));
            let result = _mm256_fmadd_pd(dp, v_up, _mm256_mul_pd(d1mp, v_down));
            _mm256_storeu_pd(values.as_mut_ptr().add(j), result);
        }
        j += 4;
    }

    // Scalar remainder
    while j < n {
        values[j] = disc_p.mul_add(values[j + 1], disc_1mp * values[j]);
        j += 1;
    }
}
