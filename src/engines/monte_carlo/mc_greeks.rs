//! Module `engines::monte_carlo::mc_greeks`.
//!
//! Implements mc greeks abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Key types and purpose: `MonteCarloGreeksEngine` define the core data contracts for this module.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::{ExerciseStyle, Greeks, OptionType, PricingError};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};

/// Monte Carlo Greeks engine for European vanilla options under GBM.
#[derive(Debug, Clone)]
pub struct MonteCarloGreeksEngine {
    /// Number of simulated paths.
    pub num_paths: usize,
    /// RNG seed.
    pub seed: u64,
    /// Enables antithetic variates for lower estimator variance.
    pub antithetic: bool,
    /// Relative bump for tangent-style pathwise gamma central differencing.
    pub spot_bump_rel: f64,
    /// Pseudo-random number generator backend.
    pub rng_kind: FastRngKind,
    /// Reproducible stream splitting mode.
    pub reproducible: bool,
}

impl MonteCarloGreeksEngine {
    /// Creates a Monte Carlo Greeks engine.
    pub fn new(num_paths: usize, seed: u64) -> Self {
        Self {
            num_paths,
            seed,
            antithetic: true,
            spot_bump_rel: 1.0e-2,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
        }
    }

    /// Enables/disables antithetic variates.
    pub fn with_antithetic(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
        self
    }

    /// Sets the relative bump used in pathwise gamma estimation.
    pub fn with_spot_bump_rel(mut self, spot_bump_rel: f64) -> Self {
        self.spot_bump_rel = spot_bump_rel.max(1.0e-6);
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

    /// Estimates Greeks using a pathwise/tangent approach.
    ///
    /// Delta is estimated via the pathwise derivative.
    /// Gamma is estimated via a common-random-number central difference of pathwise delta.
    pub fn estimate_pathwise(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<Greeks, PricingError> {
        let estimate = self.run_estimators(instrument, market)?;
        Ok(Greeks {
            delta: estimate.pathwise_delta,
            gamma: estimate.pathwise_gamma,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        })
    }

    /// Estimates Greeks using a likelihood-ratio (score function) approach.
    ///
    /// Returns delta, gamma, and vega.
    pub fn estimate_likelihood_ratio(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<Greeks, PricingError> {
        let estimate = self.run_estimators(instrument, market)?;
        Ok(Greeks {
            delta: estimate.lr_delta,
            gamma: estimate.lr_gamma,
            vega: estimate.lr_vega,
            theta: 0.0,
            rho: 0.0,
        })
    }

    fn run_estimators(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<MonteCarloGreekEstimate, PricingError> {
        instrument.validate()?;
        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "MonteCarloGreeksEngine supports European exercise only".to_string(),
            ));
        }

        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }
        if self.spot_bump_rel <= 0.0 || !self.spot_bump_rel.is_finite() {
            return Err(PricingError::InvalidInput(
                "spot_bump_rel must be finite and > 0".to_string(),
            ));
        }

        if instrument.expiry <= 0.0 {
            return Ok(MonteCarloGreekEstimate::default());
        }

        let spot = market.spot;
        if spot <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market spot must be > 0".to_string(),
            ));
        }

        let strike = instrument.strike;
        let maturity = instrument.expiry;
        let vol = market.vol_for(strike, maturity);
        if vol <= 0.0 || !vol.is_finite() {
            return Err(PricingError::InvalidInput(
                "market volatility must be finite and > 0".to_string(),
            ));
        }

        let rate = market.rate;
        let dividend = market.dividend_yield;
        let drift = (rate - dividend - 0.5 * vol * vol) * maturity;
        let sig_sqrt_t = vol * maturity.sqrt();
        let discount = (-rate * maturity).exp();

        let h = (spot * self.spot_bump_rel).max(1.0e-6);
        let spot_up = spot + h;
        let spot_dn = (spot - h).max(1.0e-8);
        let spot_span = spot_up - spot_dn;

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
            let z = sample_standard_normal(&mut rng);
            let base = single_path_contribution(
                instrument.option_type,
                z,
                spot,
                strike,
                spot_up,
                spot_dn,
                spot_span,
                vol,
                maturity,
                drift,
                sig_sqrt_t,
            );

            if self.antithetic {
                let anti = single_path_contribution(
                    instrument.option_type,
                    -z,
                    spot,
                    strike,
                    spot_up,
                    spot_dn,
                    spot_span,
                    vol,
                    maturity,
                    drift,
                    sig_sqrt_t,
                );
                MonteCarloGreekEstimate {
                    pathwise_delta: 0.5 * (base.pathwise_delta + anti.pathwise_delta),
                    pathwise_gamma: 0.5 * (base.pathwise_gamma + anti.pathwise_gamma),
                    lr_delta: 0.5 * (base.lr_delta + anti.lr_delta),
                    lr_gamma: 0.5 * (base.lr_gamma + anti.lr_gamma),
                    lr_vega: 0.5 * (base.lr_vega + anti.lr_vega),
                }
            } else {
                base
            }
        };
        let add_estimates =
            |lhs: MonteCarloGreekEstimate, rhs: MonteCarloGreekEstimate| MonteCarloGreekEstimate {
                pathwise_delta: lhs.pathwise_delta + rhs.pathwise_delta,
                pathwise_gamma: lhs.pathwise_gamma + rhs.pathwise_gamma,
                lr_delta: lhs.lr_delta + rhs.lr_delta,
                lr_gamma: lhs.lr_gamma + rhs.lr_gamma,
                lr_vega: lhs.lr_vega + rhs.lr_vega,
            };

        #[cfg(feature = "parallel")]
        let sums = (0..samples)
            .into_par_iter()
            .map(simulate_sample)
            .reduce(MonteCarloGreekEstimate::default, add_estimates);
        #[cfg(not(feature = "parallel"))]
        let sums = (0..samples)
            .map(simulate_sample)
            .fold(MonteCarloGreekEstimate::default(), add_estimates);

        let n = samples as f64;
        Ok(MonteCarloGreekEstimate {
            pathwise_delta: discount * sums.pathwise_delta / n,
            pathwise_gamma: discount * sums.pathwise_gamma / n,
            lr_delta: discount * sums.lr_delta / n,
            lr_gamma: discount * sums.lr_gamma / n,
            lr_vega: discount * sums.lr_vega / n,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MonteCarloGreekEstimate {
    pathwise_delta: f64,
    pathwise_gamma: f64,
    lr_delta: f64,
    lr_gamma: f64,
    lr_vega: f64,
}

#[allow(clippy::too_many_arguments)]
fn single_path_contribution(
    option_type: OptionType,
    z: f64,
    spot: f64,
    strike: f64,
    spot_up: f64,
    spot_dn: f64,
    spot_span: f64,
    vol: f64,
    maturity: f64,
    drift: f64,
    sig_sqrt_t: f64,
) -> MonteCarloGreekEstimate {
    let growth = sig_sqrt_t.mul_add(z, drift).exp();
    let st = spot * growth;
    let payoff = payoff(option_type, st, strike);

    let delta_pw = pathwise_delta(option_type, st, strike, growth);
    let delta_up = pathwise_delta(option_type, spot_up * growth, strike, growth);
    let delta_dn = pathwise_delta(option_type, spot_dn * growth, strike, growth);
    let gamma_pw = (delta_up - delta_dn) / spot_span;

    // Pre-compute shared denominators to avoid redundant divisions.
    let sqrt_t = maturity.sqrt();
    let inv_spot_vol_sqrt_t = 1.0 / (spot * vol * sqrt_t);
    let z2 = z * z;
    let z2m1 = z2 - 1.0;

    let w_delta = z * inv_spot_vol_sqrt_t;
    let inv_spot2_vol2_t = inv_spot_vol_sqrt_t * inv_spot_vol_sqrt_t;
    let w_gamma = z2m1 * inv_spot2_vol2_t - z * inv_spot_vol_sqrt_t / spot;
    let w_vega = (z2m1 - vol * sqrt_t * z) / vol;

    MonteCarloGreekEstimate {
        pathwise_delta: delta_pw,
        pathwise_gamma: gamma_pw,
        lr_delta: payoff * w_delta,
        lr_gamma: payoff * w_gamma,
        lr_vega: payoff * w_vega,
    }
}

#[inline(always)]
fn payoff(option_type: OptionType, st: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (st - strike).max(0.0),
        OptionType::Put => (strike - st).max(0.0),
    }
}

#[inline(always)]
fn pathwise_delta(option_type: OptionType, st: f64, strike: f64, growth: f64) -> f64 {
    match option_type {
        OptionType::Call => {
            if st > strike {
                growth
            } else {
                0.0
            }
        }
        OptionType::Put => {
            if st < strike {
                -growth
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normal_cdf, normal_pdf};

    fn setup_case() -> (VanillaOption, Market) {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .flat_vol(0.20)
            .build()
            .unwrap();
        (option, market)
    }

    #[test]
    fn pathwise_delta_converges_to_closed_form_call_delta() {
        let (option, market) = setup_case();
        let engine = MonteCarloGreeksEngine::new(100_000, 42);
        let greeks = engine.estimate_pathwise(&option, &market).unwrap();

        let d1 = ((market.spot / option.strike).ln()
            + (market.rate - market.dividend_yield + 0.5 * 0.20_f64 * 0.20_f64) * option.expiry)
            / (0.20 * option.expiry.sqrt());
        let analytical_delta = (-market.dividend_yield * option.expiry).exp() * normal_cdf(d1);

        let rel_err = ((greeks.delta - analytical_delta) / analytical_delta).abs();
        assert!(
            rel_err <= 0.02,
            "pathwise delta error too high: mc={} cf={} rel_err={}",
            greeks.delta,
            analytical_delta,
            rel_err
        );
    }

    #[test]
    fn likelihood_ratio_vega_converges_to_closed_form_call_vega() {
        let (option, market) = setup_case();
        let engine = MonteCarloGreeksEngine::new(100_000, 42);
        let greeks = engine.estimate_likelihood_ratio(&option, &market).unwrap();

        let vol = 0.20;
        let d1 = ((market.spot / option.strike).ln()
            + (market.rate - market.dividend_yield + 0.5 * vol * vol) * option.expiry)
            / (vol * option.expiry.sqrt());
        let analytical_vega = market.spot
            * (-market.dividend_yield * option.expiry).exp()
            * normal_pdf(d1)
            * option.expiry.sqrt();

        let rel_err = ((greeks.vega - analytical_vega) / analytical_vega).abs();
        assert!(
            rel_err <= 0.05,
            "LR vega error too high: mc={} cf={} rel_err={}",
            greeks.vega,
            analytical_vega,
            rel_err
        );
    }

    #[test]
    fn quantlib_reference_values_are_reasonable() {
        let (option, market) = setup_case();
        let engine = MonteCarloGreeksEngine::new(100_000, 42);

        let pw = engine.estimate_pathwise(&option, &market).unwrap();
        let lr = engine.estimate_likelihood_ratio(&option, &market).unwrap();

        assert!((pw.delta - 0.6368).abs() / 0.6368 < 0.02);
        assert!((pw.gamma - 0.01876).abs() / 0.01876 < 0.20);
        assert!((lr.vega - 37.524).abs() / 37.524 < 0.05);
    }
}
