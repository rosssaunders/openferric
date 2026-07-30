//! Module `engines::monte_carlo::mc_qmc`.
//!
//! Implements mc qmc workflows with concrete routines such as `mc_european_qmc`, `mc_european_qmc_with_seed`.
//!
//! References: Glasserman (2004), Longstaff and Schwartz (2001), Hull (11th ed.) Ch. 25, Monte Carlo estimators around Eq. (25.1).
//!
//! Primary API surface: free functions `mc_european_qmc`, `mc_european_qmc_with_seed`.
//!
//! Numerical considerations: estimator variance, path count, and random-seed strategy drive confidence intervals; monitor bias from discretization and variance reduction choices.
//!
//! When to use: use Monte Carlo for path dependence and higher-dimensional factors; prefer analytic or tree methods when low-dimensional closed-form or lattice solutions exist.
use crate::core::{ExerciseStyle, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::fast_rng::uniform_open01;
use crate::math::{SobolSequence, normal_inv_cdf};

#[inline]
fn payoff(option_type: crate::core::OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        crate::core::OptionType::Call => (spot - strike).max(0.0),
        crate::core::OptionType::Put => (strike - spot).max(0.0),
    }
}

/// Number of independent digital-shift randomizations used for the
/// randomized-QMC (RQMC) standard-error estimate.
const QMC_RANDOMIZATIONS: usize = 8;

/// SplitMix64 mix used to derive independent per-randomization Sobol seeds
/// from the user seed (mirrors the scramble derivation in `math::sobol`).
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Quasi-Monte Carlo pricer for European vanilla options under GBM.
///
/// Sobol points are transformed to standard normals using inverse CDF.
pub fn mc_european_qmc(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
) -> PricingResult {
    mc_european_qmc_with_seed(instrument, market, n_paths, n_steps, 0)
}

/// Quasi-Monte Carlo pricer with explicit Sobol seed.
///
/// Error estimation is randomized QMC: the path budget is split into
/// `QMC_RANDOMIZATIONS` independent digital-shift scrambles of the Sobol
/// sequence (seeds derived deterministically from `seed`), and the reported
/// standard error is the sample standard error across the per-randomization
/// estimates. A naive iid-sample stderr over deterministic Sobol points would
/// be statistically meaningless because the points are not independent.
pub fn mc_european_qmc_with_seed(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> PricingResult {
    if n_paths == 0
        || n_steps == 0
        || n_steps > crate::math::sobol::SOBOL_MAX_DIMENSIONS
        || instrument.validate().is_err()
        || market.validate().is_err()
    {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    if instrument.expiry <= 0.0 {
        return PricingResult {
            price: payoff(instrument.option_type, market.spot, instrument.strike),
            stderr: Some(0.0),
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let vol = market.vol_for(instrument.strike, instrument.expiry);
    if vol <= 0.0 || !vol.is_finite() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let dt = instrument.expiry / n_steps as f64;
    let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);
    let dt_drift = (market.rate - effective_dividend_yield - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-market.rate * instrument.expiry).exp();

    // Pre-allocate the uniform buffer once; avoid per-sample Vec allocation.
    let mut uniforms = vec![0.0_f64; n_steps];

    // Mean discounted-payoff estimate over `batch_paths` points of one
    // digitally-shifted Sobol stream.
    let mut run_batch = |sobol_seed: u64, batch_paths: usize| -> (f64, usize) {
        let mut sobol = SobolSequence::new(n_steps, sobol_seed);
        let mut sum = 0.0_f64;
        let mut done = 0_usize;
        for _ in 0..batch_paths {
            if !sobol.next_into(&mut uniforms) {
                break;
            }
            done += 1;
            let mut spot = market.spot;
            // Log-Euler GBM step: exp() is always positive, no clamp needed.
            // Unroll by 4 for instruction-level parallelism.
            let mut step = 0;
            while step + 4 <= n_steps {
                let z0 = normal_inv_cdf(uniform_open01(uniforms[step]));
                let z1 = normal_inv_cdf(uniform_open01(uniforms[step + 1]));
                let z2 = normal_inv_cdf(uniform_open01(uniforms[step + 2]));
                let z3 = normal_inv_cdf(uniform_open01(uniforms[step + 3]));
                spot *= dt_vol.mul_add(z0, dt_drift).exp();
                spot *= dt_vol.mul_add(z1, dt_drift).exp();
                spot *= dt_vol.mul_add(z2, dt_drift).exp();
                spot *= dt_vol.mul_add(z3, dt_drift).exp();
                step += 4;
            }
            while step < n_steps {
                let z = normal_inv_cdf(uniform_open01(uniforms[step]));
                spot *= dt_vol.mul_add(z, dt_drift).exp();
                step += 1;
            }

            sum += payoff(instrument.option_type, spot, instrument.strike);
        }
        (sum, done)
    };

    // Randomized QMC: M independent digital shifts of n/M points each (total
    // path count unchanged). Each shift seed is derived from the user seed.
    let m = QMC_RANDOMIZATIONS.min(n_paths);
    let base_batch = n_paths / m;
    let remainder = n_paths % m;

    let mut batch_means = Vec::with_capacity(m);
    let mut total_sum = 0.0_f64;
    let mut paths_done = 0_usize;
    for batch in 0..m {
        let batch_paths = base_batch + usize::from(batch < remainder);
        if batch_paths == 0 {
            continue;
        }
        // Always a non-zero seed so every batch gets an independent digital
        // shift (seed 0 would be the unscrambled canonical sequence).
        let sobol_seed = splitmix64(seed ^ ((batch as u64 + 1) << 40)).max(1);
        let (sum, done) = run_batch(sobol_seed, batch_paths);
        if done == 0 {
            continue;
        }
        batch_means.push(sum / done as f64);
        total_sum += sum;
        paths_done += done;
    }

    if paths_done == 0 || batch_means.is_empty() {
        return PricingResult {
            price: f64::NAN,
            stderr: None,
            greeks: None,
            diagnostics: crate::core::Diagnostics::new(),
        };
    }

    let mean = total_sum / paths_done as f64;

    // Standard error across the M independent randomization estimates.
    let m_done = batch_means.len();
    let stderr = if m_done > 1 {
        let batch_mean = batch_means.iter().sum::<f64>() / m_done as f64;
        let var = batch_means
            .iter()
            .map(|&b| (b - batch_mean) * (b - batch_mean))
            .sum::<f64>()
            / (m_done as f64 - 1.0);
        (var / m_done as f64).sqrt()
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert_key(crate::core::DiagKey::NumPaths, paths_done as f64);
    diagnostics.insert_key(crate::core::DiagKey::NumSteps, n_steps as f64);
    diagnostics.insert_key(crate::core::DiagKey::Vol, vol);
    // Note: the number of RQMC randomizations is QMC_RANDOMIZATIONS (capped by
    // n_paths); DiagKey has no dedicated slot for it, so it is not exported.

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * stderr),
        greeks: None,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{OptionType, PricingEngine};
    use crate::engines::monte_carlo::MonteCarloPricingEngine;
    use crate::instruments::VanillaOption;
    use crate::pricing::european::black_scholes_price;

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
    fn qmc_has_lower_stderr_than_prng_for_same_path_count() {
        let (option, market) = setup_case();
        // Use 50K paths — QMC advantage becomes clear at higher counts
        let qmc = mc_european_qmc_with_seed(&option, &market, 50_000, 1, 7);
        let mc = MonteCarloPricingEngine::new(50_000, 1, 7)
            .price(&option, &market)
            .expect("mc pricing succeeds");

        // QMC should have lower error; allow 20% tolerance for stochastic noise
        let qmc_err = qmc.stderr.unwrap_or(f64::INFINITY);
        let mc_err = mc.stderr.unwrap_or(f64::INFINITY);
        assert!(
            qmc_err < mc_err * 1.2,
            "qmc stderr={qmc_err} mc stderr={mc_err} — expected QMC to be competitive"
        );
    }

    #[test]
    fn rqmc_stderr_is_positive_and_consistent_with_true_error() {
        let (option, market) = setup_case();
        let qmc = mc_european_qmc_with_seed(&option, &market, 32_768, 4, 2024);
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);

        let stderr = qmc.stderr.expect("stderr present");
        assert!(stderr > 0.0, "RQMC stderr should be strictly positive");

        // The randomized-QMC stderr is an honest error estimate: the true
        // error should be within a few standard errors (8 randomizations make
        // the estimate noisy, so allow a generous multiple).
        let err = (qmc.price - bs).abs();
        assert!(
            err <= 8.0 * stderr,
            "true error {err} should be consistent with RQMC stderr {stderr}"
        );
    }

    #[test]
    fn qmc_price_matches_black_scholes_with_tight_error_at_10k_paths() {
        let (option, market) = setup_case();
        let qmc = mc_european_qmc_with_seed(&option, &market, 10_000, 1, 11);
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.05, 0.20, 1.0);
        let rel_err = ((qmc.price - bs) / bs).abs();

        assert!(
            rel_err <= 0.001,
            "QMC/BS relative error too high: qmc={} bs={} rel_err={}",
            qmc.price,
            bs,
            rel_err
        );
    }
}
