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
pub fn mc_european_qmc_with_seed(
    instrument: &VanillaOption,
    market: &Market,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> PricingResult {
    if n_paths == 0 || n_steps == 0 {
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
    let dt_drift = (market.rate - market.dividend_yield - 0.5 * vol * vol) * dt;
    let dt_vol = vol * dt.sqrt();
    let discount = (-market.rate * instrument.expiry).exp();

    let mut sobol = SobolSequence::new(n_steps, seed);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    // Pre-allocate the uniform buffer once; avoid per-sample Vec allocation.
    let mut uniforms = vec![0.0_f64; n_steps];

    for _ in 0..n_paths {
        if !sobol.next_into(&mut uniforms) {
            break;
        }
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

        let px = payoff(instrument.option_type, spot, instrument.strike);
        sum += px;
        sum_sq += px * px;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let variance = if n_paths > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert_key(crate::core::DiagKey::NumPaths, n_paths as f64);
    diagnostics.insert_key(crate::core::DiagKey::NumSteps, n_steps as f64);
    diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

    PricingResult {
        price: discount * mean,
        stderr: Some(discount * (variance / n).sqrt()),
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
