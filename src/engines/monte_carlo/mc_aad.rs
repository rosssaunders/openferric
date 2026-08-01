//! Module `engines::monte_carlo::mc_aad`.
//!
//! Pathwise Monte Carlo AAD routines:
//! - reverse-mode AAD through GBM simulation for European vanilla Greeks
//! - forward-mode AAD through Heston Euler simulation (delta)

use crate::core::{ExerciseStyle, Greeks, OptionType, PricingError, PricingResult};
use crate::engines::monte_carlo::mc_engine::{MonteCarloPricingEngine, RunningMoments};
use crate::instruments::VanillaOption;
use crate::market::Market;
use crate::math::aad::{AadTape, Dual, TapeCheckpoint};
use crate::math::fast_rng::{FastRng, resolve_stream_seed, sample_standard_normal};
use crate::models::Heston;

#[inline]
fn gbm_single_path_reverse(
    tape: &mut AadTape,
    checkpoint: TapeCheckpoint,
    option_type: OptionType,
    strike: f64,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    normals: &[f64],
) -> (f64, [f64; 4]) {
    tape.rewind(checkpoint);

    let s0 = tape.variable(spot);
    let r = tape.variable(rate);
    let sigma = tape.variable(vol);
    let t = tape.variable(expiry);

    let q = tape.constant(dividend_yield);
    let k = tape.constant(strike);
    let steps = tape.constant(normals.len() as f64);
    let half = tape.constant(0.5);

    let dt = tape.div(t, steps);
    let sqrt_dt = tape.sqrt(dt);
    let sigma_sq = tape.mul(sigma, sigma);
    let half_sigma_sq = tape.mul(half, sigma_sq);
    let r_minus_q = tape.sub(r, q);
    let drift_inner = tape.sub(r_minus_q, half_sigma_sq);
    let drift = tape.mul(drift_inner, dt);
    let diffusion = tape.mul(sigma, sqrt_dt);

    let mut s = s0;
    for &z in normals {
        let z_var = tape.constant(z);
        let diff_term = tape.mul(diffusion, z_var);
        let expo = tape.add(drift, diff_term);
        let growth = tape.exp(expo);
        s = tape.mul(s, growth);
    }

    let intrinsic = match option_type {
        OptionType::Call => tape.sub(s, k),
        OptionType::Put => tape.sub(k, s),
    };
    let payoff = tape.positive_part(intrinsic);

    let rt = tape.mul(r, t);
    let minus_rt = tape.neg(rt);
    let discount = tape.exp(minus_rt);
    let pv = tape.mul(discount, payoff);

    let mut grads = [0.0; 4];
    tape.gradient(pv, &[s0, r, sigma, t], &mut grads);
    (tape.value(pv), grads)
}

/// Pathwise AAD Greeks for European vanilla under GBM simulation.
pub fn mc_european_pathwise_aad(
    engine: &MonteCarloPricingEngine,
    instrument: &VanillaOption,
    market: &Market,
) -> Result<PricingResult, PricingError> {
    instrument.validate()?;
    market.validate()?;
    if !matches!(instrument.exercise, ExerciseStyle::European) {
        return Err(PricingError::InvalidInput(
            "mc pathwise AAD supports European exercise only".to_string(),
        ));
    }
    if engine.num_paths == 0 {
        return Err(PricingError::InvalidInput(
            "num_paths must be > 0".to_string(),
        ));
    }
    if engine.num_steps == 0 {
        return Err(PricingError::InvalidInput(
            "num_steps must be > 0".to_string(),
        ));
    }
    if instrument.expiry <= 0.0 {
        return Ok(PricingResult {
            price: match instrument.option_type {
                OptionType::Call => (market.spot - instrument.strike).max(0.0),
                OptionType::Put => (instrument.strike - market.spot).max(0.0),
            },
            stderr: Some(0.0),
            greeks: Some(Greeks {
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            }),
            diagnostics: crate::core::Diagnostics::new(),
        });
    }

    let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;

    let antithetic = matches!(
        engine.variance_reduction,
        super::mc_engine::VarianceReduction::Antithetic
    );
    let samples = if antithetic {
        engine.num_paths.div_ceil(2)
    } else {
        engine.num_paths
    };

    let mut tape = AadTape::with_capacity(engine.num_steps * 8 + 64);
    let cp0 = tape.checkpoint();

    let mut normals = vec![0.0_f64; engine.num_steps];
    let mut price_stats = RunningMoments::default();
    let mut grad_sum = [0.0_f64; 4];
    let effective_dividend_yield = market.effective_dividend_yield(instrument.expiry);

    for i in 0..samples {
        let seed = resolve_stream_seed(engine.seed, i, engine.reproducible);
        let mut rng = FastRng::from_seed(engine.rng_kind, seed);
        for z in &mut normals {
            *z = sample_standard_normal(&mut rng);
        }

        let (pv0, g0) = gbm_single_path_reverse(
            &mut tape,
            cp0,
            instrument.option_type,
            instrument.strike,
            market.spot,
            market.rate,
            effective_dividend_yield,
            vol,
            instrument.expiry,
            &normals,
        );

        let (pv, grads) = if antithetic {
            for z in &mut normals {
                *z = -*z;
            }
            let (pv1, g1) = gbm_single_path_reverse(
                &mut tape,
                cp0,
                instrument.option_type,
                instrument.strike,
                market.spot,
                market.rate,
                effective_dividend_yield,
                vol,
                instrument.expiry,
                &normals,
            );

            let mut avg_g = [0.0; 4];
            for j in 0..4 {
                avg_g[j] = 0.5 * (g0[j] + g1[j]);
            }
            (0.5 * (pv0 + pv1), avg_g)
        } else {
            (pv0, g0)
        };

        price_stats.record(pv);
        for j in 0..4 {
            grad_sum[j] += grads[j];
        }
    }

    let n = samples as f64;
    let mean = price_stats.mean();
    let variance = price_stats.sample_variance();
    let stderr = (variance / n).sqrt();

    let mut diagnostics = crate::core::Diagnostics::new();
    diagnostics.insert_key(crate::core::DiagKey::NumPaths, engine.num_paths as f64);
    diagnostics.insert_key(crate::core::DiagKey::NumSteps, engine.num_steps as f64);
    diagnostics.insert_key(crate::core::DiagKey::Vol, vol);

    Ok(PricingResult {
        price: mean,
        stderr: Some(stderr),
        greeks: Some(Greeks {
            delta: grad_sum[0] / n,
            gamma: 0.0,
            vega: grad_sum[2] / n,
            theta: -(grad_sum[3] / n),
            rho: grad_sum[1] / n,
        }),
        diagnostics,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct HestonAadConfig {
    pub num_paths: usize,
    pub num_steps: usize,
    pub seed: u64,
}

impl HestonAadConfig {
    #[inline]
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
        }
    }
}

#[inline]
fn heston_single_path_price_delta(
    option_type: OptionType,
    strike: f64,
    maturity: f64,
    spot: f64,
    rate: f64,
    model: Heston,
    normals1: &[f64],
    normals2: &[f64],
) -> (f64, f64) {
    let dt = maturity / normals1.len() as f64;
    let sqrt_dt = dt.sqrt();
    let rho2 = (1.0 - model.rho * model.rho).sqrt();
    let discount = (-rate * maturity).exp();

    let mut s = Dual::variable(spot);
    let mut v = model.v0;

    for (&z1, &z2) in normals1.iter().zip(normals2.iter()) {
        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();
        let zs = model.rho * z1 + rho2 * z2;
        let v_next =
            (v + model.kappa * (model.theta - v_pos) * dt + model.xi * sqrt_v * sqrt_dt * z1)
                .max(0.0);
        let growth = ((model.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * zs).exp();
        s = s * growth;
        v = v_next;
    }

    let payoff = match option_type {
        OptionType::Call => (s - strike).positive_part(),
        OptionType::Put => (Dual::constant(strike) - s).positive_part(),
    };
    let pv = payoff * discount;
    (pv.value, pv.derivative)
}

/// Heston Monte Carlo AAD price + delta (w.r.t. initial spot).
#[allow(clippy::too_many_arguments)]
pub fn heston_price_delta_aad(
    option_type: OptionType,
    strike: f64,
    maturity: f64,
    spot: f64,
    rate: f64,
    model: Heston,
    config: HestonAadConfig,
) -> Result<(f64, f64), PricingError> {
    if config.num_paths == 0 {
        return Err(PricingError::InvalidInput(
            "num_paths must be > 0".to_string(),
        ));
    }
    if config.num_steps == 0 {
        return Err(PricingError::InvalidInput(
            "num_steps must be > 0".to_string(),
        ));
    }
    if maturity <= 0.0 {
        let intrinsic = match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        };
        return Ok((intrinsic, 0.0));
    }
    if !model.validate() {
        return Err(PricingError::InvalidInput(
            "invalid Heston parameters".to_string(),
        ));
    }

    let mut z1 = vec![0.0_f64; config.num_steps];
    let mut z2 = vec![0.0_f64; config.num_steps];
    let mut sum_price = 0.0_f64;
    let mut sum_delta = 0.0_f64;

    for i in 0..config.num_paths {
        let seed = resolve_stream_seed(config.seed, i, true);
        let mut rng =
            FastRng::from_seed(crate::math::fast_rng::FastRngKind::Xoshiro256PlusPlus, seed);
        for j in 0..config.num_steps {
            z1[j] = sample_standard_normal(&mut rng);
            z2[j] = sample_standard_normal(&mut rng);
        }

        let (price, delta) = heston_single_path_price_delta(
            option_type,
            strike,
            maturity,
            spot,
            rate,
            model,
            &z1,
            &z2,
        );
        sum_price += price;
        sum_delta += delta;
    }

    let n = config.num_paths as f64;
    Ok((sum_price / n, sum_delta / n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PricingEngine;
    use crate::engines::analytic::black_scholes::{bs_delta, bs_price, bs_rho, bs_theta, bs_vega};
    use crate::engines::monte_carlo::mc_engine::{MonteCarloPricingEngine, VarianceReduction};
    use crate::math::aad::black_scholes_price_greeks_aad;

    fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        (mean, (variance / n).sqrt())
    }

    fn assert_within_four_standard_errors(label: &str, values: &[f64], exact: f64) {
        let (mean, stderr) = mean_and_standard_error(values);
        // Reverse sweeps aggregate thousands of pathwise operations; budget a
        // small multiple of machine precision when batch variance collapses.
        let roundoff = 256.0 * f64::EPSILON * exact.abs().max(1.0);
        assert!(
            (mean - exact).abs() <= 4.0 * stderr + roundoff,
            "{label} mismatch: batch_mean={mean} exact={exact} batch_stderr={stderr}"
        );
    }

    #[test]
    fn mc_pathwise_aad_recovers_bs_first_order_greeks() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.20)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        // Independent seeded batches make the uncertainty of every Greek
        // observable.  The previous fixed per-Greek absolute bands hid that
        // sampling error and had no common confidence interpretation.
        let mut prices = Vec::with_capacity(16);
        let mut deltas = Vec::with_capacity(16);
        let mut vegas = Vec::with_capacity(16);
        let mut rhos = Vec::with_capacity(16);
        let mut thetas = Vec::with_capacity(16);
        for batch in 0..16 {
            let engine = MonteCarloPricingEngine::new(7_500, 64, 4_200 + batch)
                .with_variance_reduction(VarianceReduction::Antithetic);
            let aad =
                mc_european_pathwise_aad(&engine, &option, &market).expect("aad pricing succeeds");
            let greeks = aad.greeks.expect("greeks should be present");
            prices.push(aad.price);
            deltas.push(greeks.delta);
            vegas.push(greeks.vega);
            rhos.push(greeks.rho);
            thetas.push(greeks.theta);
        }

        let ref_delta = bs_delta(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.20,
            option.expiry,
        );
        let ref_vega = bs_vega(
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.20,
            option.expiry,
        );
        let ref_rho = bs_rho(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.20,
            option.expiry,
        );
        let ref_theta = bs_theta(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.20,
            option.expiry,
        );

        let ref_price = bs_price(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.20,
            option.expiry,
        );
        assert_within_four_standard_errors("price", &prices, ref_price);
        assert_within_four_standard_errors("delta", &deltas, ref_delta);
        assert_within_four_standard_errors("vega", &vegas, ref_vega);
        assert_within_four_standard_errors("rho", &rhos, ref_rho);
        assert_within_four_standard_errors("theta", &thetas, ref_theta);
    }

    #[test]
    fn black_scholes_aad_wrapper_matches_analytic() {
        let (price, greeks) =
            black_scholes_price_greeks_aad(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let ref_price = crate::engines::analytic::black_scholes::bs_price(
            OptionType::Call,
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
        );
        let ref_delta = crate::engines::analytic::black_scholes::bs_delta(
            OptionType::Call,
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
        );
        assert!((price - ref_price).abs() < 1e-10);
        assert!((greeks.delta - ref_delta).abs() < 1e-10);
    }

    #[test]
    fn pathwise_aad_retains_near_deterministic_sampling_error() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.0)
            .dividend_yield(0.0)
            .flat_vol(1.0e-10)
            .build()
            .unwrap();
        let option = VanillaOption::european_call(1.0, 1.0);
        let result = mc_european_pathwise_aad(
            &MonteCarloPricingEngine::new(20_000, 1, 42),
            &option,
            &market,
        )
        .unwrap();
        let stderr = result.stderr.unwrap();
        assert!(stderr.is_finite());
        assert!(stderr > 0.0, "stderr={stderr}");
        assert!(stderr < 1.0e-9, "stderr={stderr}");
    }

    #[test]
    fn heston_aad_constant_variance_reduction_matches_black_scholes() {
        let rate = 0.03;
        let variance = 0.04;
        let model = Heston {
            mu: rate,
            kappa: 1.8,
            theta: variance,
            xi: 0.0,
            rho: -0.5,
            v0: variance,
        };
        let option_type = OptionType::Call;
        let strike = 100.0;
        let maturity = 1.0;
        let spot = 100.0;
        let mut prices = Vec::with_capacity(16);
        let mut deltas = Vec::with_capacity(16);

        for batch in 0..16 {
            let cfg = HestonAadConfig::new(4_000, 32, 7_000 + batch);
            let (price, delta) =
                heston_price_delta_aad(option_type, strike, maturity, spot, rate, model, cfg)
                    .expect("heston aad succeeds");
            prices.push(price);
            deltas.push(delta);
        }

        let exact_price = bs_price(
            option_type,
            spot,
            strike,
            rate,
            0.0,
            variance.sqrt(),
            maturity,
        );
        let exact_delta = bs_delta(
            option_type,
            spot,
            strike,
            rate,
            0.0,
            variance.sqrt(),
            maturity,
        );
        assert_within_four_standard_errors("constant-variance Heston price", &prices, exact_price);
        assert_within_four_standard_errors("constant-variance Heston delta", &deltas, exact_delta);
    }

    #[test]
    fn heston_aad_delta_matches_common_random_number_bump_at_roundoff_scale() {
        let model = Heston {
            mu: 0.03,
            kappa: 1.8,
            theta: 0.04,
            xi: 0.6,
            rho: -0.5,
            v0: 0.04,
        };
        let cfg = HestonAadConfig::new(60_000, 64, 7);
        let option_type = OptionType::Call;
        // Every generated path is deep in the money at this strike, making
        // the sampled payoff exactly affine in spot.  The central bump and
        // pathwise derivative should therefore differ only by floating-point
        // subtraction, not by a hand-selected Greek tolerance.
        let strike = 1.0;
        let maturity = 1.0;
        let spot = 100.0;
        let rate = 0.03;

        let (_price, delta_aad) =
            heston_price_delta_aad(option_type, strike, maturity, spot, rate, model, cfg)
                .expect("heston aad succeeds");

        let bump = 1e-2;
        let (price_up, _) =
            heston_price_delta_aad(option_type, strike, maturity, spot + bump, rate, model, cfg)
                .expect("heston up succeeds");
        let (price_dn, _) = heston_price_delta_aad(
            option_type,
            strike,
            maturity,
            (spot - bump).max(1e-8),
            rate,
            model,
            cfg,
        )
        .expect("heston dn succeeds");
        let delta_bump = (price_up - price_dn) / (2.0 * bump);

        let roundoff =
            64.0 * f64::EPSILON * (price_up.abs() + price_dn.abs()).max(1.0) / (2.0 * bump);
        assert!(
            (delta_aad - delta_bump).abs() <= roundoff,
            "AAD delta {delta_aad} vs bump {delta_bump}; roundoff budget={roundoff}"
        );
    }

    #[test]
    fn pricing_engine_trait_has_aad_entry_point_for_mc() {
        let market = Market::builder()
            .spot(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .flat_vol(0.20)
            .build()
            .expect("valid market");
        let option = VanillaOption::european_call(100.0, 1.0);
        let engine = MonteCarloPricingEngine::new(20_000, 64, 17)
            .with_variance_reduction(VarianceReduction::Antithetic);

        let res = engine
            .price_with_greeks_aad(&option, &market)
            .expect("aad trait call should succeed");
        assert!(res.greeks.is_some());
    }
}
