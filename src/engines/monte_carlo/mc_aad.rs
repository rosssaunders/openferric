use crate::core::{AadPricingResult, AadSensitivity, ExerciseStyle, OptionType, PricingError};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;
use crate::math::aad::{Tape, TapeCheckpoint, VarId};
use crate::math::fast_rng::{FastRng, FastRngKind, resolve_stream_seed, sample_standard_normal};

/// Heston parameter set used by Monte Carlo AAD pathwise pricing.
#[derive(Debug, Clone, Copy)]
pub struct HestonAadParams {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
}

impl HestonAadParams {
    fn validate(self) -> Result<(), PricingError> {
        if self.v0 < 0.0 {
            return Err(PricingError::InvalidInput(
                "heston v0 must be >= 0".to_string(),
            ));
        }
        if self.kappa <= 0.0 {
            return Err(PricingError::InvalidInput(
                "heston kappa must be > 0".to_string(),
            ));
        }
        if self.theta < 0.0 {
            return Err(PricingError::InvalidInput(
                "heston theta must be >= 0".to_string(),
            ));
        }
        if self.xi < 0.0 {
            return Err(PricingError::InvalidInput(
                "heston xi must be >= 0".to_string(),
            ));
        }
        if self.rho <= -1.0 || self.rho >= 1.0 {
            return Err(PricingError::InvalidInput(
                "heston rho must be in (-1, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

/// Pathwise Monte Carlo AAD pricer with tape checkpointing.
///
/// Memory usage is bounded in the number of time steps (not paths) by rewinding
/// the reverse tape to a base checkpoint after each path valuation.
#[derive(Debug, Clone)]
pub struct MonteCarloAadEngine {
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Number of time steps per path.
    pub num_steps: usize,
    /// Base seed.
    pub seed: u64,
    /// Use antithetic pairs.
    pub antithetic: bool,
    /// RNG backend.
    pub rng_kind: FastRngKind,
    /// Deterministic stream splitting.
    pub reproducible: bool,
}

impl MonteCarloAadEngine {
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            antithetic: true,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
            reproducible: true,
        }
    }

    pub fn with_antithetic(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
        self
    }

    pub fn with_rng_kind(mut self, rng_kind: FastRngKind) -> Self {
        self.rng_kind = rng_kind;
        if matches!(rng_kind, FastRngKind::ThreadRng) {
            self.reproducible = false;
        }
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.reproducible = true;
        self
    }

    pub fn with_randomized_streams(mut self) -> Self {
        self.reproducible = false;
        self
    }

    /// GBM Monte Carlo AAD for a European vanilla option.
    ///
    /// Gradient factors: `spot`, `rate`, `dividend_yield`, `vol`.
    pub fn price_vanilla_gbm_aad(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<AadPricingResult, PricingError> {
        self.validate_vanilla_request(instrument)?;
        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 || !vol.is_finite() {
            return Err(PricingError::InvalidInput(
                "market volatility must be finite and > 0".to_string(),
            ));
        }

        let maturity = instrument.expiry;
        if maturity <= 0.0 {
            let intrinsic = match instrument.option_type {
                OptionType::Call => (market.spot - instrument.strike).max(0.0),
                OptionType::Put => (instrument.strike - market.spot).max(0.0),
            };
            return Ok(AadPricingResult {
                price: intrinsic,
                gradient: vec![
                    AadSensitivity {
                        factor: "spot".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "rate".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "dividend_yield".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "vol".to_string(),
                        value: 0.0,
                    },
                ],
            });
        }

        let dt = maturity / self.num_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut tape = Tape::with_capacity(self.num_steps * 20 + 128);
        let inputs = [
            tape.input(market.spot),
            tape.input(market.rate),
            tape.input(market.dividend_yield),
            tape.input(vol),
        ];
        let base_checkpoint = tape.checkpoint();

        let mut stats = RunningStats::default();
        let mut gradient_sum = [0.0_f64; 4];

        let samples = if self.antithetic {
            self.num_paths.div_ceil(2)
        } else {
            self.num_paths
        };

        for i in 0..samples {
            let seed = resolve_stream_seed(self.seed, i, self.reproducible);
            let mut rng = FastRng::from_seed(self.rng_kind, seed);
            let mut normals = vec![0.0_f64; self.num_steps];
            for z in &mut normals {
                *z = sample_standard_normal(&mut rng);
            }

            let (pv, grad) = gbm_pathwise_pv_and_gradient(
                &mut tape,
                base_checkpoint,
                inputs,
                instrument.option_type,
                instrument.strike,
                maturity,
                dt,
                sqrt_dt,
                &normals,
            );

            let (pv_final, grad_final) = if self.antithetic {
                let anti: Vec<f64> = normals.iter().map(|z| -*z).collect();
                let (pv_a, grad_a) = gbm_pathwise_pv_and_gradient(
                    &mut tape,
                    base_checkpoint,
                    inputs,
                    instrument.option_type,
                    instrument.strike,
                    maturity,
                    dt,
                    sqrt_dt,
                    &anti,
                );
                (
                    0.5 * (pv + pv_a),
                    [
                        0.5 * (grad[0] + grad_a[0]),
                        0.5 * (grad[1] + grad_a[1]),
                        0.5 * (grad[2] + grad_a[2]),
                        0.5 * (grad[3] + grad_a[3]),
                    ],
                )
            } else {
                (pv, grad)
            };

            stats.push(pv_final);
            for j in 0..4 {
                gradient_sum[j] += grad_final[j];
            }
        }

        let n = samples as f64;
        let grad = gradient_sum.map(|g| g / n);

        Ok(AadPricingResult {
            price: stats.mean,
            gradient: vec![
                AadSensitivity {
                    factor: "spot".to_string(),
                    value: grad[0],
                },
                AadSensitivity {
                    factor: "rate".to_string(),
                    value: grad[1],
                },
                AadSensitivity {
                    factor: "dividend_yield".to_string(),
                    value: grad[2],
                },
                AadSensitivity {
                    factor: "vol".to_string(),
                    value: grad[3],
                },
                AadSensitivity {
                    factor: "stderr".to_string(),
                    value: stats.stderr(),
                },
            ],
        })
    }

    /// Heston Monte Carlo AAD for a European vanilla option.
    ///
    /// Gradient factors:
    /// `spot`, `rate`, `dividend_yield`, `v0`, `kappa`, `theta`, `xi`, `rho`.
    pub fn price_vanilla_heston_aad(
        &self,
        instrument: &VanillaOption,
        market: &Market,
        params: HestonAadParams,
    ) -> Result<AadPricingResult, PricingError> {
        self.validate_vanilla_request(instrument)?;
        params.validate()?;
        let maturity = instrument.expiry;
        if maturity <= 0.0 {
            let intrinsic = match instrument.option_type {
                OptionType::Call => (market.spot - instrument.strike).max(0.0),
                OptionType::Put => (instrument.strike - market.spot).max(0.0),
            };
            return Ok(AadPricingResult {
                price: intrinsic,
                gradient: vec![
                    AadSensitivity {
                        factor: "spot".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "rate".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "dividend_yield".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "v0".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "kappa".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "theta".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "xi".to_string(),
                        value: 0.0,
                    },
                    AadSensitivity {
                        factor: "rho".to_string(),
                        value: 0.0,
                    },
                ],
            });
        }

        let dt = maturity / self.num_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut tape = Tape::with_capacity(self.num_steps * 36 + 192);
        let inputs = [
            tape.input(market.spot),
            tape.input(market.rate),
            tape.input(market.dividend_yield),
            tape.input(params.v0),
            tape.input(params.kappa),
            tape.input(params.theta),
            tape.input(params.xi),
            tape.input(params.rho),
        ];
        let base_checkpoint = tape.checkpoint();

        let mut stats = RunningStats::default();
        let mut gradient_sum = [0.0_f64; 8];
        let samples = if self.antithetic {
            self.num_paths.div_ceil(2)
        } else {
            self.num_paths
        };

        for i in 0..samples {
            let seed = resolve_stream_seed(self.seed, i, self.reproducible);
            let mut rng = FastRng::from_seed(self.rng_kind, seed);
            let mut z1 = vec![0.0_f64; self.num_steps];
            let mut z2 = vec![0.0_f64; self.num_steps];
            for j in 0..self.num_steps {
                z1[j] = sample_standard_normal(&mut rng);
                z2[j] = sample_standard_normal(&mut rng);
            }

            let (pv, grad) = heston_pathwise_pv_and_gradient(
                &mut tape,
                base_checkpoint,
                inputs,
                instrument.option_type,
                instrument.strike,
                maturity,
                dt,
                sqrt_dt,
                &z1,
                &z2,
            );

            let (pv_final, grad_final) = if self.antithetic {
                let z1a: Vec<f64> = z1.iter().map(|z| -*z).collect();
                let z2a: Vec<f64> = z2.iter().map(|z| -*z).collect();
                let (pv_a, grad_a) = heston_pathwise_pv_and_gradient(
                    &mut tape,
                    base_checkpoint,
                    inputs,
                    instrument.option_type,
                    instrument.strike,
                    maturity,
                    dt,
                    sqrt_dt,
                    &z1a,
                    &z2a,
                );
                (
                    0.5 * (pv + pv_a),
                    std::array::from_fn(|k| 0.5 * (grad[k] + grad_a[k])),
                )
            } else {
                (pv, grad)
            };

            stats.push(pv_final);
            for j in 0..8 {
                gradient_sum[j] += grad_final[j];
            }
        }

        let n = samples as f64;
        let grad = gradient_sum.map(|g| g / n);

        Ok(AadPricingResult {
            price: stats.mean,
            gradient: vec![
                AadSensitivity {
                    factor: "spot".to_string(),
                    value: grad[0],
                },
                AadSensitivity {
                    factor: "rate".to_string(),
                    value: grad[1],
                },
                AadSensitivity {
                    factor: "dividend_yield".to_string(),
                    value: grad[2],
                },
                AadSensitivity {
                    factor: "v0".to_string(),
                    value: grad[3],
                },
                AadSensitivity {
                    factor: "kappa".to_string(),
                    value: grad[4],
                },
                AadSensitivity {
                    factor: "theta".to_string(),
                    value: grad[5],
                },
                AadSensitivity {
                    factor: "xi".to_string(),
                    value: grad[6],
                },
                AadSensitivity {
                    factor: "rho".to_string(),
                    value: grad[7],
                },
                AadSensitivity {
                    factor: "stderr".to_string(),
                    value: stats.stderr(),
                },
            ],
        })
    }

    fn validate_vanilla_request(&self, instrument: &VanillaOption) -> Result<(), PricingError> {
        instrument.validate()?;
        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "MonteCarloAadEngine supports European exercise only".to_string(),
            ));
        }
        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }
        if self.num_steps == 0 {
            return Err(PricingError::InvalidInput(
                "num_steps must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn gbm_pathwise_pv_and_gradient(
    tape: &mut Tape,
    checkpoint: TapeCheckpoint,
    inputs: [VarId; 4],
    option_type: OptionType,
    strike: f64,
    maturity: f64,
    dt: f64,
    sqrt_dt: f64,
    normals: &[f64],
) -> (f64, [f64; 4]) {
    tape.rewind(checkpoint);

    let spot = inputs[0];
    let rate = inputs[1];
    let dividend = inputs[2];
    let vol = inputs[3];

    let mut s = spot;
    let vol_sq = tape.mul(vol, vol);
    let half_vol_sq = tape.mul_const(vol_sq, 0.5);
    let r_minus_q = tape.sub(rate, dividend);
    let carry = tape.sub(r_minus_q, half_vol_sq);
    let drift_step = tape.mul_const(carry, dt);

    for &z in normals {
        let diffusion = tape.mul_const(vol, sqrt_dt * z);
        let exponent = tape.add(drift_step, diffusion);
        let growth = tape.exp(exponent);
        s = tape.mul(s, growth);
    }

    let intrinsic = match option_type {
        OptionType::Call => tape.sub_const(s, strike),
        OptionType::Put => tape.const_sub(strike, s),
    };
    let payoff = tape.positive_part(intrinsic);

    let neg_rt = tape.mul_const(rate, -maturity);
    let discount = tape.exp(neg_rt);
    let pv = tape.mul(discount, payoff);

    let grad = tape.gradient(pv, &inputs);
    (tape.value(pv), [grad[0], grad[1], grad[2], grad[3]])
}

#[allow(clippy::too_many_arguments)]
fn heston_pathwise_pv_and_gradient(
    tape: &mut Tape,
    checkpoint: TapeCheckpoint,
    inputs: [VarId; 8],
    option_type: OptionType,
    strike: f64,
    maturity: f64,
    dt: f64,
    sqrt_dt: f64,
    z1: &[f64],
    z2: &[f64],
) -> (f64, [f64; 8]) {
    tape.rewind(checkpoint);

    let spot = inputs[0];
    let rate = inputs[1];
    let dividend = inputs[2];
    let kappa = inputs[4];
    let theta = inputs[5];
    let xi = inputs[6];
    let rho = inputs[7];

    let mut s = spot;
    let mut v = inputs[3];

    let rho_sq = tape.mul(rho, rho);
    let one_minus_rho_sq = tape.const_sub(1.0, rho_sq);
    let corr_scale = tape.sqrt(one_minus_rho_sq);
    let r_minus_q = tape.sub(rate, dividend);

    for (&n1, &n2) in z1.iter().zip(z2.iter()) {
        let v_pos = tape.positive_part(v);
        let v_eps = tape.add_const(v_pos, 1.0e-16);
        let sqrt_v = tape.sqrt(v_eps);

        let theta_minus_v = tape.sub(theta, v_pos);
        let mean_revert = tape.mul(kappa, theta_minus_v);
        let drift_v = tape.mul_const(mean_revert, dt);
        let diff_v_scale = tape.mul(xi, sqrt_v);
        let diff_v = tape.mul_const(diff_v_scale, sqrt_dt * n1);
        let v_increment = tape.add(drift_v, diff_v);
        let v_next = tape.add(v, v_increment);
        v = tape.positive_part(v_next);

        let half_v = tape.mul_const(v_pos, 0.5);
        let drift_s_raw = tape.sub(r_minus_q, half_v);
        let drift_s = tape.mul_const(drift_s_raw, dt);

        let rho_z1 = tape.mul_const(rho, n1);
        let corr_z2 = tape.mul_const(corr_scale, n2);
        let zs = tape.add(rho_z1, corr_z2);
        let diff_s_scale = tape.mul_const(sqrt_v, sqrt_dt);
        let diff_s = tape.mul(diff_s_scale, zs);
        let expo = tape.add(drift_s, diff_s);
        let growth = tape.exp(expo);
        s = tape.mul(s, growth);
    }

    let intrinsic = match option_type {
        OptionType::Call => tape.sub_const(s, strike),
        OptionType::Put => tape.const_sub(strike, s),
    };
    let payoff = tape.positive_part(intrinsic);
    let neg_rt = tape.mul_const(rate, -maturity);
    let discount = tape.exp(neg_rt);
    let pv = tape.mul(discount, payoff);

    let grad = tape.gradient(pv, &inputs);
    (
        tape.value(pv),
        [
            grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7],
        ],
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn stderr(self) -> f64 {
        if self.n <= 1 {
            return 0.0;
        }
        let n = self.n as f64;
        let variance = self.m2 / (n - 1.0);
        (variance / n).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::analytic::black_scholes::{bs_delta, bs_rho, bs_vega};

    fn grad(result: &AadPricingResult, name: &str) -> f64 {
        result
            .gradient
            .iter()
            .find(|g| g.factor == name)
            .map(|g| g.value)
            .unwrap()
    }

    #[test]
    fn gbm_mc_aad_matches_black_scholes_first_order_greeks() {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.01)
            .flat_vol(0.2)
            .build()
            .unwrap();

        let engine = MonteCarloAadEngine::new(120_000, 96, 7).with_antithetic(true);
        let aad = engine.price_vanilla_gbm_aad(&option, &market).unwrap();

        let analytic_delta = bs_delta(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.2,
            option.expiry,
        );
        let analytic_vega = bs_vega(
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.2,
            option.expiry,
        );
        let analytic_rho = bs_rho(
            option.option_type,
            market.spot,
            option.strike,
            market.rate,
            market.dividend_yield,
            0.2,
            option.expiry,
        );

        assert!((grad(&aad, "spot") - analytic_delta).abs() < 0.02);
        assert!((grad(&aad, "vol") - analytic_vega).abs() < 0.30);
        assert!((grad(&aad, "rate") - analytic_rho).abs() < 0.30);
    }

    #[test]
    fn heston_mc_aad_is_consistent_with_bumped_mc() {
        let option = VanillaOption::european_call(100.0, 1.0);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.02)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap();
        let params = HestonAadParams {
            v0: 0.04,
            kappa: 1.4,
            theta: 0.04,
            xi: 0.5,
            rho: -0.6,
        };

        let engine = MonteCarloAadEngine::new(50_000, 64, 17).with_antithetic(true);
        let aad = engine
            .price_vanilla_heston_aad(&option, &market, params)
            .unwrap();

        let bump = 5e-3;
        let bumped = |f: fn(&mut HestonAadParams) -> &mut f64| {
            let mut up = params;
            *f(&mut up) += bump;
            let mut dn = params;
            *f(&mut dn) -= bump;
            let pu = engine
                .price_vanilla_heston_aad(&option, &market, up)
                .unwrap()
                .price;
            let pd = engine
                .price_vanilla_heston_aad(&option, &market, dn)
                .unwrap()
                .price;
            (pu - pd) / (2.0 * bump)
        };

        let fd_v0 = bumped(|p| &mut p.v0);
        let fd_kappa = bumped(|p| &mut p.kappa);
        let fd_theta = bumped(|p| &mut p.theta);
        let fd_xi = bumped(|p| &mut p.xi);
        let fd_rho = bumped(|p| &mut p.rho);

        let close = |name: &str, aad_g: f64, fd_g: f64| {
            let err = (aad_g - fd_g).abs();
            let rel = err / (1.0 + fd_g.abs());
            assert!(
                rel < 0.6,
                "{name}: aad={aad_g}, fd={fd_g}, err={err}, rel={rel}"
            );
        };

        close("v0", grad(&aad, "v0"), fd_v0);
        close("kappa", grad(&aad, "kappa"), fd_kappa);
        close("theta", grad(&aad, "theta"), fd_theta);
        close("xi", grad(&aad, "xi"), fd_xi);
        close("rho", grad(&aad, "rho"), fd_rho);
    }
}
