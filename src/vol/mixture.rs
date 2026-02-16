use nalgebra::{DMatrix, DVector};

use crate::core::OptionType;
use crate::pricing::european::black_scholes_price;

#[derive(Debug, Clone, PartialEq)]
pub struct LognormalMixture {
    pub weights: Vec<f64>,
    pub vols: Vec<f64>,
}

impl LognormalMixture {
    pub fn new(weights: Vec<f64>, vols: Vec<f64>) -> Result<Self, String> {
        if weights.len() != vols.len() || weights.len() < 2 {
            return Err("mixture requires >= 2 components with matching weights/vols".to_string());
        }
        if weights.iter().any(|&w| w < 0.0) {
            return Err("mixture weights must be >= 0".to_string());
        }
        if vols.iter().any(|&v| v <= 0.0) {
            return Err("mixture vols must be > 0".to_string());
        }

        let wsum: f64 = weights.iter().sum();
        if wsum <= 0.0 {
            return Err("mixture weights must sum to > 0".to_string());
        }

        let normalized_weights = weights.into_iter().map(|w| w / wsum).collect();
        Ok(Self {
            weights: normalized_weights,
            vols,
        })
    }

    pub fn price(
        &self,
        option_type: OptionType,
        spot: f64,
        strike: f64,
        rate: f64,
        expiry: f64,
    ) -> f64 {
        self.weights
            .iter()
            .zip(self.vols.iter())
            .map(|(&w, &sigma)| {
                w * black_scholes_price(option_type, spot, strike, rate, sigma, expiry)
            })
            .sum()
    }

    pub fn d2_call_dk2(&self, spot: f64, strike: f64, rate: f64, expiry: f64, dk: f64) -> f64 {
        let h = dk.max(1e-4);
        let k_dn = (strike - h).max(1e-8);
        let c_up = self.price(OptionType::Call, spot, strike + h, rate, expiry);
        let c_0 = self.price(OptionType::Call, spot, strike, rate, expiry);
        let c_dn = self.price(OptionType::Call, spot, k_dn, rate, expiry);
        ((c_up - 2.0 * c_0 + c_dn) / (h * h)).max(0.0)
    }

    pub fn implied_density(&self, spot: f64, strike: f64, rate: f64, expiry: f64, dk: f64) -> f64 {
        ((rate * expiry).exp() * self.d2_call_dk2(spot, strike, rate, expiry, dk)).max(0.0)
    }

    pub fn implied_density_curve(
        &self,
        spot: f64,
        strikes: &[f64],
        rate: f64,
        expiry: f64,
        dk: f64,
    ) -> Vec<(f64, f64)> {
        strikes
            .iter()
            .copied()
            .filter(|k| *k > 0.0)
            .map(|k| (k, self.implied_density(spot, k, rate, expiry, dk)))
            .collect()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let exp: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exp.into_iter().map(|x| x / sum).collect()
    }
}

fn unpack_params(params: &[f64], components: usize) -> (Vec<f64>, Vec<f64>) {
    let logits = &params[..components];
    let raw_vols = &params[components..(2 * components)];

    let weights = softmax(logits);
    let vols = raw_vols
        .iter()
        .map(|x| x.exp().clamp(1e-4, 4.0))
        .collect();

    (weights, vols)
}

fn objective(
    params: &[f64],
    option_type: OptionType,
    spot: f64,
    rate: f64,
    expiry: f64,
    strikes: &[f64],
    market_prices: &[f64],
    components: usize,
) -> f64 {
    let (weights, vols) = unpack_params(params, components);
    let mixture = LognormalMixture { weights, vols };

    let n = strikes.len().min(market_prices.len());
    let mut mse = 0.0;
    for i in 0..n {
        let model = mixture.price(option_type, spot, strikes[i], rate, expiry);
        let err = model - market_prices[i];
        mse += err * err;
    }

    mse / n as f64
}

#[allow(clippy::too_many_arguments)]
fn lm_optimize(
    mut params: Vec<f64>,
    option_type: OptionType,
    spot: f64,
    rate: f64,
    expiry: f64,
    strikes: &[f64],
    market_prices: &[f64],
    components: usize,
) -> (Vec<f64>, f64) {
    let n = strikes.len();
    let m = params.len();
    let mut lambda = 1e-3;
    let mut best_obj = objective(
        &params,
        option_type,
        spot,
        rate,
        expiry,
        strikes,
        market_prices,
        components,
    );

    for _ in 0..250 {
        let mut j = DMatrix::<f64>::zeros(n, m);
        let mut r = DVector::<f64>::zeros(n);

        let (weights, vols) = unpack_params(&params, components);
        let mixture = LognormalMixture {
            weights: weights.clone(),
            vols: vols.clone(),
        };

        for i in 0..n {
            r[i] = mixture.price(option_type, spot, strikes[i], rate, expiry) - market_prices[i];
        }

        for p_idx in 0..m {
            let eps = (params[p_idx].abs() * 1e-4).max(1e-5);
            let mut p_up = params.clone();
            p_up[p_idx] += eps;
            let mut p_dn = params.clone();
            p_dn[p_idx] -= eps;

            let (w_up, v_up) = unpack_params(&p_up, components);
            let (w_dn, v_dn) = unpack_params(&p_dn, components);
            let mix_up = LognormalMixture {
                weights: w_up,
                vols: v_up,
            };
            let mix_dn = LognormalMixture {
                weights: w_dn,
                vols: v_dn,
            };

            for i in 0..n {
                let v_plus = mix_up.price(option_type, spot, strikes[i], rate, expiry);
                let v_minus = mix_dn.price(option_type, spot, strikes[i], rate, expiry);
                j[(i, p_idx)] = (v_plus - v_minus) / (2.0 * eps);
            }
        }

        let jt = j.transpose();
        let mut a = &jt * &j;
        for d in 0..m {
            a[(d, d)] += lambda;
        }
        let g = &jt * &r;

        let Some(delta) = a.lu().solve(&g) else {
            lambda = (lambda * 10.0).min(1e8);
            continue;
        };

        let mut candidate = params.clone();
        for i in 0..m {
            candidate[i] -= delta[i];
        }

        let cand_obj = objective(
            &candidate,
            option_type,
            spot,
            rate,
            expiry,
            strikes,
            market_prices,
            components,
        );

        if cand_obj < best_obj {
            let improvement = (best_obj - cand_obj).abs();
            params = candidate;
            best_obj = cand_obj;
            lambda = (lambda * 0.5).max(1e-8);
            if improvement < 1e-12 || delta.norm() < 1e-8 {
                break;
            }
        } else {
            lambda = (lambda * 2.0).min(1e8);
        }
    }

    (params, best_obj)
}

#[allow(clippy::too_many_arguments)]
pub fn calibrate_lognormal_mixture(
    option_type: OptionType,
    spot: f64,
    rate: f64,
    expiry: f64,
    strikes: &[f64],
    market_prices: &[f64],
    components: usize,
) -> Result<LognormalMixture, String> {
    if !(2..=3).contains(&components) {
        return Err("components must be 2 or 3".to_string());
    }
    if spot <= 0.0 || expiry <= 0.0 {
        return Err("spot and expiry must be > 0".to_string());
    }
    if strikes.len() != market_prices.len() || strikes.len() < components + 2 {
        return Err("need at least components+2 strike/price points".to_string());
    }
    if strikes.windows(2).any(|w| w[1] <= w[0]) {
        return Err("strikes must be strictly increasing".to_string());
    }
    if market_prices.iter().any(|&p| p <= 0.0 || !p.is_finite()) {
        return Err("market prices must be finite and > 0".to_string());
    }

    let atm_idx = strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - spot).abs().total_cmp(&(*b - spot).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let atm_price = market_prices[atm_idx];
    let atm_guess = crate::vol::implied::implied_vol_newton(
        option_type,
        spot,
        strikes[atm_idx],
        rate,
        expiry,
        atm_price,
        1e-10,
        100,
    )
    .unwrap_or(0.2)
    .clamp(0.05, 1.0);

    let mut starts = Vec::new();

    let vol_shapes: Vec<Vec<f64>> = if components == 2 {
        vec![
            vec![0.65 * atm_guess, 1.35 * atm_guess],
            vec![0.80 * atm_guess, 1.20 * atm_guess],
            vec![0.50 * atm_guess, 1.70 * atm_guess],
        ]
    } else {
        vec![
            vec![0.60 * atm_guess, atm_guess, 1.40 * atm_guess],
            vec![0.75 * atm_guess, 1.05 * atm_guess, 1.45 * atm_guess],
            vec![0.50 * atm_guess, 0.95 * atm_guess, 1.80 * atm_guess],
        ]
    };

    let weight_shapes: Vec<Vec<f64>> = if components == 2 {
        vec![vec![0.5, 0.5], vec![0.7, 0.3], vec![0.3, 0.7]]
    } else {
        vec![
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.5, 0.3, 0.2],
            vec![0.2, 0.3, 0.5],
        ]
    };

    for vols in &vol_shapes {
        for weights in &weight_shapes {
            let mut params = Vec::with_capacity(2 * components);
            params.extend(weights.iter().map(|w| w.max(1e-8).ln()));
            params.extend(vols.iter().map(|v| v.max(1e-8).ln()));
            starts.push(params);
        }
    }

    let mut best_params = starts[0].clone();
    let mut best_obj = f64::INFINITY;

    for start in starts {
        let (p, obj) = lm_optimize(
            start,
            option_type,
            spot,
            rate,
            expiry,
            strikes,
            market_prices,
            components,
        );

        if obj < best_obj {
            best_obj = obj;
            best_params = p;
        }
    }

    let (weights, vols) = unpack_params(&best_params, components);

    // Keep deterministic ordering by vol.
    let mut pairs: Vec<(f64, f64)> = vols.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let sorted_vols: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    let sorted_weights: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    LognormalMixture::new(sorted_weights, sorted_vols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vol::implied::implied_vol_newton;

    #[test]
    fn mixture_weights_are_normalized() {
        let m = LognormalMixture::new(vec![2.0, 1.0], vec![0.2, 0.4]).unwrap();
        let sum: f64 = m.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn two_component_calibration_fits_synthetic_prices() {
        let true_mix = LognormalMixture::new(vec![0.65, 0.35], vec![0.14, 0.30]).unwrap();

        let spot = 100.0;
        let rate = 0.01;
        let expiry = 1.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let market_prices: Vec<f64> = strikes
            .iter()
            .map(|&k| true_mix.price(OptionType::Call, spot, k, rate, expiry))
            .collect();

        let fit = calibrate_lognormal_mixture(
            OptionType::Call,
            spot,
            rate,
            expiry,
            &strikes,
            &market_prices,
            2,
        )
        .unwrap();

        for (k, market_price) in strikes.iter().zip(market_prices.iter()) {
            let fit_price = fit.price(OptionType::Call, spot, *k, rate, expiry);
            let vol_market = implied_vol_newton(
                OptionType::Call,
                spot,
                *k,
                rate,
                expiry,
                *market_price,
                1e-10,
                100,
            )
            .unwrap();
            let vol_fit = implied_vol_newton(
                OptionType::Call,
                spot,
                *k,
                rate,
                expiry,
                fit_price,
                1e-10,
                100,
            )
            .unwrap();

            assert!((vol_market - vol_fit).abs() < 0.1);
        }
    }

    #[test]
    fn implied_density_is_non_negative() {
        let mix = LognormalMixture::new(vec![0.7, 0.3], vec![0.16, 0.32]).unwrap();

        for k in (60..=140).step_by(5) {
            let dens = mix.implied_density(100.0, k as f64, 0.01, 1.0, 0.5);
            assert!(dens >= 0.0);
        }
    }
}
