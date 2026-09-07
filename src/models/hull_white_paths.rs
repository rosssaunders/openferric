//! Exact joint simulation of Hull-White rates and their time integral.
//!
//! The centered OU transition and its integral are sampled jointly. The
//! deterministic shift fits the supplied discount curve without a time-step
//! approximation to discounting, including at curve interpolation pillars.

use crate::math::fast_rng::{Xoshiro256PlusPlus, uniform_open01};
use crate::math::normal_inv_cdf;
use crate::models::HullWhite;
use crate::rates::YieldCurve;

/// A risk-neutral short-rate path at the requested observation times.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HullWhitePath {
    pub times: Vec<f64>,
    pub short_rates: Vec<f64>,
    /// Stochastic money-market discount factors from valuation to each time.
    pub discount_factors: Vec<f64>,
}

impl HullWhite {
    /// Samples exact rates and money-market discounts on an arbitrary event grid.
    pub fn simulate_path(
        &self,
        curve: &YieldCurve,
        times: &[f64],
        seed: u64,
    ) -> Result<HullWhitePath, String> {
        HullWhitePathGenerator::new(self, curve, times)?
            .sample(&mut Xoshiro256PlusPlus::seed_from_u64(seed))
    }
}

struct Transition {
    persistence: f64,
    response: f64,
    rate_stddev: f64,
    integral_loading: f64,
    integral_stddev: f64,
    rate_shift: f64,
    log_discount_shift: f64,
}

pub(crate) struct HullWhitePathGenerator {
    times: Vec<f64>,
    transitions: Vec<Transition>,
}

fn response(reversion: f64, time: f64) -> f64 {
    if reversion == 0.0 {
        time
    } else {
        -(-reversion * time).exp_m1() / reversion
    }
}

fn integral_variance(reversion: f64, volatility: f64, time: f64) -> f64 {
    let scaled = reversion * time;
    if scaled.abs() < 0.01 {
        volatility.powi(2)
            * time.powi(3)
            * (1.0 / 3.0
                + scaled
                    * (-1.0 / 4.0
                        + scaled
                            * (7.0 / 60.0
                                + scaled
                                    * (-1.0 / 24.0
                                        + scaled
                                            * (31.0 / 2520.0
                                                + scaled
                                                    * (-1.0 / 320.0
                                                        + scaled * 127.0 / 181440.0))))))
    } else {
        volatility.powi(2) / reversion.powi(2)
            * (time - 2.0 * response(reversion, time) + response(2.0 * reversion, time))
    }
}

impl HullWhitePathGenerator {
    pub(crate) fn new(
        model: &HullWhite,
        curve: &YieldCurve,
        times: &[f64],
    ) -> Result<Self, String> {
        if !model.a.is_finite()
            || model.a < 0.0
            || !model.sigma.is_finite()
            || model.sigma < 0.0
            || times.is_empty()
            || times.iter().any(|time| !time.is_finite() || *time < 0.0)
            || times.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err("Hull-White paths require non-negative finite parameters and strictly increasing non-negative times".into());
        }
        let mut previous = 0.0;
        let mut transitions = Vec::with_capacity(times.len());
        for &time in times {
            let interval = time - previous;
            let response = response(model.a, interval);
            let variance = model.sigma.powi(2) * self::response(2.0 * model.a, interval);
            let covariance = 0.5 * (model.sigma * response).powi(2);
            let integral_loading = if variance > 0.0 {
                covariance / variance.sqrt()
            } else {
                0.0
            };
            let rate_shift = HullWhite::instantaneous_forward(curve, time)
                + 0.5 * (model.sigma * self::response(model.a, time)).powi(2);
            let discount = curve.discount_factor(time);
            let log_discount_shift =
                discount.ln() - 0.5 * integral_variance(model.a, model.sigma, time);
            if discount <= 0.0 || !log_discount_shift.is_finite() || !rate_shift.is_finite() {
                return Err("Hull-White paths require finite positive curve discounts".into());
            }
            transitions.push(Transition {
                persistence: (-model.a * interval).exp(),
                response,
                rate_stddev: variance.sqrt(),
                integral_loading,
                integral_stddev: (integral_variance(model.a, model.sigma, interval)
                    - integral_loading.powi(2))
                .max(0.0)
                .sqrt(),
                rate_shift,
                log_discount_shift,
            });
            previous = time;
        }
        Ok(Self {
            times: times.to_vec(),
            transitions,
        })
    }

    pub(crate) fn sample(&self, rng: &mut Xoshiro256PlusPlus) -> Result<HullWhitePath, String> {
        let mut centered_rate = 0.0;
        let mut integral = 0.0;
        let mut short_rates = Vec::with_capacity(self.times.len());
        let mut discount_factors = Vec::with_capacity(self.times.len());
        for transition in &self.transitions {
            let rate_normal = normal_inv_cdf(uniform_open01(rng.next_f64()));
            let integral_normal = normal_inv_cdf(uniform_open01(rng.next_f64()));
            integral += transition.response * centered_rate
                + transition.integral_loading * rate_normal
                + transition.integral_stddev * integral_normal;
            centered_rate =
                transition.persistence * centered_rate + transition.rate_stddev * rate_normal;
            let discount = (transition.log_discount_shift - integral).exp();
            let rate = centered_rate + transition.rate_shift;
            if !discount.is_finite() || discount <= 0.0 || !rate.is_finite() {
                return Err("non-finite Hull-White path".into());
            }
            short_rates.push(rate);
            discount_factors.push(discount);
        }
        Ok(HullWhitePath {
            times: self.times.clone(),
            short_rates,
            discount_factors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_volatility_discounts_match_nonflat_curve_on_irregular_grid() {
        let curve = YieldCurve::new(vec![(0.2, 1.002), (0.8, 0.97), (2.0, 0.89)]);
        for reversion in [0.0, 1.0e-10, 0.1] {
            let path = HullWhite::new(reversion, 0.0)
                .simulate_path(&curve, &[0.0, 0.13, 0.8, 1.71], 71)
                .unwrap();
            for (&time, &discount) in path.times.iter().zip(&path.discount_factors) {
                assert!((discount - curve.discount_factor(time)).abs() < 1.0e-15);
            }
        }
    }

    #[test]
    fn discounted_bond_is_a_martingale_with_exact_integral_simulation() {
        let curve = YieldCurve::new(vec![(0.2, 1.002), (0.8, 0.97), (2.0, 0.89), (6.0, 0.72)]);
        for reversion in [0.0, 1.0e-8, 0.15] {
            let model = HullWhite::new(reversion, 0.03);
            let generator =
                HullWhitePathGenerator::new(&model, &curve, &[0.17, 0.83, 2.1]).unwrap();
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1729);
            let count = 100_000;
            let mut totals = [0.0; 2];
            let mut squares = [0.0; 2];
            for _ in 0..count {
                let path = generator.sample(&mut rng).unwrap();
                let values = [
                    path.discount_factors[2],
                    path.discount_factors[2]
                        * model.bond_price(2.1, 6.0, path.short_rates[2], &curve),
                ];
                for index in 0..2 {
                    totals[index] += values[index];
                    squares[index] += values[index].powi(2);
                }
            }
            for (index, maturity) in [2.1, 6.0].into_iter().enumerate() {
                let mean = totals[index] / count as f64;
                let stderr =
                    ((squares[index] / count as f64 - mean.powi(2)) / (count - 1) as f64).sqrt();
                assert!((mean - curve.discount_factor(maturity)).abs() < 4.0 * stderr);
            }
        }
    }
}
