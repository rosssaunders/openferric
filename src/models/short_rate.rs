//! Module `models::short_rate`.
//!
//! Implements short rate abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull and White (1990), Brigo and Mercurio (2006) Ch. 3, short-rate calibration relations around Eq. (3.28).
//!
//! Key types and purpose: `Vasicek`, `CIR`, `HullWhite` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
use crate::rates::YieldCurve;

/// Vasicek short-rate model: `dr = a(b-r)dt + sigma dW`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vasicek {
    /// Mean reversion speed.
    pub a: f64,
    /// Long-run mean rate.
    pub b: f64,
    /// Rate volatility.
    pub sigma: f64,
}

impl Vasicek {
    /// Returns the closed-form zero-coupon bond price `P(t,T)`.
    pub fn bond_price(&self, t: f64, maturity: f64, short_rate: f64) -> f64 {
        if maturity <= t {
            return 1.0;
        }
        let tau = maturity - t;
        let b = self.bond_b(t, maturity);
        let sigma2 = self.sigma * self.sigma;
        let scaled_time = self.a * tau;
        let integrated_variance = if scaled_time.abs() < 0.01 {
            let coefficient = 1.0 / 3.0
                + scaled_time
                    * (-1.0 / 4.0
                        + scaled_time
                            * (7.0 / 60.0
                                + scaled_time
                                    * (-1.0 / 24.0
                                        + scaled_time
                                            * (31.0 / 2520.0
                                                + scaled_time
                                                    * (-1.0 / 320.0
                                                        + scaled_time * 127.0 / 181440.0)))));
            sigma2 * tau.powi(3) * coefficient
        } else {
            sigma2 / (self.a * self.a)
                * (tau - 2.0 * b - (-2.0 * scaled_time).exp_m1() / (2.0 * self.a))
        };
        (-short_rate * b - self.b * (tau - b) + 0.5 * integrated_variance).exp()
    }

    fn bond_b(&self, t: f64, maturity: f64) -> f64 {
        let tau = maturity - t;
        if tau <= 0.0 {
            0.0
        } else if self.a == 0.0 {
            tau
        } else {
            -(-self.a * tau).exp_m1() / self.a
        }
    }
}

/// Cox-Ingersoll-Ross (CIR) short-rate model: `dr = a(b-r)dt + sigma sqrt(r) dW`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIR {
    /// Mean reversion speed.
    pub a: f64,
    /// Long-run mean rate.
    pub b: f64,
    /// Volatility coefficient.
    pub sigma: f64,
}

impl CIR {
    /// Returns the closed-form zero-coupon bond price `P(t,T)`.
    pub fn bond_price(&self, t: f64, maturity: f64, short_rate: f64) -> f64 {
        if maturity <= t {
            return 1.0;
        }

        let tau = maturity - t;
        if self.sigma == 0.0 {
            let response = if self.a == 0.0 {
                tau
            } else {
                -(-self.a * tau).exp_m1() / self.a
            };
            return (-short_rate * response - self.b * (tau - response)).exp();
        }
        let gamma = (self.a * self.a + 2.0 * self.sigma * self.sigma).sqrt();
        let variance = self.sigma * self.sigma;
        let decay = (-gamma * tau).exp();
        let one_minus_decay = -(-gamma * tau).exp_m1();
        let denominator = (gamma + self.a) * one_minus_decay + 2.0 * gamma * decay;
        let response = 2.0 * one_minus_decay / denominator;
        let gamma_minus_reversion = 2.0 * variance / (gamma + self.a);
        let log_prefactor = 2.0 * self.a * self.b / variance
            * ((gamma_minus_reversion * one_minus_decay / denominator).ln_1p()
                - 0.5 * gamma_minus_reversion * tau);
        (log_prefactor - response * short_rate).exp()
    }
}

/// One-factor Hull-White model: `dr = (theta(t) - a r)dt + sigma dW`.
#[derive(Debug, Clone, PartialEq)]
pub struct HullWhite {
    /// Mean reversion speed.
    pub a: f64,
    /// Short-rate volatility.
    pub sigma: f64,
    /// Calibrated theta curve as `(time, theta)` points.
    pub theta: Vec<(f64, f64)>,
}

impl HullWhite {
    /// Creates a Hull-White model.
    pub fn new(a: f64, sigma: f64) -> Self {
        Self {
            a,
            sigma,
            theta: Vec::new(),
        }
    }

    /// Samples `theta(t)` on a grid from a differentiable initial forward curve.
    /// Forward jumps at interpolation pillars have distributional derivatives
    /// that point samples cannot represent. Pricing lattices fit discount
    /// factors through state prices instead of relying on these samples.
    pub fn calibrate_theta(&mut self, initial_curve: &YieldCurve, times: &[f64]) {
        let mut grid = times
            .iter()
            .copied()
            .filter(|t| *t >= 0.0)
            .collect::<Vec<_>>();
        grid.sort_by(|a, b| a.total_cmp(b));
        grid.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);

        self.theta = grid
            .into_iter()
            .map(|t| {
                let f = Self::instantaneous_forward(initial_curve, t);
                let dfdt = Self::forward_derivative(initial_curve, t);
                let conv = if self.a.abs() <= 1.0e-12 {
                    self.sigma * self.sigma * t
                } else {
                    (self.sigma * self.sigma) * (1.0 - (-2.0 * self.a * t).exp()) / (2.0 * self.a)
                };
                (t, dfdt + self.a * f + conv)
            })
            .collect();
    }

    /// Returns interpolated `theta(t)` from calibrated points.
    pub fn theta_at(&self, t: f64) -> f64 {
        if self.theta.is_empty() {
            return 0.0;
        }
        if t <= self.theta[0].0 {
            return self.theta[0].1;
        }

        for window in self.theta.windows(2) {
            let (t1, th1) = window[0];
            let (t2, th2) = window[1];
            if t <= t2 {
                let w = (t - t1) / (t2 - t1);
                return th1 + w * (th2 - th1);
            }
        }

        self.theta[self.theta.len() - 1].1
    }

    /// Closed-form zero-coupon bond price under calibrated Hull-White model.
    pub fn bond_price(
        &self,
        t: f64,
        maturity: f64,
        short_rate: f64,
        initial_curve: &YieldCurve,
    ) -> f64 {
        if maturity <= t {
            return 1.0;
        }

        let b = self.bond_b(t, maturity);
        let p0_t = initial_curve.discount_factor(t);
        let p0_t_maturity = initial_curve.discount_factor(maturity);
        let f0_t = Self::instantaneous_forward(initial_curve, t);

        let variance_adj = if self.a.abs() <= 1.0e-12 {
            0.5 * self.sigma * self.sigma * t * b * b
        } else {
            (self.sigma * self.sigma) * (1.0 - (-2.0 * self.a * t).exp()) * b * b / (4.0 * self.a)
        };

        let a = (p0_t_maturity / p0_t) * (b * f0_t - variance_adj).exp();
        a * (-b * short_rate).exp()
    }

    /// Instantaneous forward rate implied by the initial discount curve.
    pub fn instantaneous_forward(initial_curve: &YieldCurve, t: f64) -> f64 {
        let eps = (1.0e-4_f64).max(1.0e-4 * (1.0 + t.abs()));
        let t1 = (t - eps).max(0.0);
        let t2 = t + eps;

        let ln_p1 = initial_curve.discount_factor(t1).ln();
        let ln_p2 = initial_curve.discount_factor(t2).ln();

        -(ln_p2 - ln_p1) / (t2 - t1)
    }

    fn forward_derivative(initial_curve: &YieldCurve, t: f64) -> f64 {
        let eps = (1.0e-3_f64).max(1.0e-3 * (1.0 + t.abs()));
        let t_minus = (t - eps).max(0.0);
        let t_plus = t + eps;

        let f_minus = Self::instantaneous_forward(initial_curve, t_minus);
        let f_plus = Self::instantaneous_forward(initial_curve, t_plus);

        (f_plus - f_minus) / (t_plus - t_minus)
    }

    fn bond_b(&self, t: f64, maturity: f64) -> f64 {
        let tau = maturity - t;
        if tau <= 0.0 {
            0.0
        } else if self.a.abs() <= 1.0e-12 {
            tau
        } else {
            (1.0 - (-self.a * tau).exp()) / self.a
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn vasicek_bond_prices_match_quantlib_1_43() {
        let model = Vasicek {
            a: 0.15,
            b: 0.05,
            sigma: 0.01,
        };

        // QuantLib-Python 1.43 Vasicek(r0=.03,a=.15,b=.05,sigma=.01,
        // lambda=0).discountBond(t,T,r_t).
        let references = [
            (0.0, 0.25, 0.03, 0.992_436_413_167_702),
            (0.0, 1.0, 0.03, 0.969_075_442_577_678_9),
            (0.0, 5.0, 0.03, 0.836_593_696_385_110_6),
            (1.0, 5.0, 0.04, 0.844_319_677_072_446_5),
            (2.0, 10.0, 0.02, 0.773_808_386_486_136_3),
        ];
        for (t, maturity, short_rate, expected) in references {
            assert_relative_eq!(
                model.bond_price(t, maturity, short_rate),
                expected,
                epsilon = 3.0e-15
            );
        }
    }

    #[test]
    fn cir_bond_prices_match_quantlib_1_43() {
        let model = CIR {
            a: 0.20,
            b: 0.04,
            sigma: 0.10,
        };

        // QuantLib-Python 1.43 CoxIngersollRoss(r0=.03,theta=.04,k=.20,
        // sigma=.10).discountBond(t,T,r_t).
        let references = [
            (0.0, 0.25, 0.03, 0.992_467_794_747_472_8),
            (0.0, 1.0, 0.03, 0.969_579_552_690_866),
            (0.0, 5.0, 0.03, 0.847_811_373_675_468),
            (1.0, 5.0, 0.04, 0.854_183_588_020_182_5),
            (2.0, 10.0, 0.02, 0.792_585_079_051_245_6),
        ];
        for (t, maturity, short_rate, expected) in references {
            assert_relative_eq!(
                model.bond_price(t, maturity, short_rate),
                expected,
                epsilon = 3.0e-15
            );
        }
    }

    #[test]
    fn hull_white_calibration_reprices_input_curve_at_t0() {
        let flat_rate = 0.03;
        let initial_curve = YieldCurve::new(
            (1..=80)
                .map(|i| {
                    let t = i as f64 * 0.25;
                    (t, (-flat_rate * t).exp())
                })
                .collect(),
        );

        let mut model = HullWhite::new(0.10, 0.01);
        let theta_grid = (0..=80).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
        model.calibrate_theta(&initial_curve, &theta_grid);

        let r0 = HullWhite::instantaneous_forward(&initial_curve, 0.0);

        for maturity in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let p_model = model.bond_price(0.0, maturity, r0, &initial_curve);
            let p_curve = initial_curve.discount_factor(maturity);
            assert_relative_eq!(p_model, p_curve, epsilon = 2.0e-15);
        }

        assert!(model.theta_at(1.0).is_finite());
    }
}
