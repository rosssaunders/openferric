//! Classic one-factor short-rate models: Vasicek, CIR, and calibrated Hull-White.
//!
//! [`Vasicek`] and [`CIR`] expose closed-form zero-coupon bond prices,
//! while [`HullWhite`] adds theta calibration from an initial curve and model-consistent bond pricing.
//! References: Vasicek (1977), Cox-Ingersoll-Ross (1985), Hull and White (1990).
//! Calibration utilities approximate instantaneous forward and derivative terms via finite differences,
//! with special handling of near-zero mean-reversion limits.
//! Numerical caveat: interpolation and finite-difference step size can impact `theta(t)` smoothness.
//! Use this module for rate-model benchmarking and analytic fixed-income pricing prototypes.

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
        let a = ((self.b - sigma2 / (2.0 * self.a * self.a)) * (b - tau)
            - sigma2 * b * b / (4.0 * self.a))
            .exp();
        a * (-b * short_rate).exp()
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
        let gamma = (self.a * self.a + 2.0 * self.sigma * self.sigma).sqrt();
        let exp_gamma_tau = (gamma * tau).exp();

        let denominator = (gamma + self.a) * (exp_gamma_tau - 1.0) + 2.0 * gamma;
        let b = 2.0 * (exp_gamma_tau - 1.0) / denominator;

        let a = (2.0 * gamma * ((self.a + gamma) * tau * 0.5).exp() / denominator)
            .powf(2.0 * self.a * self.b / (self.sigma * self.sigma));

        a * (-b * short_rate).exp()
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

    /// Calibrates `theta(t)` on the provided time grid to fit an initial yield curve.
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
    fn vasicek_bond_price_matches_closed_form_value() {
        let model = Vasicek {
            a: 0.15,
            b: 0.05,
            sigma: 0.01,
        };
        let short_rate = 0.03;
        let maturity = 5.0;

        let price = model.bond_price(0.0, maturity, short_rate);
        let tau = maturity;
        let b = (1.0 - (-model.a * tau).exp()) / model.a;
        let sigma2 = model.sigma * model.sigma;
        let a = ((model.b - sigma2 / (2.0 * model.a * model.a)) * (b - tau)
            - sigma2 * b * b / (4.0 * model.a))
            .exp();
        let expected = a * (-b * short_rate).exp();

        assert_relative_eq!(price, expected, epsilon = 1e-12);
    }

    #[test]
    fn cir_bond_price_matches_closed_form_value() {
        let model = CIR {
            a: 0.20,
            b: 0.04,
            sigma: 0.10,
        };
        let short_rate = 0.03;
        let maturity = 5.0;

        let price = model.bond_price(0.0, maturity, short_rate);
        let gamma = (model.a * model.a + 2.0 * model.sigma * model.sigma).sqrt();
        let e = (gamma * maturity).exp();
        let denom = (gamma + model.a) * (e - 1.0) + 2.0 * gamma;
        let b = 2.0 * (e - 1.0) / denom;
        let a = (2.0 * gamma * ((model.a + gamma) * maturity * 0.5).exp() / denom)
            .powf(2.0 * model.a * model.b / (model.sigma * model.sigma));
        let expected = a * (-b * short_rate).exp();

        assert_relative_eq!(price, expected, epsilon = 1e-12);
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
            assert_relative_eq!(p_model, p_curve, epsilon = 3e-6);
        }

        assert!(model.theta_at(1.0).is_finite());
    }
}
