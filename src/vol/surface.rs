use crate::math::CubicSpline;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl SviParams {
    pub fn total_variance(&self, k: f64) -> f64 {
        self.a + self.b * (self.rho * (k - self.m) + ((k - self.m).powi(2) + self.sigma * self.sigma).sqrt())
    }

    pub fn dw_dk(&self, k: f64) -> f64 {
        let x = k - self.m;
        self.b * (self.rho + x / (x * x + self.sigma * self.sigma).sqrt())
    }
}

fn svi_objective(params: SviParams, points: &[(f64, f64)]) -> f64 {
    points
        .iter()
        .map(|(k, w)| {
            let err = params.total_variance(*k) - *w;
            err * err
        })
        .sum()
}

pub fn calibrate_svi(
    points: &[(f64, f64)],
    init: SviParams,
    max_iter: usize,
    learning_rate: f64,
) -> SviParams {
    fn project(mut p: SviParams) -> SviParams {
        p.a = p.a.max(1e-8);
        p.b = p.b.max(1e-8);
        p.rho = p.rho.clamp(-0.999, 0.999);
        p.sigma = p.sigma.max(1e-6);
        p
    }

    if points.is_empty() {
        return project(init);
    }

    let starts = vec![
        init,
        SviParams {
            a: init.a * 0.7,
            b: init.b * 1.2,
            rho: init.rho * 0.5,
            m: init.m - 0.1,
            sigma: init.sigma * 0.8,
        },
        SviParams {
            a: init.a * 1.3,
            b: init.b * 0.8,
            rho: (init.rho + 0.2).clamp(-0.9, 0.9),
            m: init.m + 0.1,
            sigma: init.sigma * 1.2,
        },
    ];

    let mut best = project(init);
    let mut best_obj = svi_objective(best, points);
    let eps = 1e-5;

    for start in starts {
        let mut p = project(start);
        let mut obj = svi_objective(p, points);
        let mut lr = learning_rate.max(1e-6);

        for _ in 0..max_iter {
            let mut g = [0.0_f64; 5];

            let mut p_plus = p;
            p_plus.a += eps;
            let mut p_minus = p;
            p_minus.a -= eps;
            g[0] = (svi_objective(project(p_plus), points) - svi_objective(project(p_minus), points))
                / (2.0 * eps);

            p_plus = p;
            p_plus.b += eps;
            p_minus = p;
            p_minus.b -= eps;
            g[1] = (svi_objective(project(p_plus), points) - svi_objective(project(p_minus), points))
                / (2.0 * eps);

            p_plus = p;
            p_plus.rho += eps;
            p_minus = p;
            p_minus.rho -= eps;
            g[2] = (svi_objective(project(p_plus), points) - svi_objective(project(p_minus), points))
                / (2.0 * eps);

            p_plus = p;
            p_plus.m += eps;
            p_minus = p;
            p_minus.m -= eps;
            g[3] = (svi_objective(project(p_plus), points) - svi_objective(project(p_minus), points))
                / (2.0 * eps);

            p_plus = p;
            p_plus.sigma += eps;
            p_minus = p;
            p_minus.sigma -= eps;
            g[4] = (svi_objective(project(p_plus), points) - svi_objective(project(p_minus), points))
                / (2.0 * eps);

            let grad_norm = (g.iter().map(|v| v * v).sum::<f64>()).sqrt();
            if grad_norm < 1e-12 {
                break;
            }

            let mut improved = false;
            let mut trial_lr = lr;
            for _ in 0..12 {
                let candidate = project(SviParams {
                    a: p.a - trial_lr * g[0],
                    b: p.b - trial_lr * g[1],
                    rho: p.rho - trial_lr * g[2],
                    m: p.m - trial_lr * g[3],
                    sigma: p.sigma - trial_lr * g[4],
                });
                let cand_obj = svi_objective(candidate, points);
                if cand_obj < obj {
                    p = candidate;
                    obj = cand_obj;
                    lr = trial_lr * 1.1;
                    improved = true;
                    break;
                }
                trial_lr *= 0.5;
            }

            if !improved {
                lr *= 0.5;
                if lr < 1e-10 {
                    break;
                }
            }
        }

        if obj < best_obj {
            best = p;
            best_obj = obj;
        }
    }

    best
}

pub fn sabr_implied_vol_hagan(
    forward: f64,
    strike: f64,
    maturity: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    let f = forward.max(1e-12);
    let k = strike.max(1e-12);

    if maturity <= 0.0 {
        return 0.0;
    }

    let one_minus_beta = 1.0 - beta;
    let fk = (f * k).powf(0.5 * one_minus_beta);
    let log_fk = (f / k).ln();

    let z = if alpha.abs() > 1e-12 {
        (nu / alpha) * fk * log_fk
    } else {
        0.0
    };

    let x_z = if z.abs() < 1e-10 {
        1.0 - 0.5 * rho * z + (rho * rho - 1.0 / 3.0) * z * z
    } else {
        ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho).ln() - (1.0 - rho).ln()
    };

    let log_fk2 = log_fk * log_fk;
    let log_fk4 = log_fk2 * log_fk2;

    let denominator = fk
        * (1.0
            + one_minus_beta * one_minus_beta * log_fk2 / 24.0
            + one_minus_beta.powi(4) * log_fk4 / 1920.0);

    let term_t = 1.0
        + ((one_minus_beta * one_minus_beta * alpha * alpha) / (24.0 * fk * fk)
            + (rho * beta * nu * alpha) / (4.0 * fk)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)
            * maturity;

    if (f - k).abs() < 1e-10 {
        (alpha / f.powf(one_minus_beta)) * term_t
    } else {
        let zx = if x_z.abs() > 1e-12 { z / x_z } else { 1.0 };
        (alpha / denominator) * zx * term_t
    }
}

#[derive(Debug, Clone)]
pub struct VolSurface {
    expiries: Vec<f64>,
    slices: Vec<SviParams>,
    forward: f64,
}

impl VolSurface {
    pub fn new(mut slices: Vec<(f64, SviParams)>, forward: f64) -> Result<Self, String> {
        if slices.is_empty() {
            return Err("slices cannot be empty".to_string());
        }

        slices.sort_by(|a, b| a.0.total_cmp(&b.0));
        if slices.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err("expiries must be strictly increasing".to_string());
        }

        let expiries = slices.iter().map(|(t, _)| *t).collect();
        let params = slices.iter().map(|(_, p)| *p).collect();

        Ok(Self {
            expiries,
            slices: params,
            forward,
        })
    }

    pub fn total_variance(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.len() == 1 {
            let k = (strike / self.forward).ln();
            return self.slices[0].total_variance(k).max(1e-10);
        }

        let t = expiry.clamp(self.expiries[0], self.expiries[self.expiries.len() - 1]);
        let k = (strike / self.forward).ln();

        let ws: Vec<f64> = self.slices.iter().map(|p| p.total_variance(k).max(1e-10)).collect();

        if let Ok(spline) = CubicSpline::new(self.expiries.clone(), ws.clone()) {
            spline.interpolate(t).max(1e-10)
        } else {
            // Linear fallback.
            let i = self
                .expiries
                .windows(2)
                .position(|w| t >= w[0] && t <= w[1])
                .unwrap_or(self.expiries.len() - 2);
            let t0 = self.expiries[i];
            let t1 = self.expiries[i + 1];
            let w0 = ws[i];
            let w1 = ws[i + 1];
            let wt = w0 + (w1 - w0) * (t - t0) / (t1 - t0);
            wt.max(1e-10)
        }
    }

    pub fn vol(&self, strike: f64, expiry: f64) -> f64 {
        let t = expiry.max(1e-10);
        (self.total_variance(strike, t) / t).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svi_no_butterfly_check_dw_dk_positive_on_test_range() {
        let p = SviParams {
            a: 0.02,
            b: 0.08,
            rho: 0.95,
            m: -1.0,
            sigma: 0.3,
        };

        for i in 0..40 {
            let k = -0.2 + i as f64 * 0.03;
            assert!(p.dw_dk(k) > 0.0);
        }
    }

    #[test]
    fn calibrate_svi_recovers_synthetic_slice() {
        let true_p = SviParams {
            a: 0.01,
            b: 0.2,
            rho: -0.25,
            m: 0.05,
            sigma: 0.3,
        };

        let points: Vec<(f64, f64)> = (-8..=8)
            .map(|i| {
                let k = i as f64 * 0.1;
                (k, true_p.total_variance(k))
            })
            .collect();

        let init = SviParams {
            a: 0.03,
            b: 0.1,
            rho: 0.0,
            m: 0.0,
            sigma: 0.5,
        };

        let fit = calibrate_svi(&points, init, 4_000, 5e-3);

        let mse = points
            .iter()
            .map(|(k, w)| (fit.total_variance(*k) - *w).powi(2))
            .sum::<f64>()
            / points.len() as f64;

        assert!(mse < 1e-6);
    }

    #[test]
    fn sabr_hagan_vol_is_positive() {
        let vol = sabr_implied_vol_hagan(100.0, 110.0, 1.0, 0.25, 0.7, -0.3, 0.6);
        assert!(vol > 0.0);
    }

    #[test]
    fn vol_surface_interpolates_across_expiry() {
        let p1 = SviParams {
            a: 0.01,
            b: 0.15,
            rho: -0.2,
            m: 0.0,
            sigma: 0.25,
        };
        let p2 = SviParams {
            a: 0.02,
            b: 0.18,
            rho: -0.2,
            m: 0.0,
            sigma: 0.28,
        };

        let surface = VolSurface::new(vec![(0.5, p1), (1.5, p2)], 100.0).unwrap();
        let v_short = surface.vol(100.0, 0.5);
        let v_mid = surface.vol(100.0, 1.0);
        let v_long = surface.vol(100.0, 1.5);

        assert!(v_short > 0.0);
        assert!(v_mid > 0.0);
        assert!(v_long > 0.0);
        assert!(v_mid >= v_short * 0.7);
    }
}
