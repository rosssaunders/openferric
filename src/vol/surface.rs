//! Module `vol::surface`.
//!
//! Implements surface workflows with concrete routines such as `calibrate_svi`.
//!
//! References: Gatheral (2006), Derman and Kani (1994), static-arbitrage constraints around total variance Eq. (2.2).
//!
//! Key types and purpose: `SviParams`, `VolSurface` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
use crate::math::CubicSpline;
use crate::vol::forward::{
    AtmSkewTermStructure, ForwardVarianceCurve, ForwardVarianceSource, VixSettings, VixStyleIndex,
    vix_style_index_from_surface,
};

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl SviParams {
    pub fn total_variance(&self, k: f64) -> f64 {
        self.a
            + self.b
                * (self.rho * (k - self.m)
                    + ((k - self.m).powi(2) + self.sigma * self.sigma).sqrt())
    }

    pub fn dw_dk(&self, k: f64) -> f64 {
        let x = k - self.m;
        self.b * (self.rho + x / (x * x + self.sigma * self.sigma).sqrt())
    }
}

fn svi_objective_weighted(params: SviParams, points: &[(f64, f64)], weights: &[f64]) -> f64 {
    points
        .iter()
        .zip(weights.iter())
        .map(|((k, w), &wt)| {
            let err = params.total_variance(*k) - *w;
            wt * err * err
        })
        .sum()
}

/// Analytic Jacobian row for SVI: partial derivatives of w(k) w.r.t. [a, b, rho, m, sigma].
#[inline]
pub fn svi_jacobian_row(p: &SviParams, k: f64) -> [f64; 5] {
    let x = k - p.m;
    let s2 = x * x + p.sigma * p.sigma;
    let s = s2.sqrt();
    // dw/da = 1
    // dw/db = rho*x + s
    // dw/drho = b*x
    // dw/dm = b*(-rho - x/s)
    // dw/dsigma = b*sigma/s
    [
        1.0,
        p.rho * x + s,
        p.b * x,
        p.b * (-p.rho - x / s),
        p.b * p.sigma / s,
    ]
}

#[inline]
fn project(mut p: SviParams) -> SviParams {
    // SVI allows negative a (min total variance = a + b*sigma*sqrt(1-rho^2))
    // so only enforce a loose lower bound; positivity is maintained by the data.
    p.b = p.b.max(1e-8);
    p.rho = p.rho.clamp(-0.999, 0.999);
    p.sigma = p.sigma.max(1e-6);
    p
}

/// Solve 5x5 linear system Ax = b via LU decomposition (no external dependency).
/// Returns None if singular.
fn solve5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut lu = *a;
    let mut piv = [0usize; 5];
    for i in 0..5 {
        piv[i] = i;
    }

    for col in 0..5 {
        // Partial pivot
        let mut max_val = lu[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..5 {
            let v = lu[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            lu.swap(col, max_row);
            piv.swap(col, max_row);
        }
        let diag = lu[col][col];
        for row in (col + 1)..5 {
            lu[row][col] /= diag;
            let factor = lu[row][col];
            for j in (col + 1)..5 {
                lu[row][j] -= factor * lu[col][j];
            }
        }
    }

    // Forward substitution (Ly = Pb)
    let mut y = [0.0; 5];
    for i in 0..5 {
        y[i] = b[piv[i]];
        for j in 0..i {
            y[i] -= lu[i][j] * y[j];
        }
    }
    // Back substitution (Ux = y)
    let mut x = [0.0; 5];
    for i in (0..5).rev() {
        x[i] = y[i];
        for j in (i + 1)..5 {
            x[i] -= lu[i][j] * x[j];
        }
        if lu[i][i].abs() < 1e-15 {
            return None;
        }
        x[i] /= lu[i][i];
    }
    Some(x)
}

/// Levenberg-Marquardt SVI calibration with analytic Jacobian and optional weights.
fn lm_svi(
    points: &[(f64, f64)],
    weights: &[f64],
    start: SviParams,
    max_iter: usize,
) -> (SviParams, f64) {
    let n = points.len();
    let mut p = project(start);
    let mut obj = svi_objective_weighted(p, points, weights);
    let mut lambda = 1e-3;

    for _ in 0..max_iter {
        // Build J^T W J and J^T W r using analytic Jacobian
        let mut jtj = [[0.0f64; 5]; 5];
        let mut jtr = [0.0f64; 5];

        for i in 0..n {
            let (k, w) = points[i];
            let r = p.total_variance(k) - w;
            let row = svi_jacobian_row(&p, k);
            let wi = weights[i];
            for a in 0..5 {
                jtr[a] += wi * row[a] * r;
                for b in a..5 {
                    jtj[a][b] += wi * row[a] * row[b];
                }
            }
        }
        // Fill symmetric lower triangle
        for a in 0..5 {
            for b in 0..a {
                jtj[a][b] = jtj[b][a];
            }
        }

        // Damping
        let mut damped = jtj;
        for d in 0..5 {
            damped[d][d] += lambda;
        }

        let Some(delta) = solve5(&damped, &jtr) else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
            continue;
        };

        let delta_norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let candidate = project(SviParams {
            a: p.a - delta[0],
            b: p.b - delta[1],
            rho: p.rho - delta[2],
            m: p.m - delta[3],
            sigma: p.sigma - delta[4],
        });
        let cand_obj = svi_objective_weighted(candidate, points, weights);

        if cand_obj < obj {
            let improvement = obj - cand_obj;
            p = candidate;
            obj = cand_obj;
            lambda = (lambda * 0.5).max(1e-8);
            if improvement < 1e-12 || delta_norm < 1e-8 {
                break;
            }
        } else {
            lambda *= 2.0;
            if lambda > 1e10 {
                break;
            }
        }
    }

    (p, obj)
}

pub fn calibrate_svi_weighted(
    points: &[(f64, f64)],
    weights: &[f64],
    init: SviParams,
    max_iter: usize,
) -> SviParams {
    if points.is_empty() {
        return project(init);
    }

    let starts = [
        init,
        SviParams {
            a: init.a * 0.7,
            b: init.b * 1.2,
            rho: init.rho * 0.5,
            m: init.m - 0.1,
            sigma: init.sigma * 0.8,
        },
    ];

    let mut best = project(init);
    let mut best_obj = svi_objective_weighted(best, points, weights);

    for start in starts {
        let (p, obj) = lm_svi(points, weights, start, max_iter);
        if obj < best_obj {
            best = p;
            best_obj = obj;
        }
    }

    best
}

pub fn calibrate_svi(
    points: &[(f64, f64)],
    init: SviParams,
    _max_iter: usize,
    _learning_rate: f64,
) -> SviParams {
    if points.is_empty() {
        return project(init);
    }
    let uniform: Vec<f64> = vec![1.0; points.len()];
    calibrate_svi_weighted(points, &uniform, init, 150)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

        let ws: Vec<f64> = self
            .slices
            .iter()
            .map(|p| p.total_variance(k).max(1e-10))
            .collect();

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

    /// Native expiry grid.
    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    /// Anchor forward level used to convert strike to log-moneyness.
    pub fn forward(&self) -> f64 {
        self.forward
    }

    /// Forward level at expiry (constant for this parametric representation).
    pub fn forward_price(&self, _expiry: f64) -> f64 {
        self.forward
    }

    /// Builds an ATM forward-variance curve on the provided expiry grid.
    pub fn forward_variance_curve(&self, expiries: &[f64]) -> Result<ForwardVarianceCurve, String> {
        ForwardVarianceCurve::from_surface(self, expiries)
    }

    /// Builds an ATM skew term structure on the provided expiry grid.
    pub fn atm_skew_term_structure(
        &self,
        expiries: &[f64],
    ) -> Result<AtmSkewTermStructure, String> {
        AtmSkewTermStructure::from_surface(self, expiries)
    }

    /// Computes a VIX-style index from this surface for a given risk-free rate.
    pub fn vix_style_index(
        &self,
        risk_free_rate: f64,
        settings: VixSettings,
    ) -> Result<VixStyleIndex, String> {
        vix_style_index_from_surface(self, risk_free_rate, settings)
    }
}

impl ForwardVarianceSource for VolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        VolSurface::vol(self, strike, expiry)
    }

    fn forward_price(&self, expiry: f64) -> f64 {
        VolSurface::forward_price(self, expiry)
    }

    fn expiries(&self) -> &[f64] {
        &self.expiries
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

    #[test]
    fn svi_analytic_jacobian_matches_finite_difference() {
        let p = SviParams {
            a: 0.04,
            b: 0.4,
            rho: -0.4,
            m: 0.05,
            sigma: 0.1,
        };
        let h = 1e-6;

        for i in -10..=10 {
            let k = i as f64 * 0.1;
            let analytic = svi_jacobian_row(&p, k);

            // Finite-difference partials
            let fd = [
                // dw/da
                (SviParams { a: p.a + h, ..p }.total_variance(k)
                    - SviParams { a: p.a - h, ..p }.total_variance(k))
                    / (2.0 * h),
                // dw/db
                (SviParams { b: p.b + h, ..p }.total_variance(k)
                    - SviParams { b: p.b - h, ..p }.total_variance(k))
                    / (2.0 * h),
                // dw/drho
                (SviParams { rho: p.rho + h, ..p }.total_variance(k)
                    - SviParams { rho: p.rho - h, ..p }.total_variance(k))
                    / (2.0 * h),
                // dw/dm
                (SviParams { m: p.m + h, ..p }.total_variance(k)
                    - SviParams { m: p.m - h, ..p }.total_variance(k))
                    / (2.0 * h),
                // dw/dsigma
                (SviParams { sigma: p.sigma + h, ..p }.total_variance(k)
                    - SviParams { sigma: p.sigma - h, ..p }.total_variance(k))
                    / (2.0 * h),
            ];

            for j in 0..5 {
                let err = (analytic[j] - fd[j]).abs();
                assert!(
                    err < 1e-4,
                    "Jacobian mismatch at k={k}, param {j}: analytic={}, fd={}, err={err}",
                    analytic[j], fd[j]
                );
            }
        }
    }

    #[test]
    fn calibrate_svi_weighted_uniform_matches_unweighted() {
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

        let fit_unweighted = calibrate_svi(&points, init, 150, 0.0);
        let uniform: Vec<f64> = vec![1.0; points.len()];
        let fit_weighted = calibrate_svi_weighted(&points, &uniform, init, 150);

        // Both should produce equivalent fits (same MSE within tolerance)
        let mse_uw: f64 = points
            .iter()
            .map(|(k, w)| (fit_unweighted.total_variance(*k) - *w).powi(2))
            .sum::<f64>()
            / points.len() as f64;
        let mse_w: f64 = points
            .iter()
            .map(|(k, w)| (fit_weighted.total_variance(*k) - *w).powi(2))
            .sum::<f64>()
            / points.len() as f64;

        assert!(mse_uw < 1e-6, "unweighted MSE={mse_uw}");
        assert!(mse_w < 1e-6, "weighted MSE={mse_w}");
        // Both should be very close
        assert!(
            (mse_uw - mse_w).abs() < 1e-8,
            "MSE difference too large: unweighted={mse_uw}, weighted={mse_w}"
        );
    }
}
