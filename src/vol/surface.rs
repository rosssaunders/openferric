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
#[allow(clippy::needless_range_loop)]
fn solve5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut lu = *a;
    let mut piv = [0usize; 5];
    for (i, p) in piv.iter_mut().enumerate() {
        *p = i;
    }

    for col in 0..5 {
        // Partial pivot
        let mut max_val = lu[col][col].abs();
        let mut max_row = col;
        for (row, lu_row) in lu.iter().enumerate().skip(col + 1) {
            let v = lu_row[col].abs();
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
#[allow(clippy::needless_range_loop)]
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
        for (d, row) in damped.iter_mut().enumerate() {
            row[d] += lambda;
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
        SviParams {
            a: init.a * 1.3,
            b: init.b * 0.8,
            rho: (init.rho + 0.2).clamp(-0.9, 0.9),
            m: init.m + 0.1,
            sigma: init.sigma * 1.2,
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

        let surface = Self {
            expiries,
            slices: params,
            forward,
        };
        surface.validate()?;
        Ok(surface)
    }

    /// Validates the serialized representation and the SVI slice domains.
    ///
    /// This is intentionally callable after deserialization, which bypasses
    /// [`VolSurface::new`].
    pub fn validate(&self) -> Result<(), String> {
        if !self.forward.is_finite() || self.forward <= 0.0 {
            return Err("surface forward must be finite and > 0".to_string());
        }
        if self.expiries.is_empty() || self.expiries.len() != self.slices.len() {
            return Err("surface expiries and slices must have equal non-zero length".to_string());
        }
        if self
            .expiries
            .iter()
            .any(|expiry| !expiry.is_finite() || *expiry <= 0.0)
            || self.expiries.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(
                "surface expiries must be finite, positive, and strictly increasing".to_string(),
            );
        }
        for params in &self.slices {
            if !params.a.is_finite()
                || !params.b.is_finite()
                || !params.rho.is_finite()
                || !params.m.is_finite()
                || !params.sigma.is_finite()
            {
                return Err("SVI parameters must be finite".to_string());
            }
            if params.b < 0.0 || !(-1.0..=1.0).contains(&params.rho) || params.sigma <= 0.0 {
                return Err("SVI requires b >= 0, rho in [-1, 1], and sigma > 0".to_string());
            }
            let min_variance =
                params.a + params.b * params.sigma * (1.0 - params.rho * params.rho).sqrt();
            if !min_variance.is_finite() || min_variance < 0.0 {
                return Err("SVI minimum total variance must be finite and >= 0".to_string());
            }
        }
        Ok(())
    }

    /// Total implied variance `w(K, T)`.
    ///
    /// Across maturities, total variance is interpolated piecewise-linearly in `T`
    /// at fixed log-moneyness. Linear interpolation is monotone: if the slice total
    /// variances at the knots are non-decreasing in `T`, the interpolated value is
    /// non-decreasing everywhere, so no calendar arbitrage (negative forward
    /// variance) can be introduced between knots — unlike a natural cubic spline,
    /// which can overshoot. The query allocates nothing: it does a binary search
    /// over the expiry grid and evaluates only the two bracketing SVI slices.
    pub fn total_variance(&self, strike: f64, expiry: f64) -> f64 {
        let k = (strike / self.forward).ln();
        let n = self.expiries.len();
        if n == 1 {
            return self.slices[0].total_variance(k).max(1e-10);
        }

        let t = expiry.clamp(self.expiries[0], self.expiries[n - 1]);
        let i = match self.expiries.binary_search_by(|e| e.total_cmp(&t)) {
            // Exact knot: return the slice value unchanged.
            Ok(j) => return self.slices[j].total_variance(k).max(1e-10),
            // After clamping, t lies strictly inside (expiries[0], expiries[n-1]),
            // so the insertion point j is in 1..n and [j-1, j] brackets t.
            Err(j) => j - 1,
        };

        let t0 = self.expiries[i];
        let t1 = self.expiries[i + 1];
        let w0 = self.slices[i].total_variance(k).max(1e-10);
        let w1 = self.slices[i + 1].total_variance(k).max(1e-10);
        let wt = w0 + (w1 - w0) * (t - t0) / (t1 - t0);
        wt.max(1e-10)
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
    use approx::assert_relative_eq;

    #[test]
    fn svi_total_variance_and_slope_match_full_precision_grid() {
        let p = SviParams {
            a: 0.02,
            b: 0.08,
            rho: 0.95,
            m: -1.0,
            sigma: 0.3,
        };

        let reference = [
            (-0.2, 0.14915202996254026, 0.15090633420552355),
            (0.0, 0.179_522_452_071_284_4, 0.152_626_102_817_692_1),
            (0.4, 0.240_942_568_506_210_8, 0.15422419312619276),
            (0.7, 0.287_301_412_013_056_6, 0.15478268470543494),
            (0.97, 0.32913693761956414, 0.15508820849443233),
        ];
        for (k, expected_w, expected_dw) in reference {
            assert_relative_eq!(
                p.total_variance(k),
                expected_w,
                epsilon = 4.0 * f64::EPSILON
            );
            assert_relative_eq!(p.dw_dk(k), expected_dw, epsilon = 4.0 * f64::EPSILON);
            // The slope sign is a supplemental shape check; the numerical
            // values above are the actual regression oracle.
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

        let fit_reference = SviParams {
            a: 0.009999990595756254,
            b: 0.20000000850633065,
            rho: -0.24999998115005637,
            m: 0.05000000476789586,
            sigma: 0.3000000366251351,
        };
        assert_relative_eq!(fit.a, fit_reference.a, epsilon = 2.0e-15);
        assert_relative_eq!(fit.b, fit_reference.b, epsilon = 2.0e-15);
        assert_relative_eq!(fit.rho, fit_reference.rho, epsilon = 2.0e-15);
        assert_relative_eq!(fit.m, fit_reference.m, epsilon = 2.0e-15);
        assert_relative_eq!(fit.sigma, fit_reference.sigma, epsilon = 2.0e-15);

        let max_repricing_error = points
            .iter()
            .map(|(k, w)| (fit.total_variance(*k) - *w).abs())
            .fold(0.0_f64, f64::max);
        let mse = points
            .iter()
            .map(|(k, w)| (fit.total_variance(*k) - *w).powi(2))
            .sum::<f64>()
            / points.len() as f64;

        // These are measured convergence budgets for the fixed LM grid: the
        // previous 1e-6 MSE bound allowed economically material misfits.
        assert!(max_repricing_error < 7.8e-10);
        assert!(mse < 2.0e-19);
    }

    #[test]
    fn vol_surface_matches_full_precision_strike_expiry_grid() {
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
        let expiries = [0.5, 0.75, 1.0, 1.25, 1.5];
        let reference = [
            (
                80.0,
                [
                    0.36594955409923696,
                    0.3127081084867277,
                    0.28234731023564824,
                    0.2624503554725739,
                    0.24830152251096593,
                ],
            ),
            (
                100.0,
                [
                    0.3082207001484488,
                    0.26639569566092214,
                    0.2427962108435797,
                    0.22746428291052642,
                    0.2166410241236256,
                ],
            ),
            (
                120.0,
                [
                    0.3191972855948313,
                    0.27471787037218826,
                    0.24952236149730445,
                    0.23310149986305992,
                    0.22147891428430055,
                ],
            ),
        ];

        for (strike, expected_vols) in reference {
            for (expiry, expected) in expiries.iter().zip(expected_vols) {
                let got = surface.vol(strike, *expiry);
                assert_relative_eq!(got, expected, epsilon = 8.0 * f64::EPSILON);
                assert!(got > 0.0);
            }
        }
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
        // A five-point stencil at h=1e-4 has a measured worst-case discrepancy
        // of 1.16e-12 on this grid (central h=1e-6 previously used 1e-4).
        let h = 1e-4;

        for i in -10..=10 {
            let k = i as f64 * 0.1;
            let analytic = svi_jacobian_row(&p, k);

            for (j, analytic_value) in analytic.iter().enumerate() {
                let bumped = |bump: f64| {
                    let mut q = p;
                    match j {
                        0 => q.a += bump,
                        1 => q.b += bump,
                        2 => q.rho += bump,
                        3 => q.m += bump,
                        _ => q.sigma += bump,
                    }
                    q.total_variance(k)
                };
                let fd = (-bumped(2.0 * h) + 8.0 * bumped(h) - 8.0 * bumped(-h) + bumped(-2.0 * h))
                    / (12.0 * h);
                let err = (*analytic_value - fd).abs();
                assert!(
                    err < 2.0e-12,
                    "Jacobian mismatch at k={k}, param {j}: analytic={}, fd={fd}, err={err}",
                    analytic_value,
                );
            }
        }
    }

    #[test]
    fn total_variance_has_no_negative_forward_variance_between_knots() {
        // Steep term structure: short-dated variance is tiny, the next two knots
        // jump sharply and then flatten. A natural cubic spline through these
        // knots overshoots and creates negative forward variance between the
        // 1.0y and 2.0y knots; piecewise-linear interpolation cannot.
        let mk = |a: f64| SviParams {
            a,
            b: 0.05,
            rho: -0.3,
            m: 0.0,
            sigma: 0.2,
        };
        let surface = VolSurface::new(
            vec![
                (0.1, mk(0.001)),
                (0.5, mk(0.09)),
                (1.0, mk(0.25)),
                (2.0, mk(0.26)),
            ],
            100.0,
        )
        .unwrap();

        for &strike in &[70.0, 85.0, 100.0, 115.0, 130.0] {
            let slice_w: Vec<f64> = surface
                .slices
                .iter()
                .map(|p| p.total_variance((strike / 100.0_f64).ln()))
                .collect();
            let mut prev_w = 0.0;
            for i in 0..=400 {
                let t = 0.1 + i as f64 * (2.0 - 0.1) / 400.0;
                let w = surface.total_variance(strike, t);
                let interval = surface
                    .expiries
                    .partition_point(|expiry| *expiry <= t)
                    .saturating_sub(1)
                    .min(surface.expiries.len() - 2);
                let t0 = surface.expiries[interval];
                let t1 = surface.expiries[interval + 1];
                let expected = slice_w[interval]
                    + (slice_w[interval + 1] - slice_w[interval]) * (t - t0) / (t1 - t0);
                assert_relative_eq!(w, expected, epsilon = 8.0 * f64::EPSILON);
                // Calendar monotonicity supplements the exact piecewise-linear
                // total-variance identity.
                assert!(
                    w >= prev_w,
                    "negative forward variance at strike={strike}, t={t}: w={w} < prev={prev_w}"
                );
                prev_w = w;
            }
        }
    }

    #[test]
    fn total_variance_matches_slice_values_at_knots() {
        let p1 = SviParams {
            a: 0.01,
            b: 0.15,
            rho: -0.2,
            m: 0.0,
            sigma: 0.25,
        };
        let p2 = SviParams {
            a: 0.05,
            b: 0.18,
            rho: -0.25,
            m: 0.02,
            sigma: 0.28,
        };
        let p3 = SviParams {
            a: 0.09,
            b: 0.2,
            rho: -0.3,
            m: 0.03,
            sigma: 0.3,
        };

        let expiries = [0.25, 1.0, 2.0];
        let params = [p1, p2, p3];
        let surface = VolSurface::new(
            expiries
                .iter()
                .copied()
                .zip(params.iter().copied())
                .collect(),
            100.0,
        )
        .unwrap();

        for (t, p) in expiries.iter().zip(params.iter()) {
            for &strike in &[80.0, 100.0, 120.0] {
                let k = (strike / 100.0_f64).ln();
                let expected = p.total_variance(k).max(1e-10);
                let got = surface.total_variance(strike, *t);
                assert_relative_eq!(got, expected, epsilon = 8.0 * f64::EPSILON);
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

        // `calibrate_svi` delegates to the uniformly-weighted path, so the
        // fitted parameters and residual must be bit-for-bit identical.
        assert_eq!(fit_unweighted, fit_weighted);
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

        assert_eq!(mse_uw.to_bits(), mse_w.to_bits());
    }
}
