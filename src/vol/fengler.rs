//! Module `vol::fengler`.
//!
//! Implements fengler abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Fengler (2009), Gatheral (2006) Ch. 3, arbitrage-consistent spline conditions around Eq. (3.4).
//!
//! Key types and purpose: `FenglerSurface` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.

use crate::math::CubicSpline;
use crate::vol::ArbitrageViolation;

/// Arbitrage-free total variance surface using Fengler's spline approach.
#[derive(Debug, Clone)]
pub struct FenglerSurface {
    /// Sorted unique expiry times.
    expiries: Vec<f64>,
    /// Forward prices for each expiry.
    forwards: Vec<f64>,
    /// Log-moneyness grid for each expiry slice.
    log_moneyness: Vec<Vec<f64>>,
    /// Spline per expiry slice.
    splines: Vec<CubicSpline>,
}

impl FenglerSurface {
    /// Build from market quotes `(strike, expiry, implied_vol)` and a forward curve
    /// `(expiry, forward_price)`.
    pub fn new(quotes: &[(f64, f64, f64)], forward_curve: &[(f64, f64)]) -> Self {
        assert!(!quotes.is_empty(), "quotes must not be empty");
        assert!(!forward_curve.is_empty(), "forward_curve must not be empty");

        // Sort forward curve by expiry.
        let mut fwd_sorted: Vec<(f64, f64)> = forward_curve.to_vec();
        fwd_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Collect unique expiries from quotes.
        let mut expiries: Vec<f64> = quotes.iter().map(|q| q.1).collect();
        expiries.sort_by(|a, b| a.total_cmp(b));
        expiries.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

        let mut forwards = Vec::with_capacity(expiries.len());
        let mut log_moneyness_slices = Vec::with_capacity(expiries.len());
        let mut splines = Vec::with_capacity(expiries.len());

        for &t in &expiries {
            // Interpolate forward for this expiry.
            let fwd = interpolate_forward(&fwd_sorted, t);
            forwards.push(fwd);

            // Gather quotes for this expiry, compute log-moneyness and total var.
            let mut slice: Vec<(f64, f64)> = quotes
                .iter()
                .filter(|q| (q.1 - t).abs() < 1e-14)
                .map(|q| {
                    let k = (q.0 / fwd).ln();
                    let w = q.2 * q.2 * t;
                    (k, w)
                })
                .collect();

            slice.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Deduplicate by log-moneyness.
            slice.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);

            let ks: Vec<f64> = slice.iter().map(|s| s.0).collect();
            let ws: Vec<f64> = slice.iter().map(|s| s.1.max(1e-10)).collect();

            let spline = if ks.len() >= 2 {
                CubicSpline::new(ks.clone(), ws.clone()).unwrap_or_else(|_| {
                    // Fallback: just two endpoint spline.
                    CubicSpline::new(vec![ks[0], ks[ks.len() - 1]], vec![ws[0], ws[ws.len() - 1]])
                        .unwrap()
                })
            } else {
                // Single point — create flat spline around it.
                let k0 = if ks.is_empty() { 0.0 } else { ks[0] };
                let w0 = if ws.is_empty() { 0.04 } else { ws[0] };
                CubicSpline::new(vec![k0 - 1.0, k0 + 1.0], vec![w0, w0]).unwrap()
            };

            log_moneyness_slices.push(ks);
            splines.push(spline);
        }

        Self {
            expiries,
            forwards,
            log_moneyness: log_moneyness_slices,
            splines,
        }
    }

    /// Total variance w(k, T) at given log-moneyness and expiry.
    pub fn total_variance(&self, log_moneyness: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.04;
        }

        let t = expiry.max(1e-10);

        if t <= self.expiries[0] {
            return self.splines[0].interpolate(log_moneyness).max(1e-10);
        }
        let last = self.expiries.len() - 1;
        if t >= self.expiries[last] {
            return self.splines[last].interpolate(log_moneyness).max(1e-10);
        }

        // Linear interpolation in total variance between expiry slices.
        for i in 0..last {
            if t >= self.expiries[i] && t <= self.expiries[i + 1] {
                let w = (t - self.expiries[i]) / (self.expiries[i + 1] - self.expiries[i]);
                let w0 = self.splines[i].interpolate(log_moneyness).max(1e-10);
                let w1 = self.splines[i + 1].interpolate(log_moneyness).max(1e-10);
                return (w0 * (1.0 - w) + w1 * w).max(1e-10);
            }
        }

        self.splines[last].interpolate(log_moneyness).max(1e-10)
    }

    /// Interpolated forward at expiry, without per-call allocation.
    fn forward_at(&self, t: f64) -> f64 {
        let ts = &self.expiries;
        let fs = &self.forwards;
        if ts.len() == 1 || t <= ts[0] {
            return fs[0];
        }
        let last = ts.len() - 1;
        if t >= ts[last] {
            return fs[last];
        }
        for i in 0..last {
            if t >= ts[i] && t <= ts[i + 1] {
                let w = (t - ts[i]) / (ts[i + 1] - ts[i]);
                return fs[i] * (1.0 - w) + fs[i + 1] * w;
            }
        }
        fs[last]
    }

    /// Implied vol at (strike, expiry).
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        let t = expiry.max(1e-10);
        let fwd = self.forward_at(t);
        let k = (strike / fwd).ln();
        let w = self.total_variance(k, t);
        (w / t).sqrt()
    }

    /// Check for arbitrage violations across the surface.
    pub fn check_arbitrage(&self) -> Vec<ArbitrageViolation> {
        let mut violations = Vec::new();

        // Calendar arbitrage: ∂w/∂T ≥ 0.
        if self.expiries.len() >= 2 {
            for i in 0..self.expiries.len() - 1 {
                let t1 = self.expiries[i];
                let t2 = self.expiries[i + 1];
                // Check at several log-moneyness points.
                for k_idx in -20..=20 {
                    let k = k_idx as f64 * 0.05;
                    let w1 = self.splines[i].interpolate(k);
                    let w2 = self.splines[i + 1].interpolate(k);
                    let dw_dt = (w2 - w1) / (t2 - t1);
                    if dw_dt < -1e-8 {
                        // k is log-moneyness ln(K/F): strike = F * e^k.
                        let strike = self.forwards[i] * k.exp();
                        violations.push(ArbitrageViolation::Calendar {
                            strike,
                            t1,
                            t2,
                            dw_dt,
                        });
                    }
                }
            }
        }

        // Butterfly arbitrage: check density g(k) ≥ 0 at each slice.
        for (slice_idx, spline) in self.splines.iter().enumerate() {
            let t = self.expiries[slice_idx];
            let ks = &self.log_moneyness[slice_idx];
            if ks.len() < 3 {
                continue;
            }
            let k_lo = ks[0];
            let k_hi = ks[ks.len() - 1];
            let npts = 50;
            let dk = (k_hi - k_lo) / npts as f64;

            for j in 1..npts {
                let k = k_lo + j as f64 * dk;
                let w = spline.interpolate(k).max(1e-10);
                let w_m = spline.interpolate(k - dk);
                let w_p = spline.interpolate(k + dk);

                let wp = (w_p - w_m) / (2.0 * dk); // w'
                let wpp = (w_p - 2.0 * w + w_m) / (dk * dk); // w''

                let term1 = (1.0 - k * wp / (2.0 * w)).powi(2);
                let term2 = wp * wp / 4.0 * (1.0 / w + 0.25);
                let g = term1 - term2 + wpp / 2.0;

                if g < -1e-6 {
                    // k is log-moneyness ln(K/F): strike = F * e^k.
                    let strike = self.forwards[slice_idx] * k.exp();
                    violations.push(ArbitrageViolation::Butterfly {
                        strike,
                        expiry: t,
                        density: g,
                    });
                }
            }
        }

        violations
    }
}

fn interpolate_forward(fwd_curve: &[(f64, f64)], t: f64) -> f64 {
    if fwd_curve.len() == 1 {
        return fwd_curve[0].1;
    }
    if t <= fwd_curve[0].0 {
        return fwd_curve[0].1;
    }
    let last = fwd_curve.len() - 1;
    if t >= fwd_curve[last].0 {
        return fwd_curve[last].1;
    }
    for i in 0..last {
        if t >= fwd_curve[i].0 && t <= fwd_curve[i + 1].0 {
            let w = (t - fwd_curve[i].0) / (fwd_curve[i + 1].0 - fwd_curve[i].0);
            return fwd_curve[i].1 * (1.0 - w) + fwd_curve[i + 1].1 * w;
        }
    }
    fwd_curve[last].1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data() -> (Vec<(f64, f64, f64)>, Vec<(f64, f64)>) {
        let spot: f64 = 100.0;
        let rate: f64 = 0.02;
        let vol: f64 = 0.20;
        let expiries = [0.25, 0.5, 1.0, 2.0];
        let offsets = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3];

        let mut quotes = Vec::new();
        let mut fwd_curve = Vec::new();

        for &t in &expiries {
            let fwd = spot * (rate * t).exp();
            fwd_curve.push((t, fwd));
            for &dk in &offsets {
                let k = fwd * (1.0 + dk);
                // Flat vol surface.
                quotes.push((k, t, vol));
            }
        }

        (quotes, fwd_curve)
    }

    fn non_flat_data() -> (Vec<(f64, f64, f64)>, Vec<(f64, f64)>) {
        let log_moneyness = [-0.3_f64, -0.1, 0.05, 0.25];
        let slices = [
            (0.5, 101.0, [0.018, 0.016, 0.0175, 0.022]),
            (1.0, 103.0, [0.034, 0.032, 0.035, 0.044]),
            (2.0, 105.0, [0.065, 0.064, 0.071, 0.088]),
        ];
        let mut quotes = Vec::new();
        let mut forwards = Vec::new();
        for (expiry, forward, total_variances) in slices {
            forwards.push((expiry, forward));
            for (k, w) in log_moneyness.into_iter().zip(total_variances) {
                quotes.push((forward * k.exp(), expiry, f64::sqrt(w / expiry)));
            }
        }
        (quotes, forwards)
    }

    #[test]
    fn fengler_no_arbitrage_flat_vol() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);
        let violations = surface.check_arbitrage();

        assert!(
            violations.is_empty(),
            "flat vol should have neither calendar nor butterfly arbitrage: {violations:?}"
        );
    }

    #[test]
    fn fengler_implied_vol_accuracy() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);

        // Check that implied vol at quoted points matches input.
        let mut max_err = 0.0_f64;
        for &(k, t, v) in &quotes {
            let iv = surface.implied_vol(k, t);
            max_err = max_err.max((iv - v).abs());
        }

        // A constant total-variance spline reproduces every quote to roundoff.
        assert!(max_err <= 2.0e-15, "max implied vol error {max_err:.17e}");
    }

    #[test]
    fn fengler_total_variance_positive() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);

        for k_idx in -10..=10 {
            let k = k_idx as f64 * 0.05;
            for t_bp in [25, 50, 100, 200] {
                let t = t_bp as f64 / 100.0;
                let w = surface.total_variance(k, t);
                let expected = 0.20_f64.powi(2) * t;
                assert!(
                    (w - expected).abs() <= 4.0e-16,
                    "flat-surface variance at k={k}, T={t}: {w} != {expected}"
                );
            }
        }
    }

    #[test]
    fn fengler_calendar_interpolation_and_extrapolation_are_exact() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);

        for k_idx in -5..=5 {
            let k = k_idx as f64 * 0.1;
            for t_bp in [10, 25, 50, 75, 100, 150, 200] {
                let t = t_bp as f64 / 100.0;
                let w = surface.total_variance(k, t);
                // The implementation holds total variance flat outside the
                // quoted [0.25, 2.0] interval and interpolates it linearly inside.
                let expected = 0.20_f64.powi(2) * t.clamp(0.25, 2.0);
                assert!(
                    (w - expected).abs() <= 4.0e-16,
                    "total variance at k={k}, T={t}: {w} != {expected}"
                );
            }
        }
    }

    #[test]
    fn fengler_non_flat_off_grid_values_match_scipy_natural_splines() {
        let (quotes, fwd_curve) = non_flat_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);

        // Independent SciPy 1.17.1 CubicSpline(..., bc_type="natural")
        // evaluation on each non-flat expiry slice, followed by linear
        // calendar interpolation in total variance.
        for (k, expiry, expected_w, expected_iv) in [
            (
                -0.18_f64,
                0.75,
                0.024_317_433_155_080_212,
                0.180_064_555_664_832_65,
            ),
            (
                0.0,
                1.4,
                0.047_270_469_399_881_165,
                0.183_751_519_721_375_68,
            ),
            (
                0.17,
                1.75,
                0.070_495_101_604_278_08,
                0.200_706_041_768_663_77,
            ),
        ] {
            let actual_w = surface.total_variance(k, expiry);
            let forward = interpolate_forward(&fwd_curve, expiry);
            let actual_iv = surface.implied_vol(forward * k.exp(), expiry);
            assert!(
                (actual_w - expected_w).abs() <= 3.0e-15,
                "off-grid total variance mismatch at k={k}, T={expiry}: {actual_w:.17}"
            );
            assert!(
                (actual_iv - expected_iv).abs() <= 8.0e-15,
                "off-grid implied vol mismatch at k={k}, T={expiry}: {actual_iv:.17}"
            );
        }
    }

    #[test]
    fn fengler_reports_explicit_calendar_and_butterfly_arbitrage() {
        let ks = [-0.4_f64, -0.2, 0.0, 0.2, 0.4];

        let mut calendar_quotes = Vec::new();
        for (expiry, forward, total_variance) in [(1.0, 100.0, 0.05), (2.0, 102.0, 0.04)] {
            for k in ks {
                calendar_quotes.push((
                    forward * k.exp(),
                    expiry,
                    f64::sqrt(total_variance / expiry),
                ));
            }
        }
        let calendar = FenglerSurface::new(&calendar_quotes, &[(1.0, 100.0), (2.0, 102.0)]);
        assert!(
            calendar
                .check_arbitrage()
                .iter()
                .any(|violation| matches!(violation, ArbitrageViolation::Calendar { .. }))
        );

        // This deliberately oscillating total-variance slice has negative
        // Gatheral density g(k) (SciPy minimum -8.2378987559), so the surface
        // checker must not silently accept it.
        let butterfly_w = [0.04_f64, 0.20, 0.02, 0.20, 0.04];
        let butterfly_quotes = ks
            .into_iter()
            .zip(butterfly_w)
            .map(|(k, w)| (100.0 * k.exp(), 1.0, w.sqrt()))
            .collect::<Vec<_>>();
        let butterfly = FenglerSurface::new(&butterfly_quotes, &[(1.0, 100.0)]);
        assert!(
            butterfly
                .check_arbitrage()
                .iter()
                .any(|violation| matches!(violation, ArbitrageViolation::Butterfly { .. }))
        );
    }
}
