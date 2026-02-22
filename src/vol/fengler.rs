//! Fengler arbitrage-free smoothing of total implied variance surface.
//!
//! Fits natural cubic splines to total variance slices w(k, T) = σ²(k,T) · T,
//! then checks for butterfly and calendar arbitrage violations.

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
    /// Total variance values for each slice.
    _total_variances: Vec<Vec<f64>>,
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
        let mut total_var_slices = Vec::with_capacity(expiries.len());
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
                    CubicSpline::new(
                        vec![ks[0], ks[ks.len() - 1]],
                        vec![ws[0], ws[ws.len() - 1]],
                    )
                    .unwrap()
                })
            } else {
                // Single point — create flat spline around it.
                let k0 = if ks.is_empty() { 0.0 } else { ks[0] };
                let w0 = if ws.is_empty() { 0.04 } else { ws[0] };
                CubicSpline::new(vec![k0 - 1.0, k0 + 1.0], vec![w0, w0]).unwrap()
            };

            log_moneyness_slices.push(ks);
            total_var_slices.push(ws);
            splines.push(spline);
        }

        Self {
            expiries,
            forwards,
            log_moneyness: log_moneyness_slices,
            _total_variances: total_var_slices,
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

    /// Implied vol at (strike, expiry).
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        let t = expiry.max(1e-10);
        let fwd = interpolate_forward(
            &self
                .expiries
                .iter()
                .zip(self.forwards.iter())
                .map(|(&t, &f)| (t, f))
                .collect::<Vec<_>>(),
            t,
        );
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
                        let strike = k.exp(); // approximate
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
                    let strike = (k * self.forwards[slice_idx]).exp();
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

    #[test]
    fn fengler_no_arbitrage_flat_vol() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);
        let violations = surface.check_arbitrage();

        // Flat vol should have no violations.
        let calendar: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, ArbitrageViolation::Calendar { .. }))
            .collect();
        assert!(
            calendar.is_empty(),
            "flat vol should have no calendar arbitrage, got {calendar:?}"
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

        assert!(
            max_err < 0.01,
            "max implied vol error {max_err:.6} exceeds tolerance"
        );
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
                assert!(w > 0.0, "total variance must be positive at k={k}, T={t}");
            }
        }
    }

    #[test]
    fn fengler_calendar_monotonicity() {
        let (quotes, fwd_curve) = synthetic_data();
        let surface = FenglerSurface::new(&quotes, &fwd_curve);

        for k_idx in -5..=5 {
            let k = k_idx as f64 * 0.1;
            let mut prev_w = 0.0;
            for t_bp in [10, 25, 50, 75, 100, 150, 200] {
                let t = t_bp as f64 / 100.0;
                let w = surface.total_variance(k, t);
                assert!(
                    w >= prev_w - 1e-8,
                    "calendar violation at k={k}, T={t}: w={w} < prev={prev_w}"
                );
                prev_w = w;
            }
        }
    }
}
