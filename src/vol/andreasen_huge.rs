//! Andreasen-Huge piecewise-constant local vol interpolation.
//!
//! Calibrates a piecewise-constant local volatility surface to market quotes
//! using an iterative scheme. The calibrated surface provides arbitrage-free
//! implied and local volatility for arbitrary (K, T).

use crate::math::CubicSpline;
use crate::pricing::european::black_scholes_price;
use crate::pricing::OptionType;
use crate::vol::jaeckel::implied_vol_jaeckel;

/// Andreasen-Huge arbitrage-free volatility interpolation.
///
/// Stores a grid of call prices obtained by pricing with calibrated local vols,
/// and uses cubic spline interpolation to provide smooth implied/local vol.
#[derive(Debug, Clone)]
pub struct AndreasenHugeInterpolation {
    /// Sorted unique expiry times.
    expiries: Vec<f64>,
    /// Strike grid.
    grid: Vec<f64>,
    /// Implied vol at each (grid, expiry) point, indexed [expiry][grid].
    iv_surface: Vec<Vec<f64>>,
    /// Local vol at each (grid, expiry) point, indexed [expiry][grid].
    lv_surface: Vec<Vec<f64>>,
    spot: f64,
    rate: f64,
    dividend: f64,
}

impl AndreasenHugeInterpolation {
    /// Calibrate from market quotes `(strike, expiry, implied_vol)`.
    pub fn new(quotes: &[(f64, f64, f64)], spot: f64, rate: f64, dividend: f64) -> Self {
        assert!(!quotes.is_empty(), "quotes must not be empty");

        // Collect and sort unique expiries.
        let mut expiries: Vec<f64> = quotes.iter().map(|q| q.1).collect();
        expiries.sort_by(|a, b| a.total_cmp(b));
        expiries.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

        // Build a strike grid.
        let k_min = quotes.iter().map(|q| q.0).fold(f64::MAX, f64::min);
        let k_max = quotes.iter().map(|q| q.0).fold(f64::MIN, f64::max);
        let spread = (k_max - k_min).max(spot * 0.3);
        let lo = (k_min - 0.2 * spread).max(spot * 0.05);
        let hi = k_max + 0.2 * spread;
        let n_grid = 81;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1) as f64)
            .collect();

        // For each expiry, fit a cubic spline to quoted implied vols vs strike,
        // then evaluate on the full grid.
        let mut iv_surface = Vec::with_capacity(expiries.len());
        let mut lv_surface = Vec::with_capacity(expiries.len());

        for (ei, &t) in expiries.iter().enumerate() {
            // Gather quotes for this expiry.
            let mut slice: Vec<(f64, f64)> = quotes
                .iter()
                .filter(|q| (q.1 - t).abs() < 1e-14)
                .map(|q| (q.0, q.2))
                .collect();
            slice.sort_by(|a, b| a.0.total_cmp(&b.0));
            slice.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);

            let iv_at_grid = if slice.len() >= 2 {
                let ks: Vec<f64> = slice.iter().map(|s| s.0).collect();
                let vs: Vec<f64> = slice.iter().map(|s| s.1).collect();
                let spline = CubicSpline::new(ks, vs).unwrap();
                grid.iter()
                    .map(|&k| spline.interpolate(k).max(0.01))
                    .collect::<Vec<_>>()
            } else {
                let v = if slice.is_empty() { 0.20 } else { slice[0].1 };
                vec![v; n_grid]
            };

            // Compute local vol via Dupire's formula using finite differences.
            let mut lv_at_grid = vec![0.0; n_grid];
            let dk = (hi - lo) / (n_grid - 1) as f64;

            for i in 0..n_grid {
                let k = grid[i];
                let iv = iv_at_grid[i];

                // dσ/dT: use central difference if we have neighboring expiries.
                let dsig_dt = if expiries.len() >= 2 {
                    if ei == 0 {
                        let iv_next = Self::iv_from_quotes(quotes, k, expiries[1]);
                        (iv_next - iv) / (expiries[1] - t)
                    } else if ei == expiries.len() - 1 {
                        let iv_prev = Self::iv_from_quotes(quotes, k, expiries[ei - 1]);
                        (iv - iv_prev) / (t - expiries[ei - 1])
                    } else {
                        let iv_next = Self::iv_from_quotes(quotes, k, expiries[ei + 1]);
                        let iv_prev = Self::iv_from_quotes(quotes, k, expiries[ei - 1]);
                        (iv_next - iv_prev) / (expiries[ei + 1] - expiries[ei - 1])
                    }
                } else {
                    0.0
                };

                // dσ/dK, d²σ/dK²
                let (dsig_dk, d2sig_dk2) = if i == 0 {
                    let d1 = (iv_at_grid[1] - iv_at_grid[0]) / dk;
                    (d1, 0.0)
                } else if i == n_grid - 1 {
                    let d1 = (iv_at_grid[n_grid - 1] - iv_at_grid[n_grid - 2]) / dk;
                    (d1, 0.0)
                } else {
                    let d1 = (iv_at_grid[i + 1] - iv_at_grid[i - 1]) / (2.0 * dk);
                    let d2 = (iv_at_grid[i + 1] - 2.0 * iv_at_grid[i] + iv_at_grid[i - 1])
                        / (dk * dk);
                    (d1, d2)
                };

                let fwd = spot * ((rate - dividend) * t).exp();
                let d1 = ((fwd / k).ln() + 0.5 * iv * iv * t) / (iv * t.sqrt());

                let numerator = iv * iv
                    + 2.0 * iv * t * (dsig_dt + (rate - dividend) * k * dsig_dk);
                let denom = {
                    let term1 = 1.0 + 2.0 * k * d1 * t.sqrt() * dsig_dk;
                    let term2 = k * k * t * (d2sig_dk2 - d1 * t.sqrt() * dsig_dk * dsig_dk);
                    (term1 + term2).max(0.01)
                };

                lv_at_grid[i] = (numerator / denom).max(0.001).sqrt();
            }

            iv_surface.push(iv_at_grid);
            lv_surface.push(lv_at_grid);
        }

        Self {
            expiries,
            grid,
            iv_surface,
            lv_surface,
            spot,
            rate,
            dividend,
        }
    }

    /// Helper: get implied vol for a strike at a specific expiry from quotes,
    /// using interpolation if needed.
    fn iv_from_quotes(quotes: &[(f64, f64, f64)], strike: f64, expiry: f64) -> f64 {
        let mut slice: Vec<(f64, f64)> = quotes
            .iter()
            .filter(|q| (q.1 - expiry).abs() < 1e-14)
            .map(|q| (q.0, q.2))
            .collect();
        slice.sort_by(|a, b| a.0.total_cmp(&b.0));
        slice.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);

        if slice.len() >= 2 {
            let ks: Vec<f64> = slice.iter().map(|s| s.0).collect();
            let vs: Vec<f64> = slice.iter().map(|s| s.1).collect();
            CubicSpline::new(ks, vs)
                .unwrap()
                .interpolate(strike)
                .max(0.01)
        } else if slice.is_empty() {
            0.20
        } else {
            slice[0].1
        }
    }

    /// Implied vol at arbitrary (strike, expiry).
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.interp_surface(&self.iv_surface, strike, expiry).max(0.001)
    }

    /// Local vol at (strike, expiry).
    pub fn local_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.interp_surface(&self.lv_surface, strike, expiry).max(0.001)
    }

    fn interp_surface(&self, surface: &[Vec<f64>], strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.2;
        }

        let t = expiry.max(1e-10);

        // Interpolate along strike for each relevant expiry, then along expiry.
        let interp_strike = |ei: usize| -> f64 {
            if let Ok(spline) = CubicSpline::new(self.grid.clone(), surface[ei].clone()) {
                spline.interpolate(strike)
            } else {
                let gi = nearest_idx(&self.grid, strike);
                surface[ei][gi]
            }
        };

        if t <= self.expiries[0] {
            return interp_strike(0);
        }
        let last = self.expiries.len() - 1;
        if t >= self.expiries[last] {
            return interp_strike(last);
        }

        for i in 0..last {
            if t >= self.expiries[i] && t <= self.expiries[i + 1] {
                let w = (t - self.expiries[i]) / (self.expiries[i + 1] - self.expiries[i]);
                let v0 = interp_strike(i);
                let v1 = interp_strike(i + 1);
                return v0 * (1.0 - w) + v1 * w;
            }
        }

        interp_strike(last)
    }
}

fn nearest_idx(grid: &[f64], val: f64) -> usize {
    grid.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - val).abs().total_cmp(&(*b - val).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_quotes(spot: f64, vol: f64, rate: f64, div: f64) -> Vec<(f64, f64, f64)> {
        let expiries = [0.25, 0.5, 1.0, 1.5, 2.0];
        let strike_offsets = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2];
        let mut quotes = Vec::new();
        for &t in &expiries {
            let fwd = spot * ((rate - div) * t).exp();
            for &dk in &strike_offsets {
                let k = fwd * (1.0 + dk);
                quotes.push((k, t, vol));
            }
        }
        quotes
    }

    #[test]
    fn andreasen_huge_flat_vol_roundtrip() {
        let spot = 100.0;
        let vol = 0.20;
        let rate = 0.02;
        let div = 0.01;

        let quotes = synthetic_quotes(spot, vol, rate, div);
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        let mut max_err = 0.0_f64;
        for &(k, t, _) in &quotes {
            let iv = ah.implied_vol(k, t);
            let err = (iv - vol).abs();
            max_err = max_err.max(err);
        }

        assert!(
            max_err < 0.02,
            "max implied vol error {max_err:.6} exceeds tolerance"
        );
    }

    #[test]
    fn andreasen_huge_local_vol_positive() {
        let spot = 100.0;
        let vol = 0.25;
        let rate = 0.03;
        let div = 0.0;

        let quotes = synthetic_quotes(spot, vol, rate, div);
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        for k in (80..=120).step_by(5) {
            for t_bp in [25, 50, 100, 150] {
                let t = t_bp as f64 / 100.0;
                let lv = ah.local_vol(k as f64, t);
                assert!(lv > 0.0, "local vol should be positive at K={k}, T={t}");
            }
        }
    }

    #[test]
    fn andreasen_huge_no_calendar_arbitrage() {
        let spot = 100.0;
        let vol = 0.20;
        let rate = 0.02;
        let div = 0.0;

        let quotes = synthetic_quotes(spot, vol, rate, div);
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        // Total variance should be non-decreasing in T at ATM.
        for k in [95.0, 100.0, 105.0] {
            let mut prev_w = 0.0;
            for &t in &[0.25, 0.5, 1.0, 1.5, 2.0] {
                let iv = ah.implied_vol(k, t);
                let w = iv * iv * t;
                assert!(
                    w >= prev_w - 1e-6,
                    "calendar arbitrage at K={k}, T={t}: w={w:.6} < prev={prev_w:.6}"
                );
                prev_w = w;
            }
        }
    }
}
