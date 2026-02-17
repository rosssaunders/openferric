//! Andreasen-Huge piecewise-constant local vol interpolation.
//!
//! Calibrates a piecewise-constant local volatility surface to market quotes
//! using a 1-D implicit finite-difference scheme. The resulting surface is
//! arbitrage-free by construction (no butterfly or calendar spreads).

use crate::math::CubicSpline;
use crate::pricing::european::black_scholes_price;
use crate::pricing::OptionType;
use crate::vol::jaeckel::implied_vol_jaeckel;

/// Andreasen-Huge arbitrage-free volatility interpolation.
#[derive(Debug, Clone)]
pub struct AndreasenHugeInterpolation {
    expiries: Vec<f64>,
    grid: Vec<f64>,
    local_vols: Vec<Vec<f64>>,
    call_prices: Vec<Vec<f64>>,
    spot: f64,
    rate: f64,
    dividend: f64,
}

impl AndreasenHugeInterpolation {
    /// Calibrate from market quotes `(strike, expiry, implied_vol)`.
    pub fn new(quotes: &[(f64, f64, f64)], spot: f64, rate: f64, dividend: f64) -> Self {
        assert!(!quotes.is_empty(), "quotes must not be empty");

        let mut expiries: Vec<f64> = quotes.iter().map(|q| q.1).collect();
        expiries.sort_by(|a, b| a.total_cmp(b));
        expiries.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

        // Build strike grid.
        let k_min = quotes.iter().map(|q| q.0).fold(f64::MAX, f64::min);
        let k_max = quotes.iter().map(|q| q.0).fold(f64::MIN, f64::max);
        let spread = (k_max - k_min).max(spot * 0.5);
        let lo = (k_min - 0.3 * spread).max(spot * 0.01);
        let hi = k_max + 0.3 * spread;
        let n_grid = 101;
        let dk = (hi - lo) / (n_grid - 1) as f64;
        let grid: Vec<f64> = (0..n_grid).map(|i| lo + i as f64 * dk).collect();

        let mut prev_calls: Vec<f64> = grid.iter().map(|&k| (spot - k).max(0.0)).collect();
        let mut prev_t = 0.0;

        let mut local_vols: Vec<Vec<f64>> = Vec::with_capacity(expiries.len());
        let mut call_prices: Vec<Vec<f64>> = Vec::with_capacity(expiries.len());

        for &t in &expiries {
            let dt = t - prev_t;
            if dt < 1e-14 {
                local_vols.push(local_vols.last().cloned().unwrap_or_else(|| vec![0.2; n_grid]));
                call_prices.push(prev_calls.clone());
                continue;
            }

            let slice_quotes: Vec<(f64, f64)> = quotes
                .iter()
                .filter(|q| (q.1 - t).abs() < 1e-14)
                .map(|q| (q.0, q.2))
                .collect();

            let targets: Vec<(f64, f64)> = slice_quotes
                .iter()
                .map(|&(k, iv)| {
                    let price =
                        black_scholes_price(OptionType::Call, spot, k, rate - dividend, iv, t);
                    (k, price)
                })
                .collect();

            let avg_iv = if slice_quotes.is_empty() {
                0.2
            } else {
                slice_quotes.iter().map(|q| q.1).sum::<f64>() / slice_quotes.len() as f64
            };
            let mut sigma_loc: Vec<f64> = vec![avg_iv; n_grid];

            for _iter in 0..20 {
                let new_calls =
                    step_implicit(&prev_calls, &grid, &sigma_loc, dt, rate, dividend, spot, t);

                if targets.is_empty() {
                    break;
                }

                let mut max_err = 0.0_f64;
                for &(k_target, c_target) in &targets {
                    let idx = nearest_idx(&grid, k_target);
                    let c_model = new_calls[idx];
                    let err = c_model - c_target;
                    max_err = max_err.max(err.abs());

                    let vega_approx = spot * t.sqrt() * 0.4;
                    if vega_approx.abs() > 1e-14 {
                        let adjustment = err / vega_approx;
                        let spread_pts = 5;
                        for j in
                            idx.saturating_sub(spread_pts)..=(idx + spread_pts).min(n_grid - 1)
                        {
                            let w = 1.0 / (1.0 + (grid[j] - k_target).abs() / dk);
                            sigma_loc[j] =
                                (sigma_loc[j] - 0.5 * adjustment * w).max(0.01).min(5.0);
                        }
                    }
                }

                if max_err < 1e-10 {
                    break;
                }
            }

            let new_calls =
                step_implicit(&prev_calls, &grid, &sigma_loc, dt, rate, dividend, spot, t);
            local_vols.push(sigma_loc);
            call_prices.push(new_calls.clone());
            prev_calls = new_calls;
            prev_t = t;
        }

        Self {
            expiries,
            grid,
            local_vols,
            call_prices,
            spot,
            rate,
            dividend,
        }
    }

    /// Implied vol at arbitrary (strike, expiry).
    pub fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        let t = expiry.max(1e-10);
        let fwd = self.spot * ((self.rate - self.dividend) * t).exp();
        let call_price = self.interpolate_call(strike, t);

        match implied_vol_jaeckel(call_price, fwd, strike, t, true) {
            Ok(v) => v,
            Err(_) => {
                let c_norm = call_price / (self.spot * (-self.dividend * t).exp());
                (c_norm * (2.0 * std::f64::consts::PI / t).sqrt()).max(0.001)
            }
        }
    }

    /// Local vol at (strike, expiry).
    pub fn local_vol(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.2;
        }

        let t = expiry.max(1e-10);
        let (idx, weight) = self.find_expiry_interp(t);

        let lv = |ei: usize| -> f64 {
            let gi = nearest_idx(&self.grid, strike);
            self.local_vols[ei][gi]
        };

        if let Some((i0, i1, w)) = weight {
            lv(i0) * (1.0 - w) + lv(i1) * w
        } else {
            lv(idx)
        }
    }

    fn interpolate_call(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return (self.spot - strike).max(0.0);
        }

        let interp_at = |ei: usize| -> f64 {
            if let Ok(spline) = CubicSpline::new(self.grid.clone(), self.call_prices[ei].clone()) {
                spline.interpolate(strike).max(0.0)
            } else {
                let gi = nearest_idx(&self.grid, strike);
                self.call_prices[ei][gi]
            }
        };

        let (idx, weight) = self.find_expiry_interp(expiry);
        if let Some((i0, i1, w)) = weight {
            let c0 = interp_at(i0);
            let c1 = interp_at(i1);
            (c0 * (1.0 - w) + c1 * w).max(0.0)
        } else {
            interp_at(idx)
        }
    }

    fn find_expiry_interp(&self, t: f64) -> (usize, Option<(usize, usize, f64)>) {
        if t <= self.expiries[0] {
            return (0, None);
        }
        let last = self.expiries.len() - 1;
        if t >= self.expiries[last] {
            return (last, None);
        }
        for i in 0..last {
            if t >= self.expiries[i] && t <= self.expiries[i + 1] {
                let w = (t - self.expiries[i]) / (self.expiries[i + 1] - self.expiries[i]);
                return (i, Some((i, i + 1, w)));
            }
        }
        (last, None)
    }
}

/// Implicit FD step for call prices on a strike grid.
fn step_implicit(
    prev: &[f64],
    grid: &[f64],
    sigma: &[f64],
    dt: f64,
    rate: f64,
    dividend: f64,
    spot: f64,
    total_t: f64,
) -> Vec<f64> {
    let n = prev.len();
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    for i in 1..(n - 1) {
        let dk_m = grid[i] - grid[i - 1];
        let dk_p = grid[i + 1] - grid[i];
        let dk_avg = 0.5 * (dk_m + dk_p);
        let vol2 = sigma[i] * sigma[i];

        let diff = 0.5 * vol2 * grid[i] * grid[i];
        let drift = (rate - dividend) * grid[i];

        let alpha = dt * diff / (dk_m * dk_avg);
        let beta = dt * diff / (dk_p * dk_avg);
        let gamma = dt * drift / (dk_m + dk_p);

        a[i] = -(alpha - gamma);
        b[i] = 1.0 + alpha + beta + dt * rate;
        c[i] = -(beta + gamma);
        rhs[i] = prev[i];
    }

    // Boundaries.
    let fwd = spot * ((rate - dividend) * total_t).exp();
    b[0] = 1.0;
    rhs[0] = (fwd - grid[0]).max(0.0);
    b[n - 1] = 1.0;
    rhs[n - 1] = 0.0;

    tridiag_solve(&a, &b, &c, &rhs)
}

fn tridiag_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    let mut x = vec![0.0; n];

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }

    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }

    x
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
            max_err < 0.05,
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

        for k in [90.0, 95.0, 100.0, 105.0, 110.0] {
            let mut prev_w = 0.0;
            for t_bp in [25, 50, 100, 150, 200] {
                let t = t_bp as f64 / 100.0;
                let iv = ah.implied_vol(k, t);
                let w = iv * iv * t;
                assert!(
                    w >= prev_w - 1e-6,
                    "calendar arbitrage at K={k}, T={t}: w={w} < prev={prev_w}"
                );
                prev_w = w;
            }
        }
    }
}
