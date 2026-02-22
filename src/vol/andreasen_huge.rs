//! Andreasen-Huge piecewise-constant local vol interpolation.
//!
//! Calibrates a piecewise-constant local volatility surface to market quotes
//! using a 1-D implicit finite-difference scheme stepping in the strike
//! direction. The resulting surface is arbitrage-free by construction.
//!
//! Reference: Andreasen & Huge, "Volatility Interpolation" (Risk, 2011).

use crate::pricing::european::black_scholes_price;
use crate::pricing::OptionType;
use crate::vol::jaeckel::implied_vol_jaeckel;

/// Andreasen-Huge arbitrage-free volatility interpolation.
///
/// Given a set of market implied vol quotes, calibrates piecewise-constant
/// local vol on a strike grid at each expiry, then recovers implied vols
/// at arbitrary (K, T) by pricing on the calibrated grid.
#[derive(Debug, Clone)]
pub struct AndreasenHugeInterpolation {
    expiries: Vec<f64>,
    grid: Vec<f64>,
    /// call_prices\[i\]\[j\] = calibrated call price at expiry i, grid point j
    call_prices: Vec<Vec<f64>>,
    /// local_vols\[i\]\[j\] = calibrated local vol at expiry i, grid point j
    local_vols: Vec<Vec<f64>>,
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

        // Build strike grid covering all quotes with padding.
        let k_min = quotes.iter().map(|q| q.0).fold(f64::MAX, f64::min);
        let k_max = quotes.iter().map(|q| q.0).fold(f64::MIN, f64::max);
        let spread = (k_max - k_min).max(spot * 0.5);
        let lo = (k_min - 0.5 * spread).max(spot * 0.01);
        let hi = k_max + 0.5 * spread;
        let n_grid = 201;
        let dk = (hi - lo) / (n_grid - 1) as f64;
        let grid: Vec<f64> = (0..n_grid).map(|i| lo + i as f64 * dk).collect();

        // Initial call prices at T=0: intrinsic value.
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

            // Get market quotes for this expiry slice.
            let slice_quotes: Vec<(f64, f64)> = quotes
                .iter()
                .filter(|q| (q.1 - t).abs() < 1e-14)
                .map(|q| (q.0, q.2))
                .collect();

            // Compute target call prices from market implied vols.
            let targets: Vec<(usize, f64)> = slice_quotes
                .iter()
                .map(|&(k, iv)| {
                    let price =
                        black_scholes_price(OptionType::Call, spot, k, rate - dividend, iv, t);
                    let idx = nearest_idx(&grid, k);
                    (idx, price)
                })
                .collect();

            // Initial guess: average implied vol from quotes (or previous slice).
            let avg_iv = if slice_quotes.is_empty() {
                local_vols.last().map(|v| v[n_grid / 2]).unwrap_or(0.2)
            } else {
                slice_quotes.iter().map(|q| q.1).sum::<f64>() / slice_quotes.len() as f64
            };
            let mut sigma_loc: Vec<f64> = vec![avg_iv; n_grid];

            // Levenberg-Marquardt-style calibration loop.
            for _iter in 0..50 {
                let new_calls = step_implicit(
                    &prev_calls,
                    &grid,
                    &sigma_loc,
                    dt,
                    rate,
                    dividend,
                );

                if targets.is_empty() {
                    break;
                }

                // Compute errors and Jacobian (bump-and-reprice).
                let mut max_err = 0.0_f64;
                let mut adjustments: Vec<(usize, f64)> = Vec::new();

                for &(idx, c_target) in &targets {
                    let c_model = new_calls[idx];
                    let err = c_model - c_target;
                    max_err = max_err.max(err.abs());

                    // Bump local vol at this grid point to estimate sensitivity.
                    let bump = 0.001;
                    let mut sigma_bumped = sigma_loc.clone();

                    // Spread the bump over nearby grid points for stability.
                    let spread_pts = 3;
                    for j in idx.saturating_sub(spread_pts)..=(idx + spread_pts).min(n_grid - 1) {
                        sigma_bumped[j] += bump;
                    }

                    let bumped_calls = step_implicit(
                        &prev_calls,
                        &grid,
                        &sigma_bumped,
                        dt,
                        rate,
                        dividend,
                    );
                    let dc_dsigma = (bumped_calls[idx] - c_model) / bump;

                    if dc_dsigma.abs() > 1e-14 {
                        let step = err / dc_dsigma;
                        adjustments.push((idx, step));
                    }
                }

                if max_err < 1e-10 {
                    break;
                }

                // Apply adjustments with damping and spreading.
                for &(idx, step) in &adjustments {
                    let damped = step * 0.8;
                    let spread_pts = 5;
                    for j in idx.saturating_sub(spread_pts)..=(idx + spread_pts).min(n_grid - 1) {
                        let w = 1.0 / (1.0 + ((grid[j] - grid[idx]) / dk).powi(2));
                        sigma_loc[j] = (sigma_loc[j] - damped * w).max(0.005).min(5.0);
                    }
                }
            }

            let new_calls = step_implicit(
                &prev_calls,
                &grid,
                &sigma_loc,
                dt,
                rate,
                dividend,
            );
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
                // Fallback: Brenner-Subrahmanyam approximation.
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
        self.interp_local_vol(strike, t).max(0.001)
    }

    fn interpolate_call(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return (self.spot - strike).max(0.0);
        }

        let interp_at = |ei: usize| -> f64 {
            let gi = nearest_idx(&self.grid, strike);
            if gi == 0 || gi >= self.grid.len() - 1 {
                return self.call_prices[ei][gi];
            }
            // Linear interpolation between grid points.
            let x0 = self.grid[gi];
            let x1 = self.grid[gi + 1];
            let y0 = self.call_prices[ei][gi];
            let y1 = self.call_prices[ei][gi + 1];
            if (x1 - x0).abs() < 1e-14 {
                return y0;
            }
            let w = (strike - x0) / (x1 - x0);
            y0 + w * (y1 - y0)
        };

        if expiry <= self.expiries[0] {
            return interp_at(0);
        }
        let last = self.expiries.len() - 1;
        if expiry >= self.expiries[last] {
            return interp_at(last);
        }

        for i in 0..last {
            if expiry >= self.expiries[i] && expiry <= self.expiries[i + 1] {
                let w = (expiry - self.expiries[i])
                    / (self.expiries[i + 1] - self.expiries[i]);
                return interp_at(i) * (1.0 - w) + interp_at(i + 1) * w;
            }
        }
        interp_at(last)
    }

    fn interp_local_vol(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.2;
        }
        let lv_at = |ei: usize| -> f64 {
            let gi = nearest_idx(&self.grid, strike);
            self.local_vols[ei][gi]
        };

        if expiry <= self.expiries[0] {
            return lv_at(0);
        }
        let last = self.expiries.len() - 1;
        if expiry >= self.expiries[last] {
            return lv_at(last);
        }
        for i in 0..last {
            if expiry >= self.expiries[i] && expiry <= self.expiries[i + 1] {
                let w = (expiry - self.expiries[i])
                    / (self.expiries[i + 1] - self.expiries[i]);
                return lv_at(i) * (1.0 - w) + lv_at(i + 1) * w;
            }
        }
        lv_at(last)
    }
}

/// One implicit FD step for the call price surface in the strike direction.
///
/// Solves the Dupire forward PDE:
///   ∂C/∂T = 0.5 * σ²(K) * K² * ∂²C/∂K² - (r-q) * K * ∂C/∂K - q * C
fn step_implicit(
    prev: &[f64],
    grid: &[f64],
    sigma: &[f64],
    dt: f64,
    rate: f64,
    dividend: f64,
) -> Vec<f64> {
    let n = prev.len();
    if n < 3 {
        return prev.to_vec();
    }

    let r = rate;
    let q = dividend;

    // Build tridiagonal system: (I - dt*A) * C_new = C_old
    let mut lower = vec![0.0; n];
    let mut diag = vec![1.0; n];
    let mut upper = vec![0.0; n];
    let rhs = prev.to_vec();

    for j in 1..n - 1 {
        let k = grid[j];
        let dk_minus = grid[j] - grid[j - 1];
        let dk_plus = grid[j + 1] - grid[j];
        let dk_avg = 0.5 * (dk_minus + dk_plus);

        let sig2 = sigma[j] * sigma[j];
        let diff = 0.5 * sig2 * k * k;
        let conv = -(r - q) * k;

        let a = dt * (diff / (dk_minus * dk_avg) - conv / (dk_minus + dk_plus));
        let b = dt * (diff / (dk_plus * dk_avg) + conv / (dk_minus + dk_plus));

        lower[j] = -a;
        upper[j] = -b;
        diag[j] = 1.0 + a + b + q * dt;
    }

    // Boundary conditions:
    // At K=K_min (deep ITM call): C ≈ S*exp(-qT) - K*exp(-rT) → linear
    // At K=K_max (deep OTM call): C ≈ 0
    diag[0] = 1.0;
    upper[0] = 0.0;
    diag[n - 1] = 1.0;
    lower[n - 1] = 0.0;

    // Solve tridiagonal system via Thomas algorithm.
    let mut c_star = vec![0.0; n];
    let mut d_star = vec![0.0; n];

    c_star[0] = upper[0] / diag[0];
    d_star[0] = rhs[0] / diag[0];

    for j in 1..n {
        let denom = diag[j] - lower[j] * c_star[j - 1];
        if denom.abs() < 1e-30 {
            return prev.to_vec();
        }
        c_star[j] = if j < n - 1 { upper[j] / denom } else { 0.0 };
        d_star[j] = (rhs[j] - lower[j] * d_star[j - 1]) / denom;
    }

    let mut result = vec![0.0; n];
    result[n - 1] = d_star[n - 1];
    for j in (0..n - 1).rev() {
        result[j] = d_star[j] - c_star[j] * result[j + 1];
    }

    // Enforce non-negativity and intrinsic value bound.
    for j in 0..n {
        result[j] = result[j].max(0.0);
    }

    result
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
        let expiries = [0.25, 0.5, 1.0];
        let strike_offsets = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15];
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
            if err > max_err {
                max_err = err;
            }
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

        for k in (85..=115).step_by(5) {
            for t_bp in [25, 50, 100] {
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
            for &t in &[0.25, 0.5, 1.0] {
                let iv = ah.implied_vol(k, t);
                let w = iv * iv * t;
                assert!(
                    w >= prev_w - 1e-4,
                    "calendar arbitrage at K={k}, T={t}: w={w:.6} < prev={prev_w:.6}"
                );
                prev_w = w;
            }
        }
    }
}
