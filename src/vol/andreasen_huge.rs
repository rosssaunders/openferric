//! Module `vol::andreasen_huge`.
//!
//! Implements andreasen huge abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Andreasen and Huge (2011), Gatheral (2006) Ch. 7, local-vol interpolation constraints around Eq. (7.2).
//!
//! Key types and purpose: `AndreasenHugeInterpolation` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.

use crate::engines::analytic::black_scholes::bs_price;
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
                local_vols.push(
                    local_vols
                        .last()
                        .cloned()
                        .unwrap_or_else(|| vec![0.2; n_grid]),
                );
                call_prices.push(prev_calls.clone());
                continue;
            }

            // Get market quotes for this expiry slice.
            let slice_quotes: Vec<(f64, f64)> = quotes
                .iter()
                .filter(|q| (q.1 - t).abs() < 1e-14)
                .map(|q| (q.0, q.2))
                .collect();

            // Compute target call prices from market implied vols using the true
            // dividend-adjusted discounted Black-Scholes price
            //   C = S e^{-q t} N(d1) - K e^{-r t} N(d2),
            // consistent with the discounted prices evolved by the forward PDE.
            // The target is evaluated at the grid-node strike (with the quote's
            // implied vol) so the node price the calibration matches is
            // self-consistent with the node it is assigned to.
            //
            // Strike snapping: each quote is assigned to the NEAREST grid
            // node, i.e. its strike may shift by up to dk/2 (dk = grid
            // spacing, here ~(hi-lo)/200). Quotes closer than dk to each
            // other can collide on the same node; the later quote's target
            // then effectively overwrites the earlier one in the calibration.
            let targets: Vec<(usize, f64)> = slice_quotes
                .iter()
                .map(|&(k, iv)| {
                    let idx = nearest_idx(&grid, k);
                    let price = bs_price(OptionType::Call, spot, grid[idx], rate, dividend, iv, t);
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
            for _iter in 0..200 {
                let new_calls =
                    step_implicit(&prev_calls, &grid, &sigma_loc, dt, rate, dividend, spot, t);

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
                    for sb in sigma_bumped
                        .iter_mut()
                        .take((idx + spread_pts).min(n_grid - 1) + 1)
                        .skip(idx.saturating_sub(spread_pts))
                    {
                        *sb += bump;
                    }

                    let bumped_calls = step_implicit(
                        &prev_calls,
                        &grid,
                        &sigma_bumped,
                        dt,
                        rate,
                        dividend,
                        spot,
                        t,
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
                        sigma_loc[j] = (sigma_loc[j] - damped * w).clamp(0.005, 5.0);
                    }
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

        // The grid stores discounted prices C = e^{-r t} E[(S_t - K)^+]; Jaeckel
        // expects the undiscounted forward price E[(S_t - K)^+] = C e^{r t}.
        let undiscounted = call_price * (self.rate * t).exp();
        match implied_vol_jaeckel(undiscounted, fwd, strike, t, true) {
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

        let n = self.grid.len();
        let interp_at = |ei: usize| -> f64 {
            let prices = &self.call_prices[ei];
            if n < 2 || strike <= self.grid[0] {
                return prices[0];
            }
            if strike >= self.grid[n - 1] {
                return prices[n - 1];
            }
            // Lower bracketing index: grid[gi] <= strike < grid[gi + 1].
            let gi = lower_idx(&self.grid, strike);
            let x0 = self.grid[gi];
            let x1 = self.grid[gi + 1];
            let y0 = prices[gi];
            let y1 = prices[gi + 1];
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
                let w = (expiry - self.expiries[i]) / (self.expiries[i + 1] - self.expiries[i]);
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
                let w = (expiry - self.expiries[i]) / (self.expiries[i + 1] - self.expiries[i]);
                return lv_at(i) * (1.0 - w) + lv_at(i + 1) * w;
            }
        }
        lv_at(last)
    }
}

/// One implicit FD step for the call price surface in the strike direction.
///
/// Solves the Dupire forward PDE for discounted call prices:
///   ∂C/∂T = 0.5 * σ²(K) * K² * ∂²C/∂K² - (r-q) * K * ∂C/∂K - q * C
///
/// `t_new` is the absolute maturity after the step; it sets the Dirichlet
/// boundary values consistently with the discounted-price convention.
fn step_implicit(
    prev: &[f64],
    grid: &[f64],
    sigma: &[f64],
    dt: f64,
    rate: f64,
    dividend: f64,
    spot: f64,
    t_new: f64,
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
    let mut rhs = prev.to_vec();

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

    // Boundary conditions (Dirichlet, discounted prices):
    // At K=K_min (deep ITM call): C = S*exp(-qT) - K*exp(-rT)
    // At K=K_max (deep OTM call): C ≈ 0
    diag[0] = 1.0;
    upper[0] = 0.0;
    rhs[0] = (spot * (-q * t_new).exp() - grid[0] * (-r * t_new).exp()).max(0.0);
    diag[n - 1] = 1.0;
    lower[n - 1] = 0.0;
    rhs[n - 1] = 0.0;

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
    for value in result.iter_mut().take(n) {
        *value = value.max(0.0);
    }

    result
}

/// Nearest index on a uniform grid in O(1).
fn nearest_idx(grid: &[f64], val: f64) -> usize {
    let n = grid.len();
    if n < 2 {
        return 0;
    }
    let dk = (grid[n - 1] - grid[0]) / (n - 1) as f64;
    if dk <= 0.0 || !dk.is_finite() {
        return 0;
    }
    let i = ((val - grid[0]) / dk).round();
    if i <= 0.0 { 0 } else { (i as usize).min(n - 1) }
}

/// Lower bracketing index on a uniform grid in O(1):
/// returns `i` with `grid[i] <= val < grid[i + 1]`, clamped to `[0, n - 2]`.
fn lower_idx(grid: &[f64], val: f64) -> usize {
    let n = grid.len();
    if n < 2 {
        return 0;
    }
    let dk = (grid[n - 1] - grid[0]) / (n - 1) as f64;
    if dk <= 0.0 || !dk.is_finite() {
        return 0;
    }
    let i = ((val - grid[0]) / dk).floor();
    if i <= 0.0 { 0 } else { (i as usize).min(n - 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normal_cdf, normal_pdf};

    fn synthetic_quotes(_spot: f64, vol: f64, _rate: f64, _div: f64) -> Vec<(f64, f64, f64)> {
        let expiries = [0.25, 0.5, 1.0];
        // With K_min=75 and K_max=125 the production grid is [50, 150] in
        // 0.5 increments, so every quote below is an exact grid node.  This
        // isolates calibration repricing error from the separately documented
        // off-grid strike interpolation error.
        let strikes = [75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0];
        let mut quotes = Vec::new();
        for &t in &expiries {
            for &k in &strikes {
                quotes.push((k, t, vol));
            }
        }
        quotes
    }

    fn assert_reprices_quote_nodes(
        ah: &AndreasenHugeInterpolation,
        quotes: &[(f64, f64, f64)],
        spot: f64,
        rate: f64,
        div: f64,
    ) {
        for &(strike, expiry, vol) in quotes {
            let expiry_idx = ah
                .expiries
                .iter()
                .position(|t| (*t - expiry).abs() <= 1.0e-14)
                .unwrap();
            let strike_idx = nearest_idx(&ah.grid, strike);
            assert!((ah.grid[strike_idx] - strike).abs() <= 1.0e-14);

            let model = ah.call_prices[expiry_idx][strike_idx];
            let target = bs_price(OptionType::Call, spot, strike, rate, div, vol, expiry);
            assert!(
                (model - target).abs() <= 5.0e-8,
                "K={strike} T={expiry}: model={model} target={target} residual={}",
                model - target
            );
        }
    }

    #[test]
    fn andreasen_huge_flat_vol_roundtrip() {
        let spot = 100.0;
        let vol = 0.20;
        let rate = 0.02;
        let div = 0.01;

        let quotes = synthetic_quotes(spot, vol, rate, div);
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        assert_reprices_quote_nodes(&ah, &quotes, spot, rate, div);
    }

    #[test]
    fn andreasen_huge_flat_vol_roundtrip_with_unequal_rate_and_dividend() {
        // With r != q the target prices and PDE evolution must agree; the old
        // code carried a net e^{(q-r)t} bias.  Exact quote-grid nodes isolate
        // that convention check from strike interpolation.
        let spot = 100.0;
        let vol = 0.20;
        let rate = 0.05;
        let div = 0.02;

        let quotes = synthetic_quotes(spot, vol, rate, div);
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        assert_reprices_quote_nodes(&ah, &quotes, spot, rate, div);
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
    fn andreasen_huge_off_grid_price_has_analytic_interpolation_error_bound() {
        let spot: f64 = 100.0;
        let vol: f64 = 0.20;
        let rate: f64 = 0.02;
        let div: f64 = 0.01;
        let expiry: f64 = 0.5;
        let quotes: Vec<(f64, f64, f64)> = [75.0, 99.5, 100.0, 125.0]
            .into_iter()
            .map(|strike| (strike, expiry, vol))
            .collect();
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        // 99.5 and 100.0 are adjacent calibrated nodes on the 0.5 grid.  At
        // their midpoint, the implementation must reproduce linear price
        // interpolation; the remaining error to Black-Scholes is bounded by
        // max_K C_KK * h^2/8.
        let left: f64 = 99.5;
        let right: f64 = 100.0;
        let strike: f64 = 0.5 * (left + right);
        let model = ah.interpolate_call(strike, expiry);
        let left_exact = bs_price(OptionType::Call, spot, left, rate, div, vol, expiry);
        let right_exact = bs_price(OptionType::Call, spot, right, rate, div, vol, expiry);
        let linear_reference = 0.5 * (left_exact + right_exact);
        let expiry_idx = ah
            .expiries
            .iter()
            .position(|t| (*t - expiry).abs() <= 1.0e-14)
            .unwrap();
        let left_model = ah.call_prices[expiry_idx][nearest_idx(&ah.grid, left)];
        let right_model = ah.call_prices[expiry_idx][nearest_idx(&ah.grid, right)];
        let model_linear_reference = 0.5 * (left_model + right_model);
        let roundoff = 16.0 * f64::EPSILON * model.abs().max(1.0);
        assert!(
            (model - model_linear_reference).abs() <= roundoff,
            "model={model} exact node interpolation={model_linear_reference}"
        );
        let node_calibration_error = (left_model - left_exact)
            .abs()
            .max((right_model - right_exact).abs());

        let exact = bs_price(OptionType::Call, spot, strike, rate, div, vol, expiry);
        let mut max_second_derivative = 0.0_f64;
        for i in 0..=100 {
            let k = left + (right - left) * i as f64 / 100.0;
            let d2 =
                ((spot / k).ln() + (rate - div - 0.5 * vol * vol) * expiry) / (vol * expiry.sqrt());
            let second = (-rate * expiry).exp() * normal_pdf(d2) / (k * vol * expiry.sqrt());
            max_second_derivative = max_second_derivative.max(second);
        }
        let interpolation_bound = max_second_derivative * (right - left).powi(2) / 8.0
            + node_calibration_error
            + roundoff;
        assert!(
            (model - exact).abs() <= interpolation_bound,
            "model={model} exact={exact} interpolation_bound={interpolation_bound} node_calibration_error={node_calibration_error} linear_reference={linear_reference}"
        );
    }

    #[test]
    fn andreasen_huge_nonflat_off_grid_matches_bachelier_reference() {
        let spot: f64 = 100.0;
        let rate: f64 = 0.0;
        let div: f64 = 0.0;
        let normal_vol: f64 = 20.0;
        let normal_call = |strike: f64, expiry: f64| {
            let stddev = normal_vol * expiry.sqrt();
            let d = (spot - strike) / stddev;
            (spot - strike) * normal_cdf(d) + stddev * normal_pdf(d)
        };
        let quotes: Vec<(f64, f64, f64)> = [
            0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35,
            0.375, 0.40, 0.425, 0.45, 0.475, 0.50,
        ]
        .into_iter()
        .flat_map(|expiry| {
            [
                75.0_f64, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5,
                105.0, 107.5, 110.0, 112.5, 115.0, 117.5, 120.0, 122.5, 125.0,
            ]
            .into_iter()
            .map(move |strike| {
                let price = normal_call(strike, expiry);
                let vol = implied_vol_jaeckel(price, spot, strike, expiry, true)
                    .expect("Bachelier call has a Black implied volatility");
                (strike, expiry, vol)
            })
        })
        .collect();
        let ah = AndreasenHugeInterpolation::new(&quotes, spot, rate, div);

        // These prices and node prices were independently evaluated with
        // SciPy 1.17.1 (`scipy.special.ndtr`) in the analytic Bachelier
        // formula.  A normal-vol surface induces a genuinely non-flat Black
        // implied-vol smile.  The fixed production grid has dK=0.5; its
        // calibrated-node budget is 1.1e-3 currency units.  The additional
        // off-grid budget is the exact linear-interpolation remainder
        // max(C_KK) dK^2 / 8, with Bachelier gamma
        // phi((S-K)/(sigma_N sqrt(T))) / (sigma_N sqrt(T)).
        let fixtures: [(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64); 2] = [
            (
                0.25,
                97.3,
                5.483_960_270_622_136,
                97.0,
                5.667_612_421_172_099,
                97.5,
                5.363_446_982_235_802,
                0.038_666_811_680_284_93,
                5.667_237_733_420_867,
                5.363_446_982_240_887,
                5.484_963_282_712_881,
            ),
            (
                0.5,
                108.2,
                2.464_594_633_067_921,
                108.0,
                2.521_275_914_383_213,
                108.5,
                2.381_358_578_122_543_3,
                0.024_038_532_470_982_7,
                2.521_817_358_687_558_3,
                2.382_381_380_358_338_2,
                2.466_042_967_355_869_5,
            ),
        ];
        let grid_node_budget = 1.1e-3;

        for &(
            expiry,
            strike,
            exact,
            left,
            left_exact,
            right,
            right_exact,
            max_gamma,
            left_grid_reference,
            right_grid_reference,
            off_grid_reference,
        ) in &fixtures
        {
            let model = ah.interpolate_call(strike, expiry);
            let ei = ah
                .expiries
                .iter()
                .position(|t| (*t - expiry).abs() <= 1.0e-14)
                .unwrap();
            let gi = lower_idx(&ah.grid, strike);
            assert_eq!(ah.grid[gi], left);
            assert_eq!(ah.grid[gi + 1], right);

            let left_model = ah.call_prices[ei][gi];
            let right_model = ah.call_prices[ei][gi + 1];

            // Separately lock the deterministic production finite-grid
            // outputs.  These are not economic tolerances: the allowance is
            // only for binary64 operation ordering across supported targets.
            for (label, actual, reference) in [
                ("left node", left_model, left_grid_reference),
                ("right node", right_model, right_grid_reference),
                ("off-grid interpolation", model, off_grid_reference),
            ] {
                let binary64_budget = 128.0 * f64::EPSILON * reference.abs().max(1.0);
                assert!(
                    (actual - reference).abs() <= binary64_budget,
                    "T={expiry} K={strike} {label}: actual={actual:.17e} finite-grid reference={reference:.17e} binary64_budget={binary64_budget:.3e}"
                );
            }

            assert!(
                (left_model - left_exact).abs() <= grid_node_budget,
                "T={expiry} K={left}: node={left_model} exact={left_exact} budget={grid_node_budget}"
            );
            assert!(
                (right_model - right_exact).abs() <= grid_node_budget,
                "T={expiry} K={right}: node={right_model} exact={right_exact} budget={grid_node_budget}"
            );

            let interpolation_budget = max_gamma * (right - left).powi(2) / 8.0;
            let roundoff = 32.0 * f64::EPSILON * exact.abs().max(1.0);
            let tolerance = grid_node_budget + interpolation_budget + roundoff;
            assert!(
                (model - exact).abs() <= tolerance,
                "T={expiry} K={strike}: model={model} exact={exact} grid_node_budget={grid_node_budget} interpolation_budget={interpolation_budget} tolerance={tolerance}"
            );
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

        // A flat input surface has the exact total variance w(T)=sigma^2 T.
        for k in [95.0, 100.0, 105.0] {
            for &t in &[0.25, 0.5, 1.0] {
                let iv = ah.implied_vol(k, t);
                let w = iv * iv * t;
                let expected = vol * vol * t;
                assert!(
                    (w - expected).abs() <= 5.0e-8,
                    "K={k}, T={t}: total_variance={w} exact={expected}, iv={iv}"
                );
            }
        }
    }
}
