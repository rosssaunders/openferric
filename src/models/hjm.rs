//! Module `models::hjm`.
//!
//! Implements hjm abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Heath, Jarrow, Morton (1992), Brigo and Mercurio (2006) Ch. 6, drift restrictions around HJM Eq. (3.7).
//!
//! Key types and purpose: `HjmFactorShape`, `HjmFactor`, `HjmModel` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Volatility loading shape for an HJM factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HjmFactorShape {
    /// Parallel shift of the forward curve.
    Parallel,
    /// Slope deformation of the forward curve.
    Slope,
    /// Curvature deformation of the forward curve.
    Curvature,
}

/// One HJM volatility factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HjmFactor {
    /// Factor shape (parallel/slope/curvature).
    pub shape: HjmFactorShape,
    /// Base volatility scale.
    pub volatility: f64,
    /// Exponential decay/mean reversion parameter.
    pub mean_reversion: f64,
}

impl HjmFactor {
    /// Instantaneous volatility loading `sigma_i(t, T)` for `tau = T - t`.
    pub fn sigma(&self, tau: f64) -> f64 {
        if tau <= 0.0 {
            return 0.0;
        }
        let decay = (-self.mean_reversion.max(0.0) * tau).exp();
        let shape = match self.shape {
            HjmFactorShape::Parallel => 1.0,
            HjmFactorShape::Slope => tau,
            HjmFactorShape::Curvature => tau * tau,
        };
        self.volatility * shape * decay
    }

    /// Integrated volatility loading `\int_t^T sigma_i(t, s) ds`.
    pub fn integrated_sigma(&self, tau: f64) -> f64 {
        if tau <= 0.0 {
            return 0.0;
        }
        let kappa = self.mean_reversion.max(0.0);
        let vol = self.volatility;
        let x = kappa * tau;

        match self.shape {
            HjmFactorShape::Parallel => {
                if kappa <= 1.0e-12 {
                    vol * tau
                } else {
                    vol * (1.0 - (-x).exp()) / kappa
                }
            }
            HjmFactorShape::Slope => {
                if kappa <= 1.0e-12 {
                    0.5 * vol * tau * tau
                } else {
                    vol * (1.0 - (-x).exp() * (1.0 + x)) / (kappa * kappa)
                }
            }
            HjmFactorShape::Curvature => {
                if kappa <= 1.0e-12 {
                    vol * tau * tau * tau / 3.0
                } else {
                    vol * (2.0 - (-x).exp() * (2.0 + 2.0 * x + x * x)) / (kappa * kappa * kappa)
                }
            }
        }
    }
}

/// Heath-Jarrow-Morton model with 1-3 factors.
#[derive(Debug, Clone, PartialEq)]
pub struct HjmModel {
    /// Volatility factors.
    pub factors: Vec<HjmFactor>,
    /// Factor correlation matrix.
    pub correlation: Vec<Vec<f64>>,
}

impl HjmModel {
    /// Builds a one-factor exponential-volatility HJM.
    ///
    /// `sigma(t, T) = sigma0 * exp(-kappa * (T - t))`
    pub fn single_factor_exponential(sigma0: f64, kappa: f64) -> Self {
        Self {
            factors: vec![HjmFactor {
                shape: HjmFactorShape::Parallel,
                volatility: sigma0,
                mean_reversion: kappa,
            }],
            correlation: vec![vec![1.0]],
        }
    }

    /// Builds a 2- or 3-factor HJM with parallel/slope/curvature factors.
    pub fn multi_factor_parallel_slope_curvature(
        volatilities: &[f64],
        mean_reversions: &[f64],
        correlation: Option<Vec<Vec<f64>>>,
    ) -> Result<Self, String> {
        if volatilities.len() != mean_reversions.len() {
            return Err("volatilities and mean_reversions must have the same length".to_string());
        }
        if !(2..=3).contains(&volatilities.len()) {
            return Err("multi-factor HJM expects 2 or 3 factors".to_string());
        }

        let shapes = [
            HjmFactorShape::Parallel,
            HjmFactorShape::Slope,
            HjmFactorShape::Curvature,
        ];
        let factors = volatilities
            .iter()
            .zip(mean_reversions.iter())
            .enumerate()
            .map(|(i, (&vol, &kappa))| HjmFactor {
                shape: shapes[i],
                volatility: vol,
                mean_reversion: kappa,
            })
            .collect::<Vec<_>>();

        let n = factors.len();
        let corr = correlation.unwrap_or_else(|| {
            let mut id = vec![vec![0.0; n]; n];
            for (i, row) in id.iter_mut().enumerate().take(n) {
                row[i] = 1.0;
            }
            id
        });

        let model = Self {
            factors,
            correlation: corr,
        };
        model.validate()?;
        Ok(model)
    }

    /// Builds a custom HJM model.
    pub fn new(factors: Vec<HjmFactor>, correlation: Vec<Vec<f64>>) -> Result<Self, String> {
        let model = Self {
            factors,
            correlation,
        };
        model.validate()?;
        Ok(model)
    }

    /// Validates factor count and correlation assumptions.
    pub fn validate(&self) -> Result<(), String> {
        if self.factors.is_empty() || self.factors.len() > 3 {
            return Err("HJM requires 1 to 3 factors".to_string());
        }

        let n = self.factors.len();
        if self.correlation.len() != n || self.correlation.iter().any(|row| row.len() != n) {
            return Err("correlation dimensions must match factor count".to_string());
        }

        for (i, f) in self.factors.iter().enumerate() {
            if !f.volatility.is_finite() || f.volatility < 0.0 {
                return Err(format!("factor {i} volatility must be finite and >= 0"));
            }
            if !f.mean_reversion.is_finite() || f.mean_reversion < 0.0 {
                return Err(format!("factor {i} mean_reversion must be finite and >= 0"));
            }
        }

        for i in 0..n {
            if !self.correlation[i][i].is_finite() || (self.correlation[i][i] - 1.0).abs() > 1.0e-10
            {
                return Err("correlation diagonal entries must be 1".to_string());
            }
            for j in 0..n {
                let cij = self.correlation[i][j];
                if !cij.is_finite() || !(-1.0..=1.0).contains(&cij) {
                    return Err("correlation entries must be finite and in [-1, 1]".to_string());
                }
                if (cij - self.correlation[j][i]).abs() > 1.0e-10 {
                    return Err("correlation matrix must be symmetric".to_string());
                }
            }
        }

        cholesky_lower(&self.correlation)
            .ok_or_else(|| "correlation matrix must be positive semidefinite".to_string())?;

        Ok(())
    }

    /// Factor volatility loading `sigma_i(t, T)`.
    pub fn factor_volatility(&self, factor_index: usize, t: f64, maturity: f64) -> f64 {
        if factor_index >= self.factors.len() || maturity <= t {
            return 0.0;
        }
        self.factors[factor_index].sigma(maturity - t)
    }

    /// Integrated factor loading `\int_t^T sigma_i(t, s) ds`.
    pub fn integrated_factor_volatility(&self, factor_index: usize, t: f64, maturity: f64) -> f64 {
        if factor_index >= self.factors.len() || maturity <= t {
            return 0.0;
        }
        self.factors[factor_index].integrated_sigma(maturity - t)
    }

    /// No-arbitrage risk-neutral HJM drift `mu(t, T) = sigma(t, T) * \int_t^T sigma(t, s) ds`
    /// (summed over correlated factors). Supports up to 3 factors (enforced by `validate`).
    pub fn drift(&self, t: f64, maturity: f64) -> f64 {
        if maturity <= t {
            return 0.0;
        }

        // Stack buffers: validate() caps the model at 3 factors, so avoid heap
        // allocations on this hot path.
        let n = self.factors.len().min(3);
        let mut sigma = [0.0_f64; 3];
        let mut integrated = [0.0_f64; 3];
        for i in 0..n {
            sigma[i] = self.factor_volatility(i, t, maturity);
            integrated[i] = self.integrated_factor_volatility(i, t, maturity);
        }

        let mut drift = 0.0;
        for (row, &sigma_i) in self.correlation.iter().zip(sigma.iter()).take(n) {
            for (&cij, &integrated_j) in row.iter().zip(integrated.iter()).take(n) {
                drift += cij * sigma_i * integrated_j;
            }
        }
        drift
    }

    /// Simulates one forward-rate path with Euler discretization.
    ///
    /// Returns a matrix with shape `(num_steps + 1) x maturities.len()`.
    pub fn simulate_forward_curve_euler(
        &self,
        initial_forwards: &[f64],
        maturities: &[f64],
        horizon: f64,
        num_steps: usize,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;
        validate_curve_inputs(initial_forwards, maturities, horizon, num_steps)?;

        let n = maturities.len();
        let dt = horizon / num_steps as f64;
        let sqrt_dt = dt.sqrt();
        let chol = cholesky_lower(&self.correlation)
            .ok_or_else(|| "correlation matrix is not positive semidefinite".to_string())?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indep = vec![0.0_f64; self.factors.len()];
        let mut z = vec![0.0_f64; self.factors.len()];

        let mut path = Vec::with_capacity(num_steps + 1);
        let mut forwards = initial_forwards.to_vec();
        path.push(forwards.clone());

        for step in 0..num_steps {
            let t = step as f64 * dt;
            for zi in &mut indep {
                *zi = StandardNormal.sample(&mut rng);
            }
            correlate_normals(&chol, &indep, &mut z);

            for j in 0..n {
                let maturity = maturities[j];
                if maturity <= t {
                    continue;
                }

                let drift = self.drift(t, maturity);
                let mut diffusion = 0.0;
                for (k, zk) in z.iter().enumerate().take(self.factors.len()) {
                    diffusion += self.factor_volatility(k, t, maturity) * *zk;
                }

                forwards[j] += drift * dt + diffusion * sqrt_dt;
            }

            path.push(forwards.clone());
        }

        Ok(path)
    }

    /// Zero-coupon bond price `P(t,T) = exp(-\int_t^T f(t,u) du)` from a forward curve snapshot.
    pub fn zero_coupon_bond_price(
        time: f64,
        maturity: f64,
        maturities: &[f64],
        forwards: &[f64],
    ) -> Result<f64, String> {
        if maturity <= time {
            return Ok(1.0);
        }
        if maturities.len() != forwards.len() || maturities.is_empty() {
            return Err(
                "maturities and forwards must be non-empty with matching length".to_string(),
            );
        }
        for w in maturities.windows(2) {
            if w[1] <= w[0] {
                return Err("maturities must be strictly increasing".to_string());
            }
        }
        if maturities[0] > time + 1.0e-12 {
            return Err("maturity grid must include points at or before pricing time".to_string());
        }
        if maturity > maturities[maturities.len() - 1] + 1.0e-12 {
            return Err("requested maturity exceeds forward curve grid".to_string());
        }

        let mut points_t = Vec::new();
        points_t.push(time);
        for &tm in maturities {
            if tm > time + 1.0e-12 && tm < maturity - 1.0e-12 {
                points_t.push(tm);
            }
        }
        points_t.push(maturity);

        let mut integral = 0.0;
        for window in points_t.windows(2) {
            let a = window[0];
            let b = window[1];
            let fa = linear_interp(maturities, forwards, a);
            let fb = linear_interp(maturities, forwards, b);
            integral += 0.5 * (fa + fb) * (b - a);
        }

        Ok((-integral).exp())
    }

    /// Monte Carlo price of a European payer/receiver swaption under HJM.
    ///
    /// Forwards are simulated under the risk-neutral measure (HJM no-arbitrage
    /// drift) and each path payoff is discounted with the pathwise bank account
    /// `exp(-\int_0^{T_e} r_s ds)` accumulated from the simulated short rate
    /// `r_t = f(t, t)`, keeping measure and numeraire consistent. The short rate
    /// is read off the simulated curve by interpolation; this is exact whenever
    /// the maturity grid contains the simulation step times.
    #[allow(clippy::too_many_arguments)]
    pub fn price_swaption_mc(
        &self,
        initial_forwards: &[f64],
        maturities: &[f64],
        strike: f64,
        option_expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> Result<f64, String> {
        self.price_swaption_mc_with_stderr(
            initial_forwards,
            maturities,
            strike,
            option_expiry,
            swap_start,
            swap_end,
            is_payer,
            notional,
            num_paths,
            num_steps,
            seed,
        )
        .map(|(price, _stderr)| price)
    }

    /// As [`Self::price_swaption_mc`], but also returns the Monte Carlo standard error.
    #[allow(clippy::too_many_arguments)]
    pub fn price_swaption_mc_with_stderr(
        &self,
        initial_forwards: &[f64],
        maturities: &[f64],
        strike: f64,
        option_expiry: f64,
        swap_start: f64,
        swap_end: f64,
        is_payer: bool,
        notional: f64,
        num_paths: usize,
        num_steps: usize,
        seed: u64,
    ) -> Result<(f64, f64), String> {
        self.validate()?;
        validate_curve_inputs(initial_forwards, maturities, option_expiry, num_steps)?;

        if !strike.is_finite()
            || strike <= 0.0
            || !option_expiry.is_finite()
            || option_expiry <= 0.0
            || !swap_start.is_finite()
            || !swap_end.is_finite()
            || swap_start < option_expiry - 1.0e-12
            || swap_end <= swap_start
            || !notional.is_finite()
            || notional <= 0.0
            || num_paths == 0
        {
            return Err("invalid swaption inputs".to_string());
        }

        let n = maturities.len();
        let n_factors = self.factors.len();
        let dt = option_expiry / num_steps as f64;
        let sqrt_dt = dt.sqrt();

        // Hoisted out of the path loop: the correlation Cholesky factor and the
        // path-independent drift / volatility loadings per (step, maturity).
        let chol = cholesky_lower(&self.correlation)
            .ok_or_else(|| "correlation matrix is not positive semidefinite".to_string())?;
        let mut drift_dt = vec![0.0_f64; num_steps * n];
        let mut vol_sqrt_dt = vec![0.0_f64; num_steps * n * n_factors];
        for step in 0..num_steps {
            let t = step as f64 * dt;
            for j in 0..n {
                let maturity = maturities[j];
                if maturity <= t {
                    continue;
                }
                drift_dt[step * n + j] = self.drift(t, maturity) * dt;
                for k in 0..n_factors {
                    vol_sqrt_dt[(step * n + j) * n_factors + k] =
                        self.factor_volatility(k, t, maturity) * sqrt_dt;
                }
            }
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indep = vec![0.0_f64; n_factors];
        let mut z = vec![0.0_f64; n_factors];
        let mut forwards = vec![0.0_f64; n];

        let mut payoff_sum = 0.0;
        let mut payoff_sq_sum = 0.0;
        for _ in 0..num_paths {
            forwards.copy_from_slice(initial_forwards);

            // Trapezoidal accumulation of \int_0^{T_e} r_s ds with r_t = f(t, t).
            let mut int_r = 0.0;
            let mut r_prev = linear_interp(maturities, &forwards, 0.0);
            for step in 0..num_steps {
                let t = step as f64 * dt;
                for zi in &mut indep {
                    *zi = StandardNormal.sample(&mut rng);
                }
                correlate_normals(&chol, &indep, &mut z);

                for j in 0..n {
                    if maturities[j] <= t {
                        continue;
                    }
                    let mut diffusion = 0.0;
                    for (k, zk) in z.iter().enumerate().take(n_factors) {
                        diffusion += vol_sqrt_dt[(step * n + j) * n_factors + k] * *zk;
                    }
                    forwards[j] += drift_dt[step * n + j] + diffusion;
                }

                let t_next = (step + 1) as f64 * dt;
                let r_next = linear_interp(maturities, &forwards, t_next);
                int_r += 0.5 * (r_prev + r_next) * dt;
                r_prev = r_next;
            }

            let (swap_rate, annuity) = swap_rate_and_annuity_from_forwards(
                option_expiry,
                swap_start,
                swap_end,
                maturities,
                &forwards,
            )?;

            let intrinsic = if is_payer {
                (swap_rate - strike).max(0.0)
            } else {
                (strike - swap_rate).max(0.0)
            };
            let discounted = (-int_r).exp() * notional * annuity * intrinsic;
            payoff_sum += discounted;
            payoff_sq_sum += discounted * discounted;
        }

        let m = num_paths as f64;
        let mean = payoff_sum / m;
        let stderr = if num_paths > 1 {
            ((payoff_sq_sum - m * mean * mean).max(0.0) / (m - 1.0) / m).sqrt()
        } else {
            0.0
        };
        Ok((mean, stderr))
    }
}

fn validate_curve_inputs(
    initial_forwards: &[f64],
    maturities: &[f64],
    horizon: f64,
    num_steps: usize,
) -> Result<(), String> {
    if initial_forwards.is_empty() || initial_forwards.len() != maturities.len() {
        return Err(
            "initial_forwards and maturities must have matching non-empty length".to_string(),
        );
    }
    if !horizon.is_finite() || horizon <= 0.0 || num_steps == 0 {
        return Err("horizon and num_steps must be positive".to_string());
    }
    if initial_forwards.iter().any(|f| !f.is_finite()) {
        return Err("initial forwards must be finite".to_string());
    }
    for w in maturities.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || w[1] <= w[0] {
            return Err("maturities must be finite and strictly increasing".to_string());
        }
    }
    if maturities[0] > 1.0e-12 {
        return Err("maturity grid must start at or before t=0".to_string());
    }
    if maturities[maturities.len() - 1] < horizon - 1.0e-12 {
        return Err("maturity grid must extend to simulation horizon".to_string());
    }
    Ok(())
}

fn swap_rate_and_annuity_from_forwards(
    valuation_time: f64,
    swap_start: f64,
    swap_end: f64,
    maturities: &[f64],
    forwards: &[f64],
) -> Result<(f64, f64), String> {
    let mut payment_dates = Vec::new();
    let mut prev = swap_start;
    loop {
        let next = (prev + 1.0).min(swap_end);
        if next <= prev {
            break;
        }
        payment_dates.push((prev, next));
        if next >= swap_end - 1.0e-12 {
            break;
        }
        prev = next;
    }
    if payment_dates.is_empty() {
        return Err("swap schedule has no payments".to_string());
    }

    let p_start =
        HjmModel::zero_coupon_bond_price(valuation_time, swap_start, maturities, forwards)?;
    let p_end = HjmModel::zero_coupon_bond_price(valuation_time, swap_end, maturities, forwards)?;

    let mut annuity = 0.0;
    for (a, b) in payment_dates {
        let delta = b - a;
        let p_pay = HjmModel::zero_coupon_bond_price(valuation_time, b, maturities, forwards)?;
        annuity += delta * p_pay;
    }
    if annuity <= 0.0 {
        return Err("swap annuity is non-positive".to_string());
    }

    Ok(((p_start - p_end) / annuity, annuity))
}

fn correlate_normals(chol: &[Vec<f64>], indep: &[f64], out: &mut [f64]) {
    for i in 0..chol.len() {
        let mut sum = 0.0;
        for (j, zj) in indep.iter().enumerate().take(i + 1) {
            sum += chol[i][j] * *zj;
        }
        out[i] = sum;
    }
}

fn cholesky_lower(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for (&lik, &ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
            }

            if i == j {
                if sum < -1.0e-12 {
                    return None;
                }
                l[i][j] = if sum <= 0.0 { 0.0 } else { sum.sqrt() };
            } else if l[j][j].abs() <= 1.0e-14 {
                l[i][j] = 0.0;
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

fn linear_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 0..xs.len() - 1 {
        if x <= xs[i + 1] {
            let w = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + w * (ys[i + 1] - ys[i]);
        }
    }
    ys[ys.len() - 1]
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn one_factor_drift_matches_closed_form_expression() {
        let sigma0 = 0.02;
        let kappa = 0.15;
        let model = HjmModel::single_factor_exponential(sigma0, kappa);

        let t = 1.0;
        let maturity = 4.0;
        let tau = maturity - t;
        let sigma = sigma0 * (-kappa * tau).exp();
        let integral = if kappa.abs() <= 1.0e-12 {
            sigma0 * tau
        } else {
            sigma0 * (1.0 - (-kappa * tau).exp()) / kappa
        };
        let expected = sigma * integral;

        assert_relative_eq!(model.drift(t, maturity), expected, epsilon = 1.0e-12);
    }

    #[test]
    fn zero_coupon_bond_price_from_flat_forward_curve_is_exact() {
        let maturities = (0..=40).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
        let forwards = vec![0.03; maturities.len()];
        let p = HjmModel::zero_coupon_bond_price(0.0, 5.0, &maturities, &forwards).unwrap();

        assert_relative_eq!(p, (-0.03_f64 * 5.0).exp(), epsilon = 1.0e-12);
    }

    /// Constant-sigma one-factor HJM (`kappa = 0`) is Ho-Lee, where a one-period
    /// payer swaption equals `(1 + K*tau)` puts on the zero-coupon bond
    /// `P(T0, T1)` struck at `1 / (1 + K*tau)`, with the Gaussian bond-option
    /// closed form using `sigma_p = sigma0 * (T1 - T0) * sqrt(T0)`.
    #[test]
    fn ho_lee_one_period_swaption_matches_gaussian_bond_option_closed_form() {
        use crate::math::normal_cdf;

        let sigma0 = 0.015;
        let model = HjmModel::single_factor_exponential(sigma0, 0.0);

        let f0 = 0.03;
        let t0 = 3.0_f64;
        let t1 = 4.0_f64;
        let tau = t1 - t0;
        let notional = 100.0;
        let num_steps = 60_usize;

        // Grid aligned with simulation steps up to expiry (exact short-rate
        // readout), then coarser out to the swap end.
        let dt = t0 / num_steps as f64;
        let mut maturities = (0..=num_steps).map(|i| i as f64 * dt).collect::<Vec<_>>();
        for i in 1..=10 {
            maturities.push(t0 + i as f64 * 0.1);
        }
        let forwards = vec![f0; maturities.len()];

        let p0_t0 = (-f0 * t0).exp();
        let p0_t1 = (-f0 * t1).exp();
        let strike = (p0_t0 - p0_t1) / (tau * p0_t1); // ATM forward swap rate

        // Closed form: payer swaption = (1 + K*tau) * ZBP(0, T0, T1, X).
        let x = 1.0 / (1.0 + strike * tau);
        let sigma_p = sigma0 * tau * t0.sqrt();
        let h = (p0_t1 / (x * p0_t0)).ln() / sigma_p + 0.5 * sigma_p;
        let zbp = x * p0_t0 * normal_cdf(-h + sigma_p) - p0_t1 * normal_cdf(-h);
        let closed_form = notional * (1.0 + strike * tau) * zbp;

        let (mc, stderr) = model
            .price_swaption_mc_with_stderr(
                &forwards,
                &maturities,
                strike,
                t0,
                t0,
                t1,
                true,
                notional,
                30_000,
                num_steps,
                42,
            )
            .unwrap();

        assert!(stderr > 0.0 && stderr.is_finite());
        assert!(
            (mc - closed_form).abs() <= 3.0 * stderr,
            "mc={mc} closed_form={closed_form} stderr={stderr}"
        );
    }

    #[test]
    fn swaption_price_is_non_negative() {
        let model = HjmModel::single_factor_exponential(0.01, 0.2);
        let maturities = (0..=80).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
        let forwards = vec![0.03; maturities.len()];

        let px = model
            .price_swaption_mc(
                &forwards,
                &maturities,
                0.03,
                2.0,
                2.0,
                5.0,
                true,
                1_000_000.0,
                500,
                24,
                17,
            )
            .unwrap();
        assert!(px >= 0.0);
    }
}
