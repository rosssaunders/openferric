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

    /// No-arbitrage HJM drift `mu(t, T)`.
    pub fn drift(&self, t: f64, maturity: f64) -> f64 {
        if maturity <= t {
            return 0.0;
        }

        let n = self.factors.len();
        let mut sigma = vec![0.0_f64; n];
        let mut integrated = vec![0.0_f64; n];
        for i in 0..n {
            sigma[i] = self.factor_volatility(i, t, maturity);
            integrated[i] = self.integrated_factor_volatility(i, t, maturity);
        }

        let mut drift = 0.0;
        for (i, sigma_i) in sigma.iter().enumerate().take(n) {
            for (j, integrated_j) in integrated.iter().enumerate().take(n) {
                drift += self.correlation[i][j] * sigma_i * integrated_j;
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

        let p0_expiry =
            Self::zero_coupon_bond_price(0.0, option_expiry, maturities, initial_forwards)?;
        let mut payoff_sum = 0.0;
        for path_id in 0..num_paths {
            let path = self.simulate_forward_curve_euler(
                initial_forwards,
                maturities,
                option_expiry,
                num_steps,
                seed.wrapping_add(path_id as u64),
            )?;
            let forwards_at_expiry = &path[path.len() - 1];
            let (swap_rate, annuity) = swap_rate_and_annuity_from_forwards(
                option_expiry,
                swap_start,
                swap_end,
                maturities,
                forwards_at_expiry,
            )?;

            let intrinsic = if is_payer {
                (swap_rate - strike).max(0.0)
            } else {
                (strike - swap_rate).max(0.0)
            };
            payoff_sum += notional * annuity * intrinsic;
        }

        Ok(p0_expiry * payoff_sum / num_paths as f64)
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
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
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
