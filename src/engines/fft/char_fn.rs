//! Module `engines::fft::char_fn`.
//!
//! Implements char fn abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Carr and Madan (1999), Lewis (2001), Hull (11th ed.) Ch. 19, with FFT damping/inversion forms around Eq. (19.8).
//!
//! Key types and purpose: `CharacteristicFunction`, `BlackScholesCharFn`, `HestonCharFn`, `VarianceGammaCharFn`, `CgmyCharFn` define the core data contracts for this module.
//!
//! Numerical considerations: choose damping/aliasing controls (alpha, grid spacing, FFT size) to balance truncation error against oscillation near strikes.
//!
//! When to use: choose FFT-based routines for dense strike grids under characteristic-function models; use direct quadrature or Monte Carlo for sparse-strike or path-dependent products.
use num_complex::Complex;

use crate::math::gamma::gamma;

/// Relative imaginary-part tolerance for the default numerical moment probe.
///
/// When `E[S_T^m]` exists, `phi(-i*m)` is a positive real; legitimate models
/// evaluate it with imaginary contamination on the order of f64 rounding
/// (observed <= 1e-12 relative), while moment explosions and principal-branch
/// crossings produce imaginary parts comparable to the real part.
const MOMENT_PROBE_IM_TOL: f64 = 1e-8;

/// Characteristic function interface for log-spot models.
pub trait CharacteristicFunction {
    /// Returns the characteristic function value `phi(u)`.
    fn cf(&self, u: Complex<f64>) -> Complex<f64>;

    /// Returns `true` when the exponential moment `E[S_T^order] = phi(-i*order)` is finite.
    ///
    /// Carr-Madan damping with parameter `alpha` requires the `(alpha + 1)`-th
    /// moment of the underlying to be finite (Carr & Madan 1999, Section 2);
    /// otherwise the damped integrand is not integrable and the transform
    /// silently produces garbage prices.
    ///
    /// The default implementation probes the characteristic function at
    /// `u = -i * order`. A finite moment evaluates to a positive real number;
    /// a non-finite value, a non-positive real part, or a significant
    /// imaginary part signals a moment explosion or a principal-branch
    /// crossing on the damping contour. Models with closed-form moment bounds
    /// should override this with the exact condition.
    fn moment_exists(&self, order: f64) -> bool {
        if !order.is_finite() {
            return false;
        }
        let probe = self.cf(Complex::new(0.0, -order));
        probe.is_finite() && probe.re > 0.0 && probe.im.abs() <= MOMENT_PROBE_IM_TOL * probe.re
    }

    /// Optional derivative wrt log-spot (`x = ln(S0)`).
    fn dcf_dlog_spot(&self, _u: Complex<f64>) -> Option<Complex<f64>> {
        None
    }

    /// Optional second derivative wrt log-spot (`x = ln(S0)`).
    fn d2cf_dlog_spot2(&self, _u: Complex<f64>) -> Option<Complex<f64>> {
        None
    }

    /// Optional derivative wrt a model volatility parameter.
    fn dcf_dvol(&self, _u: Complex<f64>) -> Option<Complex<f64>> {
        None
    }
}

/// Black-Scholes characteristic function for `ln(S_T)`.
#[derive(Debug, Clone, Copy)]
pub struct BlackScholesCharFn {
    pub ln_spot: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub vol: f64,
    pub maturity: f64,
}

impl BlackScholesCharFn {
    pub fn new(spot: f64, rate: f64, dividend_yield: f64, vol: f64, maturity: f64) -> Self {
        Self {
            ln_spot: spot.ln(),
            rate,
            dividend_yield,
            vol,
            maturity,
        }
    }
}

impl CharacteristicFunction for BlackScholesCharFn {
    fn cf(&self, u: Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let sigma2 = self.vol * self.vol;
        let drift = self.ln_spot + (self.rate - self.dividend_yield - 0.5 * sigma2) * self.maturity;
        let exponent = i * u * drift - 0.5 * sigma2 * u * u * self.maturity;
        exponent.exp()
    }

    fn moment_exists(&self, order: f64) -> bool {
        // The lognormal distribution has finite moments of every order.
        order.is_finite()
    }

    fn dcf_dlog_spot(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        Some(i * u * self.cf(u))
    }

    fn d2cf_dlog_spot2(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        Some(-u * u * self.cf(u))
    }

    fn dcf_dvol(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        let dlogphi_dsigma = -self.vol * self.maturity * (i * u + u * u);
        Some(self.cf(u) * dlogphi_dsigma)
    }
}

/// Gatheral-form Heston characteristic function for `ln(S_T)`.
#[derive(Debug, Clone, Copy)]
pub struct HestonCharFn {
    pub ln_spot: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub maturity: f64,
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub sigma_v: f64,
    pub rho: f64,
}

impl HestonCharFn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma_v: f64,
        rho: f64,
    ) -> Self {
        Self {
            ln_spot: spot.ln(),
            rate,
            dividend_yield,
            maturity,
            v0,
            kappa,
            theta,
            sigma_v,
            rho,
        }
    }
}

impl CharacteristicFunction for HestonCharFn {
    fn cf(&self, u: Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let one = Complex::new(1.0, 0.0);

        let sigma2 = self.sigma_v * self.sigma_v;
        let iu = i * u;
        let beta = Complex::new(self.kappa, 0.0) - self.rho * self.sigma_v * iu;

        let mut d = (beta * beta + sigma2 * (u * u + iu)).sqrt();
        if d.re < 0.0 {
            d = -d;
        }

        let mut g = (beta - d) / (beta + d);
        if g.norm() > 1.0 {
            g = Complex::new(1.0, 0.0) / g;
            d = -d;
        }

        let exp_neg_dt = (-d * self.maturity).exp();
        let log_term = ((one - g * exp_neg_dt) / (one - g)).ln();

        let a_over_sigma2 = self.kappa * self.theta / sigma2;
        let c = iu * (self.ln_spot + (self.rate - self.dividend_yield) * self.maturity)
            + Complex::new(a_over_sigma2, 0.0) * ((beta - d) * self.maturity - 2.0 * log_term);
        let d_term = ((beta - d) / sigma2) * ((one - exp_neg_dt) / (one - g * exp_neg_dt));

        (c + d_term * self.v0).exp()
    }

    fn dcf_dlog_spot(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        Some(i * u * self.cf(u))
    }

    fn d2cf_dlog_spot2(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        Some(-u * u * self.cf(u))
    }
}

/// Variance-Gamma characteristic function for `ln(S_T)`.
#[derive(Debug, Clone, Copy)]
pub struct VarianceGammaCharFn {
    pub ln_spot: f64,
    pub drift: f64,
    pub maturity: f64,
    pub sigma: f64,
    pub theta: f64,
    pub nu: f64,
}

impl VarianceGammaCharFn {
    pub fn new(spot: f64, drift: f64, maturity: f64, sigma: f64, theta: f64, nu: f64) -> Self {
        Self {
            ln_spot: spot.ln(),
            drift,
            maturity,
            sigma,
            theta,
            nu,
        }
    }

    pub fn risk_neutral(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        sigma: f64,
        theta: f64,
        nu: f64,
    ) -> Result<Self, String> {
        let martingale_term = 1.0 - theta * nu - 0.5 * sigma * sigma * nu;
        if martingale_term <= 0.0 {
            return Err("variance-gamma martingale condition violated: 1 - theta*nu - 0.5*sigma^2*nu must be > 0".to_string());
        }
        let omega = martingale_term.ln() / nu;
        Ok(Self::new(
            spot,
            rate - dividend_yield + omega,
            maturity,
            sigma,
            theta,
            nu,
        ))
    }
}

impl CharacteristicFunction for VarianceGammaCharFn {
    fn cf(&self, u: Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let denom = Complex::new(1.0, 0.0) - i * u * self.theta * self.nu
            + 0.5 * self.sigma * self.sigma * self.nu * u * u;
        let drift_term = (i * u * (self.ln_spot + self.drift * self.maturity)).exp();
        let vg_term = denom.powf(-self.maturity / self.nu);
        drift_term * vg_term
    }

    fn moment_exists(&self, order: f64) -> bool {
        // E[S_T^m] is finite iff the CF denominator stays positive at u = -i*m:
        // 1 - theta*nu*m - 0.5*sigma^2*nu*m^2 > 0 (Madan, Carr & Chang 1998).
        // When it is negative, denom^(-T/nu) continues onto the principal branch
        // and returns finite but meaningless values.
        order.is_finite()
            && 1.0
                - self.theta * self.nu * order
                - 0.5 * self.sigma * self.sigma * self.nu * order * order
                > 0.0
    }

    fn dcf_dlog_spot(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        Some(i * u * self.cf(u))
    }

    fn d2cf_dlog_spot2(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        Some(-u * u * self.cf(u))
    }

    fn dcf_dvol(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let denom = Complex::new(1.0, 0.0) - Complex::new(0.0, 1.0) * u * self.theta * self.nu
            + 0.5 * self.sigma * self.sigma * self.nu * u * u;
        let dlogphi_dsigma = -self.maturity * self.sigma * u * u / denom;
        Some(self.cf(u) * dlogphi_dsigma)
    }
}

/// CGMY (tempered stable) characteristic function for `ln(S_T)`.
#[derive(Debug, Clone, Copy)]
pub struct CgmyCharFn {
    pub ln_spot: f64,
    pub drift: f64,
    pub maturity: f64,
    pub c: f64,
    pub g: f64,
    pub m: f64,
    pub y: f64,
}

impl CgmyCharFn {
    pub fn new(spot: f64, drift: f64, maturity: f64, c: f64, g: f64, m: f64, y: f64) -> Self {
        Self {
            ln_spot: spot.ln(),
            drift,
            maturity,
            c,
            g,
            m,
            y,
        }
    }

    pub fn risk_neutral(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        c: f64,
        g: f64,
        m: f64,
        y: f64,
    ) -> Result<Self, String> {
        if m <= 1.0 {
            return Err("cgmy requires M > 1 for risk-neutral martingale correction".to_string());
        }
        if g <= 0.0 || c <= 0.0 || m <= 0.0 {
            return Err("cgmy requires C>0, G>0, M>0".to_string());
        }
        if !(y < 2.0 && y != 0.0 && y != 1.0) {
            return Err("cgmy requires Y in (-inf,2) excluding 0 and 1".to_string());
        }

        let gamma_neg_y = gamma(-y);
        let drift_correction =
            -c * gamma_neg_y * ((m - 1.0).powf(y) - m.powf(y) + (g + 1.0).powf(y) - g.powf(y));

        Ok(Self::new(
            spot,
            rate - dividend_yield + drift_correction,
            maturity,
            c,
            g,
            m,
            y,
        ))
    }
}

impl CharacteristicFunction for CgmyCharFn {
    fn cf(&self, u: Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let y_complex = Complex::new(self.y, 0.0);
        let gamma_neg_y = gamma(-self.y);

        let m_term = (Complex::new(self.m, 0.0) - i * u).powc(y_complex)
            - Complex::new(self.m.powf(self.y), 0.0);
        let g_term = (Complex::new(self.g, 0.0) + i * u).powc(y_complex)
            - Complex::new(self.g.powf(self.y), 0.0);

        let levy_exponent = self.c * gamma_neg_y * (m_term + g_term);
        let log_phi =
            i * u * (self.ln_spot + self.drift * self.maturity) + levy_exponent * self.maturity;

        log_phi.exp()
    }

    fn moment_exists(&self, order: f64) -> bool {
        // Tempered-stable exponential moments: E[e^{m*X}] < inf iff -G < m < M
        // (Carr, Geman, Madan & Yor 2002). At m >= M the base of
        // (M - i*u)^Y turns negative real on the contour and powc crosses the
        // principal branch cut.
        order.is_finite() && order < self.m && order > -self.g
    }

    fn dcf_dlog_spot(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        Some(i * u * self.cf(u))
    }

    fn d2cf_dlog_spot2(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        Some(-u * u * self.cf(u))
    }
}

/// Normal Inverse Gaussian (NIG) characteristic function for `ln(S_T)`.
///
/// NIG CF: `exp(i*u*mu*t + delta*t*(sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + i*u)^2)))`
#[derive(Debug, Clone, Copy)]
pub struct NigCharFn {
    pub ln_spot: f64,
    pub drift: f64,
    pub maturity: f64,
    pub alpha: f64,
    pub beta: f64,
    pub delta: f64,
}

impl NigCharFn {
    pub fn new(spot: f64, drift: f64, maturity: f64, alpha: f64, beta: f64, delta: f64) -> Self {
        Self {
            ln_spot: spot.ln(),
            drift,
            maturity,
            alpha,
            beta,
            delta,
        }
    }

    pub fn risk_neutral(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        alpha: f64,
        beta: f64,
        delta: f64,
    ) -> Result<Self, String> {
        if alpha <= 0.0 {
            return Err("NIG requires alpha > 0".to_string());
        }
        if beta.abs() >= alpha {
            return Err("NIG requires |beta| < alpha".to_string());
        }
        if delta <= 0.0 {
            return Err("NIG requires delta > 0".to_string());
        }
        let beta_plus_1 = beta + 1.0;
        if beta_plus_1.abs() >= alpha {
            return Err("NIG martingale condition requires |beta + 1| < alpha".to_string());
        }
        let gamma_bar = (alpha * alpha - beta * beta).sqrt();
        let gamma_bar_1 = (alpha * alpha - beta_plus_1 * beta_plus_1).sqrt();
        let omega = delta * (gamma_bar_1 - gamma_bar);
        Ok(Self::new(
            spot,
            rate - dividend_yield + omega,
            maturity,
            alpha,
            beta,
            delta,
        ))
    }
}

impl CharacteristicFunction for NigCharFn {
    fn cf(&self, u: Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let alpha2 = Complex::new(self.alpha * self.alpha, 0.0);
        let beta_iu = Complex::new(self.beta, 0.0) + i * u;
        let gamma_bar = (self.alpha * self.alpha - self.beta * self.beta).sqrt();

        let nig_exponent = Complex::new(self.delta * self.maturity, 0.0)
            * (Complex::new(gamma_bar, 0.0) - (alpha2 - beta_iu * beta_iu).sqrt());

        let log_phi = i * u * (self.ln_spot + self.drift * self.maturity) + nig_exponent;
        log_phi.exp()
    }

    fn moment_exists(&self, order: f64) -> bool {
        // NIG exponential moments: E[e^{m*X}] < inf iff |beta + m| < alpha
        // (Barndorff-Nielsen 1997). The risk-neutral constructor only enforces
        // the martingale condition |beta + 1| < alpha, which does not cover
        // higher damping moments.
        order.is_finite() && (self.beta + order).abs() < self.alpha
    }

    fn dcf_dlog_spot(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        let i = Complex::new(0.0, 1.0);
        Some(i * u * self.cf(u))
    }

    fn d2cf_dlog_spot2(&self, u: Complex<f64>) -> Option<Complex<f64>> {
        Some(-u * u * self.cf(u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bs_cf_is_one_at_zero() {
        let cf = BlackScholesCharFn::new(100.0, 0.03, 0.0, 0.2, 1.0);
        let one = cf.cf(Complex::new(0.0, 0.0));
        assert!((one.re - 1.0).abs() < 1e-12);
        assert!(one.im.abs() < 1e-12);
    }

    #[test]
    fn heston_cf_is_one_at_zero() {
        let cf = HestonCharFn::new(100.0, 0.03, 0.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7);
        let one = cf.cf(Complex::new(0.0, 0.0));
        assert!((one.re - 1.0).abs() < 1e-12);
        assert!(one.im.abs() < 1e-12);
    }

    #[test]
    fn vg_risk_neutral_constructor_validates_martingale_condition() {
        let ok = VarianceGammaCharFn::risk_neutral(100.0, 0.02, 0.0, 1.0, 0.2, -0.1, 0.2);
        assert!(ok.is_ok());

        let bad = VarianceGammaCharFn::risk_neutral(100.0, 0.02, 0.0, 1.0, 1.0, 2.0, 2.0);
        assert!(bad.is_err());
    }

    #[test]
    fn heston_moment_probe_flags_exploding_parameterization() {
        // Moment-exploding set from the FFT admissibility audit: rho=+0.7,
        // xi=1.5, T=5 has Andersen-Piterbarg discriminant
        // beta^2 + xi^2*(m - m^2) = (-2.125)^2 + 2.25*(2.5 - 6.25) < 0 at
        // m = 2.5 and the explosion time is below T=5.
        let bad = HestonCharFn::new(100.0, 0.02, 0.0, 5.0, 0.04, 0.5, 0.04, 1.5, 0.7);
        assert!(!bad.moment_exists(2.5));

        // Healthy set (rho=-0.7, xi=0.4): discriminant 4.84 - 0.6 > 0, so the
        // 2.5-th moment is finite for every maturity.
        let good = HestonCharFn::new(100.0, 0.02, 0.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7);
        assert!(good.moment_exists(2.5));
        assert!(good.moment_exists(1.0));
    }

    #[test]
    fn vg_moment_bound_matches_quadratic_condition() {
        // sigma=0.3, theta=0.3, nu=1.5: quadratic root at m = 1.7584..., so
        // the Carr-Madan default damping moment 2.5 must be rejected
        // (denominator 1 - 0.45*2.5 - 0.0675*6.25 = -0.546875).
        let cf = VarianceGammaCharFn::risk_neutral(100.0, 0.02, 0.0, 1.0, 0.3, 0.3, 1.5)
            .expect("martingale condition holds (0.4825 > 0)");
        assert!(!cf.moment_exists(2.5));
        assert!(cf.moment_exists(1.75));
        assert!(!cf.moment_exists(1.76));
    }

    #[test]
    fn cgmy_moment_bound_requires_order_below_m() {
        let cf = CgmyCharFn::risk_neutral(100.0, 0.02, 0.0, 1.0, 0.5, 5.0, 1.2, 0.5)
            .expect("constructor only requires M > 1");
        assert!(!cf.moment_exists(2.5));
        assert!(!cf.moment_exists(1.2));
        assert!(cf.moment_exists(1.1));
    }

    #[test]
    fn nig_moment_bound_requires_shifted_beta_inside_alpha() {
        let cf = NigCharFn::risk_neutral(100.0, 0.02, 0.0, 1.0, 2.0, 0.5, 0.5)
            .expect("constructor only requires |beta + 1| < alpha");
        assert!(!cf.moment_exists(2.5));
        assert!(!cf.moment_exists(1.5));
        assert!(cf.moment_exists(1.4));
    }
}
