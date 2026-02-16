use num_complex::Complex;
use statrs::function::gamma::gamma;

/// Characteristic function interface for log-spot models.
pub trait CharacteristicFunction {
    /// Returns the characteristic function value `phi(u)`.
    fn cf(&self, u: Complex<f64>) -> Complex<f64>;

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
}
