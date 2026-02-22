//! Stochastic process models used by OpenFerric engines.
//!
//! Covers equity, rates, commodity, and jump-process dynamics plus calibration helpers.

pub mod cgmy;
pub mod commodity;
pub mod hjm;
pub mod hw_calibration;
pub mod lmm;
pub mod nig;
pub mod rough_bergomi;
pub mod short_rate;
pub mod slv;
pub mod variance_gamma;

pub use commodity::{
    CommodityForwardCurve, FuturesQuote, SchwartzOneFactor, SchwartzSmithTwoFactor,
    convenience_yield_from_term_structure, implied_convenience_yield,
};
pub use hjm::{HjmFactor, HjmFactorShape, HjmModel};
pub use hw_calibration::{
    AtmSwaptionVolQuote, calibrate_hull_white_params, hw_atm_swaption_vol_approx,
};
pub use lmm::{LmmModel, LmmParams, black_swaption_price, initial_swap_rate_annuity};
pub use rough_bergomi::{
    FbmScheme, fbm_covariance, fbm_path_cholesky, fbm_path_hybrid, rbergomi_european_mc,
    rbergomi_implied_vol_surface,
};
pub use short_rate::{CIR, HullWhite, Vasicek};
pub use slv::{
    LeverageSlice, LeverageSurface, SlvParams, calibrate_leverage_surface,
    nadaraya_watson_conditional_mean, slv_mc_price, slv_mc_price_checked,
};
pub use cgmy::Cgmy;
pub use nig::Nig;
pub use variance_gamma::VarianceGamma;

#[derive(Debug, Clone, Copy)]
pub struct Gbm {
    pub mu: f64,
    pub sigma: f64,
}

impl Gbm {
    pub fn drift(&self, s: f64) -> f64 {
        self.mu * s
    }

    pub fn diffusion(&self, s: f64) -> f64 {
        self.sigma * s
    }

    pub fn step_exact(&self, s: f64, dt: f64, z: f64) -> f64 {
        s * ((self.mu - 0.5 * self.sigma * self.sigma) * dt + self.sigma * dt.sqrt() * z).exp()
    }

    pub fn step_euler(&self, s: f64, dt: f64, z: f64) -> f64 {
        s + self.drift(s) * dt + self.diffusion(s) * dt.sqrt() * z
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Heston {
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub v0: f64,
}

impl Heston {
    pub fn validate(&self) -> bool {
        self.kappa > 0.0
            && self.theta >= 0.0
            && self.xi >= 0.0
            && self.v0 >= 0.0
            && self.rho > -1.0
            && self.rho < 1.0
    }

    pub fn step_euler(&self, s: f64, v: f64, dt: f64, z1: f64, z2: f64) -> (f64, f64) {
        let v_pos = v.max(0.0);
        let sqrt_dt = dt.sqrt();

        // Correlated Brownian increments.
        let zv = z1;
        let zs = self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2;

        let v_next =
            (v + self.kappa * (self.theta - v_pos) * dt + self.xi * v_pos.sqrt() * sqrt_dt * zv)
                .max(0.0);

        let s_next = s * ((self.mu - 0.5 * v_pos) * dt + v_pos.sqrt() * sqrt_dt * zs).exp();

        (s_next, v_next)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sabr {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
}

impl Sabr {
    pub fn validate(&self) -> bool {
        self.alpha > 0.0
            && (0.0..=1.0).contains(&self.beta)
            && self.nu >= 0.0
            && self.rho > -1.0
            && self.rho < 1.0
    }

    pub fn step_euler(&self, f: f64, alpha_t: f64, dt: f64, z1: f64, z2: f64) -> (f64, f64) {
        let sqrt_dt = dt.sqrt();
        let za = z1;
        let zf = self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2;

        let alpha_next = (alpha_t + self.nu * alpha_t.max(0.0) * sqrt_dt * za).max(1e-12);
        let f_next = (f + alpha_t.max(0.0) * f.max(0.0).powf(self.beta) * sqrt_dt * zf).max(1e-12);

        (f_next, alpha_next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gbm_exact_step_is_deterministic_for_zero_noise() {
        let model = Gbm {
            mu: 0.05,
            sigma: 0.2,
        };
        let s1 = model.step_exact(100.0, 1.0, 0.0);
        assert_relative_eq!(s1, 103.045_453_395_351_7, epsilon = 1e-12);
    }

    #[test]
    fn heston_step_keeps_variance_non_negative() {
        let model = Heston {
            mu: 0.03,
            kappa: 2.0,
            theta: 0.04,
            xi: 0.7,
            rho: -0.6,
            v0: 0.04,
        };
        let (_s1, v1) = model.step_euler(100.0, 0.001, 1.0 / 252.0, -15.0, 1.2);
        assert!(v1 >= 0.0);
        assert!(model.validate());
    }

    #[test]
    fn sabr_step_keeps_forward_and_alpha_positive() {
        let model = Sabr {
            alpha: 0.2,
            beta: 0.6,
            rho: -0.3,
            nu: 0.4,
        };
        assert!(model.validate());
        let (f1, a1) = model.step_euler(100.0, model.alpha, 0.1, -4.0, 2.0);
        assert!(f1 > 0.0);
        assert!(a1 > 0.0);
    }
}
