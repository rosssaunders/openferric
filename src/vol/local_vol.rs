//! Module `vol::local_vol`.
//!
//! Implements local vol workflows with concrete routines such as `dupire_local_vol`.
//!
//! References: Dupire (1994), Gatheral (2006) Ch. 1, local-vol extraction relation around Dupire Eq. (1.10).
//!
//! Key types and purpose: `ImpliedVolSurface`, `DupireLocalVol` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
use crate::pricing::OptionType;
use crate::pricing::european::black_76_price;

pub trait ImpliedVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64;
}

impl<S: ImpliedVolSurface + ?Sized> ImpliedVolSurface for &S {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        (**self).implied_vol(strike, expiry)
    }
}

impl ImpliedVolSurface for crate::vol::surface::VolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.vol(strike, expiry)
    }
}

#[derive(Debug, Clone)]
pub struct DupireLocalVol<S> {
    surface: S,
    /// Forward anchor at T=0 (the spot); the maturity-T forward is
    /// `forward * exp((rate - dividend) * T)`.
    forward: f64,
    rate: f64,
    dividend: f64,
    strike_bump_rel: f64,
    time_bump: f64,
}

impl<S: ImpliedVolSurface> DupireLocalVol<S> {
    pub fn new(surface: S, forward: f64) -> Self {
        Self {
            surface,
            forward: forward.max(1e-12),
            rate: 0.0,
            dividend: 0.0,
            strike_bump_rel: 1e-3,
            time_bump: 1e-3,
        }
    }

    /// Sets the continuously compounded risk-free rate and dividend yield used to
    /// build the maturity-dependent forward `F(T) = forward * exp((r - q) * T)`.
    ///
    /// Defaults are `r = q = 0`, in which case the forward is maturity-independent
    /// and the formula reduces to the zero-carry Dupire relation.
    pub fn with_rates(mut self, rate: f64, dividend: f64) -> Self {
        self.rate = rate;
        self.dividend = dividend;
        self
    }

    pub fn with_bumps(mut self, strike_bump_rel: f64, time_bump: f64) -> Self {
        self.strike_bump_rel = strike_bump_rel.max(1e-6);
        self.time_bump = time_bump.max(1e-6);
        self
    }

    pub fn local_vol(&self, s: f64, t: f64) -> f64 {
        let strike = s.max(1e-8);
        let expiry = t.max(1e-6);
        let dk = (strike * self.strike_bump_rel).max(1e-5);
        let dt = self.time_bump.min(0.5 * expiry).max(1e-6);

        let c0 = self.call_price(strike, expiry);
        let c_kp = self.call_price(strike + dk, expiry);
        let c_km = self.call_price((strike - dk).max(1e-8), expiry);

        let t_minus = (expiry - dt).max(1e-6);
        let t_plus = expiry + dt;
        let c_tp = self.call_price(strike, t_plus);
        let c_tm = self.call_price(strike, t_minus);

        let dcdt = (c_tp - c_tm) / (t_plus - t_minus);
        let dcdk = (c_kp - c_km) / (2.0 * dk);
        let d2cdk2 = (c_kp - 2.0 * c0 + c_km) / (dk * dk);
        let denom = strike * strike * d2cdk2;

        // Full Dupire formula on undiscounted call prices C~(K,T) = E[(S_T - K)^+]
        // built with the maturity-dependent forward F(T) = forward * exp((r - q) T).
        // From the discounted Dupire PDE
        //   dC/dT = 0.5 sigma^2 K^2 d2C/dK2 - (r - q) K dC/dK - q C
        // and C~ = exp(r T) C one obtains
        //   dC~/dT = 0.5 sigma^2 K^2 d2C~/dK2 - (r - q) K dC~/dK + (r - q) C~,
        // i.e.
        //   sigma_loc^2 = 2 [dC~/dT + (r - q) (K dC~/dK - C~)] / (K^2 d2C~/dK2).
        // For r = q this reduces to 2 dC~/dT / (K^2 d2C~/dK2).
        let carry = self.rate - self.dividend;
        let numer = 2.0 * (dcdt + carry * (strike * dcdk - c0));
        let local_var = if denom > 1e-12 { numer / denom } else { -1.0 };
        if local_var.is_finite() && local_var > 0.0 {
            local_var.sqrt()
        } else {
            self.surface.implied_vol(strike, expiry).max(1e-8)
        }
    }

    fn call_price(&self, strike: f64, expiry: f64) -> f64 {
        let t = expiry.max(1e-8);
        let vol = self.surface.implied_vol(strike, t).max(1e-8);
        let fwd = self.forward * ((self.rate - self.dividend) * t).exp();
        black_76_price(OptionType::Call, fwd, strike, 0.0, vol, t)
    }
}

pub fn dupire_local_vol<S: ImpliedVolSurface>(
    surface: S,
    forward: f64,
    spot: f64,
    expiry: f64,
) -> f64 {
    DupireLocalVol::new(surface, forward).local_vol(spot, expiry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[derive(Debug, Clone, Copy)]
    struct FlatVolSurface {
        vol: f64,
    }

    impl ImpliedVolSurface for FlatVolSurface {
        fn implied_vol(&self, _strike: f64, _expiry: f64) -> f64 {
            self.vol
        }
    }

    #[test]
    fn dupire_local_vol_matches_flat_surface() {
        let lv = DupireLocalVol::new(FlatVolSurface { vol: 0.24 }, 100.0);

        for &t in &[0.25, 0.5, 1.0, 2.0] {
            for &k in &[70.0, 85.0, 100.0, 120.0, 140.0] {
                let sigma_loc = lv.local_vol(k, t);
                assert_relative_eq!(sigma_loc, 0.24, epsilon = 2e-3);
            }
        }
    }

    #[test]
    fn dupire_local_vol_matches_flat_surface_with_unequal_rate_and_dividend() {
        // For a flat implied-vol surface sigma(K,T) = sigma, the undiscounted call
        // is C~(K,T) = Black76(F(T), K, sigma sqrt(T)) with F(T) = S exp((r-q)T).
        // Then dC~/dT = dB/dT|_F + (r-q) F dB/dF, and by Euler homogeneity
        // F dB/dF = C~ - K dC~/dK, so the carry terms in
        //   sigma_loc^2 = 2 [dC~/dT + (r-q)(K dC~/dK - C~)] / (K^2 d2C~/dK2)
        // cancel exactly against the forward drift, leaving
        //   sigma_loc^2 = 2 dB/dT|_F / (K^2 d2B/dK2) = sigma^2.
        // Hence local vol must equal the (flat) implied vol for any r, q.
        let lv = DupireLocalVol::new(FlatVolSurface { vol: 0.2 }, 100.0).with_rates(0.05, 0.02);

        for &t in &[0.25, 0.5, 1.0, 2.0] {
            for &k in &[70.0, 85.0, 100.0, 120.0, 140.0] {
                let sigma_loc = lv.local_vol(k, t);
                assert_relative_eq!(sigma_loc, 0.2, epsilon = 2e-3);
            }
        }
    }

    #[test]
    fn dupire_local_vol_works_through_references() {
        let surface = FlatVolSurface { vol: 0.3 };
        let lv = DupireLocalVol::new(&surface, 100.0).with_rates(0.03, 0.0);
        assert_relative_eq!(lv.local_vol(100.0, 1.0), 0.3, epsilon = 2e-3);
    }
}
