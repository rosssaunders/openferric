use crate::pricing::OptionType;
use crate::pricing::european::black_76_price;

pub trait ImpliedVolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64;
}

impl ImpliedVolSurface for crate::vol::surface::VolSurface {
    fn implied_vol(&self, strike: f64, expiry: f64) -> f64 {
        self.vol(strike, expiry)
    }
}

#[derive(Debug, Clone)]
pub struct DupireLocalVol<S> {
    surface: S,
    forward: f64,
    strike_bump_rel: f64,
    time_bump: f64,
}

impl<S: ImpliedVolSurface> DupireLocalVol<S> {
    pub fn new(surface: S, forward: f64) -> Self {
        Self {
            surface,
            forward: forward.max(1e-12),
            strike_bump_rel: 1e-3,
            time_bump: 1e-3,
        }
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
        let d2cdk2 = (c_kp - 2.0 * c0 + c_km) / (dk * dk);
        let denom = strike * strike * d2cdk2;

        // Dupire formula in forward measure (r=0): sigma_loc^2 = 2*dC/dT / (K^2*d2C/dK2).
        let local_var = if denom > 1e-12 {
            (2.0 * dcdt) / denom
        } else {
            -1.0
        };
        if local_var.is_finite() && local_var > 0.0 {
            local_var.sqrt()
        } else {
            self.surface.implied_vol(strike, expiry).max(1e-8)
        }
    }

    fn call_price(&self, strike: f64, expiry: f64) -> f64 {
        let vol = self.surface.implied_vol(strike, expiry).max(1e-8);
        black_76_price(
            OptionType::Call,
            self.forward,
            strike,
            0.0,
            vol,
            expiry.max(1e-8),
        )
    }
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
}
