//! Volatility analytics for Smile.
//!
//! Module openferric::vol::smile provides smile/surface construction or volatility inversion utilities.

use crate::core::OptionType;
use crate::math::{CubicSpline, normal_cdf, normal_inv_cdf, normal_pdf};
use crate::pricing::european::black_scholes_price;
use crate::vol::builder::BuiltVolSurface;
use crate::vol::sabr::SabrParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmileDynamics {
    StickyStrike,
    StickyDelta,
}

#[derive(Debug, Clone)]
pub struct SmileSlice {
    pub strikes: Vec<f64>,
    pub vols: Vec<f64>,
    strike_spline: CubicSpline,
}

impl SmileSlice {
    pub fn new(strikes: Vec<f64>, vols: Vec<f64>) -> Result<Self, String> {
        if strikes.len() != vols.len() || strikes.len() < 2 {
            return Err("smile slice requires >= 2 strike/vol points".to_string());
        }
        if strikes.iter().any(|&k| k <= 0.0) {
            return Err("smile strikes must be > 0".to_string());
        }
        if vols.iter().any(|&v| v <= 0.0) {
            return Err("smile vols must be > 0".to_string());
        }
        if strikes.windows(2).any(|w| w[1] <= w[0]) {
            return Err("smile strikes must be strictly increasing".to_string());
        }

        let strike_spline =
            CubicSpline::new(strikes.clone(), vols.clone()).map_err(|_| "invalid spline data")?;

        Ok(Self {
            strikes,
            vols,
            strike_spline,
        })
    }

    pub fn vol_at_strike(&self, strike: f64) -> f64 {
        self.strike_spline.interpolate(strike).max(1e-8)
    }
}

#[derive(Debug, Clone)]
pub struct StickyStrikeSmile {
    pub expiry: f64,
    pub slice: SmileSlice,
}

impl StickyStrikeSmile {
    pub fn new(expiry: f64, strikes: Vec<f64>, vols: Vec<f64>) -> Result<Self, String> {
        if expiry <= 0.0 {
            return Err("expiry must be > 0".to_string());
        }
        Ok(Self {
            expiry,
            slice: SmileSlice::new(strikes, vols)?,
        })
    }

    pub fn from_built_surface(
        surface: &BuiltVolSurface,
        expiry: f64,
        strikes: Vec<f64>,
    ) -> Result<Self, String> {
        if strikes.len() < 2 {
            return Err("at least two strikes are required".to_string());
        }
        let vols = strikes
            .iter()
            .map(|&k| surface.implied_vol(k, expiry).max(1e-8))
            .collect();
        Self::new(expiry, strikes, vols)
    }

    pub fn vol(&self, strike: f64) -> f64 {
        self.slice.vol_at_strike(strike)
    }
}

#[derive(Debug, Clone)]
pub struct StickyDeltaSmile {
    deltas: Vec<f64>,
    delta_spline: CubicSpline,
}

impl StickyDeltaSmile {
    pub fn new(deltas: Vec<f64>, vols: Vec<f64>) -> Result<Self, String> {
        if deltas.len() != vols.len() || deltas.len() < 2 {
            return Err("sticky-delta smile requires >= 2 delta/vol points".to_string());
        }
        if vols.iter().any(|&v| v <= 0.0) {
            return Err("sticky-delta vols must be > 0".to_string());
        }
        if deltas.windows(2).any(|w| w[1] <= w[0]) {
            return Err("sticky-delta deltas must be strictly increasing".to_string());
        }

        let delta_spline =
            CubicSpline::new(deltas.clone(), vols).map_err(|_| "invalid delta smile data")?;
        Ok(Self {
            deltas,
            delta_spline,
        })
    }

    pub fn vol_at_delta(&self, delta: f64) -> f64 {
        let d = delta.clamp(self.deltas[0], self.deltas[self.deltas.len() - 1]);
        self.delta_spline.interpolate(d).max(1e-8)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn strike_from_delta(
        &self,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        expiry: f64,
        target_delta: f64,
        initial_strike: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<f64, String> {
        strike_from_delta_newton(
            spot,
            rate,
            dividend_yield,
            expiry,
            target_delta,
            self.vol_at_delta(target_delta),
            initial_strike,
            tol,
            max_iter,
        )
    }
}

fn d1(spot: f64, strike: f64, rate: f64, dividend_yield: f64, sigma: f64, expiry: f64) -> f64 {
    ((spot / strike).ln() + (rate - dividend_yield + 0.5 * sigma * sigma) * expiry)
        / (sigma * expiry.sqrt())
}

fn bsm_delta(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    sigma: f64,
    expiry: f64,
) -> f64 {
    if spot <= 0.0 || strike <= 0.0 || sigma <= 0.0 || expiry <= 0.0 {
        return 0.0;
    }

    let d1 = d1(spot, strike, rate, dividend_yield, sigma, expiry);
    let df_q = (-dividend_yield * expiry).exp();

    match option_type {
        OptionType::Call => df_q * normal_cdf(d1),
        OptionType::Put => df_q * (normal_cdf(d1) - 1.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn strike_from_delta_newton(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    target_delta: f64,
    sigma: f64,
    initial_strike: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    if spot <= 0.0 || sigma <= 0.0 || expiry <= 0.0 {
        return Err("spot, sigma, expiry must be > 0".to_string());
    }

    let option_type = if target_delta >= 0.0 {
        OptionType::Call
    } else {
        OptionType::Put
    };

    let mut strike = initial_strike.max(1e-8);
    for _ in 0..max_iter.max(1) {
        let d1_val = d1(spot, strike, rate, dividend_yield, sigma, expiry);
        let model_delta = bsm_delta(
            option_type,
            spot,
            strike,
            rate,
            dividend_yield,
            sigma,
            expiry,
        );
        let f = model_delta - target_delta;
        if f.abs() <= tol.max(1e-12) {
            return Ok(strike.max(1e-8));
        }

        let df_q = (-dividend_yield * expiry).exp();
        let derivative = -df_q * normal_pdf(d1_val) / (strike * sigma * expiry.sqrt());
        if derivative.abs() < 1e-14 {
            break;
        }

        let next = strike - f / derivative;
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        if (next - strike).abs() <= tol.max(1e-12) {
            return Ok(next.max(1e-8));
        }
        strike = next;
    }

    // Bisection fallback.
    let mut lo = spot * 1e-3;
    let mut hi = spot * 10.0;
    let mut flo =
        bsm_delta(option_type, spot, lo, rate, dividend_yield, sigma, expiry) - target_delta;

    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        let fm =
            bsm_delta(option_type, spot, mid, rate, dividend_yield, sigma, expiry) - target_delta;

        if fm.abs() <= tol.max(1e-12) {
            return Ok(mid.max(1e-8));
        }

        if flo * fm <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }

    Ok(0.5 * (lo + hi).max(1e-8))
}

pub fn sabr_alpha_from_atm_vol(
    forward: f64,
    expiry: f64,
    atm_vol: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    if forward <= 0.0 || expiry <= 0.0 || atm_vol <= 0.0 {
        return 0.2;
    }

    let beta = beta.clamp(0.0, 1.0);
    let mut alpha = (atm_vol * forward.powf(1.0 - beta)).max(1e-6);

    for _ in 0..100 {
        let params = SabrParams {
            alpha,
            beta,
            rho,
            nu,
        };
        let f = params.implied_vol(forward, forward, expiry) - atm_vol;
        if f.abs() < 1e-12 {
            return alpha.max(1e-8);
        }

        let eps = (alpha * 1e-4).max(1e-7);
        let f_up = SabrParams {
            alpha: alpha + eps,
            beta,
            rho,
            nu,
        }
        .implied_vol(forward, forward, expiry)
            - atm_vol;
        let f_dn = SabrParams {
            alpha: (alpha - eps).max(1e-8),
            beta,
            rho,
            nu,
        }
        .implied_vol(forward, forward, expiry)
            - atm_vol;
        let df = (f_up - f_dn) / (2.0 * eps);
        if df.abs() < 1e-14 {
            break;
        }

        let next = alpha - f / df;
        if !next.is_finite() || next <= 0.0 {
            break;
        }
        if (next - alpha).abs() < 1e-12 {
            return next.max(1e-8);
        }
        alpha = next.max(1e-8);
    }

    alpha.max(1e-8)
}

pub fn sabr_smile_from_atm(
    forward: f64,
    expiry: f64,
    atm_vol: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    strikes: &[f64],
) -> (SabrParams, Vec<(f64, f64)>) {
    let alpha = sabr_alpha_from_atm_vol(forward, expiry, atm_vol, beta, rho, nu);
    let params = SabrParams {
        alpha,
        beta: beta.clamp(0.0, 1.0),
        rho: rho.clamp(-0.999, 0.999),
        nu: nu.max(0.0),
    };

    let mut out = Vec::new();
    for &k in strikes {
        if k > 0.0 {
            out.push((k, params.implied_vol(forward, k, expiry).max(1e-8)));
        }
    }

    (params, out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VannaVolgaQuote {
    pub atm_vol: f64,
    pub rr_25d: f64,
    pub bf_25d: f64,
}

impl VannaVolgaQuote {
    pub fn new(atm_vol: f64, rr_25d: f64, bf_25d: f64) -> Self {
        Self {
            atm_vol,
            rr_25d,
            bf_25d,
        }
    }
}

fn bsm_price_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    sigma: f64,
    expiry: f64,
) -> f64 {
    if dividend_yield.abs() < 1e-14 {
        return black_scholes_price(option_type, spot, strike, rate, sigma, expiry);
    }

    if spot <= 0.0 || strike <= 0.0 || sigma <= 0.0 || expiry <= 0.0 {
        return match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        };
    }

    let d1 = d1(spot, strike, rate, dividend_yield, sigma, expiry);
    let d2 = d1 - sigma * expiry.sqrt();
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    match option_type {
        OptionType::Call => spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2),
        OptionType::Put => strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vanna_volga_price(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    quote: VannaVolgaQuote,
) -> f64 {
    let atm_vol = quote.atm_vol.max(1e-8);
    let rr = quote.rr_25d;
    let bf = quote.bf_25d;

    let base = bsm_price_with_dividend(
        option_type,
        spot,
        strike,
        rate,
        dividend_yield,
        atm_vol,
        expiry,
    );

    if spot <= 0.0 || strike <= 0.0 || expiry <= 0.0 {
        return base;
    }

    let x = (strike / spot).ln();
    let scale = (atm_vol * expiry.sqrt()).max(1e-8);

    // ATM-neutral weights: exactly zero correction at x=0.
    let vanna_weight = x / scale;
    let volga_weight = (x * x) / (scale * scale);

    let vol_25c = (atm_vol + bf + 0.5 * rr).max(1e-8);
    let vol_25p = (atm_vol + bf - 0.5 * rr).max(1e-8);

    let k_25c = strike_from_delta_newton(
        spot,
        rate,
        dividend_yield,
        expiry,
        0.25,
        vol_25c,
        spot,
        1e-12,
        100,
    )
    .unwrap_or(spot);

    let k_25p = strike_from_delta_newton(
        spot,
        rate,
        dividend_yield,
        expiry,
        -0.25,
        vol_25p,
        spot,
        1e-12,
        100,
    )
    .unwrap_or(spot);

    let call_25c_smile = bsm_price_with_dividend(
        OptionType::Call,
        spot,
        k_25c,
        rate,
        dividend_yield,
        vol_25c,
        expiry,
    );
    let call_25c_atm = bsm_price_with_dividend(
        OptionType::Call,
        spot,
        k_25c,
        rate,
        dividend_yield,
        atm_vol,
        expiry,
    );

    let put_25p_smile = bsm_price_with_dividend(
        OptionType::Put,
        spot,
        k_25p,
        rate,
        dividend_yield,
        vol_25p,
        expiry,
    );
    let put_25p_atm = bsm_price_with_dividend(
        OptionType::Put,
        spot,
        k_25p,
        rate,
        dividend_yield,
        atm_vol,
        expiry,
    );

    let rr_adjustment = (call_25c_smile - call_25c_atm) - (put_25p_smile - put_25p_atm);
    let bf_adjustment = 0.5 * ((call_25c_smile - call_25c_atm) + (put_25p_smile - put_25p_atm));

    base + vanna_weight * rr_adjustment + volga_weight * bf_adjustment
}

#[allow(clippy::too_many_arguments)]
pub fn shift_smile_for_spot_move(
    slice: &SmileSlice,
    spot_old: f64,
    spot_new: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    dynamics: SmileDynamics,
) -> Result<SmileSlice, String> {
    if spot_old <= 0.0 || spot_new <= 0.0 || expiry <= 0.0 {
        return Err("spot_old, spot_new, expiry must be > 0".to_string());
    }

    match dynamics {
        SmileDynamics::StickyStrike => SmileSlice::new(slice.strikes.clone(), slice.vols.clone()),
        SmileDynamics::StickyDelta => {
            let mut delta_vol_pairs: Vec<(f64, f64)> = slice
                .strikes
                .iter()
                .zip(slice.vols.iter())
                .map(|(&k, &v)| {
                    (
                        bsm_delta(
                            OptionType::Call,
                            spot_old,
                            k,
                            rate,
                            dividend_yield,
                            v,
                            expiry,
                        ),
                        v,
                    )
                })
                .collect();

            delta_vol_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut deltas: Vec<f64> = Vec::new();
            let mut vols: Vec<f64> = Vec::new();
            for (delta, vol) in delta_vol_pairs {
                if deltas
                    .last()
                    .is_none_or(|prev| (delta - *prev).abs() > 1e-8)
                {
                    deltas.push(delta);
                    vols.push(vol);
                }
            }

            if deltas.len() < 2 {
                return SmileSlice::new(slice.strikes.clone(), slice.vols.clone());
            }

            let delta_smile = StickyDeltaSmile::new(deltas, vols)?;
            let mut shifted_vols = Vec::with_capacity(slice.strikes.len());

            for (&k, &v0) in slice.strikes.iter().zip(slice.vols.iter()) {
                let mut vol = v0.max(1e-8);
                for _ in 0..12 {
                    let delta = bsm_delta(
                        OptionType::Call,
                        spot_new,
                        k,
                        rate,
                        dividend_yield,
                        vol,
                        expiry,
                    );
                    let next = delta_smile.vol_at_delta(delta).max(1e-8);
                    if (next - vol).abs() < 1e-10 {
                        vol = next;
                        break;
                    }
                    vol = 0.5 * (vol + next);
                }
                shifted_vols.push(vol.max(1e-8));
            }

            SmileSlice::new(slice.strikes.clone(), shifted_vols)
        }
    }
}

pub fn strike_from_delta_analytic(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    vol: f64,
    delta: f64,
) -> Option<f64> {
    if spot <= 0.0 || expiry <= 0.0 || vol <= 0.0 {
        return None;
    }

    let df_q = (-dividend_yield * expiry).exp();
    let p = if delta >= 0.0 {
        (delta / df_q).clamp(1e-12, 1.0 - 1e-12)
    } else {
        (delta / df_q + 1.0).clamp(1e-12, 1.0 - 1e-12)
    };

    let d1 = normal_inv_cdf(p);
    let denom = d1 * vol * expiry.sqrt() - (rate - dividend_yield + 0.5 * vol * vol) * expiry;
    Some(spot * (-denom).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sticky_delta_strike_solver_recovers_target_delta() {
        let smile =
            StickyDeltaSmile::new(vec![-0.25, -0.1, 0.1, 0.25], vec![0.28, 0.24, 0.22, 0.24])
                .unwrap();

        let strike = smile
            .strike_from_delta(100.0, 0.02, 0.0, 1.0, 0.25, 100.0, 1e-12, 100)
            .unwrap();

        let vol = smile.vol_at_delta(0.25);
        let delta = bsm_delta(OptionType::Call, 100.0, strike, 0.02, 0.0, vol, 1.0);

        assert_relative_eq!(delta, 0.25, epsilon = 1e-8);
    }

    #[test]
    fn sabr_smile_matches_atm_input_vol() {
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let (_params, smile) = sabr_smile_from_atm(100.0, 1.0, 0.2, 0.5, -0.2, 0.5, &strikes);

        let atm = smile
            .iter()
            .find(|(k, _)| (*k - 100.0).abs() < 1e-12)
            .map(|(_, v)| *v)
            .unwrap();

        assert_relative_eq!(atm, 0.2, epsilon = 1e-6);
    }

    #[test]
    fn vanna_volga_is_atm_neutral() {
        let quote = VannaVolgaQuote::new(0.2, 0.03, 0.01);
        let vv = vanna_volga_price(OptionType::Call, 100.0, 100.0, 0.01, 0.0, 1.0, quote);
        let bs = black_scholes_price(OptionType::Call, 100.0, 100.0, 0.01, 0.2, 1.0);

        assert_relative_eq!(vv, bs, epsilon = 1e-12);
    }
}
