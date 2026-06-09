//! Module `vol::smile`.
//!
//! Implements smile workflows with concrete routines such as `sabr_alpha_from_atm_vol`, `sabr_smile_from_atm`, `vanna_volga_price`, `shift_smile_for_spot_move`.
//!
//! References: Hagan et al. (2002), Hull (11th ed.) Ch. 18, SABR asymptotic volatility formula around Eq. (A.69).
//!
//! Key types and purpose: `SmileDynamics`, `SmileSlice`, `StickyStrikeSmile`, `StickyDeltaSmile`, `VannaVolgaQuote` define the core data contracts for this module.
//!
//! Numerical considerations: enforce positivity and no-arbitrage constraints, and guard root-finding with robust brackets for wings or short maturities.
//!
//! When to use: use these tools for smile/surface construction and implied-vol inversion; choose local/stochastic-vol models when dynamics, not just static fits, are needed.
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

    // Bisection fallback. Verify the bracket before iterating: returning an
    // unbracketed midpoint would silently produce a wrong strike when the
    // target delta is unattainable on [lo, hi].
    let mut lo = spot * 1e-3;
    let mut hi = spot * 10.0;
    let mut flo =
        bsm_delta(option_type, spot, lo, rate, dividend_yield, sigma, expiry) - target_delta;
    let fhi = bsm_delta(option_type, spot, hi, rate, dividend_yield, sigma, expiry) - target_delta;

    if flo == 0.0 {
        return Ok(lo.max(1e-8));
    }
    if fhi == 0.0 {
        return Ok(hi.max(1e-8));
    }
    if flo * fhi > 0.0 {
        return Err(format!(
            "target delta {target_delta} not bracketed by strikes [{lo}, {hi}]"
        ));
    }

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

    // Convergence verification: the bracket is valid, but the residual must be
    // small before accepting the midpoint.
    let mid = 0.5 * (lo + hi);
    let fm = bsm_delta(option_type, spot, mid, rate, dividend_yield, sigma, expiry) - target_delta;
    if fm.abs() <= 1e-6 {
        Ok(mid.max(1e-8))
    } else {
        Err(format!(
            "strike-from-delta bisection did not converge: residual {fm} at strike {mid}"
        ))
    }
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

/// Black-Scholes vega `dV/dsigma` with continuous dividend yield.
fn bsm_vega(
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
    spot * (-dividend_yield * expiry).exp() * normal_pdf(d1) * expiry.sqrt()
}

/// Pivot strikes `(k_25d_put, k_atm, k_25d_call)` used by [`vanna_volga_price`].
///
/// Pivot vols follow the module quote convention `vol_25d = atm + bf +/- rr/2`.
/// The ATM pivot is the delta-neutral straddle strike
/// `K_atm = S * exp((r - q + sigma_atm^2 / 2) * T)`.
pub fn vanna_volga_pivot_strikes(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
    quote: VannaVolgaQuote,
) -> (f64, f64, f64) {
    let atm_vol = quote.atm_vol.max(1e-8);
    let vol_25c = (atm_vol + quote.bf_25d + 0.5 * quote.rr_25d).max(1e-8);
    let vol_25p = (atm_vol + quote.bf_25d - 0.5 * quote.rr_25d).max(1e-8);

    let k_atm = spot * ((rate - dividend_yield + 0.5 * atm_vol * atm_vol) * expiry.max(0.0)).exp();

    let k_25c = strike_from_delta_newton(
        spot,
        rate,
        dividend_yield,
        expiry,
        0.25,
        vol_25c,
        k_atm,
        1e-12,
        100,
    )
    .ok()
    .or_else(|| strike_from_delta_analytic(spot, rate, dividend_yield, expiry, vol_25c, 0.25))
    .unwrap_or(spot);

    let k_25p = strike_from_delta_newton(
        spot,
        rate,
        dividend_yield,
        expiry,
        -0.25,
        vol_25p,
        k_atm,
        1e-12,
        100,
    )
    .ok()
    .or_else(|| strike_from_delta_analytic(spot, rate, dividend_yield, expiry, vol_25p, -0.25))
    .unwrap_or(spot);

    (k_25p, k_atm, k_25c)
}

/// Vanna-volga price following Castagna-Mercurio (2007), first-order form:
///
/// `VV(K) = BS(K; sigma_atm) + sum_i w_i(K) * [MKT(K_i) - BS(K_i; sigma_atm)]`
///
/// with the closed-form logarithmic weights mapped through vega ratios
///
/// `w_i(K) = (vega(K) / vega(K_i)) * y_i(K)`,
/// `y_1(K) = ln(K2/K) ln(K3/K) / (ln(K2/K1) ln(K3/K1))`,
/// `y_2(K) = ln(K/K1) ln(K3/K) / (ln(K2/K1) ln(K3/K2))`,
/// `y_3(K) = ln(K/K1) ln(K/K2) / (ln(K3/K1) ln(K3/K2))`,
///
/// where `K1 < K2 < K3` are the 25d-put, ATM, and 25d-call pivot strikes.
/// These weights solve the 3x3 system matching vega, vanna, and volga of the
/// target option at the base (ATM) vol, so the construction reprices all three
/// pivots exactly: at `K = K_i` the weights collapse to `w_i = 1`, `w_j = 0`.
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
    let vol_25c = (atm_vol + quote.bf_25d + 0.5 * quote.rr_25d).max(1e-8);
    let vol_25p = (atm_vol + quote.bf_25d - 0.5 * quote.rr_25d).max(1e-8);

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

    let (k1, k2, k3) = vanna_volga_pivot_strikes(spot, rate, dividend_yield, expiry, quote);
    if !(k1 > 0.0 && k2 > k1 && k3 > k2) {
        return base;
    }

    // Smile corrections are computed on calls; by put-call parity the
    // vol-dependent part of call and put prices is identical, so the same
    // correction applies to the target option regardless of its type.
    let pivot_vols = [vol_25p, atm_vol, vol_25c];
    let pivots = [k1, k2, k3];

    let vega_k = bsm_vega(spot, strike, rate, dividend_yield, atm_vol, expiry);
    let ln = |a: f64, b: f64| (a / b).ln();
    let y = [
        ln(k2, strike) * ln(k3, strike) / (ln(k2, k1) * ln(k3, k1)),
        ln(strike, k1) * ln(k3, strike) / (ln(k2, k1) * ln(k3, k2)),
        ln(strike, k1) * ln(strike, k2) / (ln(k3, k1) * ln(k3, k2)),
    ];

    let mut correction = 0.0;
    for i in 0..3 {
        let vega_i = bsm_vega(spot, pivots[i], rate, dividend_yield, atm_vol, expiry);
        if vega_i.abs() < 1e-14 {
            return base;
        }
        let mkt = bsm_price_with_dividend(
            OptionType::Call,
            spot,
            pivots[i],
            rate,
            dividend_yield,
            pivot_vols[i],
            expiry,
        );
        let bs = bsm_price_with_dividend(
            OptionType::Call,
            spot,
            pivots[i],
            rate,
            dividend_yield,
            atm_vol,
            expiry,
        );
        correction += (vega_k / vega_i) * y[i] * (mkt - bs);
    }

    if !correction.is_finite() {
        return base;
    }

    base + correction
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
        // At the ATM pivot strike the market vol equals the base (ATM) vol, so
        // every pivot correction term vanishes and VV == BS exactly.
        let quote = VannaVolgaQuote::new(0.2, 0.03, 0.01);
        let (_, k_atm, _) = vanna_volga_pivot_strikes(100.0, 0.01, 0.0, 1.0, quote);
        let vv = vanna_volga_price(OptionType::Call, 100.0, k_atm, 0.01, 0.0, 1.0, quote);
        let bs = black_scholes_price(OptionType::Call, 100.0, k_atm, 0.01, 0.2, 1.0);

        assert_relative_eq!(vv, bs, epsilon = 1e-12);
    }

    #[test]
    fn vanna_volga_reprices_its_pivots() {
        // Castagna-Mercurio weights collapse to w_i = 1, w_j = 0 at K = K_i, so
        // the VV price at each pivot must equal the BS price computed with that
        // pivot's market vol (atm + bf +/- rr/2 per the module convention).
        let spot = 100.0;
        let rate = 0.02;
        let q = 0.0;
        let expiry = 1.0;
        let quote = VannaVolgaQuote::new(0.2, -0.02, 0.01);

        let atm = quote.atm_vol;
        let vol_25c = atm + quote.bf_25d + 0.5 * quote.rr_25d;
        let vol_25p = atm + quote.bf_25d - 0.5 * quote.rr_25d;

        let (k_25p, k_atm, k_25c) = vanna_volga_pivot_strikes(spot, rate, q, expiry, quote);
        assert!(k_25p < k_atm && k_atm < k_25c);

        for (k, vol) in [(k_25p, vol_25p), (k_atm, atm), (k_25c, vol_25c)] {
            let vv = vanna_volga_price(OptionType::Call, spot, k, rate, q, expiry, quote);
            let market = bsm_price_with_dividend(OptionType::Call, spot, k, rate, q, vol, expiry);
            assert_relative_eq!(vv, market, epsilon = 1e-10);

            // Equivalent statement in vol space: the VV implied vol reproduces
            // the input pivot vol to high precision.
            let iv = crate::vol::implied::implied_vol(
                OptionType::Call,
                spot,
                k,
                rate,
                expiry,
                vv,
                1e-14,
                64,
            )
            .unwrap();
            assert_relative_eq!(iv, vol, epsilon = 1e-8);

            // Put corrections are identical by put-call parity.
            let vv_put = vanna_volga_price(OptionType::Put, spot, k, rate, q, expiry, quote);
            let market_put =
                bsm_price_with_dividend(OptionType::Put, spot, k, rate, q, vol, expiry);
            assert_relative_eq!(vv_put, market_put, epsilon = 1e-10);
        }
    }

    #[test]
    fn strike_from_delta_newton_errors_on_unattainable_delta() {
        // With q > 0, call delta is bounded by exp(-q T) < 1; asking for a
        // larger delta must error rather than silently returning a midpoint.
        let err = strike_from_delta_newton(100.0, 0.01, 0.05, 1.0, 0.99, 0.2, 100.0, 1e-12, 50);
        assert!(err.is_err());
    }
}
