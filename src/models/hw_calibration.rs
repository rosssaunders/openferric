//! Module `models::hw_calibration`.
//!
//! Implements hw calibration workflows with concrete routines such as `hw_atm_swaption_vol_approx`, `calibrate_hull_white_params`.
//!
//! Volatility convention: every swaption volatility in this module is an ATM
//! **normal (Bachelier)** volatility of the forward swap rate, in absolute
//! rate terms (Hull-White `sigma` scale, e.g. `0.0080` = 80 bp/yr). Lognormal
//! (Black) volatilities (e.g. `0.20` at 4% rates) are a different unit and
//! must be converted first; at the money
//! `sigma_normal ~= sigma_black * forward_swap_rate`.
//!
//! References: Hull and White (1990), Brigo and Mercurio (2006) Ch. 3, short-rate calibration relations around Eq. (3.28).
//!
//! Key types and purpose: `AtmSwaptionVolQuote` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.

/// Market ATM swaption volatility quote `(expiry, tenor, market_vol)`.
///
/// `market_vol` is an ATM **normal (Bachelier)** volatility of the forward
/// swap rate in absolute rate terms (e.g. `0.0080` = 80 bp/yr). It is *not* a
/// lognormal (Black) volatility; convert Black quotes first, using
/// `sigma_normal ~= sigma_black * forward_swap_rate` at the money.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmSwaptionVolQuote {
    pub expiry: f64,
    pub tenor: f64,
    pub market_vol: f64,
}

/// Largest admissible quoted ATM normal volatility (500 bp/yr).
///
/// The model vol satisfies `hw_atm_swaption_vol_approx(..) <= sigma` and the
/// coarse calibration grid caps `sigma` at this level (the fine grid may
/// extend somewhat above it around the coarse optimum), so materially larger
/// quotes cannot be matched and would only pin the optimizer at the grid
/// boundary.
/// Realistic ATM normal vols for rates are O(0.005-0.02); quotes above this
/// bound are almost always lognormal (Black) vols passed in the wrong unit,
/// and [`calibrate_hull_white_params`] rejects them.
pub const MAX_QUOTED_NORMAL_VOL: f64 = 0.05;

/// Approximate ATM **normal (Bachelier)** swaption volatility implied by
/// one-factor Hull-White `(a, sigma)`.
///
/// Freezing the swap-rate weights, the swap rate is approximately Gaussian
/// under Hull-White with instantaneous absolute volatility
/// `sigma * exp(-a * (expiry - t)) * B(0, tenor) / tenor`, where
/// `B(0, tau) = (1 - exp(-a * tau)) / a` is the frozen duration ratio.
/// Averaging that variance over `[0, expiry]` gives
///
/// ```text
/// sigma_n = sigma * sqrt((1 - exp(-2 a T)) / (2 a T)) * (1 - exp(-a tau)) / (a tau)
/// ```
///
/// The result is an absolute rate volatility at the Hull-White `sigma` scale
/// (e.g. `~0.01`) with no dependence on the rate level. It is **not** a
/// lognormal (Black) volatility (`~0.20` at 4% rates); at the money the two
/// are related by `sigma_black ~= sigma_n / forward_swap_rate`.
pub fn hw_atm_swaption_vol_approx(a: f64, sigma: f64, expiry: f64, tenor: f64) -> f64 {
    if !a.is_finite()
        || !sigma.is_finite()
        || !expiry.is_finite()
        || !tenor.is_finite()
        || a < 0.0
        || sigma < 0.0
        || expiry <= 0.0
        || tenor <= 0.0
    {
        return f64::NAN;
    }

    if a <= 1.0e-10 {
        return sigma;
    }

    let expiry_factor = ((1.0 - (-2.0 * a * expiry).exp()) / (2.0 * a * expiry))
        .max(0.0)
        .sqrt();
    let tenor_factor = ((1.0 - (-a * tenor).exp()) / (a * tenor)).max(0.0);
    sigma * expiry_factor * tenor_factor
}

/// Calibrates Hull-White `(a, sigma)` by minimizing squared differences of
/// ATM swaption prices per unit annuity under the Bachelier (normal) model.
///
/// Input quotes are `(expiry, tenor, market_vol)` tuples, where `market_vol`
/// is an ATM **normal (Bachelier)** volatility in absolute rate terms (see
/// [`AtmSwaptionVolQuote`]). Returns `None` when any quote is non-finite,
/// non-positive, or larger than [`MAX_QUOTED_NORMAL_VOL`]: such quotes lie
/// outside the search domain (model vols satisfy `model_vol <= sigma`) and
/// are almost always lognormal Black vols (e.g. `0.20`) passed in the wrong
/// unit, so they are rejected loudly instead of silently producing a
/// boundary-pinned degenerate fit.
pub fn calibrate_hull_white_params(quotes: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    if quotes.is_empty() {
        return None;
    }
    if quotes.iter().any(|(e, t, v)| {
        !e.is_finite()
            || !t.is_finite()
            || !v.is_finite()
            || *e <= 0.0
            || *t <= 0.0
            || *v <= 0.0
            || *v > MAX_QUOTED_NORMAL_VOL
    }) {
        return None;
    }

    let coarse = grid_search(quotes, 0.001, 0.30, 0.001, MAX_QUOTED_NORMAL_VOL, 81, 81);
    let (a0, sigma0, _) = coarse?;

    let fine_a_lo = (a0 * 0.4).max(1.0e-4);
    let fine_a_hi = (a0 * 1.6).max(fine_a_lo + 1.0e-4);
    let fine_sigma_lo = (sigma0 * 0.4).max(1.0e-4);
    let fine_sigma_hi = (sigma0 * 1.6).max(fine_sigma_lo + 1.0e-4);

    let mut best = grid_search(
        quotes,
        fine_a_lo,
        fine_a_hi,
        fine_sigma_lo,
        fine_sigma_hi,
        81,
        81,
    )?;

    // Resolve the minimum rather than returning the nearest point on the
    // relatively coarse 81x81 grid above.  Each nested grid spans two prior
    // grid spacings on either side of the incumbent and reduces both spacings
    // by a factor of five.  Six rounds make the final parameter resolution
    // about 6.4e-5 of the first fine-grid spacing.
    let mut a_step = (fine_a_hi - fine_a_lo) / 80.0;
    let mut sigma_step = (fine_sigma_hi - fine_sigma_lo) / 80.0;
    for _ in 0..6 {
        let a_lo = (best.0 - 2.0 * a_step).max(1.0e-8);
        let a_hi = best.0 + 2.0 * a_step;
        let sigma_lo = (best.1 - 2.0 * sigma_step).max(1.0e-8);
        let sigma_hi = best.1 + 2.0 * sigma_step;
        best = grid_search(quotes, a_lo, a_hi, sigma_lo, sigma_hi, 21, 21)?;
        a_step = (a_hi - a_lo) / 20.0;
        sigma_step = (sigma_hi - sigma_lo) / 20.0;
    }

    Some((best.0, best.1))
}

fn grid_search(
    quotes: &[(f64, f64, f64)],
    a_lo: f64,
    a_hi: f64,
    sigma_lo: f64,
    sigma_hi: f64,
    a_points: usize,
    sigma_points: usize,
) -> Option<(f64, f64, f64)> {
    if a_points < 2 || sigma_points < 2 || a_hi <= a_lo || sigma_hi <= sigma_lo {
        return None;
    }

    let da = (a_hi - a_lo) / (a_points as f64 - 1.0);
    let ds = (sigma_hi - sigma_lo) / (sigma_points as f64 - 1.0);

    let mut best = (0.0, 0.0, f64::INFINITY);
    for i in 0..a_points {
        let a = a_lo + i as f64 * da;
        for j in 0..sigma_points {
            let sigma = sigma_lo + j as f64 * ds;
            let err = calibration_objective(quotes, a, sigma);
            if err < best.2 {
                best = (a, sigma, err);
            }
        }
    }
    if best.2.is_finite() { Some(best) } else { None }
}

fn calibration_objective(quotes: &[(f64, f64, f64)], a: f64, sigma: f64) -> f64 {
    quotes
        .iter()
        .map(|(expiry, tenor, market_vol)| {
            let model_vol = hw_atm_swaption_vol_approx(a, sigma, *expiry, *tenor);
            let market_price = normalized_atm_bachelier_price(*market_vol, *expiry);
            let model_price = normalized_atm_bachelier_price(model_vol, *expiry);
            let err = model_price - market_price;
            err * err
        })
        .sum()
}

/// ATM payer swaption price per unit annuity under the Bachelier model.
///
/// With normal volatility `sigma_n`, the ATM forward price is
/// `E[(S_T - F)^+] = sigma_n * sqrt(T) * phi(0) = sigma_n * sqrt(T / (2*pi))`
/// (Bachelier ATM formula; see e.g. Hull, "Options, Futures, and Other
/// Derivatives", normal-model appendix). Both the market quote and the
/// Hull-White approximation are normal vols, so this keeps the objective
/// unit-consistent; the annuity factor is common to both sides and cancels.
fn normalized_atm_bachelier_price(vol: f64, expiry: f64) -> f64 {
    if !vol.is_finite() || vol < 0.0 || !expiry.is_finite() || expiry <= 0.0 {
        return f64::NAN;
    }

    vol * (expiry / (2.0 * std::f64::consts::PI)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_recovers_synthetic_hull_white_parameters() {
        let true_a = 0.05;
        let true_sigma = 0.01;

        // Self-consistency round trip: quotes generated from the model's own
        // normal-vol approximation must be recovered. This cannot detect unit
        // mismatches by construction; the external-anchor coverage lives in
        // tests/audit_hw_calibration.rs.
        let mut market_quotes = Vec::new();
        for expiry in [1.0, 2.0, 3.0, 5.0] {
            for tenor in [2.0, 5.0] {
                let vol = hw_atm_swaption_vol_approx(true_a, true_sigma, expiry, tenor);
                market_quotes.push((expiry, tenor, vol));
            }
        }

        let (cal_a, cal_sigma) = calibrate_hull_white_params(&market_quotes).unwrap();
        assert!(
            (cal_a - true_a).abs() <= 5.0e-8,
            "a: calibrated={cal_a}, target={true_a}"
        );
        assert!(
            (cal_sigma - true_sigma).abs() <= 1.0e-8,
            "sigma: calibrated={cal_sigma}, target={true_sigma}"
        );

        for &(expiry, tenor, market_vol) in &market_quotes {
            let model_vol = hw_atm_swaption_vol_approx(cal_a, cal_sigma, expiry, tenor);
            let model_price = normalized_atm_bachelier_price(model_vol, expiry);
            let market_price = normalized_atm_bachelier_price(market_vol, expiry);
            assert!(
                (model_price - market_price).abs() <= 3.0e-9,
                "expiry={expiry} tenor={tenor}: model_price={model_price} market_price={market_price} model_vol={model_vol} market_vol={market_vol}"
            );
        }
    }

    #[test]
    fn lognormal_scale_quotes_are_rejected() {
        // 0.20 is a typical ATM *Black* (lognormal) vol at ~4% rates. As a
        // normal vol it would be 2000 bp/yr, far beyond MAX_QUOTED_NORMAL_VOL
        // and unreachable by the model (model_vol <= sigma <= 0.05 on the
        // search grid). Before the unit fix this silently returned a
        // boundary-pinned degenerate fit of (a, sigma) ~ (4e-4, 0.08).
        let quotes = vec![(1.0, 5.0, 0.20), (3.0, 5.0, 0.19), (5.0, 10.0, 0.18)];
        assert!(calibrate_hull_white_params(&quotes).is_none());

        // A single wrong-unit quote poisons the set as well.
        let mixed = vec![(1.0, 5.0, 0.009), (5.0, 5.0, 0.20)];
        assert!(calibrate_hull_white_params(&mixed).is_none());

        // Quotes at the boundary itself remain admissible.
        let boundary = vec![(1.0, 5.0, MAX_QUOTED_NORMAL_VOL)];
        assert!(calibrate_hull_white_params(&boundary).is_some());
    }
}
