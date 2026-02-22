//! Module `models::hw_calibration`.
//!
//! Implements hw calibration workflows with concrete routines such as `hw_atm_swaption_vol_approx`, `calibrate_hull_white_params`.
//!
//! References: Hull and White (1990), Brigo and Mercurio (2006) Ch. 3, short-rate calibration relations around Eq. (3.28).
//!
//! Key types and purpose: `AtmSwaptionVolQuote` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.
use crate::math::normal_cdf;

/// Market ATM swaption volatility quote `(expiry, tenor, market_vol)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmSwaptionVolQuote {
    pub expiry: f64,
    pub tenor: f64,
    pub market_vol: f64,
}

/// Approximate ATM Black volatility implied by one-factor Hull-White `(a, sigma)`.
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

/// Calibrates Hull-White `(a, sigma)` by minimizing ATM swaption pricing errors.
///
/// Input quotes are `(expiry, tenor, market_vol)` tuples.
pub fn calibrate_hull_white_params(quotes: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    if quotes.is_empty() {
        return None;
    }
    if quotes.iter().any(|(e, t, v)| {
        !e.is_finite() || !t.is_finite() || !v.is_finite() || *e <= 0.0 || *t <= 0.0 || *v <= 0.0
    }) {
        return None;
    }

    let coarse = grid_search(quotes, 0.001, 0.30, 0.001, 0.05, 81, 81);
    let (a0, sigma0, _) = coarse?;

    let fine_a_lo = (a0 * 0.4).max(1.0e-4);
    let fine_a_hi = (a0 * 1.6).max(fine_a_lo + 1.0e-4);
    let fine_sigma_lo = (sigma0 * 0.4).max(1.0e-4);
    let fine_sigma_hi = (sigma0 * 1.6).max(fine_sigma_lo + 1.0e-4);

    let fine = grid_search(
        quotes,
        fine_a_lo,
        fine_a_hi,
        fine_sigma_lo,
        fine_sigma_hi,
        81,
        81,
    )?;

    Some((fine.0, fine.1))
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
            let market_price = normalized_atm_black_price(*market_vol, *expiry);
            let model_price = normalized_atm_black_price(model_vol, *expiry);
            let err = model_price - market_price;
            err * err
        })
        .sum()
}

fn normalized_atm_black_price(vol: f64, expiry: f64) -> f64 {
    if !vol.is_finite() || vol < 0.0 || !expiry.is_finite() || expiry <= 0.0 {
        return f64::NAN;
    }
    if vol == 0.0 {
        return 0.0;
    }

    let x = 0.5 * vol * expiry.sqrt();
    2.0 * normal_cdf(x) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_recovers_synthetic_hull_white_parameters() {
        let true_a = 0.05;
        let true_sigma = 0.01;

        let mut market_quotes = Vec::new();
        for expiry in [1.0, 2.0, 3.0, 5.0] {
            for tenor in [2.0, 5.0] {
                let vol = hw_atm_swaption_vol_approx(true_a, true_sigma, expiry, tenor);
                market_quotes.push((expiry, tenor, vol));
            }
        }

        let (cal_a, cal_sigma) = calibrate_hull_white_params(&market_quotes).unwrap();
        let rel_a = (cal_a - true_a).abs() / true_a;
        let rel_sigma = (cal_sigma - true_sigma).abs() / true_sigma;

        assert!(rel_a <= 0.10, "a calibration error too high: {}", rel_a);
        assert!(
            rel_sigma <= 0.10,
            "sigma calibration error too high: {}",
            rel_sigma
        );
    }
}
