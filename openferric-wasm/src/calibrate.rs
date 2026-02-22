//! Deribit-specific slice calibration from raw market quotes.
//!
//! This module contains `calibrate_slice` which performs quality filtering,
//! log-moneyness computation, ATM vol search, model calibration, and fit
//! diagnostics in a single call. The filtering thresholds are tuned for
//! Deribit market data and do not belong in the core library.

use openferric::vol::slice::{MODEL_SABR, MODEL_SVI, MODEL_VV, slice_fit_diagnostics};
use openferric::vol::surface::SviParams;

/// Calibrate a vol slice from raw market quotes.
///
/// Performs all filtering, log-moneyness computation, ATM vol search, model
/// calibration (SVI/SABR/VV), and fit diagnostics in a single call.
///
/// # Arguments
/// - `strikes`, `mark_ivs`, `bid_ivs`, `ask_ivs`, `open_interests` — parallel arrays
/// - `forward` — forward price for this expiry
/// - `t` — time to expiry in year fractions
/// - `model_type` — 0=SVI, 1=SABR, 2=VV
///
/// # Returns packed flat array
/// ```text
/// [n_points, model_type, T, forward,
///  param_0..param_N,                    // SVI:5, SABR:4, VV:3
///  rmse, atm_vol_pct, skew, kurt_proxy,
///  // per point (6 values each):
///  strike, market_iv_pct, fitted_iv_pct, bid_iv_pct, ask_iv_pct, k]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn calibrate_slice(
    strikes: &[f64],
    mark_ivs: &[f64],
    bid_ivs: &[f64],
    ask_ivs: &[f64],
    open_interests: &[f64],
    forward: f64,
    t: f64,
    model_type: u8,
) -> Vec<f64> {
    let n = strikes
        .len()
        .min(mark_ivs.len())
        .min(bid_ivs.len())
        .min(ask_ivs.len())
        .min(open_interests.len());

    if n == 0 || forward <= 0.0 || t <= 0.0 {
        return vec![0.0]; // n_points = 0
    }

    // Step 1: compute log-moneyness and build candidate list
    struct Quote {
        k: f64,
        strike: f64,
        mark_iv: f64,
        bid_iv: f64,
        ask_iv: f64,
        open_interest: f64,
    }

    let mut all_quotes: Vec<Quote> = Vec::with_capacity(n);
    for i in 0..n {
        if strikes[i] <= 0.0 || !mark_ivs[i].is_finite() || mark_ivs[i] <= 0.0 {
            continue;
        }
        all_quotes.push(Quote {
            k: (strikes[i] / forward).ln(),
            strike: strikes[i],
            mark_iv: mark_ivs[i],
            bid_iv: bid_ivs[i],
            ask_iv: ask_ivs[i],
            open_interest: open_interests[i],
        });
    }

    // Step 2: quality filter
    let mut filtered: Vec<usize> = Vec::with_capacity(all_quotes.len());
    for (i, q) in all_quotes.iter().enumerate() {
        if q.mark_iv <= 0.0 {
            continue;
        }
        if q.bid_iv <= 0.0 && q.open_interest <= 0.0 {
            continue;
        }
        if q.bid_iv > 0.0 && q.ask_iv > 0.0 {
            let spread = q.ask_iv - q.bid_iv;
            let mid = (q.ask_iv + q.bid_iv) * 0.5;
            if mid > 0.0 && spread / mid > 0.5 {
                continue;
            }
        }
        // OTM thresholds for short dates
        if t < 7.0 / 365.0 && q.k.abs() > 0.15 {
            continue;
        }
        if t < 30.0 / 365.0 && q.k.abs() > 0.5 {
            continue;
        }
        filtered.push(i);
    }

    // Fallback: if <3 pass filter, use all with mark_iv > 0
    if filtered.len() < 3 {
        filtered.clear();
        for (i, q) in all_quotes.iter().enumerate() {
            if q.mark_iv > 0.0 {
                filtered.push(i);
            }
        }
    }

    if filtered.is_empty() {
        return vec![0.0];
    }

    // Step 3: find ATM vol (min |k|)
    let mut atm_vol = 0.0_f64;
    let mut min_abs_k = f64::INFINITY;
    for &idx in &filtered {
        let q = &all_quotes[idx];
        if q.k.abs() < min_abs_k {
            min_abs_k = q.k.abs();
            atm_vol = q.mark_iv;
        }
    }

    // Step 4: calibrate model
    let n_params = match model_type {
        MODEL_SVI => 5,
        MODEL_SABR => 4,
        MODEL_VV => 3,
        _ => return vec![0.0],
    };

    let params: Vec<f64> = match model_type {
        MODEL_SVI => {
            // Build OI-weighted (k, iv^2) points
            let max_oi = filtered
                .iter()
                .map(|&i| all_quotes[i].open_interest)
                .fold(1.0_f64, f64::max);
            let mut points: Vec<(f64, f64)> = Vec::new();
            for &idx in &filtered {
                let q = &all_quotes[idx];
                let iv2 = q.mark_iv * q.mark_iv;
                let oi = q.open_interest.max(0.0);
                let reps = 1 + ((oi / max_oi) * 4.0).floor() as usize;
                for _ in 0..reps {
                    points.push((q.k, iv2));
                }
            }
            let atm_iv2 = (atm_vol * atm_vol).max(1e-4);
            let init = SviParams {
                a: atm_iv2 * 0.5,
                b: atm_iv2 * 1.5,
                rho: -0.1,
                m: 0.0,
                sigma: 0.15,
            };
            let result = openferric::vol::surface::calibrate_svi(&points, init, 3000, 0.002);
            // Scale a and b by T (SVI parameterizes total variance)
            vec![
                result.a * t,
                result.b * t,
                result.rho,
                result.m,
                result.sigma,
            ]
        }
        MODEL_SABR => {
            let cal_strikes: Vec<f64> = filtered.iter().map(|&i| all_quotes[i].strike).collect();
            let cal_vols: Vec<f64> = filtered.iter().map(|&i| all_quotes[i].mark_iv).collect();
            let result = openferric::vol::sabr::fit_sabr(forward, &cal_strikes, &cal_vols, t, 0.5);
            vec![result.alpha, result.beta, result.rho, result.nu]
        }
        MODEL_VV => {
            // Sort filtered by k, pick 15th/85th percentile
            let mut sorted: Vec<usize> = filtered.clone();
            sorted.sort_by(|&a, &b| {
                all_quotes[a]
                    .k
                    .partial_cmp(&all_quotes[b].k)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sn = sorted.len();
            let put_idx = ((sn as f64 * 0.15).round() as usize).min(sn - 1);
            let call_idx = ((sn as f64 * 0.85).round() as usize).min(sn - 1);
            let put_vol = all_quotes[sorted[put_idx]].mark_iv;
            let call_vol = all_quotes[sorted[call_idx]].mark_iv;
            let rr25 = call_vol - put_vol;
            let bf25 = 0.5 * (call_vol + put_vol) - atm_vol;
            vec![atm_vol, rr25, bf25]
        }
        _ => unreachable!(),
    };

    // Step 5: fit diagnostics
    let market_ks: Vec<f64> = filtered.iter().map(|&i| all_quotes[i].k).collect();
    let market_ivs_pct: Vec<f64> = filtered
        .iter()
        .map(|&i| all_quotes[i].mark_iv * 100.0)
        .collect();
    let diag_strikes: Vec<f64> = filtered.iter().map(|&i| all_quotes[i].strike).collect();

    let diag = slice_fit_diagnostics(
        model_type,
        &params,
        t,
        forward,
        &market_ks,
        &market_ivs_pct,
        &diag_strikes,
    );

    let rmse = diag[0];
    let skew = diag[1];
    let kurt_proxy = diag[2];

    // Step 6: pack result
    let n_filtered = filtered.len();
    let total_len = 4 + n_params + 4 + n_filtered * 6;
    let mut out = Vec::with_capacity(total_len);

    // Header
    out.push(n_filtered as f64);
    out.push(model_type as f64);
    out.push(t);
    out.push(forward);

    // Params
    out.extend_from_slice(&params);

    // Diagnostics
    out.push(rmse);
    out.push(atm_vol * 100.0);
    out.push(skew);
    out.push(kurt_proxy);

    // Per point (sorted by strike)
    let mut point_indices: Vec<usize> = filtered.clone();
    point_indices.sort_by(|&a, &b| {
        all_quotes[a]
            .strike
            .partial_cmp(&all_quotes[b].strike)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (j, &idx) in point_indices.iter().enumerate() {
        let q = &all_quotes[idx];
        let fitted_iv = if j < diag.len() - 3 {
            let filtered_pos = filtered.iter().position(|&fi| fi == idx).unwrap_or(0);
            diag[3 + filtered_pos]
        } else {
            f64::NAN
        };
        out.push(q.strike);
        out.push(q.mark_iv * 100.0);
        out.push(fitted_iv);
        out.push(q.bid_iv * 100.0);
        out.push(q.ask_iv * 100.0);
        out.push(q.k);
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_slice_svi() {
        let forward = 100.0;
        let t = 0.25;
        let strikes: Vec<f64> = (80..=120).step_by(2).map(|s| s as f64).collect();
        let mark_ivs: Vec<f64> = strikes
            .iter()
            .map(|&s| {
                let k = (s / forward).ln();
                0.20 + 0.1 * k * k
            })
            .collect();
        let bid_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 0.95).collect();
        let ask_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 1.05).collect();
        let ois: Vec<f64> = vec![100.0; strikes.len()];

        let result = calibrate_slice(
            &strikes, &mark_ivs, &bid_ivs, &ask_ivs, &ois, forward, t, MODEL_SVI,
        );

        assert!(result.len() > 4);
        let n_pts = result[0] as usize;
        assert!(n_pts > 0);
        assert_eq!(result[1] as u8, MODEL_SVI);
        assert!((result[2] - t).abs() < 1e-12);
        assert!((result[3] - forward).abs() < 1e-12);

        let rmse = result[9];
        let atm_vol_pct = result[10];
        assert!(rmse.is_finite() && rmse >= 0.0);
        assert!(atm_vol_pct > 0.0);

        assert_eq!(result.len(), 4 + 5 + 4 + n_pts * 6);
    }

    #[test]
    fn test_calibrate_slice_sabr() {
        let forward = 100.0;
        let t = 0.5;
        let strikes: Vec<f64> = (85..=115).step_by(3).map(|s| s as f64).collect();
        let mark_ivs: Vec<f64> = strikes.iter().map(|_| 0.25).collect();
        let bid_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 0.95).collect();
        let ask_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 1.05).collect();
        let ois: Vec<f64> = vec![50.0; strikes.len()];

        let result = calibrate_slice(
            &strikes, &mark_ivs, &bid_ivs, &ask_ivs, &ois, forward, t, MODEL_SABR,
        );
        let n_pts = result[0] as usize;
        assert!(n_pts > 0);
        assert_eq!(result[1] as u8, MODEL_SABR);
        assert_eq!(result.len(), 4 + 4 + 4 + n_pts * 6);
    }

    #[test]
    fn test_calibrate_slice_vv() {
        let forward = 100.0;
        let t = 0.25;
        let strikes: Vec<f64> = (80..=120).step_by(2).map(|s| s as f64).collect();
        let mark_ivs: Vec<f64> = strikes
            .iter()
            .map(|&s| {
                let k = (s / forward).ln();
                0.20 + 0.1 * k * k
            })
            .collect();
        let bid_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 0.95).collect();
        let ask_ivs: Vec<f64> = mark_ivs.iter().map(|&iv| iv * 1.05).collect();
        let ois: Vec<f64> = vec![100.0; strikes.len()];

        let result = calibrate_slice(
            &strikes, &mark_ivs, &bid_ivs, &ask_ivs, &ois, forward, t, MODEL_VV,
        );
        let n_pts = result[0] as usize;
        assert!(n_pts > 0);
        assert_eq!(result[1] as u8, MODEL_VV);
        assert_eq!(result.len(), 4 + 3 + 4 + n_pts * 6);
    }

    #[test]
    fn test_calibrate_slice_empty() {
        let result = calibrate_slice(&[], &[], &[], &[], &[], 100.0, 0.25, MODEL_SVI);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }
}
