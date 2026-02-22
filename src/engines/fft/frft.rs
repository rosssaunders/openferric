//! Module `engines::fft::frft`.
//!
//! Implements frft workflows with concrete routines such as `frft`, `carr_madan_frft_grid`, `carr_madan_price_at_strikes`, `carr_madan_price_at_strikes_with_samples`.
//!
//! References: Carr and Madan (1999), Lewis (2001), Hull (11th ed.) Ch. 19, with FFT damping/inversion forms around Eq. (19.8).
//!
//! Primary API surface: free functions `frft`, `carr_madan_frft_grid`, `carr_madan_price_at_strikes`, `carr_madan_price_at_strikes_with_samples`.
//!
//! Numerical considerations: choose damping/aliasing controls (alpha, grid spacing, FFT size) to balance truncation error against oscillation near strikes.
//!
//! When to use: choose FFT-based routines for dense strike grids under characteristic-function models; use direct quadrature or Monte Carlo for sparse-strike or path-dependent products.
use std::f64::consts::PI;

use num_complex::Complex;

use super::carr_madan::{CarrMadanParams, build_weighted_frequency_samples};
use super::char_fn::CharacteristicFunction;
use super::fft_core::{fft_forward, fft_inverse};

/// Fractional FFT using a chirp-z style convolution.
///
/// Computes:
/// `y[m] = sum_{n=0}^{N-1} x[n] * exp(-i * 2π * beta * n * m / N)`
/// for `m = 0..N-1`.
pub fn frft(input: &[Complex<f64>], beta: f64) -> Vec<Complex<f64>> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let conv_len = (2 * n).next_power_of_two();
    let mut a = vec![Complex::new(0.0, 0.0); conv_len];
    let mut b = vec![Complex::new(0.0, 0.0); conv_len];

    for idx in 0..n {
        let idx_f = idx as f64;
        let chirp_angle = PI * beta * idx_f * idx_f / n as f64;
        let chirp_pos = Complex::new(0.0, chirp_angle).exp();
        let chirp_neg = Complex::new(0.0, -chirp_angle).exp();

        a[idx] = input[idx] * chirp_neg;
        b[idx] = chirp_pos;
        if idx != 0 {
            b[conv_len - idx] = chirp_pos;
        }
    }

    fft_forward(&mut a);
    fft_forward(&mut b);

    for idx in 0..conv_len {
        a[idx] *= b[idx];
    }

    fft_inverse(&mut a);
    let mut out = Vec::with_capacity(n);
    for (m, a_m) in a.iter().enumerate().take(n) {
        let m_f = m as f64;
        let chirp_angle = PI * beta * m_f * m_f / n as f64;
        let chirp_neg = Complex::new(0.0, -chirp_angle).exp();
        out.push(a_m * chirp_neg);
    }

    out
}

/// Carr-Madan with FRFT on a uniform log-strike grid.
pub fn carr_madan_frft_grid<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    log_strike_start: f64,
    log_strike_spacing: f64,
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    let weighted_samples = build_weighted_frequency_samples(cf, rate, maturity, params);
    carr_madan_frft_grid_from_weighted_samples(
        &weighted_samples,
        log_strike_start,
        log_strike_spacing,
        params,
    )
}

fn carr_madan_frft_grid_from_weighted_samples(
    weighted_samples: &[Complex<f64>],
    log_strike_start: f64,
    log_strike_spacing: f64,
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    if params.n < 2 || !params.n.is_power_of_two() {
        return Err("carr-madan frft requires n to be a power of two >= 2".to_string());
    }
    if !params.eta.is_finite() || params.eta <= 0.0 {
        return Err("carr-madan frft requires eta > 0".to_string());
    }
    if !params.alpha.is_finite() || params.alpha <= 0.0 {
        return Err("carr-madan frft requires alpha > 0".to_string());
    }
    if !log_strike_start.is_finite() || !log_strike_spacing.is_finite() || log_strike_spacing <= 0.0
    {
        return Err(
            "carr-madan frft requires finite log_strike_start and log_strike_spacing > 0"
                .to_string(),
        );
    }
    if weighted_samples.len() != params.n {
        return Err("weighted sample length must equal params.n".to_string());
    }

    let i = Complex::new(0.0, 1.0);
    let mut input = weighted_samples.to_vec();

    for (j, xj) in input.iter_mut().enumerate() {
        let vj = j as f64 * params.eta;
        let phase = (-i * vj * log_strike_start).exp();
        *xj *= phase;
    }

    let beta = params.eta * log_strike_spacing * params.n as f64 / (2.0 * PI);
    let transformed = frft(&input, beta);

    let mut out = Vec::with_capacity(params.n);
    for (m, z) in transformed.into_iter().enumerate() {
        let k = log_strike_start + m as f64 * log_strike_spacing;
        let strike = k.exp();
        let call = ((-params.alpha * k).exp() * z.re / PI).max(0.0);
        out.push((strike, call));
    }

    Ok(out)
}

fn direct_carr_madan_at_log_strike(
    weighted_samples: &[Complex<f64>],
    eta: f64,
    alpha: f64,
    log_strike: f64,
) -> f64 {
    let i = Complex::new(0.0, 1.0);
    let mut sum = Complex::new(0.0, 0.0);

    for (j, sample) in weighted_samples.iter().enumerate() {
        let vj = j as f64 * eta;
        sum += *sample * (-i * vj * log_strike).exp();
    }

    ((-alpha * log_strike).exp() * sum.re / PI).max(0.0)
}

fn has_uniform_spacing(values: &[f64], tol: f64) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }

    let spacing = values[1] - values[0];
    if spacing <= 0.0 {
        return None;
    }

    let scale = 1.0 + spacing.abs();
    for pair in values.windows(2) {
        if ((pair[1] - pair[0]) - spacing).abs() > tol * scale {
            return None;
        }
    }

    Some(spacing)
}

/// Prices at exact user strikes.
///
/// - Uses FRFT when strikes are uniformly spaced in log-strike.
/// - Falls back to direct Carr-Madan summation otherwise.
pub fn carr_madan_price_at_strikes<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    _spot: f64,
    strikes: &[f64],
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    let weighted_samples = build_weighted_frequency_samples(cf, rate, maturity, params);
    carr_madan_price_at_strikes_with_samples(&weighted_samples, strikes, params)
}

/// Prices at exact user strikes using pre-computed weighted frequency samples.
pub fn carr_madan_price_at_strikes_with_samples(
    weighted_samples: &[Complex<f64>],
    strikes: &[f64],
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    if strikes.is_empty() {
        return Ok(Vec::new());
    }
    if params.n < 2 || !params.n.is_power_of_two() {
        return Err("carr-madan strike pricing requires n to be a power of two >= 2".to_string());
    }
    if weighted_samples.len() != params.n {
        return Err("weighted sample length must equal params.n".to_string());
    }

    let mut indexed = Vec::with_capacity(strikes.len());
    for (idx, strike) in strikes.iter().copied().enumerate() {
        if !strike.is_finite() || strike <= 0.0 {
            return Err("all strikes must be finite and > 0".to_string());
        }
        indexed.push((idx, strike, strike.ln()));
    }

    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let log_strikes_sorted: Vec<f64> = indexed.iter().map(|x| x.2).collect();
    let maybe_uniform = has_uniform_spacing(&log_strikes_sorted, 1e-10);

    let mut out = vec![(0.0, 0.0); strikes.len()];

    if let Some(dk) = maybe_uniform
        && indexed.len() <= params.n
    {
        let k0 = log_strikes_sorted[0];
        let frft_slice =
            carr_madan_frft_grid_from_weighted_samples(weighted_samples, k0, dk, params)?;

        for (orig_idx, strike, log_k) in indexed {
            let m = ((log_k - k0) / dk).round() as usize;
            let price = frft_slice[m].1;
            out[orig_idx] = (strike, price);
        }

        return Ok(out);
    }

    for (orig_idx, strike, log_k) in indexed {
        let price =
            direct_carr_madan_at_log_strike(weighted_samples, params.eta, params.alpha, log_k);
        out[orig_idx] = (strike, price);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::fft::char_fn::BlackScholesCharFn;

    #[test]
    fn frft_reduces_to_dft_for_beta_one() {
        let x = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(-0.5, 0.25),
            Complex::new(0.3, 1.2),
        ];
        let y = frft(&x, 1.0);

        let n = x.len() as f64;
        for (m, ym) in y.iter().enumerate() {
            let mut direct = Complex::new(0.0, 0.0);
            for (k, xk) in x.iter().enumerate() {
                let angle = -2.0 * PI * (k * m) as f64 / n;
                direct += *xk * Complex::new(0.0, angle).exp();
            }
            assert!((*ym - direct).norm() < 1e-8);
        }
    }

    #[test]
    fn exact_strike_path_returns_requested_order() {
        let cf = BlackScholesCharFn::new(100.0, 0.02, 0.0, 0.2, 1.0);
        let strikes = vec![120.0, 90.0, 100.0];
        let px = carr_madan_price_at_strikes(
            &cf,
            0.02,
            1.0,
            100.0,
            &strikes,
            CarrMadanParams::default(),
        )
        .unwrap();
        assert_eq!(px[0].0, 120.0);
        assert_eq!(px[1].0, 90.0);
        assert_eq!(px[2].0, 100.0);
    }
}
