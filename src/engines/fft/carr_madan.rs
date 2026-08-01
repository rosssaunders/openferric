//! Module `engines::fft::carr_madan`.
//!
//! Implements carr madan workflows with concrete routines such as `carr_madan_fft`, `carr_madan_fft_complex`, `interpolate_strike_prices`, `carr_madan_fft_strikes`.
//!
//! References: Carr and Madan (1999), Lewis (2001), Hull (11th ed.) Ch. 19, with FFT damping/inversion forms around Eq. (19.8).
//!
//! Key types and purpose: `CarrMadanParams`, `CarrMadanGreeksPoint`, `CarrMadanContext` define the core data contracts for this module.
//!
//! Numerical considerations: choose damping/aliasing controls (alpha, grid spacing, FFT size) to balance truncation error against oscillation near strikes.
//!
//! When to use: choose FFT-based routines for dense strike grids under characteristic-function models; use direct quadrature or Monte Carlo for sparse-strike or path-dependent products.
use std::f64::consts::PI;

use num_complex::Complex;

use super::char_fn::{CharacteristicFunction, HestonCharFn};
use super::fft_core::{fft_forward as fft_forward_inplace, fft_forward_real};
use super::frft::carr_madan_price_at_strikes_with_samples;

/// Default Carr-Madan grid size.
pub const DEFAULT_FFT_N: usize = 4096;
/// Optional high-resolution Carr-Madan grid size.
pub const HIGH_RES_FFT_N: usize = 16384;
/// Default Carr-Madan dampening parameter.
pub const DEFAULT_ALPHA: f64 = 1.5;
/// Default Carr-Madan frequency spacing.
pub const DEFAULT_ETA: f64 = 0.25;
const REAL_FFT_EPS: f64 = 1e-14;
/// Smallest damping alpha the automatic admissibility fallback will accept.
const MIN_ADMISSIBLE_ALPHA: f64 = 0.1;
/// Relative safety margin on the damping moment when auto-selecting a
/// fallback alpha. A moment that exists only marginally (e.g. VG with the
/// quadratic denominator barely positive) makes `E[S_T^(alpha+1)]` blow up
/// and the trapezoid quadrature cannot resolve the resulting integrand peak,
/// so fallback alphas must sit safely inside the admissible region.
const FALLBACK_MOMENT_MARGIN: f64 = 1.05;
/// Upper bound on the FFT size the admissibility fallback may grow to when it
/// refines the frequency grid alongside a reduced alpha (2^18 complex samples
/// = 4 MiB per buffer).
const MAX_FALLBACK_FFT_N: usize = 1 << 18;

/// Carr-Madan FFT configuration.
#[derive(Debug, Clone, Copy)]
pub struct CarrMadanParams {
    /// Number of frequency points (must be power-of-two).
    pub n: usize,
    /// Frequency grid spacing.
    pub eta: f64,
    /// Dampening parameter.
    pub alpha: f64,
}

impl Default for CarrMadanParams {
    fn default() -> Self {
        Self {
            n: DEFAULT_FFT_N,
            eta: DEFAULT_ETA,
            alpha: DEFAULT_ALPHA,
        }
    }
}

impl CarrMadanParams {
    /// Higher-resolution preset for surface generation.
    pub fn high_resolution() -> Self {
        Self {
            n: HIGH_RES_FFT_N,
            eta: DEFAULT_ETA,
            alpha: DEFAULT_ALPHA,
        }
    }

    fn validate(self) -> Result<(), String> {
        if self.n < 2 || !self.n.is_power_of_two() {
            return Err("carr-madan requires n to be a power of two >= 2".to_string());
        }
        if !self.eta.is_finite() || self.eta <= 0.0 {
            return Err("carr-madan requires eta > 0".to_string());
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err("carr-madan requires alpha > 0".to_string());
        }
        Ok(())
    }

    /// Log-strike spacing implied by this FFT setup.
    pub fn lambda(self) -> f64 {
        2.0 * PI / (self.n as f64 * self.eta)
    }
}

/// Single strike record with FFT Greeks.
#[derive(Debug, Clone, Copy)]
pub struct CarrMadanGreeksPoint {
    pub strike: f64,
    pub call: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

/// Reusable Carr-Madan workspace for repeated pricing under fixed model/market parameters.
///
/// The context caches weighted frequency-domain samples, which avoids repeated
/// characteristic-function evaluations when pricing multiple strike sets.
#[derive(Debug, Clone)]
pub struct CarrMadanContext {
    rate: f64,
    maturity: f64,
    spot: f64,
    params: CarrMadanParams,
    weighted_samples: Vec<Complex<f64>>,
}

impl CarrMadanContext {
    /// Builds a reusable pricing context for a fixed model/market configuration.
    ///
    /// The requested damping `params.alpha` is validated against the model's
    /// exponential moments: if `E[S_T^(alpha+1)]` is not finite the context
    /// automatically falls back to the largest admissible smaller alpha and
    /// errors when none exists. The effective parameters are reported by
    /// [`CarrMadanContext::params`].
    pub fn new<C: CharacteristicFunction>(
        cf: &C,
        rate: f64,
        maturity: f64,
        spot: f64,
        params: CarrMadanParams,
    ) -> Result<Self, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("carr-madan context requires spot > 0".to_string());
        }
        if !maturity.is_finite() || maturity < 0.0 {
            return Err("carr-madan context requires maturity >= 0".to_string());
        }

        let params = resolve_admissible_params(cf, params)?;
        let weighted_samples = build_weighted_frequency_samples(cf, rate, maturity, params)?;
        Ok(Self {
            rate,
            maturity,
            spot,
            params,
            weighted_samples,
        })
    }

    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    #[inline]
    pub fn maturity(&self) -> f64 {
        self.maturity
    }

    #[inline]
    pub fn spot(&self) -> f64 {
        self.spot
    }

    #[inline]
    pub fn params(&self) -> CarrMadanParams {
        self.params
    }

    #[inline]
    pub fn weighted_samples(&self) -> &[Complex<f64>] {
        &self.weighted_samples
    }

    /// Prices a user strike list via FRFT/direct fallback using cached weighted samples.
    pub fn price_strikes(&self, strikes: &[f64]) -> Result<Vec<(f64, f64)>, String> {
        carr_madan_price_at_strikes_with_samples(&self.weighted_samples, strikes, self.params)
    }

    /// Returns the full Carr-Madan strike grid using real-FFT optimization when possible.
    pub fn price_grid(&self) -> Result<Vec<(f64, f64)>, String> {
        carr_madan_fft_from_weighted_samples(&self.weighted_samples, self.spot, self.params, true)
    }

    /// Returns the full Carr-Madan strike grid using complex FFT only.
    pub fn price_grid_complex(&self) -> Result<Vec<(f64, f64)>, String> {
        carr_madan_fft_from_weighted_samples(&self.weighted_samples, self.spot, self.params, false)
    }
}

#[inline]
fn modified_cf<C: CharacteristicFunction>(
    cf: &C,
    u: Complex<f64>,
    rate: f64,
    maturity: f64,
    alpha: f64,
) -> Complex<f64> {
    let discount = (-rate * maturity).exp();
    let v = u.re;
    let denom = Complex::new(alpha * alpha + alpha - v * v, (2.0 * alpha + 1.0) * v);
    discount * cf.cf(u) / denom
}

#[inline]
fn quadrature_weight(index: usize, eta: f64) -> f64 {
    if index == 0 { 0.5 * eta } else { eta }
}

/// Re-sync interval for accumulated phase rotation to bound floating-point drift.
const PHASE_RESYNC_INTERVAL: usize = 128;

/// Resolves damping-admissible Carr-Madan parameters for the given model.
///
/// Carr-Madan damping with `alpha` requires `E[S_T^(alpha+1)]` to be finite
/// (Carr & Madan 1999); an inadmissible alpha makes the damped integrand
/// non-integrable and the transform returns garbage (zero or below-intrinsic
/// calls) rather than failing. When the requested alpha is inadmissible this
/// halves alpha until an admissible value is found (floor
/// `MIN_ADMISSIBLE_ALPHA`) and errors out when none exists.
///
/// A reduced alpha sharpens the damped integrand's peak near `v = 0` (the
/// denominator `alpha^2 + alpha - v^2 + i(2alpha+1)v` has magnitude
/// `~alpha(alpha+1)` at the origin, so the peak width scales with alpha).
/// The requested eta was chosen for the requested alpha and under-resolves
/// that sharper peak — with the default eta = 0.25 an alpha of 0.1875 leaves
/// a strike-independent quadrature error of ~0.9 currency units (a 10x error
/// on OTM puts recovered via parity). Fallback params therefore refine eta by
/// the same factor alpha was reduced by and grow n to preserve the frequency
/// range, capped at [`MAX_FALLBACK_FFT_N`].
pub(crate) fn resolve_admissible_params<C: CharacteristicFunction>(
    cf: &C,
    params: CarrMadanParams,
) -> Result<CarrMadanParams, String> {
    params.validate()?;
    if cf.moment_exists(params.alpha + 1.0) {
        return Ok(params);
    }

    // Auto-selected alphas demand a small moment margin so the fallback never
    // lands razor-thin against the model's moment-explosion boundary.
    let fallback_admissible = |alpha: f64| cf.moment_exists((alpha + 1.0) * FALLBACK_MOMENT_MARGIN);

    // Refines the frequency grid for a fallback alpha: eta shrinks by the
    // alpha-reduction factor and n grows to keep the integration range
    // `n * eta` unchanged. n stays a power of two; if the cap binds, eta is
    // only refined by the granted factor so the range never shrinks.
    let refine_grid = |alpha: f64| -> CarrMadanParams {
        let factor = (params.alpha / alpha).ceil() as usize;
        let factor = factor.next_power_of_two().max(1);
        let n = params
            .n
            .saturating_mul(factor)
            .min(MAX_FALLBACK_FFT_N)
            .max(params.n);
        let granted = (n / params.n).max(1);
        CarrMadanParams {
            n,
            eta: params.eta / granted as f64,
            alpha,
        }
    };

    let mut alpha = 0.5 * params.alpha;
    while alpha > MIN_ADMISSIBLE_ALPHA {
        if fallback_admissible(alpha) {
            return Ok(refine_grid(alpha));
        }
        alpha *= 0.5;
    }
    if params.alpha > MIN_ADMISSIBLE_ALPHA && fallback_admissible(MIN_ADMISSIBLE_ALPHA) {
        return Ok(refine_grid(MIN_ADMISSIBLE_ALPHA));
    }

    Err(format!(
        "carr-madan damping alpha={} is inadmissible: E[S_T^{}] is not finite for this model \
         and no fallback alpha down to {} is admissible; the model's exponential moments \
         explode, so FFT call prices would be meaningless",
        params.alpha,
        params.alpha + 1.0,
        MIN_ADMISSIBLE_ALPHA
    ))
}

pub(crate) fn build_weighted_frequency_samples<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    params: CarrMadanParams,
) -> Result<Vec<Complex<f64>>, String> {
    let mut out = vec![Complex::new(0.0, 0.0); params.n];
    for (j, out_j) in out.iter_mut().enumerate() {
        let vj = j as f64 * params.eta;
        let uj = Complex::new(vj, -(params.alpha + 1.0));
        let psi = modified_cf(cf, uj, rate, maturity, params.alpha);
        if !psi.is_finite() {
            return Err(format!(
                "carr-madan integrand is non-finite at frequency v={vj} (alpha={}): the \
                 characteristic function overflowed or is undefined on the damping contour; \
                 check model parameters",
                params.alpha
            ));
        }
        *out_j = psi * quadrature_weight(j, params.eta);
    }
    Ok(out)
}

#[inline]
fn apply_phase_in_place(input: &mut [Complex<f64>], eta: f64, k0: f64) {
    // Pre-compute the per-step phase rotation instead of calling exp() each iteration.
    // phase(j) = exp(-i * j * eta * k0), accumulated via multiplication with periodic re-sync.
    let angle_per_step = -eta * k0;
    let phase_step = Complex::new(angle_per_step.cos(), angle_per_step.sin());
    let mut phase = Complex::new(1.0, 0.0); // j=0: angle is 0

    for (j, out_j) in input.iter_mut().enumerate() {
        // Re-sync every PHASE_RESYNC_INTERVAL steps to avoid accumulated phase drift.
        if j > 0 && j % PHASE_RESYNC_INTERVAL == 0 {
            let angle = -(j as f64) * eta * k0;
            phase = Complex::new(angle.cos(), angle.sin());
        }

        *out_j *= phase;
        phase *= phase_step;
    }
}

#[inline]
fn can_use_real_fft(input: &[Complex<f64>]) -> bool {
    input
        .iter()
        .all(|z| z.im.abs() <= REAL_FFT_EPS * (1.0 + z.re.abs()))
}

#[inline]
fn transform_fft_input(mut input: Vec<Complex<f64>>, allow_real_fft: bool) -> Vec<Complex<f64>> {
    if allow_real_fft && can_use_real_fft(&input) {
        let real_input: Vec<f64> = input.iter().map(|z| z.re).collect();
        if let Ok(real_spectrum) = fft_forward_real(&real_input) {
            return real_spectrum;
        }
    }

    fft_forward_inplace(&mut input);
    input
}

fn decode_fft_output(
    transformed: Vec<Complex<f64>>,
    params: CarrMadanParams,
    k0: f64,
) -> Result<Vec<(f64, f64)>, String> {
    let lambda = params.lambda();
    // Pre-compute the exponential decay exp(-alpha * k) via accumulation.
    // k = k0 + m * lambda, so exp(-alpha * k) = exp(-alpha * k0) * exp(-alpha * lambda)^m.
    let alpha_lambda = -params.alpha * lambda;
    let exp_step = alpha_lambda.exp();
    let mut exp_alpha_k = (-params.alpha * k0).exp();
    let inv_pi = 1.0 / PI;

    let mut out = Vec::with_capacity(params.n);
    for (m, z) in transformed.into_iter().enumerate() {
        let strike = (k0 + m as f64 * lambda).exp();
        let call = exp_alpha_k * z.re * inv_pi;
        if !call.is_finite() {
            return Err(format!(
                "carr-madan FFT produced a non-finite call price at strike {strike}: \
                 characteristic function values overflowed on the damping contour"
            ));
        }
        // Only small negative truncation noise in the far wings is clamped;
        // non-finite values are propagated as errors above instead of being
        // masked to zero.
        out.push((strike, call.max(0.0)));
        exp_alpha_k *= exp_step;
    }
    Ok(out)
}

fn carr_madan_fft_from_weighted_samples(
    weighted_samples: &[Complex<f64>],
    spot: f64,
    params: CarrMadanParams,
    allow_real_fft: bool,
) -> Result<Vec<(f64, f64)>, String> {
    params.validate()?;
    if !spot.is_finite() || spot <= 0.0 {
        return Err("carr-madan requires spot > 0".to_string());
    }
    if weighted_samples.len() != params.n {
        return Err("weighted sample length must equal params.n".to_string());
    }

    let lambda = params.lambda();
    let b = 0.5 * params.n as f64 * lambda;
    let k0 = spot.ln() - b;

    let mut fft_input = weighted_samples.to_vec();
    apply_phase_in_place(&mut fft_input, params.eta, k0);
    let transformed = transform_fft_input(fft_input, allow_real_fft);
    decode_fft_output(transformed, params, k0)
}

fn carr_madan_fft_impl<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    spot: f64,
    params: CarrMadanParams,
    allow_real_fft: bool,
) -> Result<Vec<(f64, f64)>, String> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err("carr-madan requires spot > 0".to_string());
    }
    if !maturity.is_finite() || maturity < 0.0 {
        return Err("carr-madan requires maturity >= 0".to_string());
    }
    let params = resolve_admissible_params(cf, params)?;
    let weighted_samples = build_weighted_frequency_samples(cf, rate, maturity, params)?;
    carr_madan_fft_from_weighted_samples(&weighted_samples, spot, params, allow_real_fft)
}

/// Carr-Madan FFT pricing over a full strike slice in `O(N log N)`.
///
/// Returns `(strike, call_price)` pairs.
pub fn carr_madan_fft<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    spot: f64,
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    carr_madan_fft_impl(cf, rate, maturity, spot, params, true)
}

/// Carr-Madan FFT pricing using complex FFT only (benchmark/reference path).
pub fn carr_madan_fft_complex<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    spot: f64,
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    carr_madan_fft_impl(cf, rate, maturity, spot, params, false)
}

/// Interpolates call prices from a sorted strike slice.
pub fn interpolate_strike_prices(strike_slice: &[(f64, f64)], strikes: &[f64]) -> Vec<(f64, f64)> {
    if strike_slice.is_empty() {
        return strikes.iter().copied().map(|k| (k, 0.0)).collect();
    }

    let first = strike_slice[0];
    let last = strike_slice[strike_slice.len() - 1];

    strikes
        .iter()
        .copied()
        .map(|k| {
            if k <= first.0 {
                return (k, first.1);
            }
            if k >= last.0 {
                return (k, last.1);
            }

            let hi = strike_slice.partition_point(|(grid_k, _)| *grid_k < k);
            let lo = hi - 1;
            let (k0, p0) = strike_slice[lo];
            let (k1, p1) = strike_slice[hi];
            let x = k.ln();
            let x0 = k0.ln();
            let x1 = k1.ln();
            let w = (x - x0) / (x1 - x0);
            (k, p0 + w * (p1 - p0))
        })
        .collect()
}

/// Carr-Madan FFT pricing at user strikes via interpolation.
pub fn carr_madan_fft_strikes<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    spot: f64,
    strikes: &[f64],
    params: CarrMadanParams,
) -> Result<Vec<(f64, f64)>, String> {
    let full = carr_madan_fft(cf, rate, maturity, spot, params)?;
    Ok(interpolate_strike_prices(&full, strikes))
}

/// Carr-Madan FFT price and Greeks on the strike slice.
pub fn carr_madan_fft_greeks<C: CharacteristicFunction>(
    cf: &C,
    rate: f64,
    maturity: f64,
    spot: f64,
    params: CarrMadanParams,
) -> Result<Vec<CarrMadanGreeksPoint>, String> {
    if !spot.is_finite() || spot <= 0.0 {
        return Err("carr-madan greeks require spot > 0".to_string());
    }
    if !maturity.is_finite() || maturity < 0.0 {
        return Err("carr-madan greeks require maturity >= 0".to_string());
    }
    let params = resolve_admissible_params(cf, params)?;

    let lambda = params.lambda();
    let b = 0.5 * params.n as f64 * lambda;
    let k0 = spot.ln() - b;
    let alpha = params.alpha;

    let supports_d1 = cf
        .dcf_dlog_spot(Complex::new(0.0, -(alpha + 1.0)))
        .is_some();
    let supports_d2 = cf
        .d2cf_dlog_spot2(Complex::new(0.0, -(alpha + 1.0)))
        .is_some();
    let supports_dvol = cf.dcf_dvol(Complex::new(0.0, -(alpha + 1.0))).is_some();

    let mut x_price = vec![Complex::new(0.0, 0.0); params.n];
    let mut x_d1 = vec![Complex::new(0.0, 0.0); params.n];
    let mut x_d2 = vec![Complex::new(0.0, 0.0); params.n];
    let mut x_dvol = vec![Complex::new(0.0, 0.0); params.n];

    // Pre-compute discount factor once (was recomputed in each modified_cf_* call).
    let discount = (-rate * maturity).exp();
    let two_alpha_plus_one = 2.0 * alpha + 1.0;
    let alpha_sq_plus_alpha = alpha * alpha + alpha;

    // Pre-compute phase rotation for accumulation (same technique as build_fft_input).
    let angle_per_step = -params.eta * k0;
    let phase_step = Complex::new(angle_per_step.cos(), angle_per_step.sin());
    let mut phase = Complex::new(1.0, 0.0);

    for j in 0..params.n {
        // Re-sync phase periodically to bound floating-point drift.
        if j > 0 && j % PHASE_RESYNC_INTERVAL == 0 {
            let angle = -(j as f64) * params.eta * k0;
            phase = Complex::new(angle.cos(), angle.sin());
        }

        let vj = j as f64 * params.eta;
        let uj = Complex::new(vj, -(alpha + 1.0));
        let w = quadrature_weight(j, params.eta);

        // Compute shared discount / denom once for all Greeks at this frequency point.
        let v = vj; // uj.re
        let denom = Complex::new(alpha_sq_plus_alpha - v * v, two_alpha_plus_one * v);
        let common = phase * w * discount / denom;

        x_price[j] = common * cf.cf(uj);

        if supports_d1 {
            x_d1[j] = common
                * cf.dcf_dlog_spot(uj)
                    .expect("dcf_dlog_spot should be available");
        }

        if supports_d2 {
            x_d2[j] = common
                * cf.d2cf_dlog_spot2(uj)
                    .expect("d2cf_dlog_spot2 should be available");
        }

        if supports_dvol {
            x_dvol[j] = common * cf.dcf_dvol(uj).expect("dcf_dvol should be available");
        }

        if !x_price[j].is_finite()
            || !x_d1[j].is_finite()
            || !x_d2[j].is_finite()
            || !x_dvol[j].is_finite()
        {
            return Err(format!(
                "carr-madan greeks integrand is non-finite at frequency v={vj} (alpha={}): the \
                 characteristic function overflowed or is undefined on the damping contour; \
                 check model parameters",
                params.alpha
            ));
        }

        phase *= phase_step;
    }

    fft_forward_inplace(&mut x_price);
    if supports_d1 {
        fft_forward_inplace(&mut x_d1);
    }
    if supports_d2 {
        fft_forward_inplace(&mut x_d2);
    }
    if supports_dvol {
        fft_forward_inplace(&mut x_dvol);
    }

    // Pre-compute the exponential decay via accumulation (same technique as carr_madan_fft_impl).
    let alpha_lambda = -alpha * lambda;
    let exp_step = alpha_lambda.exp();
    let mut exp_alpha_k = (-alpha * k0).exp();
    let inv_pi = 1.0 / PI;

    let mut out = Vec::with_capacity(params.n);
    for m in 0..params.n {
        let strike = (k0 + m as f64 * lambda).exp();
        let pref = exp_alpha_k * inv_pi;

        let raw_call = pref * x_price[m].re;
        if !raw_call.is_finite() {
            return Err(format!(
                "carr-madan greeks FFT produced a non-finite call price at strike {strike}"
            ));
        }
        // Clamp only finite negative truncation noise; non-finite values error above.
        let call = raw_call.max(0.0);

        let dcdx = if supports_d1 {
            pref * x_d1[m].re
        } else {
            f64::NAN
        };
        let d2cdx2 = if supports_d2 {
            pref * x_d2[m].re
        } else {
            f64::NAN
        };

        let delta = if supports_d1 { dcdx / spot } else { f64::NAN };
        let gamma = if supports_d1 && supports_d2 {
            (d2cdx2 - dcdx) / (spot * spot)
        } else {
            f64::NAN
        };

        let vega = if supports_dvol {
            pref * x_dvol[m].re
        } else {
            f64::NAN
        };

        out.push(CarrMadanGreeksPoint {
            strike,
            call,
            delta,
            gamma,
            vega,
        });

        exp_alpha_k *= exp_step;
    }

    Ok(out)
}

/// Fallible Heston FFT API.
///
/// When the default damping alpha is inadmissible for the given parameters
/// (the `(alpha+1)`-th spot moment explodes), a smaller admissible alpha is
/// selected automatically; if none exists this returns an error instead of
/// silently producing zero or below-intrinsic call prices.
#[allow(clippy::too_many_arguments)]
pub fn try_heston_price_fft(
    spot: f64,
    strike_grid: &[f64],
    rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> Result<Vec<(f64, f64)>, String> {
    let cf = HestonCharFn::new(
        spot,
        rate,
        dividend_yield,
        maturity,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
    );

    // Build one reusable context and price exact strikes from cached frequency samples.
    let ctx = CarrMadanContext::new(&cf, rate, maturity, spot, CarrMadanParams::default())?;
    ctx.price_strikes(strike_grid)
}

/// Heston FFT convenience API.
///
/// Prices all strikes simultaneously. Invalid inputs — including parameter
/// sets whose damping moment explodes and admits no fallback alpha — return
/// an empty vector; use [`try_heston_price_fft`] for the error message.
#[allow(clippy::too_many_arguments)]
pub fn heston_price_fft(
    spot: f64,
    strike_grid: &[f64],
    rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    maturity: f64,
) -> Vec<(f64, f64)> {
    try_heston_price_fft(
        spot,
        strike_grid,
        rate,
        dividend_yield,
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        maturity,
    )
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::fft::char_fn::BlackScholesCharFn;
    use crate::engines::fft::frft::carr_madan_price_at_strikes;

    #[test]
    fn fft_slice_is_sorted_in_strike() {
        let cf = BlackScholesCharFn::new(100.0, 0.02, 0.0, 0.2, 1.0);
        let out = carr_madan_fft(&cf, 0.02, 1.0, 100.0, CarrMadanParams::default()).unwrap();
        assert!(!out.is_empty());
        assert!(out.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn interpolation_respects_endpoints() {
        let grid = vec![(90.0, 12.0), (100.0, 7.0), (110.0, 4.0)];
        let px = interpolate_strike_prices(&grid, &[80.0, 95.0, 120.0]);
        assert_eq!(px[0].1, 12.0);
        assert_eq!(px[2].1, 4.0);
        assert!(px[1].1 < 12.0 && px[1].1 > 7.0);
    }

    #[test]
    fn context_price_strikes_matches_direct_pricing() {
        let cf = BlackScholesCharFn::new(100.0, 0.02, 0.0, 0.2, 1.0);
        let params = CarrMadanParams::default();
        let strikes = [80.0, 95.0, 100.0, 110.0, 125.0];

        let direct = carr_madan_price_at_strikes(&cf, 0.02, 1.0, 100.0, &strikes, params)
            .expect("direct pricing succeeds");
        let ctx = CarrMadanContext::new(&cf, 0.02, 1.0, 100.0, params)
            .expect("context construction succeeds");
        let cached = ctx
            .price_strikes(&strikes)
            .expect("context pricing succeeds");

        for (lhs, rhs) in direct.iter().zip(cached.iter()) {
            assert!((lhs.0 - rhs.0).abs() < 1e-12);
            assert!((lhs.1 - rhs.1).abs() < 1e-10);
        }
    }

    #[test]
    fn context_grid_matches_fft() {
        let cf = BlackScholesCharFn::new(100.0, 0.02, 0.0, 0.2, 1.0);
        let params = CarrMadanParams::default();

        let direct = carr_madan_fft(&cf, 0.02, 1.0, 100.0, params).expect("fft pricing succeeds");
        let ctx = CarrMadanContext::new(&cf, 0.02, 1.0, 100.0, params)
            .expect("context construction succeeds");
        let cached = ctx.price_grid().expect("context grid pricing succeeds");

        for (lhs, rhs) in direct.iter().zip(cached.iter()) {
            assert!((lhs.0 - rhs.0).abs() < 1e-12);
            assert!((lhs.1 - rhs.1).abs() < 1e-10);
        }
    }
}
