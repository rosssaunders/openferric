//! Module `models::cgmy`.
//!
//! Implements cgmy abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Carr, Geman, Madan, Yor (2002), Hull (11th ed.) Ch. 19, tempered-stable exponent definitions around CGMY Eq. (2.4).
//!
//! Key types and purpose: `Cgmy` define the core data contracts for this module.
//!
//! Numerical considerations: parameter admissibility constraints are essential (positivity/integrability/stationarity) to avoid unstable simulation or invalid characteristic functions.
//!
//! When to use: select this model module when its dynamics match observed skew/tail/term-structure behavior; prefer simpler models for calibration speed or interpretability.

use num_complex::Complex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma};

use crate::calibration::{BoxConstraints, LmOptions, levenberg_marquardt};
use crate::engines::fft::carr_madan::CarrMadanParams;
use crate::engines::fft::char_fn::{CgmyCharFn, CharacteristicFunction};
use crate::engines::fft::frft::carr_madan_price_at_strikes;
use crate::math::gamma::gamma;

// Calibration deliberately uses one Carr-Madan contour and one frequency grid.
// With alpha=0.5 the transform needs the exponential moment of order 1.5.
// Requiring M>=1.6 leaves a 6.7% moment margin, so the adaptive pricing fallback
// never changes alpha or grid while LM takes finite-difference derivatives.  The
// grid is refined by four relative to the production default (and keeps the same
// frequency range) because the lower damping makes the integrand sharper at zero.
const CGMY_CALIBRATION_ALPHA: f64 = 0.5;
const CGMY_CALIBRATION_N: usize = 16_384;
const CGMY_CALIBRATION_ETA: f64 = 0.0625;
const MIN_CGMY_CALIBRATION_M: f64 = 1.6;

const MIN_CGMY_CALIBRATION_C: f64 = 1.0e-8;
const MIN_CGMY_CALIBRATION_G: f64 = 1.0e-6;
const MAX_CGMY_CALIBRATION_PARAMETER: f64 = 1.0e3;
const MIN_CGMY_CALIBRATION_Y: f64 = -5.0;
const CGMY_CALIBRATION_POLE_MARGIN: f64 = 1.0e-4;
const CGMY_CALIBRATION_PENALTY: f64 = 1.0e6;
const CGMY_JACOBIAN_RELATIVE_RANK_TOLERANCE: f64 = 1.0e-8;
const CGMY_JACOBIAN_ABSOLUTE_RANK_TOLERANCE: f64 = 1.0e-10;

fn cgmy_calibration_fft_params() -> CarrMadanParams {
    CarrMadanParams {
        n: CGMY_CALIBRATION_N,
        eta: CGMY_CALIBRATION_ETA,
        alpha: CGMY_CALIBRATION_ALPHA,
    }
}

/// CGMY (Carr-Geman-Madan-Yor) tempered stable process parameters.
///
/// The CGMY process generalises the Variance-Gamma model to allow infinite
/// activity/variation via the `Y` parameter.  The Lévy density is:
///
///   `k(x) = C * exp(-G|x|) / |x|^{1+Y}`  for x < 0
///   `k(x) = C * exp(-M*x) / x^{1+Y}`      for x > 0
///
/// Special case: `Y → 0` recovers the Variance-Gamma process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cgmy {
    /// Activity/scale parameter (C > 0).
    pub c: f64,
    /// Exponential decay of the negative jumps (G > 0).
    pub g: f64,
    /// Exponential decay of the positive jumps (M > 1 for martingale condition).
    pub m: f64,
    /// Fine structure exponent (Y < 2, Y ≠ 0, Y ≠ 1).
    pub y: f64,
}

impl Cgmy {
    pub fn validate(&self) -> Result<(), String> {
        if !self.c.is_finite() || self.c <= 0.0 {
            return Err("CGMY requires C > 0".to_string());
        }
        if !self.g.is_finite() || self.g <= 0.0 {
            return Err("CGMY requires G > 0".to_string());
        }
        if !self.m.is_finite() || self.m <= 1.0 {
            return Err("CGMY requires M > 1 for risk-neutral martingale correction".to_string());
        }
        if !self.y.is_finite() || self.y >= 2.0 || self.y == 0.0 || self.y == 1.0 {
            return Err("CGMY requires Y in (-∞, 2) excluding 0 and 1".to_string());
        }
        Ok(())
    }

    /// Risk-neutral drift correction (martingale adjustment).
    pub fn martingale_correction(&self) -> Result<f64, String> {
        self.validate()?;
        let gamma_neg_y = gamma(-self.y);
        let omega = -self.c
            * gamma_neg_y
            * ((self.m - 1.0).powf(self.y) - self.m.powf(self.y) + (self.g + 1.0).powf(self.y)
                - self.g.powf(self.y));
        Ok(omega)
    }

    /// Characteristic function of `ln(S_T)` under risk-neutral drift.
    pub fn characteristic_fn(
        &self,
        u: Complex<f64>,
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
    ) -> Result<Complex<f64>, String> {
        self.validate()?;
        let cf = CgmyCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.c,
            self.g,
            self.m,
            self.y,
        )?;
        Ok(CharacteristicFunction::cf(&cf, u))
    }

    /// European call pricing via CGMY characteristic function + Carr-Madan FFT.
    pub fn european_calls_fft(
        &self,
        spot: f64,
        strikes: &[f64],
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        params: CarrMadanParams,
    ) -> Result<Vec<(f64, f64)>, String> {
        self.validate()?;
        let cf = CgmyCharFn::risk_neutral(
            spot,
            rate,
            dividend_yield,
            maturity,
            self.c,
            self.g,
            self.m,
            self.y,
        )?;
        carr_madan_price_at_strikes(&cf, rate, maturity, spot, strikes, params)
    }

    /// Monte Carlo simulation via subordination (tempered stable subordinator
    /// approximated by a gamma process with matching cumulants).
    ///
    /// For Y ∈ (0, 1) the subordinator is a tempered stable process;
    /// we approximate it by a gamma process matching the first two cumulants:
    ///   mean = C Γ(1-Y) (G^{Y-1} + M^{Y-1}) dt
    ///   var  = C Γ(2-Y) (G^{Y-2} + M^{Y-2}) dt
    pub fn simulate_terminal_spots(
        &self,
        initial_spot: f64,
        rate: f64,
        dividend_yield: f64,
        horizon: f64,
        num_steps: usize,
        num_paths: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        if !initial_spot.is_finite() || initial_spot <= 0.0 {
            return Err("initial_spot must be finite and > 0".to_string());
        }
        if num_steps == 0 || num_paths == 0 {
            return Err("num_steps and num_paths must be > 0".to_string());
        }

        let omega = self.martingale_correction()?;
        let dt = horizon / num_steps as f64;
        let drift_dt = (rate - dividend_yield + omega) * dt;

        // Gamma approximation of positive/negative subordinators
        let gamma_1_y = gamma(1.0 - self.y);
        let gamma_2_y = gamma(2.0 - self.y);

        let mean_pos = self.c * gamma_1_y * self.m.powf(self.y - 1.0) * dt;
        let var_pos = self.c * gamma_2_y * self.m.powf(self.y - 2.0) * dt;
        let mean_neg = self.c * gamma_1_y * self.g.powf(self.y - 1.0) * dt;
        let var_neg = self.c * gamma_2_y * self.g.powf(self.y - 2.0) * dt;

        let shape_pos = mean_pos * mean_pos / var_pos;
        let scale_pos = var_pos / mean_pos;
        let shape_neg = mean_neg * mean_neg / var_neg;
        let scale_neg = var_neg / mean_neg;

        let gamma_pos = Gamma::new(shape_pos, scale_pos)
            .map_err(|e| format!("gamma distribution (pos): {e}"))?;
        let gamma_neg = Gamma::new(shape_neg, scale_neg)
            .map_err(|e| format!("gamma distribution (neg): {e}"))?;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(num_paths);

        for _ in 0..num_paths {
            let mut log_s = initial_spot.ln();
            for _ in 0..num_steps {
                // CGMY is a pure-jump process: the log-increment is the difference of the
                // positive/negative tempered-stable subordinators (gamma-approximated with
                // matched first two cumulants), so no Brownian component is sampled.
                let g_pos = gamma_pos.sample(&mut rng);
                let g_neg = gamma_neg.sample(&mut rng);
                log_s += drift_dt + g_pos - g_neg;
            }
            out.push(log_s.exp());
        }

        Ok(out)
    }

    /// Calibrates all four CGMY parameters to market call prices.
    ///
    /// Positive parameters are optimized in log space, while `Y` is kept in
    /// the same admissible interval (`Y < 0`, `0 < Y < 1`, or `1 < Y < 2`)
    /// as the supplied initial guess. This avoids stepping across the poles at
    /// `Y = 0` and `Y = 1` and gives the optimizer comparable parameter
    /// scales even when `C`, `G`, and `M` differ by orders of magnitude.
    ///
    /// Calibration uses a fixed Carr-Madan contour (`alpha=0.5`) and refined
    /// grid throughout the solve. Consequently this method requires `M >= 1.6`,
    /// a safety margin above the contour's moment order of `1.5`. At least four
    /// distinct strikes are required to identify the four fitted parameters.
    pub fn calibrate(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        maturity: f64,
        strikes: &[f64],
        market_prices: &[f64],
        initial_guess: Cgmy,
        max_iter: usize,
    ) -> Result<Cgmy, String> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err("CGMY calibration requires spot to be finite and > 0".to_string());
        }
        if !rate.is_finite() || !dividend_yield.is_finite() {
            return Err("CGMY calibration requires finite rate and dividend_yield".to_string());
        }
        if !maturity.is_finite() || maturity <= 0.0 {
            return Err("CGMY calibration requires maturity to be finite and > 0".to_string());
        }

        initial_guess.validate()?;
        if strikes.len() != market_prices.len() || strikes.len() < 4 {
            return Err(
                "CGMY calibration requires at least four strikes and matching market prices"
                    .to_string(),
            );
        }
        if strikes
            .iter()
            .any(|strike| !strike.is_finite() || *strike <= 0.0)
            || market_prices
                .iter()
                .any(|price| !price.is_finite() || *price < 0.0)
        {
            return Err(
                "CGMY calibration requires finite positive strikes and finite non-negative prices"
                    .to_string(),
            );
        }
        let mut sorted_strikes = strikes.to_vec();
        sorted_strikes.sort_by(f64::total_cmp);
        if sorted_strikes.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("CGMY calibration requires distinct strikes".to_string());
        }
        if !(MIN_CGMY_CALIBRATION_C..=MAX_CGMY_CALIBRATION_PARAMETER).contains(&initial_guess.c)
            || !(MIN_CGMY_CALIBRATION_G..=MAX_CGMY_CALIBRATION_PARAMETER).contains(&initial_guess.g)
            || !(MIN_CGMY_CALIBRATION_M..=MAX_CGMY_CALIBRATION_PARAMETER).contains(&initial_guess.m)
            || initial_guess.y < MIN_CGMY_CALIBRATION_Y
        {
            return Err(format!(
                "CGMY calibration initial guess lies outside the numerical search domain: \
                 C in [{MIN_CGMY_CALIBRATION_C}, {MAX_CGMY_CALIBRATION_PARAMETER}], \
                 G in [{MIN_CGMY_CALIBRATION_G}, {MAX_CGMY_CALIBRATION_PARAMETER}], \
                 M in [{MIN_CGMY_CALIBRATION_M}, {MAX_CGMY_CALIBRATION_PARAMETER}], \
                 Y >= {MIN_CGMY_CALIBRATION_Y}"
            ));
        }

        let params_fft = cgmy_calibration_fft_params();
        let price = |model: Cgmy| {
            model
                .european_calls_fft(spot, strikes, rate, dividend_yield, maturity, params_fft)
                .map(|values| {
                    values
                        .into_iter()
                        .map(|(_, value)| value)
                        .collect::<Vec<_>>()
                })
        };
        let scales = market_prices
            .iter()
            .map(|price| price.abs().max(1.0))
            .collect::<Vec<_>>();
        let normalized_score = |values: &[f64]| {
            0.5 * values
                .iter()
                .zip(market_prices)
                .zip(scales.iter())
                .map(|((model, market), scale)| ((model - market) / scale).powi(2))
                .sum::<f64>()
        };

        let initial_prices = price(initial_guess)
            .map_err(|error| format!("failed to price CGMY initial guess: {error}"))?;
        let initial_score = normalized_score(&initial_prices);

        let (y_lower, y_upper) = if initial_guess.y < 0.0 {
            (MIN_CGMY_CALIBRATION_Y, -CGMY_CALIBRATION_POLE_MARGIN)
        } else if initial_guess.y < 1.0 {
            (
                CGMY_CALIBRATION_POLE_MARGIN,
                1.0 - CGMY_CALIBRATION_POLE_MARGIN,
            )
        } else {
            (
                1.0 + CGMY_CALIBRATION_POLE_MARGIN,
                2.0 - CGMY_CALIBRATION_POLE_MARGIN,
            )
        };
        if initial_guess.y <= y_lower || initial_guess.y >= y_upper {
            return Err(format!(
                "CGMY calibration initial Y={} must lie strictly inside [{y_lower}, {y_upper}]",
                initial_guess.y
            ));
        }
        if max_iter == 0 {
            return Ok(initial_guess);
        }
        let bounds = BoxConstraints::new(
            vec![
                MIN_CGMY_CALIBRATION_C.ln(),
                MIN_CGMY_CALIBRATION_G.ln(),
                (MIN_CGMY_CALIBRATION_M - 1.0).ln(),
                y_lower,
            ],
            vec![
                MAX_CGMY_CALIBRATION_PARAMETER.ln(),
                MAX_CGMY_CALIBRATION_PARAMETER.ln(),
                (MAX_CGMY_CALIBRATION_PARAMETER - 1.0).ln(),
                y_upper,
            ],
        )?;
        let initial = [
            initial_guess.c.ln(),
            initial_guess.g.ln(),
            (initial_guess.m - 1.0).ln(),
            initial_guess.y,
        ];
        let decode = |x: &[f64]| Cgmy {
            c: x[0].exp(),
            g: x[1].exp(),
            m: 1.0 + x[2].exp(),
            y: x[3],
        };

        let options = LmOptions {
            max_iterations: max_iter,
            initial_lambda: 1.0e-3,
            // The residuals are normalized quote errors. A 5e-8 gradient in
            // transformed parameter space is already below market-data noise,
            // while remaining strict enough to reject short/incomplete solves.
            gradient_tolerance: 5.0e-8,
            step_tolerance: 1.0e-9,
            objective_tolerance: 1.0e-18,
            finite_diff_epsilon: 1.0e-5,
            max_stagnation: 40,
            ..LmOptions::default()
        };
        let mut pricing_failure_count = 0usize;
        let mut first_pricing_failure = None;
        let optimized = levenberg_marquardt(&initial, &bounds, options, |x| {
            let candidate = decode(x);
            match price(candidate) {
                Ok(values) => values
                    .iter()
                    .zip(market_prices)
                    .zip(scales.iter())
                    .map(|((model, market), scale)| (model - market) / scale)
                    .collect(),
                Err(error) => {
                    pricing_failure_count += 1;
                    if first_pricing_failure.is_none() {
                        first_pricing_failure = Some(error);
                    }
                    vec![CGMY_CALIBRATION_PENALTY; market_prices.len()]
                }
            }
        })?;

        if pricing_failure_count > 0 {
            return Err(format!(
                "CGMY calibration encountered {pricing_failure_count} numerical pricing \
                 failure(s); first failure: {}",
                first_pricing_failure.as_deref().unwrap_or("unknown error")
            ));
        }
        if !optimized.convergence.converged {
            return Err(format!(
                "CGMY calibration did not converge after {} iterations: {:?} \
                 (objective={}, gradient_norm={}, step_norm={})",
                optimized.convergence.iterations,
                optimized.convergence.reason,
                optimized.objective,
                optimized.convergence.gradient_norm,
                optimized.convergence.step_norm,
            ));
        }
        if bounds.hits_boundary(&optimized.x, 1.0e-7) {
            return Err(format!(
                "CGMY calibration converged on a parameter boundary: {:?}",
                optimized.x
            ));
        }

        let singular_values = optimized.jacobian.clone().svd(false, false).singular_values;
        let max_singular = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let rank_threshold = (max_singular * CGMY_JACOBIAN_RELATIVE_RANK_TOLERANCE)
            .max(CGMY_JACOBIAN_ABSOLUTE_RANK_TOLERANCE);
        let numerical_rank = singular_values
            .iter()
            .filter(|value| value.is_finite() && **value > rank_threshold)
            .count();
        if singular_values.len() < 4 || !max_singular.is_finite() || numerical_rank < 4 {
            return Err(format!(
                "CGMY calibration is non-identifiable: normalized Jacobian rank \
                 {numerical_rank}/4 (singular values={singular_values:?})"
            ));
        }

        let candidate = decode(&optimized.x);
        let candidate_prices = price(candidate)
            .map_err(|error| format!("failed to price calibrated CGMY parameters: {error}"))?;
        let candidate_score = normalized_score(&candidate_prices);
        let comparison_slack = 64.0 * f64::EPSILON * initial_score.abs().max(1.0);
        if !candidate_score.is_finite() || candidate_score > initial_score + comparison_slack {
            return Err(format!(
                "CGMY calibration worsened its normalized objective: \
                 initial={initial_score}, candidate={candidate_score}"
            ));
        }
        candidate.validate()?;
        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::fft::carr_madan::resolve_admissible_params;

    const CALIBRATION_STRIKES: [f64; 9] =
        [65.0, 75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 130.0, 150.0];
    const CALIBRATION_SPOT: f64 = 100.0;
    const CALIBRATION_RATE: f64 = 0.05;
    const CALIBRATION_DIVIDEND: f64 = 0.01;
    const CALIBRATION_MATURITY: f64 = 1.0;

    fn calibration_prices(model: Cgmy, strikes: &[f64]) -> Vec<f64> {
        model
            .european_calls_fft(
                CALIBRATION_SPOT,
                strikes,
                CALIBRATION_RATE,
                CALIBRATION_DIVIDEND,
                CALIBRATION_MATURITY,
                cgmy_calibration_fft_params(),
            )
            .unwrap()
            .into_iter()
            .map(|(_, price)| price)
            .collect()
    }

    fn normalized_score(model: Cgmy, strikes: &[f64], market_prices: &[f64]) -> f64 {
        calibration_prices(model, strikes)
            .iter()
            .zip(market_prices)
            .map(|(model_price, market_price)| {
                ((model_price - market_price) / market_price.abs().max(1.0)).powi(2)
            })
            .sum::<f64>()
            * 0.5
    }

    fn assert_parameters_close(actual: Cgmy, expected: Cgmy, relative_budget: f64) {
        for (label, actual, expected) in [
            ("C", actual.c, expected.c),
            ("G", actual.g, expected.g),
            ("M", actual.m, expected.m),
            ("Y", actual.y, expected.y),
        ] {
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= relative_budget,
                "CGMY calibration failed to recover {label}: actual={actual}, \
                 expected={expected}, relative_error={relative_error}, budget={relative_budget}"
            );
        }
    }

    #[test]
    fn cgmy_cf_is_one_at_zero() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let cf_val = cgmy
            .characteristic_fn(Complex::new(0.0, 0.0), 100.0, 0.03, 0.0, 1.0)
            .unwrap();
        assert!((cf_val.re - 1.0).abs() < 1e-10);
        assert!(cf_val.im.abs() < 1e-10);
    }

    #[test]
    fn cgmy_fft_prices_are_positive_and_bounded() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let prices = cgmy
            .european_calls_fft(100.0, &strikes, 0.03, 0.0, 0.5, CarrMadanParams::default())
            .unwrap();
        for (k, p) in &prices {
            assert!(*p > 0.0, "price at strike {k} should be > 0, got {p}");
            assert!(*p < 100.0, "price at strike {k} should be < spot, got {p}");
        }
    }

    #[test]
    fn cgmy_mc_terminal_spots_are_positive() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let spots = cgmy
            .simulate_terminal_spots(100.0, 0.03, 0.0, 1.0, 50, 200, 42)
            .unwrap();
        assert_eq!(spots.len(), 200);
        assert!(spots.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn cgmy_validation_rejects_invalid_params() {
        let bad_m = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 0.5,
            y: 0.5,
        };
        assert!(bad_m.validate().is_err());

        let bad_y = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.0,
        };
        assert!(bad_y.validate().is_err());
    }

    #[test]
    fn cgmy_martingale_correction_is_finite() {
        let cgmy = Cgmy {
            c: 1.0,
            g: 5.0,
            m: 10.0,
            y: 0.5,
        };
        let omega = cgmy.martingale_correction().unwrap();
        assert!(omega.is_finite());
    }

    #[test]
    fn calibration_rejects_invalid_or_underdetermined_inputs_before_noop() {
        let initial = Cgmy {
            c: 0.2,
            g: 6.5,
            m: 11.0,
            y: 0.4,
        };
        let strikes = [80.0, 90.0, 100.0, 110.0];
        let prices = [22.0, 14.0, 8.0, 4.0];

        for (label, spot, rate, dividend, maturity) in [
            ("zero spot", 0.0, 0.05, 0.01, 1.0),
            ("NaN spot", f64::NAN, 0.05, 0.01, 1.0),
            ("NaN rate", 100.0, f64::NAN, 0.01, 1.0),
            ("infinite dividend", 100.0, 0.05, f64::INFINITY, 1.0),
            ("zero maturity", 100.0, 0.05, 0.01, 0.0),
            ("negative maturity", 100.0, 0.05, 0.01, -1.0),
            ("NaN maturity", 100.0, 0.05, 0.01, f64::NAN),
        ] {
            let error = Cgmy::calibrate(
                spot, rate, dividend, maturity, &strikes, &prices, initial, 0,
            );
            assert!(error.is_err(), "{label} bypassed calibration validation");
        }

        let error = Cgmy::calibrate(
            100.0,
            0.05,
            0.01,
            1.0,
            &strikes[..3],
            &prices[..3],
            initial,
            0,
        )
        .unwrap_err();
        assert!(error.contains("at least four"));

        let duplicate_strikes = [80.0, 90.0, 90.0, 100.0];
        let error = Cgmy::calibrate(
            100.0,
            0.05,
            0.01,
            1.0,
            &duplicate_strikes,
            &prices,
            initial,
            0,
        )
        .unwrap_err();
        assert!(error.contains("distinct strikes"));

        let unsafe_seed = Cgmy { m: 1.1, ..initial };
        let error =
            Cgmy::calibrate(100.0, 0.05, 0.01, 1.0, &strikes, &prices, unsafe_seed, 0).unwrap_err();
        assert!(error.contains("search domain"));
    }

    #[test]
    fn calibration_fixed_grid_does_not_switch_at_default_fallback_thresholds() {
        let requested = cgmy_calibration_fft_params();
        // These values straddle every old default-alpha fallback threshold that
        // is inside the calibration domain. The method-specific alpha only needs
        // M>1.5, so all of them must retain the exact same grid and contour.
        for m in [
            MIN_CGMY_CALIBRATION_M,
            1.837_499,
            1.837_501,
            2.499_999,
            2.500_001,
        ] {
            let cf = CgmyCharFn::risk_neutral(
                CALIBRATION_SPOT,
                CALIBRATION_RATE,
                CALIBRATION_DIVIDEND,
                CALIBRATION_MATURITY,
                0.2,
                5.0,
                m,
                0.5,
            )
            .unwrap();
            let resolved = resolve_admissible_params(&cf, requested).unwrap();
            assert_eq!(resolved.n, requested.n, "grid size changed at M={m}");
            assert_eq!(resolved.eta.to_bits(), requested.eta.to_bits());
            assert_eq!(resolved.alpha.to_bits(), requested.alpha.to_bits());
        }
    }

    #[test]
    fn calibration_rejects_non_identifiable_and_non_converged_fits() {
        let reference = Cgmy {
            c: 0.35,
            g: 4.5,
            m: 8.0,
            y: 0.65,
        };
        let close_strikes = [100.0, 100.000_001, 100.000_002, 100.000_003];
        let close_prices = calibration_prices(reference, &close_strikes);
        let error = Cgmy::calibrate(
            CALIBRATION_SPOT,
            CALIBRATION_RATE,
            CALIBRATION_DIVIDEND,
            CALIBRATION_MATURITY,
            &close_strikes,
            &close_prices,
            reference,
            20,
        )
        .unwrap_err();
        assert!(
            error.contains("non-identifiable"),
            "unexpected error: {error}"
        );

        let initial = Cgmy {
            c: 0.2,
            g: 6.5,
            m: 11.0,
            y: 0.4,
        };
        let market_prices = calibration_prices(reference, &CALIBRATION_STRIKES);
        let error = Cgmy::calibrate(
            CALIBRATION_SPOT,
            CALIBRATION_RATE,
            CALIBRATION_DIVIDEND,
            CALIBRATION_MATURITY,
            &CALIBRATION_STRIKES,
            &market_prices,
            initial,
            1,
        )
        .unwrap_err();
        assert!(
            error.contains("did not converge"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn calibration_matches_independent_scipy_and_fypy_reference() {
        // Independently generated by scipy 1.18.0 `integrate.quad` over the
        // continuous Carr-Madan integral (alpha=1.5, intervals ending at
        // 25,50,100,200,500,1000, epsabs=epsrel=2e-12). The generator coded the
        // CGMY exponent directly rather than calling OpenFerric. fypy commit
        // 0e22a518c4a7c05e8faf415e93a22c5e4fa47574 independently corroborates
        // the vector with `ProjEuropeanPricer(N=2^14,L=12,order=3)` to 4.3e-9.
        let reference = Cgmy {
            c: 0.35,
            g: 4.5,
            m: 8.0,
            y: 0.65,
        };
        let market_prices = [
            37.605_233_346_151_7,
            28.768_099_386_271_196,
            20.657_066_051_117_013,
            13.726_907_277_387_058,
            10.847_963_317_020_533,
            8.409_899_654_907_32,
            4.839_675_812_092_493,
            2.028_190_167_999_301,
            0.666_804_189_320_289_5,
        ];

        let reference_prices = calibration_prices(reference, &CALIBRATION_STRIKES);
        for ((strike, actual), expected) in CALIBRATION_STRIKES
            .iter()
            .zip(reference_prices)
            .zip(market_prices)
        {
            let error = (actual - expected).abs();
            assert!(
                error <= 5.0e-11,
                "CGMY fixed-grid price disagrees with independent quadrature at \
                 K={strike}: actual={actual}, expected={expected}, error={error}"
            );
        }

        let initial = Cgmy {
            c: 0.2,
            g: 6.5,
            m: 11.0,
            y: 0.4,
        };
        let calibrated = Cgmy::calibrate(
            CALIBRATION_SPOT,
            CALIBRATION_RATE,
            CALIBRATION_DIVIDEND,
            CALIBRATION_MATURITY,
            &CALIBRATION_STRIKES,
            &market_prices,
            initial,
            150,
        )
        .unwrap();
        assert_parameters_close(calibrated, reference, 2.0e-4);

        let repriced = calibration_prices(calibrated, &CALIBRATION_STRIKES);
        for ((strike, actual), expected) in
            CALIBRATION_STRIKES.iter().zip(repriced).zip(market_prices)
        {
            assert!(
                (actual - expected).abs() <= 2.0e-8,
                "calibrated CGMY failed independent quote at K={strike}: \
                 actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn calibration_recovers_all_three_y_regimes() {
        for (reference, initial) in [
            (
                Cgmy {
                    c: 0.35,
                    g: 7.0,
                    m: 10.0,
                    y: -0.5,
                },
                Cgmy {
                    c: 0.3,
                    g: 6.0,
                    m: 9.0,
                    y: -0.4,
                },
            ),
            (
                Cgmy {
                    c: 0.35,
                    g: 4.5,
                    m: 8.0,
                    y: 0.65,
                },
                Cgmy {
                    c: 0.2,
                    g: 6.5,
                    m: 11.0,
                    y: 0.4,
                },
            ),
            (
                Cgmy {
                    c: 0.02,
                    g: 5.0,
                    m: 15.0,
                    y: 1.2,
                },
                Cgmy {
                    c: 0.03,
                    g: 6.0,
                    m: 12.0,
                    y: 1.1,
                },
            ),
        ] {
            let market_prices = calibration_prices(reference, &CALIBRATION_STRIKES);
            let calibrated = Cgmy::calibrate(
                CALIBRATION_SPOT,
                CALIBRATION_RATE,
                CALIBRATION_DIVIDEND,
                CALIBRATION_MATURITY,
                &CALIBRATION_STRIKES,
                &market_prices,
                initial,
                180,
            )
            .unwrap();
            assert_parameters_close(calibrated, reference, 3.0e-3);
            assert!(normalized_score(calibrated, &CALIBRATION_STRIKES, &market_prices) <= 1.0e-14);
        }
    }

    #[test]
    fn calibration_returns_an_interior_best_fit_for_noisy_quotes() {
        let reference = Cgmy {
            c: 0.35,
            g: 4.5,
            m: 8.0,
            y: 0.65,
        };
        let initial_guess = Cgmy {
            c: 0.20,
            g: 6.5,
            m: 11.0,
            y: 0.40,
        };
        let mut market_prices = calibration_prices(reference, &CALIBRATION_STRIKES);
        let relative_noise = [
            4.0e-4, -3.0e-4, 2.0e-4, -4.0e-4, 3.0e-4, -2.0e-4, 4.0e-4, -3.0e-4, 2.0e-4,
        ];
        for (price, noise) in market_prices.iter_mut().zip(relative_noise) {
            *price *= 1.0 + noise;
        }
        let initial_score = normalized_score(initial_guess, &CALIBRATION_STRIKES, &market_prices);

        let calibrated = Cgmy::calibrate(
            CALIBRATION_SPOT,
            CALIBRATION_RATE,
            CALIBRATION_DIVIDEND,
            CALIBRATION_MATURITY,
            &CALIBRATION_STRIKES,
            &market_prices,
            initial_guess,
            180,
        )
        .unwrap();
        let fitted_score = normalized_score(calibrated, &CALIBRATION_STRIKES, &market_prices);
        let normalized_rmse = (2.0 * fitted_score / market_prices.len() as f64).sqrt();
        assert!(fitted_score < 0.01 * initial_score);
        assert!(
            normalized_rmse <= 5.0e-4,
            "normalized RMSE={normalized_rmse}"
        );
        assert_ne!(calibrated, initial_guess);
        calibrated.validate().unwrap();
    }
}
