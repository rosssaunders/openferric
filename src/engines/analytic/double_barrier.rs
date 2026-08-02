//! Module `engines::analytic::double_barrier`.
//!
//! Implements double barrier abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13 and Ch. 26, Black-Scholes style formulas around Eq. (13.16)-(13.20), plus instrument-specific papers cited in-code.
//!
//! Key types and purpose: `DoubleBarrierAnalyticEngine` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: prefer this module for fast closed-form pricing/Greeks; use tree/PDE/Monte Carlo modules when payoffs, exercise rules, or dynamics break closed-form assumptions.
use std::f64::consts::PI;

use crate::core::{OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::double_barrier::{DoubleBarrierOption, DoubleBarrierType};
use crate::market::Market;
use crate::math::normal_cdf;

/// Analytic engine for double-barrier options using a truncated Ikeda-Kunitomo series.
#[derive(Debug, Clone)]
pub struct DoubleBarrierAnalyticEngine {
    /// Number of series terms in the expansion.
    pub series_terms: usize,
}

impl Default for DoubleBarrierAnalyticEngine {
    fn default() -> Self {
        Self { series_terms: 5 }
    }
}

impl DoubleBarrierAnalyticEngine {
    /// Creates the engine with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of series terms.
    pub fn with_series_terms(mut self, series_terms: usize) -> Self {
        self.series_terms = series_terms;
        self
    }
}

#[inline]
fn vanilla_payoff(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

#[inline]
fn bs_price_with_dividend(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 {
        return vanilla_payoff(option_type, spot, strike);
    }

    let sqrt_t = expiry.sqrt();
    let sig_sqrt_t = vol * sqrt_t;
    let d1 = ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();

    // Compute call, derive put via put-call parity to halve CDF evaluations.
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let call = spot.mul_add(df_q * nd1, -(strike * df_r * nd2));
    match option_type {
        OptionType::Call => call,
        OptionType::Put => call - spot * df_q + strike * df_r,
    }
}

#[inline]
fn exp_sin_integral(m: f64, w: f64, y0: f64, y1: f64) -> f64 {
    if y1 <= y0 {
        return 0.0;
    }
    let denom = m * m + w * w;
    let primitive = |y: f64| (m * y).exp() * (m * (w * y).sin() - w * (w * y).cos()) / denom;
    primitive(y1) - primitive(y0)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn double_knock_out_zero_rebate(
    option_type: OptionType,
    spot: f64,
    strike: f64,
    lower: f64,
    upper: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    series_terms: usize,
) -> f64 {
    let vol_sq = vol * vol;
    let std = vol * expiry.sqrt();
    let b = rate - dividend_yield;
    let mu1 = 2.0 * b / vol_sq + 1.0;
    let bsigma = (0.5 * vol).mul_add(vol, b) * expiry / std;
    let df_r = (-rate * expiry).exp();
    let df_q = (-dividend_yield * expiry).exp();
    let n_max = series_terms as i32;
    let ln_u_over_l = (upper / lower).ln();

    let mut acc1 = 0.0;
    let mut acc2 = 0.0;

    for n in -n_max..=n_max {
        let nf = n as f64;
        // Compute powers via exp(n * ln(u/l)) instead of repeated powf.
        let l_over_u_n = (-nf * ln_u_over_l).exp();
        let u_over_l_2n = (2.0 * nf * ln_u_over_l).exp();
        let l_over_u_2n = (-2.0 * nf * ln_u_over_l).exp();

        let mirror_base = (lower / spot) * l_over_u_n;
        let m1 = (mu1 * nf * ln_u_over_l).exp();
        let m2 = ((mu1 - 2.0) * nf * ln_u_over_l).exp();
        let m3 = mirror_base.powf(mu1);
        let m4 = mirror_base.powf(mu1 - 2.0);

        match option_type {
            OptionType::Call => {
                let d1 = ((spot * u_over_l_2n / strike).ln()) / std + bsigma;
                let d2 = ((spot * u_over_l_2n / upper).ln()) / std + bsigma;
                let d3 = ((lower * lower * l_over_u_2n / (strike * spot)).ln()) / std + bsigma;
                let d4 = ((lower * lower * l_over_u_2n / (upper * spot)).ln()) / std + bsigma;

                acc1 +=
                    m1 * (normal_cdf(d1) - normal_cdf(d2)) - m3 * (normal_cdf(d3) - normal_cdf(d4));
                acc2 += m2 * (normal_cdf(d1 - std) - normal_cdf(d2 - std))
                    - m4 * (normal_cdf(d3 - std) - normal_cdf(d4 - std));
            }
            OptionType::Put => {
                let y1 = ((spot * u_over_l_2n / lower).ln()) / std + bsigma;
                let y2 = ((spot * u_over_l_2n / strike).ln()) / std + bsigma;
                let y3 = ((lower * lower * l_over_u_2n / (lower * spot)).ln()) / std + bsigma;
                let y4 = ((lower * lower * l_over_u_2n / (strike * spot)).ln()) / std + bsigma;

                acc1 += m2 * (normal_cdf(y1 - std) - normal_cdf(y2 - std))
                    - m4 * (normal_cdf(y3 - std) - normal_cdf(y4 - std));
                acc2 +=
                    m1 * (normal_cdf(y1) - normal_cdf(y2)) - m3 * (normal_cdf(y3) - normal_cdf(y4));
            }
        }
    }

    match option_type {
        OptionType::Call => (spot * df_q * acc1 - strike * df_r * acc2).max(0.0),
        OptionType::Put => (strike * df_r * acc1 - spot * df_q * acc2).max(0.0),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn double_no_touch_digital_price(
    spot: f64,
    lower: f64,
    upper: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    series_terms: usize,
) -> f64 {
    let vol_sq = vol * vol;
    let width = (upper / lower).ln();
    let x = (spot / lower).ln();

    let carry = (-0.5 * vol).mul_add(vol, rate - dividend_yield);
    let alpha = -carry / vol_sq;
    let c = (-0.5 * carry).mul_add(carry / vol_sq, -rate);
    let prefactor = alpha.mul_add(x, c * expiry).exp();

    let mut sum = 0.0;
    for n in 1..=series_terms {
        let w = (n as f64) * PI / width;
        let b_n = 2.0 / width * exp_sin_integral(-alpha, w, 0.0, width);
        let decay = (-0.5 * vol * vol * w * w * expiry).exp();
        sum += b_n * (w * x).sin() * decay;
    }

    let df_r = (-rate * expiry).exp();
    (prefactor * sum).clamp(0.0, df_r)
}

/// Infinite-horizon transform `E[exp(-r*tau)]` of the first exit time from a
/// bounded log-price corridor.
///
/// This is the closed-form solution of
/// `0.5*sigma^2*f'' + carry*f' - r*f = 0`, with `f(0)=f(width)=1`.
/// Returning `None` only covers the singular negative-rate parameter sets at
/// which the infinite-horizon exponential moment does not exist; the finite
/// maturity calculation below then falls back to direct integration.
fn eventual_double_touch_discount_factor(
    carry: f64,
    vol_sq: f64,
    width: f64,
    x: f64,
    rate: f64,
) -> Option<f64> {
    let alpha = -carry / vol_sq;
    // Divide the principal eigenvalue by positive `vol_sq` before checking its
    // sign, avoiding products that can underflow for very small volatilities.
    let corridor_mode = PI / width;
    let principal_nu_over_vol_sq =
        (-0.5 * alpha).mul_add(alpha, -(rate / vol_sq)) - 0.5 * corridor_mode * corridor_mode;
    if principal_nu_over_vol_sq.is_nan() || principal_nu_over_vol_sq >= 0.0 {
        return None;
    }

    // Normalize the characteristic discriminant before evaluating it.  The
    // dimensional form `carry^2 + 2*r*vol_sq` can underflow even when the
    // characteristic root itself is order one because the root subsequently
    // divides by `vol_sq`.
    let drift_term = alpha * alpha;
    let rate_term = 2.0 * (rate / vol_sq);
    let discriminant = alpha.mul_add(alpha, rate_term);
    let coefficient_scale = drift_term.abs().max(rate_term.abs());
    if !coefficient_scale.is_finite() {
        return None;
    }
    let scale = 64.0 * f64::EPSILON * coefficient_scale;

    let (left_ratio, right_ratio) = if discriminant > scale {
        let delta = discriminant.sqrt();
        let sinh_ratio = |part: f64| {
            let whole = delta * width;
            if whole.abs() < 1.0e-5 {
                (delta * part).sinh() / whole.sinh()
            } else {
                (delta * (part - width)).exp() * (-2.0 * delta * part).exp_m1()
                    / (-2.0 * whole).exp_m1()
            }
        };
        (sinh_ratio(width - x), sinh_ratio(x))
    } else if discriminant >= -scale {
        ((width - x) / width, x / width)
    } else {
        let omega = (-discriminant).sqrt();
        let denominator = (omega * width).sin();
        if denominator.abs() <= 64.0 * f64::EPSILON {
            return None;
        }
        (
            (omega * (width - x)).sin() / denominator,
            (omega * x).sin() / denominator,
        )
    };

    let factor = (alpha * x).exp() * left_ratio + (alpha * (x - width)).exp() * right_ratio;
    (factor.is_finite() && factor >= 0.0).then_some(factor)
}

/// Expected discount factor to the first barrier-touch time,
/// `E[exp(-r*tau) * 1{tau <= T}]`, for a knock-out rebate paid at hit.
///
/// The discounted survival expansion is
/// `D(t) = sum_n a_n * exp(nu_n*t)`.  Directly integrating each term creates a
/// slowly converging `1/n^2` tail.  Instead, subtract the exponentially small
/// finite-horizon correction from the closed-form infinite-horizon transform:
/// `F(T) = F(infinity) - sum_n a_n*exp(nu_n*T)*(1 + r/nu_n)`.
/// This preserves the exact `1-D(T)` reduction at `r=0` and makes the requested
/// series term count control only an exponentially convergent remainder.
#[allow(clippy::too_many_arguments)]
#[inline]
fn double_touch_pay_at_hit_factor(
    spot: f64,
    lower: f64,
    upper: f64,
    rate: f64,
    dividend_yield: f64,
    vol: f64,
    expiry: f64,
    series_terms: usize,
) -> f64 {
    let vol_sq = vol * vol;
    let width = (upper / lower).ln();
    let x = (spot / lower).ln();

    let carry = (-0.5 * vol).mul_add(vol, rate - dividend_yield);
    let alpha = -carry / vol_sq;
    let c = (-0.5 * carry).mul_add(carry / vol_sq, -rate);
    let amp = (alpha * x).exp();

    let mut survival_disc = 0.0;
    let mut finite_horizon_correction = 0.0;
    for n in 1..=series_terms {
        let w = (n as f64) * PI / width;
        let b_n = 2.0 / width * exp_sin_integral(-alpha, w, 0.0, width);
        let a_n = amp * b_n * (w * x).sin();
        // Ordinary non-negative-rate inputs have nu_n < 0.  For sufficiently
        // negative rates a mode can be non-decaying; the acceleration helper
        // rejects that domain and the direct finite-horizon identity is used.
        let nu_n = c - 0.5 * vol_sq * w * w;
        let exp_nu_t = (nu_n * expiry).exp();
        survival_disc += a_n * exp_nu_t;
        finite_horizon_correction += a_n
            * exp_nu_t
            * if nu_n == 0.0 {
                f64::NAN
            } else {
                1.0 + rate / nu_n
            };
    }

    let df_r = (-rate * expiry).exp();
    // survival_disc = e^{-rT} * P(no touch by T), so it lies in [0, df_r] for
    // any sign of r, and the undiscounted touch probability follows exactly.
    let survival_disc = survival_disc.clamp(0.0, df_r);
    let touch_prob = (1.0 - survival_disc / df_r).clamp(0.0, 1.0);
    // Model-free bounds: on {tau <= T}, e^{-r tau} lies between min(1, e^{-rT})
    // and max(1, e^{-rT}) (the order flips for r < 0, where e^{-r t} grows in
    // t), hence E[e^{-r tau} 1{tau <= T}] is bracketed by those endpoints
    // scaled by P(touch). At r = 0 both bounds collapse to 1 - survival.
    let lo = touch_prob * df_r.min(1.0);
    let hi = touch_prob * df_r.max(1.0);
    let accelerated = eventual_double_touch_discount_factor(carry, vol_sq, width, x, rate)
        .map(|eventual| eventual - finite_horizon_correction)
        .filter(|factor| factor.is_finite());

    // At a singular negative-rate infinite-horizon transform, retain the
    // finite-maturity term-by-term identity.  This path is not needed for
    // ordinary positive/near-zero rates but keeps the routine total over the
    // validated market domain.
    let factor = accelerated.unwrap_or_else(|| {
        let mut integral_term = 0.0;
        for n in 1..=series_terms {
            let w = (n as f64) * PI / width;
            let b_n = 2.0 / width * exp_sin_integral(-alpha, w, 0.0, width);
            let a_n = amp * b_n * (w * x).sin();
            let nu_n = c - 0.5 * vol_sq * w * w;
            integral_term += if nu_n == 0.0 {
                a_n * expiry
            } else {
                a_n * (nu_n * expiry).exp_m1() / nu_n
            };
        }
        1.0 - survival_disc - rate * integral_term
    });
    factor.clamp(lo, hi)
}

impl PricingEngine<DoubleBarrierOption> for DoubleBarrierAnalyticEngine {
    fn price(
        &self,
        instrument: &DoubleBarrierOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
        instrument.validate()?;

        if self.series_terms == 0 {
            return Err(PricingError::InvalidInput(
                "double-barrier series_terms must be > 0".to_string(),
            ));
        }

        let inside =
            market.spot > instrument.lower_barrier && market.spot < instrument.upper_barrier;

        if instrument.expiry <= 0.0 {
            let vanilla = vanilla_payoff(instrument.option_type, market.spot, instrument.strike);
            let price = match instrument.barrier_type {
                DoubleBarrierType::KnockOut => {
                    if inside {
                        vanilla
                    } else {
                        instrument.rebate
                    }
                }
                DoubleBarrierType::KnockIn => {
                    if inside {
                        instrument.rebate
                    } else {
                        vanilla
                    }
                }
            };

            let mut diagnostics = crate::core::Diagnostics::new();
            diagnostics.insert("series_terms", self.series_terms as f64);
            diagnostics.insert("inside_barriers", if inside { 1.0 } else { 0.0 });
            diagnostics.insert("double_knockout_base", 0.0);
            diagnostics.insert("survival_digital", 0.0);

            return Ok(PricingResult {
                price,
                stderr: None,
                greeks: None,
                diagnostics,
            });
        }

        let vol = market.checked_vol_for(instrument.strike, instrument.expiry)?;
        let q = market.effective_dividend_yield(instrument.expiry);

        let vanilla = bs_price_with_dividend(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            q,
            vol,
            instrument.expiry,
        );

        let (double_knockout_base, survival_digital) = if inside {
            (
                double_knock_out_zero_rebate(
                    instrument.option_type,
                    market.spot,
                    instrument.strike,
                    instrument.lower_barrier,
                    instrument.upper_barrier,
                    market.rate,
                    q,
                    vol,
                    instrument.expiry,
                    self.series_terms,
                ),
                double_no_touch_digital_price(
                    market.spot,
                    instrument.lower_barrier,
                    instrument.upper_barrier,
                    market.rate,
                    q,
                    vol,
                    instrument.expiry,
                    self.series_terms,
                ),
            )
        } else {
            (0.0, 0.0)
        };

        // Knock-out rebate is paid at the hit time: weight it by the expected
        // discount factor to the first touch, E[exp(-r*tau) 1{tau <= T}],
        // rather than discounting the touch probability from expiry.
        let knock_out_with_rebate = if inside {
            let pay_at_hit = if instrument.rebate != 0.0 {
                double_touch_pay_at_hit_factor(
                    market.spot,
                    instrument.lower_barrier,
                    instrument.upper_barrier,
                    market.rate,
                    q,
                    vol,
                    instrument.expiry,
                    self.series_terms,
                )
            } else {
                0.0
            };
            double_knockout_base + instrument.rebate * pay_at_hit
        } else {
            // Already outside the corridor at inception: knocked out at t = 0,
            // rebate paid immediately.
            instrument.rebate
        };

        let knock_in_with_rebate = if inside {
            vanilla - double_knockout_base + instrument.rebate * survival_digital
        } else {
            vanilla
        };

        let price = match instrument.barrier_type {
            DoubleBarrierType::KnockOut => knock_out_with_rebate,
            DoubleBarrierType::KnockIn => knock_in_with_rebate,
        }
        .max(0.0);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("series_terms", self.series_terms as f64);
        diagnostics.insert("vol", vol);
        diagnostics.insert("inside_barriers", if inside { 1.0 } else { 0.0 });
        diagnostics.insert("double_knockout_base", double_knockout_base);
        diagnostics.insert("survival_digital", survival_digital);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;
    use crate::core::PricingEngine;
    use crate::instruments::double_barrier::{DoubleBarrierOption, DoubleBarrierType};

    fn haug_market() -> Market {
        Market::builder()
            .spot(100.0)
            .rate(0.10)
            .dividend_yield(0.0)
            .flat_vol(0.25)
            .build()
            .unwrap()
    }

    fn assert_haug_four_decimal_fixture(
        price: f64,
        exact_five_term_price: f64,
        published: f64,
        label: &str,
    ) {
        assert_eq!(
            price, exact_five_term_price,
            "{label}: five-term Ikeda-Kunitomo grid changed"
        );
        // Haug reports four decimal places, so half a unit in the last printed
        // digit is the complete source-precision interval.
        let source_rounding = 0.5e-4;
        assert!(
            (price - published).abs() <= source_rounding,
            "{label}: exact={price:.15}, published={published:.4}, source rounding={source_rounding}"
        );
    }

    #[test]
    fn double_knock_out_call_matches_haug_case_1() {
        let option = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            0.25,
            80.0,
            120.0,
            DoubleBarrierType::KnockOut,
            0.0,
        );
        let market = haug_market();
        let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(5);

        let price = engine.price(&option, &market).unwrap().price;
        let price_20 = DoubleBarrierAnalyticEngine::new()
            .with_series_terms(20)
            .price(&option, &market)
            .unwrap()
            .price;
        assert_haug_four_decimal_fixture(price, 2.638712882539764, 2.6387, "Haug case 1");
        assert_eq!(price_20, price, "series must be converged by five terms");
    }

    #[test]
    fn double_knock_out_call_matches_haug_case_2() {
        let option = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            0.25,
            85.0,
            115.0,
            DoubleBarrierType::KnockOut,
            0.0,
        );
        let market = haug_market();
        let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(5);

        let price = engine.price(&option, &market).unwrap().price;
        let price_20 = DoubleBarrierAnalyticEngine::new()
            .with_series_terms(20)
            .price(&option, &market)
            .unwrap()
            .price;
        assert_haug_four_decimal_fixture(price, 1.3401109441526664, 1.3401, "Haug case 2");
        assert_eq!(price_20, price, "series must be converged by five terms");
    }

    #[test]
    fn double_knock_out_rebate_at_hit_exceeds_pay_at_expiry_value() {
        // High rate, long maturity, tight corridor: the barrier is hit early
        // with near certainty, so a rebate paid at hit is worth close to its
        // undiscounted amount, while pay-at-expiry would cap its value at
        // rebate * exp(-r*T).
        let market = Market::builder()
            .spot(100.0)
            .rate(0.10)
            .dividend_yield(0.0)
            .flat_vol(0.25)
            .build()
            .unwrap();
        let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(20);

        let rebate = 10.0;
        let no_rebate = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            5.0,
            90.0,
            110.0,
            DoubleBarrierType::KnockOut,
            0.0,
        );
        let with_rebate = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            5.0,
            90.0,
            110.0,
            DoubleBarrierType::KnockOut,
            rebate,
        );

        let p0 = engine.price(&no_rebate, &market).unwrap().price;
        let p1 = engine.price(&with_rebate, &market).unwrap().price;
        let rebate_component = p1 - p0;

        assert_eq!(p1, 9.84266426875577, "20-term rebate price changed");
        assert_eq!(
            rebate_component, 9.842664268754929,
            "20-term rebate component changed"
        );

        let pay_at_expiry_cap = rebate * (-0.10_f64 * 5.0).exp();
        assert!(
            rebate_component > pay_at_expiry_cap,
            "rebate-at-hit component {rebate_component} should exceed pay-at-expiry cap {pay_at_expiry_cap}"
        );
        assert!(
            rebate_component <= rebate + 64.0 * f64::EPSILON * rebate,
            "discounted rebate component {rebate_component} cannot exceed undiscounted rebate {rebate}"
        );
    }

    #[test]
    fn nonzero_rebate_matches_independent_scipy_log_space_pde() {
        // Independent SciPy 1.17.1 method-of-lines reference for the complete
        // knock-out contract, including the rebate paid at first touch.  The
        // log-space Black-Scholes PDE uses Dirichlet value 7.5 at both
        // barriers and is advanced with a sparse matrix exponential.  Spot
        // and strike are exact grid nodes.  Richardson-extrapolated values
        // from 500, 1_000, and 2_000 intervals were respectively
        // 7.097463687167115, 7.097463686975762, and 7.097463687057019.
        const SCIPY_PDE_REFERENCE: f64 = 7.097_463_687_0;
        const PDE_DISCRETIZATION_BUDGET: f64 = 3.0e-9;

        let market = Market::builder()
            .spot(100.0)
            .rate(0.047)
            .dividend_yield(0.013)
            .flat_vol(0.29)
            .build()
            .unwrap();
        let option = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            1.4,
            80.0,
            125.0,
            DoubleBarrierType::KnockOut,
            7.5,
        );
        let actual = DoubleBarrierAnalyticEngine::new()
            .with_series_terms(5)
            .price(&option, &market)
            .unwrap()
            .price;
        let binary64_budget = 256.0 * f64::EPSILON * SCIPY_PDE_REFERENCE;
        let tolerance = PDE_DISCRETIZATION_BUDGET + binary64_budget;

        assert!(
            (actual - SCIPY_PDE_REFERENCE).abs() <= tolerance,
            "nonzero-rebate double barrier: actual={actual:.17e}, \
             SciPy PDE={SCIPY_PDE_REFERENCE:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn eventual_touch_transform_resolves_small_positive_discriminant() {
        // The dimensional coefficients are tiny, but their ratio is not: the
        // dimensional discriminant=1e-16 and vol^2=1e-8 imply delta=1. An absolute
        // epsilon threshold would incorrectly take the delta -> 0 limit and
        // return 1 instead of the exact symmetric-corridor value 1/cosh(0.5).
        let factor = eventual_double_touch_discount_factor(0.0, 1.0e-8, 1.0, 0.5, 5.0e-9)
            .expect("finite exit-time transform");
        let expected = 1.0 / 0.5_f64.cosh();
        let roundoff = 32.0 * f64::EPSILON * expected;
        assert!(
            (factor - expected).abs() <= roundoff,
            "small-discriminant transform: factor={factor:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn eventual_touch_transform_survives_dimensional_discriminant_underflow() {
        // Here 2*r*vol^2 = 1e-400 underflows to zero in binary64, while the
        // normalized characteristic discriminant 2*r/vol^2 is exactly one.
        // Evaluating the dimensional expression first used to return the
        // repeated-root value 1 instead of the exact 1/cosh(0.5).
        let factor = eventual_double_touch_discount_factor(0.0, 1.0e-200, 1.0, 0.5, 5.0e-201)
            .expect("finite exit-time transform");
        let expected = 1.0 / 0.5_f64.cosh();
        let roundoff = 32.0 * f64::EPSILON * expected;
        assert!(
            (factor - expected).abs() <= roundoff,
            "underflowed-dimensional transform: factor={factor:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn double_touch_pay_at_hit_factor_handles_negative_rates() {
        // r < 0: e^{-rT} > 1, so the old clamp(touch_prob_disc, 1.0) had an
        // inverted (panicking) bracket. The factor must satisfy the model-free
        // bounds P(touch) <= E[e^{-r tau} 1{tau<=T}] <= e^{-rT} * P(touch).
        let (rate, q, vol, expiry) = (-0.02_f64, 0.0, 0.25, 5.0);
        let factor = double_touch_pay_at_hit_factor(100.0, 90.0, 110.0, rate, q, vol, expiry, 20);

        let df_r = (-rate * expiry).exp();
        // Undiscounted touch probability from the r = 0 series.
        let touch_prob_r0 =
            1.0 - double_no_touch_digital_price(100.0, 90.0, 110.0, 0.0, q - rate, vol, expiry, 20);
        assert_eq!(factor, 1.003223639428337, "20-term hit factor changed");
        assert_eq!(touch_prob_r0, 1.0, "20-term touch probability changed");
        assert!(factor.is_finite());
        let roundoff = 64.0 * f64::EPSILON * df_r * touch_prob_r0;
        assert!(
            factor >= touch_prob_r0 - roundoff && factor <= df_r * touch_prob_r0 + roundoff,
            "factor {factor} outside [{touch_prob_r0}, {}]",
            df_r * touch_prob_r0
        );
    }

    #[test]
    fn double_knock_out_with_rebate_does_not_panic_for_negative_rates() {
        let market = Market::builder()
            .spot(100.0)
            .rate(-0.02)
            .dividend_yield(0.0)
            .flat_vol(0.25)
            .build()
            .unwrap();
        let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(20);
        let option = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            5.0,
            90.0,
            110.0,
            DoubleBarrierType::KnockOut,
            10.0,
        );

        let price = engine.price(&option, &market).unwrap().price;
        assert_eq!(price, 10.032236394283366, "negative-rate price changed");
        assert!(price.is_finite() && price >= 0.0);
        // With r < 0 the rebate component may exceed the undiscounted rebate
        // (paid-at-hit cash is reinvested at a negative rate relative to par),
        // but never beyond rebate * e^{-rT}.
        let cap = 10.0 * (0.02_f64 * 5.0).exp() + 10.0;
        assert!(price <= cap, "price {price} above cap {cap}");
    }

    #[test]
    fn pay_at_hit_factor_reduces_to_touch_probability_at_zero_rate() {
        let (q, vol, expiry) = (0.0, 0.25, 2.0);
        let factor = double_touch_pay_at_hit_factor(100.0, 85.0, 115.0, 0.0, q, vol, expiry, 20);
        let touch_prob =
            1.0 - double_no_touch_digital_price(100.0, 85.0, 115.0, 0.0, q, vol, expiry, 20);
        assert_eq!(
            factor, 0.998534690215092,
            "20-term zero-rate factor changed"
        );
        assert_eq!(factor, touch_prob, "r=0 reduction must be exact");
    }

    #[test]
    fn double_knock_out_call_matches_haug_case_3() {
        let option = DoubleBarrierOption::new(
            OptionType::Call,
            100.0,
            0.25,
            90.0,
            110.0,
            DoubleBarrierType::KnockOut,
            0.0,
        );
        let market = haug_market();
        let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(5);

        let price = engine.price(&option, &market).unwrap().price;
        let price_20 = DoubleBarrierAnalyticEngine::new()
            .with_series_terms(20)
            .price(&option, &market)
            .unwrap()
            .price;
        assert_haug_four_decimal_fixture(price, 0.3098238680345293, 0.3098, "Haug case 3");
        assert_eq!(price_20, price, "series must be converged by five terms");
    }
}
