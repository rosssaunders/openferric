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
    let d1 =
        ((spot / strike).ln() + (0.5 * vol).mul_add(vol, rate - dividend_yield) * expiry)
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

impl PricingEngine<DoubleBarrierOption> for DoubleBarrierAnalyticEngine {
    fn price(
        &self,
        instrument: &DoubleBarrierOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
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

        let vol = market.vol_for(instrument.strike, instrument.expiry);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let df_r = (-market.rate * instrument.expiry).exp();
        let vanilla = bs_price_with_dividend(
            instrument.option_type,
            market.spot,
            instrument.strike,
            market.rate,
            market.dividend_yield,
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
                    market.dividend_yield,
                    vol,
                    instrument.expiry,
                    self.series_terms,
                ),
                double_no_touch_digital_price(
                    market.spot,
                    instrument.lower_barrier,
                    instrument.upper_barrier,
                    market.rate,
                    market.dividend_yield,
                    vol,
                    instrument.expiry,
                    self.series_terms,
                ),
            )
        } else {
            (0.0, 0.0)
        };

        let knock_out_with_rebate = if inside {
            double_knockout_base + instrument.rebate * (df_r - survival_digital).max(0.0)
        } else {
            instrument.rebate * df_r
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
    use approx::assert_relative_eq;

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
        assert_relative_eq!(price, 2.6387, epsilon = 2e-4);
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
        assert_relative_eq!(price, 1.3401, epsilon = 2e-4);
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
        assert_relative_eq!(price, 0.3098, epsilon = 2e-4);
    }
}
