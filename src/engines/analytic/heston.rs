use std::f64::consts::PI;
use std::sync::LazyLock;

use num_complex::Complex64;

use crate::core::{ExerciseStyle, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::vanilla::VanillaOption;
use crate::market::Market;

/// Analytic Heston engine for European vanilla options.
#[derive(Debug, Clone, Copy)]
pub struct HestonEngine {
    /// Initial variance.
    pub v0: f64,
    /// Mean reversion speed.
    pub kappa: f64,
    /// Long-run variance.
    pub theta: f64,
    /// Volatility of variance.
    pub sigma_v: f64,
    /// Correlation between spot and variance Brownian motions.
    pub rho: f64,
}

impl HestonEngine {
    /// Creates a Heston analytic engine.
    pub fn new(v0: f64, kappa: f64, theta: f64, sigma_v: f64, rho: f64) -> Self {
        Self {
            v0,
            kappa,
            theta,
            sigma_v,
            rho,
        }
    }

    fn validate_params(&self) -> Result<(), PricingError> {
        if self.v0 < 0.0 {
            return Err(PricingError::InvalidInput(
                "heston v0 must be >= 0".to_string(),
            ));
        }
        if self.kappa <= 0.0 {
            return Err(PricingError::InvalidInput(
                "heston kappa must be > 0".to_string(),
            ));
        }
        if self.theta < 0.0 {
            return Err(PricingError::InvalidInput(
                "heston theta must be >= 0".to_string(),
            ));
        }
        if self.sigma_v <= 0.0 {
            return Err(PricingError::InvalidInput(
                "heston sigma_v must be > 0".to_string(),
            ));
        }
        if self.rho <= -1.0 || self.rho >= 1.0 {
            return Err(PricingError::InvalidInput(
                "heston rho must be in (-1, 1)".to_string(),
            ));
        }
        Ok(())
    }

    // Gatheral log-formulation characteristic function for log spot.
    fn characteristic_fn(
        &self,
        u: Complex64,
        ln_spot: f64,
        t: f64,
        rate: f64,
        dividend_yield: f64,
    ) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let one = Complex64::new(1.0, 0.0);

        let sigma2 = self.sigma_v * self.sigma_v;
        let inv_sigma2 = 1.0 / sigma2;
        let iu = i * u;
        let beta = Complex64::new(self.kappa, 0.0) - self.rho * self.sigma_v * iu;

        let mut d = (beta * beta + sigma2 * (u * u + iu)).sqrt();
        if d.re < 0.0 {
            d = -d;
        }

        let g = (beta - d) / (beta + d);
        let exp_neg_dt = (-d * t).exp();
        let log_term = ((one - g * exp_neg_dt) / (one - g)).ln();

        let a_term = Complex64::new(self.kappa * self.theta * inv_sigma2, 0.0);
        let c = iu * (ln_spot + (rate - dividend_yield) * t)
            + a_term * ((beta - d) * t - 2.0 * log_term);
        let d_term = ((beta - d) * inv_sigma2) * ((one - exp_neg_dt) / (one - g * exp_neg_dt));

        (c + d_term * self.v0).exp()
    }

    fn call_price_gatheral(
        &self,
        spot: f64,
        strike: f64,
        t: f64,
        rate: f64,
        dividend_yield: f64,
    ) -> Result<(f64, f64), PricingError> {
        let i = Complex64::new(0.0, 1.0);
        let half_i = Complex64::new(0.0, 0.5);
        let ln_spot = spot.ln();
        let df_r = (-rate * t).exp();
        let forward = spot * ((rate - dividend_yield) * t).exp();
        let ln_forward = forward.ln();
        let log_moneyness = (forward / strike).ln();

        // Gatheral/Lewis log-strike integral:
        // C = e^{-rT} * (F - sqrt(F*K)/pi * int_0^inf Re[e^{i u ln(F/K)} psi(u-i/2)/(u^2+1/4)] du)
        // where psi is the characteristic function of ln(S_T/F).
        let (nodes, _weights) = gauss_laguerre_32();
        let adjusted_weights = &*GL32_ADJUSTED_WEIGHTS;
        let mut integral = 0.0;

        // Closure that computes a single integrand value for quadrature node j.
        // Uses real arithmetic for the denominator (u is real, so u*u is x*x).
        let compute_integrand = |j: usize| {
            let x = nodes[j];
            let u = Complex64::new(x, 0.0);
            let shifted = u - half_i;
            let phi = self.characteristic_fn(shifted, ln_spot, t, rate, dividend_yield);
            let psi = phi / (i * shifted * ln_forward).exp();
            let numerator = (i * u * log_moneyness).exp() * psi;
            // Avoid complex multiplication: u is real so u*u = x*x (real scalar).
            let denom_re = x * x + 0.25;
            (numerator / Complex64::new(denom_re, 0.0)).re
        };

        // Unrolled loop: process 4 nodes at a time for better instruction-level parallelism.
        let mut idx = 0;
        while idx + 4 <= 32 {
            integral += adjusted_weights[idx] * compute_integrand(idx);
            integral += adjusted_weights[idx + 1] * compute_integrand(idx + 1);
            integral += adjusted_weights[idx + 2] * compute_integrand(idx + 2);
            integral += adjusted_weights[idx + 3] * compute_integrand(idx + 3);
            idx += 4;
        }
        // Handle any remainder (32 is divisible by 4, but this guards future changes).
        while idx < 32 {
            integral += adjusted_weights[idx] * compute_integrand(idx);
            idx += 1;
        }

        let call = df_r * (forward - (forward * strike).sqrt() * integral / PI);
        if !call.is_finite() || !integral.is_finite() {
            return Err(PricingError::NumericalError(
                "heston call integral returned non-finite value".to_string(),
            ));
        }

        Ok((call, integral))
    }
}

impl PricingEngine<VanillaOption> for HestonEngine {
    fn price(
        &self,
        instrument: &VanillaOption,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;
        self.validate_params()?;

        if !matches!(instrument.exercise, ExerciseStyle::European) {
            return Err(PricingError::InvalidInput(
                "HestonEngine supports European exercise only".to_string(),
            ));
        }

        if instrument.expiry <= 0.0 {
            let intrinsic = match instrument.option_type {
                OptionType::Call => (market.spot - instrument.strike).max(0.0),
                OptionType::Put => (instrument.strike - market.spot).max(0.0),
            };
            return Ok(PricingResult {
                price: intrinsic,
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let t = instrument.expiry;
        let df_r = (-market.rate * t).exp();
        let df_q = (-market.dividend_yield * t).exp();
        let (call, integral) = self.call_price_gatheral(
            market.spot,
            instrument.strike,
            t,
            market.rate,
            market.dividend_yield,
        )?;

        let price = match instrument.option_type {
            OptionType::Call => call,
            OptionType::Put => call - market.spot * df_q + instrument.strike * df_r,
        };

        if !price.is_finite() {
            return Err(PricingError::NumericalError(
                "heston option price is non-finite".to_string(),
            ));
        }

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert_key(crate::core::DiagKey::Integral, integral);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

fn gauss_laguerre_32() -> (&'static [f64; 32], &'static [f64; 32]) {
    (&GL32_NODES, &GL32_WEIGHTS)
}

/// Pre-computed weights * exp(node) for Gauss-Laguerre quadrature.
/// Computed once on first access, eliminates 32 exp() calls per Heston pricing invocation.
static GL32_ADJUSTED_WEIGHTS: LazyLock<[f64; 32]> = LazyLock::new(|| {
    let mut adj = [0.0_f64; 32];
    for i in 0..32 {
        adj[i] = GL32_WEIGHTS[i] * GL32_NODES[i].exp();
    }
    adj
});

const GL32_NODES: [f64; 32] = [
    4.448_936_583_326_695e-2,
    2.345_261_095_196_18e-1,
    5.768_846_293_018_863e-1,
    1.072_448_753_817_818_2,
    1.722_408_776_444_645_9,
    2.528_336_706_425_794,
    3.492_213_273_021_993_5,
    4.616_456_769_749_767,
    5.903_958_504_174_245,
    7.358_126_733_186_242,
    8.982_940_924_212_595,
    1.078_301_863_253_997_2e1,
    1.276_369_798_674_272_5e1,
    1.493_113_975_552_255_8e1,
    1.729_245_433_671_531_6e1,
    1.985_586_094_033_605_4e1,
    2.263_088_901_319_677_5e1,
    2.562_863_602_245_924_7e1,
    2.886_210_181_632_347_4e1,
    3.234_662_915_396_473_4e1,
    3.610_049_480_575_197e1,
    4.014_571_977_153_944e1,
    4.450_920_799_575_494e1,
    4.922_439_498_730_864e1,
    5.433_372_133_339_691e1,
    5.989_250_916_213_402e1,
    6.597_537_728_793_504_6e1,
    7.268_762_809_066_271e1,
    8.018_744_697_791_352e1,
    8.873_534_041_789_24e1,
    9.882_954_286_828_397e1,
    1.117_513_980_979_377e2,
];

const GL32_WEIGHTS: [f64; 32] = [
    1.092_183_419_523_906_5e-1,
    2.104_431_079_388_177_6e-1,
    2.352_132_296_698_383_8e-1,
    1.959_033_359_728_814_8e-1,
    1.299_837_862_860_71e-1,
    7.057_862_386_571_789e-2,
    3.176_091_250_917_504_5e-2,
    1.191_821_483_483_855_4e-2,
    3.738_816_294_611_524e-3,
    9.808_033_066_149_506e-4,
    2.148_649_188_013_647_7e-4,
    3.920_341_967_987_943_5e-5,
    5.934_541_612_868_633e-6,
    7.416_404_578_667_559e-7,
    7.604_567_879_120_781e-8,
    6.350_602_226_625_813e-9,
    4.281_382_971_040_925e-10,
    2.305_899_491_891_339_3e-11,
    9.799_379_288_727_107e-13,
    3.237_801_657_729_274_7e-14,
    8.171_823_443_420_743e-16,
    1.542_133_833_393_825_3e-17,
    2.119_792_290_163_613_1e-19,
    2.054_429_673_788_036_3e-21,
    1.346_982_586_637_393_5e-23,
    5.661_294_130_397_355e-26,
    1.418_560_545_463_052e-28,
    1.913_375_494_454_213_4e-31,
    1.192_248_760_098_223_3e-34,
    2.671_511_219_240_121e-38,
    1.338_616_942_106_27e-42,
    4.510_536_193_898_977e-48,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_laguerre_32_integrates_weighted_polynomial() {
        let (nodes, weights) = gauss_laguerre_32();

        // Integral of e^{-x} x^2 on [0, inf) equals 2.
        let approx = (0..32)
            .map(|i| weights[i] * nodes[i] * nodes[i])
            .sum::<f64>();
        assert!((approx - 2.0).abs() < 1e-12);
    }
}
