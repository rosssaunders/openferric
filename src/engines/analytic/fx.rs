use crate::core::{Greeks, OptionType, PricingEngine, PricingError, PricingResult};
use crate::instruments::fx::FxOption;
use crate::market::Market;
use crate::math::{normal_cdf, normal_pdf};

/// FX Greeks for Garman-Kohlhagen options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FxGreeks {
    /// First derivative to spot FX.
    pub delta: f64,
    /// Second derivative to spot FX.
    pub gamma: f64,
    /// First derivative to volatility.
    pub vega: f64,
    /// First derivative to time.
    pub theta: f64,
    /// First derivative to domestic rate.
    pub rho_domestic: f64,
    /// First derivative to foreign rate.
    pub rho_foreign: f64,
}

/// Analytic Garman-Kohlhagen engine for European FX options.
#[derive(Debug, Clone, Default)]
pub struct GarmanKohlhagenEngine;

impl GarmanKohlhagenEngine {
    /// Creates a Garman-Kohlhagen engine.
    pub fn new() -> Self {
        Self
    }
}

fn intrinsic(option_type: OptionType, spot: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    }
}

fn garman_kohlhagen_price_greeks(option: &FxOption) -> (f64, FxGreeks, f64, f64) {
    let sqrt_t = option.maturity.sqrt();
    let sig_sqrt_t = option.vol * sqrt_t;
    let d1 = ((option.spot_fx / option.strike_fx).ln()
        + (option.domestic_rate - option.foreign_rate + 0.5 * option.vol * option.vol)
            * option.maturity)
        / sig_sqrt_t;
    let d2 = d1 - sig_sqrt_t;

    let df_d = (-option.domestic_rate * option.maturity).exp();
    let df_f = (-option.foreign_rate * option.maturity).exp();

    let price = match option.option_type {
        OptionType::Call => {
            option.spot_fx * df_f * normal_cdf(d1) - option.strike_fx * df_d * normal_cdf(d2)
        }
        OptionType::Put => {
            option.strike_fx * df_d * normal_cdf(-d2) - option.spot_fx * df_f * normal_cdf(-d1)
        }
    };

    let delta = match option.option_type {
        OptionType::Call => df_f * normal_cdf(d1),
        OptionType::Put => df_f * (normal_cdf(d1) - 1.0),
    };
    let gamma = df_f * normal_pdf(d1) / (option.spot_fx * option.vol * sqrt_t);
    let vega = option.spot_fx * df_f * normal_pdf(d1) * sqrt_t;

    let theta = match option.option_type {
        OptionType::Call => {
            -option.spot_fx * df_f * normal_pdf(d1) * option.vol / (2.0 * sqrt_t)
                + option.foreign_rate * option.spot_fx * df_f * normal_cdf(d1)
                - option.domestic_rate * option.strike_fx * df_d * normal_cdf(d2)
        }
        OptionType::Put => {
            -option.spot_fx * df_f * normal_pdf(d1) * option.vol / (2.0 * sqrt_t)
                - option.foreign_rate * option.spot_fx * df_f * normal_cdf(-d1)
                + option.domestic_rate * option.strike_fx * df_d * normal_cdf(-d2)
        }
    };

    let rho_domestic = match option.option_type {
        OptionType::Call => option.strike_fx * option.maturity * df_d * normal_cdf(d2),
        OptionType::Put => -option.strike_fx * option.maturity * df_d * normal_cdf(-d2),
    };
    let rho_foreign = match option.option_type {
        OptionType::Call => -option.spot_fx * option.maturity * df_f * normal_cdf(d1),
        OptionType::Put => option.spot_fx * option.maturity * df_f * normal_cdf(-d1),
    };

    (
        price,
        FxGreeks {
            delta,
            gamma,
            vega,
            theta,
            rho_domestic,
            rho_foreign,
        },
        d1,
        d2,
    )
}

impl PricingEngine<FxOption> for GarmanKohlhagenEngine {
    fn price(
        &self,
        instrument: &FxOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.maturity <= 0.0 {
            return Ok(PricingResult {
                price: intrinsic(
                    instrument.option_type,
                    instrument.spot_fx,
                    instrument.strike_fx,
                ),
                stderr: None,
                greeks: Some(Greeks {
                    delta: 0.0,
                    gamma: 0.0,
                    vega: 0.0,
                    theta: 0.0,
                    rho: 0.0,
                }),
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let (price, fx_greeks, d1, d2) = garman_kohlhagen_price_greeks(instrument);

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("d1", d1);
        diagnostics.insert("d2", d2);
        diagnostics.insert("rho_domestic", fx_greeks.rho_domestic);
        diagnostics.insert("rho_foreign", fx_greeks.rho_foreign);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: Some(Greeks {
                delta: fx_greeks.delta,
                gamma: fx_greeks.gamma,
                vega: fx_greeks.vega,
                theta: fx_greeks.theta,
                rho: fx_greeks.rho_domestic,
            }),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn dummy_market() -> Market {
        Market::builder()
            .spot(1.0)
            .rate(0.01)
            .dividend_yield(0.0)
            .flat_vol(0.2)
            .build()
            .unwrap()
    }

    #[test]
    fn eurusd_call_matches_reference() {
        let option = FxOption::new(OptionType::Call, 0.05, 0.03, 1.25, 1.30, 0.10, 0.50);
        let price = GarmanKohlhagenEngine::new()
            .price(&option, &dummy_market())
            .unwrap()
            .price;

        assert_relative_eq!(price, 0.0199538, epsilon = 2e-6);
    }

    #[test]
    fn fx_put_call_parity_holds() {
        let call = FxOption::new(OptionType::Call, 0.05, 0.03, 1.25, 1.30, 0.10, 0.50);
        let put = FxOption::new(OptionType::Put, 0.05, 0.03, 1.25, 1.30, 0.10, 0.50);
        let engine = GarmanKohlhagenEngine::new();
        let market = dummy_market();

        let c = engine.price(&call, &market).unwrap().price;
        let p = engine.price(&put, &market).unwrap().price;
        let rhs = call.spot_fx * (-call.foreign_rate * call.maturity).exp()
            - call.strike_fx * (-call.domestic_rate * call.maturity).exp();

        assert_relative_eq!(c - p, rhs, epsilon = 1e-8);
    }

    #[test]
    fn rho_foreign_matches_finite_difference() {
        let option = FxOption::new(OptionType::Call, 0.05, 0.03, 1.25, 1.30, 0.10, 0.50);
        let engine = GarmanKohlhagenEngine::new();
        let market = dummy_market();
        let eps = 1.0e-5;

        let rho_foreign = *engine
            .price(&option, &market)
            .unwrap()
            .diagnostics
            .get("rho_foreign")
            .unwrap();

        let up = FxOption::new(
            option.option_type,
            option.domestic_rate,
            option.foreign_rate + eps,
            option.spot_fx,
            option.strike_fx,
            option.vol,
            option.maturity,
        );
        let dn = FxOption::new(
            option.option_type,
            option.domestic_rate,
            option.foreign_rate - eps,
            option.spot_fx,
            option.strike_fx,
            option.vol,
            option.maturity,
        );

        let p_up = engine.price(&up, &market).unwrap().price;
        let p_dn = engine.price(&dn, &market).unwrap().price;
        let fd = (p_up - p_dn) / (2.0 * eps);

        assert_relative_eq!(rho_foreign, fd, epsilon = 2e-6);
    }
}
