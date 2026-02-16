
use crate::core::{Greeks, PricingEngine, PricingError, PricingResult};
use crate::instruments::convertible::ConvertibleBond;
use crate::market::Market;

/// CRR-style binomial engine for convertible bonds.
#[derive(Debug, Clone)]
pub struct ConvertibleBinomialEngine {
    /// Number of time steps.
    pub steps: usize,
    /// Constant credit spread applied to hold-value discounting.
    pub credit_spread: f64,
}

impl Default for ConvertibleBinomialEngine {
    fn default() -> Self {
        Self {
            steps: 200,
            credit_spread: 0.0,
        }
    }
}

impl ConvertibleBinomialEngine {
    /// Creates an engine with the provided credit spread and default steps.
    pub fn new(credit_spread: f64) -> Self {
        Self {
            credit_spread,
            ..Self::default()
        }
    }

    /// Sets the number of tree steps.
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
}

fn apply_embedded_features(
    continuation: f64,
    conversion_value: f64,
    put_price: Option<f64>,
    call_price: Option<f64>,
) -> f64 {
    let mut value = continuation.max(conversion_value);
    if let Some(put) = put_price {
        value = value.max(put);
    }
    if let Some(call) = call_price {
        value = value.min(call);
    }
    value
}

impl PricingEngine<ConvertibleBond> for ConvertibleBinomialEngine {
    fn price(
        &self,
        instrument: &ConvertibleBond,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if self.steps == 0 {
            return Err(PricingError::InvalidInput(
                "convertible binomial steps must be > 0".to_string(),
            ));
        }
        if self.credit_spread < 0.0 {
            return Err(PricingError::InvalidInput(
                "convertible credit_spread must be >= 0".to_string(),
            ));
        }

        let conversion_value = instrument.conversion_ratio * market.spot;
        if instrument.maturity <= 0.0 {
            let redemption = instrument.face_value;
            let price = apply_embedded_features(
                redemption,
                conversion_value,
                instrument.put_price,
                instrument.call_price,
            );
            let mut diagnostics = crate::core::Diagnostics::new();
            diagnostics.insert("npv", price);
            diagnostics.insert("conversion_value", conversion_value);
            diagnostics.insert("delta", 0.0);
            diagnostics.insert("num_steps", self.steps as f64);
            diagnostics.insert("credit_spread", self.credit_spread);

            return Ok(PricingResult {
                price,
                stderr: None,
                greeks: Some(Greeks {
                    delta: 0.0,
                    gamma: 0.0,
                    vega: 0.0,
                    theta: 0.0,
                    rho: 0.0,
                }),
                diagnostics,
            });
        }

        let vol_strike = if instrument.conversion_ratio > 0.0 {
            instrument.face_value / instrument.conversion_ratio
        } else {
            market.spot
        };
        let vol = market.vol_for(vol_strike, instrument.maturity);
        if vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "market volatility must be > 0".to_string(),
            ));
        }

        let dt = instrument.maturity / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let growth = ((market.rate - market.dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-(market.rate + self.credit_spread) * dt).exp();
        let coupon = instrument.face_value * instrument.coupon_rate * dt;

        let mut values = vec![0.0_f64; self.steps + 1];
        for (j, value) in values.iter_mut().enumerate() {
            let st = market.spot * u.powf(j as f64) * d.powf((self.steps - j) as f64);
            let continuation = instrument.face_value;
            let conversion = instrument.conversion_ratio * st;
            *value = apply_embedded_features(
                continuation,
                conversion,
                instrument.put_price,
                instrument.call_price,
            );
        }

        let mut delta_up = if self.steps == 1 { values[1] } else { 0.0 };
        let mut delta_down = if self.steps == 1 { values[0] } else { 0.0 };

        for i in (0..self.steps).rev() {
            for j in 0..=i {
                let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j] + coupon);
                let st = market.spot * u.powf(j as f64) * d.powf((i - j) as f64);
                let conversion = instrument.conversion_ratio * st;
                values[j] = apply_embedded_features(
                    continuation,
                    conversion,
                    instrument.put_price,
                    instrument.call_price,
                );
            }

            if i == 1 {
                delta_down = values[0];
                delta_up = values[1];
            }
        }

        let s_up = market.spot * u;
        let s_down = market.spot * d;
        let delta = if (s_up - s_down).abs() > 1.0e-14 {
            (delta_up - delta_down) / (s_up - s_down)
        } else {
            0.0
        };

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("npv", values[0]);
        diagnostics.insert("conversion_value", conversion_value);
        diagnostics.insert("delta", delta);
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("vol", vol);
        diagnostics.insert("credit_spread", self.credit_spread);

        Ok(PricingResult {
            price: values[0],
            stderr: None,
            greeks: Some(Greeks {
                delta,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
            }),
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PricingEngine;
    use crate::instruments::convertible::ConvertibleBond;

    fn ql_test_market() -> Market {
        Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.20)
            .build()
            .unwrap()
    }

    #[test]
    fn convertible_no_call_dominates_straight_and_conversion_value() {
        let market = ql_test_market();
        let engine = ConvertibleBinomialEngine::new(0.03).with_steps(600);

        let no_call = ConvertibleBond::new(100.0, 0.05, 10.0, 1.0, None, None);
        let straight = ConvertibleBond::new(100.0, 0.05, 10.0, 0.0, None, None);

        let no_call_price = engine.price(&no_call, &market).unwrap().price;
        let straight_price = engine.price(&straight, &market).unwrap().price;
        let conversion_value = no_call.conversion_ratio * market.spot;

        assert!(no_call_price >= straight_price);
        assert!(no_call_price >= conversion_value);
    }

    #[test]
    fn callable_convertible_caps_upside_vs_non_callable() {
        let market = ql_test_market();
        let engine = ConvertibleBinomialEngine::new(0.03).with_steps(600);

        let no_call = ConvertibleBond::new(100.0, 0.05, 10.0, 1.0, None, None);
        let with_call = ConvertibleBond::new(100.0, 0.05, 10.0, 1.0, Some(110.0), None);

        let no_call_price = engine.price(&no_call, &market).unwrap().price;
        let call_price = engine.price(&with_call, &market).unwrap().price;

        assert!(call_price <= no_call_price);
    }
}
