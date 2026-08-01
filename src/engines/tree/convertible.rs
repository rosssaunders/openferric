//! Module `engines::tree::convertible`.
//!
//! Implements convertible abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 13, Cox-Ross-Rubinstein (1979), and backward-induction recursions around Eq. (13.10).
//!
//! Key types and purpose: `ConvertibleBinomialEngine` define the core data contracts for this module.
//!
//! Numerical considerations: convergence is first- to second-order in time-step count depending on tree parameterization; deep ITM/OTM nodes may need larger depth.
//!
//! When to use: use trees for early-exercise intuition and lattice diagnostics; use analytic formulas for plain vanillas and Monte Carlo/PDE for richer dynamics.
use crate::core::{Greeks, PricingEngine, PricingError, PricingResult};
use crate::instruments::convertible::ConvertibleBond;
use crate::market::Market;

/// CRR-style binomial engine for convertible bonds.
///
/// # Model conventions
///
/// - **Always callable / always puttable:** `call_price` and `put_price` are
///   flat levels modeled as exercisable at *every* interior time step; there
///   is no call-protection (hard/soft no-call) schedule. In particular, a
///   `call_price` below `face_value` means the issuer may redeem below par at
///   any time, so a straight (zero-conversion-ratio) bond collapses to the
///   present value of the call price rather than of face.
/// - **Maturity redemption:** at maturity the bond redeems at `face_value`
///   (best of redemption, conversion, and any put floor). The issuer call
///   does not cap the terminal payoff: a call right is meaningless once the
///   bond has matured.
/// - **Flat credit spread:** `credit_spread` is applied uniformly when
///   discounting the hold (continuation) value. This is a simplification
///   relative to Tsiveriotis-Fernandes, which splits the node value into
///   equity and debt components discounted at different rates; bond-like
///   convertibles will therefore price differently from a TF model.
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
    // Standard convertible node treatment: the issuer call caps only the hold
    // (continuation) value, after which the holder may still convert or put.
    // value = max(conversion, put, min(hold, call)). This preserves forced
    // conversion: when conversion value exceeds the call price, the holder
    // converts and receives the conversion value, not the call price.
    let mut value = match call_price {
        Some(call) => continuation.min(call),
        None => continuation,
    };
    value = value.max(conversion_value);
    if let Some(put) = put_price {
        value = value.max(put);
    }
    value
}

impl PricingEngine<ConvertibleBond> for ConvertibleBinomialEngine {
    fn price(
        &self,
        instrument: &ConvertibleBond,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;
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
            // A matured bond redeems at face value: the issuer call right is
            // meaningless at expiry and must not cap redemption. The holder
            // still takes the best of redemption, immediate conversion, and
            // the put floor.
            let mut price = instrument.face_value.max(conversion_value);
            if let Some(put) = instrument.put_price {
                price = price.max(put);
            }
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
        let vol = market.checked_vol_for(vol_strike, instrument.maturity)?;

        let dt = instrument.maturity / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let effective_dividend_yield = market.effective_dividend_yield(instrument.maturity);
        let growth = ((market.rate - effective_dividend_yield) * dt).exp();
        let p = (growth - d) / (u - d);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(PricingError::NumericalError(
                "risk-neutral probability is outside [0, 1]".to_string(),
            ));
        }
        let disc = (-(market.rate + self.credit_spread) * dt).exp();
        let coupon = instrument.face_value * instrument.coupon_rate * dt;

        // Multiplicative recurrence: spot * u^j * d^(n-j) = spot * d^n * (u/d)^j
        let ratio = u / d;
        let one_minus_p = 1.0 - p;

        let mut values = vec![0.0_f64; self.steps + 1];
        {
            let mut st = market.spot * d.powi(self.steps as i32);
            for value in values.iter_mut() {
                // Terminal payoff: redemption at face vs conversion (plus the
                // put floor). The issuer call is not applied at maturity — the
                // bond redeems at face, so a call price below face cannot cap
                // the terminal payoff. The always-exercisable call still binds
                // at every interior step, so this affects the price only at
                // O(dt). Note the final coupon is paid regardless and enters
                // through the continuation value one step earlier.
                let redemption = instrument.face_value;
                let conversion = instrument.conversion_ratio * st;
                *value =
                    apply_embedded_features(redemption, conversion, instrument.put_price, None);
                st *= ratio;
            }
        }

        let mut delta_up = if self.steps == 1 { values[1] } else { 0.0 };
        let mut delta_down = if self.steps == 1 { values[0] } else { 0.0 };

        let mut base = market.spot * d.powi((self.steps - 1) as i32);
        for i in (0..self.steps).rev() {
            let mut st = base;
            for j in 0..=i {
                let continuation = disc * (p * values[j + 1] + one_minus_p * values[j] + coupon);
                let conversion = instrument.conversion_ratio * st;
                values[j] = apply_embedded_features(
                    continuation,
                    conversion,
                    instrument.put_price,
                    instrument.call_price,
                );
                st *= ratio;
            }

            if i == 1 {
                delta_down = values[0];
                delta_up = values[1];
            }
            base *= u;
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

    #[test]
    fn apply_embedded_features_preserves_forced_conversion() {
        // Conversion value above the call price: the issuer calls, the holder
        // is forced to convert and receives the conversion value, not the
        // (lower) call price.
        let value = apply_embedded_features(130.0, 120.0, None, Some(110.0));
        assert_eq!(value, 120.0);

        // Put floor still applies on top of the capped hold value.
        let value = apply_embedded_features(100.0, 90.0, Some(105.0), Some(95.0));
        assert_eq!(value, 105.0);

        // Without forced conversion the call caps the hold value.
        let value = apply_embedded_features(130.0, 80.0, None, Some(110.0));
        assert_eq!(value, 110.0);
    }

    #[test]
    fn expired_bond_with_call_below_face_redeems_at_face() {
        // A matured bond redeems at face; an issuer call at 90 < face 100 is
        // meaningless at expiry and must not cap redemption. Previously this
        // returned min(face, call) = 90.
        let market = ql_test_market();
        let engine = ConvertibleBinomialEngine::new(0.0);

        let bond = ConvertibleBond::new(100.0, 0.0, 0.0, 0.0, Some(90.0), None);
        let price = engine.price(&bond, &market).unwrap().price;
        assert_eq!(price, 100.0, "matured bond must redeem at face");
    }

    #[test]
    fn expired_bond_takes_best_of_face_conversion_and_put() {
        let market = ql_test_market(); // spot = 100
        let engine = ConvertibleBinomialEngine::new(0.0);

        // Conversion value 1.5 * 100 = 150 dominates face 100.
        let converting = ConvertibleBond::new(100.0, 0.0, 0.0, 1.5, Some(90.0), None);
        assert_eq!(engine.price(&converting, &market).unwrap().price, 150.0);

        // Put floor 120 dominates face 100 when conversion is worthless.
        let puttable = ConvertibleBond::new(100.0, 0.0, 0.0, 0.0, None, Some(120.0));
        assert_eq!(engine.price(&puttable, &market).unwrap().price, 120.0);
    }

    #[test]
    fn deep_itm_callable_convertible_is_worth_at_least_conversion_value() {
        // Deep in-the-money conversion with a low call price: forced conversion
        // means the bond must be worth at least its immediate conversion value,
        // which exceeds the call price.
        let market = Market::builder()
            .spot(150.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.20)
            .build()
            .unwrap();
        let engine = ConvertibleBinomialEngine::new(0.03).with_steps(400);

        // conversion ratio 1.0 -> conversion value = 150 > call price = 110
        let bond = ConvertibleBond::new(100.0, 0.05, 5.0, 1.0, Some(110.0), None);
        let price = engine.price(&bond, &market).unwrap().price;
        let conversion_value = bond.conversion_ratio * market.spot;

        assert!(
            price >= conversion_value - 1e-9,
            "forced conversion lost: price {price} < conversion value {conversion_value}"
        );
        assert!(price > 110.0, "price {price} capped at call price");
    }
}
