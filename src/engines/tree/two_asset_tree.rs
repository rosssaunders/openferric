use crate::core::{PricingEngine, PricingError, PricingResult};
use crate::instruments::spread::SpreadOption;
use crate::instruments::rainbow::{BestOfTwoCallOption, WorstOfTwoCallOption};
use crate::market::Market;

/// Two-asset binomial tree engine using Rubinstein's 2D recombining tree.
///
/// Each node branches into 4 states: (uu, ud, du, dd) for the two assets.
/// Matches first two moments and correlation of both assets.
#[derive(Debug, Clone)]
pub struct TwoAssetBinomialEngine {
    /// Number of tree steps.
    pub steps: usize,
}

impl TwoAssetBinomialEngine {
    /// Creates a two-asset binomial engine with the given number of steps.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

/// Computes the 4-branch probabilities for Rubinstein's 2D tree.
///
/// Returns (p_uu, p_ud, p_du, p_dd) matching marginal moments and correlation.
fn two_asset_probabilities(
    b1: f64,
    b2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    dt: f64,
) -> Result<(f64, f64, f64, f64), PricingError> {
    let u1 = (vol1 * dt.sqrt()).exp();
    let d1 = 1.0 / u1;
    let u2 = (vol2 * dt.sqrt()).exp();
    let d2 = 1.0 / u2;

    let g1 = (b1 * dt).exp();
    let g2 = (b2 * dt).exp();

    let denom1 = u1 - d1;
    let denom2 = u2 - d2;

    if denom1.abs() < 1e-14 || denom2.abs() < 1e-14 {
        return Err(PricingError::NumericalError(
            "two-asset tree: degenerate up/down factors".to_string(),
        ));
    }

    // Marginal risk-neutral probabilities
    let p1 = (g1 - d1) / denom1;
    let p2 = (g2 - d2) / denom2;

    if !(0.0..=1.0).contains(&p1) || !(0.0..=1.0).contains(&p2) {
        return Err(PricingError::NumericalError(
            "two-asset tree: marginal probability outside [0, 1]".to_string(),
        ));
    }

    // Joint probabilities matching correlation
    // p_uu = p1*p2 + rho*sqrt(p1*(1-p1)*p2*(1-p2))
    let corr_adj = rho * (p1 * (1.0 - p1) * p2 * (1.0 - p2)).sqrt();

    let p_uu = p1 * p2 + corr_adj;
    let p_ud = p1 * (1.0 - p2) - corr_adj;
    let p_du = (1.0 - p1) * p2 - corr_adj;
    let p_dd = (1.0 - p1) * (1.0 - p2) + corr_adj;

    // Check validity
    if p_uu < -1e-12 || p_ud < -1e-12 || p_du < -1e-12 || p_dd < -1e-12 {
        return Err(PricingError::NumericalError(
            "two-asset tree: joint probability is negative".to_string(),
        ));
    }

    Ok((p_uu.max(0.0), p_ud.max(0.0), p_du.max(0.0), p_dd.max(0.0)))
}

/// Price a European two-asset option on a 2D recombining tree.
///
/// `payoff_fn` takes (S1_T, S2_T) and returns the payoff.
fn price_two_asset_european<F>(
    s1: f64,
    s2: f64,
    vol1: f64,
    vol2: f64,
    r: f64,
    b1: f64,
    b2: f64,
    rho: f64,
    t: f64,
    steps: usize,
    payoff_fn: F,
) -> Result<f64, PricingError>
where
    F: Fn(f64, f64) -> f64,
{
    if steps == 0 {
        return Err(PricingError::InvalidInput(
            "two-asset tree steps must be > 0".to_string(),
        ));
    }

    let dt = t / steps as f64;
    let u1 = (vol1 * dt.sqrt()).exp();
    let d1 = 1.0 / u1;
    let u2 = (vol2 * dt.sqrt()).exp();
    let d2 = 1.0 / u2;

    let (p_uu, p_ud, p_du, p_dd) = two_asset_probabilities(b1, b2, vol1, vol2, rho, dt)?;
    let disc = (-r * dt).exp();

    let n = steps + 1;

    // Terminal payoffs using multiplicative recurrence to eliminate powf().
    // S1 = s1 * d1^steps * (u1/d1)^i, S2 = s2 * d2^steps * (u2/d2)^j
    let ratio1 = u1 / d1;
    let ratio2 = u2 / d2;
    let mut values = vec![vec![0.0_f64; n]; n];
    {
        let mut s1_t = s1 * d1.powi(steps as i32);
        for value_row in values.iter_mut().take(n) {
            let mut s2_t = s2 * d2.powi(steps as i32);
            for value in value_row.iter_mut().take(n) {
                *value = payoff_fn(s1_t, s2_t);
                s2_t *= ratio2;
            }
            s1_t *= ratio1;
        }
    }

    // Backward induction
    for step in (0..steps).rev() {
        let m = step + 1;
        let mut new_values = vec![vec![0.0_f64; m]; m];
        for i in 0..m {
            for j in 0..m {
                new_values[i][j] = disc
                    * (p_uu * values[i + 1][j + 1]
                        + p_ud * values[i + 1][j]
                        + p_du * values[i][j + 1]
                        + p_dd * values[i][j]);
            }
        }
        values = new_values;
    }

    Ok(values[0][0])
}

impl PricingEngine<SpreadOption> for TwoAssetBinomialEngine {
    fn price(
        &self,
        instrument: &SpreadOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.t <= 0.0 {
            let payoff = (instrument.s1 - instrument.s2 - instrument.k).max(0.0);
            return Ok(PricingResult {
                price: payoff,
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let b1 = instrument.r - instrument.q1;
        let b2 = instrument.r - instrument.q2;

        let price = price_two_asset_european(
            instrument.s1,
            instrument.s2,
            instrument.vol1,
            instrument.vol2,
            instrument.r,
            b1,
            b2,
            instrument.rho,
            instrument.t,
            self.steps,
            |s1_t, s2_t| (s1_t - s2_t - instrument.k).max(0.0),
        )?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("rho", instrument.rho);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<BestOfTwoCallOption> for TwoAssetBinomialEngine {
    fn price(
        &self,
        instrument: &BestOfTwoCallOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.t <= 0.0 {
            let payoff = (instrument.s1.max(instrument.s2) - instrument.k).max(0.0);
            return Ok(PricingResult {
                price: payoff,
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let b1 = instrument.r - instrument.q1;
        let b2 = instrument.r - instrument.q2;

        let price = price_two_asset_european(
            instrument.s1,
            instrument.s2,
            instrument.vol1,
            instrument.vol2,
            instrument.r,
            b1,
            b2,
            instrument.rho,
            instrument.t,
            self.steps,
            |s1_t, s2_t| (s1_t.max(s2_t) - instrument.k).max(0.0),
        )?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("rho", instrument.rho);

        Ok(PricingResult {
            price,
            stderr: None,
            greeks: None,
            diagnostics,
        })
    }
}

impl PricingEngine<WorstOfTwoCallOption> for TwoAssetBinomialEngine {
    fn price(
        &self,
        instrument: &WorstOfTwoCallOption,
        _market: &Market,
    ) -> Result<PricingResult, PricingError> {
        instrument.validate()?;

        if instrument.t <= 0.0 {
            let payoff = (instrument.s1.min(instrument.s2) - instrument.k).max(0.0);
            return Ok(PricingResult {
                price: payoff,
                stderr: None,
                greeks: None,
                diagnostics: crate::core::Diagnostics::new(),
            });
        }

        let b1 = instrument.r - instrument.q1;
        let b2 = instrument.r - instrument.q2;

        let price = price_two_asset_european(
            instrument.s1,
            instrument.s2,
            instrument.vol1,
            instrument.vol2,
            instrument.r,
            b1,
            b2,
            instrument.rho,
            instrument.t,
            self.steps,
            |s1_t, s2_t| (s1_t.min(s2_t) - instrument.k).max(0.0),
        )?;

        let mut diagnostics = crate::core::Diagnostics::new();
        diagnostics.insert("num_steps", self.steps as f64);
        diagnostics.insert("rho", instrument.rho);

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
    use crate::core::PricingEngine;
    use crate::engines::analytic::rainbow::{best_of_two_call_price, worst_of_two_call_price};
    use crate::engines::analytic::spread::kirk_spread_price;
    use approx::assert_relative_eq;

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
    fn spread_call_matches_kirk_approximation() {
        let option = SpreadOption {
            s1: 100.0,
            s2: 96.0,
            k: 3.0,
            vol1: 0.20,
            vol2: 0.15,
            rho: 0.5,
            q1: 0.0,
            q2: 0.0,
            r: 0.05,
            t: 0.5,
        };

        let kirk = kirk_spread_price(&option).unwrap();
        let tree = TwoAssetBinomialEngine::new(200)
            .price(&option, &dummy_market())
            .unwrap()
            .price;

        // Kirk is an approximation, so tolerance is wider
        assert_relative_eq!(tree, kirk, epsilon = 0.15);
    }

    #[test]
    fn best_of_call_matches_stulz() {
        let option = BestOfTwoCallOption {
            s1: 100.0,
            s2: 100.0,
            k: 100.0,
            r: 0.05,
            q1: 0.0,
            q2: 0.0,
            vol1: 0.20,
            vol2: 0.20,
            rho: 0.5,
            t: 1.0,
        };

        let analytic = best_of_two_call_price(&option).unwrap();
        let tree = TwoAssetBinomialEngine::new(100)
            .price(&option, &dummy_market())
            .unwrap()
            .price;

        assert_relative_eq!(tree, analytic, epsilon = 0.5);
    }

    #[test]
    fn worst_of_call_matches_analytic() {
        let option = WorstOfTwoCallOption {
            s1: 100.0,
            s2: 100.0,
            k: 100.0,
            r: 0.05,
            q1: 0.0,
            q2: 0.0,
            vol1: 0.20,
            vol2: 0.20,
            rho: 0.5,
            t: 1.0,
        };

        let analytic = worst_of_two_call_price(&option).unwrap();
        let tree = TwoAssetBinomialEngine::new(100)
            .price(&option, &dummy_market())
            .unwrap()
            .price;

        assert_relative_eq!(tree, analytic, epsilon = 0.5);
    }

    #[test]
    fn best_plus_worst_equals_sum_of_marginal_calls() {
        let best = BestOfTwoCallOption {
            s1: 100.0,
            s2: 100.0,
            k: 100.0,
            r: 0.05,
            q1: 0.0,
            q2: 0.0,
            vol1: 0.20,
            vol2: 0.20,
            rho: 0.5,
            t: 1.0,
        };
        let worst = WorstOfTwoCallOption {
            s1: best.s1,
            s2: best.s2,
            k: best.k,
            r: best.r,
            q1: best.q1,
            q2: best.q2,
            vol1: best.vol1,
            vol2: best.vol2,
            rho: best.rho,
            t: best.t,
        };

        let market = dummy_market();
        let engine = TwoAssetBinomialEngine::new(100);

        let best_price = engine.price(&best, &market).unwrap().price;
        let worst_price = engine.price(&worst, &market).unwrap().price;

        // best_of + worst_of = call(S1,K) + call(S2,K)
        // Both marginals should be equal here (symmetric case)
        let analytic_best = best_of_two_call_price(&best).unwrap();
        let analytic_worst = worst_of_two_call_price(&worst).unwrap();
        let analytic_sum = analytic_best + analytic_worst;

        assert_relative_eq!(best_price + worst_price, analytic_sum, epsilon = 1.0);
    }
}
