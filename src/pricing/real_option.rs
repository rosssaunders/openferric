//! Module `pricing::real_option`.
//!
//! Implements real option workflows with concrete routines such as `price_option_to_defer`, `price_option_to_expand`, `price_option_to_abandon`, `european_abandonment_put`.
//!
//! References: Hull (11th ed.) for market conventions and payoff identities, with module-specific equations referenced by the concrete engines and models imported here.
//!
//! Key types and purpose: `RealOptionDecision`, `DecisionTreeNode`, `RealOptionValuation` define the core data contracts for this module.
//!
//! Numerical considerations: validate edge-domain inputs, preserve finite values where possible, and cross-check with reference implementations for production use.
//!
//! When to use: use these direct pricing helpers for quick valuation tasks; prefer trait-based instruments plus engines composition for larger systems and extensibility.
use crate::core::PricingError;
use crate::instruments::real_option::{
    AbandonmentOption, DeferInvestmentOption, DiscreteCashFlow, ExpandOption,
};

/// Decision made at a binomial tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealOptionDecision {
    /// Exercise/invest now.
    Invest,
    /// Wait and continue.
    Defer,
    /// Abandon and collect salvage value.
    Abandon,
}

/// Per-node valuation details for real-option decision trees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionTreeNode {
    /// Time in years of the node.
    pub time: f64,
    /// Project value state at the node.
    pub project_value: f64,
    /// Immediate invest/expand value.
    pub invest_value: f64,
    /// Continuation value from deferral.
    pub defer_value: f64,
    /// Immediate abandonment value.
    pub abandon_value: f64,
    /// Option value at this node.
    pub option_value: f64,
    /// Optimal decision at this node.
    pub decision: RealOptionDecision,
}

/// Real-option price and full decision tree.
#[derive(Debug, Clone, PartialEq)]
pub struct RealOptionValuation {
    /// Option NPV at valuation time.
    pub price: f64,
    /// Decision tree nodes by time level.
    pub nodes: Vec<Vec<DecisionTreeNode>>,
}

/// Prices an option to defer investment using a binomial tree.
pub fn price_option_to_defer(
    option: &DeferInvestmentOption,
) -> Result<RealOptionValuation, PricingError> {
    option.validate()?;
    price_with_decision_tree(
        option.model.project_value,
        option.model.volatility,
        option.model.risk_free_rate,
        option.model.maturity,
        option.model.steps,
        &option.model.cash_flows,
        |state_with_cf| (state_with_cf - option.investment_cost).max(0.0),
        |_| 0.0,
        true,
    )
}

/// Prices an expansion option (compound option on scaled project NPV).
pub fn price_option_to_expand(option: &ExpandOption) -> Result<RealOptionValuation, PricingError> {
    option.validate()?;
    price_with_decision_tree(
        option.model.project_value,
        option.model.volatility,
        option.model.risk_free_rate,
        option.model.maturity,
        option.model.steps,
        &option.model.cash_flows,
        |state_with_cf| {
            (option.expansion_multiplier * state_with_cf - option.expansion_cost).max(0.0)
        },
        |_| 0.0,
        true,
    )
}

/// Prices an abandonment option (American put on project value).
pub fn price_option_to_abandon(
    option: &AbandonmentOption,
) -> Result<RealOptionValuation, PricingError> {
    option.validate()?;
    price_with_decision_tree(
        option.model.project_value,
        option.model.volatility,
        option.model.risk_free_rate,
        option.model.maturity,
        option.model.steps,
        &option.model.cash_flows,
        |_| 0.0,
        |state_with_cf| (option.salvage_value - state_with_cf).max(0.0),
        false,
    )
}

/// European abandonment put equivalent (exercise only at maturity).
pub fn european_abandonment_put(option: &AbandonmentOption) -> Result<f64, PricingError> {
    option.validate()?;

    let dt = option.model.maturity / option.model.steps as f64;
    let (u, d, p) = crr_parameters(option.model.volatility, option.model.risk_free_rate, dt)?;
    let disc = (-option.model.risk_free_rate * dt).exp();
    let steps = option.model.steps;

    let mut values = vec![0.0_f64; steps + 1];
    for (j, value) in values.iter_mut().enumerate().take(steps + 1) {
        let state = option.model.project_value * u.powf(j as f64) * d.powf((steps - j) as f64);
        let state_with_cf = state
            + pv_future_cash_flows(
                &option.model.cash_flows,
                option.model.risk_free_rate,
                option.model.maturity,
            );
        *value = (option.salvage_value - state_with_cf).max(0.0);
    }

    for i in (0..steps).rev() {
        for j in 0..=i {
            values[j] = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
        }
    }

    Ok(values[0])
}

fn price_with_decision_tree<FI, FA>(
    project_value: f64,
    volatility: f64,
    risk_free_rate: f64,
    maturity: f64,
    steps: usize,
    cash_flows: &[DiscreteCashFlow],
    invest_payoff: FI,
    abandon_payoff: FA,
    invest_label: bool,
) -> Result<RealOptionValuation, PricingError>
where
    FI: Fn(f64) -> f64,
    FA: Fn(f64) -> f64,
{
    let dt = maturity / steps as f64;
    let (u, d, p) = crr_parameters(volatility, risk_free_rate, dt)?;
    let disc = (-risk_free_rate * dt).exp();

    let mut value_tree = (0..=steps)
        .map(|i| vec![0.0_f64; i + 1])
        .collect::<Vec<_>>();
    let mut nodes = (0..=steps)
        .map(|i| vec![default_node(); i + 1])
        .collect::<Vec<_>>();

    for i in (0..=steps).rev() {
        let t = i as f64 * dt;
        let cf_pv = pv_future_cash_flows(cash_flows, risk_free_rate, t);

        for j in 0..=i {
            let state = project_value * u.powf(j as f64) * d.powf((i - j) as f64);
            let state_with_cf = state + cf_pv;
            let invest_value = invest_payoff(state_with_cf).max(0.0);
            let abandon_value = abandon_payoff(state_with_cf).max(0.0);

            let (defer_value, option_value, decision) = if i == steps {
                if invest_label {
                    if invest_value > 0.0 {
                        (0.0, invest_value, RealOptionDecision::Invest)
                    } else {
                        (0.0, 0.0, RealOptionDecision::Defer)
                    }
                } else if abandon_value > 0.0 {
                    (0.0, abandon_value, RealOptionDecision::Abandon)
                } else {
                    (0.0, 0.0, RealOptionDecision::Defer)
                }
            } else {
                let continuation =
                    disc * (p * value_tree[i + 1][j + 1] + (1.0 - p) * value_tree[i + 1][j]);
                if invest_label {
                    if invest_value >= continuation {
                        (continuation, invest_value, RealOptionDecision::Invest)
                    } else {
                        (continuation, continuation, RealOptionDecision::Defer)
                    }
                } else if abandon_value >= continuation {
                    (continuation, abandon_value, RealOptionDecision::Abandon)
                } else {
                    (continuation, continuation, RealOptionDecision::Defer)
                }
            };

            value_tree[i][j] = option_value;
            nodes[i][j] = DecisionTreeNode {
                time: t,
                project_value: state,
                invest_value,
                defer_value,
                abandon_value,
                option_value,
                decision,
            };
        }
    }

    Ok(RealOptionValuation {
        price: value_tree[0][0],
        nodes,
    })
}

fn crr_parameters(
    volatility: f64,
    risk_free_rate: f64,
    dt: f64,
) -> Result<(f64, f64, f64), PricingError> {
    if dt <= 0.0 || !dt.is_finite() {
        return Err(PricingError::InvalidInput(
            "time step must be finite and > 0".to_string(),
        ));
    }

    if volatility <= 1.0e-14 {
        let growth = (risk_free_rate * dt).exp();
        return Ok((growth, growth, 1.0));
    }

    let u = (volatility * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = (risk_free_rate * dt).exp();
    let p = (growth - d) / (u - d);
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(PricingError::NumericalError(
            "risk-neutral probability is outside [0, 1]".to_string(),
        ));
    }
    Ok((u, d, p))
}

fn pv_future_cash_flows(cash_flows: &[DiscreteCashFlow], risk_free_rate: f64, time: f64) -> f64 {
    cash_flows
        .iter()
        .filter(|cf| cf.time >= time - 1.0e-12)
        .map(|cf| cf.amount * (-risk_free_rate * (cf.time - time)).exp())
        .sum()
}

fn default_node() -> DecisionTreeNode {
    DecisionTreeNode {
        time: 0.0,
        project_value: 0.0,
        invest_value: 0.0,
        defer_value: 0.0,
        abandon_value: 0.0,
        option_value: 0.0,
        decision: RealOptionDecision::Defer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::real_option::{
        AbandonmentOption, DeferInvestmentOption, ExpandOption, RealOptionBinomialSpec,
    };
    use statrs::distribution::{ContinuousCDF, Normal};

    fn black_scholes_price(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        maturity: f64,
        is_call: bool,
    ) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sigma_sqrt_t = volatility * maturity.sqrt();
        let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * maturity)
            / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        if is_call {
            spot * normal.cdf(d1) - strike * (-rate * maturity).exp() * normal.cdf(d2)
        } else {
            strike * (-rate * maturity).exp() * normal.cdf(-d2) - spot * normal.cdf(-d1)
        }
    }

    #[test]
    fn defer_option_matches_black_scholes_call_with_crr_budget() {
        // With no project distributions, early exercise of this American call
        // is suboptimal, so Black-Scholes is an independent analytic oracle.
        const CRR_DISCRETIZATION_BUDGET: f64 = 2.3e-3;
        let option = DeferInvestmentOption {
            model: RealOptionBinomialSpec {
                project_value: 120.0,
                volatility: 0.25,
                risk_free_rate: 0.05,
                maturity: 1.0,
                steps: 400,
                cash_flows: vec![],
            },
            investment_cost: 100.0,
        };
        let actual = price_option_to_defer(&option).unwrap().price;
        let reference = black_scholes_price(120.0, 100.0, 0.05, 0.25, 1.0, true);

        assert!(
            (actual - reference).abs() <= CRR_DISCRETIZATION_BUDGET,
            "CRR defer price {actual:.12} differs from BS reference {reference:.12}"
        );
        // Supplemental economic invariant.
        assert!(actual >= 20.0);
    }

    #[test]
    fn expand_option_matches_scaled_black_scholes_call_with_crr_budget() {
        // max(m*S-C, 0) = m*max(S-C/m, 0). With no project
        // distributions this American expansion right is therefore a scaled
        // European Black-Scholes call.
        const CRR_DISCRETIZATION_BUDGET: f64 = 6.0e-3;
        let option = ExpandOption {
            model: RealOptionBinomialSpec {
                project_value: 100.0,
                volatility: 0.30,
                risk_free_rate: 0.04,
                maturity: 1.0,
                steps: 800,
                cash_flows: vec![],
            },
            expansion_multiplier: 1.5,
            expansion_cost: 150.0,
        };

        let actual = price_option_to_expand(&option).unwrap().price;
        let reference = 1.5 * black_scholes_price(100.0, 100.0, 0.04, 0.30, 1.0, true);
        assert!(
            (actual - reference).abs() <= CRR_DISCRETIZATION_BUDGET,
            "CRR expansion price {actual:.12} differs from scaled BS reference {reference:.12}"
        );
    }

    #[test]
    fn abandonment_matches_closed_form_and_deterministic_references() {
        // The European helper is a CRR approximation to a Black-Scholes put.
        const CRR_DISCRETIZATION_BUDGET: f64 = 1.0e-3;
        let option = AbandonmentOption {
            model: RealOptionBinomialSpec {
                project_value: 95.0,
                volatility: 0.3,
                risk_free_rate: 0.04,
                maturity: 1.5,
                steps: 1_000,
                cash_flows: vec![],
            },
            salvage_value: 100.0,
        };

        let european = european_abandonment_put(&option).unwrap();
        let reference = black_scholes_price(95.0, 100.0, 0.04, 0.30, 1.5, false);
        assert!(
            (european - reference).abs() <= CRR_DISCRETIZATION_BUDGET,
            "CRR European abandonment price {european:.12} differs from BS put {reference:.12}"
        );

        // At zero volatility and a positive rate the project grows
        // deterministically, so an in-the-money abandonment right is exercised
        // immediately and is exactly its intrinsic value.
        let deterministic = AbandonmentOption {
            model: RealOptionBinomialSpec {
                project_value: 95.0,
                volatility: 0.0,
                risk_free_rate: 0.04,
                maturity: 1.5,
                steps: 24,
                cash_flows: vec![],
            },
            salvage_value: 100.0,
        };
        let american_deterministic = price_option_to_abandon(&deterministic).unwrap().price;
        assert!((american_deterministic - 5.0).abs() <= 1.0e-12);

        // Supplemental economic invariant for the stochastic contract.
        let american = price_option_to_abandon(&option).unwrap().price;
        assert!(american >= european - 1.0e-10);
    }

    #[test]
    fn multi_stage_cash_flows_match_independent_decimal_crr_references() {
        // These finite-grid targets were generated independently with Python's
        // 60-digit `decimal` arithmetic.  The generator materialized complete
        // CRR time slices and, at every node, independently evaluated
        //
        //   max(exercise(S + sum_{T_k >= t} CF_k exp(-r(T_k-t))),
        //       exp(-r dt) E[V_next]).
        //
        // No OpenFerric code or output was used by that calculation.  The
        // cash-flow dates are aligned to the 256-step grid, making these exact
        // references for this specified finite-grid methodology rather than a
        // broad economic range.
        fn model(steps: usize) -> RealOptionBinomialSpec {
            RealOptionBinomialSpec {
                project_value: 82.0,
                volatility: 0.28,
                risk_free_rate: 0.045,
                maturity: 2.0,
                steps,
                cash_flows: vec![
                    DiscreteCashFlow {
                        time: 0.5,
                        amount: 6.0,
                    },
                    DiscreteCashFlow {
                        time: 1.0,
                        amount: 9.0,
                    },
                    DiscreteCashFlow {
                        time: 1.5,
                        amount: 12.0,
                    },
                    DiscreteCashFlow {
                        time: 2.0,
                        amount: 20.0,
                    },
                ],
            }
        }

        let defer = DeferInvestmentOption {
            model: model(256),
            investment_cost: 115.0,
        };
        let expand = ExpandOption {
            model: model(256),
            expansion_multiplier: 1.4,
            expansion_cost: 135.0,
        };
        let abandon = AbandonmentOption {
            model: model(256),
            salvage_value: 95.0,
        };

        let defer_valuation = price_option_to_defer(&defer).unwrap();
        let expand_valuation = price_option_to_expand(&expand).unwrap();
        let abandon_valuation = price_option_to_abandon(&abandon).unwrap();
        let defer_price = defer_valuation.price;
        let expand_price = expand_valuation.price;
        let abandon_price = abandon_valuation.price;

        const DECIMAL_REFERENCE_ROUNDOFF_BUDGET: f64 = 2.0e-10;
        for (label, actual, reference) in [
            ("defer", defer_price, 16.332_294_078_471_08),
            ("expand", expand_price, 44.466_113_732_262_36),
            ("abandon", abandon_price, 6.681_512_608_900_297),
        ] {
            assert!(
                (actual - reference).abs() <= DECIMAL_REFERENCE_ROUNDOFF_BUDGET,
                "multi-stage {label} finite-grid price: actual={actual:.17e}, \
                 independent decimal reference={reference:.17e}, \
                 roundoff budget={DECIMAL_REFERENCE_ROUNDOFF_BUDGET:.3e}"
            );
        }

        // Refinement budgets are stated separately because they measure CRR
        // discretization, not accuracy of the finite-grid implementation.
        let refined_defer = price_option_to_defer(&DeferInvestmentOption {
            model: model(512),
            investment_cost: 115.0,
        })
        .unwrap()
        .price;
        let refined_expand = price_option_to_expand(&ExpandOption {
            model: model(512),
            expansion_multiplier: 1.4,
            expansion_cost: 135.0,
        })
        .unwrap()
        .price;
        let refined_abandon = price_option_to_abandon(&AbandonmentOption {
            model: model(512),
            salvage_value: 95.0,
        })
        .unwrap()
        .price;

        for (label, coarse, refined, grid_budget) in [
            ("defer", defer_price, refined_defer, 1.3e-3),
            ("expand", expand_price, refined_expand, 3.4e-3),
            ("abandon", abandon_price, refined_abandon, 2.0e-3),
        ] {
            assert!(
                (refined - coarse).abs() <= grid_budget,
                "multi-stage {label} 256-to-512 grid change: p256={coarse:.12}, \
                 p512={refined:.12}, budget={grid_budget:.3e}"
            );
        }

        // The root is not a trivial immediate-exercise reduction: defer and
        // expand retain time value, while abandonment is initially OTM and is
        // valuable only in later low-project states.
        assert_eq!(defer.model.cash_flows.len(), 4);
        let immediate_defer = (defer.model.project_value
            + pv_future_cash_flows(&defer.model.cash_flows, defer.model.risk_free_rate, 0.0)
            - defer.investment_cost)
            .max(0.0);
        assert!(defer_price > immediate_defer);
        let immediate_expand = (expand.expansion_multiplier
            * (expand.model.project_value
                + pv_future_cash_flows(
                    &expand.model.cash_flows,
                    expand.model.risk_free_rate,
                    0.0,
                ))
            - expand.expansion_cost)
            .max(0.0);
        assert!(expand_price > immediate_expand);
        let immediate_abandon = (abandon.salvage_value
            - abandon.model.project_value
            - pv_future_cash_flows(&abandon.model.cash_flows, abandon.model.risk_free_rate, 0.0))
        .max(0.0);
        assert_eq!(immediate_abandon, 0.0);
        assert!(abandon_price > 0.0);

        let contains_decision = |valuation: &RealOptionValuation, wanted| {
            valuation
                .nodes
                .iter()
                .flatten()
                .any(|node| node.decision == wanted)
        };
        for valuation in [&defer_valuation, &expand_valuation] {
            assert!(contains_decision(valuation, RealOptionDecision::Invest));
            assert!(contains_decision(valuation, RealOptionDecision::Defer));
        }
        assert!(contains_decision(
            &abandon_valuation,
            RealOptionDecision::Abandon
        ));
        assert!(contains_decision(
            &abandon_valuation,
            RealOptionDecision::Defer
        ));
    }
}
