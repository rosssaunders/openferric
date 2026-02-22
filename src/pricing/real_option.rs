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
        AbandonmentOption, DeferInvestmentOption, RealOptionBinomialSpec,
    };

    #[test]
    fn defer_option_value_is_at_least_intrinsic() {
        let option = DeferInvestmentOption {
            model: RealOptionBinomialSpec {
                project_value: 120.0,
                volatility: 0.25,
                risk_free_rate: 0.05,
                maturity: 1.0,
                steps: 200,
                cash_flows: vec![],
            },
            investment_cost: 100.0,
        };
        let px = price_option_to_defer(&option).unwrap().price;
        assert!(px >= 20.0);
    }

    #[test]
    fn american_abandonment_dominates_european() {
        let option = AbandonmentOption {
            model: RealOptionBinomialSpec {
                project_value: 95.0,
                volatility: 0.3,
                risk_free_rate: 0.04,
                maturity: 1.5,
                steps: 150,
                cash_flows: vec![],
            },
            salvage_value: 100.0,
        };

        let am = price_option_to_abandon(&option).unwrap().price;
        let eu = european_abandonment_put(&option).unwrap();
        assert!(am >= eu - 1.0e-10);
    }
}
