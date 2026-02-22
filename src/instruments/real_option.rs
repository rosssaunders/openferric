use crate::core::{Instrument, PricingError};

/// Discrete project cash flow paid at a given year fraction.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiscreteCashFlow {
    /// Payment time in years from valuation date.
    pub time: f64,
    /// Cash-flow amount.
    pub amount: f64,
}

/// Shared binomial settings for real-option valuation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RealOptionBinomialSpec {
    /// Current project NPV proxy (underlying state variable).
    pub project_value: f64,
    /// Project-value volatility.
    pub volatility: f64,
    /// Risk-free discount rate.
    pub risk_free_rate: f64,
    /// Option maturity in years.
    pub maturity: f64,
    /// Number of binomial time steps.
    pub steps: usize,
    /// Optional discrete cash flows used in exercise value.
    pub cash_flows: Vec<DiscreteCashFlow>,
}

impl RealOptionBinomialSpec {
    /// Validates shared settings.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.project_value <= 0.0 || !self.project_value.is_finite() {
            return Err(PricingError::InvalidInput(
                "project_value must be finite and > 0".to_string(),
            ));
        }
        if self.volatility < 0.0 || !self.volatility.is_finite() {
            return Err(PricingError::InvalidInput(
                "volatility must be finite and >= 0".to_string(),
            ));
        }
        if !self.risk_free_rate.is_finite() {
            return Err(PricingError::InvalidInput(
                "risk_free_rate must be finite".to_string(),
            ));
        }
        if self.maturity <= 0.0 || !self.maturity.is_finite() {
            return Err(PricingError::InvalidInput(
                "maturity must be finite and > 0".to_string(),
            ));
        }
        if self.steps == 0 {
            return Err(PricingError::InvalidInput("steps must be > 0".to_string()));
        }

        for (idx, cf) in self.cash_flows.iter().enumerate() {
            if !cf.time.is_finite() || !cf.amount.is_finite() || cf.time < 0.0 {
                return Err(PricingError::InvalidInput(format!(
                    "cash_flows[{idx}] must have finite values with non-negative time"
                )));
            }
        }

        Ok(())
    }
}

/// Option to defer an investment decision (American call on project NPV).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeferInvestmentOption {
    /// Shared tree settings.
    pub model: RealOptionBinomialSpec,
    /// Investment cost (strike).
    pub investment_cost: f64,
}

impl DeferInvestmentOption {
    /// Input validation.
    pub fn validate(&self) -> Result<(), PricingError> {
        self.model.validate()?;
        if self.investment_cost <= 0.0 || !self.investment_cost.is_finite() {
            return Err(PricingError::InvalidInput(
                "investment_cost must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Option to expand project scale (compound option on scaled NPV).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExpandOption {
    /// Shared tree settings.
    pub model: RealOptionBinomialSpec,
    /// Multiplicative scale applied to project value upon expansion.
    pub expansion_multiplier: f64,
    /// Expansion cost paid at exercise.
    pub expansion_cost: f64,
}

impl ExpandOption {
    /// Input validation.
    pub fn validate(&self) -> Result<(), PricingError> {
        self.model.validate()?;
        if self.expansion_multiplier <= 0.0 || !self.expansion_multiplier.is_finite() {
            return Err(PricingError::InvalidInput(
                "expansion_multiplier must be finite and > 0".to_string(),
            ));
        }
        if self.expansion_cost <= 0.0 || !self.expansion_cost.is_finite() {
            return Err(PricingError::InvalidInput(
                "expansion_cost must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Option to abandon a project (American put with salvage value strike).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AbandonmentOption {
    /// Shared tree settings.
    pub model: RealOptionBinomialSpec,
    /// Salvage value received upon abandonment.
    pub salvage_value: f64,
}

impl AbandonmentOption {
    /// Input validation.
    pub fn validate(&self) -> Result<(), PricingError> {
        self.model.validate()?;
        if self.salvage_value <= 0.0 || !self.salvage_value.is_finite() {
            return Err(PricingError::InvalidInput(
                "salvage_value must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Unified real-option instrument enum.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RealOptionInstrument {
    /// Option to defer investment.
    Defer(DeferInvestmentOption),
    /// Option to expand project scale.
    Expand(ExpandOption),
    /// Option to abandon project.
    Abandon(AbandonmentOption),
}

impl RealOptionInstrument {
    /// Input validation.
    pub fn validate(&self) -> Result<(), PricingError> {
        match self {
            Self::Defer(spec) => spec.validate(),
            Self::Expand(spec) => spec.validate(),
            Self::Abandon(spec) => spec.validate(),
        }
    }
}

impl Instrument for RealOptionInstrument {
    fn instrument_type(&self) -> &str {
        "RealOption"
    }
}
