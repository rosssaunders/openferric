//! Instrument definition for Variance Swap contracts.
//!
//! Module openferric::instruments::variance_swap contains payoff parameters and validation logic.

use crate::core::{Instrument, PricingError};

/// Option quote used for variance/volatility swap replication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceOptionQuote {
    /// Option strike.
    pub strike: f64,
    /// Call premium for this strike.
    pub call_price: f64,
    /// Put premium for this strike.
    pub put_price: f64,
}

impl VarianceOptionQuote {
    /// Creates a quote tuple for replication.
    pub fn new(strike: f64, call_price: f64, put_price: f64) -> Self {
        Self {
            strike,
            call_price,
            put_price,
        }
    }

    /// Validates quote fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(PricingError::InvalidInput(
                "variance quote strike must be finite and > 0".to_string(),
            ));
        }
        if !self.call_price.is_finite() || self.call_price < -1.0e-12 {
            return Err(PricingError::InvalidInput(
                "variance quote call_price must be finite and >= 0".to_string(),
            ));
        }
        if !self.put_price.is_finite() || self.put_price < -1.0e-12 {
            return Err(PricingError::InvalidInput(
                "variance quote put_price must be finite and >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Variance swap instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceSwap {
    /// Vega notional used to derive variance notional via `N_var = N_vega / (2*K_vol)`.
    pub notional_vega: f64,
    /// Volatility strike `K_vol`; variance strike is `K_vol^2`.
    pub strike_vol: f64,
    /// Maturity in years.
    pub expiry: f64,
    /// Optional realized variance (annualized) used for mark-to-market.
    pub observed_realized_var: Option<f64>,
    /// OTM strip quotes used for fair variance strike replication.
    pub option_quotes: Vec<VarianceOptionQuote>,
}

impl VarianceSwap {
    /// Creates a variance swap.
    pub fn new(
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        option_quotes: Vec<VarianceOptionQuote>,
    ) -> Self {
        Self {
            notional_vega,
            strike_vol,
            expiry,
            observed_realized_var: None,
            option_quotes,
        }
    }

    /// Creates a variance swap with observed realized variance.
    pub fn with_observed_realized_var(mut self, observed_realized_var: f64) -> Self {
        self.observed_realized_var = Some(observed_realized_var);
        self
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.notional_vega.is_finite() || self.notional_vega == 0.0 {
            return Err(PricingError::InvalidInput(
                "variance swap notional_vega must be finite and non-zero".to_string(),
            ));
        }
        if !self.strike_vol.is_finite() || self.strike_vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "variance swap strike_vol must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry <= 0.0 {
            return Err(PricingError::InvalidInput(
                "variance swap expiry must be finite and > 0".to_string(),
            ));
        }
        if self.option_quotes.len() < 2 {
            return Err(PricingError::InvalidInput(
                "variance swap requires at least two option quotes".to_string(),
            ));
        }
        for quote in &self.option_quotes {
            quote.validate()?;
        }
        if let Some(observed_realized_var) = self.observed_realized_var
            && (!observed_realized_var.is_finite() || observed_realized_var < 0.0)
        {
            return Err(PricingError::InvalidInput(
                "variance swap observed_realized_var must be finite and >= 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for VarianceSwap {
    fn instrument_type(&self) -> &str {
        "VarianceSwap"
    }
}

/// Volatility swap instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct VolatilitySwap {
    /// Vega notional (linear in realized volatility).
    pub notional_vega: f64,
    /// Volatility strike.
    pub strike_vol: f64,
    /// Maturity in years.
    pub expiry: f64,
    /// Optional realized variance (annualized) used for mark-to-market.
    pub observed_realized_var: Option<f64>,
    /// Replication quotes used to infer fair variance before convexity adjustment.
    pub option_quotes: Vec<VarianceOptionQuote>,
    /// Variance of variance used in convexity adjustment.
    pub var_of_var: f64,
}

impl VolatilitySwap {
    /// Creates a volatility swap.
    pub fn new(
        notional_vega: f64,
        strike_vol: f64,
        expiry: f64,
        option_quotes: Vec<VarianceOptionQuote>,
        var_of_var: f64,
    ) -> Self {
        Self {
            notional_vega,
            strike_vol,
            expiry,
            observed_realized_var: None,
            option_quotes,
            var_of_var,
        }
    }

    /// Creates a volatility swap with observed realized variance.
    pub fn with_observed_realized_var(mut self, observed_realized_var: f64) -> Self {
        self.observed_realized_var = Some(observed_realized_var);
        self
    }

    /// Validates instrument fields.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.notional_vega.is_finite() || self.notional_vega == 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility swap notional_vega must be finite and non-zero".to_string(),
            ));
        }
        if !self.strike_vol.is_finite() || self.strike_vol <= 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility swap strike_vol must be finite and > 0".to_string(),
            ));
        }
        if !self.expiry.is_finite() || self.expiry <= 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility swap expiry must be finite and > 0".to_string(),
            ));
        }
        if self.option_quotes.len() < 2 {
            return Err(PricingError::InvalidInput(
                "volatility swap requires at least two option quotes".to_string(),
            ));
        }
        for quote in &self.option_quotes {
            quote.validate()?;
        }
        if !self.var_of_var.is_finite() || self.var_of_var < 0.0 {
            return Err(PricingError::InvalidInput(
                "volatility swap var_of_var must be finite and >= 0".to_string(),
            ));
        }
        if let Some(observed_realized_var) = self.observed_realized_var
            && (!observed_realized_var.is_finite() || observed_realized_var < 0.0)
        {
            return Err(PricingError::InvalidInput(
                "volatility swap observed_realized_var must be finite and >= 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Instrument for VolatilitySwap {
    fn instrument_type(&self) -> &str {
        "VolatilitySwap"
    }
}
