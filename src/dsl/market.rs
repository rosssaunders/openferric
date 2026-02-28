//! Multi-asset market data for DSL products.

use crate::core::PricingError;

/// Per-asset market data for the multi-asset DSL MC engine.
#[derive(Debug, Clone)]
pub struct AssetData {
    /// Initial spot price.
    pub spot: f64,
    /// Flat volatility.
    pub vol: f64,
    /// Continuous dividend yield.
    pub dividend_yield: f64,
}

/// Multi-asset market snapshot used by `DslMonteCarloEngine`.
#[derive(Debug, Clone)]
pub struct MultiAssetMarket {
    /// Per-asset data, indexed by asset index.
    pub assets: Vec<AssetData>,
    /// Correlation matrix (n x n, where n = assets.len()).
    pub correlation: Vec<Vec<f64>>,
    /// Risk-free rate (continuously compounded).
    pub rate: f64,
}

impl MultiAssetMarket {
    /// Creates a single-asset market for simple products.
    pub fn single(spot: f64, vol: f64, rate: f64, dividend_yield: f64) -> Self {
        Self {
            assets: vec![AssetData {
                spot,
                vol,
                dividend_yield,
            }],
            correlation: vec![vec![1.0]],
            rate,
        }
    }

    /// Validates the market data.
    pub fn validate(&self) -> Result<(), PricingError> {
        if self.assets.is_empty() {
            return Err(PricingError::InvalidInput(
                "multi-asset market requires at least one asset".to_string(),
            ));
        }
        let n = self.assets.len();
        if self.correlation.len() != n {
            return Err(PricingError::InvalidInput(format!(
                "correlation matrix rows ({}) must match number of assets ({n})",
                self.correlation.len()
            )));
        }
        for (i, row) in self.correlation.iter().enumerate() {
            if row.len() != n {
                return Err(PricingError::InvalidInput(format!(
                    "correlation matrix row {i} has {} columns, expected {n}",
                    row.len()
                )));
            }
        }
        for asset in &self.assets {
            if asset.spot <= 0.0 {
                return Err(PricingError::InvalidInput(
                    "asset spot must be > 0".to_string(),
                ));
            }
            if asset.vol <= 0.0 || !asset.vol.is_finite() {
                return Err(PricingError::InvalidInput(
                    "asset vol must be finite and > 0".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Returns a vector of initial spot prices.
    pub fn initial_spots(&self) -> Vec<f64> {
        self.assets.iter().map(|a| a.spot).collect()
    }
}
