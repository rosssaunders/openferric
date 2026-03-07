//! Multi-asset market data for DSL products.

use crate::core::PricingError;

/// Per-asset market data for the multi-asset DSL MC engine.
///
/// Each variant carries the market data appropriate for its asset type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum AssetMarketData {
    Equity {
        spot: f64,
        vol: f64,
        dividend_yield: f64,
    },
    Fx {
        spot: f64,
        vol: f64,
        domestic_rate: f64,
        foreign_rate: f64,
    },
    Commodity {
        spot: f64,
        vol: f64,
        convenience_yield: f64,
        /// Mean-reversion speed (Schwartz one-factor).
        kappa: f64,
        /// Long-run log-price mean (Schwartz one-factor).
        mu: f64,
    },
    Rate {
        initial_rate: f64,
        vol: f64,
        /// Mean-reversion speed (Vasicek/Hull-White).
        mean_reversion: f64,
        /// Long-run mean rate.
        long_run_mean: f64,
    },
}

impl AssetMarketData {
    /// The initial value used for path generation and performance calculations.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        match self {
            Self::Equity { spot, .. } => *spot,
            Self::Fx { spot, .. } => *spot,
            Self::Commodity { spot, .. } => *spot,
            Self::Rate { initial_rate, .. } => *initial_rate,
        }
    }

    /// The flat volatility.
    #[inline]
    pub fn vol(&self) -> f64 {
        match self {
            Self::Equity { vol, .. } => *vol,
            Self::Fx { vol, .. } => *vol,
            Self::Commodity { vol, .. } => *vol,
            Self::Rate { vol, .. } => *vol,
        }
    }

    /// Return a copy with the initial value (spot / initial_rate) bumped by `amount`.
    pub fn with_spot_bump(&self, amount: f64) -> Self {
        match self {
            Self::Equity {
                spot,
                vol,
                dividend_yield,
            } => Self::Equity {
                spot: spot + amount,
                vol: *vol,
                dividend_yield: *dividend_yield,
            },
            Self::Fx {
                spot,
                vol,
                domestic_rate,
                foreign_rate,
            } => Self::Fx {
                spot: spot + amount,
                vol: *vol,
                domestic_rate: *domestic_rate,
                foreign_rate: *foreign_rate,
            },
            Self::Commodity {
                spot,
                vol,
                convenience_yield,
                kappa,
                mu,
            } => Self::Commodity {
                spot: spot + amount,
                vol: *vol,
                convenience_yield: *convenience_yield,
                kappa: *kappa,
                mu: *mu,
            },
            Self::Rate {
                initial_rate,
                vol,
                mean_reversion,
                long_run_mean,
            } => Self::Rate {
                initial_rate: initial_rate + amount,
                vol: *vol,
                mean_reversion: *mean_reversion,
                long_run_mean: *long_run_mean,
            },
        }
    }

    /// Return a copy with vol bumped by `amount`.
    pub fn with_vol_bump(&self, amount: f64) -> Self {
        match self {
            Self::Equity {
                spot,
                vol,
                dividend_yield,
            } => Self::Equity {
                spot: *spot,
                vol: vol + amount,
                dividend_yield: *dividend_yield,
            },
            Self::Fx {
                spot,
                vol,
                domestic_rate,
                foreign_rate,
            } => Self::Fx {
                spot: *spot,
                vol: vol + amount,
                domestic_rate: *domestic_rate,
                foreign_rate: *foreign_rate,
            },
            Self::Commodity {
                spot,
                vol,
                convenience_yield,
                kappa,
                mu,
            } => Self::Commodity {
                spot: *spot,
                vol: vol + amount,
                convenience_yield: *convenience_yield,
                kappa: *kappa,
                mu: *mu,
            },
            Self::Rate {
                initial_rate,
                vol,
                mean_reversion,
                long_run_mean,
            } => Self::Rate {
                initial_rate: *initial_rate,
                vol: vol + amount,
                mean_reversion: *mean_reversion,
                long_run_mean: *long_run_mean,
            },
        }
    }
}

/// Multi-asset market snapshot used by `DslMonteCarloEngine`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiAssetMarket {
    /// Per-asset data, indexed by asset index.
    pub assets: Vec<AssetMarketData>,
    /// Correlation matrix (n x n, where n = assets.len()).
    pub correlation: Vec<Vec<f64>>,
    /// Risk-free rate (continuously compounded).
    pub rate: f64,
}

impl MultiAssetMarket {
    /// Creates a single-asset equity market for simple products.
    pub fn single(spot: f64, vol: f64, rate: f64, dividend_yield: f64) -> Self {
        Self {
            assets: vec![AssetMarketData::Equity {
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
            match asset {
                AssetMarketData::Rate { vol, .. } => {
                    // Rate initial_value can be negative (negative rates).
                    if *vol <= 0.0 || !vol.is_finite() {
                        return Err(PricingError::InvalidInput(
                            "rate asset vol must be finite and > 0".to_string(),
                        ));
                    }
                }
                _ => {
                    if asset.initial_value() <= 0.0 {
                        return Err(PricingError::InvalidInput(
                            "asset spot must be > 0".to_string(),
                        ));
                    }
                    if asset.vol() <= 0.0 || !asset.vol().is_finite() {
                        return Err(PricingError::InvalidInput(
                            "asset vol must be finite and > 0".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns a vector of initial values (spot prices / initial rates).
    pub fn initial_spots(&self) -> Vec<f64> {
        self.assets.iter().map(|a| a.initial_value()).collect()
    }
}
