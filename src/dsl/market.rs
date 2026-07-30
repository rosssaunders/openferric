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
        fn require_finite(name: &str, value: f64) -> Result<(), PricingError> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(PricingError::InvalidInput(format!("{name} must be finite")))
            }
        }

        if self.assets.is_empty() {
            return Err(PricingError::InvalidInput(
                "multi-asset market requires at least one asset".to_string(),
            ));
        }
        require_finite("market rate", self.rate)?;

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
            for (j, &correlation) in row.iter().enumerate() {
                require_finite(&format!("correlation[{i}][{j}]"), correlation)?;
            }
        }
        for (index, asset) in self.assets.iter().enumerate() {
            match asset {
                AssetMarketData::Equity {
                    spot,
                    vol,
                    dividend_yield,
                } => {
                    require_finite(&format!("equity asset {index} spot"), *spot)?;
                    require_finite(&format!("equity asset {index} vol"), *vol)?;
                    require_finite(
                        &format!("equity asset {index} dividend yield"),
                        *dividend_yield,
                    )?;
                    if *spot <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "equity asset {index} spot must be > 0"
                        )));
                    }
                    if *vol <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "equity asset {index} vol must be > 0"
                        )));
                    }
                }
                AssetMarketData::Fx {
                    spot,
                    vol,
                    domestic_rate,
                    foreign_rate,
                } => {
                    require_finite(&format!("FX asset {index} spot"), *spot)?;
                    require_finite(&format!("FX asset {index} vol"), *vol)?;
                    require_finite(&format!("FX asset {index} domestic rate"), *domestic_rate)?;
                    require_finite(&format!("FX asset {index} foreign rate"), *foreign_rate)?;
                    if *spot <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "FX asset {index} spot must be > 0"
                        )));
                    }
                    if *vol <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "FX asset {index} vol must be > 0"
                        )));
                    }
                }
                AssetMarketData::Commodity {
                    spot,
                    vol,
                    convenience_yield,
                    kappa,
                    mu,
                } => {
                    require_finite(&format!("commodity asset {index} spot"), *spot)?;
                    require_finite(&format!("commodity asset {index} vol"), *vol)?;
                    require_finite(
                        &format!("commodity asset {index} convenience yield"),
                        *convenience_yield,
                    )?;
                    require_finite(&format!("commodity asset {index} kappa"), *kappa)?;
                    require_finite(&format!("commodity asset {index} mu"), *mu)?;
                    if *spot <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "commodity asset {index} spot must be > 0"
                        )));
                    }
                    if *vol <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "commodity asset {index} vol must be > 0"
                        )));
                    }
                    if *kappa <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "commodity asset {index} kappa must be > 0"
                        )));
                    }
                }
                AssetMarketData::Rate {
                    initial_rate,
                    vol,
                    mean_reversion,
                    long_run_mean,
                } => {
                    // Rate initial_value can be negative (negative rates).
                    require_finite(&format!("rate asset {index} initial rate"), *initial_rate)?;
                    require_finite(&format!("rate asset {index} vol"), *vol)?;
                    require_finite(
                        &format!("rate asset {index} mean reversion"),
                        *mean_reversion,
                    )?;
                    require_finite(&format!("rate asset {index} long-run mean"), *long_run_mean)?;
                    if *vol <= 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "rate asset {index} vol must be > 0"
                        )));
                    }
                    if *mean_reversion < 0.0 {
                        return Err(PricingError::InvalidInput(format!(
                            "rate asset {index} mean reversion must be >= 0"
                        )));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn market_with(asset: AssetMarketData) -> MultiAssetMarket {
        MultiAssetMarket {
            assets: vec![asset],
            correlation: vec![vec![1.0]],
            rate: 0.03,
        }
    }

    #[test]
    fn validate_rejects_non_finite_top_level_market_values() {
        let mut market = MultiAssetMarket::single(100.0, 0.2, f64::NAN, 0.01);
        assert!(market.validate().unwrap_err().to_string().contains("rate"));

        market.rate = 0.03;
        market.correlation[0][0] = f64::INFINITY;
        assert!(
            market
                .validate()
                .unwrap_err()
                .to_string()
                .contains("correlation[0][0]")
        );
    }

    #[test]
    fn validate_rejects_non_finite_parameter_for_every_asset_variant() {
        let assets = [
            AssetMarketData::Equity {
                spot: f64::NAN,
                vol: 0.2,
                dividend_yield: 0.01,
            },
            AssetMarketData::Fx {
                spot: 1.1,
                vol: 0.15,
                domestic_rate: f64::NAN,
                foreign_rate: 0.02,
            },
            AssetMarketData::Commodity {
                spot: 75.0,
                vol: 0.3,
                convenience_yield: 0.01,
                kappa: f64::NAN,
                mu: 4.0,
            },
            AssetMarketData::Rate {
                initial_rate: 0.03,
                vol: 0.01,
                mean_reversion: 0.1,
                long_run_mean: f64::NAN,
            },
        ];

        for asset in assets {
            assert!(market_with(asset).validate().is_err());
        }
    }

    #[test]
    fn validate_rejects_invalid_mean_reversion_domains() {
        let commodity = AssetMarketData::Commodity {
            spot: 75.0,
            vol: 0.3,
            convenience_yield: 0.01,
            kappa: 0.0,
            mu: 4.0,
        };
        let error = market_with(commodity).validate().unwrap_err().to_string();
        assert!(error.contains("kappa must be > 0"), "unexpected: {error}");

        let rate = AssetMarketData::Rate {
            initial_rate: 0.03,
            vol: 0.01,
            mean_reversion: -0.1,
            long_run_mean: 0.04,
        };
        let error = market_with(rate).validate().unwrap_err().to_string();
        assert!(
            error.contains("mean reversion must be >= 0"),
            "unexpected: {error}"
        );
    }
}
