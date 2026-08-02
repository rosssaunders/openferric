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

    fn assert_invalid(asset: AssetMarketData, expected: &str) {
        let error = market_with(asset).validate().unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?} in validation error, got {error:?}"
        );
    }

    fn valid_equity() -> AssetMarketData {
        AssetMarketData::Equity {
            spot: 100.0,
            vol: 0.2,
            dividend_yield: 0.01,
        }
    }

    fn valid_fx() -> AssetMarketData {
        AssetMarketData::Fx {
            spot: 1.1,
            vol: 0.15,
            domestic_rate: 0.03,
            foreign_rate: 0.02,
        }
    }

    fn valid_commodity() -> AssetMarketData {
        AssetMarketData::Commodity {
            spot: 75.0,
            vol: 0.3,
            convenience_yield: 0.01,
            kappa: 0.4,
            mu: 4.0,
        }
    }

    fn valid_rate() -> AssetMarketData {
        AssetMarketData::Rate {
            initial_rate: -0.005,
            vol: 0.01,
            mean_reversion: 0.1,
            long_run_mean: 0.04,
        }
    }

    #[test]
    fn accessors_and_bumps_cover_every_asset_variant_without_changing_other_fields() {
        let equity = valid_equity();
        assert_eq!(equity.initial_value(), 100.0);
        assert_eq!(equity.vol(), 0.2);
        match equity.with_spot_bump(2.5) {
            AssetMarketData::Equity {
                spot,
                vol,
                dividend_yield,
            } => {
                assert_eq!(spot, 102.5);
                assert_eq!(vol, 0.2);
                assert_eq!(dividend_yield, 0.01);
            }
            _ => panic!("spot bump changed equity variant"),
        }
        match equity.with_vol_bump(0.025) {
            AssetMarketData::Equity {
                spot,
                vol,
                dividend_yield,
            } => {
                assert_eq!(spot, 100.0);
                assert_eq!(vol, 0.225);
                assert_eq!(dividend_yield, 0.01);
            }
            _ => panic!("vol bump changed equity variant"),
        }

        let fx = valid_fx();
        assert_eq!(fx.initial_value(), 1.1);
        assert_eq!(fx.vol(), 0.15);
        match fx.with_spot_bump(0.02) {
            AssetMarketData::Fx {
                spot,
                vol,
                domestic_rate,
                foreign_rate,
            } => {
                assert_eq!(spot, 1.1 + 0.02);
                assert_eq!(vol, 0.15);
                assert_eq!(domestic_rate, 0.03);
                assert_eq!(foreign_rate, 0.02);
            }
            _ => panic!("spot bump changed FX variant"),
        }
        match fx.with_vol_bump(-0.01) {
            AssetMarketData::Fx {
                spot,
                vol,
                domestic_rate,
                foreign_rate,
            } => {
                assert_eq!(spot, 1.1);
                assert_eq!(vol, 0.15 - 0.01);
                assert_eq!(domestic_rate, 0.03);
                assert_eq!(foreign_rate, 0.02);
            }
            _ => panic!("vol bump changed FX variant"),
        }

        let commodity = valid_commodity();
        assert_eq!(commodity.initial_value(), 75.0);
        assert_eq!(commodity.vol(), 0.3);
        match commodity.with_spot_bump(-1.5) {
            AssetMarketData::Commodity {
                spot,
                vol,
                convenience_yield,
                kappa,
                mu,
            } => {
                assert_eq!(spot, 73.5);
                assert_eq!(vol, 0.3);
                assert_eq!(convenience_yield, 0.01);
                assert_eq!(kappa, 0.4);
                assert_eq!(mu, 4.0);
            }
            _ => panic!("spot bump changed commodity variant"),
        }
        match commodity.with_vol_bump(0.05) {
            AssetMarketData::Commodity {
                spot,
                vol,
                convenience_yield,
                kappa,
                mu,
            } => {
                assert_eq!(spot, 75.0);
                assert_eq!(vol, 0.3 + 0.05);
                assert_eq!(convenience_yield, 0.01);
                assert_eq!(kappa, 0.4);
                assert_eq!(mu, 4.0);
            }
            _ => panic!("vol bump changed commodity variant"),
        }

        let rate = valid_rate();
        assert_eq!(rate.initial_value(), -0.005);
        assert_eq!(rate.vol(), 0.01);
        match rate.with_spot_bump(0.001) {
            AssetMarketData::Rate {
                initial_rate,
                vol,
                mean_reversion,
                long_run_mean,
            } => {
                assert_eq!(initial_rate, -0.005 + 0.001);
                assert_eq!(vol, 0.01);
                assert_eq!(mean_reversion, 0.1);
                assert_eq!(long_run_mean, 0.04);
            }
            _ => panic!("spot bump changed rate variant"),
        }
        match rate.with_vol_bump(0.0025) {
            AssetMarketData::Rate {
                initial_rate,
                vol,
                mean_reversion,
                long_run_mean,
            } => {
                assert_eq!(initial_rate, -0.005);
                assert_eq!(vol, 0.0125);
                assert_eq!(mean_reversion, 0.1);
                assert_eq!(long_run_mean, 0.04);
            }
            _ => panic!("vol bump changed rate variant"),
        }
    }

    #[test]
    fn mixed_asset_market_validates_and_returns_values_in_asset_order() {
        let market = MultiAssetMarket {
            assets: vec![valid_equity(), valid_fx(), valid_commodity(), valid_rate()],
            correlation: vec![
                vec![1.0, 0.1, -0.2, 0.0],
                vec![0.1, 1.0, 0.25, -0.1],
                vec![-0.2, 0.25, 1.0, 0.15],
                vec![0.0, -0.1, 0.15, 1.0],
            ],
            rate: 0.03,
        };

        market.validate().unwrap();
        assert_eq!(market.initial_spots(), [100.0, 1.1, 75.0, -0.005]);

        let single = MultiAssetMarket::single(90.0, 0.25, 0.04, 0.015);
        single.validate().unwrap();
        assert_eq!(single.initial_spots(), [90.0]);
        assert_eq!(single.correlation, [[1.0]]);
        match &single.assets[0] {
            AssetMarketData::Equity {
                spot,
                vol,
                dividend_yield,
            } => assert_eq!((*spot, *vol, *dividend_yield), (90.0, 0.25, 0.015)),
            _ => panic!("single market must contain equity data"),
        }
    }

    #[test]
    fn validate_rejects_empty_and_misshaped_correlation_matrices() {
        let empty = MultiAssetMarket {
            assets: vec![],
            correlation: vec![],
            rate: 0.03,
        };
        assert!(
            empty
                .validate()
                .unwrap_err()
                .to_string()
                .contains("at least one asset")
        );

        let wrong_rows = MultiAssetMarket {
            assets: vec![valid_equity(), valid_fx()],
            correlation: vec![vec![1.0, 0.2]],
            rate: 0.03,
        };
        assert!(
            wrong_rows
                .validate()
                .unwrap_err()
                .to_string()
                .contains("rows (1) must match number of assets (2)")
        );

        let wrong_columns = MultiAssetMarket {
            assets: vec![valid_equity(), valid_fx()],
            correlation: vec![vec![1.0, 0.2], vec![0.2]],
            rate: 0.03,
        };
        assert!(
            wrong_columns
                .validate()
                .unwrap_err()
                .to_string()
                .contains("row 1 has 1 columns, expected 2")
        );
    }

    #[test]
    fn validate_rejects_non_positive_spots_and_volatilities() {
        for (asset, expected) in [
            (
                AssetMarketData::Equity {
                    spot: 0.0,
                    vol: 0.2,
                    dividend_yield: 0.01,
                },
                "equity asset 0 spot must be > 0",
            ),
            (
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.0,
                    dividend_yield: 0.01,
                },
                "equity asset 0 vol must be > 0",
            ),
            (
                AssetMarketData::Fx {
                    spot: -1.0,
                    vol: 0.15,
                    domestic_rate: 0.03,
                    foreign_rate: 0.02,
                },
                "FX asset 0 spot must be > 0",
            ),
            (
                AssetMarketData::Fx {
                    spot: 1.1,
                    vol: -0.15,
                    domestic_rate: 0.03,
                    foreign_rate: 0.02,
                },
                "FX asset 0 vol must be > 0",
            ),
            (
                AssetMarketData::Commodity {
                    spot: 0.0,
                    vol: 0.3,
                    convenience_yield: 0.01,
                    kappa: 0.4,
                    mu: 4.0,
                },
                "commodity asset 0 spot must be > 0",
            ),
            (
                AssetMarketData::Commodity {
                    spot: 75.0,
                    vol: 0.0,
                    convenience_yield: 0.01,
                    kappa: 0.4,
                    mu: 4.0,
                },
                "commodity asset 0 vol must be > 0",
            ),
            (
                AssetMarketData::Rate {
                    initial_rate: -0.01,
                    vol: 0.0,
                    mean_reversion: 0.1,
                    long_run_mean: 0.04,
                },
                "rate asset 0 vol must be > 0",
            ),
        ] {
            assert_invalid(asset, expected);
        }
    }

    #[test]
    fn validate_rejects_each_non_finite_asset_parameter() {
        for (asset, expected) in [
            (
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: f64::INFINITY,
                    dividend_yield: 0.01,
                },
                "equity asset 0 vol must be finite",
            ),
            (
                AssetMarketData::Equity {
                    spot: 100.0,
                    vol: 0.2,
                    dividend_yield: f64::NEG_INFINITY,
                },
                "equity asset 0 dividend yield must be finite",
            ),
            (
                AssetMarketData::Fx {
                    spot: f64::NAN,
                    vol: 0.15,
                    domestic_rate: 0.03,
                    foreign_rate: 0.02,
                },
                "FX asset 0 spot must be finite",
            ),
            (
                AssetMarketData::Fx {
                    spot: 1.1,
                    vol: f64::NAN,
                    domestic_rate: 0.03,
                    foreign_rate: 0.02,
                },
                "FX asset 0 vol must be finite",
            ),
            (
                AssetMarketData::Fx {
                    spot: 1.1,
                    vol: 0.15,
                    domestic_rate: 0.03,
                    foreign_rate: f64::NAN,
                },
                "FX asset 0 foreign rate must be finite",
            ),
            (
                AssetMarketData::Commodity {
                    spot: f64::NAN,
                    vol: 0.3,
                    convenience_yield: 0.01,
                    kappa: 0.4,
                    mu: 4.0,
                },
                "commodity asset 0 spot must be finite",
            ),
            (
                AssetMarketData::Commodity {
                    spot: 75.0,
                    vol: f64::INFINITY,
                    convenience_yield: 0.01,
                    kappa: 0.4,
                    mu: 4.0,
                },
                "commodity asset 0 vol must be finite",
            ),
            (
                AssetMarketData::Commodity {
                    spot: 75.0,
                    vol: 0.3,
                    convenience_yield: f64::NAN,
                    kappa: 0.4,
                    mu: 4.0,
                },
                "commodity asset 0 convenience yield must be finite",
            ),
            (
                AssetMarketData::Commodity {
                    spot: 75.0,
                    vol: 0.3,
                    convenience_yield: 0.01,
                    kappa: 0.4,
                    mu: f64::NAN,
                },
                "commodity asset 0 mu must be finite",
            ),
            (
                AssetMarketData::Rate {
                    initial_rate: f64::NAN,
                    vol: 0.01,
                    mean_reversion: 0.1,
                    long_run_mean: 0.04,
                },
                "rate asset 0 initial rate must be finite",
            ),
            (
                AssetMarketData::Rate {
                    initial_rate: 0.03,
                    vol: f64::NAN,
                    mean_reversion: 0.1,
                    long_run_mean: 0.04,
                },
                "rate asset 0 vol must be finite",
            ),
            (
                AssetMarketData::Rate {
                    initial_rate: 0.03,
                    vol: 0.01,
                    mean_reversion: f64::NAN,
                    long_run_mean: 0.04,
                },
                "rate asset 0 mean reversion must be finite",
            ),
        ] {
            assert_invalid(asset, expected);
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
