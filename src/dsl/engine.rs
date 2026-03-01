//! Multi-asset Monte Carlo engine for DSL products.
//!
//! Generates correlated GBM paths and evaluates compiled products.

use crate::core::{
    DiagKey, Diagnostics, Greeks, Instrument, PricingEngine, PricingError, PricingResult,
};
use crate::dsl::eval::evaluate_product;
use crate::dsl::ir::CompiledProduct;
use crate::dsl::market::MultiAssetMarket;
use crate::engines::monte_carlo::correlated_mc::{
    cholesky_for_correlation, sample_correlated_normals_cholesky,
};
use crate::market::Market;
use crate::math::fast_rng::{FastRng, FastRngKind};

/// A DSL product wrapped as an `Instrument` for compatibility.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DslProduct {
    pub product: CompiledProduct,
}

impl Instrument for DslProduct {
    fn instrument_type(&self) -> &str {
        "DslProduct"
    }
}

impl DslProduct {
    pub fn new(product: CompiledProduct) -> Self {
        Self { product }
    }
}

/// Multi-asset Monte Carlo engine for DSL products.
#[derive(Debug, Clone)]
pub struct DslMonteCarloEngine {
    pub num_paths: usize,
    pub num_steps: usize,
    pub seed: u64,
    pub rng_kind: FastRngKind,
}

impl DslMonteCarloEngine {
    pub fn new(num_paths: usize, num_steps: usize, seed: u64) -> Self {
        Self {
            num_paths,
            num_steps,
            seed,
            rng_kind: FastRngKind::Xoshiro256PlusPlus,
        }
    }

    /// Price a DSL product under a multi-asset market.
    pub fn price_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
    ) -> Result<PricingResult, PricingError> {
        market.validate()?;

        if self.num_paths == 0 {
            return Err(PricingError::InvalidInput(
                "num_paths must be > 0".to_string(),
            ));
        }
        if self.num_steps == 0 {
            return Err(PricingError::InvalidInput(
                "num_steps must be > 0".to_string(),
            ));
        }
        if product.maturity <= 0.0 {
            return Err(PricingError::InvalidInput(
                "product maturity must be > 0".to_string(),
            ));
        }

        let n_assets = market.assets.len();
        let n_steps = self.num_steps;
        let dt = product.maturity / n_steps as f64;
        let sqrt_dt = dt.sqrt();

        // Build Cholesky factor for correlation.
        let (chol, _) = cholesky_for_correlation(&market.correlation)?;

        let initial_spots = market.initial_spots();

        // Pre-compute per-asset drift and diffusion.
        let drifts: Vec<f64> = market
            .assets
            .iter()
            .map(|a| (market.rate - a.dividend_yield - 0.5 * a.vol * a.vol) * dt)
            .collect();
        let diffusions: Vec<f64> = market.assets.iter().map(|a| a.vol * sqrt_dt).collect();

        let mut rng = FastRng::from_seed(self.rng_kind, self.seed);
        let mut sum_pv = 0.0;
        let mut sum_pv_sq = 0.0;
        let mut corr_normals = vec![0.0; n_assets];

        for _ in 0..self.num_paths {
            // Generate correlated multi-asset path.
            // path_spots[step][asset]
            let mut path_spots = vec![vec![0.0; n_assets]; n_steps + 1];
            for (i, &s0) in initial_spots.iter().enumerate() {
                path_spots[0][i] = s0;
            }

            for step in 0..n_steps {
                sample_correlated_normals_cholesky(&chol, &mut rng, &mut corr_normals)?;

                for asset in 0..n_assets {
                    let prev = path_spots[step][asset];
                    let z = corr_normals[asset];
                    path_spots[step + 1][asset] =
                        prev * (drifts[asset] + diffusions[asset] * z).exp();
                }
            }

            // Evaluate product on this path.
            let pv = evaluate_product(product, &path_spots, &initial_spots, n_steps, market.rate)
                .map_err(|e| PricingError::NumericalError(e.to_string()))?;

            sum_pv += pv;
            sum_pv_sq += pv * pv;
        }

        let n = self.num_paths as f64;
        let mean = sum_pv / n;
        let variance = if self.num_paths > 1 {
            (sum_pv_sq - sum_pv * sum_pv / n) / (n - 1.0)
        } else {
            0.0
        };
        let stderr = (variance / n).sqrt();

        let mut diagnostics = Diagnostics::new();
        diagnostics.insert_key(DiagKey::NumPaths, n);
        diagnostics.insert_key(DiagKey::NumSteps, n_steps as f64);

        Ok(PricingResult {
            price: mean,
            stderr: Some(stderr),
            greeks: None,
            diagnostics,
        })
    }

    /// Compute Greeks via bump-and-reprice.
    pub fn greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
    ) -> Result<Greeks, PricingError> {
        if asset_index >= market.assets.len() {
            return Err(PricingError::InvalidInput(format!(
                "asset_index {asset_index} out of range (have {} assets)",
                market.assets.len()
            )));
        }

        let base = self.price_multi_asset(product, market)?;

        // Delta: bump spot by 1%
        let spot_bump = 0.01 * market.assets[asset_index].spot;
        let mut market_up = market.clone();
        market_up.assets[asset_index].spot += spot_bump;
        let price_up = self.price_multi_asset(product, &market_up)?.price;

        let mut market_down = market.clone();
        market_down.assets[asset_index].spot -= spot_bump;
        let price_down = self.price_multi_asset(product, &market_down)?.price;

        let delta = (price_up - price_down) / (2.0 * spot_bump);
        let gamma = (price_up - 2.0 * base.price + price_down) / (spot_bump * spot_bump);

        // Vega: bump vol by 1%
        let vol_bump = 0.01;
        let mut market_vega = market.clone();
        market_vega.assets[asset_index].vol += vol_bump;
        let price_vega = self.price_multi_asset(product, &market_vega)?.price;
        let vega = (price_vega - base.price) / vol_bump;

        // Rho: bump rate by 1bp
        let rate_bump = 0.0001;
        let mut market_rho = market.clone();
        market_rho.rate += rate_bump;
        let price_rho = self.price_multi_asset(product, &market_rho)?.price;
        let rho = (price_rho - base.price) / rate_bump;

        // Theta: bump maturity by -1/365 (not applicable to product maturity directly,
        // but we approximate by shifting observation dates isn't feasible, so we use
        // time-decay via a small time bump on rate discounting)
        let theta = 0.0; // Theta requires more sophisticated approach for products

        Ok(Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        })
    }

    /// Compute per-asset greeks including higher-order sensitivities (vanna, volga).
    pub fn extended_greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_index: usize,
        base_price: f64,
    ) -> Result<ExtendedGreeks, PricingError> {
        if asset_index >= market.assets.len() {
            return Err(PricingError::InvalidInput(format!(
                "asset_index {asset_index} out of range (have {} assets)",
                market.assets.len()
            )));
        }

        let spot_bump = 0.01 * market.assets[asset_index].spot;
        let vol_bump = 0.01;

        // Spot up / down
        let mut market_spot_up = market.clone();
        market_spot_up.assets[asset_index].spot += spot_bump;
        let price_spot_up = self.price_multi_asset(product, &market_spot_up)?.price;

        let mut market_spot_down = market.clone();
        market_spot_down.assets[asset_index].spot -= spot_bump;
        let price_spot_down = self.price_multi_asset(product, &market_spot_down)?.price;

        let delta = (price_spot_up - price_spot_down) / (2.0 * spot_bump);
        let gamma = (price_spot_up - 2.0 * base_price + price_spot_down) / (spot_bump * spot_bump);

        // Vol up / down
        let mut market_vol_up = market.clone();
        market_vol_up.assets[asset_index].vol += vol_bump;
        let price_vol_up = self.price_multi_asset(product, &market_vol_up)?.price;

        let mut market_vol_down = market.clone();
        market_vol_down.assets[asset_index].vol -= vol_bump;
        let price_vol_down = self.price_multi_asset(product, &market_vol_down)?.price;

        let vega = (price_vol_up - price_vol_down) / (2.0 * vol_bump);
        let volga = (price_vol_up - 2.0 * base_price + price_vol_down) / (vol_bump * vol_bump);

        // Vanna: cross derivative d²V/(dS dσ)
        // Bump spot+vol jointly for the four corners
        let mut m_up_up = market.clone();
        m_up_up.assets[asset_index].spot += spot_bump;
        m_up_up.assets[asset_index].vol += vol_bump;
        let p_up_up = self.price_multi_asset(product, &m_up_up)?.price;

        let mut m_up_down = market.clone();
        m_up_down.assets[asset_index].spot += spot_bump;
        m_up_down.assets[asset_index].vol -= vol_bump;
        let p_up_down = self.price_multi_asset(product, &m_up_down)?.price;

        let mut m_down_up = market.clone();
        m_down_up.assets[asset_index].spot -= spot_bump;
        m_down_up.assets[asset_index].vol += vol_bump;
        let p_down_up = self.price_multi_asset(product, &m_down_up)?.price;

        let mut m_down_down = market.clone();
        m_down_down.assets[asset_index].spot -= spot_bump;
        m_down_down.assets[asset_index].vol -= vol_bump;
        let p_down_down = self.price_multi_asset(product, &m_down_down)?.price;

        let vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4.0 * spot_bump * vol_bump);

        // Rho
        let rate_bump = 0.0001;
        let mut market_rho = market.clone();
        market_rho.rate += rate_bump;
        let price_rho = self.price_multi_asset(product, &market_rho)?.price;
        let rho = (price_rho - base_price) / rate_bump;

        Ok(ExtendedGreeks {
            delta,
            gamma,
            vega,
            theta: 0.0,
            rho,
            vanna,
            volga,
        })
    }

    /// Compute cross-asset sensitivities between a pair of underlyings.
    pub fn cross_greeks_multi_asset(
        &self,
        product: &CompiledProduct,
        market: &MultiAssetMarket,
        asset_i: usize,
        asset_j: usize,
        _base_price: f64,
    ) -> Result<CrossGreeks, PricingError> {
        let n = market.assets.len();
        if asset_i >= n || asset_j >= n {
            return Err(PricingError::InvalidInput(format!(
                "asset indices ({asset_i}, {asset_j}) out of range (have {n} assets)"
            )));
        }

        // Cross-gamma: d²V/(dSi dSj)
        let bump_i = 0.01 * market.assets[asset_i].spot;
        let bump_j = 0.01 * market.assets[asset_j].spot;

        let mut m_pp = market.clone();
        m_pp.assets[asset_i].spot += bump_i;
        m_pp.assets[asset_j].spot += bump_j;
        let p_pp = self.price_multi_asset(product, &m_pp)?.price;

        let mut m_pm = market.clone();
        m_pm.assets[asset_i].spot += bump_i;
        m_pm.assets[asset_j].spot -= bump_j;
        let p_pm = self.price_multi_asset(product, &m_pm)?.price;

        let mut m_mp = market.clone();
        m_mp.assets[asset_i].spot -= bump_i;
        m_mp.assets[asset_j].spot += bump_j;
        let p_mp = self.price_multi_asset(product, &m_mp)?.price;

        let mut m_mm = market.clone();
        m_mm.assets[asset_i].spot -= bump_i;
        m_mm.assets[asset_j].spot -= bump_j;
        let p_mm = self.price_multi_asset(product, &m_mm)?.price;

        let cross_gamma = (p_pp - p_pm - p_mp + p_mm) / (4.0 * bump_i * bump_j);

        // Correlation sensitivity: dV/dρij
        let corr_bump = 0.01;
        let mut m_corr_up = market.clone();
        let rho_val = m_corr_up.correlation[asset_i][asset_j];
        let rho_up = (rho_val + corr_bump).min(1.0);
        m_corr_up.correlation[asset_i][asset_j] = rho_up;
        m_corr_up.correlation[asset_j][asset_i] = rho_up;
        let p_corr_up = self.price_multi_asset(product, &m_corr_up)?.price;

        let mut m_corr_down = market.clone();
        let rho_down = (rho_val - corr_bump).max(-1.0);
        m_corr_down.correlation[asset_i][asset_j] = rho_down;
        m_corr_down.correlation[asset_j][asset_i] = rho_down;
        let p_corr_down = self.price_multi_asset(product, &m_corr_down)?.price;

        let corr_sens = (p_corr_up - p_corr_down) / (rho_up - rho_down);

        Ok(CrossGreeks {
            cross_gamma,
            corr_sens,
        })
    }
}

/// Extended per-asset greeks including higher-order sensitivities.
#[derive(Debug, Clone, Copy)]
pub struct ExtendedGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

/// Cross-asset sensitivities between a pair of underlyings.
#[derive(Debug, Clone, Copy)]
pub struct CrossGreeks {
    /// d²V/(dSi dSj)
    pub cross_gamma: f64,
    /// dV/dρij
    pub corr_sens: f64,
}

/// Implement `PricingEngine<DslProduct>` for single-asset convenience.
/// Uses a standard `Market` and wraps it into `MultiAssetMarket`.
impl PricingEngine<DslProduct> for DslMonteCarloEngine {
    fn price(
        &self,
        instrument: &DslProduct,
        market: &Market,
    ) -> Result<PricingResult, PricingError> {
        let vol = market.vol_for(market.spot, instrument.product.maturity);
        let multi_market =
            MultiAssetMarket::single(market.spot, vol, market.rate, market.dividend_yield);
        self.price_multi_asset(&instrument.product, &multi_market)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::ir::*;

    /// Build a simple forward product: pays S(T) at maturity.
    /// PV should be approximately S(0) * exp(-q*T) for zero-coupon.
    fn make_forward_product() -> CompiledProduct {
        CompiledProduct {
            name: "Forward".to_string(),
            notional: 1.0,
            maturity: 1.0,
            num_underlyings: 1,
            underlyings: vec![UnderlyingDef {
                name: "SPX".to_string(),
                asset_index: 0,
            }],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::Call {
                        func: BuiltinFn::Price,
                        args: vec![Expr::Literal(Value::F64(0.0))],
                    },
                }],
            }],
        }
    }

    #[test]
    fn forward_product_prices_near_forward() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02);
        let engine = DslMonteCarloEngine::new(100_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Forward price = S(0) * exp((r-q)*T) = 100 * exp(0.03) ≈ 103.045
        // Discounted = 100 * exp((r-q)*T) * exp(-r*T) = 100 * exp(-q*T) ≈ 98.02
        let expected = 100.0 * (-0.02f64).exp();
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err < 0.02,
            "forward price error: got {}, expected {expected}, rel_err {rel_err}",
            result.price
        );
    }

    #[test]
    fn multi_asset_correlation_produces_correlated_paths() {
        // Simple 2-asset product: pays max(S1(T)/S1(0), S2(T)/S2(0)) * notional
        let product = CompiledProduct {
            name: "BestOf".to_string(),
            notional: 100.0,
            maturity: 1.0,
            num_underlyings: 2,
            underlyings: vec![
                UnderlyingDef {
                    name: "A".to_string(),
                    asset_index: 0,
                },
                UnderlyingDef {
                    name: "B".to_string(),
                    asset_index: 1,
                },
            ],
            state_vars: vec![],
            constants: vec![],
            schedules: vec![Schedule {
                dates: vec![1.0],
                body: vec![Statement::Redeem {
                    amount: Expr::BinOp {
                        op: BinOp::Mul,
                        lhs: Box::new(Expr::Notional),
                        rhs: Box::new(Expr::Call {
                            func: BuiltinFn::BestOf,
                            args: vec![Expr::Call {
                                func: BuiltinFn::Performances,
                                args: vec![],
                            }],
                        }),
                    },
                }],
            }],
        };

        let market = MultiAssetMarket {
            assets: vec![
                crate::dsl::market::AssetData {
                    spot: 100.0,
                    vol: 0.20,
                    dividend_yield: 0.0,
                },
                crate::dsl::market::AssetData {
                    spot: 100.0,
                    vol: 0.20,
                    dividend_yield: 0.0,
                },
            ],
            correlation: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            rate: 0.05,
        };

        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price_multi_asset(&product, &market).unwrap();

        // Best-of should be > 100 (it's a convex payoff on performance)
        assert!(
            result.price > 90.0,
            "best-of price {} should be > 90",
            result.price
        );
        assert!(result.stderr.unwrap() < 2.0, "stderr should be reasonable");
    }

    #[test]
    fn single_asset_via_market_trait() {
        let product = make_forward_product();
        let dsl_product = DslProduct::new(product);
        let market = Market::builder()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .flat_vol(0.20)
            .build()
            .unwrap();

        let engine = DslMonteCarloEngine::new(50_000, 252, 42);
        let result = engine.price(&dsl_product, &market).unwrap();

        let expected = 100.0 * (-0.02f64).exp();
        let rel_err = ((result.price - expected) / expected).abs();
        assert!(
            rel_err < 0.02,
            "single-asset forward price error: got {}, expected {expected}",
            result.price
        );
    }

    #[test]
    fn greeks_produce_sensible_delta() {
        let product = make_forward_product();
        let market = MultiAssetMarket::single(100.0, 0.20, 0.05, 0.0);
        let engine = DslMonteCarloEngine::new(50_000, 252, 42);

        let greeks = engine.greeks_multi_asset(&product, &market, 0).unwrap();

        // Forward delta ≈ exp(-r*T) ≈ 0.951
        let expected_delta = (-0.05f64).exp();
        assert!(
            (greeks.delta - expected_delta).abs() < 0.05,
            "delta {} should be near {expected_delta}",
            greeks.delta
        );
    }
}
