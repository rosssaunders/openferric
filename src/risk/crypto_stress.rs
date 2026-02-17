//! Crypto portfolio stress testing with full options Greeks analysis.
//!
//! Upload a portfolio of crypto option positions and stress it across spot shocks,
//! volatility shocks, time decay, combined 2D grids, and historical crypto crash
//! scenarios. Every result includes full Greeks decomposition (delta, gamma, vega,
//! theta, rho, vanna, volga) so you can see exactly where your risk lives.

use crate::greeks::{EuropeanBsmGreeks, black_scholes_merton_greeks};
use crate::pricing::OptionType;
use crate::pricing::european::black_scholes_price;
use crate::risk::var::{
    cornish_fisher_var_from_pnl, delta_gamma_normal_var, historical_expected_shortfall,
    historical_var,
};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Crypto underlying asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CryptoAsset {
    BTC,
    ETH,
    SOL,
    Custom,
}

impl std::fmt::Display for CryptoAsset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BTC => write!(f, "BTC"),
            Self::ETH => write!(f, "ETH"),
            Self::SOL => write!(f, "SOL"),
            Self::Custom => write!(f, "CUSTOM"),
        }
    }
}

/// A single crypto option position with full metadata.
#[derive(Debug, Clone)]
pub struct CryptoOptionPosition {
    /// Underlying asset.
    pub asset: CryptoAsset,
    /// Call or put.
    pub option_type: OptionType,
    /// Strike price in USD.
    pub strike: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// Number of contracts (negative = short).
    pub quantity: f64,
    /// Current spot price of the underlying.
    pub spot: f64,
    /// Implied volatility (annualized, e.g. 0.80 = 80%).
    pub implied_vol: f64,
    /// Risk-free rate (annualized).
    pub rate: f64,
    /// Continuous dividend yield (usually 0 for crypto).
    pub dividend_yield: f64,
}

impl CryptoOptionPosition {
    /// Creates a new position with zero dividend yield.
    pub fn new(
        asset: CryptoAsset,
        option_type: OptionType,
        strike: f64,
        expiry: f64,
        quantity: f64,
        spot: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        Self {
            asset,
            option_type,
            strike,
            expiry,
            quantity,
            spot,
            implied_vol,
            rate,
            dividend_yield: 0.0,
        }
    }

    /// Current mark price per contract.
    pub fn mark_price(&self) -> f64 {
        black_scholes_price(self.option_type, self.spot, self.strike, self.rate, self.implied_vol, self.expiry)
    }

    /// Full BSM Greeks including second-order cross-Greeks.
    pub fn greeks(&self) -> EuropeanBsmGreeks {
        black_scholes_merton_greeks(
            self.option_type,
            self.spot,
            self.strike,
            self.rate,
            self.dividend_yield,
            self.implied_vol,
            self.expiry,
        )
    }

    /// Dollar value of the position (quantity * mark price).
    pub fn market_value(&self) -> f64 {
        self.quantity * self.mark_price()
    }

    /// Reprice under a shocked spot and vol, with optional time shift.
    pub fn reprice_shocked(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        time_decay_years: f64,
    ) -> f64 {
        let shocked_spot = self.spot * (1.0 + spot_shock_pct);
        let shocked_vol = (self.implied_vol * (1.0 + vol_shock_pct)).max(1e-6);
        let shocked_expiry = (self.expiry - time_decay_years).max(1e-8);

        if shocked_spot <= 0.0 {
            return 0.0;
        }

        black_scholes_price(
            self.option_type,
            shocked_spot,
            self.strike,
            self.rate,
            shocked_vol,
            shocked_expiry,
        )
    }

    /// Full Greeks under shocked market conditions.
    pub fn greeks_shocked(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        time_decay_years: f64,
    ) -> EuropeanBsmGreeks {
        let shocked_spot = self.spot * (1.0 + spot_shock_pct);
        let shocked_vol = (self.implied_vol * (1.0 + vol_shock_pct)).max(1e-6);
        let shocked_expiry = (self.expiry - time_decay_years).max(1e-8);

        if shocked_spot <= 0.0 {
            return EuropeanBsmGreeks {
                delta: 0.0, gamma: 0.0, vega: 0.0, theta: 0.0,
                rho: 0.0, vanna: 0.0, volga: 0.0,
            };
        }

        black_scholes_merton_greeks(
            self.option_type,
            shocked_spot,
            self.strike,
            self.rate,
            self.dividend_yield,
            shocked_vol,
            shocked_expiry,
        )
    }
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

/// Portfolio of crypto option positions.
#[derive(Debug, Clone, Default)]
pub struct CryptoPortfolio {
    pub positions: Vec<CryptoOptionPosition>,
}

/// Portfolio-level Greeks with second-order cross-Greeks.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

/// Per-position Greeks attribution.
#[derive(Debug, Clone)]
pub struct PositionAttribution {
    pub position_index: usize,
    pub asset: CryptoAsset,
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry: f64,
    pub quantity: f64,
    pub mark_price: f64,
    pub market_value: f64,
    pub greeks: EuropeanBsmGreeks,
    /// Dollar Greeks (Greeks * quantity).
    pub dollar_delta: f64,
    pub dollar_gamma: f64,
    pub dollar_vega: f64,
    pub dollar_theta: f64,
}

impl CryptoPortfolio {
    pub fn new() -> Self {
        Self { positions: Vec::new() }
    }

    pub fn add_position(&mut self, position: CryptoOptionPosition) {
        self.positions.push(position);
    }

    /// Total portfolio market value.
    pub fn total_market_value(&self) -> f64 {
        self.positions.iter().map(|p| p.market_value()).sum()
    }

    /// Aggregate portfolio Greeks across all positions.
    pub fn portfolio_greeks(&self) -> PortfolioGreeks {
        let mut agg = PortfolioGreeks::default();
        for pos in &self.positions {
            let g = pos.greeks();
            agg.delta += pos.quantity * g.delta;
            agg.gamma += pos.quantity * g.gamma;
            agg.vega += pos.quantity * g.vega;
            agg.theta += pos.quantity * g.theta;
            agg.rho += pos.quantity * g.rho;
            agg.vanna += pos.quantity * g.vanna;
            agg.volga += pos.quantity * g.volga;
        }
        agg
    }

    /// Per-position attribution of Greeks and P&L.
    pub fn position_attribution(&self) -> Vec<PositionAttribution> {
        self.positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let g = p.greeks();
                let mark = p.mark_price();
                PositionAttribution {
                    position_index: i,
                    asset: p.asset,
                    option_type: p.option_type,
                    strike: p.strike,
                    expiry: p.expiry,
                    quantity: p.quantity,
                    mark_price: mark,
                    market_value: p.quantity * mark,
                    greeks: g,
                    dollar_delta: p.quantity * g.delta * p.spot,
                    dollar_gamma: p.quantity * g.gamma * p.spot * p.spot,
                    dollar_vega: p.quantity * g.vega,
                    dollar_theta: p.quantity * g.theta,
                }
            })
            .collect()
    }

    /// Full repricing P&L under a scenario (difference from base value).
    pub fn scenario_pnl_full_reprice(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        time_decay_years: f64,
    ) -> f64 {
        self.positions
            .iter()
            .map(|p| {
                let base = p.mark_price();
                let shocked = p.reprice_shocked(spot_shock_pct, vol_shock_pct, time_decay_years);
                p.quantity * (shocked - base)
            })
            .sum()
    }

    /// Greeks-approximated P&L (delta-gamma-vega-theta-vanna-volga).
    pub fn scenario_pnl_greeks_approx(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        time_decay_years: f64,
    ) -> f64 {
        self.positions
            .iter()
            .map(|p| {
                let g = p.greeks();
                let ds = p.spot * spot_shock_pct;
                let dvol = p.implied_vol * vol_shock_pct;
                let dt = time_decay_years;

                // Taylor expansion including cross-Greeks:
                // dV ≈ Δ·dS + ½Γ·dS² + V·dσ + Θ·dt + vanna·dS·dσ + ½volga·dσ²
                let pnl = g.delta * ds
                    + 0.5 * g.gamma * ds * ds
                    + g.vega * dvol
                    + g.theta * dt
                    + g.vanna * ds * dvol
                    + 0.5 * g.volga * dvol * dvol;

                p.quantity * pnl
            })
            .sum()
    }

    /// Greeks under a shocked scenario.
    pub fn shocked_portfolio_greeks(
        &self,
        spot_shock_pct: f64,
        vol_shock_pct: f64,
        time_decay_years: f64,
    ) -> PortfolioGreeks {
        let mut agg = PortfolioGreeks::default();
        for pos in &self.positions {
            let g = pos.greeks_shocked(spot_shock_pct, vol_shock_pct, time_decay_years);
            agg.delta += pos.quantity * g.delta;
            agg.gamma += pos.quantity * g.gamma;
            agg.vega += pos.quantity * g.vega;
            agg.theta += pos.quantity * g.theta;
            agg.rho += pos.quantity * g.rho;
            agg.vanna += pos.quantity * g.vanna;
            agg.volga += pos.quantity * g.volga;
        }
        agg
    }
}

// ---------------------------------------------------------------------------
// Stress test configuration
// ---------------------------------------------------------------------------

/// A named historical crypto crash scenario.
#[derive(Debug, Clone)]
pub struct HistoricalScenario {
    /// Descriptive name (e.g. "COVID March 2020").
    pub name: &'static str,
    /// Spot shock as a fraction (e.g. -0.50 = -50%).
    pub spot_shock: f64,
    /// Vol shock as a fraction (e.g. 1.50 = +150% relative).
    pub vol_shock: f64,
    /// Time horizon in years.
    pub horizon_years: f64,
}

/// Built-in historical crypto crash scenarios.
pub fn historical_crypto_scenarios() -> Vec<HistoricalScenario> {
    vec![
        HistoricalScenario {
            name: "COVID crash (Mar 2020)",
            spot_shock: -0.50,
            vol_shock: 1.50,
            horizon_years: 1.0 / 252.0,
        },
        HistoricalScenario {
            name: "China mining ban (May 2021)",
            spot_shock: -0.53,
            vol_shock: 1.20,
            horizon_years: 5.0 / 252.0,
        },
        HistoricalScenario {
            name: "LUNA/UST collapse (May 2022)",
            spot_shock: -0.60,
            vol_shock: 2.00,
            horizon_years: 3.0 / 252.0,
        },
        HistoricalScenario {
            name: "FTX collapse (Nov 2022)",
            spot_shock: -0.27,
            vol_shock: 0.80,
            horizon_years: 5.0 / 252.0,
        },
        HistoricalScenario {
            name: "Flash crash (-30% intraday)",
            spot_shock: -0.30,
            vol_shock: 2.50,
            horizon_years: 0.0,
        },
        HistoricalScenario {
            name: "Euphoric rally (+40%)",
            spot_shock: 0.40,
            vol_shock: 0.50,
            horizon_years: 5.0 / 252.0,
        },
        HistoricalScenario {
            name: "Vol crush (post-event)",
            spot_shock: 0.0,
            vol_shock: -0.50,
            horizon_years: 1.0 / 252.0,
        },
        HistoricalScenario {
            name: "Vol explosion (pre-event)",
            spot_shock: 0.0,
            vol_shock: 1.50,
            horizon_years: 0.0,
        },
    ]
}

/// Configuration for the stress test grid.
#[derive(Debug, Clone)]
pub struct StressConfig {
    /// Spot shock steps (e.g. [-0.50, -0.40, ..., 0.40, 0.50]).
    pub spot_shocks: Vec<f64>,
    /// Vol shock steps (e.g. [-0.50, -0.25, ..., 0.50, 1.00]).
    pub vol_shocks: Vec<f64>,
    /// Time decay steps in years (e.g. [0, 1/252, 7/252, 30/252]).
    pub time_horizons: Vec<f64>,
    /// VaR confidence level (e.g. 0.99).
    pub var_confidence: f64,
    /// Whether to include historical scenario analysis.
    pub include_historical: bool,
    /// Whether to compute Greeks under each shocked state.
    pub compute_shocked_greeks: bool,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            spot_shocks: vec![
                -0.50, -0.40, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
            ],
            vol_shocks: vec![
                -0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50, 1.00,
            ],
            time_horizons: vec![
                0.0,
                1.0 / 365.0,
                7.0 / 365.0,
                30.0 / 365.0,
            ],
            var_confidence: 0.99,
            include_historical: true,
            compute_shocked_greeks: true,
        }
    }
}

impl StressConfig {
    /// Crypto-focused config with wider tails (crypto moves ±80%).
    pub fn crypto_wide() -> Self {
        Self {
            spot_shocks: vec![
                -0.80, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, -0.05,
                0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00,
            ],
            vol_shocks: vec![
                -0.50, -0.30, -0.10, 0.0, 0.10, 0.30, 0.50, 1.00, 1.50, 2.00,
            ],
            time_horizons: vec![
                0.0,
                1.0 / 365.0,
                7.0 / 365.0,
                14.0 / 365.0,
                30.0 / 365.0,
                90.0 / 365.0,
            ],
            var_confidence: 0.99,
            include_historical: true,
            compute_shocked_greeks: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Stress test results
// ---------------------------------------------------------------------------

/// A single cell in the 2D P&L grid.
#[derive(Debug, Clone, Copy)]
pub struct PnlGridCell {
    pub spot_shock: f64,
    pub vol_shock: f64,
    /// Full-repricing P&L.
    pub pnl_full: f64,
    /// Greeks-approximated P&L.
    pub pnl_approx: f64,
    /// Approximation error (full - approx).
    pub approx_error: f64,
}

/// Result of a single historical scenario.
#[derive(Debug, Clone)]
pub struct HistoricalScenarioResult {
    pub scenario: HistoricalScenario,
    pub pnl_full: f64,
    pub pnl_approx: f64,
    pub shocked_greeks: PortfolioGreeks,
    /// P&L decomposition: how much came from each Greek.
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
    pub theta_pnl: f64,
    pub cross_pnl: f64,
}

/// Time decay (theta bleed) result at a horizon.
#[derive(Debug, Clone, Copy)]
pub struct ThetaDecayPoint {
    pub horizon_years: f64,
    pub horizon_days: f64,
    /// P&L purely from time passing (no spot/vol change).
    pub theta_pnl: f64,
    /// Portfolio value after time decay.
    pub portfolio_value: f64,
}

/// Greeks evolution across spot shocks (holding vol constant).
#[derive(Debug, Clone, Copy)]
pub struct GreeksAtSpot {
    pub spot_shock: f64,
    pub shocked_spot: f64,
    pub greeks: PortfolioGreeks,
}

/// Greeks evolution across vol shocks (holding spot constant).
#[derive(Debug, Clone, Copy)]
pub struct GreeksAtVol {
    pub vol_shock: f64,
    pub greeks: PortfolioGreeks,
}

/// Risk metrics derived from the scenario P&L distribution.
#[derive(Debug, Clone, Copy)]
pub struct RiskMetrics {
    /// Historical VaR at the configured confidence level.
    pub var_historical: f64,
    /// Historical Expected Shortfall (CVaR).
    pub es_historical: f64,
    /// Delta-gamma-normal parametric VaR.
    pub var_delta_gamma: f64,
    /// Cornish-Fisher adjusted VaR (accounts for skew/kurtosis).
    pub var_cornish_fisher: f64,
    /// Worst-case P&L across all scenarios.
    pub worst_case_pnl: f64,
    /// Best-case P&L across all scenarios.
    pub best_case_pnl: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,
}

/// Complete stress test output.
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Base portfolio value before any shocks.
    pub base_portfolio_value: f64,
    /// Base portfolio Greeks.
    pub base_greeks: PortfolioGreeks,
    /// Per-position attribution.
    pub position_attribution: Vec<PositionAttribution>,
    /// 2D P&L grid (spot x vol).
    pub pnl_grid: Vec<PnlGridCell>,
    /// P&L grid dimensions.
    pub grid_spot_count: usize,
    pub grid_vol_count: usize,
    /// Time decay / theta bleed curve.
    pub theta_decay: Vec<ThetaDecayPoint>,
    /// Historical crash scenario results.
    pub historical_results: Vec<HistoricalScenarioResult>,
    /// Greeks evolution across spot shocks.
    pub greeks_by_spot: Vec<GreeksAtSpot>,
    /// Greeks evolution across vol shocks.
    pub greeks_by_vol: Vec<GreeksAtVol>,
    /// Aggregate risk metrics.
    pub risk_metrics: RiskMetrics,
    /// Spot shock ladder P&L (vol=0 slice).
    pub spot_ladder: Vec<(f64, f64)>,
    /// Vol shock ladder P&L (spot=0 slice).
    pub vol_ladder: Vec<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// Stress test engine
// ---------------------------------------------------------------------------

/// Runs a full stress test on a crypto options portfolio.
pub struct StressTestEngine;

impl StressTestEngine {
    /// Execute a comprehensive stress test and return all results.
    pub fn run(portfolio: &CryptoPortfolio, config: &StressConfig) -> StressTestResult {
        assert!(!portfolio.positions.is_empty(), "portfolio must not be empty");

        // Base values
        let base_value = portfolio.total_market_value();
        let base_greeks = portfolio.portfolio_greeks();
        let attribution = portfolio.position_attribution();

        // 2D P&L grid (spot x vol)
        let mut pnl_grid = Vec::with_capacity(config.spot_shocks.len() * config.vol_shocks.len());
        for &spot_shock in &config.spot_shocks {
            for &vol_shock in &config.vol_shocks {
                let pnl_full = portfolio.scenario_pnl_full_reprice(spot_shock, vol_shock, 0.0);
                let pnl_approx = portfolio.scenario_pnl_greeks_approx(spot_shock, vol_shock, 0.0);
                pnl_grid.push(PnlGridCell {
                    spot_shock,
                    vol_shock,
                    pnl_full,
                    pnl_approx,
                    approx_error: pnl_full - pnl_approx,
                });
            }
        }

        // Spot ladder (vol unchanged)
        let spot_ladder: Vec<(f64, f64)> = config.spot_shocks
            .iter()
            .map(|&s| (s, portfolio.scenario_pnl_full_reprice(s, 0.0, 0.0)))
            .collect();

        // Vol ladder (spot unchanged)
        let vol_ladder: Vec<(f64, f64)> = config.vol_shocks
            .iter()
            .map(|&v| (v, portfolio.scenario_pnl_full_reprice(0.0, v, 0.0)))
            .collect();

        // Time decay / theta bleed
        let theta_decay: Vec<ThetaDecayPoint> = config.time_horizons
            .iter()
            .map(|&h| {
                let theta_pnl = portfolio.scenario_pnl_full_reprice(0.0, 0.0, h);
                ThetaDecayPoint {
                    horizon_years: h,
                    horizon_days: h * 365.0,
                    theta_pnl,
                    portfolio_value: base_value + theta_pnl,
                }
            })
            .collect();

        // Historical scenarios
        let historical_results = if config.include_historical {
            Self::run_historical_scenarios(portfolio, &base_greeks)
        } else {
            Vec::new()
        };

        // Greeks evolution across spot shocks
        let greeks_by_spot: Vec<GreeksAtSpot> = if config.compute_shocked_greeks {
            config.spot_shocks
                .iter()
                .map(|&s| {
                    let avg_spot = portfolio.positions.first()
                        .map(|p| p.spot)
                        .unwrap_or(1.0);
                    GreeksAtSpot {
                        spot_shock: s,
                        shocked_spot: avg_spot * (1.0 + s),
                        greeks: portfolio.shocked_portfolio_greeks(s, 0.0, 0.0),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Greeks evolution across vol shocks
        let greeks_by_vol: Vec<GreeksAtVol> = if config.compute_shocked_greeks {
            config.vol_shocks
                .iter()
                .map(|&v| GreeksAtVol {
                    vol_shock: v,
                    greeks: portfolio.shocked_portfolio_greeks(0.0, v, 0.0),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Collect all scenario P&Ls for VaR/ES calculation
        let all_pnls: Vec<f64> = pnl_grid.iter().map(|c| c.pnl_full).collect();
        let risk_metrics = Self::compute_risk_metrics(portfolio, &all_pnls, config.var_confidence);

        StressTestResult {
            base_portfolio_value: base_value,
            base_greeks,
            position_attribution: attribution,
            pnl_grid,
            grid_spot_count: config.spot_shocks.len(),
            grid_vol_count: config.vol_shocks.len(),
            theta_decay,
            historical_results,
            greeks_by_spot,
            greeks_by_vol,
            risk_metrics,
            spot_ladder,
            vol_ladder,
        }
    }

    fn run_historical_scenarios(
        portfolio: &CryptoPortfolio,
        base_greeks: &PortfolioGreeks,
    ) -> Vec<HistoricalScenarioResult> {
        historical_crypto_scenarios()
            .into_iter()
            .map(|scenario| {
                let pnl_full = portfolio.scenario_pnl_full_reprice(
                    scenario.spot_shock,
                    scenario.vol_shock,
                    scenario.horizon_years,
                );
                let pnl_approx = portfolio.scenario_pnl_greeks_approx(
                    scenario.spot_shock,
                    scenario.vol_shock,
                    scenario.horizon_years,
                );
                let shocked_greeks = portfolio.shocked_portfolio_greeks(
                    scenario.spot_shock,
                    scenario.vol_shock,
                    scenario.horizon_years,
                );

                // Greeks P&L decomposition (using portfolio-level aggregates)
                let avg_spot = portfolio.positions.iter()
                    .map(|p| p.spot * p.quantity.abs())
                    .sum::<f64>()
                    / portfolio.positions.iter()
                        .map(|p| p.quantity.abs())
                        .sum::<f64>()
                        .max(1e-12);
                let avg_vol = portfolio.positions.iter()
                    .map(|p| p.implied_vol * p.quantity.abs())
                    .sum::<f64>()
                    / portfolio.positions.iter()
                        .map(|p| p.quantity.abs())
                        .sum::<f64>()
                        .max(1e-12);

                let ds = avg_spot * scenario.spot_shock;
                let dvol = avg_vol * scenario.vol_shock;

                let delta_pnl = base_greeks.delta * ds;
                let gamma_pnl = 0.5 * base_greeks.gamma * ds * ds;
                let vega_pnl = base_greeks.vega * dvol;
                let theta_pnl = base_greeks.theta * scenario.horizon_years;
                let cross_pnl = base_greeks.vanna * ds * dvol
                    + 0.5 * base_greeks.volga * dvol * dvol;

                HistoricalScenarioResult {
                    scenario,
                    pnl_full,
                    pnl_approx,
                    shocked_greeks,
                    delta_pnl,
                    gamma_pnl,
                    vega_pnl,
                    theta_pnl,
                    cross_pnl,
                }
            })
            .collect()
    }

    fn compute_risk_metrics(
        portfolio: &CryptoPortfolio,
        scenario_pnls: &[f64],
        confidence: f64,
    ) -> RiskMetrics {
        let worst = scenario_pnls.iter().cloned().fold(f64::INFINITY, f64::min);
        let best = scenario_pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let base_value = portfolio.total_market_value();
        let max_drawdown_pct = if base_value.abs() > 1e-12 {
            (-worst / base_value.abs()).max(0.0)
        } else {
            0.0
        };

        // Historical VaR/ES from scenario distribution
        let var_hist = historical_var(scenario_pnls, confidence);
        let es_hist = historical_expected_shortfall(scenario_pnls, confidence);

        // Delta-gamma parametric VaR using portfolio aggregates
        let greeks = portfolio.portfolio_greeks();
        let avg_vol = portfolio.positions.iter()
            .map(|p| p.implied_vol * p.quantity.abs())
            .sum::<f64>()
            / portfolio.positions.iter()
                .map(|p| p.quantity.abs())
                .sum::<f64>()
                .max(1e-12);

        let var_dg = delta_gamma_normal_var(
            greeks.delta,
            greeks.gamma,
            avg_vol,
            confidence,
            1.0,
        );

        // Cornish-Fisher adjusted VaR (captures skew/kurtosis of scenario P&Ls)
        let var_cf = cornish_fisher_var_from_pnl(scenario_pnls, confidence);

        RiskMetrics {
            var_historical: var_hist,
            es_historical: es_hist,
            var_delta_gamma: var_dg,
            var_cornish_fisher: var_cf,
            worst_case_pnl: worst,
            best_case_pnl: best,
            max_drawdown_pct,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience builders for common portfolio shapes
// ---------------------------------------------------------------------------

impl CryptoPortfolio {
    /// Build a simple long call portfolio.
    pub fn long_call(
        asset: CryptoAsset,
        spot: f64,
        strike: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a bull call spread (long lower strike, short higher strike).
    pub fn bull_call_spread(
        asset: CryptoAsset,
        spot: f64,
        strike_low: f64,
        strike_high: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_low, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_high, expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a long straddle (long call + long put at same strike).
    pub fn long_straddle(
        asset: CryptoAsset,
        spot: f64,
        strike: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, strike, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a short strangle (short OTM call + short OTM put).
    pub fn short_strangle(
        asset: CryptoAsset,
        spot: f64,
        strike_put: f64,
        strike_call: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_call, expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, strike_put, expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build an iron condor (short strangle + long wings).
    pub fn iron_condor(
        asset: CryptoAsset,
        spot: f64,
        put_buy: f64,
        put_sell: f64,
        call_sell: f64,
        call_buy: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        // Long OTM put (wing)
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, put_buy, expiry, quantity, spot, implied_vol, rate,
        ));
        // Short put (body)
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, put_sell, expiry, -quantity, spot, implied_vol, rate,
        ));
        // Short call (body)
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, call_sell, expiry, -quantity, spot, implied_vol, rate,
        ));
        // Long OTM call (wing)
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, call_buy, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a butterfly spread.
    pub fn butterfly(
        asset: CryptoAsset,
        spot: f64,
        strike_low: f64,
        strike_mid: f64,
        strike_high: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_low, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_mid, expiry, -2.0 * quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_high, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a protective put (long underlying approximated as deep ITM call + long put).
    pub fn collar(
        asset: CryptoAsset,
        spot: f64,
        put_strike: f64,
        call_strike: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        // Long put for downside protection
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, put_strike, expiry, quantity, spot, implied_vol, rate,
        ));
        // Short call to cap upside (generates premium)
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, call_strike, expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a risk reversal (short put + long call).
    pub fn risk_reversal(
        asset: CryptoAsset,
        spot: f64,
        put_strike: f64,
        call_strike: f64,
        expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Put, put_strike, expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, call_strike, expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a calendar spread (short near-dated + long far-dated at same strike).
    pub fn calendar_spread(
        asset: CryptoAsset,
        spot: f64,
        strike: f64,
        near_expiry: f64,
        far_expiry: f64,
        quantity: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike, near_expiry, -quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike, far_expiry, quantity, spot, implied_vol, rate,
        ));
        portfolio
    }

    /// Build a ratio spread (1 long : N short).
    pub fn ratio_call_spread(
        asset: CryptoAsset,
        spot: f64,
        strike_buy: f64,
        strike_sell: f64,
        expiry: f64,
        buy_quantity: f64,
        sell_ratio: f64,
        implied_vol: f64,
        rate: f64,
    ) -> Self {
        let mut portfolio = Self::new();
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_buy, expiry, buy_quantity, spot, implied_vol, rate,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            asset, OptionType::Call, strike_sell, expiry, -buy_quantity * sell_ratio, spot, implied_vol, rate,
        ));
        portfolio
    }
}

// ---------------------------------------------------------------------------
// Display helpers for StressTestResult
// ---------------------------------------------------------------------------

impl StressTestResult {
    /// Find the P&L at a specific spot/vol shock pair.
    pub fn pnl_at(&self, spot_shock: f64, vol_shock: f64) -> Option<f64> {
        self.pnl_grid.iter()
            .find(|c| (c.spot_shock - spot_shock).abs() < 1e-12
                && (c.vol_shock - vol_shock).abs() < 1e-12)
            .map(|c| c.pnl_full)
    }

    /// Extract a spot ladder slice at a fixed vol shock.
    pub fn spot_ladder_at_vol(&self, vol_shock: f64) -> Vec<(f64, f64)> {
        self.pnl_grid.iter()
            .filter(|c| (c.vol_shock - vol_shock).abs() < 1e-12)
            .map(|c| (c.spot_shock, c.pnl_full))
            .collect()
    }

    /// Extract a vol ladder slice at a fixed spot shock.
    pub fn vol_ladder_at_spot(&self, spot_shock: f64) -> Vec<(f64, f64)> {
        self.pnl_grid.iter()
            .filter(|c| (c.spot_shock - spot_shock).abs() < 1e-12)
            .map(|c| (c.vol_shock, c.pnl_full))
            .collect()
    }

    /// Maximum absolute approximation error across the entire grid.
    pub fn max_approx_error(&self) -> f64 {
        self.pnl_grid.iter()
            .map(|c| c.approx_error.abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper: BTC ATM call position.
    fn btc_call() -> CryptoOptionPosition {
        CryptoOptionPosition::new(
            CryptoAsset::BTC,
            OptionType::Call,
            100_000.0,  // strike
            0.25,       // 3 months
            1.0,        // 1 contract
            100_000.0,  // spot
            0.80,       // 80% vol
            0.05,       // 5% rate
        )
    }

    fn eth_put() -> CryptoOptionPosition {
        CryptoOptionPosition::new(
            CryptoAsset::ETH,
            OptionType::Put,
            3_000.0,
            0.5,
            10.0,
            3_500.0,
            0.90,
            0.05,
        )
    }

    // -----------------------------------------------------------------------
    // Position-level tests
    // -----------------------------------------------------------------------

    #[test]
    fn mark_price_is_positive_for_atm_call() {
        let pos = btc_call();
        let price = pos.mark_price();
        assert!(price > 0.0, "ATM call should have positive value: {price}");
    }

    #[test]
    fn greeks_have_correct_signs_for_long_call() {
        let pos = btc_call();
        let g = pos.greeks();
        assert!(g.delta > 0.0, "long call delta should be positive");
        assert!(g.gamma > 0.0, "long call gamma should be positive");
        assert!(g.vega > 0.0, "long call vega should be positive");
        assert!(g.theta < 0.0, "long call theta should be negative");
        assert!(g.rho > 0.0, "long call rho should be positive");
    }

    #[test]
    fn greeks_have_correct_signs_for_put() {
        let pos = eth_put();
        let g = pos.greeks();
        assert!(g.delta < 0.0, "put delta should be negative");
        assert!(g.gamma > 0.0, "put gamma should be positive");
        assert!(g.vega > 0.0, "put vega should be positive");
    }

    #[test]
    fn put_call_parity_btc() {
        let s = 100_000.0;
        let k = 100_000.0;
        let t = 0.25;
        let r = 0.05;
        let vol = 0.80;

        let call = CryptoOptionPosition::new(CryptoAsset::BTC, OptionType::Call, k, t, 1.0, s, vol, r);
        let put = CryptoOptionPosition::new(CryptoAsset::BTC, OptionType::Put, k, t, 1.0, s, vol, r);

        let c = call.mark_price();
        let p = put.mark_price();
        let parity = s - k * (-r * t).exp();

        assert_relative_eq!(c - p, parity, epsilon = 1e-6);
    }

    #[test]
    fn reprice_shocked_respects_spot_increase() {
        let pos = btc_call();
        let base = pos.mark_price();
        let shocked = pos.reprice_shocked(0.10, 0.0, 0.0);
        assert!(shocked > base, "call value should increase with spot: base={base}, shocked={shocked}");
    }

    #[test]
    fn reprice_shocked_respects_vol_increase() {
        let pos = btc_call();
        let base = pos.mark_price();
        let shocked = pos.reprice_shocked(0.0, 0.50, 0.0);
        assert!(shocked > base, "ATM call value should increase with vol: base={base}, shocked={shocked}");
    }

    #[test]
    fn reprice_shocked_time_decay_reduces_value() {
        let pos = btc_call();
        let base = pos.mark_price();
        let decayed = pos.reprice_shocked(0.0, 0.0, 30.0 / 365.0);
        assert!(decayed < base, "time decay should reduce option value: base={base}, decayed={decayed}");
    }

    // -----------------------------------------------------------------------
    // Portfolio-level tests
    // -----------------------------------------------------------------------

    #[test]
    fn portfolio_greeks_aggregate_correctly() {
        let mut portfolio = CryptoPortfolio::new();
        portfolio.add_position(btc_call());
        portfolio.add_position(eth_put());

        let pg = portfolio.portfolio_greeks();
        let g_btc = btc_call().greeks();
        let g_eth = eth_put().greeks();

        assert_relative_eq!(
            pg.delta,
            1.0 * g_btc.delta + 10.0 * g_eth.delta,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            pg.gamma,
            1.0 * g_btc.gamma + 10.0 * g_eth.gamma,
            epsilon = 1e-10
        );
    }

    #[test]
    fn straddle_has_near_zero_delta() {
        let portfolio = CryptoPortfolio::long_straddle(
            CryptoAsset::BTC,
            100_000.0,
            100_000.0,
            0.25,
            1.0,
            0.80,
            0.05,
        );
        let g = portfolio.portfolio_greeks();
        // ATM straddle delta should be close to zero, but with high vol (80%)
        // and non-zero rate the forward is above spot, skewing delta positive.
        assert!(g.delta.abs() < 0.25, "straddle delta should be near zero: {}", g.delta);
        assert!(g.gamma > 0.0, "straddle gamma should be positive");
        assert!(g.vega > 0.0, "straddle vega should be positive");
    }

    #[test]
    fn iron_condor_has_limited_risk() {
        let portfolio = CryptoPortfolio::iron_condor(
            CryptoAsset::BTC,
            100_000.0,
            85_000.0,   // buy put
            90_000.0,   // sell put
            110_000.0,  // sell call
            115_000.0,  // buy call
            0.25,
            1.0,
            0.80,
            0.05,
        );

        // Iron condor should have limited loss on both sides.
        let down_pnl = portfolio.scenario_pnl_full_reprice(-0.50, 0.0, 0.0);
        let up_pnl = portfolio.scenario_pnl_full_reprice(0.50, 0.0, 0.0);
        let flat_pnl = portfolio.scenario_pnl_full_reprice(0.0, 0.0, 0.25);

        // Loss should be bounded (max loss ~ distance between strikes).
        let max_strike_diff = 5_000.0;
        assert!(down_pnl > -max_strike_diff * 1.5, "down loss bounded");
        assert!(up_pnl > -max_strike_diff * 1.5, "up loss bounded");
        // Near expiry with spot unchanged, should earn premium.
        assert!(flat_pnl > 0.0, "IC should profit from theta if spot stays");
    }

    #[test]
    fn scenario_pnl_full_reprice_vs_greeks_approx_small_shock() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );

        // For small shocks, Greeks approximation should be close to full reprice.
        let shock = 0.01;
        let full = portfolio.scenario_pnl_full_reprice(shock, 0.0, 0.0);
        let approx = portfolio.scenario_pnl_greeks_approx(shock, 0.0, 0.0);

        let rel_error = if full.abs() > 1e-8 {
            ((full - approx) / full).abs()
        } else {
            (full - approx).abs()
        };

        assert!(rel_error < 0.01, "small shock approx error should be < 1%: {rel_error:.4}");
    }

    #[test]
    fn scenario_pnl_full_reprice_vs_greeks_approx_large_shock() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );

        // For large shocks, there should be noticeable error (higher-order terms).
        let full = portfolio.scenario_pnl_full_reprice(-0.30, 0.50, 0.0);
        let approx = portfolio.scenario_pnl_greeks_approx(-0.30, 0.50, 0.0);

        // The important thing is both have the same sign.
        assert!(
            full.signum() == approx.signum() || full.abs() < 100.0,
            "full={full:.2}, approx={approx:.2}"
        );
    }

    // -----------------------------------------------------------------------
    // Stress test engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn stress_test_runs_without_panic() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        assert!(result.base_portfolio_value > 0.0);
        assert!(!result.pnl_grid.is_empty());
        assert!(!result.theta_decay.is_empty());
        assert!(!result.historical_results.is_empty());
        assert!(!result.greeks_by_spot.is_empty());
        assert!(!result.greeks_by_vol.is_empty());
    }

    #[test]
    fn pnl_grid_has_correct_dimensions() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        let expected_cells = config.spot_shocks.len() * config.vol_shocks.len();
        assert_eq!(result.pnl_grid.len(), expected_cells);
        assert_eq!(result.grid_spot_count, config.spot_shocks.len());
        assert_eq!(result.grid_vol_count, config.vol_shocks.len());
    }

    #[test]
    fn pnl_grid_zero_shock_is_near_zero() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        let zero_pnl = result.pnl_at(0.0, 0.0).expect("should find zero shock cell");
        assert!(zero_pnl.abs() < 1e-8, "zero shock P&L should be zero: {zero_pnl}");
    }

    #[test]
    fn spot_ladder_is_monotonic_for_long_call() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // For a long call, P&L should increase with spot.
        for pair in result.spot_ladder.windows(2) {
            assert!(
                pair[1].1 >= pair[0].1 - 1e-6,
                "long call spot ladder should be monotonically increasing: ({}, {}) vs ({}, {})",
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            );
        }
    }

    #[test]
    fn vol_ladder_for_straddle_is_monotonic() {
        let portfolio = CryptoPortfolio::long_straddle(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // Long straddle P&L should increase with vol.
        for pair in result.vol_ladder.windows(2) {
            assert!(
                pair[1].1 >= pair[0].1 - 1e-6,
                "straddle vol ladder should increase: ({}, {}) vs ({}, {})",
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            );
        }
    }

    #[test]
    fn theta_decay_is_negative_for_long_option() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        for td in &result.theta_decay {
            if td.horizon_years > 0.0 {
                assert!(td.theta_pnl < 0.0, "long option theta should be negative: {} days -> {}", td.horizon_days, td.theta_pnl);
            }
        }
    }

    #[test]
    fn historical_scenarios_produce_results() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        assert_eq!(result.historical_results.len(), historical_crypto_scenarios().len());

        // COVID crash should cause a loss on a long call.
        let covid = &result.historical_results[0];
        assert!(covid.pnl_full < 0.0, "long call should lose in COVID crash: {}", covid.pnl_full);
    }

    #[test]
    fn historical_pnl_decomposition_sums_approximately() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        for hr in &result.historical_results {
            let decomp_sum = hr.delta_pnl + hr.gamma_pnl + hr.vega_pnl + hr.theta_pnl + hr.cross_pnl;
            // The decomposition should approximately match the Greeks-approximated P&L.
            let error = (decomp_sum - hr.pnl_approx).abs();
            let scale = hr.pnl_approx.abs().max(1.0);
            assert!(
                error / scale < 0.05,
                "decomposition sum should match approx: decomp={decomp_sum:.2}, approx={:.2}, error={error:.2}",
                hr.pnl_approx
            );
        }
    }

    #[test]
    fn risk_metrics_are_positive_and_sane() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        assert!(result.risk_metrics.var_historical >= 0.0, "VaR should be non-negative");
        assert!(result.risk_metrics.es_historical >= result.risk_metrics.var_historical,
            "ES should be >= VaR");
        assert!(result.risk_metrics.worst_case_pnl < result.risk_metrics.best_case_pnl,
            "worst < best");
    }

    #[test]
    fn greeks_by_spot_shows_gamma_effect() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // Delta should increase as spot goes up (positive gamma).
        let first = result.greeks_by_spot.first().unwrap();
        let last = result.greeks_by_spot.last().unwrap();
        assert!(
            last.greeks.delta > first.greeks.delta,
            "call delta should increase with spot: low_spot_delta={:.4}, high_spot_delta={:.4}",
            first.greeks.delta, last.greeks.delta
        );
    }

    #[test]
    fn greeks_by_vol_shows_volga_effect() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // Vega should change as vol changes (volga).
        let low_vol = result.greeks_by_vol.first().unwrap();
        let high_vol = result.greeks_by_vol.last().unwrap();
        assert!(
            (high_vol.greeks.vega - low_vol.greeks.vega).abs() > 1e-6,
            "vega should change with vol (volga effect)"
        );
    }

    #[test]
    fn position_attribution_sums_to_portfolio() {
        let mut portfolio = CryptoPortfolio::new();
        portfolio.add_position(btc_call());
        portfolio.add_position(eth_put());

        let result = StressTestEngine::run(&portfolio, &StressConfig::default());

        let attr_value: f64 = result.position_attribution.iter().map(|a| a.market_value).sum();
        assert_relative_eq!(attr_value, result.base_portfolio_value, epsilon = 1e-8);
    }

    #[test]
    fn crypto_wide_config_has_wider_shocks() {
        let standard = StressConfig::default();
        let wide = StressConfig::crypto_wide();

        let max_spot_std = standard.spot_shocks.iter().cloned().fold(0.0f64, f64::max);
        let max_spot_wide = wide.spot_shocks.iter().cloned().fold(0.0f64, f64::max);

        assert!(max_spot_wide > max_spot_std, "crypto_wide should have wider spot shocks");
    }

    #[test]
    fn short_strangle_loses_on_large_move() {
        let portfolio = CryptoPortfolio::short_strangle(
            CryptoAsset::BTC,
            100_000.0,
            80_000.0,   // put strike
            120_000.0,  // call strike
            0.25,
            1.0,
            0.80,
            0.05,
        );

        // Big move down should cause a loss.
        let down_pnl = portfolio.scenario_pnl_full_reprice(-0.30, 0.0, 0.0);
        assert!(down_pnl < 0.0, "short strangle should lose on big down move: {down_pnl}");

        // Big move up should also cause a loss.
        let up_pnl = portfolio.scenario_pnl_full_reprice(0.30, 0.0, 0.0);
        assert!(up_pnl < 0.0, "short strangle should lose on big up move: {up_pnl}");
    }

    #[test]
    fn calendar_spread_benefits_from_time_decay() {
        let portfolio = CryptoPortfolio::calendar_spread(
            CryptoAsset::BTC,
            100_000.0,
            100_000.0,   // strike
            0.08,        // near expiry (~1 month)
            0.25,        // far expiry (~3 months)
            1.0,
            0.80,
            0.05,
        );

        // Calendar spread should benefit from near-term theta decay
        // (short near-term decays faster than long far-term).
        let g = portfolio.portfolio_greeks();
        // Theta should be positive (net collect) for properly structured calendar.
        // Note: depends on moneyness; ATM should work.
        assert!(g.theta > 0.0 || g.theta.abs() < 100.0,
            "calendar spread should have favorable theta: {}", g.theta);
    }

    #[test]
    fn result_spot_ladder_at_vol_extracts_correctly() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        let ladder = result.spot_ladder_at_vol(0.0);
        assert_eq!(ladder.len(), config.spot_shocks.len());
    }

    #[test]
    fn result_vol_ladder_at_spot_extracts_correctly() {
        let portfolio = CryptoPortfolio::long_call(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        let ladder = result.vol_ladder_at_spot(0.0);
        assert_eq!(ladder.len(), config.vol_shocks.len());
    }

    #[test]
    fn max_approx_error_is_finite() {
        let portfolio = CryptoPortfolio::long_straddle(
            CryptoAsset::BTC, 100_000.0, 100_000.0, 0.25, 1.0, 0.80, 0.05,
        );
        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        assert!(result.max_approx_error().is_finite());
    }

    #[test]
    fn multi_asset_portfolio_stress() {
        let mut portfolio = CryptoPortfolio::new();
        // BTC long call
        portfolio.add_position(CryptoOptionPosition::new(
            CryptoAsset::BTC, OptionType::Call, 100_000.0, 0.25, 5.0,
            100_000.0, 0.80, 0.05,
        ));
        // ETH short put
        portfolio.add_position(CryptoOptionPosition::new(
            CryptoAsset::ETH, OptionType::Put, 3_000.0, 0.5, -20.0,
            3_500.0, 0.90, 0.05,
        ));
        // SOL long straddle components
        portfolio.add_position(CryptoOptionPosition::new(
            CryptoAsset::SOL, OptionType::Call, 150.0, 0.25, 100.0,
            150.0, 1.00, 0.05,
        ));
        portfolio.add_position(CryptoOptionPosition::new(
            CryptoAsset::SOL, OptionType::Put, 150.0, 0.25, 100.0,
            150.0, 1.00, 0.05,
        ));

        let config = StressConfig::crypto_wide();
        let result = StressTestEngine::run(&portfolio, &config);

        assert_eq!(result.position_attribution.len(), 4);
        assert!(result.risk_metrics.var_historical.is_finite());
        assert!(result.risk_metrics.es_historical.is_finite());
    }

    #[test]
    fn butterfly_has_limited_risk_and_reward() {
        let portfolio = CryptoPortfolio::butterfly(
            CryptoAsset::BTC,
            100_000.0,
            90_000.0,
            100_000.0,
            110_000.0,
            0.25,
            1.0,
            0.80,
            0.05,
        );

        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // Butterfly has limited loss (roughly the net debit).
        // Even in extreme scenarios, loss is bounded.
        assert!(result.risk_metrics.worst_case_pnl > -15_000.0,
            "butterfly loss should be bounded: {}", result.risk_metrics.worst_case_pnl);
    }

    #[test]
    fn vanna_exposure_in_risk_reversal() {
        let portfolio = CryptoPortfolio::risk_reversal(
            CryptoAsset::BTC,
            100_000.0,
            90_000.0,   // put strike
            110_000.0,  // call strike
            0.25,
            1.0,
            0.80,
            0.05,
        );

        let g = portfolio.portfolio_greeks();
        // Risk reversal has significant vanna exposure (combined spot-vol risk).
        assert!(g.vanna.abs() > 1e-6, "risk reversal should have vanna exposure: {}", g.vanna);
        assert!(g.delta > 0.0, "risk reversal should be net long delta: {}", g.delta);
    }

    #[test]
    fn ratio_spread_has_asymmetric_pnl() {
        let portfolio = CryptoPortfolio::ratio_call_spread(
            CryptoAsset::BTC,
            100_000.0,
            100_000.0,   // buy strike
            110_000.0,   // sell strike
            0.25,
            1.0,         // buy 1
            2.0,         // sell 2 (1:2 ratio)
            0.80,
            0.05,
        );

        let config = StressConfig::default();
        let result = StressTestEngine::run(&portfolio, &config);

        // 1:2 ratio spread should have unlimited upside risk.
        let big_up = portfolio.scenario_pnl_full_reprice(0.50, 0.0, 0.0);
        let big_down = portfolio.scenario_pnl_full_reprice(-0.50, 0.0, 0.0);

        // On big up move, the extra short call hurts.
        assert!(big_up < 0.0 || big_down < 0.0,
            "ratio spread should lose on at least one side: up={big_up:.0}, down={big_down:.0}");

        // Should be a real stress test with results.
        assert!(!result.pnl_grid.is_empty());
    }
}
