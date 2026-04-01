#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod credit;
mod fft;
mod funding;
mod helpers;
mod pricing;
mod rates;
mod risk;
mod vol;
mod calibration;
mod core;
mod engines;
mod instruments;
mod market;
mod math_bindings;
mod mc;
mod models;

#[pymodule]
pub fn openferric(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pricing::py_bs_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_bs_greeks, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_barrier_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_american_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_heston_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_fx_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_digital_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_spread_price, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_lookback_floating, module)?)?;
    module.add_function(wrap_pyfunction!(pricing::py_lookback_fixed, module)?)?;
    module.add_function(wrap_pyfunction!(vol::py_implied_vol, module)?)?;
    module.add_function(wrap_pyfunction!(vol::py_sabr_vol, module)?)?;
    module.add_function(wrap_pyfunction!(credit::py_cds_npv, module)?)?;
    module.add_function(wrap_pyfunction!(credit::py_survival_prob, module)?)?;
    module.add_function(wrap_pyfunction!(fft::py_heston_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(fft::py_heston_fft_prices, module)?)?;
    module.add_function(wrap_pyfunction!(fft::py_vg_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(fft::py_cgmy_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(fft::py_nig_fft_price, module)?)?;
    module.add_function(wrap_pyfunction!(rates::py_swaption_price, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_cva, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_sa_ccr_ead, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_historical_var, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_historical_es, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_delta_normal_var, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_delta_gamma_normal_var, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_normal_expected_shortfall,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_cornish_fisher_var, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_cornish_fisher_var_from_pnl,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_historical_var_from_prices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_historical_expected_shortfall_from_prices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_rolling_historical_var_from_prices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_backtest_historical_var_from_prices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_funding_exposure_profile, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_fva_from_profile, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_mva_from_profile, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_regulatory_capital, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_kva_from_profile, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_netting_set_exposure, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_parallel_dv01, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_bucket_dv01, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_key_rate_duration, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_gamma_ladder, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_cross_gamma, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_vega_by_expiry_bucket, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_vega_by_strike_expiry_bucket,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_fx_delta, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_commodity_delta, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_jacobian_via_bootstrap, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_map_risk_class, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_to_crif_csv, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_compute_risk_charges, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_pnl_explain, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_scenario_pnl_report, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_risk_contribution_per_trade,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_explained_pnl_components, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_apply_market_shock, module)?)?;
    module.add_function(wrap_pyfunction!(risk::py_diff_market_snapshots, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_historical_replay_from_diff,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_run_scenario_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_run_scenario_batch_with_pricer,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(risk::py_day_over_day_attribution, module)?)?;
    module.add_function(wrap_pyfunction!(
        risk::py_day_over_day_attribution_with_pricer,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(funding::funding_rate_swap_mtm, module)?)?;
    module.add_function(wrap_pyfunction!(funding::funding_rate_swap_dv01, module)?)?;
    module.add_function(wrap_pyfunction!(funding::funding_rate_swap_theta, module)?)?;
    module.add_function(wrap_pyfunction!(funding::funding_rate_swap_risks, module)?)?;
    module.add_function(wrap_pyfunction!(vol::py_svi_vol, module)?)?;
    module.add_class::<funding::FundingRateSnapshot>()?;
    module.add_class::<funding::FundingRateCurve>()?;
    module.add_class::<funding::FundingRateStats>()?;
    module.add_class::<funding::MultiVenueFundingCurve>()?;
    module.add_class::<funding::FundingRateSwap>()?;
    module.add_class::<funding::FundingRateSwapRisks>()?;
    module.add_class::<risk::Greeks>()?;
    module.add_class::<risk::YieldCurve>()?;
    module.add_class::<risk::SurvivalCurve>()?;
    module.add_class::<risk::KupiecBacktestResult>()?;
    module.add_class::<risk::ChristoffersenBacktestResult>()?;
    module.add_class::<risk::VarBacktestResult>()?;
    module.add_class::<risk::SampledVolSurface>()?;
    module.add_class::<risk::VolSource>()?;
    module.add_class::<risk::Market>()?;
    module.add_class::<risk::ForwardCurveSnapshot>()?;
    module.add_class::<risk::CreditCurveSnapshot>()?;
    module.add_class::<risk::MarketSnapshot>()?;
    module.add_class::<risk::MarginParams>()?;
    module.add_class::<risk::MarginCalculator>()?;
    module.add_class::<risk::InherentLeverage>()?;
    module.add_class::<risk::Vasicek>()?;
    module.add_class::<risk::FundingRateModel>()?;
    module.add_class::<risk::LiquidationPosition>()?;
    module.add_class::<risk::LiquidationSimulator>()?;
    module.add_class::<risk::LiquidationRisk>()?;
    module.add_class::<risk::StressTestResult>()?;
    module.add_class::<risk::StressScenario>()?;
    module.add_class::<risk::XvaCalculator>()?;
    module.add_class::<risk::CsaTerms>()?;
    module.add_class::<risk::SimmRiskClass>()?;
    module.add_class::<risk::SimmMargin>()?;
    module.add_class::<risk::SaCcrAssetClass>()?;
    module.add_class::<risk::AggregatedGreeks>()?;
    module.add_class::<risk::Position>()?;
    module.add_class::<risk::Portfolio>()?;
    module.add_class::<risk::WwrResult>()?;
    module.add_class::<risk::AlphaWWR>()?;
    module.add_class::<risk::CopulaWWR>()?;
    module.add_class::<risk::HullWhiteWWR>()?;
    module.add_class::<risk::BumpSize>()?;
    module.add_class::<risk::DifferencingScheme>()?;
    module.add_class::<risk::CurveBumpMode>()?;
    module.add_class::<risk::CurveBumpConfig>()?;
    module.add_class::<risk::SurfaceBumpMode>()?;
    module.add_class::<risk::SurfaceBumpConfig>()?;
    module.add_class::<risk::SpotBumpConfig>()?;
    module.add_class::<risk::BucketSensitivity>()?;
    module.add_class::<risk::KeyRateDurationPoint>()?;
    module.add_class::<risk::GammaLadderPoint>()?;
    module.add_class::<risk::VegaExpiryPoint>()?;
    module.add_class::<risk::VegaStrikeExpiryPoint>()?;
    module.add_class::<risk::QuoteVolSurface>()?;
    module.add_class::<risk::ChainRuleJacobian>()?;
    module.add_class::<risk::RegulatoryRiskClass>()?;
    module.add_class::<risk::SensitivityMeasure>()?;
    module.add_class::<risk::SensitivityRecord>()?;
    module.add_class::<risk::CrifRecord>()?;
    module.add_class::<risk::RiskClassChargeConfig>()?;
    module.add_class::<risk::RiskChargeConfig>()?;
    module.add_class::<risk::ClassRiskCharge>()?;
    module.add_class::<risk::RiskChargeSummary>()?;
    module.add_class::<risk::ScenarioShock>()?;
    module.add_class::<risk::PnlExplain>()?;
    module.add_class::<risk::ScenarioPnlRow>()?;
    module.add_class::<risk::TradeRiskContribution>()?;
    module.add_class::<risk::ScenarioKind>()?;
    module.add_class::<risk::ShockFactor>()?;
    module.add_class::<risk::MarketShock>()?;
    module.add_class::<risk::HistoricalReplayDefinition>()?;
    module.add_class::<risk::HypotheticalScenarioDefinition>()?;
    module.add_class::<risk::StressAxis>()?;
    module.add_class::<risk::ParametricStress2dDefinition>()?;
    module.add_class::<risk::ReverseStressDefinition>()?;
    module.add_class::<risk::ScenarioDefinition>()?;
    module.add_class::<risk::StressGridPoint>()?;
    module.add_class::<risk::ResolvedScenario>()?;
    module.add_class::<risk::ScenarioTrade>()?;
    module.add_class::<risk::ExplainedPnlComponents>()?;
    module.add_class::<risk::ScenarioTradePnlRow>()?;
    module.add_class::<risk::ScenarioPortfolioPnlRow>()?;
    module.add_class::<risk::ScenarioResultTable>()?;
    module.add_class::<risk::StressHeatmap2d>()?;
    module.add_class::<risk::ScenarioRunResult>()?;
    module.add_class::<risk::MarketLevelDiff>()?;
    module.add_class::<risk::SpotPriceDiff>()?;
    module.add_class::<risk::YieldCurveDiff>()?;
    module.add_class::<risk::CreditCurveDiff>()?;
    module.add_class::<risk::MarketSnapshotDiff>()?;
    module.add_class::<risk::DayOverDayAttribution>()?;
    Ok(())
}
