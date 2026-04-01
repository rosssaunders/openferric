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
    Ok(())
}
