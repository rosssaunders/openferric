use pyo3::prelude::*;

fn copy(
    root: &Bound<'_, PyModule>,
    destination: &Bound<'_, PyModule>,
    names: &[&str],
) -> PyResult<()> {
    for name in names {
        destination.add(*name, root.getattr(*name)?)?;
    }
    Ok(())
}

fn child<'py>(
    root: &Bound<'py, PyModule>,
    parent: &Bound<'py, PyModule>,
    name: &str,
    names: &[&str],
) -> PyResult<Bound<'py, PyModule>> {
    let full_name = format!("{}.{}", parent.name()?, name);
    let module = PyModule::new(root.py(), &full_name)?;
    copy(root, &module, names)?;
    parent.add(name, &module)?;
    root.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(full_name, &module)?;
    Ok(module)
}

pub(crate) fn register(root: &Bound<'_, PyModule>) -> PyResult<()> {
    let credit = root.getattr("credit")?.cast_into::<PyModule>()?;
    copy(root, &credit, &["cash_settle_date", "step_in_date"])?;
    for name in [
        "price_isda_flat",
        "price_isda_flat_legacy_analytic",
        "price_isda_flat_with_calendar",
        "price_midpoint_flat",
        "price_midpoint_flat_with_calendar",
    ] {
        let function = root.getattr("DatedCds")?.getattr(name)?;
        credit.add(name, &function)?;
        root.add(name, function)?;
    }
    copy(
        root,
        &credit,
        &[
            "bootstrap_survival_curve_from_cds_spreads",
            "vasicek_portfolio_loss_cdf",
            "first_to_default_spread_copula",
            "fair_spread_from_hazard",
            "risky_annuity",
            "cash_settle_date_with_calendar",
            "generate_imm_schedule",
            "hazard_from_par_spread",
            "next_imm_twentieth",
            "previous_imm_twentieth",
            "step_in_date_with_calendar",
        ],
    )?;
    let market = root.getattr("market")?.cast_into::<PyModule>()?;
    market.add("fx_delta", root.getattr("market_fx_delta")?)?;
    copy(root, &market, &["atm_strike", "strike_from_delta"])?;
    copy(
        root,
        &market,
        &[
            "bootstrap_dividend_curve_from_put_call_parity",
            "canonical_pair",
            "convert_premium",
            "fx_option_premium",
        ],
    )?;
    let models = root.getattr("models")?.cast_into::<PyModule>()?;
    copy(
        root,
        &models,
        &[
            "Vasicek",
            "VarianceGamma",
            "Cgmy",
            "Nig",
            "CurveStructure",
            "ForwardInterpolation",
            "SeasonalityMode",
            "FbmScheme",
            "MAX_QUOTED_NORMAL_VOL",
        ],
    )?;
    let calibration = root.getattr("calibration")?.cast_into::<PyModule>()?;
    calibration.add("diagnostics", root.getattr("calibration_diagnostics")?)?;
    copy(
        root,
        &calibration,
        &["TerminationReason", "CalibrationWarningFlag"],
    )?;
    let pricing = root.getattr("pricing")?.cast_into::<PyModule>()?;
    copy(
        root,
        &pricing,
        &[
            "FundingRateSwapRisks",
            "funding_rate_swap_mtm",
            "funding_rate_swap_dv01",
            "funding_rate_swap_vega",
            "funding_rate_swap_theta",
            "funding_rate_swap_risks",
            "funding_rate_swap_discount_dv01",
        ],
    )?;
    copy(
        root,
        &pricing,
        &[
            "crr_binomial_american",
            "price_autocallable_with_greeks",
            "price_phoenix_autocallable_with_greeks",
            "barrier_price_closed_form_with_carry_and_rebate",
            "longstaff_schwartz_bermudan",
            "black_76_price",
            "FUNDING_RATE_BUMP_BP",
            "FUNDING_RATE_VOL_BUMP",
        ],
    )?;
    copy(
        root,
        &root.getattr("vol")?.cast_into::<PyModule>()?,
        &["vanna_volga_pivot_strikes"],
    )?;
    copy(
        root,
        &root.getattr("risk")?.cast_into::<PyModule>()?,
        &["historical_expected_shortfall"],
    )?;
    copy(
        root,
        &root.getattr("dsl")?.cast_into::<PyModule>()?,
        &["DslError", "ExecutionBackend", "ExecutionPolicy"],
    )?;
    let math = root.getattr("math")?.cast_into::<PyModule>()?;
    for name in ["sample_standard_normal", "fill_standard_normals"] {
        let function = root.getattr("FastRng")?.getattr(name)?;
        math.add(name, &function)?;
        root.add(name, function)?;
    }
    for (alias, name) in [
        ("Pcg64Rng", "Pcg64"),
        ("Xoshiro256Rng", "Xoshiro256PlusPlus"),
    ] {
        let value = root.getattr(name)?;
        root.add(alias, &value)?;
        math.add(alias, value)?;
    }
    copy(
        root,
        &math,
        &[
            "MathError",
            "InterpolationError",
            "normal_cdf_approx",
            "normal_cdf_batch_approx",
            "normal_cdf_batch_approx_into",
            "BatchSimdBackend",
            "detected_batch_simd_backend",
        ],
    )?;
    let mc = root.getattr("mc")?.cast_into::<PyModule>()?;
    copy(
        root,
        &mc,
        &[
            "MonteCarloPricingEngine",
            "ArithmeticAsianMC",
            "SpreadMonteCarloEngine",
            "MonteCarloGreeksEngine",
            "VarianceReduction",
            "ExecutionPolicy",
            "ExecutionBackend",
            "AccuracyTier",
            "PricingArena",
            "mc_european_with_arena",
        ],
    )?;
    let fft = root.getattr("fft")?.cast_into::<PyModule>()?;
    copy(root, &fft, &["heston_price_fft", "try_heston_price_fft"])?;
    let engines = root.getattr("engines")?.cast_into::<PyModule>()?;
    child(
        root,
        &engines,
        "tree",
        &[
            "BinomialTreeEngine",
            "TrinomialTreeEngine",
            "GeneralizedBinomialEngine",
            "SwingTreeEngine",
            "ConvertibleBinomialEngine",
            "TwoAssetBinomialEngine",
            "BermudanSwaptionEngine",
        ],
    )?;
    child(
        root,
        &engines,
        "pde",
        &[
            "CrankNicolsonEngine",
            "ExplicitFdEngine",
            "ImplicitFdEngine",
            "HopscotchEngine",
            "AdiHestonEngine",
            "AdiScheme",
            "BermudanPdeOutput",
            "PdeExerciseBoundaryPoint",
        ],
    )?;
    child(
        root,
        &engines,
        "lsm",
        &[
            "LongstaffSchwartzEngine",
            "LsmDynamics",
            "ExerciseBoundaryPoint",
            "BermudanLsmOutput",
        ],
    )?;
    child(
        root,
        &engines,
        "numerical",
        &["AmericanBinomialEngine", "PricingArena"],
    )?;
    let analytic = child(
        root,
        &engines,
        "analytic",
        &[
            "BlackScholesEngine",
            "Black76Engine",
            "GeometricAsianEngine",
            "BarrierAnalyticEngine",
            "DigitalAnalyticEngine",
            "DoubleBarrierAnalyticEngine",
            "ExoticAnalyticEngine",
            "GarmanKohlhagenEngine",
            "PowerOptionEngine",
            "RainbowAnalyticEngine",
            "SpreadAnalyticEngine",
            "SpreadAnalyticMethod",
            "VarianceSwapEngine",
            "FxGreeks",
            "BinaryBarrierType",
            "BatchSimdBackend",
            "detected_batch_simd_backend",
            "black_scholes",
            "bs_delta",
            "bs_gamma",
            "bs_vega",
            "bs_theta",
            "bs_rho",
            "norm_cdf",
            "norm_pdf",
            "bs_price_asm",
            "has_fma_bs_kernel",
            "bs_price_batch",
            "bs_price_batch_into",
            "bs_greeks_batch",
            "bs_greeks_batch_into",
            "normal_cdf_approx",
            "normal_cdf_batch_approx",
            "normal_cdf_batch_approx_into",
        ],
    )?;
    let helper = root.getattr("AnalyticEngine")?;
    for name in helper.dir()?.try_iter()? {
        let name: String = name?.extract()?;
        if !name.starts_with('_') && !analytic.hasattr(&name)? {
            analytic.add(&name, helper.getattr(&name)?)?;
        }
    }
    for (name, module) in [("fft", &fft), ("monte_carlo", &mc)] {
        engines.add(name, module)?;
        root.py()
            .import("sys")?
            .getattr("modules")?
            .set_item(format!("openferric.engines.{name}"), module)?;
    }
    #[cfg(feature = "gpu")]
    child(
        root,
        &engines,
        "gpu",
        &[
            "GpuMcResult",
            "gpu_is_ready",
            "prewarm_gpu",
            "mc_european_gpu",
        ],
    )?;
    root.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
