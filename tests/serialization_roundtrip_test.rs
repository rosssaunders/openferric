use chrono::NaiveDate;
use openferric::core::*;
use openferric::credit::cds_option::CdsOption;
use openferric::credit::{
    CdoTranche, Cds, CdsDateRule, CdsIndex, DatedCds, NthToDefaultBasket, ProtectionSide,
    SyntheticCdo,
};
use openferric::instruments::mbs::{ConstantCpr, MbsPassThrough, PrepaymentModel};
use openferric::instruments::*;
use openferric::market::Market;
use openferric::rates::cms::{CmsSpreadOption, CmsSpreadOptionType};
use openferric::rates::*;
use openferric::vol::surface::{SviParams, VolSurface as SviSurface};

fn roundtrip_json<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value).expect("json serialize");
    let decoded: T = serde_json::from_str(&json).expect("json deserialize");
    assert_eq!(*value, decoded);
}

fn roundtrip_msgpack<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = rmp_serde::to_vec_named(value).expect("msgpack serialize");
    let decoded: T = rmp_serde::from_slice(&bytes).expect("msgpack deserialize");
    assert_eq!(*value, decoded);
}

fn sample_trade_products() -> Vec<TradeProduct> {
    let start = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
    let end = NaiveDate::from_ymd_opt(2030, 1, 2).unwrap();

    let real_model = RealOptionBinomialSpec {
        project_value: 120.0,
        volatility: 0.25,
        risk_free_rate: 0.03,
        maturity: 3.0,
        steps: 50,
        cash_flows: vec![DiscreteCashFlow {
            time: 1.0,
            amount: 2.0,
        }],
    };

    vec![
        TradeProduct::VanillaOption(VanillaOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            exercise: ExerciseStyle::European,
        }),
        TradeProduct::BarrierOption(BarrierOption {
            option_type: OptionType::Put,
            strike: 95.0,
            expiry: 1.2,
            barrier: BarrierSpec {
                direction: BarrierDirection::Down,
                style: BarrierStyle::Out,
                level: 80.0,
                rebate: 1.0,
            },
        }),
        TradeProduct::AsianOption(AsianOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            asian: AsianSpec {
                averaging: Averaging::Arithmetic,
                strike_type: StrikeType::Fixed,
                observation_times: vec![0.25, 0.5, 0.75, 1.0],
            },
        }),
        TradeProduct::BasketOption(BasketOption {
            weights: vec![0.6, 0.4],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        }),
        TradeProduct::CashOrNothingOption(CashOrNothingOption {
            option_type: OptionType::Call,
            strike: 100.0,
            cash: 10.0,
            expiry: 1.0,
        }),
        TradeProduct::AssetOrNothingOption(AssetOrNothingOption {
            option_type: OptionType::Put,
            strike: 90.0,
            expiry: 1.0,
        }),
        TradeProduct::GapOption(GapOption {
            option_type: OptionType::Call,
            payoff_strike: 100.0,
            trigger_strike: 105.0,
            expiry: 1.0,
        }),
        TradeProduct::DoubleBarrierOption(DoubleBarrierOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            lower_barrier: 80.0,
            upper_barrier: 130.0,
            barrier_type: DoubleBarrierType::KnockOut,
            rebate: 0.5,
        }),
        TradeProduct::SpreadOption(SpreadOption {
            s1: 100.0,
            s2: 95.0,
            k: 2.0,
            vol1: 0.2,
            vol2: 0.25,
            rho: 0.4,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.0,
        }),
        TradeProduct::FxOption(FxOption {
            option_type: OptionType::Call,
            domestic_rate: 0.03,
            foreign_rate: 0.01,
            spot_fx: 1.10,
            strike_fx: 1.15,
            vol: 0.14,
            maturity: 1.0,
        }),
        TradeProduct::FuturesOption(FuturesOption {
            forward: 100.0,
            strike: 102.0,
            vol: 0.2,
            r: 0.03,
            t: 1.0,
            option_type: OptionType::Put,
        }),
        TradeProduct::ForwardStartOption(ForwardStartOption {
            option_type: OptionType::Call,
            spot: 100.0,
            strike_ratio: 1.0,
            rate: 0.02,
            dividend_yield: 0.01,
            vol: 0.25,
            t_start: 0.5,
            expiry: 1.0,
        }),
        TradeProduct::CommodityForward(CommodityForward {
            spot: 80.0,
            strike: 82.0,
            notional: 10_000.0,
            risk_free_rate: 0.03,
            storage_cost: 0.01,
            convenience_yield: 0.005,
            maturity: 1.5,
            is_long: true,
        }),
        TradeProduct::CommodityFutures(CommodityFutures {
            contract_price: 75.0,
            contract_size: 1_000.0,
            is_long: false,
        }),
        TradeProduct::CommodityOption(CommodityOption {
            forward: 90.0,
            strike: 95.0,
            vol: 0.3,
            risk_free_rate: 0.03,
            maturity: 1.0,
            notional: 5_000.0,
            option_type: OptionType::Call,
        }),
        TradeProduct::CommoditySpreadOption(CommoditySpreadOption {
            option_type: OptionType::Call,
            forward_1: 100.0,
            forward_2: 95.0,
            strike: 2.0,
            quantity_1: 1.0,
            quantity_2: 1.2,
            vol_1: 0.25,
            vol_2: 0.22,
            rho: 0.5,
            risk_free_rate: 0.03,
            maturity: 1.0,
            notional: 2_000.0,
        }),
        TradeProduct::ConvertibleBond(ConvertibleBond {
            face_value: 100.0,
            coupon_rate: 0.03,
            maturity: 5.0,
            conversion_ratio: 1.2,
            call_price: Some(110.0),
            put_price: Some(95.0),
        }),
        TradeProduct::EmployeeStockOption(EmployeeStockOption {
            option_type: OptionType::Call,
            strike: 100.0,
            maturity: 5.0,
            vesting_period: 1.0,
            expected_life: 4.0,
            early_exercise_multiple: Some(1.8),
            forfeiture_rate: 0.03,
            shares_outstanding: 1_000_000.0,
            options_granted: 50_000.0,
        }),
        TradeProduct::ExoticOption(ExoticOption::Chooser(ChooserOption {
            strike: 100.0,
            expiry: 1.0,
            choose_time: 0.5,
        })),
        TradeProduct::PowerOption(PowerOption {
            option_type: OptionType::Put,
            strike: 100.0,
            alpha: 1.3,
            expiry: 1.0,
        }),
        TradeProduct::BestOfTwoCallOption(BestOfTwoCallOption {
            s1: 100.0,
            s2: 98.0,
            k: 100.0,
            vol1: 0.2,
            vol2: 0.22,
            rho: 0.4,
            q1: 0.01,
            q2: 0.01,
            r: 0.03,
            t: 1.0,
        }),
        TradeProduct::WorstOfTwoCallOption(WorstOfTwoCallOption {
            s1: 100.0,
            s2: 98.0,
            k: 100.0,
            vol1: 0.2,
            vol2: 0.22,
            rho: 0.4,
            q1: 0.01,
            q2: 0.01,
            r: 0.03,
            t: 1.0,
        }),
        TradeProduct::TwoAssetCorrelationOption(TwoAssetCorrelationOption {
            option_type: OptionType::Call,
            s1: 100.0,
            s2: 99.0,
            k1: 100.0,
            k2: 97.0,
            vol1: 0.2,
            vol2: 0.23,
            rho: 0.3,
            q1: 0.01,
            q2: 0.01,
            r: 0.03,
            t: 1.0,
        }),
        TradeProduct::RangeAccrual(RangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.04,
            lower_bound: 0.01,
            upper_bound: 0.05,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }),
        TradeProduct::DualRangeAccrual(DualRangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.03,
            lower_bound: -0.01,
            upper_bound: 0.02,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }),
        TradeProduct::DeferInvestmentOption(DeferInvestmentOption {
            model: real_model.clone(),
            investment_cost: 80.0,
        }),
        TradeProduct::ExpandOption(ExpandOption {
            model: real_model.clone(),
            expansion_multiplier: 1.5,
            expansion_cost: 30.0,
        }),
        TradeProduct::AbandonmentOption(AbandonmentOption {
            model: real_model.clone(),
            salvage_value: 25.0,
        }),
        TradeProduct::RealOptionInstrument(RealOptionInstrument::Expand(ExpandOption {
            model: real_model.clone(),
            expansion_multiplier: 1.3,
            expansion_cost: 20.0,
        })),
        TradeProduct::SwingOption(SwingOption {
            min_exercises: 1,
            max_exercises: 3,
            exercise_dates: vec![0.5, 1.0, 1.5],
            strike: 70.0,
            payoff_per_exercise: 10.0,
        }),
        TradeProduct::Tarf(Tarf {
            strike: 100.0,
            notional_per_fixing: 1000.0,
            ko_barrier: 130.0,
            target_profit: 10_000.0,
            downside_leverage: 2.0,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            tarf_type: TarfType::Standard,
        }),
        TradeProduct::VarianceSwap(VarianceSwap {
            notional_vega: 50_000.0,
            strike_vol: 0.2,
            expiry: 1.0,
            observed_realized_var: Some(0.03),
            option_quotes: vec![
                VarianceOptionQuote::new(80.0, 25.0, 1.0),
                VarianceOptionQuote::new(120.0, 1.0, 22.0),
            ],
        }),
        TradeProduct::VolatilitySwap(VolatilitySwap {
            notional_vega: 50_000.0,
            strike_vol: 0.22,
            expiry: 1.0,
            observed_realized_var: Some(0.04),
            option_quotes: vec![
                VarianceOptionQuote::new(80.0, 25.0, 1.0),
                VarianceOptionQuote::new(120.0, 1.0, 22.0),
            ],
            var_of_var: 0.1,
        }),
        TradeProduct::WeatherSwap(WeatherSwap {
            index_type: DegreeDayType::HDD,
            strike: 100.0,
            tick_size: 10.0,
            notional: 1_000.0,
            is_payer: true,
            discount_rate: 0.02,
            maturity: 1.0,
        }),
        TradeProduct::WeatherOption(WeatherOption {
            index_type: DegreeDayType::CDD,
            option_type: OptionType::Call,
            strike: 150.0,
            tick_size: 8.0,
            notional: 1_000.0,
            discount_rate: 0.02,
            maturity: 1.0,
        }),
        TradeProduct::CatastropheBond(CatastropheBond {
            principal: 1_000_000.0,
            coupon_rate: 0.06,
            risk_free_rate: 0.03,
            maturity: 3.0,
            coupon_frequency: 4,
            loss_intensity: 0.2,
            expected_loss_per_event: 0.4,
        }),
        TradeProduct::Autocallable(Autocallable {
            underlyings: vec![0, 1],
            notional: 1000.0,
            autocall_dates: vec![0.5, 1.0],
            autocall_barrier: 1.05,
            coupon_rate: 0.08,
            ki_barrier: 0.7,
            ki_strike: 1.0,
            maturity: 1.0,
        }),
        TradeProduct::PhoenixAutocallable(PhoenixAutocallable {
            underlyings: vec![0, 1],
            notional: 1000.0,
            autocall_dates: vec![0.5, 1.0],
            autocall_barrier: 1.05,
            coupon_barrier: 0.8,
            coupon_rate: 0.08,
            memory: true,
            ki_barrier: 0.7,
            ki_strike: 1.0,
            maturity: 1.0,
        }),
        TradeProduct::MbsPassThrough(MbsPassThrough {
            original_balance: 100_000.0,
            coupon_rate: 0.05,
            servicing_fee: 0.0025,
            original_term: 360,
            age: 24,
            prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.06 }),
        }),
        TradeProduct::FixedRateBond(FixedRateBond {
            face_value: 100.0,
            coupon_rate: 0.04,
            frequency: 2,
            maturity: 5.0,
            day_count: DayCountConvention::Act365Fixed,
        }),
        TradeProduct::CapFloor(CapFloor {
            notional: 1_000_000.0,
            strike: 0.03,
            start_date: start,
            end_date: end,
            frequency: Frequency::Quarterly,
            day_count: DayCountConvention::Act360,
            is_cap: true,
        }),
        TradeProduct::ForwardRateAgreement(ForwardRateAgreement {
            notional: 1_000_000.0,
            fixed_rate: 0.03,
            start_date: start,
            end_date: NaiveDate::from_ymd_opt(2026, 7, 2).unwrap(),
            day_count: DayCountConvention::Act360,
        }),
        TradeProduct::Future(Future {
            underlying_spot: 100.0,
            risk_free_rate: 0.03,
            dividend_yield: 0.01,
            storage_cost: 0.0,
            convenience_yield: 0.0,
            expiry: 0.5,
        }),
        TradeProduct::InterestRateSwap(InterestRateSwap {
            notional: 1_000_000.0,
            fixed_rate: 0.03,
            float_spread: 0.001,
            start_date: start,
            end_date: end,
            fixed_freq: Frequency::Annual,
            float_freq: Frequency::Quarterly,
            fixed_day_count: DayCountConvention::Act365Fixed,
            float_day_count: DayCountConvention::Act360,
        }),
        TradeProduct::Swaption(Swaption {
            notional: 1_000_000.0,
            strike: 0.03,
            option_expiry: 1.0,
            swap_tenor: 5.0,
            is_payer: true,
        }),
        TradeProduct::OvernightIndexSwap(OvernightIndexSwap {
            notional: 1_000_000.0,
            fixed_rate: 0.028,
            float_spread: 0.0,
            tenor: 3.0,
        }),
        TradeProduct::BasisSwap(BasisSwap {
            notional: 1_000_000.0,
            spread_on_short_leg: 0.001,
            tenor: 3.0,
            short_leg_payments_per_year: 4,
            long_leg_payments_per_year: 2,
        }),
        TradeProduct::XccySwap(XccySwap {
            notional1: 1_000_000.0,
            notional2: 900_000.0,
            fixed_rate: 0.03,
            float_spread: 0.001,
            tenor: 5.0,
            fx_spot: 1.1,
        }),
        TradeProduct::ZeroCouponInflationSwap(ZeroCouponInflationSwap {
            notional: 1_000_000.0,
            cpi_base: 250.0,
            fixed_rate: 0.025,
            tenor: 5.0,
            receive_inflation: true,
        }),
        TradeProduct::YearOnYearInflationSwap(YearOnYearInflationSwap {
            notional: 1_000_000.0,
            fixed_rate: 0.02,
            maturity_years: 5,
            receive_inflation: true,
        }),
        TradeProduct::InflationIndexedBond(InflationIndexedBond {
            face_value: 100.0,
            coupon_rate: 0.01,
            maturity_years: 10,
            coupon_frequency: 2,
            cpi_base: 250.0,
        }),
        TradeProduct::CmsSpreadOption(CmsSpreadOption {
            strike: 0.01,
            option_type: CmsSpreadOptionType::Call,
            notional: 1_000_000.0,
            expiry: 1.0,
        }),
        TradeProduct::Cds(Cds {
            notional: 1_000_000.0,
            spread: 0.01,
            maturity: 5.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        }),
        TradeProduct::CdsOption(CdsOption {
            notional: 1_000_000.0,
            strike_spread: 0.01,
            option_expiry: 1.0,
            cds_maturity: 5.0,
            is_payer: true,
            recovery_rate: 0.4,
        }),
        TradeProduct::CdoTranche(CdoTranche {
            attachment: 0.03,
            detachment: 0.07,
            notional: 10_000_000.0,
            spread: 0.04,
        }),
        TradeProduct::SyntheticCdo(SyntheticCdo {
            num_names: 125,
            pool_spread: 0.02,
            recovery_rate: 0.4,
            correlation: 0.3,
            risk_free_rate: 0.03,
            maturity: 5.0,
            payment_freq: 4,
        }),
        TradeProduct::CdsIndex(CdsIndex {
            constituents: vec![Cds {
                notional: 1_000_000.0,
                spread: 0.01,
                maturity: 5.0,
                recovery_rate: 0.4,
                payment_freq: 4,
            }],
            weights: vec![1.0],
        }),
        TradeProduct::NthToDefaultBasket(NthToDefaultBasket {
            n: 1,
            notional: 1_000_000.0,
            maturity: 5.0,
            recovery_rate: 0.4,
            payment_freq: 4,
        }),
        TradeProduct::DatedCds(DatedCds {
            side: ProtectionSide::Buyer,
            notional: 1_000_000.0,
            running_spread: 0.01,
            recovery_rate: 0.4,
            issue_date: start,
            maturity_date: end,
            coupon_interval_months: 3,
            date_rule: CdsDateRule::QuarterlyImm,
        }),
    ]
}

#[test]
fn trade_roundtrip_json_and_msgpack_for_every_product_variant() {
    for (idx, product) in sample_trade_products().into_iter().enumerate() {
        let trade = Trade {
            metadata: TradeMetadata {
                trade_id: format!("TRD-{idx:04}"),
                version: 1,
                timestamp: "2026-02-22T12:00:00Z".to_string(),
            },
            market_data_ref: Some("SNAP-2026-02-22".to_string()),
            product,
        };

        roundtrip_json(&trade);
        roundtrip_msgpack(&trade);
    }
}

#[test]
fn market_snapshot_roundtrip_json_and_msgpack() {
    let snapshot = MarketSnapshot {
        snapshot_id: "SNAP-2026-02-22".to_string(),
        as_of: "2026-02-22T12:00:00Z".to_string(),
        yield_curves: [(
            "USD-OIS".to_string(),
            YieldCurveSnapshot {
                pillars: vec![0.5, 1.0, 2.0, 5.0],
                rates: vec![0.03, 0.031, 0.032, 0.034],
                interpolation: InterpolationMethod::LogLinear,
                value_type: CurveValueType::ZeroRate,
            },
        )]
        .into_iter()
        .collect(),
        credit_curves: [(
            "ACME".to_string(),
            CreditCurveSnapshot {
                pillars: vec![1.0, 3.0, 5.0],
                survival_probabilities: vec![0.98, 0.93, 0.88],
                recovery_rate: 0.4,
                interpolation: InterpolationMethod::LogLinear,
            },
        )]
        .into_iter()
        .collect(),
        vol_surfaces: [(
            "SPX".to_string(),
            VolSurfaceSnapshot {
                expiries: vec![0.5, 1.0],
                strikes: vec![90.0, 100.0, 110.0],
                vols: vec![vec![0.22, 0.2, 0.21], vec![0.24, 0.22, 0.23]],
                model: Some(VolSurfaceModel::Svi {
                    parameters_by_expiry: vec![
                        (
                            0.5,
                            SviParams {
                                a: 0.02,
                                b: 0.1,
                                rho: -0.2,
                                m: 0.0,
                                sigma: 0.3,
                            },
                        ),
                        (
                            1.0,
                            SviParams {
                                a: 0.03,
                                b: 0.11,
                                rho: -0.25,
                                m: 0.0,
                                sigma: 0.35,
                            },
                        ),
                    ],
                }),
            },
        )]
        .into_iter()
        .collect(),
        spots: [("SPX".to_string(), 5100.0)].into_iter().collect(),
        forwards: [(
            "WTI".to_string(),
            ForwardCurveSnapshot {
                points: vec![(0.5, 76.0), (1.0, 77.2), (2.0, 79.5)],
                interpolation: InterpolationMethod::Linear,
            },
        )]
        .into_iter()
        .collect(),
    };

    roundtrip_json(&snapshot);
    roundtrip_msgpack(&snapshot);
}

#[test]
fn pricing_audit_roundtrip_json_and_msgpack() {
    let mut diagnostics = Diagnostics::new();
    diagnostics.insert("npv", 12.34);
    diagnostics.insert("vol", 0.22);

    let base_result = PricingResult {
        price: 12.34,
        stderr: Some(0.01),
        greeks: Some(Greeks {
            delta: 0.5,
            gamma: 0.02,
            vega: 0.11,
            theta: -0.03,
            rho: 0.09,
        }),
        diagnostics,
    };

    let trade = Trade {
        metadata: TradeMetadata {
            trade_id: "TRD-AUDIT-001".to_string(),
            version: 3,
            timestamp: "2026-02-22T12:00:00Z".to_string(),
        },
        market_data_ref: Some("SNAP-2026-02-22".to_string()),
        product: TradeProduct::VanillaOption(VanillaOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            exercise: ExerciseStyle::European,
        }),
    };

    let audit = PricingAuditTrail {
        trade,
        market_snapshot_id: "SNAP-2026-02-22".to_string(),
        engine_name: "analytic.black_scholes".to_string(),
        model_name: "black_scholes".to_string(),
        model_parameters: [("rate".to_string(), 0.03), ("vol".to_string(), 0.22)]
            .into_iter()
            .collect(),
        base_result: base_result.clone(),
        scenario_results: vec![ScenarioPricingResult {
            scenario_name: "spot_up_1pct".to_string(),
            bump_description: Some("spot +1%".to_string()),
            result: PricingResult {
                price: 12.95,
                ..base_result
            },
        }],
        generated_at: "2026-02-22T12:00:01Z".to_string(),
        notes: vec!["desk=eqd".to_string(), "book=macro".to_string()],
    };

    roundtrip_json(&audit);
    roundtrip_msgpack(&audit);
}

#[test]
fn market_runtime_type_roundtrip_for_flat_and_surface_vol() {
    let flat_market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .flat_vol(0.2)
        .reference_date("2026-02-22")
        .build()
        .unwrap();

    let flat_json = serde_json::to_string(&flat_market).unwrap();
    let flat_back: Market = serde_json::from_str(&flat_json).unwrap();
    assert!((flat_back.spot - flat_market.spot).abs() < 1e-12);
    assert!((flat_back.vol_for(100.0, 1.0) - flat_market.vol_for(100.0, 1.0)).abs() < 1e-12);

    let surface = SviSurface::new(
        vec![(
            1.0,
            SviParams {
                a: 0.02,
                b: 0.1,
                rho: -0.2,
                m: 0.0,
                sigma: 0.3,
            },
        )],
        100.0,
    )
    .unwrap();

    let surface_market = Market::builder()
        .spot(100.0)
        .rate(0.03)
        .dividend_yield(0.01)
        .vol_surface(Box::new(surface))
        .reference_date("2026-02-22")
        .build()
        .unwrap();

    let surface_json = serde_json::to_string(&surface_market).unwrap();
    let surface_back: Market = serde_json::from_str(&surface_json).unwrap();
    let v1 = surface_market.vol_for(100.0, 1.0);
    let v2 = surface_back.vol_for(100.0, 1.0);
    assert!((v1 - v2).abs() < 1e-12);
}
