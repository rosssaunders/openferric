use std::fmt::Debug;

use openferric::core::{
    AsianSpec, Averaging, BarrierDirection, BarrierSpec, BarrierStyle, Diagnostics, ExerciseStyle,
    Greeks, OptionType, PricingError, PricingResult, StrikeType,
};
use openferric::credit::SurvivalCurve;
use openferric::instruments::{
    AbandonmentOption, AsianOption, AssetOrNothingOption, Autocallable, BarrierOption,
    BasketOption, BasketType, BestOfTwoCallOption, CashOrNothingOption, CatastropheBond,
    ChooserOption, CliquetOption, CommodityForward, CommodityFutures, CommodityOption,
    CommoditySpreadOption, CompoundOption, ConstantCpr, ConvertibleBond, DeferInvestmentOption,
    DegreeDayType, DiscreteCashFlow, DoubleBarrierOption, DoubleBarrierType, DualRangeAccrual,
    EmployeeStockOption, ExoticOption, ExpandOption, ForwardStartOption, FuturesOption, FxOption,
    GapOption, LookbackFixedOption, LookbackFloatingOption, MbsCashflow, MbsPassThrough,
    PhoenixAutocallable, Portfolio, PowerOption, PrepaymentModel, PsaModel, QuantoOption,
    RangeAccrual, RealOptionBinomialSpec, RealOptionInstrument, SpreadOption, SwingOption, Tarf,
    TarfType, Trade, TradeInstrument, TradeMetadata, TwoAssetCorrelationOption, VanillaOption,
    VarianceOptionQuote, VarianceSwap, VolatilitySwap, WeatherOption, WeatherSwap,
    WorstOfTwoCallOption,
};
use openferric::market::{
    CreditCurveSnapshot, ForwardCurveSnapshot, Market, MarketSnapshot, SampledVolSurface, VolSource,
};
use openferric::math::interpolation::ExtrapolationMode;
use openferric::rates::{
    YieldCurve, YieldCurveInterpolationMethod, YieldCurveInterpolationSettings,
};
use openferric::vol::surface::{SviParams, VolSurface as ParametricVolSurface};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn assert_roundtrip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let json = serde_json::to_vec_pretty(value).expect("json serialize");
    let from_json: T = serde_json::from_slice(&json).expect("json deserialize");
    assert_eq!(from_json, *value, "json roundtrip mismatch");

    let msgpack = rmp_serde::to_vec_named(value).expect("msgpack serialize");
    let from_msgpack: T = rmp_serde::from_slice(&msgpack).expect("msgpack deserialize");
    assert_eq!(from_msgpack, *value, "msgpack roundtrip mismatch");
}

#[derive(Debug, Clone)]
struct QuadraticSmile {
    base: f64,
    spot: f64,
    skew: f64,
}

impl openferric::market::VolSurface for QuadraticSmile {
    fn vol(&self, strike: f64, expiry: f64) -> f64 {
        let x = (strike / self.spot).ln();
        let t_adj = (expiry - 1.0).abs() * 0.01;
        (self.base + self.skew * x * x + t_adj).max(0.01)
    }
}

fn sample_yield_curve() -> YieldCurve {
    YieldCurve::new_with_settings(
        vec![(0.25, 0.996), (0.5, 0.990), (1.0, 0.978), (2.0, 0.952)],
        YieldCurveInterpolationSettings {
            method: YieldCurveInterpolationMethod::MonotoneConvex,
            extrapolation: ExtrapolationMode::Linear,
        },
    )
    .expect("valid curve")
}

fn sample_vol_surface() -> ParametricVolSurface {
    ParametricVolSurface::new(
        vec![
            (
                0.5,
                SviParams {
                    a: 0.02,
                    b: 0.10,
                    rho: -0.20,
                    m: 0.0,
                    sigma: 0.25,
                },
            ),
            (
                1.0,
                SviParams {
                    a: 0.03,
                    b: 0.12,
                    rho: -0.25,
                    m: 0.0,
                    sigma: 0.30,
                },
            ),
        ],
        100.0,
    )
    .expect("valid vol surface")
}

fn sample_trade_instruments() -> Vec<TradeInstrument> {
    let asian_spec = AsianSpec {
        averaging: Averaging::Arithmetic,
        strike_type: StrikeType::Fixed,
        observation_times: vec![0.25, 0.5, 0.75, 1.0],
    };

    let forward_start = ForwardStartOption {
        option_type: OptionType::Call,
        spot: 100.0,
        strike_ratio: 1.0,
        rate: 0.02,
        dividend_yield: 0.01,
        vol: 0.2,
        t_start: 0.5,
        expiry: 1.0,
    };

    let common_real_model = RealOptionBinomialSpec {
        project_value: 120.0,
        volatility: 0.25,
        risk_free_rate: 0.03,
        maturity: 2.0,
        steps: 50,
        cash_flows: vec![DiscreteCashFlow {
            time: 1.0,
            amount: 5.0,
        }],
    };

    vec![
        TradeInstrument::AsianOption(AsianOption::new(OptionType::Call, 100.0, 1.0, asian_spec)),
        TradeInstrument::Autocallable(Autocallable {
            underlyings: vec![0, 1],
            notional: 1_000_000.0,
            autocall_dates: vec![0.5, 1.0],
            autocall_barrier: 1.0,
            coupon_rate: 0.08,
            ki_barrier: 0.7,
            ki_strike: 1.0,
            maturity: 1.0,
        }),
        TradeInstrument::PhoenixAutocallable(PhoenixAutocallable {
            underlyings: vec![0, 1],
            notional: 1_000_000.0,
            autocall_dates: vec![0.5, 1.0],
            autocall_barrier: 1.0,
            coupon_barrier: 0.8,
            coupon_rate: 0.1,
            memory: true,
            ki_barrier: 0.6,
            ki_strike: 1.0,
            maturity: 1.0,
        }),
        TradeInstrument::BarrierOption(
            BarrierOption::builder()
                .call()
                .strike(100.0)
                .expiry(1.0)
                .down_and_out(80.0)
                .rebate(1.0)
                .build()
                .expect("valid barrier"),
        ),
        TradeInstrument::BasketOption(BasketOption {
            weights: vec![0.6, 0.4],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        }),
        TradeInstrument::FuturesOption(FuturesOption::new(
            100.0,
            102.0,
            0.25,
            0.03,
            1.0,
            OptionType::Call,
        )),
        TradeInstrument::CliquetOption(CliquetOption {
            option_type: OptionType::Put,
            ..forward_start
        }),
        TradeInstrument::ForwardStartOption(forward_start),
        TradeInstrument::CommodityForward(CommodityForward {
            spot: 70.0,
            strike: 72.0,
            notional: 1000.0,
            risk_free_rate: 0.02,
            storage_cost: 0.01,
            convenience_yield: 0.005,
            maturity: 1.0,
            is_long: true,
        }),
        TradeInstrument::CommodityFutures(CommodityFutures {
            contract_price: 75.0,
            contract_size: 1000.0,
            is_long: false,
        }),
        TradeInstrument::CommodityOption(CommodityOption {
            forward: 75.0,
            strike: 74.0,
            vol: 0.30,
            risk_free_rate: 0.02,
            maturity: 1.0,
            notional: 500.0,
            option_type: OptionType::Put,
        }),
        TradeInstrument::CommoditySpreadOption(CommoditySpreadOption {
            option_type: OptionType::Call,
            forward_1: 80.0,
            forward_2: 70.0,
            strike: 5.0,
            quantity_1: 1.0,
            quantity_2: 1.0,
            vol_1: 0.25,
            vol_2: 0.20,
            rho: 0.4,
            risk_free_rate: 0.02,
            maturity: 1.0,
            notional: 100.0,
        }),
        TradeInstrument::ConvertibleBond(ConvertibleBond::new(
            1000.0,
            0.04,
            5.0,
            10.0,
            Some(1050.0),
            Some(950.0),
        )),
        TradeInstrument::CashOrNothingOption(CashOrNothingOption::new(
            OptionType::Call,
            100.0,
            10.0,
            1.0,
        )),
        TradeInstrument::AssetOrNothingOption(AssetOrNothingOption::new(
            OptionType::Put,
            95.0,
            1.0,
        )),
        TradeInstrument::GapOption(GapOption::new(OptionType::Call, 100.0, 105.0, 1.0)),
        TradeInstrument::DoubleBarrierOption(DoubleBarrierOption::new(
            OptionType::Put,
            100.0,
            1.0,
            80.0,
            120.0,
            DoubleBarrierType::KnockOut,
            0.5,
        )),
        TradeInstrument::EmployeeStockOption(EmployeeStockOption::new(
            OptionType::Call,
            100.0,
            5.0,
            1.0,
            3.0,
            Some(2.0),
            0.02,
            1_000_000.0,
            50_000.0,
        )),
        TradeInstrument::LookbackFloatingOption(LookbackFloatingOption {
            option_type: OptionType::Call,
            expiry: 1.0,
            observed_extreme: Some(95.0),
        }),
        TradeInstrument::LookbackFixedOption(LookbackFixedOption {
            option_type: OptionType::Put,
            strike: 100.0,
            expiry: 1.0,
            observed_extreme: Some(80.0),
        }),
        TradeInstrument::ChooserOption(ChooserOption {
            strike: 100.0,
            expiry: 1.0,
            choose_time: 0.5,
        }),
        TradeInstrument::QuantoOption(QuantoOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            fx_rate: 1.1,
            foreign_rate: 0.02,
            fx_vol: 0.15,
            asset_fx_corr: -0.3,
        }),
        TradeInstrument::CompoundOption(CompoundOption {
            option_type: OptionType::Call,
            underlying_option_type: OptionType::Put,
            compound_strike: 4.0,
            underlying_strike: 100.0,
            compound_expiry: 0.5,
            underlying_expiry: 1.0,
        }),
        TradeInstrument::ExoticOption(ExoticOption::Chooser(ChooserOption {
            strike: 95.0,
            expiry: 1.0,
            choose_time: 0.25,
        })),
        TradeInstrument::FxOption(FxOption::new(
            OptionType::Call,
            0.03,
            0.01,
            1.1,
            1.05,
            0.2,
            1.0,
        )),
        TradeInstrument::PowerOption(PowerOption::new(OptionType::Call, 100.0, 1.2, 1.0)),
        TradeInstrument::BestOfTwoCallOption(BestOfTwoCallOption {
            s1: 100.0,
            s2: 110.0,
            k: 100.0,
            vol1: 0.25,
            vol2: 0.3,
            rho: 0.4,
            q1: 0.01,
            q2: 0.01,
            r: 0.03,
            t: 1.0,
        }),
        TradeInstrument::WorstOfTwoCallOption(WorstOfTwoCallOption {
            s1: 100.0,
            s2: 90.0,
            k: 85.0,
            vol1: 0.2,
            vol2: 0.28,
            rho: 0.35,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.0,
        }),
        TradeInstrument::TwoAssetCorrelationOption(TwoAssetCorrelationOption {
            option_type: OptionType::Put,
            s1: 100.0,
            s2: 95.0,
            k1: 100.0,
            k2: 90.0,
            vol1: 0.22,
            vol2: 0.25,
            rho: 0.5,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.0,
        }),
        TradeInstrument::RangeAccrual(RangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.05,
            lower_bound: 0.01,
            upper_bound: 0.04,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }),
        TradeInstrument::DualRangeAccrual(DualRangeAccrual {
            notional: 1_000_000.0,
            coupon_rate: 0.04,
            lower_bound: -0.01,
            upper_bound: 0.01,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            payment_time: 1.0,
        }),
        TradeInstrument::DeferInvestmentOption(DeferInvestmentOption {
            model: common_real_model.clone(),
            investment_cost: 80.0,
        }),
        TradeInstrument::ExpandOption(ExpandOption {
            model: common_real_model.clone(),
            expansion_multiplier: 1.25,
            expansion_cost: 20.0,
        }),
        TradeInstrument::AbandonmentOption(AbandonmentOption {
            model: common_real_model.clone(),
            salvage_value: 30.0,
        }),
        TradeInstrument::RealOptionInstrument(RealOptionInstrument::Defer(DeferInvestmentOption {
            model: common_real_model,
            investment_cost: 75.0,
        })),
        TradeInstrument::SpreadOption(SpreadOption {
            s1: 100.0,
            s2: 105.0,
            k: 3.0,
            vol1: 0.2,
            vol2: 0.18,
            rho: 0.4,
            q1: 0.01,
            q2: 0.02,
            r: 0.03,
            t: 1.0,
        }),
        TradeInstrument::SwingOption(SwingOption::new(1, 3, vec![0.25, 0.5, 0.75], 100.0, 10.0)),
        TradeInstrument::Tarf(Tarf {
            strike: 1.10,
            notional_per_fixing: 100_000.0,
            ko_barrier: 1.25,
            target_profit: 15_000.0,
            downside_leverage: 2.0,
            fixing_times: vec![0.25, 0.5, 0.75, 1.0],
            tarf_type: TarfType::Standard,
        }),
        TradeInstrument::VanillaOption(VanillaOption {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 1.0,
            exercise: ExerciseStyle::Bermudan {
                dates: vec![0.5, 0.75, 1.0],
            },
        }),
        TradeInstrument::VarianceSwap(VarianceSwap::new(
            100_000.0,
            0.20,
            1.0,
            vec![VarianceOptionQuote::new(90.0, 12.0, 2.0)],
        )),
        TradeInstrument::VolatilitySwap(VolatilitySwap::new(
            100_000.0,
            0.22,
            1.0,
            vec![VarianceOptionQuote::new(100.0, 8.0, 7.5)],
            0.06,
        )),
        TradeInstrument::WeatherSwap(WeatherSwap {
            index_type: DegreeDayType::CDD,
            strike: 250.0,
            tick_size: 10.0,
            notional: 200_000.0,
            is_payer: true,
            discount_rate: 0.03,
            maturity: 1.0,
        }),
        TradeInstrument::WeatherOption(WeatherOption {
            index_type: DegreeDayType::HDD,
            option_type: OptionType::Put,
            strike: 500.0,
            tick_size: 15.0,
            notional: 250_000.0,
            discount_rate: 0.03,
            maturity: 1.0,
        }),
        TradeInstrument::CatastropheBond(CatastropheBond {
            principal: 1_000_000.0,
            coupon_rate: 0.08,
            risk_free_rate: 0.03,
            maturity: 3.0,
            coupon_frequency: 2,
            loss_intensity: 0.15,
            expected_loss_per_event: 0.25,
        }),
        TradeInstrument::MbsPassThrough(MbsPassThrough {
            original_balance: 100_000.0,
            coupon_rate: 0.05,
            servicing_fee: 0.0025,
            original_term: 360,
            age: 24,
            prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.08 }),
        }),
    ]
}

#[test]
fn core_and_pricing_result_roundtrip() {
    assert_roundtrip(&OptionType::Call);
    assert_roundtrip(&ExerciseStyle::American);
    assert_roundtrip(&BarrierDirection::Down);
    assert_roundtrip(&BarrierStyle::Out);
    assert_roundtrip(&BarrierSpec {
        direction: BarrierDirection::Up,
        style: BarrierStyle::In,
        level: 120.0,
        rebate: 2.0,
    });
    assert_roundtrip(&Averaging::Geometric);
    assert_roundtrip(&StrikeType::Floating);
    assert_roundtrip(&AsianSpec {
        averaging: Averaging::Arithmetic,
        strike_type: StrikeType::Fixed,
        observation_times: vec![0.2, 0.4, 0.6, 0.8, 1.0],
    });

    let mut diagnostics = Diagnostics::new();
    diagnostics.insert("delta", 0.5);
    diagnostics.insert("vol", 10.0);

    assert_roundtrip(&diagnostics);
    assert_roundtrip(&Greeks {
        delta: 0.5,
        gamma: 0.02,
        vega: 12.0,
        theta: -1.5,
        rho: 4.1,
    });

    let result = PricingResult {
        price: 12.34,
        stderr: Some(0.01),
        greeks: Some(Greeks {
            delta: 0.55,
            gamma: 0.021,
            vega: 11.5,
            theta: -1.2,
            rho: 5.0,
        }),
        diagnostics,
    };
    assert_roundtrip(&result);
    assert_roundtrip(&PricingError::MarketDataMissing(
        "curve missing".to_string(),
    ));
}

#[test]
fn market_data_roundtrip_json_and_msgpack() {
    let curve = sample_yield_curve();
    assert_roundtrip(&curve);

    let json = serde_json::to_vec(&curve).expect("serialize curve");
    let curve_back: YieldCurve = serde_json::from_slice(&json).expect("deserialize curve");
    let t = 0.9;
    assert!((curve.discount_factor(t) - curve_back.discount_factor(t)).abs() < 1.0e-12);

    let surface = sample_vol_surface();
    assert_roundtrip(&surface);

    let survival = SurvivalCurve::new(vec![(1.0, 0.98), (3.0, 0.92), (5.0, 0.86)]);
    assert_roundtrip(&survival);

    let sampled = SampledVolSurface::new(
        vec![80.0, 100.0, 120.0],
        vec![0.5, 1.0],
        vec![vec![0.25, 0.2, 0.22], vec![0.24, 0.21, 0.23]],
    )
    .expect("valid sampled surface");
    assert_roundtrip(&sampled);

    let market_from_custom = Market::builder()
        .spot(100.0)
        .rate(0.02)
        .dividend_yield(0.01)
        .vol_surface(Box::new(QuadraticSmile {
            base: 0.2,
            spot: 100.0,
            skew: 0.15,
        }))
        .reference_date("2026-02-22")
        .build()
        .expect("market build");
    let market_custom_json =
        serde_json::to_vec(&market_from_custom).expect("serialize custom market");
    let market_from_custom_json: Market =
        serde_json::from_slice(&market_custom_json).expect("deserialize custom market");
    assert_eq!(market_from_custom_json.spot, market_from_custom.spot);
    assert_eq!(market_from_custom_json.rate, market_from_custom.rate);
    assert_eq!(
        market_from_custom_json.dividend_yield,
        market_from_custom.dividend_yield
    );
    assert_eq!(
        market_from_custom_json.reference_date,
        market_from_custom.reference_date
    );
    for &(k, t) in &[(90.0, 0.5), (100.0, 1.0), (110.0, 2.0)] {
        let a = market_from_custom.vol_for(k, t);
        let b = market_from_custom_json.vol_for(k, t);
        assert!(
            (a - b).abs() <= 1.0e-10,
            "custom market vol mismatch at k={k}, t={t}"
        );
    }

    let market_custom_msgpack =
        rmp_serde::to_vec_named(&market_from_custom).expect("serialize custom market msgpack");
    let market_from_custom_msgpack: Market =
        rmp_serde::from_slice(&market_custom_msgpack).expect("deserialize custom market msgpack");
    for &(k, t) in &[(90.0, 0.5), (100.0, 1.0), (110.0, 2.0)] {
        let a = market_from_custom.vol_for(k, t);
        let b = market_from_custom_msgpack.vol_for(k, t);
        assert!(
            (a - b).abs() <= 1.0e-10,
            "custom market msgpack vol mismatch at k={k}, t={t}"
        );
    }

    let market_from_parametric = Market::builder()
        .spot(100.0)
        .rate(0.02)
        .dividend_yield(0.0)
        .vol_surface(Box::new(surface.clone()))
        .build()
        .expect("market build");
    assert_roundtrip(&market_from_parametric);

    assert_roundtrip(&VolSource::Flat(0.2));
    assert_roundtrip(&VolSource::Parametric(surface.clone()));
    assert_roundtrip(&VolSource::Sampled(sampled.clone()));

    let snapshot = MarketSnapshot {
        snapshot_id: "snap-001".to_string(),
        timestamp_unix_ms: 1_771_744_000_000,
        markets: vec![("equity-us".to_string(), market_from_parametric.clone())],
        yield_curves: vec![("usd-discount".to_string(), curve)],
        vol_surfaces: vec![("spx-svi".to_string(), surface)],
        credit_curves: vec![CreditCurveSnapshot {
            curve_id: "acme-5y".to_string(),
            survival_curve: survival,
            recovery_rate: 0.4,
        }],
        spot_prices: vec![("SPX".to_string(), 5100.0)],
        forward_curves: vec![ForwardCurveSnapshot {
            asset_id: "WTI".to_string(),
            points: vec![(0.5, 72.0), (1.0, 74.0)],
        }],
    };
    assert_roundtrip(&snapshot);
}

#[test]
fn instrument_and_portfolio_roundtrip() {
    assert_roundtrip(&PsaModel { psa_speed: 1.5 });
    assert_roundtrip(&ConstantCpr { annual_cpr: 0.07 });
    assert_roundtrip(&PrepaymentModel::Psa(PsaModel { psa_speed: 2.0 }));
    assert_roundtrip(&MbsCashflow {
        month: 12,
        interest: 400.0,
        scheduled_principal: 300.0,
        prepayment: 100.0,
        total_principal: 400.0,
        remaining_balance: 99_600.0,
        total_cashflow: 800.0,
    });
    assert_roundtrip(&VarianceOptionQuote::new(100.0, 8.0, 7.5));
    assert_roundtrip(&DiscreteCashFlow {
        time: 1.0,
        amount: 10.0,
    });
    assert_roundtrip(&TarfType::Decumulator);
    assert_roundtrip(&BasketType::WorstOf);
    assert_roundtrip(&DoubleBarrierType::KnockIn);
    assert_roundtrip(&DegreeDayType::CDD);

    let instruments = sample_trade_instruments();
    for instrument in instruments.iter().cloned() {
        assert_roundtrip(&instrument);
    }

    let trades = instruments
        .into_iter()
        .enumerate()
        .map(|(idx, instrument)| Trade {
            metadata: TradeMetadata {
                trade_id: format!("T-{idx:04}"),
                version: 1,
                timestamp_unix_ms: 1_771_744_000_000 + idx as i64,
            },
            instrument,
        })
        .collect::<Vec<_>>();

    let portfolio = Portfolio {
        portfolio_id: "book-alpha".to_string(),
        market_snapshot_id: Some("snap-001".to_string()),
        trades,
    };

    assert_roundtrip(&portfolio);
}
