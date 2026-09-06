//! Exact and independently sourced equity/exotics pricing references.
//!
//! Reference provenance:
//! - `vendor/QuantLib/test-suite/lookbackoptions.cpp` (Haug 1998 and
//!   Broadie--Glasserman--Kou 1999 cached values).
//! - `vendor/QuantLib/test-suite/compoundoption.cpp` (Haug 2007, Sitmo, and
//!   MathFinance cached values).
//! - `vendor/QuantLib/test-suite/doublebarrieroption.cpp` (Haug 2006 cached
//!   values).
//! - `vendor/QuantLib/test-suite/basketoption.cpp` (three-asset payoff reduced
//!   to the published two-asset max/min basket cases).
//! - SciPy 1.17.1 `scipy.stats.norm.cdf`, evaluated independently for the
//!   cash/asset/gap digital, quanto, FX, and one-asset basket values below.
//! - SciPy 1.17.1 independently scrambled Sobol replicates and NumPy 2.4.3
//!   Cholesky correlation for the general stochastic basket values below.
//! - Deterministic limiting cases are derived directly from their discounted
//!   contractual cash flows and therefore have no Monte Carlo tolerance.

use openferric::core::{OptionType, PricingEngine};
use openferric::engines::analytic::{
    BlackScholesEngine, DigitalAnalyticEngine, DoubleBarrierAnalyticEngine, ExoticAnalyticEngine,
    GarmanKohlhagenEngine,
};
use openferric::engines::tree::{ConvertibleBinomialEngine, SwingTreeEngine};
use openferric::instruments::structured_notes::{
    CallableRateNote, CouponScheduleBuilder, ExerciseSchedule,
};
use openferric::instruments::{
    AssetOrNothingOption, Autocallable, BasketOption, BasketType, CashOrNothingOption,
    CompoundOption, ConvertibleBond, DeferInvestmentOption, DoubleBarrierOption, DoubleBarrierType,
    DualRangeAccrual, ExoticOption, FxOption, GapOption, LookbackFixedOption,
    LookbackFloatingOption, OutperformanceBasketOption, PhoenixAutocallable, QuantoBasketOption,
    QuantoOption, RangeAccrual, RealOptionBinomialSpec, SwingOption, VanillaOption,
};
use openferric::market::Market;
use openferric::math::FactorCorrelationModel;
use openferric::pricing::autocallable::{price_autocallable, price_phoenix_autocallable};
use openferric::pricing::basket::{
    BasketCopula, BasketMomentMatchingMethod, price_basket_mc_with_copula,
    price_basket_mc_with_factor_model, price_basket_moment_matching,
    price_outperformance_basket_mc, price_quanto_basket_mc,
};
use openferric::pricing::range_accrual::{dual_range_accrual_mc_price, range_accrual_mc_price};
use openferric::pricing::real_option::price_option_to_defer;
use openferric::rates::{Frequency, YieldCurve};

fn market(spot: f64, rate: f64, dividend_yield: f64, vol: f64) -> Market {
    Market::builder()
        .spot(spot)
        .rate(rate)
        .dividend_yield(dividend_yield)
        .flat_vol(vol)
        .build()
        .expect("valid test market")
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label}: expected={expected:.15}, actual={actual:.15}, error={error:.3e}, tolerance={tolerance:.3e}"
    );
}

fn assert_mc_matches_qmc(
    label: &str,
    actual: f64,
    implementation_stderr: Option<f64>,
    reference: f64,
    reference_stderr: f64,
) {
    let implementation_stderr = implementation_stderr.expect("MC reports standard error");
    assert!(
        implementation_stderr.is_finite() && implementation_stderr > 0.0,
        "{label}: invalid implementation standard error {implementation_stderr}"
    );
    let combined_stderr = implementation_stderr.hypot(reference_stderr);
    let roundoff = 32.0 * f64::EPSILON * actual.abs().max(reference.abs()).max(1.0);
    let budget = 4.0 * combined_stderr + roundoff;
    let error = (actual - reference).abs();
    assert!(
        error <= budget,
        "{label}: reference={reference:.17e} +/- {reference_stderr:.3e}, \
         actual={actual:.17e} +/- {implementation_stderr:.3e}, \
         error={error:.3e}, four-combined-stderr budget={budget:.3e}"
    );
}

fn price_exotic(option: ExoticOption, market: &Market) -> f64 {
    ExoticAnalyticEngine::new()
        .price(&option, market)
        .expect("exotic pricing succeeds")
        .price
}

#[test]
fn digitals_match_independent_scipy_closed_form_values() {
    let engine = DigitalAnalyticEngine::new();
    let m = market(100.0, 0.06, 0.06, 0.35);

    let cash_call = engine
        .price(
            &CashOrNothingOption::new(OptionType::Call, 80.0, 10.0, 0.75),
            &m,
        )
        .unwrap()
        .price;
    let cash_put = engine
        .price(
            &CashOrNothingOption::new(OptionType::Put, 80.0, 10.0, 0.75),
            &m,
        )
        .unwrap()
        .price;
    assert_close(
        "cash-or-nothing call",
        cash_call,
        6.888_929_133_869_653,
        2.0e-12,
    );
    assert_close(
        "cash-or-nothing put",
        cash_put,
        2.671_045_684_461_347,
        2.0e-12,
    );

    let asset_put = engine
        .price(
            &AssetOrNothingOption::new(OptionType::Put, 65.0, 0.50),
            &market(70.0, 0.07, 0.05, 0.27),
        )
        .unwrap()
        .price;
    assert_close(
        "asset-or-nothing put",
        asset_put,
        20.206_947_298_368_55,
        2.0e-12,
    );

    let gap_call = engine
        .price(
            &GapOption::new(OptionType::Call, 57.0, 50.0, 0.50),
            &market(50.0, 0.09, 0.0, 0.20),
        )
        .unwrap()
        .price;
    assert_close("gap call", gap_call, -0.005_252_489_258_786_852, 2.0e-12);
}

#[test]
fn floating_lookbacks_match_quantlib_reference_grid() {
    // option type, running extreme, spot, q, r, T, vol, expected
    let cases = [
        (
            OptionType::Call,
            100.0,
            120.0,
            0.06,
            0.10,
            0.50,
            0.30,
            25.3533,
        ),
        (
            OptionType::Call,
            100.0,
            100.0,
            0.00,
            0.05,
            1.00,
            0.30,
            23.7884,
        ),
        (
            OptionType::Call,
            100.0,
            100.0,
            0.00,
            0.05,
            0.20,
            0.30,
            10.7190,
        ),
        (
            OptionType::Call,
            100.0,
            110.0,
            0.00,
            0.05,
            0.20,
            0.30,
            14.4597,
        ),
        (
            OptionType::Put,
            100.0,
            100.0,
            0.00,
            0.10,
            0.50,
            0.30,
            15.3526,
        ),
        (
            OptionType::Put,
            110.0,
            100.0,
            0.00,
            0.10,
            0.50,
            0.30,
            16.8468,
        ),
        (
            OptionType::Put,
            120.0,
            100.0,
            0.00,
            0.10,
            0.50,
            0.30,
            21.0645,
        ),
    ];

    for (idx, (option_type, extreme, spot, q, r, expiry, vol, expected)) in
        cases.into_iter().enumerate()
    {
        let option = ExoticOption::LookbackFloating(LookbackFloatingOption {
            option_type,
            expiry,
            observed_extreme: Some(extreme),
        });
        let actual = price_exotic(option, &market(spot, r, q, vol));
        assert_close(
            &format!("floating lookback case {idx}"),
            actual,
            expected,
            1.1e-4,
        );
    }
}

#[test]
fn fixed_lookbacks_match_full_quantlib_haug_grid() {
    // option type, strike, T, vol, expected. S = running extreme = 100,
    // q = 0, and r = 10% for every Haug case.
    let cases = [
        (OptionType::Call, 95.0, 0.50, 0.10, 13.2687),
        (OptionType::Call, 95.0, 0.50, 0.20, 18.9263),
        (OptionType::Call, 95.0, 0.50, 0.30, 24.9857),
        (OptionType::Call, 100.0, 0.50, 0.10, 8.5126),
        (OptionType::Call, 100.0, 0.50, 0.20, 14.1702),
        (OptionType::Call, 100.0, 0.50, 0.30, 20.2296),
        (OptionType::Call, 105.0, 0.50, 0.10, 4.3908),
        (OptionType::Call, 105.0, 0.50, 0.20, 9.8905),
        (OptionType::Call, 105.0, 0.50, 0.30, 15.8512),
        (OptionType::Call, 95.0, 1.00, 0.10, 18.3241),
        (OptionType::Call, 95.0, 1.00, 0.20, 26.0731),
        (OptionType::Call, 95.0, 1.00, 0.30, 34.7116),
        (OptionType::Call, 100.0, 1.00, 0.10, 13.8000),
        (OptionType::Call, 100.0, 1.00, 0.20, 21.5489),
        (OptionType::Call, 100.0, 1.00, 0.30, 30.1874),
        (OptionType::Call, 105.0, 1.00, 0.10, 9.5445),
        (OptionType::Call, 105.0, 1.00, 0.20, 17.2965),
        (OptionType::Call, 105.0, 1.00, 0.30, 25.9002),
        (OptionType::Put, 95.0, 0.50, 0.10, 0.6899),
        (OptionType::Put, 95.0, 0.50, 0.20, 4.4448),
        (OptionType::Put, 95.0, 0.50, 0.30, 8.9213),
        (OptionType::Put, 100.0, 0.50, 0.10, 3.3917),
        (OptionType::Put, 100.0, 0.50, 0.20, 8.3177),
        (OptionType::Put, 100.0, 0.50, 0.30, 13.1579),
        (OptionType::Put, 105.0, 0.50, 0.10, 8.1478),
        (OptionType::Put, 105.0, 0.50, 0.20, 13.0739),
        (OptionType::Put, 105.0, 0.50, 0.30, 17.9140),
        (OptionType::Put, 95.0, 1.00, 0.10, 1.0534),
        (OptionType::Put, 95.0, 1.00, 0.20, 6.2813),
        (OptionType::Put, 95.0, 1.00, 0.30, 12.2376),
        (OptionType::Put, 100.0, 1.00, 0.10, 3.8079),
        (OptionType::Put, 100.0, 1.00, 0.20, 10.1294),
        (OptionType::Put, 100.0, 1.00, 0.30, 16.3889),
        (OptionType::Put, 105.0, 1.00, 0.10, 8.3321),
        (OptionType::Put, 105.0, 1.00, 0.20, 14.6536),
        (OptionType::Put, 105.0, 1.00, 0.30, 20.9130),
    ];

    for (idx, (option_type, strike, expiry, vol, expected)) in cases.into_iter().enumerate() {
        let option = ExoticOption::LookbackFixed(LookbackFixedOption {
            option_type,
            strike,
            expiry,
            observed_extreme: Some(100.0),
        });
        let actual = price_exotic(option, &market(100.0, 0.10, 0.0, vol));
        assert_close(
            &format!("fixed lookback case {idx}"),
            actual,
            expected,
            1.1e-4,
        );
    }
}

#[test]
fn compound_options_match_quantlib_reference_grid() {
    // outer type, inner type, outer K, inner K, S, q, r, T1, T2, vol, expected
    let cases = [
        (
            OptionType::Put,
            OptionType::Call,
            50.0,
            520.0,
            500.0,
            0.03,
            0.08,
            0.25,
            0.5,
            0.35,
            21.1965,
        ),
        (
            OptionType::Call,
            OptionType::Call,
            50.0,
            520.0,
            500.0,
            0.03,
            0.08,
            0.25,
            0.5,
            0.35,
            17.5945,
        ),
        (
            OptionType::Call,
            OptionType::Put,
            50.0,
            520.0,
            500.0,
            0.03,
            0.08,
            0.25,
            0.5,
            0.35,
            18.7128,
        ),
        (
            OptionType::Put,
            OptionType::Put,
            50.0,
            520.0,
            500.0,
            0.03,
            0.08,
            0.25,
            0.5,
            0.35,
            15.2601,
        ),
        (
            OptionType::Call,
            OptionType::Call,
            0.05,
            1.14,
            1.20,
            0.0,
            0.01,
            0.5,
            2.0,
            0.11,
            0.0729,
        ),
        (
            OptionType::Call,
            OptionType::Put,
            0.05,
            1.14,
            1.20,
            0.0,
            0.01,
            0.5,
            2.0,
            0.11,
            0.0074,
        ),
        (
            OptionType::Put,
            OptionType::Call,
            0.05,
            1.14,
            1.20,
            0.0,
            0.01,
            0.5,
            2.0,
            0.11,
            0.0021,
        ),
        (
            OptionType::Put,
            OptionType::Put,
            0.05,
            1.14,
            1.20,
            0.0,
            0.01,
            0.5,
            2.0,
            0.11,
            0.0192,
        ),
        (
            OptionType::Call,
            OptionType::Call,
            10.0,
            122.0,
            120.0,
            0.06,
            0.02,
            0.1,
            0.7,
            0.22,
            0.4419,
        ),
        (
            OptionType::Call,
            OptionType::Put,
            10.0,
            122.0,
            120.0,
            0.06,
            0.02,
            0.1,
            0.7,
            0.22,
            2.6112,
        ),
        (
            OptionType::Put,
            OptionType::Call,
            10.0,
            122.0,
            120.0,
            0.06,
            0.02,
            0.1,
            0.7,
            0.22,
            4.1616,
        ),
        (
            OptionType::Put,
            OptionType::Put,
            10.0,
            122.0,
            120.0,
            0.06,
            0.02,
            0.1,
            0.7,
            0.22,
            1.0914,
        ),
        (
            OptionType::Call,
            OptionType::Call,
            0.4,
            8.2,
            8.0,
            0.05,
            0.0,
            2.0,
            3.0,
            0.08,
            0.0099,
        ),
        (
            OptionType::Call,
            OptionType::Put,
            0.4,
            8.2,
            8.0,
            0.05,
            0.0,
            2.0,
            3.0,
            0.08,
            0.9826,
        ),
        (
            OptionType::Put,
            OptionType::Call,
            0.4,
            8.2,
            8.0,
            0.05,
            0.0,
            2.0,
            3.0,
            0.08,
            0.3585,
        ),
        (
            OptionType::Put,
            OptionType::Put,
            0.4,
            8.2,
            8.0,
            0.05,
            0.0,
            2.0,
            3.0,
            0.08,
            0.0168,
        ),
        (
            OptionType::Call,
            OptionType::Call,
            0.02,
            1.6,
            1.6,
            0.013,
            0.022,
            0.45,
            0.5,
            0.17,
            0.0680,
        ),
        (
            OptionType::Call,
            OptionType::Put,
            0.02,
            1.6,
            1.6,
            0.013,
            0.022,
            0.45,
            0.5,
            0.17,
            0.0605,
        ),
        (
            OptionType::Put,
            OptionType::Call,
            0.02,
            1.6,
            1.6,
            0.013,
            0.022,
            0.45,
            0.5,
            0.17,
            0.0081,
        ),
        (
            OptionType::Put,
            OptionType::Put,
            0.02,
            1.6,
            1.6,
            0.013,
            0.022,
            0.45,
            0.5,
            0.17,
            0.0078,
        ),
    ];

    for (idx, (outer, inner, k1, k2, spot, q, r, t1, t2, vol, expected)) in
        cases.into_iter().enumerate()
    {
        let option = ExoticOption::Compound(CompoundOption {
            option_type: outer,
            underlying_option_type: inner,
            compound_strike: k1,
            underlying_strike: k2,
            compound_expiry: t1,
            underlying_expiry: t2,
        });
        let actual = price_exotic(option, &market(spot, r, q, vol));
        assert_close(&format!("compound case {idx}"), actual, expected, 1.1e-3);
    }
}

#[test]
fn compound_option_split_preserves_put_call_parity_for_both_daughters() {
    let m = market(100.0, 0.04, 0.01, 0.25);
    let compound_strike = 6.0;
    let underlying_strike = 100.0;
    let t1 = 0.4;
    let t2 = 1.2;
    let exotic_engine = ExoticAnalyticEngine::new();
    let vanilla_engine = BlackScholesEngine::new();

    for inner in [OptionType::Call, OptionType::Put] {
        let price_outer = |outer| {
            exotic_engine
                .price(
                    &ExoticOption::Compound(CompoundOption {
                        option_type: outer,
                        underlying_option_type: inner,
                        compound_strike,
                        underlying_strike,
                        compound_expiry: t1,
                        underlying_expiry: t2,
                    }),
                    &m,
                )
                .unwrap()
                .price
        };
        let daughter = vanilla_engine
            .price(
                &match inner {
                    OptionType::Call => VanillaOption::european_call(underlying_strike, t2),
                    OptionType::Put => VanillaOption::european_put(underlying_strike, t2),
                },
                &m,
            )
            .unwrap()
            .price;
        let lhs = price_outer(OptionType::Call) + compound_strike * (-m.rate * t1).exp();
        let rhs = price_outer(OptionType::Put) + daughter;
        assert_close("compound put-call parity", lhs, rhs, 3.0e-12);
    }
}

#[test]
fn compound_call_with_unreachable_put_value_is_exactly_zero() {
    // At T1 the daughter put is bounded above by K2*exp(-r*(T2-T1)),
    // which is below the mother strike in this case.
    let option = ExoticOption::Compound(CompoundOption {
        option_type: OptionType::Call,
        underlying_option_type: OptionType::Put,
        compound_strike: 120.0,
        underlying_strike: 100.0,
        compound_expiry: 0.5,
        underlying_expiry: 1.0,
    });
    let actual = price_exotic(option, &market(100.0, 0.04, 0.01, 0.25));
    assert_eq!(actual, 0.0);
}

#[test]
fn quanto_and_fx_match_independent_scipy_values() {
    let quanto = ExoticOption::Quanto(QuantoOption {
        option_type: OptionType::Call,
        strike: 100.0,
        expiry: 1.0,
        fx_rate: 1.5,
        foreign_rate: 0.02,
        fx_vol: 0.10,
        asset_fx_corr: -0.30,
    });
    let quanto_price = price_exotic(quanto, &market(100.0, 0.05, 0.0, 0.20));
    assert_close("quanto call", quanto_price, 13.491_372_579_960_611, 3.0e-12);

    let fx = FxOption::new(OptionType::Call, 0.05, 0.03, 1.25, 1.30, 0.10, 0.50);
    let fx_price = GarmanKohlhagenEngine::new()
        .price(&fx, &market(1.0, 0.01, 0.0, 0.20))
        .unwrap()
        .price;
    assert_close(
        "Garman-Kohlhagen call",
        fx_price,
        0.019_953_800_723_103_543,
        2.0e-14,
    );
}

#[test]
fn double_barriers_match_quantlib_haug_call_put_and_knock_in_values() {
    // barrier type, option type, lower, upper, T, vol, expected
    let cases = [
        (
            DoubleBarrierType::KnockOut,
            OptionType::Put,
            80.0,
            120.0,
            0.25,
            0.25,
            2.6866,
        ),
        (
            DoubleBarrierType::KnockOut,
            OptionType::Put,
            90.0,
            110.0,
            0.50,
            0.25,
            0.0491,
        ),
        (
            DoubleBarrierType::KnockIn,
            OptionType::Call,
            50.0,
            150.0,
            0.25,
            0.25,
            0.0900,
        ),
        (
            DoubleBarrierType::KnockIn,
            OptionType::Call,
            60.0,
            140.0,
            0.50,
            0.25,
            3.2439,
        ),
        (
            DoubleBarrierType::KnockIn,
            OptionType::Call,
            80.0,
            120.0,
            0.25,
            0.35,
            6.7007,
        ),
        (
            DoubleBarrierType::KnockIn,
            OptionType::Call,
            90.0,
            110.0,
            0.50,
            0.25,
            9.5382,
        ),
    ];
    let engine = DoubleBarrierAnalyticEngine::new().with_series_terms(30);

    for (idx, (kind, option_type, lower, upper, expiry, vol, expected)) in
        cases.into_iter().enumerate()
    {
        let option = DoubleBarrierOption::new(option_type, 100.0, expiry, lower, upper, kind, 0.0);
        let actual = engine
            .price(&option, &market(100.0, 0.10, 0.0, vol))
            .unwrap()
            .price;
        assert_close(
            &format!("double barrier case {idx}"),
            actual,
            expected,
            2.0e-4,
        );
    }
}

#[test]
fn one_asset_basket_moment_matching_is_exact_black_scholes() {
    let basket = BasketOption {
        weights: vec![1.0],
        strike: 100.0,
        maturity: 1.4,
        is_call: true,
        basket_type: BasketType::Average,
    };
    for method in [
        BasketMomentMatchingMethod::Levy,
        BasketMomentMatchingMethod::Gentle,
    ] {
        let actual = price_basket_moment_matching(
            &basket,
            &[105.0],
            &[0.24],
            &[vec![1.0]],
            0.03,
            &[0.012],
            method,
        )
        .unwrap();
        assert_close(
            "one-asset basket call",
            actual,
            15.300_792_597_601_458,
            3.0e-12,
        );
    }
}

#[test]
fn zero_vol_basket_mc_methods_equal_discounted_terminal_payoffs() {
    let spots = [100.0, 120.0];
    let vols = [0.0, 0.0];
    let dividends = [0.01, 0.03];
    let corr = [vec![1.0, 0.35], vec![0.35, 1.0]];
    let rate: f64 = 0.04;
    let maturity: f64 = 1.25;
    let terminal = [
        spots[0] * ((rate - dividends[0]) * maturity).exp(),
        spots[1] * ((rate - dividends[1]) * maturity).exp(),
    ];
    let discount = (-rate * maturity).exp();

    let average = BasketOption {
        weights: vec![0.4, 0.6],
        strike: 105.0,
        maturity,
        is_call: true,
        basket_type: BasketType::Average,
    };
    let average_expected = discount * (0.4 * terminal[0] + 0.6 * terminal[1] - 105.0);
    for copula in [
        BasketCopula::Gaussian,
        BasketCopula::StudentT {
            degrees_of_freedom: 5,
        },
    ] {
        let result = price_basket_mc_with_copula(
            &average, &spots, &vols, &corr, rate, &dividends, 1, copula,
        );
        assert_close("matrix basket MC", result.price, average_expected, 2.0e-14);
        assert_eq!(result.stderr, Some(0.0));
    }

    let factor_result = price_basket_mc_with_factor_model(
        &average,
        &spots,
        &vols,
        &FactorCorrelationModel::OneFactor {
            loadings: vec![0.5, 0.7],
        },
        rate,
        &dividends,
        1,
        BasketCopula::Gaussian,
    );
    assert_close(
        "factor basket MC",
        factor_result.price,
        average_expected,
        2.0e-14,
    );
    assert_eq!(factor_result.stderr, Some(0.0));

    for (basket_type, terminal_underlying) in [
        (
            BasketType::BestOf,
            (terminal[0] / spots[0]).max(terminal[1] / spots[1]),
        ),
        (
            BasketType::WorstOf,
            (terminal[0] / spots[0]).min(terminal[1] / spots[1]),
        ),
    ] {
        let basket = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity,
            is_call: true,
            basket_type,
        };
        let result = price_basket_mc_with_copula(
            &basket,
            &spots,
            &vols,
            &corr,
            rate,
            &dividends,
            1,
            BasketCopula::Gaussian,
        );
        assert_close(
            "best/worst basket MC",
            result.price,
            discount * (terminal_underlying - 1.0),
            2.0e-14,
        );
        assert_eq!(result.stderr, Some(0.0));
    }
}

#[test]
fn best_and_worst_basket_mc_match_quantlib_stulz_references() {
    // QuantLib's cached values are 21.619 and 6.844 for S1=S2=K=100.
    // BasketType::BestOf/WorstOf is expressed on normalized performances, so
    // the corresponding unit-notional prices are 0.21619 and 0.06844.
    let spots = [100.0, 100.0];
    let vols = [0.30, 0.30];
    let dividends = [0.0, 0.0];
    let corr = [vec![1.0, 0.50], vec![0.50, 1.0]];

    for (basket_type, expected) in [
        (BasketType::BestOf, 0.21619),
        (BasketType::WorstOf, 0.06844),
    ] {
        let basket = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type,
        };
        let result = price_basket_mc_with_copula(
            &basket,
            &spots,
            &vols,
            &corr,
            0.05,
            &dividends,
            1_000_000,
            BasketCopula::Gaussian,
        );
        let tolerance = 4.0 * result.stderr.expect("MC stderr") + 1.0e-5;
        assert_close("QuantLib basket MC", result.price, expected, tolerance);
    }
}

#[test]
fn general_stochastic_baskets_match_independent_scipy_sobol_references() {
    // Each cached oracle below is the mean of 32 independently Owen-scrambled
    // SciPy 1.17.1 Sobol replicates with 2^20 points per replicate. NumPy 2.4.3
    // maps the Sobol uniforms through the inverse normal CDF and applies a
    // Cholesky factor to the stated correlation matrix. REFERENCE_SE is the
    // sample standard deviation of the 32 replicate prices divided by sqrt(32).
    // The implementation uses an unrelated Xoshiro pseudorandom stream.
    const IMPLEMENTATION_PATHS: usize = 500_000;

    let spots3 = [100.0, 98.0, 102.0];
    let vols3 = [0.25, 0.22, 0.20];
    let dividends3 = [0.0; 3];
    let corr3 = [
        vec![1.0, 0.40, 0.20],
        vec![0.40, 1.0, 0.30],
        vec![0.20, 0.30, 1.0],
    ];
    // Scramble seeds 273000..273031. BasketType::BestOf/WorstOf prices calls
    // on max/min_i(S_i(T)/S_i(0)) with K=1, r=1%, q_i=0, and T=1.
    for (basket_type, label, reference, reference_stderr) in [
        (
            BasketType::BestOf,
            "three-asset best-of call",
            0.193_257_668_769_514_9,
            8.057_553_280_924_98e-8,
        ),
        (
            BasketType::WorstOf,
            "three-asset worst-of call",
            0.017_982_413_968_364_988,
            2.996_601_505_311_016e-8,
        ),
    ] {
        let basket = BasketOption {
            weights: vec![],
            strike: 1.0,
            maturity: 1.0,
            is_call: true,
            basket_type,
        };
        let result = price_basket_mc_with_copula(
            &basket,
            &spots3,
            &vols3,
            &corr3,
            0.01,
            &dividends3,
            IMPLEMENTATION_PATHS,
            BasketCopula::Gaussian,
        );
        assert_mc_matches_qmc(
            label,
            result.price,
            result.stderr,
            reference,
            reference_stderr,
        );
    }

    // Five-asset weighted-average call with K=100, r=2%, q_i=0, T=1 and
    // the exact two-factor loadings below. The independent reference uses the
    // implied correlation rho_ij=sum_k(beta_ik*beta_jk), not the factor sampler
    // under test. Scramble seeds are 183000..183031.
    let factor_basket = BasketOption {
        weights: vec![0.30, 0.25, 0.20, 0.15, 0.10],
        strike: 100.0,
        maturity: 1.0,
        is_call: true,
        basket_type: BasketType::Average,
    };
    let factor_model = FactorCorrelationModel::MultiFactor {
        loadings: vec![
            vec![0.60, 0.10],
            vec![0.50, -0.10],
            vec![0.40, 0.20],
            vec![0.35, -0.20],
            vec![0.25, 0.15],
        ],
    };
    let factor_result = price_basket_mc_with_factor_model(
        &factor_basket,
        &[100.0, 101.0, 99.0, 97.0, 103.0],
        &[0.20, 0.22, 0.18, 0.21, 0.19],
        &factor_model,
        0.02,
        &[0.0; 5],
        IMPLEMENTATION_PATHS,
        BasketCopula::Gaussian,
    );
    assert_mc_matches_qmc(
        "five-asset two-factor average call",
        factor_result.price,
        factor_result.stderr,
        5.919_332_695_780_863,
        1.178_336_468_285_617_3e-5,
    );

    // Three-asset outperformance call: S=[100,95,105], sigma=[20%,22%,21%],
    // rho=[[1,.4,.35],[.4,1,.3],[.35,.3,1]], r=2%, q_i=0, T=1,
    // leader=asset 0, lagger=.5*S1+.5*S2, K=1. Seeds 293000..293031.
    let outperformance = OutperformanceBasketOption {
        leader_index: 0,
        lagger_weights: vec![0.0, 0.5, 0.5],
        strike: 1.0,
        maturity: 1.0,
        option_type: OptionType::Call,
    };
    let outperformance_result = price_outperformance_basket_mc(
        &outperformance,
        &[100.0, 95.0, 105.0],
        &[0.20, 0.22, 0.21],
        &[
            vec![1.0, 0.40, 0.35],
            vec![0.40, 1.0, 0.30],
            vec![0.35, 0.30, 1.0],
        ],
        0.02,
        &[0.0; 3],
        IMPLEMENTATION_PATHS,
    );
    assert_mc_matches_qmc(
        "three-asset outperformance call",
        outperformance_result.price,
        outperformance_result.stderr,
        0.083_727_546_602_651_91,
        4.063_382_457_167_203_4e-8,
    );

    // Two-asset quanto average call: S=[100,98], weights=[.6,.4], K=100,
    // sigma=[20%,22%], rho12=.4, q_i=1%, T=1, FX=1.2, sigma_FX=12%,
    // rho_asset,FX=[.3,.25], rd=3%, rf=2%. Seeds 203000..203031.
    let quanto = QuantoBasketOption {
        basket: BasketOption {
            weights: vec![0.60, 0.40],
            strike: 100.0,
            maturity: 1.0,
            is_call: true,
            basket_type: BasketType::Average,
        },
        fx_rate: 1.20,
        fx_vol: 0.12,
        asset_fx_corr: vec![0.30, 0.25],
        domestic_rate: 0.03,
        foreign_rate: 0.02,
    };
    let quanto_result = price_quanto_basket_mc(
        &quanto,
        &[100.0, 98.0],
        &[0.20, 0.22],
        &[vec![1.0, 0.40], vec![0.40, 1.0]],
        &[0.01, 0.01],
        IMPLEMENTATION_PATHS,
    );
    assert_mc_matches_qmc(
        "two-asset quanto average call",
        quanto_result.price,
        quanto_result.stderr,
        7.820_441_594_319_476,
        1.553_014_744_327_432_6e-6,
    );
}

#[test]
fn student_t_copula_basket_matches_independent_scipy_sobol_reference() {
    // Independent oracle: SciPy 1.17.1 / NumPy 2.4.3, using a four-dimensional
    // Sobol sequence (three correlated Gaussian drivers plus the common
    // chi-square scale), SciPy's t CDF, and inverse-normal marginal mapping.
    // The cached price is the mean of 32 independently Owen-scrambled 2^19
    // point replicates (seeds 367000..367031); the quoted uncertainty is the
    // replicate standard deviation divided by sqrt(32).
    let basket = BasketOption {
        weights: vec![0.40, 0.35, 0.25],
        strike: 100.0,
        maturity: 1.25,
        is_call: true,
        basket_type: BasketType::Average,
    };
    let result = price_basket_mc_with_copula(
        &basket,
        &[100.0, 96.0, 104.0],
        &[0.20, 0.25, 0.18],
        &[
            vec![1.0, 0.45, 0.20],
            vec![0.45, 1.0, 0.30],
            vec![0.20, 0.30, 1.0],
        ],
        0.03,
        &[0.01, 0.02, 0.0],
        500_000,
        BasketCopula::StudentT {
            degrees_of_freedom: 5,
        },
    );
    assert_mc_matches_qmc(
        "three-asset Student-t copula average call",
        result.price,
        result.stderr,
        7.828_172_306_368_756_5,
        1.791_229_479_546_623e-5,
    );
}

#[test]
fn zero_vol_outperformance_and_quanto_baskets_equal_discounted_payoffs() {
    let spots = [100.0, 120.0];
    let vols = [0.0, 0.0];
    let dividends = [0.01, 0.03];
    let corr = [vec![1.0, 0.35], vec![0.35, 1.0]];
    let rate: f64 = 0.04;
    let maturity: f64 = 1.25;
    let terminal = [
        spots[0] * ((rate - dividends[0]) * maturity).exp(),
        spots[1] * ((rate - dividends[1]) * maturity).exp(),
    ];

    let outperformance = OutperformanceBasketOption {
        leader_index: 0,
        lagger_weights: vec![0.0, 1.0],
        strike: 0.75,
        maturity,
        option_type: OptionType::Call,
    };
    let outperformance_result =
        price_outperformance_basket_mc(&outperformance, &spots, &vols, &corr, rate, &dividends, 1);
    let outperformance_expected = (-rate * maturity).exp() * (terminal[0] / terminal[1] - 0.75);
    assert_close(
        "outperformance basket",
        outperformance_result.price,
        outperformance_expected,
        2.0e-14,
    );
    assert_eq!(outperformance_result.stderr, Some(0.0));

    let foreign_rate = 0.02;
    let domestic_rate = 0.03;
    let fx_rate = 1.25;
    let quanto_terminal = [
        spots[0] * ((foreign_rate - dividends[0]) * maturity).exp(),
        spots[1] * ((foreign_rate - dividends[1]) * maturity).exp(),
    ];
    let quanto = QuantoBasketOption {
        basket: BasketOption {
            weights: vec![0.4, 0.6],
            strike: 105.0,
            maturity,
            is_call: true,
            basket_type: BasketType::Average,
        },
        fx_rate,
        fx_vol: 0.15,
        asset_fx_corr: vec![0.3, -0.2],
        domestic_rate,
        foreign_rate,
    };
    let quanto_result = price_quanto_basket_mc(&quanto, &spots, &vols, &corr, &dividends, 1);
    let quanto_expected = (-domestic_rate * maturity).exp()
        * fx_rate
        * (0.4 * quanto_terminal[0] + 0.6 * quanto_terminal[1] - 105.0);
    assert_close(
        "quanto basket",
        quanto_result.price,
        quanto_expected,
        2.0e-14,
    );
    assert_eq!(quanto_result.stderr, Some(0.0));
}

#[test]
fn zero_vol_range_accruals_equal_discounted_contractual_coupon() {
    let fixing_times = vec![0.25, 0.5, 0.75, 1.0];
    let discount_rate = 0.03;
    let expected = 1_000_000.0 * 0.05 * (-discount_rate * 1.0_f64).exp();
    let single = RangeAccrual {
        notional: 1_000_000.0,
        coupon_rate: 0.05,
        accrual_factor: 1.0,
        lower_bound: 0.02,
        upper_bound: 0.06,
        fixing_times: fixing_times.clone(),
        payment_time: 1.0,
    };
    let single_result =
        range_accrual_mc_price(&single, 0.04, 1.0, 0.04, 0.0, discount_rate, 1, 42).unwrap();
    assert_close(
        "single range accrual",
        single_result.price,
        expected,
        1.0e-11,
    );
    assert_eq!(single_result.expected_accrual_fraction, 1.0);
    assert_eq!(single_result.std_error, 0.0);

    let dual = DualRangeAccrual {
        notional: 1_000_000.0,
        coupon_rate: 0.05,
        accrual_factor: 1.0,
        lower_bound: 0.01,
        upper_bound: 0.03,
        fixing_times,
        payment_time: 1.0,
    };
    let dual_result = dual_range_accrual_mc_price(
        &dual,
        0.05,
        0.03,
        1.0,
        0.05,
        0.0,
        1.0,
        0.03,
        0.0,
        -0.7,
        discount_rate,
        1,
        7,
    )
    .unwrap();
    assert_close("dual range accrual", dual_result.price, expected, 1.0e-11);
    assert_eq!(dual_result.expected_accrual_fraction, 1.0);
    assert_eq!(dual_result.std_error, 0.0);
}

#[test]
fn zero_vol_autocalls_equal_first_observation_cashflow() {
    let standard = Autocallable {
        underlyings: vec![0],
        notional: 1_000.0,
        autocall_dates: vec![0.25, 0.5, 1.0],
        autocall_barrier: 1.0,
        coupon_rate: 0.08,
        ki_barrier: 0.60,
        ki_strike: 1.0,
        maturity: 1.0,
    };
    let r = 0.03;
    let expected = 1_000.0 * (1.0 + 0.08 * 0.25) * (-r * 0.25_f64).exp();
    let result = price_autocallable(&standard, &[100.0], &[0.0], &[vec![1.0]], r, r, 1, 20);
    assert_close("standard autocall", result.price, expected, 2.0e-12);
    assert_eq!(result.stderr, Some(0.0));

    let phoenix = PhoenixAutocallable {
        underlyings: vec![0],
        notional: 1_000.0,
        autocall_dates: vec![0.25, 0.5, 1.0],
        autocall_barrier: 1.0,
        coupon_barrier: 0.80,
        coupon_rate: 0.08,
        memory: true,
        ki_barrier: 0.60,
        ki_strike: 1.0,
        maturity: 1.0,
    };
    let phoenix_result =
        price_phoenix_autocallable(&phoenix, &[100.0], &[0.0], &[vec![1.0]], r, r, 1, 20);
    assert_close("phoenix autocall", phoenix_result.price, expected, 2.0e-12);
    assert_eq!(phoenix_result.stderr, Some(0.0));
}

#[test]
fn deterministic_real_option_matches_discounted_investment_cashflow() {
    let option = DeferInvestmentOption {
        model: RealOptionBinomialSpec {
            project_value: 120.0,
            volatility: 0.0,
            risk_free_rate: 0.05,
            maturity: 2.0,
            steps: 8,
            cash_flows: vec![],
        },
        investment_cost: 100.0,
    };
    let expected = 120.0 - 100.0 * (-0.05_f64 * 2.0).exp();
    let actual = price_option_to_defer(&option).unwrap().price;
    assert_close("deterministic defer option", actual, expected, 3.0e-13);
}

#[test]
fn straight_convertible_tree_matches_its_discrete_coupon_annuity() {
    let face = 100.0;
    let coupon_rate = 0.06;
    let maturity = 2.0;
    let steps = 8;
    let rate = 0.04;
    let credit_spread = 0.015;
    let bond = ConvertibleBond::new(face, coupon_rate, maturity, 0.0, None, None);
    let actual = ConvertibleBinomialEngine::new(credit_spread)
        .with_steps(steps)
        .price(&bond, &market(100.0, rate, 0.01, 0.20))
        .unwrap()
        .price;

    let dt = maturity / steps as f64;
    let disc = (-(rate + credit_spread) * dt).exp();
    let coupon = face * coupon_rate * dt;
    let expected = face * disc.powi(steps as i32)
        + disc * coupon * (1.0 - disc.powi(steps as i32)) / (1.0 - disc);
    assert_close("straight convertible", actual, expected, 2.0e-12);
}

#[test]
fn forced_deep_itm_swing_matches_sum_of_discounted_forwards() {
    let exercise_dates = vec![0.25, 0.50, 0.75, 1.0];
    let spot = 200.0;
    let strike = 50.0;
    let quantity = 2.5;
    let rate = 0.04;
    let dividend_yield = 0.01;
    let swing = SwingOption::new(4, 4, exercise_dates.clone(), strike, quantity);
    let actual = SwingTreeEngine::new(4)
        .price(&swing, &market(spot, rate, dividend_yield, 0.10))
        .unwrap()
        .price;
    let expected = quantity
        * exercise_dates
            .iter()
            .map(|t| spot * (-dividend_yield * t).exp() - strike * (-rate * t).exp())
            .sum::<f64>();
    assert_close("forced deep-ITM swing", actual, expected, 2.0e-10);
}

#[test]
fn fixed_rate_note_hold_value_matches_discounted_cashflows() {
    let rate = 0.03;
    let curve = YieldCurve::new(
        (1..=8)
            .map(|i| {
                let t = i as f64 * 0.25;
                (t, (-rate * t).exp())
            })
            .collect(),
    );
    let coupons = CouponScheduleBuilder::new(0.0, 2.0, Frequency::SemiAnnual)
        .unwrap()
        .build_fixed(0.05)
        .unwrap();
    let note = CallableRateNote {
        notional: 1_000.0,
        redemption: 1_000.0,
        call_price: 1_000.0,
        maturity: 2.0,
        coupon_schedule: coupons,
        exercise_schedule: ExerciseSchedule::new(vec![1.0], 0.0).unwrap(),
    };
    let actual = note
        .hold_to_maturity_value(&curve, &[0.0; 4], &[0.0; 4])
        .unwrap();
    let expected = (1..=4)
        .map(|i| 1_000.0 * 0.05 * 0.5 * (-rate * (i as f64 * 0.5)).exp())
        .sum::<f64>()
        + 1_000.0 * (-rate * 2.0_f64).exp();
    assert_close("fixed note hold value", actual, expected, 2.0e-12);
}
