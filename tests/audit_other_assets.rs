use approx::assert_relative_eq;
use openferric::instruments::mbs::{IoStrip, PoStrip};
use openferric::instruments::{
    CatastropheBond, CommodityFutures, ConstantCpr, MbsPassThrough, PrepaymentModel,
};

fn short_mortgage() -> MbsPassThrough {
    MbsPassThrough {
        original_balance: 100.0,
        coupon_rate: 0.0,
        servicing_fee: 0.0,
        original_term: 2,
        age: 0,
        prepayment: PrepaymentModel::ConstantCpr(ConstantCpr { annual_cpr: 0.0 }),
    }
}

#[test]
fn catastrophe_bond_prorates_final_coupon_including_subperiod_tenors() {
    for maturity in [0.1, 1.1] {
        let bond = CatastropheBond {
            principal: 100.0,
            coupon_rate: 0.06,
            maturity,
            coupon_frequency: 2,
            risk_free_rate: 0.03,
            loss_intensity: 0.2,
            expected_loss_per_event: 0.4,
        };
        let mut expected = 100.0 * (-0.11 * maturity).exp();
        let mut previous = 0.0;
        for payment in [0.5_f64, 1.0, 1.1] {
            let end = payment.min(maturity);
            expected += 6.0 * (end - previous) * (-0.11 * end).exp();
            previous = end;
            if end == maturity {
                break;
            }
        }
        assert_relative_eq!(bond.price().unwrap(), expected, epsilon = 3.0e-13);
    }
}

#[test]
fn mbs_spread_uses_every_discount_curve_point() {
    let mortgage = short_mortgage();
    for spread in [-0.025_f64, 0.0, 0.015, 0.1] {
        let market_price =
            50.0 / (1.0 + (0.01 + spread) / 12.0) + 50.0 / (1.0 + (0.06 + spread) / 12.0).powi(2);
        assert_relative_eq!(
            mortgage.oas(market_price, &[0.01, 0.06]),
            spread,
            epsilon = 3.0e-12
        );
    }
    assert!(mortgage.oas(-1.0, &[0.01]).is_nan());
    assert!(mortgage.oas(f64::NAN, &[0.01]).is_nan());
    assert!(mortgage.oas(100.0, &[0.01, 0.02, 0.03]).is_nan());
}

#[test]
fn mortgage_and_strips_reject_nonpositive_monthly_discount_bases() {
    let mortgage = short_mortgage();
    for yield_rate in [-24.0, -12.0, f64::NAN, f64::INFINITY] {
        assert!(mortgage.price(yield_rate).is_nan());
        assert!(IoStrip { mbs: &mortgage }.price(yield_rate).is_nan());
        assert!(PoStrip { mbs: &mortgage }.price(yield_rate).is_nan());
    }
    let expected = 50.0 / 1.005 + 50.0 / 1.005_f64.powi(2);
    assert_relative_eq!(mortgage.price(0.06), expected, epsilon = 3.0e-13);
    assert_eq!(IoStrip { mbs: &mortgage }.price(0.06), 0.0);
    assert_relative_eq!(
        PoStrip { mbs: &mortgage }.price(0.06),
        expected,
        epsilon = 3.0e-13
    );
}

#[test]
fn linear_commodity_futures_allow_zero_and_negative_settlements() {
    let future = CommodityFutures {
        contract_price: 20.0,
        contract_size: 1000.0,
        is_long: true,
    };
    for settlement in [0.0, -37.63] {
        assert_relative_eq!(
            future.value(settlement).unwrap(),
            1000.0 * (settlement - 20.0),
            epsilon = 1.0e-10
        );
    }
    let short = CommodityFutures {
        contract_price: -37.63,
        is_long: false,
        ..future
    };
    assert_relative_eq!(short.value(10.0).unwrap(), -47_630.0, epsilon = 1.0e-10);
}

#[test]
fn storage_lsm_does_not_round_away_initial_inventory() {
    use openferric::models::commodity::{
        CommodityForwardCurve, CommodityStorageContract, FuturesQuote, StorageLsmConfig,
        value_storage_intrinsic_extrinsic,
    };
    let contract = CommodityStorageContract {
        decision_times: vec![1.0],
        min_inventory: 0.0,
        max_inventory: 200.0,
        initial_inventory: 150.0,
        max_injection: 1.0,
        max_withdrawal: 1.0,
        variable_cost: 0.0,
        terminal_inventory_target: None,
    };
    let curve = CommodityForwardCurve::from_futures_quotes(&[FuturesQuote {
        maturity: 1.0,
        price: 100.0,
    }])
    .unwrap();
    let result = value_storage_intrinsic_extrinsic(
        &contract,
        &curve,
        0.03,
        3,
        StorageLsmConfig {
            num_paths: 8,
            kappa: 1.0,
            sigma: 0.0,
            seed: 42,
        },
    )
    .unwrap();
    let expected = 15_000.0 * (-0.03_f64).exp();
    assert_relative_eq!(result.intrinsic, expected, epsilon = 3.0e-10);
    assert_relative_eq!(result.total, expected, epsilon = 3.0e-10);
    assert_relative_eq!(result.extrinsic, 0.0, epsilon = 3.0e-10);
}

#[test]
fn volume_swing_reports_grid_infeasibility_instead_of_infinite_price() {
    use openferric::core::OptionType;
    use openferric::models::commodity::{
        CommodityForwardCurve, FuturesQuote, VolumeConstrainedSwing,
    };
    let contract = VolumeConstrainedSwing {
        exercise_times: vec![1.0, 2.0],
        strike: 90.0,
        option_type: OptionType::Call,
        min_period_volume: 0.3,
        max_period_volume: 0.3,
        min_total_volume: 0.6,
        max_total_volume: 1.0,
    };
    let curve = CommodityForwardCurve::from_futures_quotes(&[FuturesQuote {
        maturity: 2.0,
        price: 100.0,
    }])
    .unwrap();
    assert!(contract.validate().is_ok());
    assert!(contract.intrinsic_value(&curve, 0.0, 3).is_err());
    assert_relative_eq!(
        contract.intrinsic_value(&curve, 0.0, 11).unwrap(),
        6.0,
        epsilon = 1.0e-12
    );
}
