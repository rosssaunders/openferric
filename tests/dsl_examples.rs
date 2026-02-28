//! Integration tests that parse and price every DSL example file.
//!
//! Ensures all examples in `examples/dsl/` are valid DSL that compiles
//! and produces a finite price under standard market conditions.

use openferric::dsl::{parse_and_compile, AssetData, DslMonteCarloEngine, MultiAssetMarket};

fn single_asset_market() -> MultiAssetMarket {
    MultiAssetMarket::single(100.0, 0.20, 0.05, 0.02)
}

fn three_asset_market() -> MultiAssetMarket {
    MultiAssetMarket {
        assets: vec![
            AssetData {
                spot: 100.0,
                vol: 0.20,
                dividend_yield: 0.02,
            },
            AssetData {
                spot: 100.0,
                vol: 0.22,
                dividend_yield: 0.03,
            },
            AssetData {
                spot: 100.0,
                vol: 0.25,
                dividend_yield: 0.01,
            },
        ],
        correlation: vec![
            vec![1.0, 0.7, 0.5],
            vec![0.7, 1.0, 0.6],
            vec![0.5, 0.6, 1.0],
        ],
        rate: 0.03,
    }
}

fn two_asset_market() -> MultiAssetMarket {
    MultiAssetMarket {
        assets: vec![
            AssetData {
                spot: 100.0,
                vol: 0.20,
                dividend_yield: 0.02,
            },
            AssetData {
                spot: 100.0,
                vol: 0.22,
                dividend_yield: 0.03,
            },
        ],
        correlation: vec![vec![1.0, 0.7], vec![0.7, 1.0]],
        rate: 0.03,
    }
}

fn engine() -> DslMonteCarloEngine {
    DslMonteCarloEngine::new(20_000, 100, 42)
}

/// Helper: load, parse, compile, price a DSL file and assert the price is finite.
fn test_example(filename: &str, market: &MultiAssetMarket) {
    let path = format!("examples/dsl/{filename}");
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let product =
        parse_and_compile(&source).unwrap_or_else(|e| panic!("compile {filename}: {e}"));
    let result = engine()
        .price_multi_asset(&product, market)
        .unwrap_or_else(|e| panic!("price {filename}: {e}"));
    assert!(
        result.price.is_finite(),
        "{filename}: price is not finite ({})",
        result.price
    );
    eprintln!("{filename}: {:<40} price = {:>14.2}", product.name, result.price);
}

#[test]
fn example_01_zero_coupon() {
    test_example("01_zero_coupon.of", &single_asset_market());
}

#[test]
fn example_02_fixed_coupon_note() {
    test_example("02_fixed_coupon_note.of", &single_asset_market());
}

#[test]
fn example_03_forward() {
    test_example("03_forward.of", &single_asset_market());
}

#[test]
fn example_04_reverse_convertible() {
    test_example("04_reverse_convertible.of", &single_asset_market());
}

#[test]
fn example_05_barrier_reverse_convertible() {
    test_example("05_barrier_reverse_convertible.of", &single_asset_market());
}

#[test]
fn example_06_autocallable_worst_of() {
    test_example("06_autocallable_worst_of.of", &three_asset_market());
}

#[test]
fn example_07_step_down_autocallable() {
    test_example("07_step_down_autocallable.of", &two_asset_market());
}

#[test]
fn example_08_phoenix_memory_coupon() {
    test_example("08_phoenix_memory_coupon.of", &single_asset_market());
}

#[test]
fn example_09_athena_autocallable() {
    test_example("09_athena_autocallable.of", &two_asset_market());
}

#[test]
fn example_10_snowball_note() {
    test_example("10_snowball_note.of", &single_asset_market());
}

#[test]
fn example_11_digital_coupon_note() {
    test_example("11_digital_coupon_note.of", &single_asset_market());
}

#[test]
fn example_12_twin_win() {
    test_example("12_twin_win.of", &single_asset_market());
}

#[test]
fn example_13_capital_protected_note() {
    test_example("13_capital_protected_note.of", &single_asset_market());
}

#[test]
fn example_14_bonus_certificate() {
    test_example("14_bonus_certificate.of", &single_asset_market());
}

#[test]
fn example_15_cliquet() {
    test_example("15_cliquet.of", &single_asset_market());
}

#[test]
fn example_16_accumulator_tarf() {
    test_example("16_accumulator_tarf.of", &single_asset_market());
}

#[test]
fn example_17_range_accrual() {
    test_example("17_range_accrual.of", &single_asset_market());
}

#[test]
fn example_18_shark_fin() {
    test_example("18_shark_fin.of", &single_asset_market());
}

#[test]
fn example_19_outperformance_certificate() {
    test_example("19_outperformance_certificate.of", &single_asset_market());
}

#[test]
fn example_20_wedding_cake() {
    test_example("20_wedding_cake.of", &three_asset_market());
}

#[test]
fn example_21_best_of_call() {
    test_example("21_best_of_call.of", &three_asset_market());
}

#[test]
fn example_22_worst_of_put() {
    test_example("22_worst_of_put.of", &two_asset_market());
}

#[test]
fn example_23_express_certificate() {
    test_example("23_express_certificate.of", &single_asset_market());
}

#[test]
fn example_24_lookback_coupon() {
    test_example("24_lookback_coupon.of", &single_asset_market());
}

#[test]
fn example_25_callable_note() {
    test_example("25_callable_note.of", &single_asset_market());
}

#[test]
fn example_26_asian_payout() {
    test_example("26_asian_payout.of", &single_asset_market());
}

#[test]
fn example_27_airbag_certificate() {
    test_example("27_airbag_certificate.of", &single_asset_market());
}

#[test]
fn example_28_double_barrier_ki_ko() {
    test_example("28_double_barrier_ki_ko.of", &single_asset_market());
}

#[test]
fn example_29_dispersion_note() {
    test_example("29_dispersion_note.of", &three_asset_market());
}

#[test]
fn example_30_napoleon() {
    test_example("30_napoleon.of", &two_asset_market());
}
