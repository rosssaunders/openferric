"""Exercise the native DSL pipeline and its independently known cashflows."""

import math
import platform

import openferric as of
import pytest

FORWARD = """product "Forward"
    notional: 100
    maturity: 1
    underlyings
        SPX = asset(0)
    schedule quarterly from 0.25 to 1
        pay 2
        if is_final then
            redeem notional
"""


def expected_price(rate=0.05):
    return sum(2 * math.exp(-rate * time) for time in (0.25, 0.5, 0.75, 1)) + 100 * math.exp(-rate)


def test_compile_serialize_parse_and_evaluate_cashflows():
    product = of.parse_and_compile(FORWARD)
    assert product.name == "Forward"
    assert product.notional == 100
    assert product.num_underlyings == 1
    assert of.ProductDef(FORWARD).compile().to_dict() == product.to_dict()
    assert of.compile(of.ProductDef(FORWARD)).to_dict() == product.to_dict()
    for restored in (of.CompiledProduct.from_json(product.to_json()), of.CompiledProduct.from_dict(product.to_dict())):
        restored.validate()
        path = [[100], [110], [105], [120], [115]]
        evaluator = of.ProductEvaluator(restored, 4, 0.05)
        for _ in range(2):
            assert evaluator.evaluate(path, [100]) == pytest.approx(expected_price(), abs=3e-14, rel=0)
        assert of.evaluate_product(restored, path, [100], 4, 0.05) == pytest.approx(expected_price(), abs=3e-14, rel=0)


@pytest.mark.parametrize("policy", [of.ExecutionPolicy.Scalar, of.ExecutionPolicy.Parallel, of.ExecutionPolicy.Auto])
def test_mc_backends_price_the_same_deterministic_contract(policy):
    if policy == of.ExecutionPolicy.Parallel and not of.build_features()["parallel"]:
        pytest.skip("built without parallel support")
    product = of.CompiledProduct(FORWARD)
    market = of.MultiAssetMarket.single(100, 0.2, 0.05)
    engine = of.DslMonteCarloEngine(256, 4, 42)
    price = engine.price_multi_asset(product, market, policy)
    assert price.price == pytest.approx(expected_price(), abs=3e-13, rel=0)
    assert price.stderr == 0
    backend = engine.resolve_execution_backend(policy)
    assert of.ExecutionBackend.from_diagnostic_code(backend.diagnostic_code()) == backend
    assert of.ExecutionBackend.from_diagnostic_code(math.nan) is None
    with pytest.raises(ValueError, match="GPU"):
        engine.price_multi_asset(product, market, of.ExecutionPolicy.Gpu)


def test_jit_path_evaluator_and_reusable_scratch():
    if not hasattr(of, "JitProductEvaluator"):
        pytest.skip("built without JIT support")
    product = of.CompiledProduct(FORWARD)
    if platform.machine().lower() not in ("x86_64", "amd64"):
        with pytest.raises(ValueError, match="unsupported"):
            of.JitProductEvaluator(product, 4, 0.05)
        return
    evaluator = of.JitProductEvaluator(product, 4, 0.05)
    scratch = evaluator.new_scratch()
    path = [[100.0]] * 5
    assert evaluator.evaluate_path(path, [100]) == pytest.approx(expected_price(), abs=3e-14, rel=0)
    assert evaluator.evaluate_path_with_scratch(path, [100], scratch) == pytest.approx(
        expected_price(), abs=3e-14, rel=0
    )
    with pytest.raises(ValueError):
        evaluator.evaluate_path(path[:-1], [100])


def test_invalid_compiled_products_paths_and_markets_raise_python_errors():
    with pytest.raises(ValueError):
        of.parse_and_compile("not a product")
    product = of.CompiledProduct(FORWARD)
    data = product.to_dict()
    data["maturity"] = -1
    with pytest.raises(ValueError):
        of.CompiledProduct.from_dict(data)
    with pytest.raises(ValueError):
        of.ProductEvaluator(product, 4, 0.05).evaluate([[100]], [100])
    with pytest.raises(ValueError):
        of.MultiAssetMarket.single(math.nan, 0.2, 0.05)
    with pytest.raises(ValueError):
        of.DslMonteCarloEngine(0, 4, 42)


def test_analysis_diagnostics_and_utf8_boundaries():
    ast, product, diagnostics = of.parse_and_diagnose(FORWARD)
    assert product is not None
    assert not diagnostics
    symbols = of.SymbolTable(ast, FORWARD)
    assert symbols.to_dict()["declarations"]
    assert of.completions(FORWARD, symbols, len(FORWARD))
    assert of.semantic_token_data(FORWARD, symbols)
    _, failed, diagnostics = of.parse_and_diagnose("invalid source")
    assert failed is None
    assert diagnostics[0]["severity"] == "Error"
    assert of.offset_to_line_col("é\nx", 3) == (1, 0)
    assert of.line_col_to_offset("é\nx", 0, 1) == 0
    with pytest.raises(ValueError, match="UTF-8"):
        of.offset_to_line_col("é", 1)


def test_all_multi_asset_market_variants_roundtrip():
    assets = [
        of.AssetMarketData.equity(100, 0.2, 0.01),
        of.AssetMarketData.fx(1.2, 0.1, 0.03, 0.02),
        of.AssetMarketData.commodity(80, 0.3, 0.02, 0.5, math.log(80)),
        of.AssetMarketData.rate(-0.01, 0.01, 0.1, 0.02),
    ]
    correlation = [[float(row == column) for column in range(4)] for row in range(4)]
    market = of.MultiAssetMarket(assets, correlation, 0.03)
    assert market.initial_spots() == [100, 1.2, 80, -0.01]
    assert of.MultiAssetMarket.from_dict(market.to_dict()).to_dict() == market.to_dict()
    for asset in assets:
        assert asset.with_spot_bump(1).initial_value() == pytest.approx(asset.initial_value() + 1)
        assert asset.with_vol_bump(0.01).vol() == pytest.approx(asset.vol() + 0.01)
