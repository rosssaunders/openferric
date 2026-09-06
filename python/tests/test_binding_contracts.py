"""Cross-module ownership, serialization, callback, and backend contracts."""

import importlib
import json
import math
from pathlib import Path

import numpy as np
import openferric as of
import pytest
from test_dsl import FORWARD


@pytest.mark.parametrize(
    "domain",
    [
        "core",
        "credit",
        "dsl",
        "engines",
        "engines.analytic",
        "engines.tree",
        "engines.pde",
        "engines.lsm",
        "engines.numerical",
        "engines.monte_carlo",
        "engines.fft",
        "fft",
        "funding",
        "instruments",
        "market",
        "math",
        "math.timeseries",
        "mc",
        "models",
        "calibration",
        "pricing",
        "rates",
        "risk",
        "greeks",
        "timeseries",
        "vol",
    ],
)
def test_domains_are_importable(domain):
    module = importlib.import_module(f"openferric.{domain}")
    assert any(not name.startswith("_") for name in dir(module))


def test_ast_tokens_records_and_compiled_fields_roundtrip():
    tokens = of.tokenize(FORWARD)
    assert isinstance(tokens[0], of.Token)
    assert tokens[0].kind == "Product"
    assert tokens[0].span == {"start": 0, "end": 7}
    reconstructed = [of.Token.from_dict(token.to_dict()) for token in tokens]
    ast = of.parse(reconstructed)
    assert of.ProductDef.from_dict(ast.to_dict()).compile().to_dict() == of.CompiledProduct(FORWARD).to_dict()
    assert ast.body
    product = ast.compile()
    assert product.underlyings[0].name == "SPX"
    assert product.underlyings[0].asset_index == 0
    assert product.schedules[0].dates == [0.25, 0.5, 0.75, 1]
    assert product.state_vars == []
    symbols = of.build_symbol_table(ast, FORWARD)
    assert symbols.declarations[0].name
    assert [value.to_dict() for value in symbols.declarations] == symbols.to_dict()["declarations"]
    assert [value.to_dict() for value in symbols.references] == symbols.to_dict()["references"]
    span = of.Span(start=1, end=4)
    assert span.start == 1
    assert of.Token(kind=of.TokenKind({"Number": 2.5}), span=span).kind == {"Number": 2.5}
    assert of.Value({"F64": 3.5}).as_f64() == 3.5
    assert of.Value({"Bool": True}).as_bool()
    assert of.ScheduleFreq("Quarterly").generate_dates(0.25, 1) == [0.25, 0.5, 0.75, 1]
    with pytest.raises(AttributeError):
        _ = span.unknown_field
    with pytest.raises(TypeError, match="not both"):
        of.Span({"start": 1, "end": 4}, start=2)
    cyclic = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="nest"):
        of.Span(cyclic)


TRADE_FIXTURES = json.loads((Path(__file__).parent / "data" / "trade_instruments.json").read_text())


@pytest.mark.parametrize("fixture", TRADE_FIXTURES, ids=lambda fixture: fixture["type"])
def test_every_trade_variant_roundtrips_native_serde(fixture):
    instrument = of.TradeInstrument(fixture["type"], fixture["data"])
    assert instrument.to_dict() == fixture
    assert of.TradeInstrument.from_dict(fixture).to_dict() == fixture
    trade = of.Trade(of.TradeMetadata("trade", 3, 0), instrument)
    portfolio = of.instruments.Portfolio("book", [trade], "market")
    restored = of.InstrumentPortfolio.from_dict(portfolio.to_dict())
    assert restored.trades[0].instrument.to_dict() == fixture
    assert restored.trades[0].metadata.version == 3
    assert restored.market_snapshot_id == "market"


@pytest.mark.parametrize(
    "instrument",
    [
        of.VanillaOption.european_call(100, 1),
        of.ChooserOption(100, 1, 0.5),
        of.VarianceSwap(100, 0.2, 1, [of.VarianceOptionQuote(90, 12, 2), of.VarianceOptionQuote(100, 8, 8)]),
        of.DslProduct.from_source(FORWARD),
    ],
)
def test_typed_trade_conversion(instrument):
    payload = of.TradeInstrument.from_instrument(instrument)
    assert of.Trade(of.TradeMetadata("native", 1, 0), instrument).to_dict()["instrument"] == payload.to_dict()
    with pytest.raises(ValueError, match="kind"):
        of.TradeInstrument("NotTheInstrumentType", instrument)


def test_batch_output_arrays_match_scalar_greeks_and_check_borrows():
    spots = np.array([80, 100, 120], dtype=float)
    strikes = np.full(3, 100.0)
    arguments = (spots, strikes, 0.05, 0.01, 0.2, 1.0, True)
    prices = np.empty(3)
    of.bs_price_batch_into(*arguments, prices)
    scalar = [of.bs_price(spot * math.exp(-0.01), 100, 1, 0.2, 0.05, "call") for spot in spots]
    assert prices == pytest.approx(scalar, abs=1e-7, rel=0)
    outputs = [np.empty(3) for _ in range(4)]
    of.bs_greeks_batch_into(*arguments, *outputs)
    for actual, expected in zip(outputs, of.bs_greeks_batch(*arguments)):
        assert actual == pytest.approx(expected, abs=1e-13, rel=0)
    of.bs_price_batch_into(*arguments, spots)
    assert spots == pytest.approx(prices, abs=1e-13, rel=0)
    with pytest.raises((ValueError, TypeError)):
        of.bs_price_batch_into(*arguments, np.empty(2))
    with pytest.raises((ValueError, TypeError)):
        of.bs_price_batch_into(*arguments, np.empty(6)[::2])
    readonly = np.zeros(3)
    readonly.flags.writeable = False
    with pytest.raises((ValueError, TypeError)):
        of.bs_price_batch_into(*arguments, readonly)
    with pytest.raises((ValueError, TypeError)):
        of.bs_greeks_batch_into(*arguments, prices, prices, prices, prices)
    cdf = np.empty(3)
    of.normal_cdf_batch_approx_into([-1, 0, 1], cdf)
    assert cdf == pytest.approx([0.15865525393145707, 0.5, 0.8413447460685429], abs=1e-7, rel=0)


def test_finite_difference_callbacks_and_aad_match_analytic_greeks():
    def pricer(spot, strike, rate, vol, expiry):
        return of.bs_price(spot, strike, expiry, vol, rate, "call")

    analytic = of.black_scholes_merton_greeks(of.OptionType.Call, 100, 100, 0.05, 0, 0.2, 1)
    finite = of.finite_difference_greeks(pricer, 100, 100, 0.05, 0.2, 1, 0.01, 0.0001, 0.0001, 0.0001)
    for name in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga"):
        assert getattr(finite, name) == pytest.approx(getattr(analytic, name), abs=0.001, rel=0)
    assert analytic.vomma() == analytic.volga

    def failing(*_arguments):
        raise LookupError("pricer failure")

    with pytest.raises(LookupError, match="pricer failure"):
        of.finite_difference_greeks(failing, 100, 100, 0.05, 0.2, 1, 0.01, 0.0001, 0.0001, 0.0001)
    with pytest.raises(ValueError, match="finite"):
        of.bump_and_reprice(lambda *_arguments: math.nan, 100, 100, 0.05, 0.2, 1, 0.01, 0.0001, 0.0001, 0.0001)
    market = of.Market.builder().spot(100).rate(0.05).flat_vol(0.2).build()
    result = of.BlackScholesEngine().price_with_greeks_aad(of.VanillaOption.european_call(100, 1), market)
    assert result.price == pytest.approx(pricer(100, 100, 0.05, 0.2, 1), abs=1e-12, rel=0)
    assert result.greeks.delta == pytest.approx(analytic.delta, abs=1e-12, rel=0)


def test_arena_rng_interpolator_and_sobol_ownership():
    arena = of.PricingArena(256, 64)
    assert arena.path_slice(4) == [0] * 4
    copied = arena.path_buffer
    copied[0] = 2
    assert arena.path_buffer[0] == 0
    arena.path_buffer = copied
    assert arena.path_slice(2) == [2, 0]
    for generator_type in (of.Pcg64, of.Xoshiro256PlusPlus):
        first = generator_type(42)
        second = generator_type.seed_from_u64(42)
        assert [first.next_u64() for _ in range(20)] == [second.next_u64() for _ in range(20)]
    assert of.math.Pcg64Rng is of.Pcg64
    first, second = of.FastRng("pcg", 42), of.FastRng("pcg", 42)
    assert of.math.fill_standard_normals(first, 8) == second.fill_standard_normals(8)
    linear = of.LinearInterpolator([0, 1, 2], [1, 3, 5], of.ExtrapolationMode.flat())
    generic = of.AnyInterpolator(linear)
    assert generic.value(0.5) == 2
    assert generic.derivative(0.5) == 2
    assert generic.jacobian(0.5) == [0.5, 0.5, 0]
    assert generic.x() == [0, 1, 2]
    sequence = of.SobolSequence(2)
    point = np.empty(2)
    assert sequence.next_into(point)
    assert point.tolist() == pytest.approx([0.5, 0.5], abs=2e-16, rel=0)
    with pytest.raises(ValueError):
        sequence.next_into(np.empty(1))
    readonly = np.zeros(2)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        sequence.next_into(readonly)


def test_custom_payoff_policy_and_model_readback():
    generator = of.GbmPathGenerator(0, 0, 100, 1, 4)
    assert generator.model.mu == 0

    def evaluator(path):
        return path[-1]

    control = of.ControlVariate(100, evaluator)
    assert control.evaluator is evaluator
    engine = of.MonteCarloEngine(256, 17).with_control_variate(control)
    assert engine.control_variate.expected == 100
    assert engine.execution_policy == of.CpuExecutionPolicy.Scalar
    assert engine.run(generator, evaluator, 1) == (100, 0)
    assert engine.run_fallible(generator, evaluator, 1) == (100, 0)
    if of.build_features()["parallel"]:
        assert engine.with_execution_policy(of.CpuExecutionPolicy.Parallel).run(generator, evaluator, 1) == (100, 0)
    else:
        with pytest.raises(ValueError, match="parallel"):
            engine.with_execution_policy(of.CpuExecutionPolicy.Parallel)


def test_python_vol_surface_sampling_preserves_errors_and_snapshot():
    parametric = of.VolSurface([(1, of.SviParams(0.02, 0.1, -0.3, 0, 0.2))], 105)
    source = of.VolSource.from_surface(parametric, 100)
    assert source.parametric_spec() == ([(1, (0.02, 0.1, -0.3, 0, 0.2))], 105)
    assert source.vol(100, 1) == parametric.vol(100, 1)
    surface = of.SampledVolSurface.from_surface(lambda _strike, _expiry: 0.2, 100)
    assert surface.vol(100, 1) == pytest.approx(0.2, abs=1e-15, rel=0)
    market = of.Market.builder().spot(100).rate(0.05).vol_surface(surface).build()
    assert market.vol_for(100, 1) == pytest.approx(0.2, abs=1e-15, rel=0)

    def failing(_strike, _expiry):
        raise LookupError("surface failure")

    with pytest.raises(LookupError, match="surface failure"):
        of.Market.builder().spot(100).vol_surface(failing)
    with pytest.raises(ValueError, match="finite"):
        of.SampledVolSurface.from_surface(lambda _strike, _expiry: math.nan, 100)


def test_slv_uses_full_market_and_validates_instrument_dispatch():
    market = of.Market.builder().spot(100).rate(0.05).vol_surface(lambda _strike, _expiry: 0.2).build()
    params = of.SlvParams(0.04, 1, 0.04, 0, 0)
    surface = of.calibrate_leverage_surface(market, params, 1, 256, 4)
    assert surface.value(100, 0.5) == pytest.approx(1, abs=1e-5, rel=0)
    option = of.VanillaOption.european_call(100, 1)
    result = of.slv_mc_price_checked(option, market, params, 2048, 4)
    assert abs(result.price - 10.450583572185565) < 5 * result.stderr
    assert of.slv_mc_price(option, market, params, 2048, 4).price == result.price
    with pytest.raises(TypeError, match="unsupported SLV"):
        of.slv_mc_price_checked(object(), market, params, 256, 4)
    with pytest.raises(ValueError):
        of.slv_mc_price_checked(option, market, of.SlvParams(-1, 1, 0.04, 0, 0), 256, 4)


def test_engine_configuration_fields_and_feature_contracts():
    assert of.__version__ == "0.1.0"
    model = of.Heston(0.05, 2, 0.04, 0.2, -0.5, 0.04)
    engine = of.AdiHestonEngine(model, 100, 80, 40).with_scheme(of.AdiScheme.CraigSneyd)
    assert engine.model.v0 == 0.04
    assert engine.scheme == of.AdiScheme.CraigSneyd
    assert (engine.time_steps, engine.spot_steps, engine.variance_steps) == (100, 80, 40)
    mc = of.MonteCarloPricingEngine(256, 4, 7)
    assert mc.accuracy_tier is None
    assert mc.with_accuracy_tier(of.AccuracyTier.High).accuracy_tier == of.AccuracyTier.High
    assert mc.with_execution_policy(of.ExecutionPolicy.Scalar).execution_policy == of.ExecutionPolicy.Scalar
    assert (
        mc.with_variance_reduction(of.VarianceReduction.Antithetic).variance_reduction
        == of.VarianceReduction.Antithetic
    )
    greeks = of.MonteCarloGreeksEngine(256, 7).with_antithetic(True).with_spot_bump_rel(0.02)
    assert (greeks.num_paths, greeks.seed, greeks.antithetic, greeks.spot_bump_rel) == (256, 7, True, 0.02)
    dsl = of.DslMonteCarloEngine(256, 4, 7)
    assert (dsl.num_paths, dsl.num_steps, dsl.seed) == (256, 4, 7)
    features = of.build_features()
    assert hasattr(of, "JitProductEvaluator") == features["jit"]
    assert hasattr(of, "mc_european_parallel") == features["parallel"]
    assert hasattr(of, "mc_european_gpu") == features["gpu"]
    if features["gpu"]:
        assert isinstance(of.gpu_is_ready(), bool)
    for feature, policy in (("parallel", of.ExecutionPolicy.Parallel), ("jit", of.ExecutionPolicy.Jit)):
        if not features[feature]:
            with pytest.raises(ValueError):
                dsl.resolve_execution_backend(policy)


def test_heston_aad_prices_constant_variance_limit():
    price, delta = of.heston_price_delta_aad(
        of.OptionType.Call,
        100,
        1,
        100,
        0.05,
        of.Heston(0.05, 2, 0.04, 0, 0, 0.04),
        of.HestonAadConfig(8192, 8, 42),
    )
    assert price == pytest.approx(10.450583572185565, abs=0.65, rel=0)
    assert delta == pytest.approx(0.6368306511756191, abs=0.025, rel=0)


@pytest.mark.parametrize(
    "convention",
    [
        of.FxDeltaConvention.Spot,
        of.FxDeltaConvention.Forward,
        of.FxDeltaConvention.PremiumAdjustedSpot,
        of.FxDeltaConvention.PremiumAdjustedForward,
    ],
)
@pytest.mark.parametrize("currency", [of.PremiumCurrency.Domestic, of.PremiumCurrency.Foreign])
@pytest.mark.parametrize("option_type, target", [(of.OptionType.Call, 0.25), (of.OptionType.Put, -0.25)])
def test_fx_delta_conventions_roundtrip_strikes(convention, currency, option_type, target):
    strike = of.market.strike_from_delta(1.2, 0.04, 0.02, 0.15, 1, target, convention, currency)
    assert of.market.fx_delta(option_type, 1.2, strike, 0.04, 0.02, 0.15, 1, convention, currency) == pytest.approx(
        target, abs=1e-10, rel=0
    )
    assert of.market.fx_delta is of.market_fx_delta
    assert of.market.fx_delta is not of.risk.fx_delta
