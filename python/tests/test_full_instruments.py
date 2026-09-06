"""Typed contracts tested against independent prices and cashflow identities."""

import math

import openferric as of
import pytest


def market(volatility=0.2):
    return of.Market.builder().spot(100).rate(0.05).dividend_yield(0).flat_vol(volatility).build()


def test_domain_imports_and_native_vanilla_engines():
    from openferric.engines import BlackScholesEngine
    from openferric.math.timeseries import simple_returns

    assert BlackScholesEngine is of.BlackScholesEngine
    assert simple_returns([100, 110]) == pytest.approx([0.1], abs=2e-16, rel=0)
    option = of.VanillaOption.european_call(100, 1)
    reference = 10.450583572185565
    assert BlackScholesEngine().price(option, market()).price == pytest.approx(reference, abs=1e-12, rel=0)
    for engine in (of.BinomialTreeEngine(400), of.TrinomialTreeEngine(400), of.GeneralizedBinomialEngine(400, 0.05)):
        assert engine.price(option, market()).price == pytest.approx(reference, abs=0.015, rel=0)
    with pytest.raises(TypeError, match="does not support"):
        BlackScholesEngine().price(object(), market())


@pytest.mark.parametrize(
    "engine_type", [of.CrankNicolsonEngine, of.ImplicitFdEngine, of.ExplicitFdEngine, of.HopscotchEngine]
)
def test_pde_engines_price_contracts(engine_type):
    option = of.VanillaOption.european_put(100, 1)
    price = engine_type(1000, 150).price(option, market()).price
    assert price == pytest.approx(5.573526022256971, abs=0.08, rel=0)


def test_exotics_are_validated_contracts_not_opaque_payloads():
    chooser = of.ChooserOption(100, 1, 0)
    engine = of.ExoticAnalyticEngine()
    expected = of.BlackScholesEngine().price(of.VanillaOption.european_call(100, 1), market()).price
    assert engine.price(chooser, market()).price == pytest.approx(expected, abs=1e-12, rel=0)
    wrapped = of.ExoticOption("Chooser", chooser)
    assert wrapped.kind == "Chooser"
    assert wrapped.payload["choose_time"] == 0
    assert engine.price(wrapped, market()).price == engine.price(chooser, market()).price
    with pytest.raises(ValueError):
        of.ChooserOption(100, 1, 2)
    with pytest.raises(ValueError):
        of.ExoticOption("Unknown", {})


def test_coupon_builders_and_noncallable_rate_notes():
    curve = of.YieldCurve([(0, 1), (1, math.exp(-0.03))])
    schedule = of.CouponScheduleBuilder(0, 1, of.Frequency.quarterly()).build_floating()
    note = of.InverseFloaterNote(100, 100, 0.08, 2, None, None, schedule)
    expected = sum(0.5 * math.exp(-0.03 * time) for time in (0.25, 0.5, 0.75, 1)) + 100 * math.exp(-0.03)
    assert note.price([0.03] * 4, curve) == pytest.approx(expected, abs=2e-12, rel=0)
    assert of.InverseFloaterNote.from_dict(note.to_dict()).price([0.03] * 4, curve) == note.price([0.03] * 4, curve)
    fixed = of.CouponScheduleBuilder(0, 1, of.Frequency.quarterly()).payment_lag(0.02).build_fixed(0.04)
    assert fixed[0].payment_time == 0.27
    assert fixed[0].coupon.kind == "Fixed"
    with pytest.raises(ValueError):
        of.ExerciseSchedule([0.5, 0.5], 0)


def test_tarn_and_snowball_expose_cashflow_results():
    curve = of.YieldCurve([(0, 1), (1, 1)])
    schedule = of.CouponScheduleBuilder(0, 1, of.Frequency.quarterly()).build_floating()
    result = of.TargetRedemptionNote(100, 100, 3, 0, None, None, schedule).price([0.08] * 4, curve)
    assert result.knocked_out
    assert result.knockout_time == 0.5
    assert result.accrued_coupon == 3
    assert result.price == 103
    snowball = of.SnowballNote(100, 100, 0.08, 0, 0, None, schedule).price([0] * 4, curve)
    assert len(snowball.coupon_path) == 4
    assert snowball.price == pytest.approx(100 + 25 * sum(snowball.coupon_path), abs=2e-14, rel=0)


def test_callable_note_tree_and_hold_value():
    curve = of.YieldCurve([(0, 1), (1, math.exp(-0.03))])
    schedule = of.CouponScheduleBuilder(0, 1, of.Frequency.quarterly()).build_fixed(0.04)
    note = of.CallableRateNote(100, 100, 1e12, 1, of.ExerciseSchedule([1], 0), schedule)
    expected = sum(math.exp(-0.03 * time) for time in (0.25, 0.5, 0.75, 1)) + 100 * math.exp(-0.03)
    assert note.hold_to_maturity_value(curve, [0] * 4, [0] * 4) == pytest.approx(expected, abs=2e-12, rel=0)
    assert note.price_hull_white_tree(of.HullWhite(0.1, 0.01), curve, 480) == pytest.approx(expected, abs=1e-5, rel=0)
    assert note.price_hull_white_tree(of.HullWhite(0.1, 0), curve, 120) == pytest.approx(expected, abs=2e-12, rel=0)


def test_monte_carlo_reproducibility_qmc_aad_and_owned_paths():
    option = of.VanillaOption.european_call(100, 1)
    engine = of.MonteCarloPricingEngine(8192, 8, 123).with_execution_policy(of.ExecutionPolicy.Scalar)
    first = engine.price(option, market())
    assert first.price == engine.price(option, market()).price
    assert abs(first.price - 10.450583572185565) < 5 * first.stderr
    result = of.mc_european_qmc_with_seed(option, market(), 4096, 8, 42)
    assert abs(result.price - 10.450583572185565) < 0.1
    assert result.stderr > 0
    aad = of.mc_european_pathwise_aad(engine, option, market())
    assert aad.greeks is not None
    assert abs(aad.greeks.delta - 0.6368306511756191) < 0.03
    paths = of.simulate_gbm_paths_soa_scalar(100, 0.05, 0, 0, 1, 8, 4, 42)
    assert paths.num_paths == 8
    assert len(paths.levels) == 5
    assert paths.terminal() == pytest.approx([100 * math.exp(0.05)] * 8, abs=2e-12, rel=0)


def test_hjm_factors_drift_and_deterministic_discounting():
    factor = of.HjmFactor(of.HjmFactorShape.Parallel, 0.01, 0.2)
    assert factor.sigma(2) == pytest.approx(0.01 * math.exp(-0.4), abs=1e-17, rel=0)
    assert factor.integrated_sigma(2) == pytest.approx(0.01 * -math.expm1(-0.4) / 0.2, abs=1e-17, rel=0)
    model = of.HjmModel([factor], [[1]])
    assert model.drift(0, 2) == pytest.approx(factor.sigma(2) * factor.integrated_sigma(2), abs=1e-18, rel=0)
    assert of.HjmModel.zero_coupon_bond_price(0, 2, [0, 1, 2], [0.03] * 3) == pytest.approx(
        math.exp(-0.06), abs=2e-15, rel=0
    )
    with pytest.raises(IndexError):
        model.factor_volatility(1, 0, 2)
