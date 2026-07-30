"""Monte Carlo binding coverage for native and custom-payoff APIs."""

import math

import pytest
from conftest import assert_releases_gil
from openferric import ControlVariate, GbmPathGenerator, HestonPathGenerator, McEngine, MonteCarloEngine


def vanilla_price(**overrides):
    parameters = dict(
        option_type="call",
        spot=100.0,
        strike=100.0,
        expiry=1.0,
        rate=0.05,
        dividend_yield=0.0,
        vol=0.2,
        num_paths=100_000,
        num_steps=64,
        seed=42,
        variance_reduction="antithetic",
        rng_kind="xoshiro",
        reproducible=True,
        accuracy_tier="high",
        exercise_style="european",
        bermudan_dates=None,
    )
    parameters.update(overrides)
    return McEngine.vanilla_price(**parameters)


def test_vanilla_price_uses_exact_terminal_native_engine():
    result = vanilla_price()

    assert result.price == pytest.approx(10.4506, abs=max(4.0 * result.stderr, 0.15))
    assert result.stderr > 0.0
    assert result.diagnostics.get("num_steps") == 1.0


def test_gbm_custom_payoff_preserves_arbitrary_drift():
    generator = GbmPathGenerator(0.3, 0.2, 100.0, 0.25, 8)
    engine = MonteCarloEngine(2_000, 7)

    price, stderr = engine.run_gbm(generator, lambda path: path[-1], 1.0)

    assert price == pytest.approx(100.0 * math.exp(0.3 * 0.25), abs=4.0 * stderr)
    assert stderr > 0.0


def test_heston_custom_payoff_remains_supported():
    generator = HestonPathGenerator(
        mu=0.07,
        kappa=1.5,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        v0=0.04,
        s0=100.0,
        maturity=0.25,
        steps=8,
    )
    engine = MonteCarloEngine(2_000, 11)

    price, stderr = engine.run_heston(generator, lambda path: path[-1], 1.0)

    assert math.isfinite(price)
    assert price > 0.0
    assert stderr > 0.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_paths": 0}, "num_paths"),
        ({"num_steps": 0}, "num_steps"),
        ({"option_type": "invalid"}, "option_type"),
    ],
)
def test_vanilla_price_validates_inputs(override, message):
    with pytest.raises(ValueError, match=message):
        vanilla_price(**override)


def test_custom_payoff_validates_callback():
    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    engine = MonteCarloEngine(100, 7)

    with pytest.raises(ValueError, match="payoff must be callable"):
        engine.run_gbm(generator, 42, 1.0)


def test_custom_engine_rejects_zero_paths():
    with pytest.raises(ValueError, match="num_paths"):
        MonteCarloEngine(0, 7)


@pytest.mark.parametrize("discount_factor", [math.nan, math.inf, -math.inf, -0.1])
def test_custom_payoff_rejects_invalid_discount_factor(discount_factor):
    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    engine = MonteCarloEngine(100, 7)

    with pytest.raises(ValueError, match="discount_factor"):
        engine.run_gbm(generator, lambda path: path[-1], discount_factor)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_custom_payoff_rejects_non_finite_callback_values(value):
    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    engine = MonteCarloEngine(10_000, 7)
    calls = 0

    def payoff(_path):
        nonlocal calls
        calls += 1
        return value

    with pytest.raises(ValueError, match="payoff callback must return a finite"):
        engine.run_gbm(generator, payoff, 1.0)
    assert calls == 1


@pytest.mark.parametrize("expected", [math.nan, math.inf, -math.inf])
def test_control_variate_rejects_non_finite_expected_value(expected):
    with pytest.raises(ValueError, match="expected value must be finite"):
        ControlVariate(expected, lambda path: path[-1])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_control_variate_rejects_non_finite_callback_values(value):
    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    control = ControlVariate(100.0, lambda _path: value)
    engine = MonteCarloEngine(10_000, 7).with_control_variate(control)

    with pytest.raises(ValueError, match="control variate callback must return a finite"):
        engine.run_gbm(generator, lambda path: path[-1], 1.0)


def test_gbm_callback_failure_stops_immediately_and_preserves_exception():
    class PayoffFailureError(RuntimeError):
        pass

    calls = 0

    def failing_payoff(path):
        nonlocal calls
        calls += 1
        raise PayoffFailureError(f"failed at terminal value {path[-1]}")

    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    engine = MonteCarloEngine(10_000, 7)

    with pytest.raises(PayoffFailureError, match="failed at terminal value"):
        engine.run_gbm(generator, failing_payoff, 1.0)
    assert calls == 1


def test_heston_callback_failure_stops_immediately_and_preserves_exception():
    calls = 0

    def failing_payoff(_path):
        nonlocal calls
        calls += 1
        raise LookupError("heston payoff failed")

    generator = HestonPathGenerator(
        mu=0.07,
        kappa=1.5,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        v0=0.04,
        s0=100.0,
        maturity=0.25,
        steps=8,
    )
    engine = MonteCarloEngine(10_000, 11)

    with pytest.raises(LookupError, match="heston payoff failed"):
        engine.run_heston(generator, failing_payoff, 1.0)
    assert calls == 1


def test_control_variate_failure_stops_immediately_and_preserves_exception():
    control_calls = 0
    payoff_calls = 0

    def payoff(path):
        nonlocal payoff_calls
        payoff_calls += 1
        return path[-1]

    def failing_control(_path):
        nonlocal control_calls
        control_calls += 1
        raise ArithmeticError("control variate failed")

    generator = GbmPathGenerator(0.0, 0.2, 100.0, 0.25, 8)
    control = ControlVariate(100.0, failing_control)
    engine = MonteCarloEngine(10_000, 7).with_control_variate(control)

    with pytest.raises(ArithmeticError, match="control variate failed"):
        engine.run_gbm(generator, payoff, 1.0)
    assert payoff_calls == 1
    assert control_calls == 1


def test_pure_rust_mc_entry_points_release_gil():
    operations = {
        "arithmetic Asian": lambda: McEngine.arithmetic_asian_price(
            option_type="call",
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            vol=0.2,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            paths=200_000,
            steps=32,
            seed=42,
            control_variate=True,
            rng_kind="xoshiro",
            reproducible=True,
        ),
        "pathwise Greeks": lambda: McEngine.greeks_pathwise(
            option_type="call",
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            vol=0.2,
            num_paths=400_000,
            seed=42,
            antithetic=True,
            spot_bump_rel=0.01,
            rng_kind="xoshiro",
            reproducible=True,
        ),
        "likelihood-ratio Greeks": lambda: McEngine.greeks_likelihood_ratio(
            option_type="call",
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.0,
            vol=0.2,
            num_paths=400_000,
            seed=42,
            antithetic=True,
            spot_bump_rel=0.01,
            rng_kind="xoshiro",
            reproducible=True,
        ),
        "spread": lambda: McEngine.spread_price(
            s1=100.0,
            s2=90.0,
            k=5.0,
            vol1=0.2,
            vol2=0.25,
            rho=0.5,
            q1=0.0,
            q2=0.0,
            rate=0.05,
            expiry=1.0,
            num_paths=400_000,
            seed=42,
            antithetic=True,
            rng_kind="xoshiro",
            reproducible=True,
        ),
    }

    for name, operation in operations.items():
        result = assert_releases_gil(operation)
        assert result is not None, name


def spread_price(**overrides):
    parameters = dict(
        s1=100.0,
        s2=90.0,
        k=5.0,
        vol1=0.2,
        vol2=0.25,
        rho=0.5,
        q1=0.0,
        q2=0.0,
        rate=0.05,
        expiry=1.0,
        num_paths=4_096,
        seed=42,
        antithetic=True,
        rng_kind="xoshiro",
        reproducible=True,
    )
    parameters.update(overrides)
    return McEngine.spread_price(**parameters)


@pytest.mark.parametrize("field", ["s1", "s2", "k", "vol1", "vol2", "q1", "q2", "rate", "expiry"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_spread_price_rejects_non_finite_inputs(field, bad_value):
    with pytest.raises(ValueError):
        spread_price(**{field: bad_value})


@pytest.mark.parametrize(
    "override",
    [
        {"s1": 0.0},
        {"s2": -1.0},
        {"k": -1.0},
        {"vol1": -0.1},
        {"vol2": -0.1},
        {"rho": math.nan},
        {"rho": -1.01},
        {"rho": 1.01},
        {"expiry": -0.1},
        {"num_paths": 0},
    ],
)
def test_spread_price_rejects_out_of_domain_inputs(override):
    with pytest.raises(ValueError):
        spread_price(**override)


def test_spread_price_accepts_zero_volatility_and_expiry():
    deterministic = spread_price(vol1=0.0, vol2=0.0)
    assert math.isfinite(deterministic.price)
    assert deterministic.stderr == 0.0

    expired = spread_price(expiry=0.0)
    assert expired.price == pytest.approx(5.0)
    assert expired.stderr == 0.0
