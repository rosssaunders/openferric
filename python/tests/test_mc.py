"""Monte Carlo binding coverage for native and custom-payoff APIs."""

import math

import pytest
from openferric import GbmPathGenerator, HestonPathGenerator, McEngine, MonteCarloEngine


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
