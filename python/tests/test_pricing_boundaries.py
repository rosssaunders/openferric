"""Binding regressions for the September 2026 pricing-boundary audit."""

import math

import pytest
from openferric import AnalyticEngine, CapFloor, CdsOption, McEngine


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [("call", 13.096840337494951), ("put", 2.416146705239445)],
)
def test_asian_fixing_before_payment_matches_scipy(option_type, expected):
    parameters = dict(
        option_type=option_type,
        spot=100.0,
        strike=95.0,
        expiry=1.5,
        rate=0.2,
        dividend_yield=0.02,
        vol=0.3,
        observation_times=[0.5],
    )
    analytic = AnalyticEngine.geometric_asian_engine_price(**parameters)
    simulated = McEngine.arithmetic_asian_price(
        **parameters,
        paths=1024,
        steps=6,
        seed=42,
        control_variate=True,
        rng_kind="xoshiro",
        reproducible=True,
    )
    assert analytic.price == pytest.approx(expected, rel=0.0, abs=2e-12)
    assert simulated.price == pytest.approx(expected, rel=0.0, abs=2e-11)


@pytest.mark.parametrize(("option_type", "forward", "sign"), [("call", 120.0, 1.0), ("put", 80.0, -1.0)])
def test_black76_zero_volatility_greeks(option_type, forward, sign):
    greeks = AnalyticEngine.black76_greeks(option_type, forward, 100.0, 0.05, 0.0, 1.5)
    discount = math.exp(-0.05 * 1.5)
    price = 20.0 * discount
    assert greeks.delta == pytest.approx(sign * discount, rel=0.0, abs=1e-14)
    assert greeks.gamma == 0.0
    assert greeks.vega == 0.0
    assert greeks.theta == pytest.approx(0.05 * price, rel=0.0, abs=1e-13)
    assert greeks.rho == pytest.approx(-1.5 * price, rel=0.0, abs=1e-13)


def test_black76_zero_volatility_atm_does_not_report_finite_delta():
    greeks = AnalyticEngine.black76_greeks("call", 100.0, 100.0, 0.05, 0.0, 1.5)
    assert math.isnan(greeks.delta)
    assert math.isnan(greeks.gamma)
    assert greeks.vega == pytest.approx(math.exp(-0.075) * 100.0 * math.sqrt(1.5 / math.tau), rel=0.0, abs=1e-13)


def test_zero_strike_caplet_and_invalid_lognormal_forward():
    assert CapFloor.black_caplet(1_000_000.0, 0.95, 0.25, 0.04, 0.0, 0.3, 1.0) == pytest.approx(
        9500.0, rel=0.0, abs=1e-10
    )
    assert CapFloor.black_floorlet(1_000_000.0, 0.95, 0.25, 0.0, 0.04, 0.3, 1.0) == pytest.approx(
        9500.0, rel=0.0, abs=1e-10
    )
    assert math.isnan(CapFloor.black_floorlet(1_000_000.0, 0.95, 0.25, -0.01, 0.04, 0.3, 1.0))


def test_cds_option_at_expiry_pays_intrinsic():
    option = CdsOption(1_000_000.0, 0.02, 0.0, 5.0, True, 0.4)
    assert option.black_price(0.03, 0.3, 4.0) == pytest.approx(40_000.0, rel=0.0, abs=1e-10)
