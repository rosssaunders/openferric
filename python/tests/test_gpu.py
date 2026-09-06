"""GPU payoff regression locks; reference provenance is in docs/PERFORMANCE.md."""

import math

import openferric as of
import pytest


@pytest.fixture(scope="module")
def gpu():
    if not of.build_features()["gpu"]:
        pytest.skip("GPU feature is not compiled")
    try:
        of.engines.gpu.prewarm_gpu()
    except RuntimeError as error:
        if str(error).startswith("No GPU adapter found:"):
            pytest.skip(str(error))
        raise
    return of.engines.gpu


@pytest.mark.parametrize(
    "spot,strike,rate,is_call,expected",
    [
        (100.0, 100.0, 0.05, True, 4.877057549928599),
        (100.0, 100.0, -0.05, False, 5.127109637602405),
        (100.000001, 100.0, 0.0, True, 9.999999974752427e-7),
        (100.0, 100.000001, 0.0, False, 9.999999974752427e-7),
    ],
)
def test_gpu_deterministic_payoffs(gpu, spot, strike, rate, is_call, expected):
    result = gpu.mc_european_gpu(spot, strike, rate, 0.0, 1.0, 513, 1, 42, is_call)
    assert result.price == pytest.approx(expected, rel=2**-23, abs=0)
    assert result.stderr == 0.0


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize(
    "vol,expected",
    [(1e-9, 3.989422804014327e-8), (1e-6, 3.98942280401416e-5), (0.2, 7.965567455405796)],
)
def test_gpu_atm_small_volatility(gpu, vol, expected, is_call):
    result = gpu.mc_european_gpu(100.0, 100.0, 0.0, vol, 1.0, 131_072, 1, 42, is_call)
    assert math.isfinite(result.stderr) and result.stderr > 0
    assert abs(result.price - expected) <= 4 * result.stderr + 8 * 2**-23 * expected
