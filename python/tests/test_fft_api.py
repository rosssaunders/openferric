"""Complex-number, characteristic-function, and transform interfaces."""

import cmath
import math

import openferric as of
import pytest


def test_fft_context_and_characteristic_function_martingale():
    characteristic = of.BlackScholesCharFn(100, 0.05, 0, 0.2, 1)
    assert characteristic(0j) == 1
    assert characteristic(-1j) == pytest.approx(100 * math.exp(0.05), abs=3e-13, rel=0)
    context = of.CarrMadanContext(characteristic, 0.05, 1, 100)
    price = context.price_strikes([100])[0][1]
    assert price == pytest.approx(10.450583572185565, abs=1e-7, rel=0)
    assert context.params().n == of.DEFAULT_FFT_N
    assert len(context.weighted_samples()) == of.DEFAULT_FFT_N
    assert (
        of.carr_madan_price_at_strikes_with_samples(context.weighted_samples(), [100], context.params())[0][1] == price
    )
    assert of.carr_madan_fft_strikes(characteristic, 0.05, 1, 100, [100])[0][1] == pytest.approx(
        price, abs=1e-12, rel=0
    )


def test_characteristic_function_callbacks_and_failures():
    def characteristic(argument):
        return cmath.exp(1j * argument * (math.log(100) + 0.03) - 0.02 * argument**2)

    price = of.CarrMadanContext(characteristic, 0.05, 1, 100).price_strikes([100])[0][1]
    assert price == pytest.approx(10.450583572185565, abs=1e-7, rel=0)

    def failing(_argument):
        raise LookupError("cf failure")

    with pytest.raises(LookupError, match="cf failure"):
        of.CarrMadanContext(failing, 0.05, 1, 100)
    with pytest.raises(ValueError, match="finite complex"):
        of.CarrMadanContext(lambda _argument: complex(math.nan, 0), 0.05, 1, 100)


def test_frft_agrees_with_direct_discrete_transform():
    values = [1 + 2j, 3 - 1j, -2 + 0.5j, 4 + 0j]
    beta = 0.17
    expected = [
        sum(value * cmath.exp(-2j * math.pi * beta * row * column / len(values)) for column, value in enumerate(values))
        for row in range(len(values))
    ]
    assert of.frft(values, beta) == pytest.approx(expected, abs=2e-13, rel=0)


@pytest.mark.parametrize("model", [of.VarianceGamma(0.2, -0.1, 0.2), of.Cgmy(0.2, 4, 6, 0.5), of.Nig(8, -2, 0.2)])
def test_levy_models_expose_risk_neutral_moments(model):
    model.validate()
    assert math.isfinite(model.martingale_correction())
    assert model.characteristic_fn(0j, 100, 0.05, 0.01, 1) == pytest.approx(1, abs=2e-14, rel=0)
    assert model.characteristic_fn(-1j, 100, 0.05, 0.01, 1) == pytest.approx(100 * math.exp(0.04), abs=4e-13, rel=0)
    prices = model.european_calls_fft(100, [90, 100, 110], 0.05, 0.01, 1)
    assert prices[0][1] >= prices[1][1] >= prices[2][1] >= 0
