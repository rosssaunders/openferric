"""Numerical utilities are checked against elementary analytic identities."""

import math

import openferric as of
import pytest


def test_normal_functions_and_quadrature():
    assert of.normal_pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-16, rel=0)
    assert of.normal_cdf(0) == 0.5
    assert of.normal_inv_cdf(0.5) == pytest.approx(0, abs=1e-15, rel=0)
    assert of.bivariate_normal_cdf(0, 0, 0) == 0.25
    assert of.erfc_cody(0) == 1
    assert of.gamma(5) == pytest.approx(24, abs=2e-13, rel=0)
    nodes, weights = of.gauss_legendre_nodes_weights(8)
    assert sum(weights) == pytest.approx(2, abs=2e-14, rel=0)
    assert sum(weight * node**6 for node, weight in zip(nodes, weights)) == pytest.approx(2 / 7, abs=2e-14, rel=0)
    assert of.gauss_legendre_integrate(lambda value: value**6, -1, 1, 8) == pytest.approx(2 / 7, abs=2e-14, rel=0)
    assert of.newton_raphson(lambda value: value**2 - 2, lambda value: 2 * value, 1) == pytest.approx(
        math.sqrt(2), abs=1e-12, rel=0
    )


def test_callback_exceptions_are_not_replaced_with_numerical_results():
    def failing_callback(_value):
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        of.gauss_legendre_integrate(failing_callback, 0, 1, 8)
    with pytest.raises(RuntimeError, match="callback failure"):
        of.newton_raphson(failing_callback, lambda value: value, 1)
    with pytest.raises(ValueError, match="non-finite"):
        of.gauss_legendre_integrate(lambda _value: math.nan, 0, 1, 8)


def test_sobol_iteration_and_owned_batches():
    sequence = of.SobolSequence(2)
    points = sequence.fill_points(4)
    for actual, expected in zip(points, [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.375, 0.375]]):
        assert actual == pytest.approx(expected, abs=2e-16, rel=0)
    reference = of.SobolSequence(2)
    assert [next(reference) for _ in range(4)] == points
    assert sequence.dimensions() == 2
    assert of.SobolSequence(2).fill_points(0) == []
    with pytest.raises(ValueError):
        of.SobolSequence(of.SOBOL_MAX_DIMENSIONS + 1)


def test_interpolation_values_derivatives_and_accuracy_policy():
    spline = of.CubicSpline([0, 1, 2], [1, 3, 5])
    assert spline.interpolate(0.5) == pytest.approx(2, abs=1e-15, rel=0)
    step = of.PiecewiseConstantInterpolator([0, 1, 2], [1, 3, 5], "error")
    assert step.value(0.5) == 1
    assert step.value(1) == 3
    assert step.derivative(0.5) == 0
    assert step.jacobian(0.5) == [1, 0, 0]
    with pytest.raises(ValueError):
        step.value(-1)
    assert of.AccuracyTier.for_mc(100, 10) == of.AccuracyTier.High
    assert of.AccuracyTier.for_analytic() == of.AccuracyTier.High
    assert of.tiered_exp(1, of.AccuracyTier.High) == math.exp(1)


def test_timeseries_moments_and_normal_fit():
    assert of.simple_returns([100, 110, 99]) == pytest.approx([0.1, -0.1], abs=2e-16, rel=0)
    assert of.log_returns([1, math.e, math.e**2]) == pytest.approx([1, 1], abs=2e-16, rel=0)
    assert of.rolling_mean([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
    fitted = of.fit_normal_distribution([-1, 0, 1])
    variance = 2 / 3
    log_likelihood = -1.5 * (math.log(2 * math.pi * variance) + 1)
    assert fitted.mean == 0
    assert fitted.std_dev == pytest.approx(math.sqrt(variance), abs=1e-15, rel=0)
    assert fitted.log_likelihood == pytest.approx(log_likelihood, abs=2e-15, rel=0)
    assert fitted.aic == pytest.approx(4 - 2 * log_likelihood, abs=3e-15, rel=0)
    assert of.sample_correlation_matrix([[-1, 0, 1, 0], [0, -1, 0, 1]]) == [[1, 0], [0, 1]]
    shrunk = of.ledoit_wolf_correlation_matrix([[-1, 0, 1, 0], [0, -1, 0, 1]])
    assert shrunk.correlation == [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        of.simple_returns([100])
    with pytest.raises(ValueError):
        of.fit_normal_distribution([math.nan])


def test_var_backtesting_outputs_are_accessible():
    losses = [0, 2, 0, 0, 0, 2, 0, 0, 0, 0]
    forecasts = [1] * len(losses)
    result = of.backtest_var(losses, forecasts, 0.8)
    assert result.kupiec.exceptions == 2
    assert result.exception_rate == 0.2
    assert result.kupiec.lr_statistic == pytest.approx(0, abs=2e-14, rel=0)
    assert of.var_breach_indicators(losses, forecasts) == [value > 1 for value in losses]
    assert of.kupiec_test(losses, forecasts, 0.8).p_value == pytest.approx(1, abs=1e-14, rel=0)
    assert of.christoffersen_test(losses, forecasts, 0.8).n11 == 0
