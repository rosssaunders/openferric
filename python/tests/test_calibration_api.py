"""Native optimizer callbacks and full calibration result surfaces."""

import math

import openferric as of
import pytest


def test_native_optimizers_minimize_explicit_quadratics():
    bounds = of.BoxConstraints([-5, -5], [5, 5])

    def residual(values):
        return [values[0] - 1, values[1] + 2]

    def objective(values):
        return (values[0] - 1) ** 2 + (values[1] + 2) ** 2

    result = of.levenberg_marquardt([0, 0], bounds, residual)
    assert result.x == pytest.approx([1, -2], abs=1e-6, rel=0)
    assert result.objective == pytest.approx(sum(value**2 for value in result.residuals) / 2, abs=1e-20, rel=0)
    assert result.jacobian[0] == pytest.approx([1, 0], abs=1e-10, rel=0)
    simplex = of.nelder_mead([0, 0], bounds, objective, of.NelderMeadOptions(tolerance=1e-12))
    assert simplex.objective <= 1e-12
    population = of.differential_evolution(bounds, objective, of.DifferentialEvolutionOptions(seed=19))
    assert population.objective <= 1e-8


def test_optimizer_callback_errors_and_residual_shape_changes_propagate():
    bounds = of.BoxConstraints([-2], [2])

    def failing(_values):
        raise LookupError("objective failed")

    with pytest.raises(LookupError, match="objective failed"):
        of.nelder_mead([0], bounds, failing)
    with pytest.raises(LookupError, match="objective failed"):
        of.differential_evolution(bounds, failing)
    with pytest.raises(LookupError, match="objective failed"):
        of.levenberg_marquardt([0], bounds, failing)
    with pytest.raises(ValueError, match="same length"):
        of.levenberg_marquardt([0], bounds, lambda values: [1, 2] if values[0] == 0 else [1])


@pytest.mark.parametrize("parameterization", [of.SviParameterization.Raw, of.SviParameterization.JumpWings])
def test_svi_calibration_exposes_raw_and_jump_wings_results(parameterization):
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]
    quotes = []
    for strike in strikes:
        coordinate = math.log(strike / 100)
        variance = 0.02 + 0.15 * (-0.3 * coordinate + math.sqrt(coordinate**2 + 0.25**2))
        quotes.append(of.OptionVolQuote(str(strike), strike, 1, math.sqrt(variance)))
    result = of.SviCalibrator(100, 1, parameterization).calibrate(quotes)
    assert result.params_type == "svi"
    assert result.params.parameterization == parameterization
    assert result.svi_params().parameterization == parameterization
    assert result.heston_params() is None
    assert result.sabr_params() is None
    assert max(record.abs_error for record in result.per_instrument_error) < 1e-9
    if parameterization == of.SviParameterization.Raw:
        params = result.params.raw_params()
        for quote, record in zip(quotes, result.per_instrument_error):
            coordinate = math.log(quote.strike / 100) - params.m
            implied = math.sqrt(
                params.a + params.b * (params.rho * coordinate + math.sqrt(coordinate**2 + params.sigma**2))
            )
            assert record.model == pytest.approx(implied, abs=2e-15, rel=0)
    else:
        assert result.params.jump_wings_params().maturity == 1


def test_bid_ask_diagnostics_and_matrix_conditioning():
    quote = of.OptionVolQuote("quoted", 100, 1, 0.2)
    quote.bid_vol = 0.19
    quote.ask_vol = 0.21
    signed, effective, within = of.bid_ask_aware_error(quote, 0.205)
    assert signed == 0.205 - 0.2
    assert effective == 0
    assert within
    record = of.make_error_record(quote, 0.205)
    assert of.fit_quality([record]).liquid_rmse == abs(0.205 - 0.2)
    stability = of.parameter_stability(["alpha"], [1], [1.1], 0.2)
    assert stability.stable
    assert of.matrix_condition_number([[2, 0], [0, 1]]) == 2
    with pytest.raises(ValueError, match="equal lengths"):
        of.matrix_condition_number([[1], [2, 3]])


def test_sabr_calibrator_recovers_constant_lognormal_volatility():
    quotes = [of.OptionVolQuote(str(strike), strike, 1, 0.2) for strike in (70, 80, 90, 100, 110, 120, 130)]
    result = of.SabrCalibrator(100, 1, 1).calibrate(quotes)
    assert result.params_type == "sabr"
    assert result.params.beta == 1
    assert result.params.alpha == pytest.approx(0.2, abs=1e-5, rel=0)
    assert max(record.abs_error for record in result.per_instrument_error) < 1e-5
    assert result.svi_params() is None
    fit = of.fit_quality(result.per_instrument_error)
    bounds = of.BoxConstraints([1e-8, -0.999, 1e-8], [5, 0.999, 5])
    params = [result.params.alpha, result.params.rho, result.params.nu]
    assert (
        of.warning_flags(result.convergence, result.condition_number, fit, bounds, params)
        == result.diagnostics.warning_flags
    )


def test_diagnostic_enums_work_as_inputs_and_outputs():
    convergence = of.ConvergenceInfo(1, 1, 0, 0, True, of.TerminationReason.ObjectiveTolerance)
    assert of.sanitize_convergence(convergence).reason == of.TerminationReason.ObjectiveTolerance
    warnings = of.warning_flags(convergence, 1e12, of.FitQuality(0.01, 0.01, 0.01, 0.01))
    assert warnings == [of.CalibrationWarningFlag.IllConditioned, of.CalibrationWarningFlag.PoorFit]
