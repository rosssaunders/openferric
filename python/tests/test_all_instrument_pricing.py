"""Independent cashflow checks for the full-instrument pricing audit."""

import math

import openferric as of
import pytest


def test_fra_separates_accrual_and_discount_curve_clocks():
    curve = of.YieldCurve([(2.0, math.exp(-0.06))])
    parameters = dict(
        notional=100.0,
        fixed_rate=0.02,
        start_date="2026-07-01",
        end_date="2027-01-01",
        day_count=of.DayCountConvention.act360(),
        valuation_date="2026-01-01",
    )
    for clock, denominator in [(None, 365.0), (of.DayCountConvention.act360(), 360.0)]:
        contract = of.ForwardRateAgreement(**parameters, curve_day_count=clock)
        accrual = 184.0 / 360.0
        start_discount = math.exp(-0.03 * 181.0 / denominator)
        end_discount = math.exp(-0.03 * 365.0 / denominator)
        expected_forward = math.expm1(0.03 * 184.0 / denominator) / accrual
        expected_price = 100.0 * (start_discount - end_discount * (1.0 + 0.02 * accrual))
        assert contract.forward_rate(curve) == pytest.approx(expected_forward, rel=0.0, abs=5e-16)
        assert contract.npv(curve) == pytest.approx(expected_price, rel=0.0, abs=4e-14)
        assert repr(contract.curve_day_count) == repr(clock or of.DayCountConvention.act365_fixed())


def test_swap_builder_exposes_curve_clock_independently_of_both_legs():
    curve = of.YieldCurve([(2.0, math.exp(-0.06))])
    builder = (
        of.InterestRateSwap.builder()
        .notional(100.0)
        .fixed_rate(0.0)
        .start_date("2025-01-02")
        .end_date("2026-01-02")
        .fixed_day_count(of.DayCountConvention.thirty360())
        .float_day_count(of.DayCountConvention.act360())
    )
    for clock, denominator in [(of.DayCountConvention.act365_fixed(), 365.0), (of.DayCountConvention.act360(), 360.0)]:
        swap = builder.curve_day_count(clock).build()
        expected_floating = 100.0 * -math.expm1(-0.03 * 365.0 / denominator)
        assert swap.float_leg_pv(curve) == pytest.approx(expected_floating, rel=0.0, abs=5e-14)
        assert repr(swap.curve_day_count) == repr(clock)


@pytest.mark.parametrize("instrument_class", [of.RangeAccrual, of.DualRangeAccrual])
def test_range_coupon_has_explicit_accrual_separate_from_payment_lag(instrument_class):
    contract = instrument_class(
        notional=100.0,
        coupon_rate=0.08,
        lower_bound=0.01,
        upper_bound=0.06,
        fixing_times=[0.25],
        payment_time=0.5,
        accrual_factor=0.25,
    )
    if instrument_class is of.RangeAccrual:
        result = of.py_range_accrual_mc_price(contract, 0.04, 0.1, 0.04, 0.0, 0.03, 1, 42)
    else:
        result = of.py_dual_range_accrual_mc_price(
            contract, 0.05, 0.03, 0.1, 0.05, 0.0, 0.1, 0.03, 0.0, 0.5, 0.03, 1, 42
        )
    assert result.price == pytest.approx(2.0 * math.exp(-0.015), rel=0.0, abs=2e-14)
    assert contract.accrual_factor == 0.25


@pytest.mark.parametrize("instrument_class", [of.RangeAccrual, of.DualRangeAccrual])
@pytest.mark.parametrize("accrual_factor", [0.0, -0.25, math.nan, math.inf])
def test_range_accrual_rejects_invalid_periods(instrument_class, accrual_factor):
    with pytest.raises(ValueError):
        instrument_class(100.0, 0.08, 0.01, 0.06, [0.25], 0.5, accrual_factor)
