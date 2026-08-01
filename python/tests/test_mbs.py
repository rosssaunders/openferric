"""Exact references for rate-dependent MBS prepayment scenarios."""

import math

import pytest
from openferric import (
    ConstantCpr,
    MbsPassThrough,
    PrepaymentModel,
    PsaModel,
    RateIncentivePrepayment,
)


def make_mbs():
    return MbsPassThrough(
        original_balance=100.0,
        coupon_rate=0.06,
        servicing_fee=0.005,
        original_term=360,
        age=0,
        prepayment=PrepaymentModel("constant_cpr", ConstantCpr(0.0)),
    )


def test_rate_incentive_formula_matches_decimal_reference():
    model = RateIncentivePrepayment()
    assert model.refinancing_incentive(1.089, 1.0) == pytest.approx(0.2406, rel=0.0, abs=2e-16)
    assert model.cpr(30, 0.06, 0.045) == pytest.approx(0.34510460867313806, rel=0.0, abs=3e-16)
    assert model.smm(30, 0.06, 0.045) == pytest.approx(0.03465846083701067, rel=0.0, abs=3e-16)


def test_rate_scenario_matches_decimal_cashflow_price_and_duration():
    model = RateIncentivePrepayment(first_payment_month=1)
    mbs = make_mbs()

    cashflows = mbs.cashflows_with_refinancing_rates(model, [0.045])
    assert len(cashflows) == 360
    assert cashflows[0].prepayment == pytest.approx(0.0983799592970102, rel=0.0, abs=2e-14)
    assert cashflows[29].prepayment == pytest.approx(2.0104719506561097, rel=0.0, abs=3e-13)
    assert mbs.price_with_rate_scenario(model, [0.045], [0.05]) == pytest.approx(101.44972305453936, rel=0.0, abs=2e-12)
    assert mbs.wal_with_refinancing_rates(model, [0.045]) == pytest.approx(3.248112193095725, rel=0.0, abs=2e-13)
    assert mbs.effective_duration_with_rate_scenario(model, [0.045], [0.05], 0.0001) == pytest.approx(
        2.6886232461836676, rel=0.0, abs=3e-11
    )


def test_rate_scenario_rejects_ambiguous_curves():
    with pytest.raises(ValueError, match="first_payment_month"):
        RateIncentivePrepayment(first_payment_month=0)

    model = RateIncentivePrepayment()
    mbs = make_mbs()
    with pytest.raises(ValueError, match="one value per remaining payment"):
        mbs.cashflows_with_refinancing_rates(model, [0.04, 0.05])
    with pytest.raises(ValueError, match="must be > 0"):
        mbs.cashflows_with_refinancing_rates(model, [0.0])


def test_psa_speed_boundary_and_rejection():
    boundary = PsaModel(1.0 / 0.06)
    assert boundary.cpr(30) == pytest.approx(1.0, rel=0.0, abs=2e-16)
    assert boundary.smm(30) == pytest.approx(1.0, rel=0.0, abs=2e-16)
    boundary_mbs = MbsPassThrough(
        original_balance=100.0,
        coupon_rate=0.06,
        servicing_fee=0.005,
        original_term=360,
        age=0,
        prepayment=PrepaymentModel("psa", boundary),
    )
    assert math.isfinite(boundary_mbs.price(0.05))
    with pytest.raises(ValueError, match="psa_speed"):
        PsaModel(20.0)


def test_aged_pool_december_january_rollover_and_curve_indexing():
    model = RateIncentivePrepayment(first_payment_month=1)
    mbs = MbsPassThrough(
        original_balance=100.0,
        coupon_rate=0.06,
        servicing_fee=0.005,
        original_term=15,
        age=10,
        prepayment=PrepaymentModel("constant_cpr", ConstantCpr(0.0)),
    )
    refinancing_rates = [0.07, 0.06, 0.05, 0.04, 0.03]
    discount_yields = [0.03, 0.035, 0.04, 0.045, 0.05]

    cashflows = mbs.cashflows_with_refinancing_rates(model, refinancing_rates)
    assert [cashflow.prepayment for cashflow in cashflows[:4]] == pytest.approx(
        [
            0.3378140601145549,
            0.3465372645924251,
            0.4642691264846021,
            0.25369606760371995,
        ],
        rel=0.0,
        abs=3e-14,
    )
    assert cashflows[0].interest == pytest.approx(100.0 * (0.06 - 0.005) / 12.0, rel=0.0, abs=2e-16)
    assert mbs.price_with_rate_scenario(model, refinancing_rates, discount_yields) == pytest.approx(
        100.29149462440617, rel=0.0, abs=3e-14
    )


def test_tiny_coupon_annuity_denominator_is_stable():
    mbs = MbsPassThrough(
        original_balance=100.0,
        coupon_rate=1.2e-11,
        servicing_fee=0.0,
        original_term=2,
        age=0,
        prepayment=PrepaymentModel("constant_cpr", ConstantCpr(0.0)),
    )

    cashflows = mbs.cashflows()
    monthly_coupon = mbs.coupon_rate / 12.0
    assert cashflows[0].scheduled_principal == pytest.approx(
        mbs.original_balance / (2.0 + monthly_coupon), rel=0.0, abs=2e-14
    )
