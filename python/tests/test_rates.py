"""Tests for rates functions: swaption pricing."""

import pytest
from conftest import is_nan
from openferric import py_swaption_price

# =========================================================================
# 21. py_swaption_price
# =========================================================================


class TestSwaptionPrice:
    @pytest.fixture
    def swaption_params(self):
        return dict(notional=1_000_000.0, strike=0.03, swap_tenor=5.0, option_expiry=1.0, vol=0.15, discount_rate=0.03)

    def test_payer_positive(self, swaption_params):
        price = py_swaption_price(**swaption_params, option_type="payer")
        assert price > 0.0

    def test_receiver_positive(self, swaption_params):
        price = py_swaption_price(**swaption_params, option_type="receiver")
        assert price > 0.0

    def test_call_alias(self, swaption_params):
        """'call' should be same as 'payer'."""
        payer = py_swaption_price(**swaption_params, option_type="payer")
        call = py_swaption_price(**swaption_params, option_type="call")
        assert payer == pytest.approx(call, rel=1e-10)

    def test_put_alias(self, swaption_params):
        """'put' should be same as 'receiver'."""
        receiver = py_swaption_price(**swaption_params, option_type="receiver")
        put = py_swaption_price(**swaption_params, option_type="put")
        assert receiver == pytest.approx(put, rel=1e-10)

    def test_higher_vol_higher_price(self, swaption_params):
        low_vol = py_swaption_price(**{**swaption_params, "vol": 0.10}, option_type="payer")
        high_vol = py_swaption_price(**{**swaption_params, "vol": 0.30}, option_type="payer")
        assert high_vol > low_vol

    def test_invalid_option_type(self, swaption_params):
        assert is_nan(py_swaption_price(**swaption_params, option_type="straddle"))


# =========================================================================
# ForwardRateAgreement: forward-start NPV matches the curve identity
# =========================================================================


class TestForwardRateAgreement:
    def test_forward_start_npv_matches_curve_identity(self):
        """NPV == notional * (fwd - fixed) * tau * df(t2), with the simple
        forward over [t1, t2] and discounting from t2."""
        import math

        from openferric import DayCountConvention, ForwardRateAgreement, YieldCurve

        r = 0.03
        # Flat continuously-compounded curve: log-linear interpolation on
        # discount factors reproduces exp(-r * t) exactly between nodes.
        nodes = [(t, math.exp(-r * t)) for t in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
        curve = YieldCurve(nodes)

        notional = 1_000_000.0
        fixed = 0.025
        fra = ForwardRateAgreement(
            notional=notional,
            fixed_rate=fixed,
            start_date="2026-07-01",
            end_date="2027-01-01",
            day_count=DayCountConvention.act365_fixed(),
            valuation_date="2026-01-01",
        )

        # Same Act/365F year fractions the instrument uses.
        t1 = 181.0 / 365.0  # 2026-01-01 -> 2026-07-01
        tau = 184.0 / 365.0  # 2026-07-01 -> 2027-01-01
        t2 = t1 + tau

        df1 = math.exp(-r * t1)
        df2 = math.exp(-r * t2)
        fwd = (df1 / df2 - 1.0) / tau
        expected = notional * (fwd - fixed) * tau * df2

        assert fra.forward_rate(curve) == pytest.approx(fwd, rel=1e-10)
        assert fra.npv(curve) == pytest.approx(expected, rel=1e-10)
        assert "valuation_date" in repr(fra)

    def test_omitted_valuation_date_anchors_at_start(self):
        """Without valuation_date the FRA is spot-started: t1 == 0."""
        import math

        from openferric import DayCountConvention, ForwardRateAgreement, YieldCurve

        r = 0.03
        nodes = [(t, math.exp(-r * t)) for t in (0.25, 0.5, 1.0, 1.5, 2.0)]
        curve = YieldCurve(nodes)

        fra = ForwardRateAgreement(
            notional=100.0,
            fixed_rate=0.02,
            start_date="2026-07-01",
            end_date="2027-01-01",
            day_count=DayCountConvention.act365_fixed(),
        )
        # Valuation defaults to start_date, so the accrual period starts at
        # time zero on the curve.
        tau = 184.0 / 365.0
        df = math.exp(-r * tau)
        fwd = (1.0 / df - 1.0) / tau
        expected = 100.0 * (fwd - 0.02) * tau * df
        assert fra.npv(curve) == pytest.approx(expected, rel=1e-10)
