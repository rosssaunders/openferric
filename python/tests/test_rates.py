"""Tests for rates functions: swaption pricing."""

import math

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
        # Independent SciPy 1.17.1 Black-76 evaluation on OpenFerric's
        # documented annual fixed-leg schedule. The 256-ULP allowance covers
        # curve construction, Black-76 operations, and the independent CDF
        # implementations; it is not an economic price range.
        reference = 9070.129921696538
        assert price == pytest.approx(reference, rel=0.0, abs=256 * math.ulp(reference))

    def test_receiver_positive(self, swaption_params):
        price = py_swaption_price(**swaption_params, option_type="receiver")
        # Same SciPy 1.17.1 Black-76 fixture and binary64-only budget as payer.
        reference = 7052.63808672894
        assert price == pytest.approx(reference, rel=0.0, abs=256 * math.ulp(reference))

    def test_call_alias(self, swaption_params):
        """'call' should be same as 'payer'."""
        payer = py_swaption_price(**swaption_params, option_type="payer")
        call = py_swaption_price(**swaption_params, option_type="call")
        assert payer == call

    def test_put_alias(self, swaption_params):
        """'put' should be same as 'receiver'."""
        receiver = py_swaption_price(**swaption_params, option_type="receiver")
        put = py_swaption_price(**swaption_params, option_type="put")
        assert receiver == put

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

        assert fra.forward_rate(curve) == pytest.approx(fwd, rel=0.0, abs=32 * math.ulp(fwd))
        assert fra.npv(curve) == pytest.approx(expected, rel=0.0, abs=32 * math.ulp(expected))
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
        assert fra.npv(curve) == pytest.approx(expected, rel=0.0, abs=32 * math.ulp(expected))


class TestXccySwapFrequencies:
    def test_quarterly_dual_curve_matches_decimal_reference(self):
        import math

        from openferric import Frequency, XccySwap, YieldCurve

        def curve(rate):
            return YieldCurve([(float(t), math.exp(-rate * t)) for t in range(1, 6)])

        ccy1_discount = curve(0.04)
        ccy2_discount = curve(0.03)
        ccy2_projection = curve(0.035)
        quarterly = Frequency.quarterly()
        swap = XccySwap(100_000_000.0, 90_000_000.0, 0.03, 0.0025, 5.0, 1.1111)

        fixed = swap.fixed_leg_pv_ccy1_with_frequency(ccy1_discount, quarterly)
        floating = swap.float_leg_pv_ccy2_with_frequency(ccy2_discount, ccy2_projection, quarterly)
        npv = swap.npv_dual_curve_with_frequencies(
            ccy1_discount,
            ccy2_discount,
            ccy2_projection,
            quarterly,
            quarterly,
            True,
        )

        roundoff = 8 * math.ulp(max(fixed, floating))
        assert fixed == pytest.approx(95_400_406.1524443, rel=0.0, abs=roundoff)
        assert floating == pytest.approx(93_139_314.12169167, rel=0.0, abs=roundoff)
        assert npv == pytest.approx(8_086_685.768167322, rel=0.0, abs=roundoff)
