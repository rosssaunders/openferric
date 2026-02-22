"""Tests for rates functions: swaption pricing."""

import pytest
from openferric import py_swaption_price
from conftest import REL_TOL, is_nan


# =========================================================================
# 21. py_swaption_price
# =========================================================================

class TestSwaptionPrice:
    @pytest.fixture
    def swaption_params(self):
        return dict(notional=1_000_000.0, strike=0.03, swap_tenor=5.0,
                    option_expiry=1.0, vol=0.15, discount_rate=0.03)

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
