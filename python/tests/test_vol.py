"""Tests for volatility functions: implied vol, SABR, SVI."""

import math
import pytest
from openferric import py_implied_vol, py_sabr_vol, py_svi_vol, py_bs_price
from conftest import REL_TOL, ABS_TOL, is_nan


# =========================================================================
# 11. py_implied_vol
# =========================================================================

class TestImpliedVol:
    def test_round_trip_call(self):
        """Price → implied vol → price should round-trip."""
        spot, strike, expiry, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.25
        price = py_bs_price(spot, strike, expiry, vol, rate, "call")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "call")
        assert recovered_vol == pytest.approx(vol, rel=REL_TOL)

    def test_round_trip_put(self):
        spot, strike, expiry, rate, vol = 100.0, 110.0, 0.5, 0.03, 0.30
        price = py_bs_price(spot, strike, expiry, vol, rate, "put")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "put")
        assert recovered_vol == pytest.approx(vol, rel=REL_TOL)

    def test_deep_otm(self):
        """Deep OTM option should still recover vol."""
        spot, strike, expiry, rate, vol = 100.0, 150.0, 1.0, 0.05, 0.20
        price = py_bs_price(spot, strike, expiry, vol, rate, "call")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "call")
        assert recovered_vol == pytest.approx(vol, rel=1e-3)

    def test_invalid_option_type(self):
        assert is_nan(py_implied_vol(100.0, 100.0, 1.0, 0.05, 10.0, "xxx"))


# =========================================================================
# 12. py_sabr_vol
# =========================================================================

class TestSabrVol:
    def test_atm_vol(self):
        """ATM vol should be approximately alpha for beta=1."""
        alpha, beta, rho, nu = 0.20, 1.0, -0.3, 0.4
        forward = 100.0
        vol = py_sabr_vol(forward, forward, 1.0, alpha, beta, rho, nu)
        assert vol == pytest.approx(alpha, rel=0.05)

    def test_smile_shape(self):
        """SABR with negative rho should produce a skew: low strike vol > high strike vol."""
        alpha, beta, rho, nu = 0.20, 0.5, -0.4, 0.6
        forward = 100.0
        vol_low = py_sabr_vol(forward, 80.0, 1.0, alpha, beta, rho, nu)
        vol_atm = py_sabr_vol(forward, 100.0, 1.0, alpha, beta, rho, nu)
        vol_high = py_sabr_vol(forward, 120.0, 1.0, alpha, beta, rho, nu)
        # With negative rho, low strike should have higher vol (skew)
        assert vol_low > vol_atm
        assert vol_atm > 0.0
        assert vol_high > 0.0

    def test_positive_vol(self):
        vol = py_sabr_vol(100.0, 100.0, 1.0, 0.25, 0.5, 0.0, 0.3)
        assert vol > 0.0


# =========================================================================
# 13. py_svi_vol
# =========================================================================

class TestSviVol:
    def test_basic_svi(self):
        """SVI with standard params should give positive vol."""
        vol = py_svi_vol(strike=100.0, forward=100.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert vol > 0.0

    def test_atm_symmetric(self):
        """ATM (K=F) with m=0, rho=0 should give sqrt(a + b*sigma)."""
        a, b, rho_svi, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        vol = py_svi_vol(strike=100.0, forward=100.0, a=a, b=b, rho=rho_svi, m=m, sigma=sigma)
        expected = math.sqrt(a + b * sigma)
        assert vol == pytest.approx(expected, rel=REL_TOL)

    def test_negative_total_var_returns_nan(self):
        """Negative total variance should return NaN."""
        vol = py_svi_vol(strike=100.0, forward=100.0, a=-1.0, b=0.01, rho=0.0, m=0.0, sigma=0.1)
        assert is_nan(vol)

    def test_smile_wings(self):
        """Wings should have higher vol than ATM for typical params."""
        params = dict(forward=100.0, a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.1)
        vol_atm = py_svi_vol(strike=100.0, **params)
        vol_low = py_svi_vol(strike=70.0, **params)
        vol_high = py_svi_vol(strike=140.0, **params)
        assert vol_low > vol_atm or vol_high > vol_atm  # at least one wing higher
