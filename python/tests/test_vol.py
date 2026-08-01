"""Tests for volatility functions: implied vol, SABR, SVI."""

import math

import pytest
from conftest import is_nan
from openferric import py_bs_price, py_fit_sabr, py_implied_vol, py_sabr_vol, py_svi_vol

# =========================================================================
# 11. py_implied_vol
# =========================================================================


class TestImpliedVol:
    def test_round_trip_call(self):
        """Price → implied vol → price should round-trip."""
        spot, strike, expiry, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.25
        price = py_bs_price(spot, strike, expiry, vol, rate, "call")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "call")
        assert recovered_vol == pytest.approx(vol, abs=1e-12)

    def test_round_trip_put(self):
        spot, strike, expiry, rate, vol = 100.0, 110.0, 0.5, 0.03, 0.30
        price = py_bs_price(spot, strike, expiry, vol, rate, "put")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "put")
        assert recovered_vol == pytest.approx(vol, abs=1e-12)

    def test_deep_otm(self):
        """Deep OTM option should still recover vol."""
        spot, strike, expiry, rate, vol = 100.0, 150.0, 1.0, 0.05, 0.20
        price = py_bs_price(spot, strike, expiry, vol, rate, "call")
        recovered_vol = py_implied_vol(spot, strike, expiry, rate, price, "call")
        assert recovered_vol == pytest.approx(vol, abs=1e-12)

    def test_invalid_option_type(self):
        assert is_nan(py_implied_vol(100.0, 100.0, 1.0, 0.05, 10.0, "xxx"))


# =========================================================================
# 12. py_sabr_vol
# =========================================================================


class TestSabrVol:
    def test_atm_vol_matches_hagan_formula(self):
        alpha, beta, rho, nu = 0.20, 1.0, -0.3, 0.4
        forward = 100.0
        vol = py_sabr_vol(forward, forward, 1.0, alpha, beta, rho, nu)
        correction = rho * beta * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        expected = alpha * (1.0 + correction)
        assert vol == pytest.approx(expected, abs=2e-16)

    def test_smile_shape(self):
        """SABR with negative rho should produce a skew: low strike vol > high strike vol."""
        alpha, beta, rho, nu = 0.20, 0.5, -0.4, 0.6
        forward = 100.0
        vol_low = py_sabr_vol(forward, 80.0, 1.0, alpha, beta, rho, nu)
        vol_atm = py_sabr_vol(forward, 100.0, 1.0, alpha, beta, rho, nu)
        vol_high = py_sabr_vol(forward, 120.0, 1.0, alpha, beta, rho, nu)
        assert [vol_low, vol_atm, vol_high] == pytest.approx(
            [0.060309788022575306, 0.02044408333333333, 0.03876535566992095],
            abs=2e-16,
        )

    def test_second_atm_hagan_reference(self):
        vol = py_sabr_vol(100.0, 100.0, 1.0, 0.25, 0.5, 0.0, 0.3)
        assert vol == pytest.approx(0.02518766276041667, abs=2e-16)

    def test_synthetic_calibration_reprices_quotes(self):
        alpha, beta, rho, nu = 0.25, 0.5, -0.2, 0.4
        forward, expiry = 100.0, 1.0
        strikes = [float(k) for k in range(80, 121, 5)]
        quotes = [py_sabr_vol(forward, k, expiry, alpha, beta, rho, nu) for k in strikes]
        fit = py_fit_sabr(forward, strikes, quotes, expiry, beta)
        repriced = [py_sabr_vol(forward, k, expiry, fit.alpha, fit.beta, fit.rho, fit.nu) for k in strikes]
        assert repriced == pytest.approx(quotes, abs=1e-9)


# =========================================================================
# 13. py_svi_vol
# =========================================================================


class TestSviVol:
    def test_basic_svi(self):
        """SVI binding matches the independently evaluated total-variance formula."""
        vol = py_svi_vol(strike=100.0, forward=100.0, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert vol == pytest.approx(math.sqrt(0.04 + 0.1 * 0.1), abs=2e-16)

    def test_atm_symmetric(self):
        """ATM (K=F) with m=0, rho=0 should give sqrt(a + b*sigma)."""
        a, b, rho_svi, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        vol = py_svi_vol(strike=100.0, forward=100.0, a=a, b=b, rho=rho_svi, m=m, sigma=sigma)
        expected = math.sqrt(a + b * sigma)
        assert vol == pytest.approx(expected, abs=2e-16)

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
        assert [vol_low, vol_atm, vol_high] == pytest.approx(
            [0.29013154612540676, 0.22360679774997896, 0.261481060983064],
            abs=2e-16,
        )
