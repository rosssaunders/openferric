"""Tests for FFT pricing functions: Heston, VG, CGMY, NIG."""

import math
import pytest
from openferric import (
    py_heston_fft_price,
    py_heston_fft_prices,
    py_vg_fft_price,
    py_cgmy_fft_price,
    py_nig_fft_price,
    py_heston_price,
)
from conftest import is_nan


# =========================================================================
# 14. py_heston_fft_price
# =========================================================================

class TestHestonFftPrice:
    def test_matches_heston_price(self, heston_params):
        """FFT and semi-analytic should agree closely."""
        p = heston_params
        fft_price = py_heston_fft_price(
            spot=p["spot"], strike=100.0, expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        semi_price = py_heston_price(
            spot=p["spot"], strike=100.0, expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"], option_type="call",
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        assert fft_price == pytest.approx(semi_price, rel=1e-3)

    def test_alan_lewis_reference(self, heston_params):
        """Check against Alan Lewis K=80 call ≈ 26.775."""
        p = heston_params
        price = py_heston_fft_price(
            spot=p["spot"], strike=80.0, expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        assert price == pytest.approx(26.774758743998854, rel=1e-3)


# =========================================================================
# 15. py_heston_fft_prices
# =========================================================================

class TestHestonFftPrices:
    def test_batch_pricing(self, heston_params):
        p = heston_params
        prices = py_heston_fft_prices(
            spot=p["spot"], strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
            expiry=p["expiry"], rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        assert len(prices) == 5
        # All prices should be positive and decreasing for calls
        for price in prices:
            assert price > 0.0
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]

    def test_empty_strikes(self, heston_params):
        p = heston_params
        prices = py_heston_fft_prices(
            spot=p["spot"], strikes=[], expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        assert prices == []

    def test_single_strike_matches_scalar(self, heston_params):
        p = heston_params
        batch = py_heston_fft_prices(
            spot=p["spot"], strikes=[100.0], expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        scalar = py_heston_fft_price(
            spot=p["spot"], strike=100.0, expiry=p["expiry"],
            rate=p["rate"], div_yield=p["div_yield"],
            v0=p["v0"], kappa=p["kappa"], theta=p["theta"],
            sigma_v=p["sigma_v"], rho=p["rho"],
        )
        assert batch[0] == pytest.approx(scalar, rel=1e-10)


# =========================================================================
# 16. py_vg_fft_price
# =========================================================================

class TestVgFftPrice:
    def test_positive_price(self):
        price = py_vg_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                                rate=0.05, div_yield=0.0,
                                sigma=0.20, theta_vg=-0.14, nu=0.20)
        assert price > 0.0

    def test_otm_lower(self):
        atm = py_vg_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                               rate=0.05, div_yield=0.0,
                               sigma=0.20, theta_vg=-0.14, nu=0.20)
        otm = py_vg_fft_price(spot=100.0, strike=120.0, expiry=1.0,
                               rate=0.05, div_yield=0.0,
                               sigma=0.20, theta_vg=-0.14, nu=0.20)
        assert atm > otm


# =========================================================================
# 17. py_cgmy_fft_price
# =========================================================================

class TestCgmyFftPrice:
    def test_positive_price(self):
        price = py_cgmy_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                                  rate=0.05, div_yield=0.0,
                                  c=1.0, g=5.0, m=5.0, y=0.5)
        assert price > 0.0

    def test_otm_lower(self):
        atm = py_cgmy_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                                 rate=0.05, div_yield=0.0,
                                 c=1.0, g=5.0, m=5.0, y=0.5)
        otm = py_cgmy_fft_price(spot=100.0, strike=120.0, expiry=1.0,
                                 rate=0.05, div_yield=0.0,
                                 c=1.0, g=5.0, m=5.0, y=0.5)
        assert atm > otm


# =========================================================================
# 18. py_nig_fft_price
# =========================================================================

class TestNigFftPrice:
    def test_positive_price(self):
        price = py_nig_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                                 rate=0.05, div_yield=0.0,
                                 alpha=15.0, beta=-5.0, delta=0.5)
        assert price > 0.0

    def test_otm_lower(self):
        atm = py_nig_fft_price(spot=100.0, strike=100.0, expiry=1.0,
                                rate=0.05, div_yield=0.0,
                                alpha=15.0, beta=-5.0, delta=0.5)
        otm = py_nig_fft_price(spot=100.0, strike=120.0, expiry=1.0,
                                rate=0.05, div_yield=0.0,
                                alpha=15.0, beta=-5.0, delta=0.5)
        assert atm > otm
