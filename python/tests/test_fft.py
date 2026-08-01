"""Tests for FFT pricing functions: Heston, VG, CGMY, NIG."""

import math

import pytest
from openferric import (
    py_cgmy_fft_price,
    py_heston_fft_price,
    py_heston_fft_prices,
    py_heston_price,
    py_nig_fft_price,
    py_vg_fft_price,
)

# =========================================================================
# 14. py_heston_fft_price
# =========================================================================


class TestHestonFftPrice:
    def test_matches_heston_price(self, heston_params):
        """FFT and semi-analytic should agree closely."""
        p = heston_params
        fft_price = py_heston_fft_price(
            spot=p["spot"],
            strike=100.0,
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        semi_price = py_heston_price(
            spot=p["spot"],
            strike=100.0,
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            option_type="call",
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        assert fft_price == pytest.approx(semi_price, rel=0.0, abs=5e-12)

    def test_alan_lewis_reference(self, heston_params):
        """Check against Alan Lewis K=80 call ≈ 26.775."""
        p = heston_params
        price = py_heston_fft_price(
            spot=p["spot"],
            strike=80.0,
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        assert price == pytest.approx(26.774758743998854, rel=0.0, abs=5e-12)


# =========================================================================
# 15. py_heston_fft_prices
# =========================================================================


class TestHestonFftPrices:
    def test_batch_pricing(self, heston_params):
        p = heston_params
        prices = py_heston_fft_prices(
            spot=p["spot"],
            strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        # Alan Lewis' five-strike table, as cached by QuantLib's Heston tests.
        expected = [
            26.774758743998854,
            20.93334900059671,
            16.070154917028834,
            12.132211516709845,
            9.024913483457836,
        ]
        assert prices == pytest.approx(expected, rel=0.0, abs=5e-12)

    def test_empty_strikes(self, heston_params):
        p = heston_params
        prices = py_heston_fft_prices(
            spot=p["spot"],
            strikes=[],
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        assert prices == []

    def test_single_strike_matches_scalar(self, heston_params):
        p = heston_params
        batch = py_heston_fft_prices(
            spot=p["spot"],
            strikes=[100.0],
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        scalar = py_heston_fft_price(
            spot=p["spot"],
            strike=100.0,
            expiry=p["expiry"],
            rate=p["rate"],
            div_yield=p["div_yield"],
            v0=p["v0"],
            kappa=p["kappa"],
            theta=p["theta"],
            sigma_v=p["sigma_v"],
            rho=p["rho"],
        )
        # Both wrappers route the identical strike through the same finite FFT
        # grid and interpolation path, so this compatibility check is exact.
        assert batch[0] == scalar

    def test_small_alpha_fallback_finite_grid_lock(self):
        """Pin the deterministic auto-refined small-alpha production grid."""
        prices = py_heston_fft_prices(
            spot=100.0,
            strikes=[50.0, 100.0, 150.0],
            expiry=5.0,
            rate=0.02,
            div_yield=0.0,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=1.5,
            rho=0.7,
        )
        expected = [
            54.860113436677324,
            18.62391817429515,
            9.818703664151924,
        ]
        for actual, locked in zip(prices, expected, strict=True):
            binary64_budget = 256 * math.ulp(max(abs(locked), 1.0))
            assert actual == pytest.approx(locked, rel=0.0, abs=binary64_budget)


# =========================================================================
# 16. py_vg_fft_price
# =========================================================================


class TestVgFftPrice:
    def test_positive_price(self):
        """Match fypy's high-resolution Carr--Madan reference."""
        price = py_vg_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.01, sigma=0.20, theta_vg=0.10, nu=0.85
        )
        assert price == pytest.approx(10.13935062748614, rel=0.0, abs=5e-9)

    def test_otm_lower(self):
        atm = py_vg_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.0, sigma=0.20, theta_vg=-0.14, nu=0.20
        )
        otm = py_vg_fft_price(
            spot=100.0, strike=120.0, expiry=1.0, rate=0.05, div_yield=0.0, sigma=0.20, theta_vg=-0.14, nu=0.20
        )
        assert atm > otm


# =========================================================================
# 17. py_cgmy_fft_price
# =========================================================================


class TestCgmyFftPrice:
    def test_positive_price(self):
        """Match fypy's high-resolution Carr--Madan reference."""
        price = py_cgmy_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.01, c=0.02, g=5.0, m=15.0, y=1.2
        )
        assert price == pytest.approx(5.80222163947386, rel=0.0, abs=5e-6)

    def test_default_finite_grid_lock(self):
        """Pin the default finite grid separately from fypy's converged value."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        expected = [
            23.005036990350163,
            13.817853639255867,
            5.802224526781332,
            1.2927982063779289,
            0.18496095469813645,
        ]
        prices = [
            py_cgmy_fft_price(
                spot=100.0,
                strike=strike,
                expiry=1.0,
                rate=0.05,
                div_yield=0.01,
                c=0.02,
                g=5.0,
                m=15.0,
                y=1.2,
            )
            for strike in strikes
        ]
        for actual, locked in zip(prices, expected, strict=True):
            binary64_budget = 256 * math.ulp(max(abs(locked), 1.0))
            assert actual == pytest.approx(locked, rel=0.0, abs=binary64_budget)

    def test_otm_lower(self):
        atm = py_cgmy_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.0, c=1.0, g=5.0, m=5.0, y=0.5
        )
        otm = py_cgmy_fft_price(
            spot=100.0, strike=120.0, expiry=1.0, rate=0.05, div_yield=0.0, c=1.0, g=5.0, m=5.0, y=0.5
        )
        assert atm > otm


# =========================================================================
# 18. py_nig_fft_price
# =========================================================================


class TestNigFftPrice:
    def test_positive_price(self):
        """Match fypy's high-resolution Carr--Madan reference."""
        price = py_nig_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.01, alpha=15.0, beta=-5.0, delta=0.5
        )
        assert price == pytest.approx(9.63000693130414, rel=0.0, abs=5e-12)

    def test_otm_lower(self):
        atm = py_nig_fft_price(
            spot=100.0, strike=100.0, expiry=1.0, rate=0.05, div_yield=0.0, alpha=15.0, beta=-5.0, delta=0.5
        )
        otm = py_nig_fft_price(
            spot=100.0, strike=120.0, expiry=1.0, rate=0.05, div_yield=0.0, alpha=15.0, beta=-5.0, delta=0.5
        )
        assert atm > otm
