"""Exact-reference tests for the Python risk-function bindings."""

import math

import pytest
from conftest import is_nan
from openferric import py_cva, py_historical_es, py_historical_var, py_sa_ccr_ead

# =========================================================================
# 22. py_cva
# =========================================================================


class TestCva:
    def test_flat_curve_cva_matches_discrete_closed_form(self):
        """The binding must preserve the core's stated right-endpoint CVA sum."""
        times = [0.5, 1.0, 2.0, 3.0, 5.0]
        ee = [100_000.0, 90_000.0, 80_000.0, 60_000.0, 40_000.0]
        discount_rate = 0.03
        hazard_rate = 0.02
        lgd = 0.60
        cva = py_cva(
            times=times,
            ee_profile=ee,
            discount_rate=discount_rate,
            hazard_rate=hazard_rate,
            lgd=lgd,
        )
        expected = -lgd * sum(
            math.exp(-discount_rate * t) * exposure * (math.exp(-hazard_rate * previous_t) - math.exp(-hazard_rate * t))
            for previous_t, t, exposure in zip([0.0, *times[:-1]], times, ee)
        )
        assert cva == pytest.approx(expected, rel=0.0, abs=16 * math.ulp(expected))

    def test_zero_exposure_zero_cva(self):
        """Zero exposure → zero CVA."""
        times = [1.0, 2.0, 3.0]
        ee = [0.0, 0.0, 0.0]
        cva = py_cva(times=times, ee_profile=ee, discount_rate=0.03, hazard_rate=0.02, lgd=0.60)
        assert cva == 0.0

    @pytest.mark.parametrize("hazard_rate", [0.01, 0.05])
    def test_cva_hazard_scenarios_match_closed_form(self, hazard_rate):
        times = [1.0, 2.0, 3.0]
        ee = [100_000.0, 80_000.0, 50_000.0]
        discount_rate = 0.03
        lgd = 0.60
        actual = py_cva(times, ee, discount_rate, hazard_rate, lgd)
        expected = -lgd * sum(
            math.exp(-discount_rate * t) * exposure * (math.exp(-hazard_rate * previous_t) - math.exp(-hazard_rate * t))
            for previous_t, t, exposure in zip([0.0, *times[:-1]], times, ee)
        )
        assert actual == pytest.approx(expected, rel=0.0, abs=16 * math.ulp(expected))


# =========================================================================
# 23. py_sa_ccr_ead
# =========================================================================


class TestSaCcrEad:
    @pytest.mark.parametrize(
        ("asset_class", "expected"),
        [
            ("ir", 147_000.0),
            ("fx", 196_000.0),
            ("credit", 210_000.0),
            ("equity", 588_000.0),
            ("commodity", 392_000.0),
        ],
    )
    def test_ead_matches_stated_sa_ccr_reduction(self, asset_class, expected):
        """MF=1, so EAD = 1.4 * (100k + supervisory_factor * 1m)."""
        ead = py_sa_ccr_ead(replacement_cost=100_000.0, notional=1_000_000.0, maturity=5.0, asset_class=asset_class)
        assert ead == pytest.approx(expected, rel=0.0, abs=8 * math.ulp(expected))

    def test_interest_rate_alias(self):
        ead1 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "ir")
        ead2 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "interest_rate")
        assert ead1 == ead2 == 147_000.0

    def test_fx_alias(self):
        ead1 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "fx")
        ead2 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "foreign_exchange")
        assert ead1 == ead2 == 196_000.0

    def test_invalid_asset_class(self):
        assert is_nan(py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "crypto"))


# =========================================================================
# 24. py_historical_var
# =========================================================================


class TestHistoricalVar:
    def test_known_percentile(self):
        """The interpolated 95% loss quantile is exactly 4 + 0.55 * (5 - 4)."""
        returns = list(range(-5, 5))  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        var = py_historical_var(returns=returns, confidence=0.95)
        assert var == pytest.approx(4.55, rel=0.0, abs=2 * math.ulp(4.55))

    @pytest.mark.parametrize(("confidence", "expected"), [(0.90, 0.032), (0.99, 0.0482)])
    def test_confidence_quantiles_are_exact(self, confidence, expected):
        # Sorted losses are [-.05, -.04, -.03, -.02, -.01, 0, .01, .02, .03, .05].
        # The type-7 rank p*(n-1) therefore interpolates between .03 and .05.
        returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        actual = py_historical_var(returns=returns, confidence=confidence)
        assert actual == pytest.approx(expected, rel=0.0, abs=4 * math.ulp(expected))

    def test_all_positive_returns(self):
        """The loss-positive convention floors a negative loss quantile at zero."""
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        var = py_historical_var(returns=returns, confidence=0.95)
        assert var == 0.0


# =========================================================================
# 25. py_historical_es
# =========================================================================


class TestHistoricalEs:
    @pytest.mark.parametrize(
        ("confidence", "expected_var", "expected_es"),
        [(0.80, 0.034, 0.075), (0.90, 0.055, 0.10), (0.95, 0.0775, 0.10), (0.99, 0.0955, 0.10)],
    )
    def test_var_and_discrete_tail_mean_are_exact(self, confidence, expected_var, expected_es):
        returns = [-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
        var = py_historical_var(returns=returns, confidence=confidence)
        es = py_historical_es(returns=returns, confidence=confidence)
        assert var == pytest.approx(expected_var, rel=0.0, abs=4 * math.ulp(expected_var))
        assert es == pytest.approx(expected_es, rel=0.0, abs=4 * math.ulp(expected_es))
