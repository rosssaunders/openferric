"""Tests for risk functions: CVA, SA-CCR, VaR, ES."""

import math
import pytest
from openferric import py_cva, py_sa_ccr_ead, py_historical_var, py_historical_es
from conftest import REL_TOL, ABS_TOL, is_nan


# =========================================================================
# 22. py_cva
# =========================================================================

class TestCva:
    def test_nonzero_cva(self):
        """CVA should be nonzero for a positive exposure profile."""
        times = [0.5, 1.0, 2.0, 3.0, 5.0]
        ee = [100_000.0, 90_000.0, 80_000.0, 60_000.0, 40_000.0]
        cva = py_cva(times=times, ee_profile=ee, discount_rate=0.03,
                     hazard_rate=0.02, lgd=0.60)
        assert cva != 0.0

    def test_zero_exposure_zero_cva(self):
        """Zero exposure → zero CVA."""
        times = [1.0, 2.0, 3.0]
        ee = [0.0, 0.0, 0.0]
        cva = py_cva(times=times, ee_profile=ee, discount_rate=0.03,
                     hazard_rate=0.02, lgd=0.60)
        assert cva == pytest.approx(0.0, abs=ABS_TOL)

    def test_higher_hazard_larger_magnitude_cva(self):
        """Higher hazard rate → larger magnitude CVA."""
        times = [1.0, 2.0, 3.0]
        ee = [100_000.0, 80_000.0, 50_000.0]
        cva_low = py_cva(times=times, ee_profile=ee, discount_rate=0.03,
                          hazard_rate=0.01, lgd=0.60)
        cva_high = py_cva(times=times, ee_profile=ee, discount_rate=0.03,
                           hazard_rate=0.05, lgd=0.60)
        assert abs(cva_high) > abs(cva_low)


# =========================================================================
# 23. py_sa_ccr_ead
# =========================================================================

class TestSaCcrEad:
    @pytest.mark.parametrize("asset_class", ["ir", "fx", "credit", "equity", "commodity"])
    def test_positive_ead(self, asset_class):
        ead = py_sa_ccr_ead(replacement_cost=100_000.0, notional=1_000_000.0,
                            maturity=5.0, asset_class=asset_class)
        assert ead > 0.0

    def test_interest_rate_alias(self):
        ead1 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "ir")
        ead2 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "interest_rate")
        assert ead1 == pytest.approx(ead2, rel=1e-10)

    def test_fx_alias(self):
        ead1 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "fx")
        ead2 = py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "foreign_exchange")
        assert ead1 == pytest.approx(ead2, rel=1e-10)

    def test_invalid_asset_class(self):
        assert is_nan(py_sa_ccr_ead(100_000.0, 1_000_000.0, 5.0, "crypto"))


# =========================================================================
# 24. py_historical_var
# =========================================================================

class TestHistoricalVar:
    def test_known_percentile(self):
        """For sorted returns [-5,-4,-3,-2,-1,0,1,2,3,4], 95% VaR should be ~4."""
        returns = list(range(-5, 5))  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        var = py_historical_var(returns=returns, confidence=0.95)
        assert var > 0.0
        assert var == pytest.approx(4.0, abs=1.5)

    def test_higher_confidence_higher_var(self):
        returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        var_90 = py_historical_var(returns=returns, confidence=0.90)
        var_99 = py_historical_var(returns=returns, confidence=0.99)
        assert var_99 >= var_90

    def test_all_positive_returns(self):
        """All positive returns → VaR should be small or zero."""
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        var = py_historical_var(returns=returns, confidence=0.95)
        assert var <= 0.01  # very small loss or zero


# =========================================================================
# 25. py_historical_es
# =========================================================================

class TestHistoricalEs:
    def test_es_ge_var(self):
        """Expected Shortfall >= VaR for the same confidence level."""
        returns = [-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
        var = py_historical_var(returns=returns, confidence=0.95)
        es = py_historical_es(returns=returns, confidence=0.95)
        assert es >= var - 1e-10

    def test_positive_es(self):
        returns = [-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
        es = py_historical_es(returns=returns, confidence=0.95)
        assert es > 0.0

    def test_higher_confidence_higher_es(self):
        returns = [-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
        es_90 = py_historical_es(returns=returns, confidence=0.90)
        es_99 = py_historical_es(returns=returns, confidence=0.99)
        assert es_99 >= es_90
