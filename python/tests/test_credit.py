"""Tests for credit functions: CDS NPV and survival probability."""

import math
import pytest
from openferric import py_cds_npv, py_survival_prob
from conftest import REL_TOL, ABS_TOL, is_nan


# =========================================================================
# 19. py_cds_npv
# =========================================================================

class TestCdsNpv:
    def test_fair_spread_near_zero(self):
        """At the fair CDS spread, NPV should be approximately zero.
        For a rough fair spread: (1-R)*hazard_rate ≈ spread."""
        hazard_rate = 0.02
        recovery = 0.40
        fair_spread = (1.0 - recovery) * hazard_rate  # ≈ 0.012
        npv = py_cds_npv(
            notional=1_000_000.0, spread=fair_spread, maturity=5.0,
            recovery_rate=recovery, payment_freq=4,
            discount_rate=0.03, hazard_rate=hazard_rate,
        )
        assert abs(npv) < 5000.0  # NPV close to zero (not exact due to discretization)

    def test_protection_buyer_positive_npv(self):
        """If spread < fair spread, protection buyer benefits (positive NPV)."""
        npv = py_cds_npv(
            notional=1_000_000.0, spread=0.005, maturity=5.0,
            recovery_rate=0.40, payment_freq=4,
            discount_rate=0.03, hazard_rate=0.05,
        )
        assert npv > 0.0

    def test_protection_buyer_negative_npv(self):
        """If spread > fair spread, protection buyer overpays (negative NPV)."""
        npv = py_cds_npv(
            notional=1_000_000.0, spread=0.10, maturity=5.0,
            recovery_rate=0.40, payment_freq=4,
            discount_rate=0.03, hazard_rate=0.01,
        )
        assert npv < 0.0

    def test_zero_payment_freq_returns_nan(self):
        assert is_nan(py_cds_npv(
            notional=1_000_000.0, spread=0.01, maturity=5.0,
            recovery_rate=0.40, payment_freq=0,
            discount_rate=0.03, hazard_rate=0.02,
        ))


# =========================================================================
# 20. py_survival_prob
# =========================================================================

class TestSurvivalProb:
    def test_t_zero(self):
        """Survival probability at t=0 should be 1.0."""
        assert py_survival_prob(hazard_rate=0.05, t=0.0) == 1.0

    def test_negative_t(self):
        """Negative t should also return 1.0."""
        assert py_survival_prob(hazard_rate=0.05, t=-1.0) == 1.0

    def test_exponential_decay(self):
        """Survival prob ≈ exp(-λt) for constant hazard rate."""
        hazard_rate = 0.05
        t = 3.0
        expected = math.exp(-hazard_rate * t)
        actual = py_survival_prob(hazard_rate, t)
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_high_hazard_rate(self):
        """High hazard rate → low survival probability."""
        prob = py_survival_prob(hazard_rate=1.0, t=5.0)
        assert prob < 0.01

    def test_zero_hazard_rate(self):
        """Zero hazard rate → survival prob = 1.0."""
        assert py_survival_prob(hazard_rate=0.0, t=10.0) == pytest.approx(1.0, abs=ABS_TOL)
