"""Tests for credit functions: CDS NPV and survival probability."""

import math

import pytest
from conftest import ABS_TOL, is_nan
from openferric import py_cds_npv, py_survival_prob

# =========================================================================
# 19. py_cds_npv
# =========================================================================


class TestCdsNpv:
    def test_flat_hazard_cashflows_match_independent_midpoint_sum(self):
        """Assert the exact quarterly midpoint-model NPV, not a near-zero band."""
        hazard_rate = 0.02
        recovery = 0.40
        npv = py_cds_npv(
            notional=1_000_000.0,
            spread=0.012,
            maturity=5.0,
            recovery_rate=recovery,
            payment_freq=4,
            discount_rate=0.03,
            hazard_rate=hazard_rate,
        )
        # Independently summed from exp(-r t), exp(-lambda t), quarterly
        # coupons, midpoint protection, and half-period accrued premium.
        assert npv == pytest.approx(198.09845254999126, abs=2e-9)

    def test_protection_buyer_positive_npv(self):
        """If spread < fair spread, protection buyer benefits (positive NPV)."""
        npv = py_cds_npv(
            notional=1_000_000.0,
            spread=0.005,
            maturity=5.0,
            recovery_rate=0.40,
            payment_freq=4,
            discount_rate=0.03,
            hazard_rate=0.05,
        )
        assert npv == pytest.approx(103100.31322752044, abs=2e-9)

    def test_protection_buyer_negative_npv(self):
        """If spread > fair spread, protection buyer overpays (negative NPV)."""
        npv = py_cds_npv(
            notional=1_000_000.0,
            spread=0.10,
            maturity=5.0,
            recovery_rate=0.40,
            payment_freq=4,
            discount_rate=0.03,
            hazard_rate=0.01,
        )
        assert npv == pytest.approx(-424287.2115125937, abs=2e-9)

    def test_zero_payment_freq_returns_nan(self):
        assert is_nan(
            py_cds_npv(
                notional=1_000_000.0,
                spread=0.01,
                maturity=5.0,
                recovery_rate=0.40,
                payment_freq=0,
                discount_rate=0.03,
                hazard_rate=0.02,
            )
        )


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
        """Survival probability equals exp(-lambda*t) for constant hazard."""
        hazard_rate = 0.05
        t = 3.0
        expected = math.exp(-hazard_rate * t)
        actual = py_survival_prob(hazard_rate, t)
        assert actual == pytest.approx(expected, abs=2e-15)

    def test_high_hazard_rate(self):
        """The high-hazard case still gets an exact exponential oracle."""
        prob = py_survival_prob(hazard_rate=1.0, t=5.0)
        assert prob == pytest.approx(math.exp(-5.0), abs=2e-16)

    def test_zero_hazard_rate(self):
        """Zero hazard rate → survival prob = 1.0."""
        assert py_survival_prob(hazard_rate=0.0, t=10.0) == pytest.approx(1.0, abs=ABS_TOL)
