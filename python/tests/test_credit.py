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
        assert npv == pytest.approx(198.09845254999126, rel=0.0, abs=2e-9)

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
        assert npv == pytest.approx(103100.31322752044, rel=0.0, abs=2e-9)

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
        assert npv == pytest.approx(-424287.2115125937, rel=0.0, abs=2e-9)

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
        assert actual == pytest.approx(expected, rel=0.0, abs=2e-15)

    def test_high_hazard_rate(self):
        """The high-hazard case still gets an exact exponential oracle."""
        prob = py_survival_prob(hazard_rate=1.0, t=5.0)
        assert prob == pytest.approx(math.exp(-5.0), rel=0.0, abs=2e-16)

    def test_zero_hazard_rate(self):
        """Zero hazard rate → survival prob = 1.0."""
        assert py_survival_prob(hazard_rate=0.0, t=10.0) == pytest.approx(1.0, rel=0.0, abs=ABS_TOL)


class TestDatedCdsCalendar:
    def test_legacy_flat_analytic_compatibility_method(self):
        from openferric import CdsDateRule, DatedCds, IsdaConventions, ProtectionSide

        cds = DatedCds(
            ProtectionSide.Buyer,
            10_000_000.0,
            0.01,
            0.4,
            "2026-03-20",
            "2026-06-20",
            3,
            CdsDateRule.QuarterlyImm,
        )
        result = cds.price_isda_flat_legacy_analytic("2026-03-20", 0.02, 0.03, IsdaConventions(0, 0))

        # Independent 80-decimal mpmath closed-form integral fixture, shared
        # with the Rust API test. The allowance is binary64 roundoff only.
        price_roundoff = 128 * math.ulp(30_471.572581091194)
        assert result.clean_npv == pytest.approx(5_175.727878884433, rel=0.0, abs=price_roundoff)
        assert result.premium_leg_pv == pytest.approx(25_295.84470220676, rel=0.0, abs=price_roundoff)
        assert result.protection_leg_pv == pytest.approx(30_471.572581091194, rel=0.0, abs=price_roundoff)
        assert result.fair_spread == pytest.approx(
            0.012046078294603427,
            rel=0.0,
            abs=128 * math.ulp(0.012046078294603427),
        )

    def test_target_calendar_price_matches_quantlib_isda_engine(self):
        import math

        from openferric import Calendar, DatedCds, ProtectionSide

        target = Calendar.target()
        cds = DatedCds.standard_imm_with_calendar(
            ProtectionSide.Buyer,
            "2026-04-02",
            5,
            10_000_000.0,
            0.01,
            0.4,
            target,
        )
        result = cds.price_isda_flat_with_calendar("2026-04-02", 0.02, 0.03, target, None)

        assert result.step_in_date == "2026-04-03"
        assert result.cash_settle_date == "2026-04-09"
        # QuantLib-Python 1.43, IsdaCdsEngine(Taylor, HalfDayBias,
        # Piecewise), with flat continuous Act/365F h=2%, r=3%.
        price_roundoff = 256 * math.ulp(551_249.8184500821)
        assert result.clean_npv == pytest.approx(87_274.59749599213, rel=0.0, abs=price_roundoff)
        assert result.dirty_npv == pytest.approx(83_387.9454065, rel=0.0, abs=price_roundoff)
        assert result.premium_leg_pv == pytest.approx(467_861.8730435821, rel=0.0, abs=price_roundoff)
        assert result.protection_leg_pv == pytest.approx(551_249.8184500821, rel=0.0, abs=price_roundoff)
        assert result.accrued_premium_pv == pytest.approx(3_886.652089492108, rel=0.0, abs=price_roundoff)
        assert result.fair_spread == pytest.approx(
            0.011881018501732185,
            rel=0.0,
            abs=128 * math.ulp(0.011881018501732185),
        )


class TestHeterogeneousSyntheticCdo:
    @staticmethod
    def build_model():
        from openferric import (
            CdoReferenceName,
            HeterogeneousSyntheticCdo,
            SurvivalCurve,
        )

        notionals = [1.0, 2.0, 3.0, 4.0]
        recoveries = [0.40, 0.50, 0.60, 0.75]
        hazards = [0.008, 0.015, 0.030, 0.060]
        loadings = [-0.15, 0.20, 0.45, 0.65]
        names = [
            CdoReferenceName(
                notionals[index],
                recoveries[index],
                loadings[index],
                SurvivalCurve.from_piecewise_hazard([5.0], [hazards[index]]),
            )
            for index in range(4)
        ]
        return HeterogeneousSyntheticCdo(
            names=names,
            risk_free_rate=0.03,
            maturity=5.0,
            payment_freq=4,
            loss_unit=0.02,
            factor_integration_points=64,
        )

    @staticmethod
    def build_tranche():
        from openferric import CdoTranche

        return CdoTranche(0.08, 0.24, 1.0, 0.0)

    def test_full_heterogeneous_cashflows_match_exact_scipy_reference(self):
        model = self.build_model()
        tranche = self.build_tranche()

        # SciPy 1.17.1 roots_hermitenorm(128), with each conditional loss
        # evaluated by both direct Bernoulli convolution and enumeration of
        # all 2^4 default states.
        assert model.portfolio_expected_loss(5.0) == pytest.approx(0.0522115057788266, rel=0.0, abs=3e-12)
        assert model.expected_loss_fraction(tranche, 5.0) == pytest.approx(0.1222569787570317, rel=0.0, abs=5e-12)
        assert model.protection_leg_pv(tranche) == pytest.approx(0.11304357859029632, rel=0.0, abs=5e-12)
        assert model.premium_leg_pv(tranche, 1.0) == pytest.approx(4.366093039976794, rel=0.0, abs=2e-11)
        assert model.fair_spread(tranche) == pytest.approx(0.025891243625650533, rel=0.0, abs=2e-12)

    def test_base_correlation_and_par_npv_match_exact_reference(self):
        from openferric import BaseCorrelationPair

        model = self.build_model()
        tranche = self.build_tranche()
        correlations = BaseCorrelationPair(0.18, 0.52)

        assert model.expected_loss_fraction_base_correlation(tranche, 5.0, correlations) == pytest.approx(
            0.10915105847793727, rel=0.0, abs=5e-12
        )
        assert model.protection_leg_pv_base_correlation(tranche, correlations) == pytest.approx(
            0.10100191177676004, rel=0.0, abs=5e-12
        )
        assert model.premium_leg_pv_base_correlation(tranche, 1.0, correlations) == pytest.approx(
            4.391377364905903, rel=0.0, abs=2e-11
        )
        assert model.fair_spread_base_correlation(tranche, correlations) == pytest.approx(
            0.023000052918231564, rel=0.0, abs=2e-12
        )

        tranche.spread = model.fair_spread_base_correlation(tranche, correlations)
        assert model.npv_base_correlation(tranche, correlations) == pytest.approx(0.0, rel=0.0, abs=3e-15)

    def test_non_commensurate_loss_unit_raises_value_error(self):
        model = self.build_model()
        model.loss_unit = 0.03
        with pytest.raises(ValueError, match="not a positive integer multiple"):
            model.expected_loss_fraction(self.build_tranche(), 5.0)

    @staticmethod
    def build_near_unit_model(factor_loading, factor_points=64):
        from openferric import (
            CdoReferenceName,
            HeterogeneousSyntheticCdo,
            SurvivalCurve,
        )

        names = [
            CdoReferenceName(
                1.0,
                0.40,
                factor_loading,
                SurvivalCurve([(5.0, 0.98)]),
            )
            for _ in range(10)
        ]
        return HeterogeneousSyntheticCdo(names, 0.03, 5.0, 4, 0.06, factor_points)

    def test_near_unit_loading_and_base_correlation_are_refined(self):
        from openferric import BaseCorrelationPair, CdoTranche

        tranche = CdoTranche(0.03, 0.06, 1.0, 0.0)
        reference = 0.0322516005452895
        model = self.build_near_unit_model(0.99)
        assert model.expected_loss_fraction(tranche, 5.0) == pytest.approx(reference, rel=0.0, abs=5e-12)

        base_model = self.build_near_unit_model(0.0)
        correlations = BaseCorrelationPair(0.99**2, 0.99**2)
        assert base_model.expected_loss_fraction_base_correlation(tranche, 5.0, correlations) == pytest.approx(
            reference, rel=0.0, abs=5e-12
        )

    def test_nonconvergence_and_low_initial_order_raise_value_error(self):
        from openferric import CdoTranche

        tranche = CdoTranche(0.03, 0.06, 1.0, 0.0)
        nonconvergent = self.build_near_unit_model(0.9999, 2_048)
        with pytest.raises(ValueError, match="did not converge"):
            nonconvergent.expected_loss_fraction(tranche, 5.0)

        low_order = self.build_near_unit_model(0.5, 31)
        with pytest.raises(ValueError, match=r"must be in \[32, 2048\]"):
            low_order.expected_loss_fraction(tranche, 5.0)

    def test_directly_mutated_survival_curve_is_rejected(self):
        from openferric import (
            CdoReferenceName,
            CdoTranche,
            HeterogeneousSyntheticCdo,
            SurvivalCurve,
        )

        curve = SurvivalCurve([(1.0, 0.90), (2.0, 0.80)])
        curve.tenors = [(2.0, 0.80), (1.0, 0.70)]
        name = CdoReferenceName(1.0, 0.40, 0.30, curve)
        model = HeterogeneousSyntheticCdo([name], 0.03, 5.0, 4, 0.60, 64)
        with pytest.raises(ValueError, match="strictly increasing"):
            model.expected_loss_fraction(CdoTranche(0.0, 0.6, 1.0, 0.0), 5.0)
