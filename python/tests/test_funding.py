"""Tests for Boros funding-rate Python bindings."""

import math

import pytest

from conftest import ABS_TOL
from openferric import (
    YieldCurve,
    FundingRateCurve,
    FundingRateModel,
    FundingRateSnapshot,
    FundingRateStats,
    FundingRateSwap,
    InherentLeverage,
    LiquidationPosition,
    LiquidationSimulator,
    MarginCalculator,
    MarginParams,
    MultiVenueFundingCurve,
    StressScenario,
    Vasicek,
    funding_rate_swap_discount_dv01,
    funding_rate_swap_dv01,
    funding_rate_swap_mtm,
    funding_rate_swap_risks,
    funding_rate_swap_theta,
)


def dt(day: int, hour: int) -> str:
    return f"2026-03-{day:02d}T{hour:02d}:00:00Z"


def build_snapshots(venue: str, asset: str, rates: list[float]) -> list[FundingRateSnapshot]:
    snapshots = []
    day = 24
    hour = 0
    for rate in rates:
        snapshots.append(FundingRateSnapshot(venue, asset, rate, dt(day, hour)))
        hour += 8
        if hour >= 24:
            hour -= 24
            day += 1
    return snapshots


class TestFundingRateCurve:
    def test_curve_snapshots_stats_and_nodes(self):
        snapshots = build_snapshots(
            "Binance",
            "BTCUSDT",
            [0.00010, 0.00012, 0.00014, 0.00016],
        )
        curve = FundingRateCurve(snapshots)

        assert curve.anchor_timestamp() == "2026-03-24T00:00:00Z"
        assert len(curve.snapshots()) == 4
        assert curve.snapshots()[0].venue == "Binance"
        assert len(curve.nodes()) == 4
        assert curve.forward_rate(0.0) == pytest.approx(0.00010, abs=ABS_TOL)
        assert curve.forward_rate(1.0 / (1095.0 * 2.0)) == pytest.approx(0.00011, abs=1e-12)
        assert curve.discount_factor(2.0 / 1095.0) < 1.0

        stats = FundingRateStats.from_rates([snapshot.rate for snapshot in snapshots])
        assert stats.window_size == 4
        assert stats.mean == pytest.approx(0.00013, abs=1e-12)

        rolling = curve.rolling_stats(3)
        assert len(rolling) == 2
        assert rolling[-1]["timestamp"] == "2026-03-25T00:00:00Z"
        assert math.isfinite(rolling[-1]["vol"])

    def test_multi_venue_weighting(self):
        venue_a = FundingRateCurve(
            build_snapshots("Binance", "BTCUSDT", [0.0001, 0.0001])
        )
        venue_b = FundingRateCurve(build_snapshots("Bybit", "BTCUSDT", [0.0003, 0.0003]))
        composite = MultiVenueFundingCurve([(venue_a, 1.0), (venue_b, 3.0)])

        assert composite.forward_rate(0.0) == pytest.approx(0.00025, abs=1e-12)
        assert composite.cumulative_index(1.0 / 1095.0) == pytest.approx(0.00025, abs=1e-12)


class TestFundingRateSwap:
    def test_schedule_realized_and_pricing_functions(self):
        swap = FundingRateSwap(
            1_000.0,
            0.10,
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "Boros blended index",
            "BTCUSDT",
        )
        curve = FundingRateCurve.flat(0.13)

        assert swap.validate() is None
        assert swap.settlement_schedule() == [
            "2026-01-01T08:00:00Z",
            "2026-01-01T16:00:00Z",
            "2026-01-02T00:00:00Z",
        ]

        realized = swap.realized_pnl(
            [
                ("2026-01-01T08:00:00Z", 0.12),
                ("2026-01-01T16:00:00Z", 0.08),
                ("2026-01-02T00:00:00Z", 0.11),
            ]
        )
        expected_realized = ((0.12 - 0.10) + (0.08 - 0.10) + (0.11 - 0.10)) * 1_000.0 * (8.0 / 8760.0)
        assert realized == pytest.approx(expected_realized, abs=1e-12)

        mtm = funding_rate_swap_mtm(swap, curve, "2026-01-01T08:00:00Z")
        assert mtm == pytest.approx(2.0 * (0.13 - 0.10) * 1_000.0 * (8.0 / 8760.0), abs=1e-12)

        discount_curve = YieldCurve([
            (8.0 / 8760.0, 0.99),
            (16.0 / 8760.0, 0.97),
        ])
        discounted_mtm = funding_rate_swap_mtm(
            swap,
            curve,
            "2026-01-01T08:00:00Z",
            discount_curve=discount_curve,
        )
        expected_interval = (0.13 - 0.10) * 1_000.0 * (8.0 / 8760.0)
        assert discounted_mtm == pytest.approx(expected_interval * 0.99 + expected_interval * 0.97, abs=1e-12)

        dv01 = funding_rate_swap_dv01(
            FundingRateSwap(
                5_000.0,
                0.04,
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "OKX",
                "ETHUSDT",
            ),
            FundingRateCurve.flat(0.05),
            "2026-01-01T00:00:00Z",
        )
        assert dv01 == pytest.approx(3.0 * 5_000.0 * (8.0 / 8760.0) * 1.0e-4, abs=1e-12)

        discount_dv01 = funding_rate_swap_discount_dv01(
            swap,
            curve,
            "2026-01-01T08:00:00Z",
            discount_curve=discount_curve,
        )
        assert discount_dv01 < 0.0

        theta = funding_rate_swap_theta(swap, curve, "2026-01-01T00:00:00Z")
        assert theta == pytest.approx(-(0.13 - 0.10) * 1_000.0 * (8.0 / 8760.0), abs=1e-12)

        risks = funding_rate_swap_risks(
            swap,
            curve,
            "2026-01-01T08:00:00Z",
            discount_curve=discount_curve,
        )
        assert risks.mtm == pytest.approx(discounted_mtm, abs=1e-12)
        assert set(risks.to_dict()) == {"mtm", "dv01", "vega", "theta"}


class TestMarginAndLiquidation:
    def test_margin_leverage_and_liquidation(self):
        params = MarginParams(0.18, 0.12, 0.20, 30.0 / 365.0, 0.0001)

        initial_margin = MarginCalculator.initial_margin(5_000_000.0, params)
        maintenance_margin = MarginCalculator.maintenance_margin(5_000_000.0, params)
        assert initial_margin > maintenance_margin > 0.0

        health_ratio = MarginCalculator.health_ratio(initial_margin * 1.2, 5_000_000.0, 0.0, params)
        assert health_ratio > 1.0

        liquidation_rate = MarginCalculator.liquidation_rate(
            0.12, initial_margin * 1.2, -5_000_000.0, params
        )
        assert math.isfinite(liquidation_rate)

        leverage = InherentLeverage.leverage(5_000_000.0, 125_000.0)
        assert leverage == pytest.approx(40.0, abs=ABS_TOL)
        assert InherentLeverage.leveraged_return(0.01, leverage) == pytest.approx(0.4, abs=ABS_TOL)

        position = LiquidationPosition(
            -5_000_000.0,
            0.12,
            initial_margin * 1.2,
            params,
        )
        model = FundingRateModel.vasicek(Vasicek(8.0, 0.115, 0.06))
        simulator = LiquidationSimulator(position, model, 0.12, 256, 32, 7)

        baseline = simulator.simulate()
        assert 0.0 <= baseline.prob_liquidation <= 1.0
        assert math.isfinite(baseline.worst_case_funding_rate)

        scenarios = [StressScenario.baseline()]
        scenarios.extend(StressScenario.cascade_suite())
        scenarios.extend(StressScenario.mean_shift_suite(0.03))
        results = simulator.run_stress_scenarios(scenarios)

        assert len(results) == 6
        assert results[0].scenario.kind == "baseline"
        assert all(0.0 <= result.risk.prob_liquidation <= 1.0 for result in results)
