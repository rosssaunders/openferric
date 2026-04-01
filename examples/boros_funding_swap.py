"""Pendle Boros funding-rate swap walkthrough using the Python bindings."""

from __future__ import annotations

from dataclasses import dataclass

from openferric import (
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
    funding_rate_swap_dv01,
    funding_rate_swap_mtm,
    funding_rate_swap_risks,
    funding_rate_swap_theta,
)


HOURS_PER_YEAR = 8760.0
BINANCE_WEIGHT = 0.65
BYBIT_WEIGHT = 0.35


@dataclass
class MarketContext:
    binance_curve: FundingRateCurve
    bybit_curve: FundingRateCurve
    composite_curve: MultiVenueFundingCurve
    pricing_curve: FundingRateCurve


def dt(day: int, hour: int) -> str:
    return f"2026-03-{day:02d}T{hour:02d}:00:00Z"


def hours_to_years(hours: float) -> float:
    return hours / HOURS_PER_YEAR


def build_snapshots(venue: str, asset: str, anchor_day: int, rates: list[float]) -> list[FundingRateSnapshot]:
    snapshots: list[FundingRateSnapshot] = []
    day = anchor_day
    hour = 0
    for rate in rates:
        snapshots.append(FundingRateSnapshot(venue, asset, rate, dt(day, hour)))
        hour += 8
        if hour >= 24:
            hour -= 24
            day += 1
    return snapshots


def build_reference_market_curves() -> MarketContext:
    binance_rates = [
        0.00010, 0.00011, 0.00010, 0.00011, 0.00012, 0.00011, 0.00012, 0.00012, 0.00013, 0.00012,
        0.00013, 0.00012, 0.00013, 0.00014, 0.00013, 0.00014, 0.00015, 0.00014, 0.00015, 0.00015,
        0.00016, 0.00015, 0.00016, 0.00017, 0.00016, 0.00017, 0.00018, 0.00017, 0.00016, 0.00017,
    ]
    bybit_rates = [
        max(rate * 0.96 + adj, 0.000095)
        for rate, adj in zip(
            binance_rates,
            [-0.000004, 0.000001, -0.000002, 0.000002, -0.000001, 0.000001] * 5,
        )
    ]

    binance_snapshots = build_snapshots("Binance", "BTCUSDT", 18, binance_rates)
    bybit_snapshots = build_snapshots("Bybit", "BTCUSDT", 18, bybit_rates)
    blended_snapshots = [
        FundingRateSnapshot(
            "BorosComposite",
            "BTCUSDT",
            (BINANCE_WEIGHT * b.rate + BYBIT_WEIGHT * y.rate) / (BINANCE_WEIGHT + BYBIT_WEIGHT),
            b.timestamp,
        )
        for b, y in zip(binance_snapshots, bybit_snapshots)
    ]

    binance_curve = FundingRateCurve(binance_snapshots)
    bybit_curve = FundingRateCurve(bybit_snapshots)
    composite_curve = MultiVenueFundingCurve(
        [(binance_curve, BINANCE_WEIGHT), (bybit_curve, BYBIT_WEIGHT)]
    )
    pricing_curve = FundingRateCurve(blended_snapshots)
    return MarketContext(binance_curve, bybit_curve, composite_curve, pricing_curve)


def scenario_label(scenario: StressScenario) -> str:
    if scenario.kind == "baseline":
        return "Baseline"
    if scenario.kind == "liquidation_cascade":
        return f"Cascade vol x{scenario.vol_multiplier:.0f}"
    return f"Mean shift {scenario.shift:+.1%}"


def main() -> None:
    market = build_reference_market_curves()
    anchor = market.binance_curve.anchor_timestamp()
    latest_snapshot = market.binance_curve.snapshots()[-1]
    latest_apr = FundingRateCurve.per_period_rate_to_apr(latest_snapshot.rate)
    rolling = market.binance_curve.rolling_stats(9)
    overall = FundingRateStats.from_rates([snapshot.rate for snapshot in market.binance_curve.snapshots()])

    print("\nPendle Boros Funding Rate Swap Reference Example\n")
    print(f"Anchor UTC: {anchor}")
    print(f"Latest Binance funding: {latest_apr:.4%} APR")
    print(f"Seven-day composite APR: {FundingRateCurve.per_period_rate_to_apr(market.composite_curve.forward_rate(7 / 365)):.4%}")
    print(
        "Overall Binance per-8h stats:",
        f"mean={overall.mean:.6f}",
        f"vol={overall.vol:.6f}",
        f"skew={overall.skew:.4f}",
        f"kurtosis={overall.kurtosis:.4f}",
    )
    print("Recent rolling stats:")
    for row in rolling[-3:]:
        print(
            f"  {row['timestamp']}: mean={row['mean']:.6f} vol={row['vol']:.6f} "
            f"skew={row['skew']:.4f} kurtosis={row['kurtosis']:.4f}"
        )

    swap = FundingRateSwap(
        5_000_000.0,
        0.10,
        "2026-03-24T03:00:00Z",
        "2026-03-29T00:00:00Z",
        "Boros blended index",
        "BTCUSDT",
    )
    swap.validate()
    as_of = "2026-03-27T00:00:00Z"

    fixing_lookup = {
        snapshot.timestamp: FundingRateCurve.per_period_rate_to_apr(snapshot.rate)
        for snapshot in market.pricing_curve.snapshots()
    }
    realized_fixings = [
        (settlement, fixing_lookup[settlement])
        for settlement in swap.settlement_schedule()
        if settlement <= as_of and settlement in fixing_lookup
    ]

    realized_pnl = swap.realized_pnl(realized_fixings)
    mtm = funding_rate_swap_mtm(swap, market.pricing_curve, as_of)
    risks = funding_rate_swap_risks(swap, market.pricing_curve, as_of)

    print("\nFunding-rate swap pricing")
    print(f"Realized PnL: {realized_pnl:,.2f}")
    print(f"Forward MTM:   {mtm:,.2f}")
    print(f"DV01:          {funding_rate_swap_dv01(swap, market.pricing_curve, as_of):,.4f}")
    print(f"Theta:         {funding_rate_swap_theta(swap, market.pricing_curve, as_of):,.4f}")
    print(f"Risk report:   {risks.to_dict()}")

    current_rate = market.pricing_curve.expected_rate(
        as_of, as_of, "2026-03-27T08:00:00Z"
    )
    margin_params = MarginParams(0.18, 0.12, 0.20, hours_to_years(30 * 24), 0.0001)
    initial_margin = MarginCalculator.initial_margin(swap.notional, margin_params)
    maintenance_margin = MarginCalculator.maintenance_margin(swap.notional, margin_params)
    collateral = initial_margin * 1.20
    position = LiquidationPosition(-swap.notional, current_rate, collateral, margin_params)
    leverage = InherentLeverage.leverage(swap.notional, 125_000.0)

    print("\nMargin and leverage")
    print(f"Current funding APR:      {current_rate:.4%}")
    print(f"Initial margin:           {initial_margin:,.2f}")
    print(f"Maintenance margin:       {maintenance_margin:,.2f}")
    print(
        f"Health ratio:             "
        f"{MarginCalculator.health_ratio(collateral, swap.notional, 0.0, margin_params):.4f}"
    )
    print(
        f"Liquidation funding APR:  "
        f"{MarginCalculator.liquidation_rate(current_rate, collateral, position.size, margin_params):.4%}"
    )
    print(f"Inherent leverage:        {leverage:.2f}x")
    print(f"100bp levered return:     {InherentLeverage.leveraged_return(0.01, leverage):.2%}")

    simulator = LiquidationSimulator(
        position,
        FundingRateModel.vasicek(Vasicek(8.0, current_rate - 0.005, 0.06)),
        current_rate,
        5_000,
        90,
        7,
    )
    scenarios = [StressScenario.baseline(), *StressScenario.cascade_suite(), *StressScenario.mean_shift_suite(0.03)]

    print("\nMonte Carlo liquidation")
    for result in simulator.run_stress_scenarios(scenarios):
        days = (
            f"{result.risk.expected_time_to_liquidation * 365:.2f}"
            if result.risk.expected_time_to_liquidation is not None
            else "n/a"
        )
        print(
            f"{scenario_label(result.scenario):<22} "
            f"liq_prob={result.risk.prob_liquidation:>7.2%} "
            f"exp_liq_days={days:>8} "
            f"worst_funding={result.risk.worst_case_funding_rate:>8.4%}"
        )


if __name__ == "__main__":
    main()
