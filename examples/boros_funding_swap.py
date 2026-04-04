"""Pendle Boros funding-rate swap walkthrough using live Binance and Bybit history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

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
    funding_rate_swap_vega,
)

HOURS_PER_YEAR = 8760.0
BINANCE_WEIGHT = 0.65
BYBIT_WEIGHT = 0.35
HTTP_TIMEOUT_SECONDS = 20
FUNDING_LIMIT = 30


@dataclass
class MarketContext:
    binance_curve: FundingRateCurve
    bybit_curve: FundingRateCurve
    composite_curve: MultiVenueFundingCurve
    pricing_curve: FundingRateCurve


def hours_to_years(hours: float) -> float:
    return hours / HOURS_PER_YEAR


def isoformat_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "openferric-python-example/0.1"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def fetch_binance_snapshots(symbol: str = "BTCUSDT", limit: int = FUNDING_LIMIT) -> list[FundingRateSnapshot]:
    payload = fetch_json(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Binance funding API returned no rows")

    rows = sorted(payload, key=lambda row: int(row["fundingTime"]))
    return [
        FundingRateSnapshot(
            "Binance",
            symbol,
            float(row["fundingRate"]),
            isoformat_utc(datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=timezone.utc)),
        )
        for row in rows
    ]


def fetch_bybit_snapshots(symbol: str = "BTCUSDT", limit: int = FUNDING_LIMIT) -> list[FundingRateSnapshot]:
    payload = fetch_json(
        f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit={limit}"
    )
    if not isinstance(payload, dict) or payload.get("retCode") not in (0, "0"):
        raise RuntimeError(f"Bybit funding API error: {payload}")

    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError("Bybit funding API returned no rows")

    rows = sorted(rows, key=lambda row: int(row["fundingRateTimestamp"]))
    return [
        FundingRateSnapshot(
            "Bybit",
            symbol,
            float(row["fundingRate"]),
            isoformat_utc(datetime.fromtimestamp(int(row["fundingRateTimestamp"]) / 1000, tz=timezone.utc)),
        )
        for row in rows
    ]


def blend_snapshots(
    binance_snapshots: list[FundingRateSnapshot], bybit_snapshots: list[FundingRateSnapshot]
) -> list[FundingRateSnapshot]:
    bybit_by_timestamp = {snapshot.timestamp: snapshot for snapshot in bybit_snapshots}
    common_timestamps = [
        snapshot.timestamp for snapshot in binance_snapshots if snapshot.timestamp in bybit_by_timestamp
    ]
    if not common_timestamps:
        raise RuntimeError("Binance and Bybit histories have no common funding timestamps")

    blended: list[FundingRateSnapshot] = []
    for timestamp in common_timestamps:
        binance = next(snapshot for snapshot in binance_snapshots if snapshot.timestamp == timestamp)
        bybit = bybit_by_timestamp[timestamp]
        blended.append(
            FundingRateSnapshot(
                "BorosComposite",
                binance.asset,
                (BINANCE_WEIGHT * binance.rate + BYBIT_WEIGHT * bybit.rate) / (BINANCE_WEIGHT + BYBIT_WEIGHT),
                timestamp,
            )
        )
    return blended


def build_reference_market_curves() -> MarketContext:
    binance_snapshots = fetch_binance_snapshots()
    bybit_snapshots = fetch_bybit_snapshots()
    blended_snapshots = blend_snapshots(binance_snapshots, bybit_snapshots)

    binance_curve = FundingRateCurve(binance_snapshots)
    bybit_curve = FundingRateCurve(bybit_snapshots)
    composite_curve = MultiVenueFundingCurve([(binance_curve, BINANCE_WEIGHT), (bybit_curve, BYBIT_WEIGHT)])
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
    latest_snapshot = market.pricing_curve.snapshots()[-1]
    latest_timestamp = parse_utc(latest_snapshot.timestamp)
    entry_timestamp = market.pricing_curve.snapshots()[-8].timestamp
    maturity_timestamp = isoformat_utc(latest_timestamp + timedelta(hours=32))
    as_of = latest_snapshot.timestamp

    rolling_window = max(2, min(9, len(market.binance_curve.snapshots())))
    rolling = market.binance_curve.rolling_stats(rolling_window)
    overall = FundingRateStats.from_rates([snapshot.rate for snapshot in market.binance_curve.snapshots()])

    print("\nPendle Boros Funding Rate Swap Reference Example\n")
    print(f"Anchor UTC: {market.binance_curve.anchor_timestamp()}")
    print(
        f"Latest composite funding: "
        f"{FundingRateCurve.per_period_rate_to_apr(latest_snapshot.rate):.4%} APR "
        f"at {latest_snapshot.timestamp}"
    )
    print(
        "Seven-day composite APR:",
        f"{FundingRateCurve.per_period_rate_to_apr(market.composite_curve.forward_rate(7 / 365)):.4%}",
    )
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
        entry_timestamp,
        maturity_timestamp,
        "Boros blended index",
        "BTCUSDT",
    )
    swap.validate()

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
    print(f"Swap entry:    {entry_timestamp}")
    print(f"As of:         {as_of}")
    print(f"Maturity:      {maturity_timestamp}")
    print(f"Realized PnL:  {realized_pnl:,.2f}")
    print(f"Forward MTM:   {mtm:,.2f}")
    print(f"DV01:          {funding_rate_swap_dv01(swap, market.pricing_curve, as_of):,.4f}")
    print(f"Vega:          {funding_rate_swap_vega(swap, market.pricing_curve, as_of):,.4f}")
    print(f"Theta:         {funding_rate_swap_theta(swap, market.pricing_curve, as_of):,.4f}")
    print(f"Risk report:   {risks.to_dict()}")

    next_settlement = isoformat_utc(latest_timestamp + timedelta(hours=8))
    current_rate = market.pricing_curve.expected_rate(as_of, as_of, next_settlement)
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
        f"Health ratio:             {MarginCalculator.health_ratio(collateral, swap.notional, 0.0, margin_params):.4f}"
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
    scenarios = [
        StressScenario.baseline(),
        *StressScenario.cascade_suite(),
        *StressScenario.mean_shift_suite(0.03),
    ]

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
