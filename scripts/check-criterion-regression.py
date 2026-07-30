#!/usr/bin/env python3
"""Gate Criterion benchmark medians against a saved same-host baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def median(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    try:
        value = float(document["median"]["point_estimate"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{path}: missing numeric median.point_estimate") from error
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"{path}: median.point_estimate must be finite and positive")
    return value


def estimates_by_benchmark(group: Path, label: str) -> dict[str, Path]:
    """Return estimates for one Criterion label, keyed by benchmark path."""
    estimates: dict[str, Path] = {}
    for path in sorted(group.rglob("estimates.json")):
        if path.parent.name != label:
            continue
        benchmark_dir = path.parent.parent
        benchmark_name = benchmark_dir.relative_to(group).as_posix()
        estimates[benchmark_name] = path
    return estimates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=Path, help="Criterion benchmark-group directory")
    parser.add_argument("--baseline", default="main")
    parser.add_argument("--current", default="candidate")
    parser.add_argument("--max-regression-percent", type=float, default=20.0)
    parser.add_argument("--expected-benchmarks", type=int)
    args = parser.parse_args()
    if (
        not math.isfinite(args.max_regression_percent)
        or args.max_regression_percent < 0.0
    ):
        raise SystemExit("--max-regression-percent must be finite and non-negative")
    if args.expected_benchmarks is not None and args.expected_benchmarks <= 0:
        raise SystemExit("--expected-benchmarks must be positive")

    baseline_estimates = estimates_by_benchmark(args.group, args.baseline)
    current_estimates = estimates_by_benchmark(args.group, args.current)
    if (
        args.expected_benchmarks is not None
        and len(current_estimates) != args.expected_benchmarks
    ):
        raise SystemExit(
            f"{args.group}: expected {args.expected_benchmarks} benchmark estimates, "
            f"found {len(current_estimates)}"
        )
    if not current_estimates:
        raise SystemExit(
            f"{args.group}: no Criterion estimates found for current label "
            f"{args.current!r}"
        )
    if not baseline_estimates:
        raise SystemExit(
            f"{args.group}: no Criterion estimates found for baseline label "
            f"{args.baseline!r}"
        )

    failures: list[str] = []
    baseline_names = set(baseline_estimates)
    current_names = set(current_estimates)
    for benchmark_name in sorted(baseline_names - current_names):
        expected_path = (
            baseline_estimates[benchmark_name].parent.parent
            / args.current
            / "estimates.json"
        )
        failures.append(
            f"{benchmark_name}: current estimate {expected_path} is missing"
        )
    for benchmark_name in sorted(current_names - baseline_names):
        expected_path = (
            current_estimates[benchmark_name].parent.parent
            / args.baseline
            / "estimates.json"
        )
        failures.append(
            f"{benchmark_name}: baseline estimate {expected_path} is missing"
        )

    for benchmark_name in sorted(baseline_names & current_names):
        baseline_path = baseline_estimates[benchmark_name]
        current_path = current_estimates[benchmark_name]
        baseline_ns = median(baseline_path)
        current_ns = median(current_path)
        change_percent = (current_ns / baseline_ns - 1.0) * 100.0
        print(
            f"{benchmark_name}: baseline={baseline_ns:.3f}ns "
            f"current={current_ns:.3f}ns change={change_percent:+.2f}%"
        )
        if change_percent > args.max_regression_percent:
            failures.append(
                f"{benchmark_name}: median regressed {change_percent:.2f}% "
                f"(limit {args.max_regression_percent:.2f}%)"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
