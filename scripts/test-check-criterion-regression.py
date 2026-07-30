#!/usr/bin/env python3
"""Self-tests for the dependency-free Criterion regression gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("check-criterion-regression.py")


def write_estimate(group: Path, benchmark: str, baseline: str, value: float) -> None:
    path = group / benchmark / baseline / "estimates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"median": {"point_estimate": value}}),
        encoding="utf-8",
    )


def run_gate(
    group: Path, expected: int | None = 2
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(GATE),
        str(group),
        "--baseline",
        "main",
        "--current",
        "candidate",
        "--max-regression-percent",
        "20",
    ]
    if expected is not None:
        command.extend(["--expected-benchmarks", str(expected)])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


class CriterionRegressionGateTests(unittest.TestCase):
    def test_nested_benchmarks_are_discovered_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "function/100000", "main", 100.0)
            write_estimate(group, "function/100000", "candidate", 110.0)
            write_estimate(group, "function/1000000", "main", 200.0)
            write_estimate(group, "function/1000000", "candidate", 180.0)

            result = run_gate(group)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("function/100000:", result.stdout)
            self.assertIn("change=-10.00%", result.stdout)

    def test_missing_baseline_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)
            write_estimate(group, "a", "candidate", 100.0)
            write_estimate(group, "b", "candidate", 100.0)

            result = run_gate(group)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("baseline", result.stdout + result.stderr)

    def test_baseline_only_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)

            result = run_gate(group, expected=None)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("current label 'candidate'", result.stdout + result.stderr)

    def test_stale_new_results_do_not_satisfy_candidate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)
            write_estimate(group, "a", "new", 100.0)

            result = run_gate(group, expected=None)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("current label 'candidate'", result.stdout + result.stderr)

    def test_missing_candidate_benchmark_fails_exact_set_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)
            write_estimate(group, "a", "candidate", 100.0)
            write_estimate(group, "b", "main", 100.0)
            write_estimate(group, "b", "new", 100.0)

            result = run_gate(group, expected=None)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("b: current estimate", result.stdout)
            self.assertIn("/candidate/estimates.json is missing", result.stdout)

    def test_expected_benchmark_count_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)
            write_estimate(group, "a", "candidate", 100.0)

            result = run_gate(group, expected=2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 2 benchmark estimates, found 1", result.stderr)

    def test_injected_regression_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            write_estimate(group, "a", "main", 100.0)
            write_estimate(group, "a", "candidate", 120.01)

            result = run_gate(group, expected=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regressed 20.01%", result.stdout)


if __name__ == "__main__":
    unittest.main()
