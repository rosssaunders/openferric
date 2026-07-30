"""Shared fixtures and constants for openferric Python tests."""

import math
import sys
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

REL_TOL = 1e-4  # relative tolerance for pricing values
ABS_TOL = 1e-8  # absolute tolerance for near-zero values
LOOSE_TOL = 1e-2  # looser tolerance for MC / FFT comparisons


# ---------------------------------------------------------------------------
# Standard market parameters (used across many tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def std_market():
    """Standard Black-Scholes market: S=100, K=100, T=1, vol=20%, r=5%."""
    return dict(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05)


@pytest.fixture
def heston_params():
    """Alan Lewis reference parameters from QuantLib test-suite."""
    return dict(
        spot=100.0,
        v0=0.04,
        rho=-0.5,
        sigma_v=1.0,
        kappa=4.0,
        theta=0.25,
        rate=0.01,
        div_yield=0.02,
        expiry=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_nan(value):
    """Check if a value is NaN."""
    return math.isnan(value)


def assert_releases_gil(operation):
    """Assert that a native operation lets another Python thread make progress.

    A long switch interval prevents the assertion itself from handing the GIL
    to the ticker between the native call's return and the counter snapshot.
    Operations passed here should perform enough native work to give the
    already-waiting ticker thread a scheduling opportunity.
    """

    started = threading.Event()
    stop = threading.Event()
    ticks = 0

    def ticker():
        nonlocal ticks
        started.set()
        while not stop.is_set():
            ticks += 1

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.05)
    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        assert started.wait(timeout=2.0)
        # Let the ticker consume one complete timeslice, then snapshot it from
        # the calling thread immediately before entering the native function.
        time.sleep(0.06)
        before = ticks
        result = operation()
        progressed = ticks - before
    finally:
        stop.set()
        thread.join(timeout=2.0)
        sys.setswitchinterval(previous_interval)

    assert progressed > 100, "native operation held the GIL for its complete execution"
    return result
