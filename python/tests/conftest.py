"""Shared fixtures and constants for openferric Python tests."""

import math
import pytest


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

REL_TOL = 1e-4       # relative tolerance for pricing values
ABS_TOL = 1e-8       # absolute tolerance for near-zero values
LOOSE_TOL = 1e-2     # looser tolerance for MC / FFT comparisons


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
        spot=100.0, v0=0.04, rho=-0.5, sigma_v=1.0,
        kappa=4.0, theta=0.25, rate=0.01, div_yield=0.02, expiry=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_nan(value):
    """Check if a value is NaN."""
    return math.isnan(value)
