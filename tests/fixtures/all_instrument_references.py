"""Regenerate additional audit references without importing OpenFerric.

Run with Python and SciPy. Cashflow, accrual, probability-mass and curve-fit
identities are assembled directly in the Rust audit tests; this file records
the independent Gaussian-tail calculation used by audit_equity_extended.
"""

import math

from scipy.special import ndtr


def power_put_reference():
    spot, strike, rate, dividend, volatility, exponent, expiry = (
        100.0, 1000.0, 0.03, 0.01, 0.2, 2.0, 1.0
    )
    prepaid = spot**exponent * math.exp(
        ((exponent - 1.0) * (rate + 0.5 * exponent * volatility**2) - exponent * dividend)
        * expiry
    )
    discounted_strike = strike * math.exp(-rate * expiry)
    deviation = exponent * volatility * math.sqrt(expiry)
    first = math.log(prepaid / discounted_strike) / deviation + 0.5 * deviation
    second = first - deviation
    return discounted_strike * ndtr(-second) - prepaid * ndtr(-first)


if __name__ == "__main__":
    print(f"Power put: {power_put_reference():.17g}")
