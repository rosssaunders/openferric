"""Regenerate audit_pricing_boundaries references with SciPy, not OpenFerric.

Run with Python and SciPy installed. Output includes QUADPACK's integration
error estimates; the Rust suite consumes cached values and needs no Python.
"""

import json
import math

import scipy
from scipy.integrate import quad
from scipy.stats import norm


def single_fixing_references():
    references = []
    for rate in [-0.05, 0.2]:
        for sign in [1, -1]:
            for fixing in [0.0, 0.5, 1.5]:
                forward = 100.0 * math.exp((rate - 0.02) * fixing)
                if fixing == 0.0:
                    payoff = max(sign * (forward - 95.0), 0.0)
                else:
                    width = 0.3 * math.sqrt(fixing)
                    first = math.log(forward / 95.0) / width + 0.5 * width
                    payoff = sign * (
                        forward * norm.cdf(sign * first)
                        - 95.0 * norm.cdf(sign * (first - width))
                    )
                references.append((rate, sign, fixing, math.exp(-rate * 1.5) * payoff))
    return references


def arithmetic_references():
    references = []
    for fixing in [0.25, 0.37]:
        for sign in [1, -1]:

            def integrand(driver):
                middle = 100.0 * math.exp(
                    (0.1 - 0.02 - 0.5 * 0.3**2) * fixing
                    + 0.3 * math.sqrt(fixing) * driver
                )
                forward = middle * math.exp((0.1 - 0.02) * (1.0 - fixing))
                strike = 300.0 - 100.0 - middle
                if strike <= 0.0:
                    payoff = forward - strike if sign == 1 else 0.0
                else:
                    width = 0.3 * math.sqrt(1.0 - fixing)
                    first = math.log(forward / strike) / width + 0.5 * width
                    payoff = sign * (
                        forward * norm.cdf(sign * first)
                        - strike * norm.cdf(sign * (first - width))
                    )
                return math.exp(-0.1) * payoff / 3.0 * norm.pdf(driver)

            price, error = quad(integrand, -12.0, 12.0, epsabs=1e-11, epsrel=1e-11, limit=400)
            references.append((fixing, sign, price, error))
    return references


def double_barrier_references():
    lower, upper, spot = 80.0, 120.0, 100.0
    rate, dividend, volatility, expiry = 0.03, 0.01, 0.25, 1.0
    width = math.log(upper / lower)
    position = math.log(spot / lower)
    drift = rate - dividend - 0.5 * volatility**2

    def killed_density(level):
        series = sum(
            math.sin(mode * math.pi * position / width)
            * math.sin(mode * math.pi * level / width)
            * math.exp(-0.5 * volatility**2 * (mode * math.pi / width)**2 * expiry)
            for mode in range(1, 65)
        )
        return 2.0 / width * math.exp(
            drift / volatility**2 * (level - position)
            - drift**2 * expiry / (2.0 * volatility**2)
        ) * series

    references = []
    for sign in [1, -1]:
        for strike in [60.0, 80.0, 100.0, 120.0, 140.0]:

            def integrand(level):
                terminal = lower * math.exp(level)
                return (
                    math.exp(-rate * expiry)
                    * max(sign * (terminal - strike), 0.0)
                    * killed_density(level)
                )

            points = [math.log(strike / lower)] if lower < strike < upper else None
            price, error = quad(
                integrand, 0.0, width, points=points, epsabs=1e-12, epsrel=1e-12
            )
            references.append((sign, strike, price, error))
    return references


if __name__ == "__main__":
    print(json.dumps({
        "scipy_version": scipy.__version__,
        "single_fixing": single_fixing_references(),
        "arithmetic": arithmetic_references(),
        "double_barrier": double_barrier_references(),
    }, indent=2))
