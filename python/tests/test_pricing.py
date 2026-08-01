"""Tests for the 10 pricing functions in openferric."""

import math

import numpy as np
import pytest
from conftest import ABS_TOL, assert_releases_gil, is_nan
from openferric import (
    AnalyticEngine,
    AsianStrike,
    Autocallable,
    bs_price_batch,
    py_american_price,
    py_arithmetic_asian_price_mc,
    py_barrier_price,
    py_black76_price,
    py_bs_greeks,
    py_bs_price,
    py_digital_price,
    py_fx_price,
    py_heston_price,
    py_lookback_fixed,
    py_lookback_floating,
    py_price_autocallable,
    py_spread_price,
)


def test_free_monte_carlo_pricer_releases_gil():
    result = assert_releases_gil(
        lambda: py_arithmetic_asian_price_mc(
            AsianStrike.fixed(100.0),
            spot=100.0,
            rate=0.05,
            vol=0.2,
            expiry=1.0,
            steps=32,
            num_paths=200_000,
            seed=42,
            option_type="call",
        )
    )
    assert all(math.isfinite(value) for value in result)


# =========================================================================
# 1. py_bs_price
# =========================================================================


class TestBsPrice:
    def test_atm_call(self, std_market):
        price = py_bs_price(**std_market, option_type="call")
        # Independent SciPy 1.17.1 norm.cdf evaluation of Black--Scholes.
        assert price == pytest.approx(10.450583572185565, abs=2e-12)

    def test_atm_put(self, std_market):
        price = py_bs_price(**std_market, option_type="put")
        assert price == pytest.approx(5.573526022256971, abs=2e-12)

    def test_put_call_parity(self, std_market):
        """C - P = S - K * exp(-rT)."""
        call = py_bs_price(**std_market, option_type="call")
        put = py_bs_price(**std_market, option_type="put")
        s, k, r, t = std_market["spot"], std_market["strike"], std_market["rate"], std_market["expiry"]
        parity = call - put - (s - k * math.exp(-r * t))
        assert parity == pytest.approx(0.0, abs=ABS_TOL)

    def test_deep_itm_call(self):
        price = py_bs_price(spot=100.0, strike=50.0, expiry=1.0, vol=0.20, rate=0.05, option_type="call")
        assert price == pytest.approx(52.438862117161854, abs=2e-12)

    def test_deep_otm_put(self):
        price = py_bs_price(spot=100.0, strike=50.0, expiry=1.0, vol=0.20, rate=0.05, option_type="put")
        assert price == pytest.approx(0.000333342197556874, abs=2e-12)

    def test_invalid_option_type(self, std_market):
        assert is_nan(py_bs_price(**std_market, option_type="straddle"))

    def test_numpy_batch_matches_scalar(self):
        spots = np.array([80.0, 100.0, 120.0], dtype=np.float64)
        strikes = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        actual = bs_price_batch(spots, strikes, 0.05, 0.0, 0.2, 1.0, True)
        expected = np.array([py_bs_price(float(spot), 100.0, 1.0, 0.2, 0.05, "call") for spot in spots])
        # The vector path uses the fast A&S CDF approximation. Its documented
        # CDF error propagates to a price-level absolute bound below 5e-5 for
        # this grid, while the scalar binding uses the high-accuracy CDF.
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-5)

    def test_numpy_batch_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            bs_price_batch(
                np.array([100.0], dtype=np.float64),
                np.array([90.0, 100.0], dtype=np.float64),
                0.05,
                0.0,
                0.2,
                1.0,
                True,
            )

    def test_numpy_batch_accepts_empty_arrays(self):
        values = bs_price_batch(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.05,
            0.0,
            0.2,
            1.0,
            True,
        )
        assert values.dtype == np.float64
        assert values.shape == (0,)

    def test_numpy_batch_rejects_non_contiguous_arrays(self):
        base = np.array([80.0, 90.0, 100.0, 110.0], dtype=np.float64)
        with pytest.raises(ValueError, match="spots must be a contiguous"):
            bs_price_batch(base[::2], np.array([100.0, 100.0]), 0.05, 0.0, 0.2, 1.0, True)
        with pytest.raises(ValueError, match="strikes must be a contiguous"):
            bs_price_batch(np.array([80.0, 100.0]), base[::2], 0.05, 0.0, 0.2, 1.0, True)

    @pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 8, 9, 15, 17])
    def test_numpy_batch_simd_tails_match_scalar(self, length):
        spots = np.linspace(70.0, 130.0, length, dtype=np.float64)
        strikes = np.linspace(95.0, 105.0, length, dtype=np.float64)
        actual = bs_price_batch(spots, strikes, 0.05, 0.0, 0.2, 1.0, True)
        expected = np.array(
            [py_bs_price(float(spot), float(strike), 1.0, 0.2, 0.05, "call") for spot, strike in zip(spots, strikes)]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-5)

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("input_name", ["spots", "strikes"])
    def test_numpy_batch_rejects_non_finite_array_values(self, input_name, bad_value):
        inputs = {
            "spots": np.array([100.0], dtype=np.float64),
            "strikes": np.array([100.0], dtype=np.float64),
        }
        inputs[input_name][0] = bad_value
        with pytest.raises(ValueError, match=input_name):
            bs_price_batch(**inputs, rate=0.05, dividend_yield=0.0, vol=0.2, expiry=1.0, is_call=True)

    @pytest.mark.parametrize(
        ("input_name", "bad_value"),
        [
            ("rate", math.nan),
            ("rate", math.inf),
            ("dividend_yield", math.nan),
            ("dividend_yield", -math.inf),
            ("vol", math.nan),
            ("vol", math.inf),
            ("vol", -0.1),
            ("expiry", math.nan),
            ("expiry", math.inf),
            ("expiry", -0.1),
        ],
    )
    def test_numpy_batch_rejects_invalid_scalar_values(self, input_name, bad_value):
        inputs = dict(
            spots=np.array([100.0], dtype=np.float64),
            strikes=np.array([100.0], dtype=np.float64),
            rate=0.05,
            dividend_yield=0.0,
            vol=0.2,
            expiry=1.0,
            is_call=True,
        )
        inputs[input_name] = bad_value
        with pytest.raises(ValueError, match=input_name):
            bs_price_batch(**inputs)

    def test_numpy_batch_handles_zero_vol_and_expiry_deterministically(self):
        spots = np.array([80.0, 100.0, 120.0])
        strikes = np.array([100.0, 100.0, 100.0])

        zero_expiry = bs_price_batch(spots, strikes, 0.05, 0.02, 0.2, 0.0, True)
        np.testing.assert_array_equal(zero_expiry, np.array([0.0, 0.0, 20.0]))

        zero_vol = bs_price_batch(spots, strikes, 0.05, 0.02, 0.0, 1.0, True)
        expected = np.maximum(spots * math.exp(-0.02) - strikes * math.exp(-0.05), 0.0)
        np.testing.assert_allclose(zero_vol, expected, rtol=0.0, atol=1e-12)

    def test_large_numpy_batch_releases_gil(self):
        length = 2_000_003
        spots = np.linspace(80.0, 120.0, length, dtype=np.float64)
        strikes = np.full(length, 100.0, dtype=np.float64)

        values = assert_releases_gil(lambda: bs_price_batch(spots, strikes, 0.05, 0.0, 0.2, 1.0, True))

        assert values.shape == (length,)
        assert np.isfinite(values[[0, -1]]).all()


class TestLegacyBlack76Price:
    def test_matches_validated_analytic_entry_point(self):
        legacy = py_black76_price(
            forward=103.0,
            strike=100.0,
            expiry=1.25,
            vol=0.24,
            rate=0.03,
            option_type="call",
        )
        validated = AnalyticEngine.black76_price(
            option_type="call",
            forward=103.0,
            strike=100.0,
            rate=0.03,
            vol=0.24,
            expiry=1.25,
        )
        assert legacy == validated

    @pytest.mark.parametrize(
        ("input_name", "bad_value"),
        [
            ("forward", -1.0),
            ("forward", math.nan),
            ("strike", 0.0),
            ("strike", math.inf),
            ("expiry", -1.0),
            ("expiry", math.nan),
            ("vol", -0.1),
            ("vol", math.inf),
            ("rate", math.nan),
            ("rate", math.inf),
        ],
    )
    def test_rejects_invalid_numeric_domain(self, input_name, bad_value):
        inputs = dict(
            forward=100.0,
            strike=100.0,
            expiry=1.0,
            vol=0.2,
            rate=0.03,
            option_type="call",
        )
        inputs[input_name] = bad_value
        with pytest.raises(ValueError):
            py_black76_price(**inputs)

    def test_rejects_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            py_black76_price(100.0, 100.0, 1.0, 0.2, 0.03, "straddle")

    def test_rejects_non_finite_computed_price(self):
        with pytest.raises(ValueError, match="non-finite"):
            py_black76_price(100.0, 100.0, 1.0, 0.2, -1e308, "call")


# =========================================================================
# 2. py_bs_greeks
# =========================================================================


class TestBsGreeks:
    @pytest.fixture
    def greeks_params(self):
        return dict(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0)

    def test_call_delta(self, greeks_params):
        delta = py_bs_greeks(**greeks_params, option_type="call", greek="delta")
        assert delta == pytest.approx(0.6368306511756191, abs=2e-14)

    def test_put_delta(self, greeks_params):
        delta = py_bs_greeks(**greeks_params, option_type="put", greek="delta")
        assert delta == pytest.approx(-0.3631693488243809, abs=2e-14)

    def test_gamma_positive(self, greeks_params):
        gamma = py_bs_greeks(**greeks_params, option_type="call", greek="gamma")
        assert gamma == pytest.approx(0.018762017345846895, abs=2e-15)

    def test_vega_positive(self, greeks_params):
        vega = py_bs_greeks(**greeks_params, option_type="call", greek="vega")
        assert vega == pytest.approx(37.52403469169379, abs=2e-12)

    def test_call_theta_negative(self, greeks_params):
        theta = py_bs_greeks(**greeks_params, option_type="call", greek="theta")
        assert theta == pytest.approx(-6.414027546438197, abs=2e-12)

    def test_vanna(self, greeks_params):
        vanna = py_bs_greeks(**greeks_params, option_type="call", greek="vanna")
        assert vanna == pytest.approx(-0.28143026018770345, abs=2e-14)

    def test_volga(self, greeks_params):
        volga = py_bs_greeks(**greeks_params, option_type="call", greek="volga")
        assert volga == pytest.approx(9.850059106569622, abs=2e-12)

    def test_vomma_alias(self, greeks_params):
        volga = py_bs_greeks(**greeks_params, option_type="call", greek="volga")
        vomma = py_bs_greeks(**greeks_params, option_type="call", greek="vomma")
        assert volga == pytest.approx(vomma, abs=ABS_TOL)

    def test_invalid_greek_name(self, greeks_params):
        assert is_nan(py_bs_greeks(**greeks_params, option_type="call", greek="charm"))

    def test_invalid_option_type(self, greeks_params):
        assert is_nan(py_bs_greeks(**greeks_params, option_type="xxx", greek="delta"))


# =========================================================================
# 3. py_barrier_price
# =========================================================================


class TestBarrierPrice:
    @pytest.fixture
    def barrier_params(self):
        return dict(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0, rebate=0.0)

    def test_up_out_call(self, barrier_params):
        price = py_barrier_price(
            **barrier_params, barrier=120.0, option_type="call", barrier_type="out", barrier_dir="up"
        )
        # QuantLib 1.43 AnalyticBarrierEngine, Actual/360 over 360 days.
        assert price == pytest.approx(1.1760653996503727, abs=2e-11)

    def test_down_in_put(self, barrier_params):
        price = py_barrier_price(
            **barrier_params, barrier=80.0, option_type="put", barrier_type="in", barrier_dir="down"
        )
        assert price == pytest.approx(3.9525105131751994, abs=2e-11)

    def test_in_plus_out_approx_vanilla(self, barrier_params):
        """In + Out ≈ Vanilla (with zero rebate)."""
        in_price = py_barrier_price(
            **barrier_params, barrier=120.0, option_type="call", barrier_type="in", barrier_dir="up"
        )
        out_price = py_barrier_price(
            **barrier_params, barrier=120.0, option_type="call", barrier_type="out", barrier_dir="up"
        )
        vanilla = py_bs_price(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="call")
        assert in_price == pytest.approx(9.274518172535206, abs=2e-11)
        assert out_price == pytest.approx(1.1760653996503727, abs=2e-11)
        assert (in_price + out_price) == pytest.approx(vanilla, abs=2e-12)

    def test_invalid_barrier_type(self, barrier_params):
        assert is_nan(
            py_barrier_price(
                **barrier_params, barrier=120.0, option_type="call", barrier_type="through", barrier_dir="up"
            )
        )

    def test_invalid_barrier_dir(self, barrier_params):
        assert is_nan(
            py_barrier_price(
                **barrier_params, barrier=120.0, option_type="call", barrier_type="out", barrier_dir="left"
            )
        )


# =========================================================================
# 4. py_american_price
# =========================================================================


class TestAmericanPrice:
    def test_american_put_ge_european(self):
        """OpenFerric's 500-step CRR value matches an independent recurrence."""
        am = py_american_price(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="put", steps=500)
        assert am == pytest.approx(6.088810110703037, abs=2e-11)

    def test_american_call_no_div_approx_european(self):
        """The finite 500-step CRR price is asserted, not a percent band."""
        am = py_american_price(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="call", steps=500)
        assert am == pytest.approx(10.44658513644654, abs=2e-11)

    def test_deep_itm_put(self):
        price = py_american_price(
            spot=50.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="put", steps=300
        )
        assert price == pytest.approx(50.0, abs=2e-12)

    def test_invalid_option_type(self):
        assert is_nan(
            py_american_price(
                spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="butterfly", steps=100
            )
        )


# =========================================================================
# 5. py_heston_price
# =========================================================================


class TestHestonPrice:
    def test_call_reference(self, heston_params):
        """Alan Lewis reference: call K=100, expected ≈ 16.070."""
        price = py_heston_price(
            spot=heston_params["spot"],
            strike=100.0,
            expiry=heston_params["expiry"],
            rate=heston_params["rate"],
            div_yield=heston_params["div_yield"],
            option_type="call",
            v0=heston_params["v0"],
            kappa=heston_params["kappa"],
            theta=heston_params["theta"],
            sigma_v=heston_params["sigma_v"],
            rho=heston_params["rho"],
        )
        assert price == pytest.approx(16.070154917028834, abs=5e-12)

    def test_put_reference(self, heston_params):
        """Alan Lewis reference: put K=100, expected ≈ 17.055."""
        price = py_heston_price(
            spot=heston_params["spot"],
            strike=100.0,
            expiry=heston_params["expiry"],
            rate=heston_params["rate"],
            div_yield=heston_params["div_yield"],
            option_type="put",
            v0=heston_params["v0"],
            kappa=heston_params["kappa"],
            theta=heston_params["theta"],
            sigma_v=heston_params["sigma_v"],
            rho=heston_params["rho"],
        )
        assert price == pytest.approx(17.055270961270109, abs=5e-12)

    def test_invalid_option_type(self, heston_params):
        assert is_nan(
            py_heston_price(
                spot=100.0,
                strike=100.0,
                expiry=1.0,
                rate=0.01,
                div_yield=0.0,
                option_type="xxx",
                v0=0.04,
                kappa=4.0,
                theta=0.25,
                sigma_v=1.0,
                rho=-0.5,
            )
        )


# =========================================================================
# 6. py_fx_price
# =========================================================================


class TestFxPrice:
    def test_garman_kohlhagen_call(self):
        price = py_fx_price(
            spot_fx=1.30,
            strike_fx=1.30,
            maturity=0.5,
            vol=0.10,
            domestic_rate=0.05,
            foreign_rate=0.03,
            option_type="call",
        )
        # Independent SciPy evaluation of Garman--Kohlhagen.
        assert price == pytest.approx(0.04266908230609778, abs=2e-14)

    def test_garman_kohlhagen_put(self):
        price = py_fx_price(
            spot_fx=1.30,
            strike_fx=1.30,
            maturity=0.5,
            vol=0.10,
            domestic_rate=0.05,
            foreign_rate=0.03,
            option_type="put",
        )
        assert price == pytest.approx(0.029926446458948597, abs=2e-14)

    def test_fx_put_call_parity(self):
        """C - P = S*exp(-rf*T) - K*exp(-rd*T)."""
        s, k, t, vol, rd, rf = 1.30, 1.30, 0.5, 0.10, 0.05, 0.03
        call = py_fx_price(s, k, t, vol, rd, rf, "call")
        put = py_fx_price(s, k, t, vol, rd, rf, "put")
        parity = call - put - (s * math.exp(-rf * t) - k * math.exp(-rd * t))
        assert parity == pytest.approx(0.0, abs=2e-15)

    def test_invalid_option_type(self):
        assert is_nan(py_fx_price(1.30, 1.30, 0.5, 0.10, 0.05, 0.03, "straddle"))


# =========================================================================
# 7. py_digital_price
# =========================================================================


class TestDigitalPrice:
    @pytest.fixture
    def digital_params(self):
        return dict(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0)

    def test_cash_or_nothing_call(self, digital_params):
        price = py_digital_price(**digital_params, option_type="call", digital_type="cash", cash=1.0)
        assert price == pytest.approx(0.5323248154537634, abs=2e-14)

    def test_cash_or_nothing_put(self, digital_params):
        price = py_digital_price(**digital_params, option_type="put", digital_type="cash", cash=1.0)
        assert price == pytest.approx(0.41890460904695065, abs=2e-14)

    def test_asset_or_nothing_call(self, digital_params):
        price = py_digital_price(**digital_params, option_type="call", digital_type="asset", cash=0.0)
        assert price == pytest.approx(63.68306511756191, abs=2e-12)

    def test_cash_call_plus_put_equals_pv(self, digital_params):
        """Cash-or-nothing call + put equals PV(cash)."""
        call = py_digital_price(**digital_params, option_type="call", digital_type="cash", cash=1.0)
        put = py_digital_price(**digital_params, option_type="put", digital_type="cash", cash=1.0)
        pv = math.exp(-digital_params["rate"] * digital_params["expiry"])
        assert (call + put) == pytest.approx(pv, abs=2e-15)

    def test_invalid_digital_type(self, digital_params):
        assert is_nan(py_digital_price(**digital_params, option_type="call", digital_type="binary", cash=1.0))


# =========================================================================
# 8. py_spread_price
# =========================================================================


class TestSpreadPrice:
    def test_kirk_positive(self):
        price = py_spread_price(
            s1=100.0, s2=96.0, k=3.0, vol1=0.20, vol2=0.15, rho=0.5, q1=0.0, q2=0.0, r=0.05, t=0.5, method="kirk"
        )
        # Independent SciPy 1.17.1 evaluation of the Kirk approximation.
        assert price == pytest.approx(5.577024131239917, abs=2e-12)

    def test_margrabe_exchange(self):
        """Margrabe: exchange option (K=0) on two assets."""
        price = py_spread_price(
            s1=100.0, s2=105.0, k=0.0, vol1=0.20, vol2=0.15, rho=0.5, q1=0.04, q2=0.06, r=0.05, t=1.0, method="margrabe"
        )
        # Independent SciPy 1.17.1 evaluation of Margrabe's formula.
        assert price == pytest.approx(5.687142248583889, abs=2e-12)

    def test_margrabe_equal_assets(self):
        """Margrabe with identical assets → positive but small."""
        price = py_spread_price(
            s1=100.0, s2=100.0, k=0.0, vol1=0.20, vol2=0.20, rho=0.99, q1=0.0, q2=0.0, r=0.05, t=1.0, method="margrabe"
        )
        assert price >= 0.0

    def test_invalid_method(self):
        assert is_nan(py_spread_price(100.0, 90.0, 5.0, 0.20, 0.25, 0.5, 0.0, 0.0, 0.05, 1.0, "monte_carlo"))


# =========================================================================
# 9. py_lookback_floating
# =========================================================================


class TestLookbackFloating:
    def test_floating_call(self):
        price = py_lookback_floating(
            spot=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0, option_type="call", observed_extreme=0.0
        )
        # QuantLib 1.43 AnalyticContinuousFloatingLookbackEngine.
        assert price == pytest.approx(17.216802237360877, abs=3e-11)

    def test_floating_put(self):
        price = py_lookback_floating(
            spot=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0, option_type="put", observed_extreme=0.0
        )
        assert price == pytest.approx(14.29056770740371, abs=3e-11)

    def test_floating_call_ge_vanilla(self):
        """Lookback floating call >= vanilla ATM call."""
        lookback = py_lookback_floating(
            spot=100.0, expiry=1.0, vol=0.20, rate=0.05, div_yield=0.0, option_type="call", observed_extreme=0.0
        )
        vanilla = py_bs_price(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="call")
        assert lookback >= vanilla

    def test_invalid_option_type(self):
        assert is_nan(py_lookback_floating(100.0, 1.0, 0.20, 0.05, 0.0, "xxx", 0.0))


# =========================================================================
# 10. py_lookback_fixed
# =========================================================================


class TestLookbackFixed:
    def test_fixed_call(self):
        price = py_lookback_fixed(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            vol=0.20,
            rate=0.05,
            div_yield=0.0,
            option_type="call",
            observed_extreme=0.0,
        )
        # QuantLib 1.43 AnalyticContinuousFixedLookbackEngine.
        assert price == pytest.approx(19.16762525733232, abs=3e-11)

    def test_fixed_put(self):
        price = py_lookback_fixed(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            vol=0.20,
            rate=0.05,
            div_yield=0.0,
            option_type="put",
            observed_extreme=0.0,
        )
        assert price == pytest.approx(12.339744687432269, abs=3e-11)

    def test_fixed_call_ge_vanilla(self):
        """Lookback fixed call >= vanilla call (same strike)."""
        lookback = py_lookback_fixed(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            vol=0.20,
            rate=0.05,
            div_yield=0.0,
            option_type="call",
            observed_extreme=0.0,
        )
        vanilla = py_bs_price(spot=100.0, strike=100.0, expiry=1.0, vol=0.20, rate=0.05, option_type="call")
        assert lookback >= vanilla

    def test_invalid_option_type(self):
        assert is_nan(py_lookback_fixed(100.0, 100.0, 1.0, 0.20, 0.05, 0.0, "xxx", 0.0))


# =========================================================================
# Digital greeks: vega/rho are RAW units (per unit vol / per unit rate)
# =========================================================================


class TestDigitalGreeksRawUnits:
    @pytest.fixture
    def params(self):
        return dict(
            option_type="call",
            spot=100.0,
            strike=100.0,
            cash=10.0,
            expiry=1.0,
            rate=0.05,
            dividend_yield=0.02,
            vol=0.20,
        )

    def test_vega_matches_scipy_analytic_reference(self, params):
        """Vega is raw per-unit-vol, independently evaluated with SciPy."""
        greeks = AnalyticEngine.digital_cash_price(**params).greeks
        assert greeks is not None

        assert greeks.vega == pytest.approx(-4.737644688752179, abs=3e-12)

    def test_rho_matches_scipy_analytic_reference(self, params):
        """Rho is raw per-unit-rate, independently evaluated with SciPy."""
        greeks = AnalyticEngine.digital_cash_price(**params).greeks
        assert greeks is not None

        assert greeks.rho == pytest.approx(14.004767844476481, abs=3e-12)


# =========================================================================
# py_price_autocallable
# =========================================================================


class TestPriceAutocallable:
    @pytest.fixture
    def autocall(self):
        return Autocallable(
            underlyings=[0],
            notional=1.0,
            autocall_dates=[0.5, 1.0],
            autocall_barrier=1.0,
            coupon_rate=0.05,
            ki_barrier=0.6,
            ki_strike=1.0,
            maturity=1.0,
        )

    @pytest.fixture
    def market_args(self):
        return dict(
            spots=[100.0],
            vols=[0.0],
            corr_matrix=[[1.0]],
            rate=0.03,
            # With q=r the path is constant, so the first observation pays a
            # known contractual cash flow without Monte Carlo uncertainty.
            div_yield=0.03,
            num_paths=1,
            num_steps=50,
        )

    def test_price_is_finite(self, autocall, market_args):
        result = py_price_autocallable(autocall, **market_args)
        expected = 1.0 * (1.0 + 0.05 * 0.5) * math.exp(-0.03 * 0.5)
        assert result.price == pytest.approx(expected, abs=2e-14)
        assert result.stderr == 0.0
        # Default does not compute greeks.
        assert result.greeks is None

    def test_with_greeks_returns_greeks(self, autocall, market_args):
        # An unreachable autocall and certain KI reduce the zero-vol payoff to
        # N * S_T / (2 * S_fix).  The internal fixed-strike spot bump therefore
        # has the exact delta exp(-qT) / (2 * S_fix).
        linear_note = Autocallable(
            underlyings=[0],
            notional=1.0,
            autocall_dates=[1.0],
            autocall_barrier=10.0,
            coupon_rate=0.0,
            ki_barrier=5.0,
            ki_strike=2.0,
            maturity=1.0,
        )
        result = py_price_autocallable(linear_note, **market_args, with_greeks=True)
        expected_price = 0.5 * math.exp(-0.03)
        expected_delta = math.exp(-0.03) / 200.0
        assert result.price == pytest.approx(expected_price, abs=2e-14)
        assert result.greeks is not None
        assert result.greeks.delta == pytest.approx(expected_delta, abs=2e-14)
