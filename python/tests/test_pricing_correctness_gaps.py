"""Price-level binding tests, sharing independent references with Rust tests."""

import math

import openferric as of
import pytest


def curve(rate=0.03):
    return of.YieldCurve([(30, math.exp(-rate * 30))])


def test_exact_short_rate_paths_and_invalid_observation_grids():
    model = of.HullWhite(0.1, 0)
    times = [0, 0.13, 0.8, 1.71]
    path = model.simulate_path(curve(), times, 71)
    assert path.times == times
    assert path.discount_factors == pytest.approx([math.exp(-0.03 * time) for time in times], abs=1e-15, rel=0)
    assert of.models.HullWhitePath.from_dict(path.to_dict()).to_dict() == path.to_dict()
    with pytest.raises(ValueError):
        model.simulate_path(curve(), [0.5, 0.5], 71)


def test_heston_full_truncation_keeps_auxiliary_state():
    model = of.Heston(0.03, 2, 0.04, 0.7, -0.6, 0.001)
    spot, auxiliary = model.step_full_truncation(100, 0.001, 1 / 252, -15, 1.2)
    assert auxiliary < 0
    next_spot, next_auxiliary = model.step_full_truncation(spot, auxiliary, 1 / 252, 99, -99)
    assert next_auxiliary == pytest.approx(auxiliary + 0.08 / 252, abs=1e-16, rel=0)
    assert next_spot == pytest.approx(spot * math.exp(0.03 / 252), abs=1e-13, rel=0)


def test_hull_white_calibration_reprices_quantlib_quotes():
    quotes = [
        of.SwaptionVolQuote("1x1", 1, 1, 0.010679657425268224),
        of.SwaptionVolQuote("1x10", 1, 10, 0.00839433699169527),
        of.SwaptionVolQuote("5x1", 5, 1, 0.00954011021686305),
        of.SwaptionVolQuote("5x10", 5, 10, 0.007492784574779565),
    ]
    result = of.HullWhiteCalibrator(curve()).calibrate(quotes)
    assert result.params.a == pytest.approx(0.06, abs=3.3e-8, rel=0)
    assert result.params.sigma == pytest.approx(0.011, abs=1.3e-9, rel=0)
    for quote in quotes:
        probe = of.Swaption(100, 0, quote.expiry, quote.tenor, True)
        swaption = of.Swaption(100, probe.forward_swap_rate(curve()), quote.expiry, quote.tenor, True)
        expected = 100 * swaption.annuity_factor(curve()) * quote.market_vol * math.sqrt(quote.expiry / (2 * math.pi))
        assert swaption.price_hull_white(curve(), of.HullWhite(result.params.a, result.params.sigma)) == pytest.approx(
            expected, abs=2e-8, rel=0
        )


def test_negative_rate_stub_swaption_matches_independent_gaussian_integral():
    swaption = of.Swaption(100, -0.02, 1, 2.5, True)
    assert swaption.price_hull_white(curve(-0.015), of.HullWhite(0.1, 0.01)) == pytest.approx(
        1.682397693379852, abs=2e-12, rel=0
    )


def test_callable_daily_range_and_quantlib_bond_option_prices():
    schedule = [of.CouponPeriod(0, 2, 2, of.CouponType.fixed(0.05))]
    note = of.CallableRateNote(100, 100, 100, 2, of.ExerciseSchedule([1], 0), schedule)
    assert note.price_hull_white_tree(of.HullWhite(0.1, 0.02), curve(), 1200) == pytest.approx(
        97.04448357435724, abs=1e-4, rel=0
    )
    daily = of.CallableRangeAccrualNote(
        100, 1, of.Frequency.annual(), 0.08, 0.01, 0.02, 0.045, 1e12, of.ExerciseSchedule([1], 0)
    )
    assert daily.price_hull_white_tree(of.HullWhite(0.1, 0.02), curve(), 1460) == pytest.approx(
        102.587013455667, abs=0.01, rel=0
    )


def test_seasoned_rate_notes_preserve_fixings_target_and_recursive_coupon():
    schedule = of.CouponScheduleBuilder(0, 2, of.Frequency.semi_annual()).build_floating()
    history = of.RateNoteHistory(valuation_time=0.75, fixings=[(0, 0.06), (0.5, 0.06)])
    tarn = of.TargetRedemptionNote(100, 100, 5, 0, 0, None, schedule)
    result = tarn.price_hull_white_mc(of.HullWhite(0.1, 0.02), curve(), history, 100, 7)
    assert result.price == pytest.approx(102 * math.exp(-0.03 * 0.25), abs=2e-12, rel=0)
    assert result.stderr == 0
    snowball = of.SnowballNote(100, 100, 0.08, 0.01, 0, None, schedule)
    snow_history = of.RateNoteHistory(valuation_time=0.75, fixings=[(0, 0.06), (0.5, 0.03)])
    expected = 0.5 * math.exp(-0.03 * 0.25) + 100 * math.exp(-0.03 * 1.25)
    assert snowball.price_hull_white_mc(of.HullWhite(0.1, 0), curve(), snow_history, 2, 7).price == pytest.approx(
        expected, abs=2e-12, rel=0
    )
    with pytest.raises(ValueError, match="missing historical fixing"):
        tarn.price_hull_white_mc(
            of.HullWhite(0.1, 0), curve(), of.RateNoteHistory(valuation_time=0.75, fixings=[]), 2, 7
        )


def test_stochastic_rate_notes_match_independent_sobol_prices():
    schedule = of.CouponScheduleBuilder(0, 2, of.Frequency.semi_annual()).build_floating()
    notes = [
        (of.TargetRedemptionNote(100, 100, 5, 0.01, 0, 0.12, schedule), 100.6501593058501),
        (of.SnowballNote(100, 100, 0.06, 0.02, 0, 0.12, schedule), 102.05819173482063),
    ]
    history = of.RateNoteHistory(valuation_time=0, fixings=[])
    for note, reference in notes:
        result = note.price_hull_white_mc(of.HullWhite(0.1, 0.03), curve(), history, 100_000, 77)
        assert abs(result.price - reference) < 4 * result.stderr + 5e-6


def test_seasoned_callable_coupon_survives_call_at_valuation():
    schedule = of.CouponScheduleBuilder(0, 1, of.Frequency.annual()).build_floating(0.01)
    note = of.CallableRateNote(100, 100, 90, 1, of.ExerciseSchedule([0.5], 0), schedule)
    history = of.RateNoteHistory(valuation_time=0.5, fixings=[(0, 0.04)])
    assert note.price_hull_white_tree_with_history(of.HullWhite(0.1, 0.02), curve(), 200, history) == pytest.approx(
        90 + 5 * math.exp(-0.015), abs=1e-9, rel=0
    )


def test_fra_settlement_and_known_fixing():
    fra = of.ForwardRateAgreement(1e6, 0.04, "2025-01-01", "2025-07-01", of.DayCountConvention.act360(), "2025-04-01")
    expected = 1e6 * 0.02 * 181 / 360 * math.exp(-0.03 * 91 / 365)
    assert fra.npv_with_fixing(curve(), 0.06, False, False) == pytest.approx(expected, abs=1e-10, rel=0)
    assert fra.npv_with_fixing(curve(), None, True, False) == 0
    with pytest.raises(ValueError, match="fixing"):
        fra.npv_with_fixing(curve(), None, False, False)
    zero_period = of.ForwardRateAgreement(
        1e6, 0.04, "2025-01-01", "2025-01-01", of.DayCountConvention.act360(), "2025-01-01"
    )
    assert zero_period.npv_with_fixing(curve(), None, True, True) == 0


def test_historical_knock_in_price_and_greeks_are_vanilla():
    market = of.Market.builder().spot(100).rate(0.03).flat_vol(0.2).build()
    barrier = of.BarrierOption.builder().put().strike(100).expiry(1).down_and_in(80).build()
    actual = barrier.price_with_history(market, True)
    expected = of.BlackScholesEngine().price(of.VanillaOption.european_put(100, 1), market)
    assert actual.price == expected.price
    for name in ["delta", "gamma", "vega", "theta", "rho"]:
        assert getattr(actual.greeks, name) == getattr(expected.greeks, name)


def test_stochastic_mbs_oas_and_explicit_deterministic_spread():
    pool = of.MbsPassThrough(100, 0.06, 0.005, 24, 0, of.PrepaymentModel("constant_cpr", of.ConstantCpr(0)))
    config = of.MbsHullWhiteConfig(
        num_paths=100, seed=21, refinancing_tenor=5, refinancing_spread=0.02, refinancing_floor=0.001
    )
    model = of.HullWhite(0.1, 0.03)
    prepayment = of.RateIncentivePrepayment()
    result = pool.price_hull_white_mc(prepayment, model, curve(), config, 0.004)
    assert result.stderr > 0
    assert pool.oas_hull_white(result.price, prepayment, model, curve(), config) == pytest.approx(
        0.004, abs=1e-12, rel=0
    )
    assert pool.z_spread(pool.price(0.04), [0.03]) == pytest.approx(0.01, abs=1e-12, rel=0)


def test_stochastic_mbs_matches_independent_payment_measure_quadrature():
    pool = of.MbsPassThrough(100, 0.06, 0.005, 23, 20, of.PrepaymentModel("constant_cpr", of.ConstantCpr(0)))
    config = of.MbsHullWhiteConfig(
        num_paths=80_000, seed=21, refinancing_tenor=5, refinancing_spread=0.02, refinancing_floor=0.001
    )
    result = pool.price_hull_white_mc(of.RateIncentivePrepayment(), of.HullWhite(0.1, 0.08), curve(), config, 0)
    assert abs(result.price - 100.40734088342208) < 4 * result.stderr + 3e-11


def test_coterminal_and_rolling_swaptions_are_distinct_contracts():
    swaption = of.Swaption(1e6, 0.04, 1, 5, True)
    engine = of.BermudanSwaptionEngine(of.HullWhite(0.05, 0.01), 300)
    coterminal = engine.price(swaption, [1, 2, 3], curve(0.05))
    rolling = engine.price_rolling_tenor(swaption, [1, 2, 3], curve(0.05))
    assert math.isfinite(coterminal)
    assert rolling - coterminal > 1000
