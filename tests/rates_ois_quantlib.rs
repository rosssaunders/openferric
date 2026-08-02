//! OIS (Overnight Index Swap) pricing reference tests derived from QuantLib's
//! overnightindexedswap.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/overnightindexedswap.cpp — testCachedValue
//!
//! Our API uses annual periods with continuous-rate projection and OIS
//! discounting. QuantLib uses daily compounding with business-day calendars,
//! so its dated cached NPV is not a like-for-like target.  The tests below pin
//! this API's prices to exact independently assembled cashflow sums; QuantLib's
//! suite supplies the product/convention scenarios and structural properties.

use approx::assert_relative_eq;

use openferric::rates::{BasisSwap, OvernightIndexSwap, YieldCurve};

/// Build a flat continuous yield curve.
fn flat_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let n = max_tenor.ceil() as usize + 1;
    let points: Vec<(f64, f64)> = (1..=n)
        .map(|i| {
            let t = i as f64;
            (t, (-rate * t).exp())
        })
        .collect();
    YieldCurve::new(points)
}

// ── Cached value test ───────────────────────────────────────────────────────

/// Reference: QuantLib overnightindexedswap.cpp testCachedValue.
/// Setup: flat 5% (continuous) OIS curve, 1Y tenor, notional 100.
///
/// The float leg compounds the overnight rate over each annual period,
/// which on a flat continuous curve pays the simple annual equivalent
/// e^r - 1, so that is the at-par fixed rate (NPV = 0).
#[test]
fn ois_cached_value_flat_curve_at_par() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let ois = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: rate.exp() - 1.0,
        float_spread: 0.0,
        tenor: 1.0,
    };

    let fixed_pv = ois.fixed_leg_pv(&curve);
    let floating_pv = ois.floating_leg_pv(&curve, &curve);
    let expected_leg = 100.0 * (rate.exp() - 1.0) * (-rate).exp();
    assert_relative_eq!(fixed_pv, expected_leg, epsilon = 1.0e-12);
    assert_relative_eq!(floating_pv, expected_leg, epsilon = 1.0e-12);

    // NPV (pay fixed) is exactly zero at the simple annual par rate
    let npv = ois.npv(&curve, &curve, true);
    assert_relative_eq!(npv, 0.0, epsilon = 1.0e-10,);
}

/// Reference: QuantLib overnightindexedswap.cpp testCachedValue.
/// 1Y OIS, notional 100, 5% flat, NPV = 0.001730450147 (QuantLib cached).
/// OpenFerric's annual-period model has the exact closed-form value below;
/// the dated QuantLib cached value is not used as a loose substitute oracle.
#[test]
fn ois_off_market_npv() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    // Off-market: fixed rate slightly below curve rate
    let ois = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: 0.049,
        float_spread: 0.0,
        tenor: 1.0,
    };

    let npv = ois.npv(&curve, &curve, true);
    let expected = 100.0 * ((rate.exp() - 1.0) - 0.049) * (-rate).exp();
    assert_relative_eq!(npv, expected, epsilon = 1.0e-12);
    assert_relative_eq!(npv, 0.216_033_369_875_107_6, epsilon = 1.0e-12);

    let npv_recv = ois.npv(&curve, &curve, false);
    assert_relative_eq!(npv, -npv_recv, epsilon = 1.0e-10);
}

// ── Par rate ────────────────────────────────────────────────────────────────

/// Par fixed rate on a flat continuous curve equals the simple annual
/// equivalent e^r - 1 (the compounded overnight rate over unit periods).
#[test]
fn ois_par_rate_flat_curve() {
    let rate = 0.05;
    let curve = flat_curve(rate, 5.0);

    for tenor in [1.0, 2.0, 3.0, 5.0] {
        let ois = OvernightIndexSwap {
            notional: 1_000_000.0,
            fixed_rate: 0.0, // doesn't matter for par rate calc
            float_spread: 0.0,
            tenor,
        };

        let par = ois.par_fixed_rate(&curve, &curve);
        assert_relative_eq!(par, rate.exp() - 1.0, epsilon = 1.0e-10,);
    }
}

/// Par rate with a spread: the par fixed rate should shift up by the spread.
#[test]
fn ois_par_rate_with_spread() {
    let rate = 0.05;
    let spread = 0.001; // 10 bps
    let curve = flat_curve(rate, 3.0);

    let ois_no_spread = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: 0.0,
        float_spread: 0.0,
        tenor: 1.0,
    };

    let ois_with_spread = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: 0.0,
        float_spread: spread,
        tenor: 1.0,
    };

    let par_no_spread = ois_no_spread.par_fixed_rate(&curve, &curve);
    let par_with_spread = ois_with_spread.par_fixed_rate(&curve, &curve);

    // Both legs share the same annual discounted accruals, so the shift is exact.
    assert_relative_eq!(par_with_spread - par_no_spread, spread, epsilon = 1.0e-12);
}

// ── Notional and tenor dependence ───────────────────────────────────────────

/// NPV scales linearly with notional.
#[test]
fn ois_npv_scales_with_notional() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let ois1 = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: 0.04,
        float_spread: 0.0,
        tenor: 1.0,
    };

    let ois2 = OvernightIndexSwap {
        notional: 200.0,
        ..ois1
    };

    let npv1 = ois1.npv(&curve, &curve, true);
    let npv2 = ois2.npv(&curve, &curve, true);

    assert_relative_eq!(npv2, 2.0 * npv1, epsilon = 1.0e-10);
}

/// Fixed and floating leg PVs increase with tenor.
#[test]
fn ois_leg_pvs_increase_with_tenor() {
    let rate = 0.05;
    let curve = flat_curve(rate, 12.0);

    let tenors = [1.0, 2.0, 5.0, 10.0];
    let mut prev_fixed = 0.0;
    let mut prev_float = 0.0;

    for &tenor in &tenors {
        let ois = OvernightIndexSwap {
            notional: 100.0,
            fixed_rate: 0.05,
            float_spread: 0.0,
            tenor,
        };

        let fixed = ois.fixed_leg_pv(&curve);
        let floating = ois.floating_leg_pv(&curve, &curve);

        assert!(fixed > prev_fixed, "Fixed PV must increase with tenor");
        assert!(floating > prev_float, "Float PV must increase with tenor");

        prev_fixed = fixed;
        prev_float = floating;
    }
}

// ── Dual-curve pricing ──────────────────────────────────────────────────────

/// When projection curve > discount curve, floating leg is worth more,
/// so pay-fixed NPV > 0 for at-par fixed rate on discount curve.
#[test]
fn ois_dual_curve_projection_above_discount() {
    let disc_rate = 0.04;
    let proj_rate = 0.05;
    let disc_curve = flat_curve(disc_rate, 5.0);
    let proj_curve = flat_curve(proj_rate, 5.0);

    let ois = OvernightIndexSwap {
        notional: 1_000_000.0,
        fixed_rate: disc_rate,
        float_spread: 0.0,
        tenor: 5.0,
    };

    let npv = ois.npv(&disc_curve, &proj_curve, true);
    let expected: f64 = (1..=5)
        .map(|year| {
            ois.notional
                * (proj_rate.exp() - 1.0 - ois.fixed_rate)
                * (-disc_rate * year as f64).exp()
        })
        .sum();
    assert_relative_eq!(npv, expected, epsilon = 1.0e-9);
    assert_relative_eq!(npv, 50_062.837_387_888_765, epsilon = 1.0e-8);
}

// ── Edge cases ──────────────────────────────────────────────────────────────

/// Zero notional → zero PV.
#[test]
fn ois_zero_notional_returns_zero() {
    let curve = flat_curve(0.05, 3.0);
    let ois = OvernightIndexSwap {
        notional: 0.0,
        fixed_rate: 0.05,
        float_spread: 0.0,
        tenor: 1.0,
    };

    assert_eq!(ois.fixed_leg_pv(&curve), 0.0);
    assert_eq!(ois.floating_leg_pv(&curve, &curve), 0.0);
    assert_eq!(ois.npv(&curve, &curve, true), 0.0);
}

/// Negative tenor → zero PV.
#[test]
fn ois_negative_tenor_returns_zero() {
    let curve = flat_curve(0.05, 3.0);
    let ois = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: 0.05,
        float_spread: 0.0,
        tenor: -1.0,
    };

    assert_eq!(ois.fixed_leg_pv(&curve), 0.0);
    assert_eq!(ois.floating_leg_pv(&curve, &curve), 0.0);
}

#[test]
fn basis_swap_par_spread_reprices_to_zero() {
    // Source context: QuantLib overnightindexedswap.cpp fair-spread tests.
    // This simplified basis-swap model uses the same zero-NPV fair-spread invariant.
    let discount_curve = flat_curve(0.03, 5.0);
    let short_curve = flat_curve(0.04, 5.0);
    let long_curve = flat_curve(0.05, 5.0);
    let swap = BasisSwap {
        notional: 10_000_000.0,
        spread_on_short_leg: 0.0,
        tenor: 3.0,
        short_leg_payments_per_year: 4,
        long_leg_payments_per_year: 2,
    };

    let par_spread = swap.par_spread_on_short_leg(&discount_curve, &short_curve, &long_curve);
    let par_swap = BasisSwap {
        spread_on_short_leg: par_spread,
        ..swap
    };

    let short_leg: f64 = (1..=12)
        .map(|i| 10_000_000.0 * (0.04_f64 * 0.25).exp_m1() * (-0.03 * i as f64 * 0.25).exp())
        .sum();
    let long_leg: f64 = (1..=6)
        .map(|i| 10_000_000.0 * (0.05_f64 * 0.5).exp_m1() * (-0.03 * i as f64 * 0.5).exp())
        .sum();
    let spread_pv01: f64 = (1..=12)
        .map(|i| 10_000_000.0 * 0.25 * (-0.03 * i as f64 * 0.25).exp())
        .sum();
    let expected_par_spread = (long_leg - short_leg) / spread_pv01;
    assert_relative_eq!(par_spread, expected_par_spread, epsilon = 1.0e-12);
    assert_relative_eq!(par_spread, 0.010_239_710_198_232_398, epsilon = 1.0e-12);
    let par_npv = par_swap.npv(&discount_curve, &short_curve, &long_curve, true);
    // The par NPV subtracts two independently accumulated legs.  Bound that
    // cancellation by their scale instead of admitting a fixed currency band.
    let cancellation_roundoff = 64.0 * f64::EPSILON * short_leg.abs().max(long_leg.abs()).max(1.0);
    assert!(
        par_npv.abs() <= cancellation_roundoff,
        "basis-swap par residual {par_npv:e} exceeds cancellation budget {cancellation_roundoff:e}"
    );
}

#[test]
fn basis_swap_pay_receive_orientation_reverses_sign() {
    let discount_curve = flat_curve(0.03, 5.0);
    let short_curve = flat_curve(0.04, 5.0);
    let long_curve = flat_curve(0.05, 5.0);
    let swap = BasisSwap {
        notional: 1_000_000.0,
        spread_on_short_leg: 0.001,
        tenor: 2.5,
        short_leg_payments_per_year: 4,
        long_leg_payments_per_year: 2,
    };

    let pay_short = swap.npv(&discount_curve, &short_curve, &long_curve, true);
    let receive_short = swap.npv(&discount_curve, &short_curve, &long_curve, false);

    let short_leg: f64 = (1..=10)
        .map(|i| {
            let accrual = 0.25;
            let forward = (0.04_f64 * accrual).exp_m1() / accrual;
            swap.notional
                * (forward + swap.spread_on_short_leg)
                * accrual
                * (-0.03 * i as f64 * accrual).exp()
        })
        .sum();
    let long_leg: f64 = (1..=5)
        .map(|i| {
            let accrual = 0.5;
            let forward = (0.05_f64 * accrual).exp_m1() / accrual;
            swap.notional * forward * accrual * (-0.03 * i as f64 * accrual).exp()
        })
        .sum();
    assert_relative_eq!(pay_short, long_leg - short_leg, epsilon = 1.0e-9);
    assert_relative_eq!(pay_short, 22_170.958_869_815_862, epsilon = 1.0e-8);
    assert_relative_eq!(pay_short, -receive_short, epsilon = 1.0e-9);
}

#[test]
fn basis_swap_invalid_inputs_return_zero_or_nan() {
    let curve = flat_curve(0.03, 2.0);
    let zero_notional = BasisSwap {
        notional: 0.0,
        spread_on_short_leg: 0.0,
        tenor: 1.0,
        short_leg_payments_per_year: 4,
        long_leg_payments_per_year: 2,
    };
    let zero_frequency = BasisSwap {
        notional: 1_000_000.0,
        spread_on_short_leg: 0.0,
        tenor: 1.0,
        short_leg_payments_per_year: 0,
        long_leg_payments_per_year: 2,
    };

    assert_eq!(zero_notional.npv(&curve, &curve, &curve, true), 0.0);
    assert!(
        zero_notional
            .par_spread_on_short_leg(&curve, &curve, &curve)
            .is_nan()
    );
    assert!(zero_frequency.npv(&curve, &curve, &curve, true) > 0.0);
    assert!(
        zero_frequency
            .par_spread_on_short_leg(&curve, &curve, &curve)
            .is_nan()
    );
}
