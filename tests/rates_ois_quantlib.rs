//! OIS (Overnight Index Swap) pricing reference tests derived from QuantLib's
//! overnightindexedswap.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/overnightindexedswap.cpp — testCachedValue
//!
//! Our API uses annual periods with continuous-rate projection and OIS
//! discounting. QuantLib uses daily compounding with business-day calendars,
//! so exact NPV values differ. We verify structural properties and
//! approximate values.

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
/// Setup: flat 5% OIS curve, 1Y tenor, notional 100, fixed rate 5%.
///
/// When discount = projection curve and fixed_rate = curve rate, the
/// OIS should be approximately at-par (NPV ≈ 0).
#[test]
fn ois_cached_value_flat_curve_at_par() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let ois = OvernightIndexSwap {
        notional: 100.0,
        fixed_rate: rate,
        float_spread: 0.0,
        tenor: 1.0,
    };

    // Fixed and floating legs should approximately match on a flat curve
    // where fixed_rate equals the curve rate
    let fixed_pv = ois.fixed_leg_pv(&curve);
    let floating_pv = ois.floating_leg_pv(&curve, &curve);

    // Both PVs should be positive
    assert!(fixed_pv > 0.0, "Fixed leg PV must be positive: {fixed_pv}");
    assert!(
        floating_pv > 0.0,
        "Floating leg PV must be positive: {floating_pv}"
    );

    // NPV (pay fixed) should be approximately zero for at-par swap
    let npv = ois.npv(&curve, &curve, true);
    assert_relative_eq!(npv, 0.0, epsilon = 0.01,);
}

/// Reference: QuantLib overnightindexedswap.cpp testCachedValue.
/// 1Y OIS, notional 100, 5% flat, NPV = 0.001730450147 (QuantLib cached).
/// Our simplified annual model won't match exactly, but the sign and
/// magnitude should be consistent.
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

    // Pay-fixed NPV should be positive when fixed_rate < forward rate
    let npv = ois.npv(&curve, &curve, true);
    assert!(
        npv > 0.0,
        "Pay-fixed NPV should be positive when fixed < floating: {npv}"
    );

    // And conversely, receive-fixed NPV should be negative
    let npv_recv = ois.npv(&curve, &curve, false);
    assert!(npv_recv < 0.0);
    assert_relative_eq!(npv, -npv_recv, epsilon = 1.0e-10);
}

// ── Par rate ────────────────────────────────────────────────────────────────

/// Par fixed rate on a flat curve should approximately equal the curve rate.
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
        assert_relative_eq!(par, rate, epsilon = 1.0e-6,);
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

    // Spread on float leg should increase par fixed rate by approximately
    // the same amount
    assert_relative_eq!(par_with_spread - par_no_spread, spread, epsilon = 1.0e-6,);
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

    // Pay-fixed at discount rate, floating projects higher → NPV > 0
    let npv = ois.npv(&disc_curve, &proj_curve, true);
    assert!(
        npv > 0.0,
        "Pay-fixed NPV should be positive when projection > discount: {npv}"
    );
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

    assert!(par_spread > 0.0);
    assert_relative_eq!(
        par_swap.npv(&discount_curve, &short_curve, &long_curve, true),
        0.0,
        epsilon = 1.0e-5
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

    assert!(pay_short.is_finite());
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
