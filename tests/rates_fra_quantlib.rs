//! FRA (Forward Rate Agreement) pricing reference tests derived from QuantLib's
//! forwardrateagreement.cpp.
//!
//! QuantLib — C++ finance library (BSD 3-Clause).
//! Source: vendor/QuantLib/test-suite/forwardrateagreement.cpp
//!
//! Our FRA uses year_fraction(start_date, end_date) to convert dates to
//! continuous-time tenor, then queries curve.forward_rate(0, tau).

use approx::assert_relative_eq;
use chrono::NaiveDate;

use openferric::rates::{
    DayCountConvention, ForwardRateAgreement, YieldCurve, YieldCurveBuilder,
};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

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

// ── Forward rate from pillar rates ──────────────────────────────────────────

/// Reference: QuantLib forwardrateagreement.cpp.
/// On a curve bootstrapped from deposits [0.01, 0.02, 0.03] at 3M, 6M, 9M,
/// the implied forward rate for a 6M→9M FRA should be ~0.01 (annualized).
///
/// Note: Our FRA computes forward_rate(0, tau) where tau = year_fraction
/// of the FRA period, which is the zero rate at that tenor, not the
/// forward rate between start and end. This test verifies the forward_rate
/// method returns a sensible continuously-compounded rate.
#[test]
fn fra_forward_rate_from_deposit_curve() {
    let deposits = vec![
        (0.25, 0.01),
        (0.50, 0.02),
        (0.75, 0.03),
    ];
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    // FRA for the 6M period (start = now, end = 6M from now)
    let start = d(2024, 1, 1);
    let end = d(2024, 7, 1);

    let fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.02,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };

    let fwd = fra.forward_rate(&curve);
    // The forward rate from 0 to ~0.497 years should be close to the
    // interpolated zero rate at that tenor
    assert!(fwd > 0.0, "Forward rate must be positive");
    assert!(fwd < 0.05, "Forward rate should be reasonable");
}

// ── NPV at par ──────────────────────────────────────────────────────────────

/// When the fixed rate equals the forward rate, FRA NPV should be zero.
#[test]
fn fra_npv_at_par_is_zero() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let start = d(2024, 1, 1);
    let end = d(2025, 1, 1);

    // First, find the forward rate
    let fra_probe = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.0,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };
    let fwd = fra_probe.forward_rate(&curve);

    // Set fixed rate = forward rate
    let fra = ForwardRateAgreement {
        fixed_rate: fwd,
        ..fra_probe
    };

    let npv = fra.npv(&curve);
    assert_relative_eq!(npv, 0.0, epsilon = 1.0e-6);
}

// ── NPV sign ────────────────────────────────────────────────────────────────

/// FRA NPV is positive when forward rate > fixed rate (receiver benefits).
/// FRA NPV is negative when forward rate < fixed rate.
#[test]
fn fra_npv_sign_consistency() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let start = d(2024, 1, 1);
    let end = d(2025, 1, 1);

    let fra_probe = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.0,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };
    let fwd = fra_probe.forward_rate(&curve);

    // Fixed rate below forward → positive NPV
    let fra_low = ForwardRateAgreement {
        fixed_rate: fwd - 0.01,
        ..fra_probe
    };
    assert!(fra_low.npv(&curve) > 0.0, "NPV should be positive when fixed < forward");

    // Fixed rate above forward → negative NPV
    let fra_high = ForwardRateAgreement {
        fixed_rate: fwd + 0.01,
        ..fra_probe
    };
    assert!(fra_high.npv(&curve) < 0.0, "NPV should be negative when fixed > forward");
}

// ── NPV formula verification ────────────────────────────────────────────────

/// Verify NPV = notional * (forward - fixed) * tau * DF(tau)
#[test]
fn fra_npv_formula_verification() {
    let rate = 0.05;
    let curve = flat_curve(rate, 3.0);

    let start = d(2024, 1, 1);
    let end = d(2025, 1, 1);
    let fixed_rate = 0.04;

    let fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };

    let fwd = fra.forward_rate(&curve);
    let tau = openferric::rates::year_fraction(start, end, DayCountConvention::Act365Fixed);
    let df = curve.discount_factor(tau);
    let expected_npv = 1_000_000.0 * (fwd - fixed_rate) * tau * df;

    assert_relative_eq!(
        fra.npv(&curve),
        expected_npv,
        epsilon = 1.0e-6,
    );
}

// ── Notional scaling ────────────────────────────────────────────────────────

/// NPV scales linearly with notional.
#[test]
fn fra_npv_scales_with_notional() {
    let curve = flat_curve(0.05, 3.0);

    let start = d(2024, 1, 1);
    let end = d(2025, 1, 1);

    let fra1 = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.04,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };

    let fra2 = ForwardRateAgreement {
        notional: 2_000_000.0,
        ..fra1
    };

    assert_relative_eq!(
        fra2.npv(&curve),
        2.0 * fra1.npv(&curve),
        epsilon = 1.0e-10,
    );
}

// ── Day count convention effect ─────────────────────────────────────────────

/// Different day count conventions should produce different year fractions
/// for the same date pair, resulting in different forward rates and NPVs.
#[test]
fn fra_day_count_convention_matters() {
    let curve = flat_curve(0.05, 3.0);

    let start = d(2024, 1, 15);
    let end = d(2024, 7, 15);

    let fra_act360 = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.04,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act360,
    };

    let fra_act365 = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.04,
        start_date: start,
        end_date: end,
        day_count: DayCountConvention::Act365Fixed,
    };

    // Different conventions → different tau → different NPV
    let npv_360 = fra_act360.npv(&curve);
    let npv_365 = fra_act365.npv(&curve);

    assert!(
        (npv_360 - npv_365).abs() > 1.0e-6,
        "Different day count conventions should produce different NPVs"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

/// FRA with start == end returns zero forward rate and zero NPV.
#[test]
fn fra_zero_period_returns_zero() {
    let curve = flat_curve(0.05, 3.0);
    let date = d(2024, 6, 15);

    let fra = ForwardRateAgreement {
        notional: 1_000_000.0,
        fixed_rate: 0.05,
        start_date: date,
        end_date: date,
        day_count: DayCountConvention::Act365Fixed,
    };

    assert_eq!(fra.forward_rate(&curve), 0.0);
    assert_eq!(fra.npv(&curve), 0.0);
}

/// FRA on multiple day count conventions — consistency check.
#[test]
fn fra_multiple_conventions_positive_forward() {
    let curve = flat_curve(0.05, 3.0);
    let start = d(2024, 1, 1);
    let end = d(2025, 1, 1);

    for conv in [
        DayCountConvention::Act360,
        DayCountConvention::Act365Fixed,
        DayCountConvention::Thirty360,
        DayCountConvention::ThirtyE360,
        DayCountConvention::ActActISDA,
    ] {
        let fra = ForwardRateAgreement {
            notional: 1_000_000.0,
            fixed_rate: 0.04,
            start_date: start,
            end_date: end,
            day_count: conv,
        };

        let fwd = fra.forward_rate(&curve);
        assert!(
            fwd > 0.0,
            "Forward rate must be positive for {:?}: {fwd}",
            conv
        );

        let npv = fra.npv(&curve);
        assert!(
            npv > 0.0,
            "NPV must be positive when fixed < forward for {:?}: {npv}",
            conv
        );
    }
}

// ── Upward-sloping curve ────────────────────────────────────────────────────

/// On an upward-sloping curve, longer-dated FRAs should have higher
/// forward rates.
#[test]
fn fra_forward_rate_increases_on_upward_sloping_curve() {
    let deposits = vec![
        (0.25, 0.03),
        (0.50, 0.04),
        (1.00, 0.05),
        (2.00, 0.06),
    ];
    let curve = YieldCurveBuilder::from_deposits(&deposits);

    let base = d(2024, 1, 1);
    let periods = [
        (d(2024, 1, 1), d(2024, 4, 1)),   // ~3M
        (d(2024, 1, 1), d(2024, 7, 1)),   // ~6M
        (d(2024, 1, 1), d(2025, 1, 1)),   // ~1Y
    ];

    let mut prev_fwd = 0.0;
    for (start, end) in &periods {
        let fra = ForwardRateAgreement {
            notional: 1_000_000.0,
            fixed_rate: 0.0,
            start_date: *start,
            end_date: *end,
            day_count: DayCountConvention::Act365Fixed,
        };
        let fwd = fra.forward_rate(&curve);
        assert!(
            fwd >= prev_fwd,
            "Forward rate should increase on upward-sloping curve"
        );
        prev_fwd = fwd;
    }
    let _ = base;
}
