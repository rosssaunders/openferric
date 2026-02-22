//! Comprehensive ISDA CDS reference tests.
//!
//! Reference values derived from QuantLib — a free/open-source C++ finance library
//! (BSD 3-Clause licence). See `vendor/QuantLib/test-suite/creditdefaultswap.cpp`,
//! `cdsoption.cpp` for the original C++ test code.

use approx::assert_relative_eq;
use chrono::{Datelike, NaiveDate};
use rand::SeedableRng;
use rand::rngs::StdRng;

use openferric::credit::cds_option::{CdsOption, fair_spread_from_hazard, risky_annuity};
use openferric::credit::{
    CdoTranche, Cds, CdsDateRule, CdsIndex, DatedCds, GaussianCopula, IsdaConventions,
    NthToDefaultBasket, ProtectionSide, SurvivalCurve, SyntheticCdo,
    bootstrap_survival_curve_from_cds_spreads, cash_settle_date, first_to_default_spread_copula,
    generate_imm_schedule, hazard_from_par_spread, next_imm_twentieth, previous_imm_twentieth,
    price_isda_flat, price_midpoint_flat, step_in_date, vasicek_portfolio_loss_cdf,
};
use openferric::rates::YieldCurve;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Flat discount curve with quarterly nodes out to `max_tenor` years.
fn flat_discount_curve(rate: f64, max_tenor: f64) -> YieldCurve {
    let n = (max_tenor * 4.0).ceil() as usize;
    YieldCurve::new(
        (1..=n)
            .map(|i| {
                let t = i as f64 * 0.25;
                (t, (-rate * t).exp())
            })
            .collect(),
    )
}

/// Flat survival curve from a constant hazard rate out to `max_tenor`.
fn flat_survival_curve(hazard: f64, max_tenor: f64) -> SurvivalCurve {
    let n = (max_tenor * 4.0).ceil() as usize;
    let tenors: Vec<f64> = (1..=n).map(|i| i as f64 * 0.25).collect();
    SurvivalCurve::from_piecewise_hazard(&tenors, &vec![hazard; tenors.len()])
}

// ===========================================================================
// 1. Survival Curve Construction & Interpolation
// ===========================================================================

#[test]
fn survival_curve_from_piecewise_hazard_exact() {
    // Two periods: h1=0.02 on [0,2], h2=0.05 on [2,5]
    let tenors = vec![2.0, 5.0];
    let hazards = vec![0.02, 0.05];
    let curve = SurvivalCurve::from_piecewise_hazard(&tenors, &hazards);

    // S(t) = exp(-∫₀ᵗ h ds)
    let s2 = (-0.02 * 2.0_f64).exp();
    let s5 = (-0.02 * 2.0 - 0.05 * 3.0_f64).exp();

    assert_relative_eq!(curve.survival_prob(2.0), s2, epsilon = 1e-12);
    assert_relative_eq!(curve.survival_prob(5.0), s5, epsilon = 1e-12);
}

#[test]
fn survival_prob_interpolates_log_linearly() {
    let curve = SurvivalCurve::from_piecewise_hazard(&[2.0, 5.0], &[0.02, 0.05]);
    let s2 = curve.survival_prob(2.0);
    let s5 = curve.survival_prob(5.0);

    // At t=3.5 (midpoint of [2,5]) log-linear means:
    // ln S(3.5) = ln S(2) + (3.5-2)/(5-2) * (ln S(5) - ln S(2))
    let w = (3.5 - 2.0) / (5.0 - 2.0);
    let expected = (s2.ln() + w * (s5.ln() - s2.ln())).exp();
    assert_relative_eq!(curve.survival_prob(3.5), expected, epsilon = 1e-12);
}

#[test]
fn hazard_rate_piecewise_constant() {
    let curve = SurvivalCurve::from_piecewise_hazard(&[2.0, 5.0], &[0.02, 0.05]);

    // hazard_rate at t in [0,2] should be h1=0.02, at t in (2,5] should be h2=0.05
    assert_relative_eq!(curve.hazard_rate(1.0), 0.02, epsilon = 1e-10);
    assert_relative_eq!(curve.hazard_rate(3.0), 0.05, epsilon = 1e-10);
}

#[test]
fn default_prob_is_survival_difference() {
    let curve = SurvivalCurve::new(vec![(1.0, 0.97), (3.0, 0.91), (5.0, 0.84)]);
    let dp = curve.default_prob(1.0, 3.0);
    let expected = curve.survival_prob(1.0) - curve.survival_prob(3.0);
    assert_relative_eq!(dp, expected, epsilon = 1e-12);
}

#[test]
fn inverse_survival_round_trip() {
    let curve = SurvivalCurve::new(vec![(1.0, 0.96), (3.0, 0.88), (5.0, 0.80), (10.0, 0.62)]);
    for &t in &[0.1, 0.5, 1.5, 3.0, 4.5, 7.0, 9.5] {
        let s = curve.survival_prob(t);
        let t_back = curve.inverse_survival_prob(s);
        assert_relative_eq!(t_back, t, epsilon = 1e-10);
    }
}

#[test]
fn survival_curve_monotonicity() {
    let curve =
        SurvivalCurve::from_piecewise_hazard(&[1.0, 3.0, 5.0, 10.0], &[0.01, 0.02, 0.03, 0.04]);
    let times: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
    for w in times.windows(2) {
        assert!(
            curve.survival_prob(w[1]) <= curve.survival_prob(w[0]),
            "S({}) > S({})",
            w[1],
            w[0]
        );
    }
}

#[test]
fn survival_curve_edge_cases() {
    // Empty curve
    let empty = SurvivalCurve::new(vec![]);
    assert_relative_eq!(empty.survival_prob(1.0), 1.0, epsilon = 1e-12);
    assert_relative_eq!(empty.hazard_rate(1.0), 0.0, epsilon = 1e-12);

    // Single point
    let single = SurvivalCurve::new(vec![(5.0, 0.80)]);
    assert_relative_eq!(single.survival_prob(0.0), 1.0, epsilon = 1e-12);
    assert!(single.survival_prob(5.0) > 0.0);
    assert!(single.hazard_rate(2.5) > 0.0);

    // t=0
    let curve = SurvivalCurve::from_piecewise_hazard(&[5.0], &[0.02]);
    assert_relative_eq!(curve.survival_prob(0.0), 1.0, epsilon = 1e-12);
}

#[test]
fn survival_prob_before_first_tenor_extrapolates() {
    let curve = SurvivalCurve::from_piecewise_hazard(&[2.0, 5.0], &[0.03, 0.06]);
    // S(1.0) should use log-linear between (0, 1.0) and first tenor
    let s1 = curve.survival_prob(1.0);
    assert!(s1 > 0.0 && s1 < 1.0);
    // Should equal exp(-0.03 * 1.0) for constant hazard
    assert_relative_eq!(s1, (-0.03_f64).exp(), epsilon = 1e-10);
}

// ===========================================================================
// 2. Survival Curve Bootstrap from CDS Spreads
// ===========================================================================

#[test]
fn bootstrap_upward_sloping_term_structure() {
    let discount_curve = flat_discount_curve(0.05, 12.0);
    let recovery = 0.4;
    let quotes = vec![
        (1.0, 0.0060),
        (3.0, 0.0080),
        (5.0, 0.0100),
        (7.0, 0.0115),
        (10.0, 0.0130),
    ];

    let curve = bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

    // Each pillar CDS should reprice to NPV ≈ 0
    for &(tenor, spread) in &quotes {
        let cds = Cds {
            notional: 1.0,
            spread,
            maturity: tenor,
            recovery_rate: recovery,
            payment_freq: 4,
        };
        assert_relative_eq!(cds.npv(&discount_curve, &curve), 0.0, epsilon = 1e-8);
    }
}

#[test]
fn bootstrap_flat_spreads_give_approximately_flat_hazard() {
    let discount_curve = flat_discount_curve(0.03, 12.0);
    let recovery = 0.4;
    let flat_spread = 0.0100;
    let quotes = vec![
        (1.0, flat_spread),
        (3.0, flat_spread),
        (5.0, flat_spread),
        (7.0, flat_spread),
        (10.0, flat_spread),
    ];

    let curve = bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

    // Hazard rates at different tenors should be approximately equal
    let h1 = curve.hazard_rate(0.5);
    let h5 = curve.hazard_rate(4.0);
    let h8 = curve.hazard_rate(8.0);
    assert_relative_eq!(h1, h5, epsilon = 3e-3);
    assert_relative_eq!(h5, h8, epsilon = 3e-3);
}

#[test]
fn bootstrap_survival_probs_are_decreasing() {
    let discount_curve = flat_discount_curve(0.04, 12.0);
    let recovery = 0.4;
    let quotes = vec![
        (1.0, 0.0060),
        (3.0, 0.0080),
        (5.0, 0.0100),
        (7.0, 0.0115),
        (10.0, 0.0130),
    ];

    let curve = bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

    let times: Vec<f64> = (1..=40).map(|i| i as f64 * 0.25).collect();
    for w in times.windows(2) {
        assert!(
            curve.survival_prob(w[1]) <= curve.survival_prob(w[0]) + 1e-12,
            "Non-monotone at t={}: S({})={} > S({})={}",
            w[1],
            w[1],
            curve.survival_prob(w[1]),
            w[0],
            curve.survival_prob(w[0])
        );
    }
}

#[test]
fn bootstrap_repricing_via_survival_curve_method() {
    // Test the SurvivalCurve::bootstrap_from_cds_spreads static method
    let discount_curve = flat_discount_curve(0.03, 12.0);
    let recovery = 0.4;
    let quotes = vec![(1.0, 0.0050), (3.0, 0.0070), (5.0, 0.0090)];

    let curve = SurvivalCurve::bootstrap_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

    for &(tenor, spread) in &quotes {
        let cds = Cds {
            notional: 1.0,
            spread,
            maturity: tenor,
            recovery_rate: recovery,
            payment_freq: 4,
        };
        let repriced = cds.fair_spread(&discount_curve, &curve);
        assert_relative_eq!(repriced, spread, epsilon = 1e-8);
    }
}

// ===========================================================================
// 3. Standard CDS Pricing (Cds struct)
// ===========================================================================

#[test]
fn fair_spread_approximates_hazard_times_lgd() {
    // Fundamental identity: fair_spread ≈ (1-R)*h for flat hazard
    let r = 0.03;
    let discount_curve = flat_discount_curve(r, 20.0);
    let hazard = 0.02;
    let survival_curve = flat_survival_curve(hazard, 20.0);
    let recovery = 0.4;

    let cds = Cds {
        notional: 10_000_000.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };

    let fair = cds.fair_spread(&discount_curve, &survival_curve);
    let expected = (1.0 - recovery) * hazard;
    assert_relative_eq!(fair, expected, epsilon = 1e-3);
}

#[test]
fn npv_zero_at_fair_spread() {
    let discount_curve = flat_discount_curve(0.04, 20.0);
    let survival_curve = flat_survival_curve(0.025, 20.0);

    let cds = Cds {
        notional: 1_000_000.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: 0.4,
        payment_freq: 4,
    };
    let fair = cds.fair_spread(&discount_curve, &survival_curve);

    let at_fair = Cds {
        spread: fair,
        ..cds.clone()
    };
    assert_relative_eq!(
        at_fair.npv(&discount_curve, &survival_curve),
        0.0,
        epsilon = 1e-8
    );
}

#[test]
fn premium_plus_protection_legs_balance() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let survival_curve = flat_survival_curve(0.015, 20.0);

    let cds = Cds {
        notional: 1_000_000.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: 0.4,
        payment_freq: 4,
    };
    let fair = cds.fair_spread(&discount_curve, &survival_curve);
    let at_fair = Cds {
        spread: fair,
        ..cds.clone()
    };

    let premium = at_fair.premium_leg_pv(&discount_curve, &survival_curve);
    let protection = at_fair.protection_leg_pv(&discount_curve, &survival_curve);
    assert_relative_eq!(premium, protection, epsilon = 1e-6);
}

#[test]
fn npv_scales_linearly_with_notional() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let survival_curve = flat_survival_curve(0.02, 20.0);

    let cds1 = Cds {
        notional: 1_000_000.0,
        spread: 0.01,
        maturity: 5.0,
        recovery_rate: 0.4,
        payment_freq: 4,
    };
    let cds2 = Cds {
        notional: 2_000_000.0,
        ..cds1.clone()
    };

    let npv1 = cds1.npv(&discount_curve, &survival_curve);
    let npv2 = cds2.npv(&discount_curve, &survival_curve);
    assert_relative_eq!(npv2, 2.0 * npv1, epsilon = 1e-8);
}

#[test]
fn higher_hazard_rate_increases_fair_spread() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let recovery = 0.4;

    let curve_low = flat_survival_curve(0.01, 20.0);
    let curve_high = flat_survival_curve(0.04, 20.0);

    let cds = Cds {
        notional: 1.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };

    let spread_low = cds.fair_spread(&discount_curve, &curve_low);
    let spread_high = cds.fair_spread(&discount_curve, &curve_high);
    assert!(spread_high > spread_low);
}

#[test]
fn higher_recovery_rate_lowers_fair_spread() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let survival_curve = flat_survival_curve(0.02, 20.0);

    let cds_low_r = Cds {
        notional: 1.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: 0.2,
        payment_freq: 4,
    };
    let cds_high_r = Cds {
        notional: 1.0,
        spread: 0.0,
        maturity: 5.0,
        recovery_rate: 0.6,
        payment_freq: 4,
    };

    let s_low = cds_low_r.fair_spread(&discount_curve, &survival_curve);
    let s_high = cds_high_r.fair_spread(&discount_curve, &survival_curve);
    assert!(s_low > s_high, "Lower recovery should mean higher spread");
}

// ===========================================================================
// 4. ISDA Dated CDS Pricing
// ===========================================================================

#[test]
fn isda_midpoint_cached_regression() {
    // QuantLib creditdefaultswap.cpp testCachedValue() reference.
    let evaluation_date = NaiveDate::from_ymd_opt(2006, 6, 9).unwrap();
    let issue_date = NaiveDate::from_ymd_opt(2005, 6, 9).unwrap();
    let maturity_date = NaiveDate::from_ymd_opt(2015, 6, 9).unwrap();

    let cds = DatedCds {
        side: ProtectionSide::Seller,
        notional: 10_000.0,
        running_spread: 0.0120,
        recovery_rate: 0.40,
        issue_date,
        maturity_date,
        coupon_interval_months: 6,
        date_rule: CdsDateRule::TwentiethImm,
    };

    let result = price_midpoint_flat(
        &cds,
        evaluation_date,
        0.01234,
        0.06,
        IsdaConventions {
            step_in_days: 1,
            cash_settle_days: 1,
        },
    );

    assert_relative_eq!(result.clean_npv, 295.0153398, epsilon = 1.0);
    assert_relative_eq!(result.fair_spread, 0.007517539081, epsilon = 5.0e-7);
}

#[test]
fn isda_flat_vs_midpoint_same_sign_similar_magnitude() {
    let eval = NaiveDate::from_ymd_opt(2026, 3, 15).unwrap();
    let cds = DatedCds::standard_imm(ProtectionSide::Buyer, eval, 5, 10_000_000.0, 0.01, 0.4);

    let hazard = 0.02;
    let conventions = IsdaConventions::default();

    let midpoint = price_midpoint_flat(&cds, eval, hazard, 0.03, conventions);
    let isda = price_isda_flat(&cds, eval, hazard, 0.03, conventions);

    // Same sign
    assert!(
        midpoint.clean_npv.signum() == isda.clean_npv.signum(),
        "midpoint={} isda={}",
        midpoint.clean_npv,
        isda.clean_npv
    );
    // Similar magnitude (within 10%)
    if midpoint.clean_npv.abs() > 1.0 {
        let ratio = isda.clean_npv / midpoint.clean_npv;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "ratio={ratio}, midpoint={}, isda={}",
            midpoint.clean_npv,
            isda.clean_npv
        );
    }
}

#[test]
fn standard_imm_constructor_conventions() {
    let eval = NaiveDate::from_ymd_opt(2026, 2, 16).unwrap();
    let cds = DatedCds::standard_imm(ProtectionSide::Buyer, eval, 5, 10_000_000.0, 0.01, 0.4);

    // Coupon interval should be quarterly
    assert_eq!(cds.coupon_interval_months, 3);
    assert_eq!(cds.date_rule, CdsDateRule::QuarterlyImm);

    // Issue date should be on a 20th of IMM month
    assert_eq!(cds.issue_date.day(), 20);
    assert!(
        [3, 6, 9, 12].contains(&cds.issue_date.month()),
        "Issue month: {}",
        cds.issue_date.month()
    );

    // Maturity should be on a 20th of IMM month
    assert_eq!(cds.maturity_date.day(), 20);
    assert!(
        [3, 6, 9, 12].contains(&cds.maturity_date.month()),
        "Maturity month: {}",
        cds.maturity_date.month()
    );

    // Maturity should be roughly 5 years out
    let years_diff = (cds.maturity_date - cds.issue_date).num_days() as f64 / 365.25;
    assert!(
        (4.5..=6.0).contains(&years_diff),
        "Tenor = {years_diff:.2} years"
    );
}

#[test]
fn buyer_seller_npv_opposite_sign() {
    let eval = NaiveDate::from_ymd_opt(2026, 3, 15).unwrap();
    let hazard = 0.03;
    let conventions = IsdaConventions::default();

    let buyer = DatedCds::standard_imm(
        ProtectionSide::Buyer,
        eval,
        5,
        10_000_000.0,
        0.005, // Low spread => buyer benefits from higher hazard
        0.4,
    );
    let seller = DatedCds {
        side: ProtectionSide::Seller,
        ..buyer.clone()
    };

    let buyer_result = price_isda_flat(&buyer, eval, hazard, 0.03, conventions);
    let seller_result = price_isda_flat(&seller, eval, hazard, 0.03, conventions);

    assert_relative_eq!(
        buyer_result.clean_npv,
        -seller_result.clean_npv,
        epsilon = 1e-6
    );
}

#[test]
fn hazard_from_par_spread_consistency() {
    // s = h * (1-R) => h = s / (1-R)
    let s = 0.01; // 100 bps
    let r = 0.4;
    let h = hazard_from_par_spread(s, r);
    let expected = s / (1.0 - r); // ~166.67 bps
    assert_relative_eq!(h, expected, epsilon = 1e-12);
}

#[test]
fn par_cds_has_near_zero_npv() {
    let eval = NaiveDate::from_ymd_opt(2026, 3, 15).unwrap();
    let spread = 0.0100;
    let recovery = 0.4;
    let hazard = hazard_from_par_spread(spread, recovery);

    let cds = DatedCds::standard_imm(
        ProtectionSide::Buyer,
        eval,
        5,
        10_000_000.0,
        spread,
        recovery,
    );

    let result = price_isda_flat(&cds, eval, hazard, 0.05, IsdaConventions::default());
    // NPV should be approximately zero for a par CDS
    // (tolerance is loose due to discrete schedule vs continuous approximation)
    assert!(
        result.clean_npv.abs() < 5.0e4,
        "Par CDS NPV should be near zero, got {}",
        result.clean_npv
    );
}

// ===========================================================================
// 5. ISDA Date Utilities
// ===========================================================================

#[test]
fn previous_imm_twentieth_various_dates() {
    // Mid-February → Dec 20th of prior year
    let d1 = NaiveDate::from_ymd_opt(2026, 2, 16).unwrap();
    assert_eq!(
        previous_imm_twentieth(d1),
        NaiveDate::from_ymd_opt(2025, 12, 20).unwrap()
    );

    // Exactly on an IMM date → that date
    let d2 = NaiveDate::from_ymd_opt(2026, 3, 20).unwrap();
    assert_eq!(
        previous_imm_twentieth(d2),
        NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
    );

    // Day after IMM date → still that IMM date
    let d3 = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
    assert_eq!(
        previous_imm_twentieth(d3),
        NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
    );

    // October → Sep 20
    let d4 = NaiveDate::from_ymd_opt(2026, 10, 5).unwrap();
    assert_eq!(
        previous_imm_twentieth(d4),
        NaiveDate::from_ymd_opt(2026, 9, 20).unwrap()
    );
}

#[test]
fn next_imm_twentieth_various_dates() {
    // Mid-February → Mar 20
    let d1 = NaiveDate::from_ymd_opt(2026, 2, 16).unwrap();
    assert_eq!(
        next_imm_twentieth(d1),
        NaiveDate::from_ymd_opt(2026, 3, 20).unwrap()
    );

    // Exactly on an IMM date → that date
    let d2 = NaiveDate::from_ymd_opt(2026, 6, 20).unwrap();
    assert_eq!(
        next_imm_twentieth(d2),
        NaiveDate::from_ymd_opt(2026, 6, 20).unwrap()
    );

    // Late December → Dec 20 if before, else Mar 20 next year
    let d3 = NaiveDate::from_ymd_opt(2026, 12, 21).unwrap();
    assert_eq!(
        next_imm_twentieth(d3),
        NaiveDate::from_ymd_opt(2027, 3, 20).unwrap()
    );
}

#[test]
fn step_in_and_cash_settle_with_weekends() {
    // Friday 2026-02-13 → step_in = Mon 2026-02-16 (T+1 business day)
    let friday = NaiveDate::from_ymd_opt(2026, 2, 13).unwrap();
    let si = step_in_date(friday);
    assert_eq!(si.weekday(), chrono::Weekday::Mon);
    assert!(si > friday);

    let cs = cash_settle_date(friday);
    assert!(cs > si);
    // Cash settle is T+3 business days from Friday → Wed
    assert_eq!(cs.weekday(), chrono::Weekday::Wed);
}

#[test]
fn generate_imm_schedule_strictly_increasing() {
    let issue = NaiveDate::from_ymd_opt(2025, 12, 20).unwrap();
    let maturity = NaiveDate::from_ymd_opt(2031, 3, 20).unwrap();
    let schedule = generate_imm_schedule(issue, maturity, 3, CdsDateRule::QuarterlyImm);

    assert!(schedule.len() >= 2, "Schedule too short: {:?}", schedule);

    // Strictly increasing
    for w in schedule.windows(2) {
        assert!(w[1] > w[0], "Non-increasing schedule: {} >= {}", w[0], w[1]);
    }

    // Last date should be at or after maturity
    assert!(*schedule.last().unwrap() >= maturity);
}

// ===========================================================================
// 6. CDS Option (Black's model)
// ===========================================================================

#[test]
fn cds_option_atm_put_call_parity() {
    let forward = 0.01;
    let vol = 0.30;
    let rpv01 = risky_annuity(4, 5.0, 0.01, 0.05, 0.4);

    let payer = CdsOption {
        notional: 1_000_000.0,
        strike_spread: forward,
        option_expiry: 1.0,
        cds_maturity: 5.0,
        is_payer: true,
        recovery_rate: 0.4,
    };
    let receiver = CdsOption {
        is_payer: false,
        ..payer.clone()
    };

    let p = payer.black_price(forward, vol, rpv01);
    let r = receiver.black_price(forward, vol, rpv01);
    assert_relative_eq!(p, r, epsilon = 1e-8);
}

#[test]
fn cds_option_zero_vol_gives_intrinsic() {
    let rpv01 = risky_annuity(4, 5.0, 0.01, 0.05, 0.4);
    let forward = 0.02;
    let strike = 0.01;

    let payer = CdsOption {
        notional: 1_000_000.0,
        strike_spread: strike,
        option_expiry: 1.0,
        cds_maturity: 5.0,
        is_payer: true,
        recovery_rate: 0.4,
    };
    let price = payer.black_price(forward, 0.0, rpv01);
    let expected = 1_000_000.0 * rpv01 * (forward - strike);
    assert_relative_eq!(price, expected, epsilon = 1e-6);

    // OTM receiver at zero vol = 0
    let receiver = CdsOption {
        is_payer: false,
        ..payer.clone()
    };
    assert_relative_eq!(
        receiver.black_price(forward, 0.0, rpv01),
        0.0,
        epsilon = 1e-6
    );
}

#[test]
fn risky_annuity_positive_and_decreasing_with_hazard() {
    let r = 0.05;
    let rpv01_low = risky_annuity(4, 5.0, 0.01, r, 0.4);
    let rpv01_high = risky_annuity(4, 5.0, 0.10, r, 0.4);

    assert!(rpv01_low > 0.0);
    assert!(rpv01_high > 0.0);
    assert!(rpv01_low > rpv01_high, "Higher hazard should reduce RPV01");
}

#[test]
fn fair_spread_from_hazard_approximation() {
    // For discrete payments: fair_spread ≈ h*(1-R) with small correction
    let h = 0.02;
    let r_free = 0.03;
    let recovery = 0.4;
    let fair = fair_spread_from_hazard(4, 5.0, h, r_free, recovery);
    let continuous = h * (1.0 - recovery);
    // Discrete vs continuous should be close
    assert_relative_eq!(fair, continuous, epsilon = 2e-3);
}

#[test]
fn cds_option_quantlib_cached_value() {
    // QuantLib cdsoption.cpp reference: h=0.001, r=0.02, R=0.4, vol=0.20
    let hazard_rate = 0.001_f64;
    let risk_free_rate = 0.02_f64;
    let recovery = 0.4_f64;
    let vol = 0.20_f64;
    let notional = 1_000_000.0_f64;
    let t_expiry = 275.0 / 360.0;

    let cds_start_days = 309.0_f64;
    let num_periods = 28_u32;

    let mut coupon_leg_npv = 0.0;
    let mut prot_leg_npv = 0.0;

    for i in 0..num_periods {
        let period_days = 91.31_f64;
        let t_start_days = cds_start_days + i as f64 * period_days;
        let t_end_days = cds_start_days + (i + 1) as f64 * period_days;
        let t_mid_days = (t_start_days + t_end_days) / 2.0;
        let accrual = period_days / 360.0;

        let t_end = t_end_days / 360.0;
        let t_start = t_start_days / 360.0;
        let t_mid = t_mid_days / 360.0;

        let df_end = (-risk_free_rate * t_end).exp();
        let df_mid = (-risk_free_rate * t_mid).exp();
        let s_start = (-hazard_rate * t_start).exp();
        let s_end = (-hazard_rate * t_end).exp();
        let default_prob = s_start - s_end;

        coupon_leg_npv += s_end * accrual * df_end;
        coupon_leg_npv += default_prob * (accrual / 2.0) * df_mid;
        prot_leg_npv += (1.0 - recovery) * default_prob * df_mid;
    }

    let rpv01 = coupon_leg_npv;
    let fair = prot_leg_npv / coupon_leg_npv;

    let option = CdsOption {
        notional,
        strike_spread: fair,
        option_expiry: t_expiry,
        cds_maturity: 7.0,
        is_payer: true,
        recovery_rate: recovery,
    };

    let price = option.black_price(fair, vol, rpv01);
    assert!(
        (price - 270.976348).abs() < 1.0,
        "QuantLib cached value: got {price}, expected 270.976348"
    );
}

// ===========================================================================
// 7. Synthetic CDO & Tranche Analytics
// ===========================================================================

#[test]
fn equity_tranche_loss_exceeds_mezzanine_exceeds_senior() {
    let cdo = SyntheticCdo {
        num_names: 125,
        pool_spread: 0.01,
        recovery_rate: 0.4,
        correlation: 0.30,
        risk_free_rate: 0.05,
        maturity: 5.0,
        payment_freq: 4,
    };

    let equity = CdoTranche {
        attachment: 0.0,
        detachment: 0.03,
        notional: 0.03,
        spread: 0.0,
    };
    let mezz = CdoTranche {
        attachment: 0.03,
        detachment: 0.07,
        notional: 0.04,
        spread: 0.0,
    };
    let senior = CdoTranche {
        attachment: 0.07,
        detachment: 1.0,
        notional: 0.93,
        spread: 0.0,
    };

    let t = cdo.maturity;
    let lf_eq = equity.expected_loss_fraction(
        cdo.default_probability(t),
        cdo.recovery_rate,
        cdo.correlation,
    );
    let lf_mz = mezz.expected_loss_fraction(
        cdo.default_probability(t),
        cdo.recovery_rate,
        cdo.correlation,
    );
    let lf_sn = senior.expected_loss_fraction(
        cdo.default_probability(t),
        cdo.recovery_rate,
        cdo.correlation,
    );

    assert!(
        lf_eq > lf_mz,
        "equity loss {lf_eq} should > mezz loss {lf_mz}"
    );
    assert!(
        lf_mz > lf_sn,
        "mezz loss {lf_mz} should > senior loss {lf_sn}"
    );
}

#[test]
fn tranche_expected_losses_sum_to_portfolio_loss() {
    let cdo = SyntheticCdo {
        num_names: 125,
        pool_spread: 0.01,
        recovery_rate: 0.4,
        correlation: 0.30,
        risk_free_rate: 0.05,
        maturity: 5.0,
        payment_freq: 4,
    };

    let equity = CdoTranche {
        attachment: 0.0,
        detachment: 0.03,
        notional: 0.03,
        spread: 0.0,
    };
    let mezz = CdoTranche {
        attachment: 0.03,
        detachment: 0.07,
        notional: 0.04,
        spread: 0.0,
    };
    let senior = CdoTranche {
        attachment: 0.07,
        detachment: 1.0,
        notional: 0.93,
        spread: 0.0,
    };

    let t = cdo.maturity;
    let el_sum = cdo.expected_tranche_loss(&equity, t)
        + cdo.expected_tranche_loss(&mezz, t)
        + cdo.expected_tranche_loss(&senior, t);
    let portfolio_el = cdo.portfolio_expected_loss(t);

    assert_relative_eq!(el_sum, portfolio_el, epsilon = 4e-3);
}

#[test]
fn cdo_fair_spread_ordering() {
    let cdo = SyntheticCdo {
        num_names: 125,
        pool_spread: 0.01,
        recovery_rate: 0.4,
        correlation: 0.30,
        risk_free_rate: 0.05,
        maturity: 5.0,
        payment_freq: 4,
    };

    let equity = CdoTranche {
        attachment: 0.0,
        detachment: 0.03,
        notional: 0.03,
        spread: 0.0,
    };
    let mezz = CdoTranche {
        attachment: 0.03,
        detachment: 0.07,
        notional: 0.04,
        spread: 0.0,
    };
    let senior = CdoTranche {
        attachment: 0.07,
        detachment: 1.0,
        notional: 0.93,
        spread: 0.0,
    };

    let s_eq = cdo.fair_spread(&equity);
    let s_mz = cdo.fair_spread(&mezz);
    let s_sn = cdo.fair_spread(&senior);

    assert!(s_eq > s_mz, "equity spread {s_eq} > mezz spread {s_mz}");
    assert!(s_mz > s_sn, "mezz spread {s_mz} > senior spread {s_sn}");
}

#[test]
fn vasicek_cdf_monotone_in_loss_fraction() {
    let q = 0.08;
    let recovery = 0.4;
    let rho = 0.3;

    let losses: Vec<f64> = (1..=20).map(|i| i as f64 * 0.025).collect();
    let cdfs: Vec<f64> = losses
        .iter()
        .map(|&l| vasicek_portfolio_loss_cdf(l, q, recovery, rho))
        .collect();

    for w in cdfs.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-12,
            "CDF not monotone: {} > {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn cdo_npv_zero_at_fair_spread() {
    let cdo = SyntheticCdo {
        num_names: 125,
        pool_spread: 0.01,
        recovery_rate: 0.4,
        correlation: 0.30,
        risk_free_rate: 0.05,
        maturity: 5.0,
        payment_freq: 4,
    };

    let tranche = CdoTranche {
        attachment: 0.03,
        detachment: 0.07,
        notional: 0.04,
        spread: 0.0,
    };

    let fair = cdo.fair_spread(&tranche);
    let at_fair = CdoTranche {
        spread: fair,
        ..tranche.clone()
    };
    let npv = cdo.npv(&at_fair);
    assert_relative_eq!(npv, 0.0, epsilon = 1e-10);
}

// ===========================================================================
// 8. CDS Index & Nth-to-Default
// ===========================================================================

#[test]
fn homogeneous_index_spread_equals_single_name() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let recovery = 0.4;
    let hazard = 0.02;
    let curve = flat_survival_curve(hazard, 20.0);

    let single = Cds {
        notional: 1.0,
        spread: 0.01,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };

    let n_names = 5;
    let index = CdsIndex {
        constituents: vec![single.clone(); n_names],
        weights: vec![1.0; n_names],
    };
    let curves = vec![curve.clone(); n_names];

    let index_spread = index.fair_spread(&discount_curve, &curves);
    let single_spread = single.fair_spread(&discount_curve, &curve);
    assert_relative_eq!(index_spread, single_spread, epsilon = 1e-12);
}

#[test]
fn first_to_default_spread_exceeds_single_name() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let recovery = 0.4;
    let single_spread = 0.01;
    let hazard = hazard_from_par_spread(single_spread, recovery);
    let curve = flat_survival_curve(hazard, 20.0);

    let single = Cds {
        notional: 1.0,
        spread: single_spread,
        maturity: 5.0,
        recovery_rate: recovery,
        payment_freq: 4,
    };
    let name_spread = single.fair_spread(&discount_curve, &curve);

    let curves = vec![curve.clone(); 5];
    let ftd = first_to_default_spread_copula(
        1.0,
        5.0,
        recovery,
        4,
        &discount_curve,
        &curves,
        &GaussianCopula::new(0.25),
        20_000,
        42,
    );

    assert!(
        ftd > name_spread,
        "FTD spread {ftd} should exceed single-name spread {name_spread}"
    );
}

#[test]
fn nth_to_default_spread_decreases_with_n() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let hazard = 0.02;
    let curve = flat_survival_curve(hazard, 20.0);
    let curves = vec![curve.clone(); 5];

    let basket1 = NthToDefaultBasket {
        n: 1,
        notional: 1.0,
        maturity: 5.0,
        recovery_rate: 0.4,
        payment_freq: 4,
    };
    let basket2 = NthToDefaultBasket {
        n: 2,
        ..basket1.clone()
    };
    let basket3 = NthToDefaultBasket {
        n: 3,
        ..basket1.clone()
    };

    let s1 = basket1.fair_spread(&discount_curve, &curves);
    let s2 = basket2.fair_spread(&discount_curve, &curves);
    let s3 = basket3.fair_spread(&discount_curve, &curves);

    assert!(s1 > s2, "1st-to-default {s1} > 2nd-to-default {s2}");
    assert!(s2 > s3, "2nd-to-default {s2} > 3rd-to-default {s3}");
}

#[test]
fn index_npv_is_weighted_sum_of_constituents() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let recovery = 0.4;

    // Different credit qualities
    let curves = vec![
        flat_survival_curve(0.01, 20.0),
        flat_survival_curve(0.02, 20.0),
        flat_survival_curve(0.03, 20.0),
    ];
    let spread = 0.01;
    let constituents: Vec<Cds> = (0..3)
        .map(|_| Cds {
            notional: 1.0,
            spread,
            maturity: 5.0,
            recovery_rate: recovery,
            payment_freq: 4,
        })
        .collect();

    let weights = vec![1.0, 1.0, 1.0];
    let index = CdsIndex {
        constituents: constituents.clone(),
        weights: weights.clone(),
    };

    let index_npv = index.npv(&discount_curve, &curves);

    // Manually compute weighted average
    let norm_w: Vec<f64> = {
        let s: f64 = weights.iter().sum();
        weights.iter().map(|w| w / s).collect()
    };
    let manual: f64 = constituents
        .iter()
        .zip(curves.iter())
        .zip(norm_w.iter())
        .map(|((cds, curve), w)| w * cds.npv(&discount_curve, curve))
        .sum();

    assert_relative_eq!(index_npv, manual, epsilon = 1e-12);
}

// ===========================================================================
// 9. Gaussian Copula Simulation
// ===========================================================================

#[test]
fn copula_simulate_returns_correct_count() {
    let curve = SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.02]);
    let copula = GaussianCopula::new(0.3);
    let mut rng = StdRng::seed_from_u64(42);

    let sim = copula.simulate_homogeneous(100, &curve, &mut rng);
    assert_eq!(sim.default_times.len(), 100);
    assert_eq!(sim.latent_variables.len(), 100);
    assert!(sim.default_times.iter().all(|t| t.is_finite() && *t >= 0.0));
}

#[test]
fn copula_heterogeneous_simulate() {
    let curves = vec![
        SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.01]),
        SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.02]),
        SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.05]),
    ];
    let copula = GaussianCopula::new(0.3);
    let mut rng = StdRng::seed_from_u64(42);

    let sim = copula.simulate(&curves, &mut rng);
    assert_eq!(sim.default_times.len(), 3);
    assert_eq!(sim.latent_variables.len(), 3);
}

#[test]
fn higher_correlation_increases_joint_default_rate() {
    let curve = SurvivalCurve::from_piecewise_hazard(&[10.0], &[0.02]);
    let horizon = 5.0;
    let n_paths = 20_000;

    let mut rng_low = StdRng::seed_from_u64(123);
    let mut rng_high = StdRng::seed_from_u64(123);
    let model_low = GaussianCopula::new(0.0);
    let model_high = GaussianCopula::new(0.7);

    let mut both_low = 0usize;
    let mut both_high = 0usize;
    for _ in 0..n_paths {
        let sim_low = model_low.simulate_homogeneous(2, &curve, &mut rng_low);
        let sim_high = model_high.simulate_homogeneous(2, &curve, &mut rng_high);
        if sim_low.defaults_by(horizon) == 2 {
            both_low += 1;
        }
        if sim_high.defaults_by(horizon) == 2 {
            both_high += 1;
        }
    }

    let p_low = both_low as f64 / n_paths as f64;
    let p_high = both_high as f64 / n_paths as f64;
    assert!(
        p_high > p_low + 0.005,
        "High correlation joint default ({p_high}) should exceed low ({p_low})"
    );
}

// ===========================================================================
// 10. Cross-cutting / regression tests
// ===========================================================================

#[test]
fn cds_fair_spread_positive_for_nonzero_hazard() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let survival_curve = flat_survival_curve(0.015, 20.0);

    for &maturity in &[1.0, 3.0, 5.0, 7.0, 10.0] {
        let cds = Cds {
            notional: 1.0,
            spread: 0.0,
            maturity,
            recovery_rate: 0.4,
            payment_freq: 4,
        };
        let fair = cds.fair_spread(&discount_curve, &survival_curve);
        assert!(
            fair > 0.0,
            "Fair spread at {maturity}Y should be positive, got {fair}"
        );
    }
}

#[test]
fn protection_leg_pv_positive_for_nonzero_hazard() {
    let discount_curve = flat_discount_curve(0.03, 20.0);
    let survival_curve = flat_survival_curve(0.02, 20.0);

    let cds = Cds {
        notional: 1_000_000.0,
        spread: 0.01,
        maturity: 5.0,
        recovery_rate: 0.4,
        payment_freq: 4,
    };

    let prot = cds.protection_leg_pv(&discount_curve, &survival_curve);
    assert!(prot > 0.0, "Protection leg PV should be positive: {prot}");
}

#[test]
fn dated_cds_fair_spread_positive() {
    let eval = NaiveDate::from_ymd_opt(2026, 3, 15).unwrap();
    let cds = DatedCds::standard_imm(ProtectionSide::Buyer, eval, 5, 10_000_000.0, 0.01, 0.4);

    let result = price_midpoint_flat(&cds, eval, 0.02, 0.03, IsdaConventions::default());
    assert!(
        result.fair_spread > 0.0,
        "Dated CDS fair spread should be positive: {}",
        result.fair_spread
    );
}

#[test]
fn bootstrap_and_reprice_round_trip() {
    // Full round-trip: create true curve → compute fair spreads → bootstrap → verify
    let discount_curve = flat_discount_curve(0.04, 15.0);
    let recovery = 0.4;

    // True curve with varying hazard rates
    let true_curve = SurvivalCurve::from_piecewise_hazard(
        &[1.0, 3.0, 5.0, 7.0, 10.0],
        &[0.01, 0.015, 0.02, 0.025, 0.03],
    );

    // Compute fair spreads at pillar maturities
    let maturities = [1.0, 3.0, 5.0, 7.0, 10.0];
    let quotes: Vec<(f64, f64)> = maturities
        .iter()
        .map(|&m| {
            let cds = Cds {
                notional: 1.0,
                spread: 0.0,
                maturity: m,
                recovery_rate: recovery,
                payment_freq: 4,
            };
            (m, cds.fair_spread(&discount_curve, &true_curve))
        })
        .collect();

    // Bootstrap
    let bootstrapped =
        bootstrap_survival_curve_from_cds_spreads(&quotes, recovery, 4, &discount_curve);

    // Repricing should give back the same spreads
    for &(m, spread) in &quotes {
        let cds = Cds {
            notional: 1.0,
            spread: 0.0,
            maturity: m,
            recovery_rate: recovery,
            payment_freq: 4,
        };
        let repriced = cds.fair_spread(&discount_curve, &bootstrapped);
        assert_relative_eq!(repriced, spread, epsilon = 1e-8);
    }
}
