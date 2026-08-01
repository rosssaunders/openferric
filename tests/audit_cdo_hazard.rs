//! Audit regression tests: `SyntheticCdo` must convert its pool par spread to a
//! hazard rate via the credit triangle `h = s / (1 - R)` (O'Kane 2008, Ch. 5),
//! and must discount with the given flat rate even when it is negative.
//!
//! Historical bug: `SyntheticCdo::hazard_rate()` returned the par spread
//! directly (h = s), understating the 5y pool default probability for
//! s = 100 bps, R = 40% as 4.88% instead of ~8.00%, biasing all tranche
//! expected losses and fair spreads 36-60% low. Separately,
//! `discount_factor()` floored the risk-free rate at zero, silently treating
//! negative rates as 0%.

use approx::assert_relative_eq;
use openferric::credit::isda::hazard_from_par_spread;
use openferric::credit::{CdoTranche, SyntheticCdo};

fn assert_binary64_close(label: &str, actual: f64, expected: f64) {
    let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "{label}: expected={expected:.17e}, actual={actual:.17e}, \
         error={error:.3e}, binary64 budget={tolerance:.3e}"
    );
}

fn base_cdo() -> SyntheticCdo {
    SyntheticCdo {
        num_names: 125,
        pool_spread: 0.01,
        recovery_rate: 0.4,
        correlation: 0.30,
        risk_free_rate: 0.05,
        maturity: 5.0,
        payment_freq: 4,
    }
}

fn equity_tranche() -> CdoTranche {
    CdoTranche {
        attachment: 0.0,
        detachment: 0.03,
        notional: 0.03,
        spread: 0.0,
    }
}

fn mezz_tranche() -> CdoTranche {
    CdoTranche {
        attachment: 0.03,
        detachment: 0.07,
        notional: 0.04,
        spread: 0.0,
    }
}

fn senior_tranche() -> CdoTranche {
    CdoTranche {
        attachment: 0.07,
        detachment: 1.0,
        notional: 0.93,
        spread: 0.0,
    }
}

/// The CDO hazard rate must be pinned to the crate's canonical ISDA-style
/// credit-triangle conversion for a grid of spreads and recoveries.
#[test]
fn cdo_hazard_rate_pins_to_isda_credit_triangle() {
    let mut cdo = base_cdo();
    for &s in &[0.0, 0.0025, 0.01, 0.05, 0.12] {
        for &r in &[0.0, 0.25, 0.4, 0.75, 0.95] {
            cdo.pool_spread = s;
            cdo.recovery_rate = r;
            assert_relative_eq!(
                cdo.hazard_rate(),
                hazard_from_par_spread(s, r),
                epsilon = 0.0
            );
            // Closed form, independently: h = s / (1 - R).
            assert_binary64_close("credit-triangle hazard", cdo.hazard_rate(), s / (1.0 - r));
        }
    }
}

/// External anchor: with a flat hazard, the pool default probability is
/// 1 - exp(-h t) with h = s/(1-R). For s = 100 bps, R = 40%, T = 5y:
/// h = 0.01/0.6 = 1/60, h*T = 1/12, PD = 1 - exp(-1/12) = 0.0799555853706767
/// (hand computation; exp(-1/12) = 0.9200444146293233).
/// The pre-fix code (h = s) gave PD = 1 - exp(-0.05) = 0.048771, which this
/// test rejects.
#[test]
fn pool_default_probability_matches_hand_computed_anchor() {
    let cdo = base_cdo();
    let pd_5y = cdo.default_probability(5.0);
    assert_binary64_close(
        "5y pool default probability",
        pd_5y,
        0.079_955_585_370_676_71,
    );
    // Explicitly reject the pre-fix value.
    assert!((pd_5y - 0.048_771).abs() > 0.02);
}

/// External anchor: pool expected loss fraction must equal
/// (1-R) * (1 - exp(-s T / (1-R))). For s = 0.01, R = 0.4, T = 5:
/// 0.6 * (1 - exp(-1/12)) = 0.0479733512224060 (hand computation).
#[test]
fn portfolio_expected_loss_matches_credit_triangle_closed_form() {
    let cdo = base_cdo();
    let t = 5.0;
    let closed_form = (1.0 - cdo.recovery_rate)
        * (1.0 - (-cdo.pool_spread * t / (1.0 - cdo.recovery_rate)).exp());
    assert_binary64_close(
        "portfolio expected-loss closed form",
        closed_form,
        0.047_973_351_222_406_03,
    );
    assert_binary64_close(
        "portfolio expected loss",
        cdo.portfolio_expected_loss(t),
        closed_form,
    );
}

/// Hand-computed tranche anchor at zero correlation, where the LHP pool loss
/// is deterministic: L = (1-R) * PD = 0.6 * (1 - exp(-1/12)) = 0.0479733512.
/// Equity 0-3%: tranche loss = min(L, 0.03) = 0.03, i.e. fraction 1.0 of the
/// tranche. Mezz 3-7%: clamp(L - 0.03, 0, 0.04) = 0.0179733512, fraction
/// 0.0179733512 / 0.04 = 0.4493337806. Senior 7-100%: 0. All derived by hand
/// from the credit triangle and the tranche payoff definition.
#[test]
fn zero_correlation_tranche_losses_match_hand_computation() {
    let mut cdo = base_cdo();
    cdo.correlation = 0.0;
    let t = 5.0;
    let pool_loss = 0.047_973_351_222_406_03;

    let equity = equity_tranche();
    let mezz = mezz_tranche();
    let senior = senior_tranche();

    // Expected tranche loss in currency units = notional * loss fraction.
    assert_binary64_close(
        "zero-correlation equity loss",
        cdo.expected_tranche_loss(&equity, t),
        equity.notional,
    );
    assert_binary64_close(
        "zero-correlation mezzanine loss",
        cdo.expected_tranche_loss(&mezz, t),
        mezz.notional * ((pool_loss - 0.03) / 0.04),
    );
    assert_binary64_close(
        "zero-correlation senior loss",
        cdo.expected_tranche_loss(&senior, t),
        0.0,
    );
}

/// Consistency: with correlation on, tranche expected losses spanning
/// 0-100% must still sum to the (corrected) pool expected loss.
#[test]
fn tranche_expected_losses_sum_to_corrected_pool_loss() {
    let cdo = base_cdo();
    let t = cdo.maturity;
    let sum = cdo.expected_tranche_loss(&equity_tranche(), t)
        + cdo.expected_tranche_loss(&mezz_tranche(), t)
        + cdo.expected_tranche_loss(&senior_tranche(), t);
    // Supplemental partition identity. The primary SciPy LHP quadrature suite
    // (`credit_isda_quantlib`) establishes the individual tranche values; the
    // remaining 2e-10 is its documented inverse-normal/quadrature budget.
    assert_relative_eq!(sum, cdo.portfolio_expected_loss(t), epsilon = 2.0e-10);
    // The pool loss itself is the hand-computed anchor, so the sum is
    // pinned to an external value, not a self-referential one.
    assert_relative_eq!(sum, 0.047_973_351_222_406_03, epsilon = 2.0e-10);
}

/// Negative risk-free rates must change PVs. The pre-fix discount_factor
/// floored the rate at 0, making r = -2% and r = 0% produce identical
/// results (verified empirically on the old code: mezz fair spread
/// 417.137926 bps and NPV 0.00398866 at both rates).
#[test]
fn negative_rates_are_not_floored_to_zero() {
    let mezz = CdoTranche {
        spread: 0.02,
        ..mezz_tranche()
    };

    let mut cdo_neg = base_cdo();
    cdo_neg.risk_free_rate = -0.02;
    let mut cdo_zero = base_cdo();
    cdo_zero.risk_free_rate = 0.0;

    // Independent SciPy 1.17.1 LHP factor quadrature over quarterly payment
    // dates, epsabs=epsrel=1e-14 and split at both tranche kinks. The 5e-11
    // PV and 2e-10 spread budgets isolate the crate's documented inverse-
    // normal approximation error; they are not economic-price ranges.
    assert_relative_eq!(
        cdo_neg.protection_leg_pv(&mezz),
        0.014_120_759_741_736_854,
        epsilon = 5.0e-11
    );
    assert_relative_eq!(
        cdo_neg.premium_leg_pv(&mezz, mezz.spread),
        0.003_553_350_899_221_205_6,
        epsilon = 5.0e-11
    );
    assert_relative_eq!(
        cdo_neg.fair_spread(&mezz),
        0.079_478_554_987_810_12,
        epsilon = 2.0e-10
    );
    assert_relative_eq!(
        cdo_neg.npv(&mezz),
        0.010_567_408_842_515_648,
        epsilon = 5.0e-11
    );
    assert_relative_eq!(
        cdo_zero.protection_leg_pv(&mezz),
        0.013_376_354_771_248_778,
        epsilon = 5.0e-11
    );
    assert_relative_eq!(
        cdo_zero.premium_leg_pv(&mezz, mezz.spread),
        0.003_382_112_652_435_168,
        epsilon = 5.0e-11
    );
    assert_relative_eq!(
        cdo_zero.fair_spread(&mezz),
        0.079_100_586_798_122_29,
        epsilon = 2.0e-10
    );
    assert_relative_eq!(
        cdo_zero.npv(&mezz),
        0.009_994_242_118_813_61,
        epsilon = 5.0e-11
    );

    // Supplemental monotonicity checks preserve the historical bug signature.
    // With r = -2%, every discount factor exceeds 1, so both legs must be
    // strictly larger than at r = 0%.
    assert!(
        cdo_neg.premium_leg_pv(&mezz, mezz.spread) > cdo_zero.premium_leg_pv(&mezz, mezz.spread)
    );
    assert!(cdo_neg.protection_leg_pv(&mezz) > cdo_zero.protection_leg_pv(&mezz));

    // Fair spreads and NPVs must differ between r = -2% and r = 0%.
    let fs_neg = cdo_neg.fair_spread(&mezz);
    let fs_zero = cdo_zero.fair_spread(&mezz);
    assert!(
        (fs_neg - fs_zero).abs() > 1.0e-6,
        "fair spreads identical: {fs_neg} vs {fs_zero}"
    );
    assert!(
        (cdo_neg.npv(&mezz) - cdo_zero.npv(&mezz)).abs() > 1.0e-8,
        "NPVs identical under negative vs zero rates"
    );
}

/// NPV at the fair spread must be zero, including under negative rates.
#[test]
fn npv_is_zero_at_fair_spread_under_negative_rates() {
    let mut cdo = base_cdo();
    cdo.risk_free_rate = -0.015;
    for tranche in [equity_tranche(), mezz_tranche(), senior_tranche()] {
        let fair = cdo.fair_spread(&tranche);
        let at_par = CdoTranche {
            spread: fair,
            ..tranche
        };
        assert_binary64_close("NPV at fair spread", cdo.npv(&at_par), 0.0);
    }
}
